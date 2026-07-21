"""Joint PC/orientation residual diagnostics for Pt EBSD Kikuchi patterns.

This script keeps H5 PC/orientation as priors, then tests whether the remaining
mismatch is better explained by detector geometry (PC) or by a small rigid
crystal-frame rotation (orientation). It uses the H5/OHP Kikuchi band lines as
extra evidence instead of relying on a single image NCC score.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from batch_pt_kikuchi_spherical_calibration import (
    DEFAULT_H5,
    DEFAULT_MASTER,
    DEFAULT_UP2_ROOT,
    choose_quality_indices,
    detector_mask_for_pattern,
    matched_pt_maps,
    norm3,
    projection_with_pc,
    rotation_matrix_from_deg,
    scan_position_pc,
    score_orientation_matrix,
)
from export_publication_h5_kikuchi_bands import PUBLICATION_VARIANT, line_segment_from_band, read_bands, read_ohp_header
from project_edax_oim_to_sphere import EdaxMapInputs, build_master_lon_colat, read_edax_inputs, sample_master
from single_kikuchi_pc_finetune import (
    build_master_samplers,
    build_preprocessed_images,
    choose_orientation_matrix,
    detector_directions_with_pc,
    make_stride_indices,
    project_crystal_patch,
    score_with_directions,
    zscore,
)


DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_joint_pc_orientation_explainability"


@dataclass(frozen=True)
class BandProfileGrid:
    offsets: np.ndarray
    profile_indices: list[list[np.ndarray]]
    center_indices: np.ndarray
    band_labels: list[str]
    band_widths: np.ndarray
    band_strengths: np.ndarray


@dataclass
class PatternCase:
    key: str
    h5_row: dict[str, Any]
    selection: dict[str, Any]
    projection: Any
    map_pc: tuple[float, float, float]
    scan_pc: tuple[float, float, float]
    scan_meta: dict[str, Any]
    mask: np.ndarray
    circle: tuple[int, int, int]
    mask_mode: str
    images: dict[str, np.ndarray]
    orientation_name: str
    base_matrix: np.ndarray
    band_grid: BandProfileGrid


@dataclass(frozen=True)
class CandidateScore:
    pc: tuple[float, float, float]
    rot_deg: tuple[float, float, float]
    matrix: np.ndarray
    intensity_score: float
    image_band_score: float
    image_score: float
    band_center_score: float
    band_profile_score: float
    band_width_score: float
    mean_center_error_px: float
    std_center_error_px: float
    mean_width_error_px: float
    pc_prior_penalty: float
    orientation_prior_penalty: float
    total_score: float
    center_offsets_px: tuple[float, ...]
    exp_widths_px: tuple[float, ...]
    master_widths_px: tuple[float, ...]


def safe_key(text: str) -> str:
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    return "_".join("".join(out).split("_"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def weighted_width(offsets: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.nanmin(values)
    total = float(np.nansum(values))
    if total <= 1e-10:
        return 0.0
    center = float(np.nansum(offsets * values) / total)
    var = float(np.nansum(((offsets - center) ** 2) * values) / total)
    return math.sqrt(max(var, 0.0))


def profile_peak_offset(offsets: np.ndarray, values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return 0.0
    return float(offsets[int(np.nanargmax(values))])


def profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    return float(np.mean(zscore(a.astype(np.float64)) * zscore(b.astype(np.float64))))


def line_sample_indices(
    row0: float,
    col0: float,
    row1: float,
    col1: float,
    shape: tuple[int, int],
    mask: np.ndarray,
    offsets: np.ndarray,
    max_points: int,
) -> list[np.ndarray]:
    height, width = shape
    drow = row1 - row0
    dcol = col1 - col0
    length = math.hypot(drow, dcol)
    if length < 2:
        return [np.array([], dtype=np.int64) for _ in offsets]
    t = np.linspace(0.04, 0.96, max_points)
    rows = row0 + t * drow
    cols = col0 + t * dcol
    normal_row = -dcol / length
    normal_col = drow / length
    out: list[np.ndarray] = []
    for offset in offsets:
        rr = np.rint(rows + offset * normal_row).astype(np.int64)
        cc = np.rint(cols + offset * normal_col).astype(np.int64)
        in_bounds = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        valid = in_bounds.copy()
        if np.any(in_bounds):
            valid[in_bounds] &= mask[rr[in_bounds], cc[in_bounds]]
        if not np.any(valid):
            out.append(np.array([], dtype=np.int64))
            continue
        indices = np.unique((rr[valid] * width + cc[valid]).astype(np.int64))
        out.append(indices)
    return out


def build_band_profile_grid(
    h5_path: Path,
    map_group_path: str,
    pattern_index: int,
    shape: tuple[int, int],
    mask: np.ndarray,
    args: argparse.Namespace,
) -> BandProfileGrid:
    with h5py.File(h5_path, "r") as h5:
        group = h5[map_group_path]
        header = read_ohp_header(group)
        bands = read_bands(group, pattern_index)[: args.max_ohp_bands]

    offsets = np.arange(-args.profile_radius_px, args.profile_radius_px + 1e-9, args.profile_step_px, dtype=np.float64)
    profile_indices: list[list[np.ndarray]] = []
    center_indices: list[np.ndarray] = []
    labels: list[str] = []
    widths: list[float] = []
    strengths: list[float] = []

    for band_id, band in enumerate(bands):
        segment = line_segment_from_band(
            band=band,
            header=header,
            height=shape[0],
            width=shape[1],
            variant=PUBLICATION_VARIANT,
            band_index=band_id,
        )
        if segment is None:
            continue
        samples = line_sample_indices(
            row0=segment.row0,
            col0=segment.col0,
            row1=segment.row1,
            col1=segment.col1,
            shape=shape,
            mask=mask,
            offsets=offsets,
            max_points=args.profile_points_per_band,
        )
        if not samples or sum(len(item) for item in samples) < args.profile_points_per_band:
            continue
        zero_index = int(np.argmin(np.abs(offsets)))
        if samples[zero_index].size:
            center_indices.append(samples[zero_index])
        profile_indices.append(samples)
        labels.append(f"b{band_id}: rho={band.rho_bin:.0f}, theta={band.theta_deg:.0f}")
        widths.append(float(band.width))
        strengths.append(float(band.intensity))

    if center_indices:
        center = np.unique(np.concatenate(center_indices)).astype(np.int64)
    else:
        center = np.array([], dtype=np.int64)
    return BandProfileGrid(
        offsets=offsets,
        profile_indices=profile_indices,
        center_indices=center,
        band_labels=labels,
        band_widths=np.asarray(widths, dtype=np.float64),
        band_strengths=np.asarray(strengths, dtype=np.float64),
    )


def sample_master_band_at_indices(case: PatternCase, pc: tuple[float, float, float], matrix: np.ndarray, indices: np.ndarray, cache: dict[tuple[float, float, float], np.ndarray], samplers) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=np.float64)
    key = tuple(round(float(v), 8) for v in pc)
    detector_directions = cache.get(key)
    if detector_directions is None:
        detector_directions = detector_directions_with_pc(case.projection, pc).reshape(-1, 3)
        cache[key] = detector_directions
    vectors = detector_directions[indices] @ matrix
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return sample_master(vectors, samplers.upper_band, samplers.lower_band)


def evaluate_candidate(
    case: PatternCase,
    pc: tuple[float, float, float],
    rot_deg: tuple[float, float, float],
    args: argparse.Namespace,
    samplers,
    direction_cache: dict[tuple[float, float, float], np.ndarray],
) -> CandidateScore:
    matrix = case.base_matrix @ rotation_matrix_from_deg(*rot_deg).T
    intensity, image_band, image_score = score_orientation_matrix(
        projection=case.projection,
        pc=pc,
        matrix=matrix,
        mask=case.mask,
        images=case.images,
        samplers=samplers,
        args=args,
    )

    grid = case.band_grid
    center_master = sample_master_band_at_indices(case, pc, matrix, grid.center_indices, direction_cache, samplers)
    if center_master.size:
        line_center_score = float(np.nanmean(zscore(center_master)))
    else:
        line_center_score = 0.0

    profile_scores: list[float] = []
    center_errors: list[float] = []
    width_errors: list[float] = []
    exp_widths: list[float] = []
    master_widths: list[float] = []
    exp_band_flat = case.images["band"].ravel()
    for samples in grid.profile_indices:
        exp_profile = []
        master_profile = []
        for indices in samples:
            if indices.size == 0:
                exp_profile.append(np.nan)
                master_profile.append(np.nan)
                continue
            exp_profile.append(float(np.nanmean(exp_band_flat[indices])))
            master_values = sample_master_band_at_indices(case, pc, matrix, indices, direction_cache, samplers)
            master_profile.append(float(np.nanmean(master_values)) if master_values.size else np.nan)
        exp_profile_arr = np.asarray(exp_profile, dtype=np.float64)
        master_profile_arr = np.asarray(master_profile, dtype=np.float64)
        valid = np.isfinite(exp_profile_arr) & np.isfinite(master_profile_arr)
        if valid.sum() < 5:
            continue
        offsets = grid.offsets[valid]
        exp_valid = exp_profile_arr[valid]
        master_valid = master_profile_arr[valid]
        profile_scores.append(profile_corr(exp_valid, master_valid))
        exp_peak = profile_peak_offset(offsets, exp_valid)
        master_peak = profile_peak_offset(offsets, master_valid)
        center_errors.append(master_peak - exp_peak)
        exp_w = weighted_width(offsets, exp_valid)
        master_w = weighted_width(offsets, master_valid)
        exp_widths.append(exp_w)
        master_widths.append(master_w)
        width_errors.append(abs(master_w - exp_w))

    if profile_scores:
        band_profile_score = float(np.mean(profile_scores))
        mean_center_error = float(np.mean(np.abs(center_errors)))
        std_center_error = float(np.std(center_errors))
        mean_width_error = float(np.mean(width_errors))
    else:
        band_profile_score = 0.0
        mean_center_error = args.profile_radius_px
        std_center_error = args.profile_radius_px
        mean_width_error = args.profile_radius_px

    band_center_score = math.exp(-mean_center_error / max(args.center_error_scale_px, 1e-6))
    band_width_score = math.exp(-mean_width_error / max(args.width_error_scale_px, 1e-6))
    pc_delta = np.asarray(pc, dtype=np.float64) - np.asarray(case.scan_pc, dtype=np.float64)
    pc_scale = np.asarray(args.pc_prior_scale, dtype=np.float64)
    pc_prior = float(np.mean((pc_delta / (pc_scale + 1e-12)) ** 2))
    rot = np.asarray(rot_deg, dtype=np.float64)
    ori_prior = float(np.mean((rot / max(args.orientation_bound_deg, 1e-6)) ** 2))
    total = (
        args.image_score_weight * image_score
        + args.band_center_weight * band_center_score
        + args.band_profile_weight * band_profile_score
        + args.band_width_weight * band_width_score
        - args.pc_prior_weight * pc_prior
        - args.orientation_prior_weight * ori_prior
    )
    return CandidateScore(
        pc=pc,
        rot_deg=rot_deg,
        matrix=matrix,
        intensity_score=float(intensity),
        image_band_score=float(image_band),
        image_score=float(image_score),
        band_center_score=float(band_center_score),
        band_profile_score=float(band_profile_score),
        band_width_score=float(band_width_score),
        mean_center_error_px=mean_center_error,
        std_center_error_px=std_center_error,
        mean_width_error_px=mean_width_error,
        pc_prior_penalty=pc_prior,
        orientation_prior_penalty=ori_prior,
        total_score=float(total),
        center_offsets_px=tuple(float(v) for v in center_errors),
        exp_widths_px=tuple(float(v) for v in exp_widths),
        master_widths_px=tuple(float(v) for v in master_widths),
    )


def optimize_pc_grid(
    case: PatternCase,
    current_pc: tuple[float, float, float],
    rot_deg: tuple[float, float, float],
    args: argparse.Namespace,
    samplers,
    cache: dict[tuple[float, float, float], np.ndarray],
) -> CandidateScore:
    ranges = np.asarray(args.pc_search_range, dtype=np.float64)
    steps = np.asarray(args.pc_search_step, dtype=np.float64)
    offset_axes = [np.arange(-rng, rng + 0.5 * step, step, dtype=np.float64) for rng, step in zip(ranges, steps)]
    best: CandidateScore | None = None
    for dx in offset_axes[0]:
        for dy in offset_axes[1]:
            for dz in offset_axes[2]:
                pc = (current_pc[0] + float(dx), current_pc[1] + float(dy), current_pc[2] + float(dz))
                if pc[2] <= 0.05:
                    continue
                score = evaluate_candidate(case, pc, rot_deg, args, samplers, cache)
                if best is None or score.total_score > best.total_score:
                    best = score
    if best is None:
        raise RuntimeError(f"No valid PC candidate for {case.key}")
    return best


def optimize_orientation_coordinate(
    case: PatternCase,
    pc: tuple[float, float, float],
    current_rot: tuple[float, float, float],
    args: argparse.Namespace,
    samplers,
    cache: dict[tuple[float, float, float], np.ndarray],
) -> CandidateScore:
    values = np.arange(-args.orientation_bound_deg, args.orientation_bound_deg + 0.5 * args.orientation_step_deg, args.orientation_step_deg)
    rot = np.asarray(current_rot, dtype=np.float64)
    best = evaluate_candidate(case, pc, tuple(float(v) for v in rot), args, samplers, cache)
    for _iteration in range(args.joint_iterations):
        improved = False
        for axis in range(3):
            axis_best = best
            for value in values:
                trial = rot.copy()
                trial[axis] = float(value)
                score = evaluate_candidate(case, pc, tuple(float(v) for v in trial), args, samplers, cache)
                if score.total_score > axis_best.total_score:
                    axis_best = score
            if axis_best.total_score > best.total_score + 1e-12:
                best = axis_best
                rot = np.asarray(best.rot_deg, dtype=np.float64)
                improved = True
        if not improved:
            break
    return best


def optimize_joint(case: PatternCase, args: argparse.Namespace, samplers) -> tuple[CandidateScore, CandidateScore, CandidateScore, CandidateScore, list[CandidateScore]]:
    cache: dict[tuple[float, float, float], np.ndarray] = {}
    zero_rot = (0.0, 0.0, 0.0)
    scan = evaluate_candidate(case, case.scan_pc, zero_rot, args, samplers, cache)
    pc_only = optimize_pc_grid(case, case.scan_pc, zero_rot, args, samplers, cache)
    ori_only = optimize_orientation_coordinate(case, case.scan_pc, zero_rot, args, samplers, cache)

    current_pc = case.scan_pc
    current_rot = zero_rot
    trace = [scan]
    best = scan
    for _iteration in range(args.joint_iterations):
        pc_step = optimize_pc_grid(case, current_pc, current_rot, args, samplers, cache)
        current_pc = pc_step.pc
        trace.append(pc_step)
        rot_step = optimize_orientation_coordinate(case, current_pc, current_rot, args, samplers, cache)
        current_rot = rot_step.rot_deg
        trace.append(rot_step)
        if rot_step.total_score > best.total_score:
            best = rot_step
    return scan, pc_only, ori_only, best, trace


def master_band_detector_image(case: PatternCase, score: CandidateScore, samplers) -> np.ndarray:
    flat_indices = np.flatnonzero(case.mask.ravel()).astype(np.int64)
    cache: dict[tuple[float, float, float], np.ndarray] = {}
    values = sample_master_band_at_indices(case, score.pc, score.matrix, flat_indices, cache, samplers)
    image = np.full(case.mask.size, np.nan, dtype=np.float32)
    image[flat_indices] = values.astype(np.float32)
    return image.reshape(case.mask.shape)


def residual_interpretation(pc_like_evidence: float, orientation_like_evidence: float, pc_norm: float, rot_norm: float) -> str:
    if orientation_like_evidence > 0.75 and pc_like_evidence <= 0.0:
        return "orientation_dominant_pc_not_supported_by_width"
    if orientation_like_evidence > 0.75 and pc_like_evidence > 0.02:
        return "mixed_orientation_and_pc_supported"
    if pc_like_evidence > 0.02 and pc_norm > 1e-6:
        return "pc_dominant_width_distortion_reduced"
    if rot_norm > 1e-6:
        return "weak_orientation_like"
    return "ambiguous_or_already_aligned"


def draw_ohp_segments(ax, case: PatternCase) -> None:
    shape = case.mask.shape
    with h5py.File(case.h5_row["source_h5"], "r") as h5:
        group = h5[case.h5_row["h5_path"]]
        header = read_ohp_header(group)
        bands = read_bands(group, int(case.selection["pattern_index"]))[: len(case.band_grid.band_labels)]
    for band_id, band in enumerate(bands):
        segment = line_segment_from_band(
            band=band,
            header=header,
            height=shape[0],
            width=shape[1],
            variant=PUBLICATION_VARIANT,
            band_index=band_id,
        )
        if segment is None:
            continue
        ax.plot([segment.col0, segment.col1], [segment.row0, segment.row1], lw=1.2, alpha=0.95)


def plot_candidate_patch(ax, case: PatternCase, master_texture: np.ndarray, score: CandidateScore, title: str) -> None:
    patch = project_crystal_patch(case.projection, score.pc, score.matrix, case.images["enhanced"], case.mask)
    ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    ax.imshow(
        patch[0],
        cmap="magma",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=np.where(patch[1], 0.88, 0.0),
    )
    ax.set_title(title)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("colatitude (deg)")


def save_case_visualization(
    path: Path,
    case: PatternCase,
    scan: CandidateScore,
    pc_only: CandidateScore,
    ori_only: CandidateScore,
    joint: CandidateScore,
    master_texture: np.ndarray,
    samplers,
) -> None:
    exp_band = np.where(case.mask, case.images["band"], np.nan)
    scan_master = master_band_detector_image(case, scan, samplers)
    joint_master = master_band_detector_image(case, joint, samplers)
    scan_diff = np.where(case.mask, exp_band - scan_master, np.nan)
    joint_diff = np.where(case.mask, exp_band - joint_master, np.nan)

    fig = plt.figure(figsize=(21, 15))
    axes = [fig.add_subplot(3, 4, i + 1) for i in range(12)]
    axes[0].imshow(case.projection.pattern, cmap="gray")
    axes[0].contour(case.mask, levels=[0.5], colors=["#ff3030"], linewidths=0.8)
    draw_ohp_segments(axes[0], case)
    axes[0].set_title(f"Raw + full mask + H5/OHP bands\n{case.key}")
    axes[0].axis("off")

    axes[1].imshow(np.where(case.mask, case.images["enhanced"], np.nan), cmap="gray")
    axes[1].set_title("Preprocessed Kikuchi")
    axes[1].axis("off")

    axes[2].imshow(exp_band, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Experimental band-enhanced image")
    axes[2].axis("off")

    axes[3].imshow(joint_master, cmap="magma")
    axes[3].set_title("Master band sampled to detector\njoint PC + orientation")
    axes[3].axis("off")

    vmax = np.nanpercentile(np.abs(scan_diff), 97.0)
    axes[4].imshow(scan_diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[4].set_title("Residual map before joint fit\nexp band - master band")
    axes[4].axis("off")

    vmax2 = np.nanpercentile(np.abs(joint_diff), 97.0)
    axes[5].imshow(joint_diff, cmap="coolwarm", vmin=-vmax2, vmax=vmax2)
    axes[5].set_title("Residual map after joint fit")
    axes[5].axis("off")

    plot_candidate_patch(axes[6], case, master_texture, scan, f"Start: scan-position PC\nscore={scan.total_score:+.4f}")
    plot_candidate_patch(axes[7], case, master_texture, joint, f"Joint PC + orientation\nscore={joint.total_score:+.4f}")

    names = ["scan", "PC only", "ori only", "joint"]
    scores = [scan, pc_only, ori_only, joint]
    x = np.arange(len(names))
    width = 0.18
    axes[8].bar(x - 1.5 * width, [s.image_score for s in scores], width, label="image NCC")
    axes[8].bar(x - 0.5 * width, [s.band_center_score for s in scores], width, label="band center")
    axes[8].bar(x + 0.5 * width, [s.band_profile_score for s in scores], width, label="profile")
    axes[8].bar(x + 1.5 * width, [s.band_width_score for s in scores], width, label="width")
    axes[8].set_xticks(x, names, rotation=15)
    axes[8].set_title("Loss components")
    axes[8].legend(fontsize=8)

    axes[9].bar(names, [s.total_score for s in scores], color=["#ffbf00", "#1f77b4", "#2ca02c", "#d62728"])
    axes[9].set_title("Total objective with priors")
    axes[9].tick_params(axis="x", rotation=15)

    axes[10].plot(scan.center_offsets_px, "o-", label="scan")
    axes[10].plot(joint.center_offsets_px, "o-", label="joint")
    axes[10].axhline(0, color="black", lw=0.8)
    axes[10].set_title("OHP band center residual offsets")
    axes[10].set_xlabel("H5/OHP band")
    axes[10].set_ylabel("peak offset (px)")
    axes[10].legend()

    axes[11].axis("off")
    pc_delta = np.asarray(joint.pc) - np.asarray(case.scan_pc)
    pc_like = scan.mean_width_error_px - joint.mean_width_error_px
    orientation_like = scan.mean_center_error_px - joint.mean_center_error_px
    interpretation = residual_interpretation(pc_like, orientation_like, norm3(pc_delta), norm3(joint.rot_deg))
    summary = (
        f"orientation variant: {case.orientation_name}\n"
        f"scan PC: ({case.scan_pc[0]:.6f}, {case.scan_pc[1]:.6f}, {case.scan_pc[2]:.6f})\n"
        f"joint PC: ({joint.pc[0]:.6f}, {joint.pc[1]:.6f}, {joint.pc[2]:.6f})\n"
        f"delta PC: ({pc_delta[0]:+.6f}, {pc_delta[1]:+.6f}, {pc_delta[2]:+.6f}), |dPC|={norm3(pc_delta):.6f}\n"
        f"delta R deg: ({joint.rot_deg[0]:+.3f}, {joint.rot_deg[1]:+.3f}, {joint.rot_deg[2]:+.3f}), |dR|={norm3(joint.rot_deg):.3f}\n"
        f"center error: {scan.mean_center_error_px:.3f} -> {joint.mean_center_error_px:.3f} px\n"
        f"width error:  {scan.mean_width_error_px:.3f} -> {joint.mean_width_error_px:.3f} px\n"
        f"profile NCC:  {scan.band_profile_score:+.3f} -> {joint.band_profile_score:+.3f}\n"
        f"PC-like evidence: {pc_like:+.3f}; orientation-like evidence: {orientation_like:+.3f}\n"
        f"interpretation: {interpretation}"
    )
    axes[11].text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=9.5)

    fig.suptitle("Joint PC/orientation explainability: image + H5/OHP band constraints", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_case(args: argparse.Namespace, h5_row: dict[str, Any], selection: dict[str, Any], samplers) -> PatternCase:
    pattern_index = int(selection["pattern_index"])
    key = safe_key(f"{h5_row['specimen']}_{h5_row['area']}_{h5_row['map_name']}_idx{pattern_index}")
    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=Path(h5_row["up2_actual_path"]),
            map_group=h5_row["h5_path"],
            pattern_index=pattern_index,
        )
    )
    map_pc = projection.pc_edax
    scan_pc, scan_meta = scan_position_pc(args, h5_row["h5_path"], pattern_index, map_pc)
    projection = projection_with_pc(projection, scan_pc)
    mask, circle, mask_mode = detector_mask_for_pattern(projection, args)
    images = build_preprocessed_images(projection.pattern, mask)
    orientation_name, base_matrix, _orientation_rows = choose_orientation_matrix(
        projection=projection,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    band_grid = build_band_profile_grid(
        h5_path=args.h5,
        map_group_path=h5_row["h5_path"],
        pattern_index=pattern_index,
        shape=projection.pattern.shape,
        mask=mask,
        args=args,
    )
    return PatternCase(
        key=key,
        h5_row={**h5_row, "source_h5": str(args.h5)},
        selection=selection,
        projection=projection,
        map_pc=map_pc,
        scan_pc=scan_pc,
        scan_meta=scan_meta,
        mask=mask,
        circle=circle,
        mask_mode=mask_mode,
        images=images,
        orientation_name=orientation_name,
        base_matrix=base_matrix,
        band_grid=band_grid,
    )


def select_one_mapping(args: argparse.Namespace) -> dict[str, Any]:
    matched = matched_pt_maps(args.h5, args.up2_root or [DEFAULT_UP2_ROOT], args.specimen)
    if args.area:
        matched = [row for row in matched if row["area"] == args.area]
    if args.map_group:
        matched = [row for row in matched if row["h5_path"].strip("/") == args.map_group.strip("/")]
    if not matched:
        raise RuntimeError("No matched H5/UP2 mapping found for the requested Pt selection.")
    return matched[0]


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samplers = build_master_samplers(args.master)
    master_texture = build_master_lon_colat(samplers.upper_corrected, samplers.lower_corrected)
    h5_row = select_one_mapping(args)

    with h5py.File(args.h5, "r") as h5:
        map_group = h5[h5_row["h5_path"]]
        selections = choose_quality_indices(
            map_group=map_group,
            count=int(h5_row["point_count"]),
            patterns_per_map=args.pattern_count,
            ci_min=args.ci_min,
            min_scan_distance_fraction=args.min_scan_distance_fraction,
        )
    if not selections:
        raise RuntimeError("No patterns passed the selection filters.")

    rows: list[dict[str, Any]] = []
    for idx, selection in enumerate(selections, start=1):
        print(f"[{idx}/{len(selections)}] {h5_row['area']} idx={selection['pattern_index']}")
        case = build_case(args, h5_row, selection, samplers)
        scan, pc_only, ori_only, joint, trace = optimize_joint(case, args, samplers)
        case_dir = args.output_dir / "per_pattern" / case.key
        figure_path = case_dir / f"{case.key}_joint_pc_orientation_explainability.png"
        save_case_visualization(figure_path, case, scan, pc_only, ori_only, joint, master_texture, samplers)
        write_csv(
            case_dir / "joint_optimization_trace.csv",
            [
                {
                    "step": step,
                    "pc_x": score.pc[0],
                    "pc_y": score.pc[1],
                    "pc_z": score.pc[2],
                    "rot_x_deg": score.rot_deg[0],
                    "rot_y_deg": score.rot_deg[1],
                    "rot_z_deg": score.rot_deg[2],
                    "total_score": score.total_score,
                    "image_score": score.image_score,
                    "band_center_score": score.band_center_score,
                    "band_profile_score": score.band_profile_score,
                    "band_width_score": score.band_width_score,
                    "mean_center_error_px": score.mean_center_error_px,
                    "mean_width_error_px": score.mean_width_error_px,
                }
                for step, score in enumerate(trace)
            ],
        )
        pc_delta = np.asarray(joint.pc) - np.asarray(case.scan_pc)
        pc_like = scan.mean_width_error_px - joint.mean_width_error_px
        orientation_like = scan.mean_center_error_px - joint.mean_center_error_px
        interpretation = residual_interpretation(
            pc_like_evidence=pc_like,
            orientation_like_evidence=orientation_like,
            pc_norm=norm3(pc_delta),
            rot_norm=norm3(joint.rot_deg),
        )
        rows.append(
            {
                "area": h5_row["area"],
                "map_name": h5_row["map_name"],
                "pattern_index": selection["pattern_index"],
                "row": selection["row"],
                "col": selection["col"],
                "IQ": selection["IQ"],
                "CI": selection["CI"],
                "orientation_variant": case.orientation_name,
                "ohp_band_count": len(case.band_grid.profile_indices),
                "scan_total_score": scan.total_score,
                "pc_only_total_score": pc_only.total_score,
                "ori_only_total_score": ori_only.total_score,
                "joint_total_score": joint.total_score,
                "scan_image_score": scan.image_score,
                "joint_image_score": joint.image_score,
                "scan_center_error_px": scan.mean_center_error_px,
                "joint_center_error_px": joint.mean_center_error_px,
                "scan_width_error_px": scan.mean_width_error_px,
                "joint_width_error_px": joint.mean_width_error_px,
                "scan_profile_score": scan.band_profile_score,
                "joint_profile_score": joint.band_profile_score,
                "scan_pc_x": case.scan_pc[0],
                "scan_pc_y": case.scan_pc[1],
                "scan_pc_z": case.scan_pc[2],
                "joint_pc_x": joint.pc[0],
                "joint_pc_y": joint.pc[1],
                "joint_pc_z": joint.pc[2],
                "delta_pcx": pc_delta[0],
                "delta_pcy": pc_delta[1],
                "delta_pcz": pc_delta[2],
                "delta_pc_norm": norm3(pc_delta),
                "rot_x_deg": joint.rot_deg[0],
                "rot_y_deg": joint.rot_deg[1],
                "rot_z_deg": joint.rot_deg[2],
                "rot_norm_deg": norm3(joint.rot_deg),
                "pc_like_evidence": pc_like,
                "orientation_like_evidence": orientation_like,
                "residual_interpretation": interpretation,
                "figure": str(figure_path),
            }
        )

    summary_csv = args.output_dir / "joint_pc_orientation_explainability_summary.csv"
    write_csv(summary_csv, rows)
    save_contact_sheet(args.output_dir / "joint_pc_orientation_explainability_contact_sheet.png", rows)
    print(f"Summary CSV: {summary_csv}")
    print(f"Contact sheet: {args.output_dir / 'joint_pc_orientation_explainability_contact_sheet.png'}")


def save_contact_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = min(2, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 8.5, rows_n * 6.2), dpi=160)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, row in zip(axes_arr.ravel(), rows):
        image = plt.imread(row["figure"])
        ax.imshow(image)
        ax.set_title(
            f"{row['area']} idx={row['pattern_index']} "
            f"score {float(row['scan_total_score']):+.3f}->{float(row['joint_total_score']):+.3f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Joint PC/orientation explainability on one Pt EBSD mapping", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint PC/orientation residual diagnostics with H5/OHP Kikuchi band constraints."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", action="append", type=Path, default=None)
    parser.add_argument("--specimen", default="Pt-3")
    parser.add_argument("--area", default="Area 3-90")
    parser.add_argument("--map-group", default="")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern-count", type=int, default=3)
    parser.add_argument("--ci-min", type=float, default=0.30)
    parser.add_argument("--min-scan-distance-fraction", type=float, default=0.18)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--mask-mode", choices=("centered", "estimated"), default="centered")
    parser.add_argument("--mask-radius-fraction", type=float, default=0.49)
    parser.add_argument("--pc-initial", choices=("scan_position", "map"), default="scan_position")
    parser.add_argument("--pc-x-sign", type=float, default=-1.0)
    parser.add_argument("--pc-y-sign", type=float, default=1.0)
    parser.add_argument("--pc-y-scale", type=float, default=math.cos(math.radians(70.0)))
    parser.add_argument("--pc-z-sign", type=float, default=1.0)
    parser.add_argument("--pc-z-scale", type=float, default=math.sin(math.radians(70.0)))
    parser.add_argument("--pc-search-range", nargs=3, type=float, default=(0.004, 0.004, 0.008))
    parser.add_argument("--pc-search-step", nargs=3, type=float, default=(0.002, 0.002, 0.004))
    parser.add_argument("--pc-prior-scale", nargs=3, type=float, default=(0.010, 0.010, 0.020))
    parser.add_argument("--orientation-bound-deg", type=float, default=0.5)
    parser.add_argument("--orientation-step-deg", type=float, default=0.05)
    parser.add_argument("--joint-iterations", type=int, default=3)
    parser.add_argument("--max-ohp-bands", type=int, default=8)
    parser.add_argument("--profile-radius-px", type=float, default=8.0)
    parser.add_argument("--profile-step-px", type=float, default=1.0)
    parser.add_argument("--profile-points-per-band", type=int, default=90)
    parser.add_argument("--center-error-scale-px", type=float, default=2.0)
    parser.add_argument("--width-error-scale-px", type=float, default=2.5)
    parser.add_argument("--image-score-weight", type=float, default=0.45)
    parser.add_argument("--band-center-weight", type=float, default=0.20)
    parser.add_argument("--band-profile-weight", type=float, default=0.25)
    parser.add_argument("--band-width-weight", type=float, default=0.10)
    parser.add_argument("--pc-prior-weight", type=float, default=0.050)
    parser.add_argument("--orientation-prior-weight", type=float, default=0.020)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
