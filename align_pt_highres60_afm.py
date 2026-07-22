from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import exposure, filters

from afm_ebsd_surface_index import (
    height_to_normals,
    normal_direction_rgb,
    save_normalmap_with_legend,
    save_scalar_image,
)
from align_pt_afm_sem_ipf import (
    Candidate,
    afm_feature_images,
    find_best_alignment,
    make_lightglue_models,
    read_afm_channels,
    robust_rescale,
    save_match_figure,
    sem_feature_images,
    warp_to_sem,
    write_candidates,
)
from export_pt_highres_data_overview import read_ipf_map, read_reference_ipf
from pt_highres_30deg_lightglue_calibration import DEFAULT_H5, build_map_specs, normalize_gray


DEFAULT_AFM = Path(r"D:\EBSD project\3d数据\pt-afm\Pt-2high resolution.ibw")
DEFAULT_EDAX_IPF = Path(r"E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution\60.bmp")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_highres60_afm_alignment_corrected"
DEFAULT_ANGLE = 60


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_ebsd60(args: argparse.Namespace) -> dict[str, Any]:
    specs = {spec.angle_deg: spec for spec in build_map_specs()}
    spec = specs[args.angle]
    with h5py.File(args.h5, "r") as h5:
        group = h5[spec.h5_group]
        sem_raw = normalize_gray(np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
        # EDAX's exported IPF-Z bitmap and the PRIAS SEM image in this H5 are
        # stored with opposite row direction.  Flip only the SEM image; IPF and
        # EBSD scan-row data stay in the software/export frame.
        sem = np.flipud(sem_raw) if args.flip_sem_y else sem_raw
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        step_x_um = float(np.asarray(group["Sample/Step X"][()]).reshape(-1)[0])
        step_y_um = float(np.asarray(group["Sample/Step Y"][()]).reshape(-1)[0])
        ipf_h5 = read_ipf_map(group, sem.shape)
    ipf_ref = read_reference_ipf(args.edax_ipf) if args.edax_ipf.exists() else ipf_h5
    ipf_sem = cv2.resize(ipf_ref, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return {
        "spec": spec,
        "sem_raw": sem_raw,
        "sem": sem,
        "ipf_sem": ipf_sem,
        "ipf_h5": ipf_h5,
        "sem_flip_y_to_match_ipf": bool(args.flip_sem_y),
        "nrows": nrows,
        "ncols": ncols,
        "step_x_um": step_x_um,
        "step_y_um": step_y_um,
        "physical_width_um": ncols * step_x_um,
        "physical_height_um": nrows * step_y_um,
    }


def choose_match_scales(scan_size_um: float, ebsd_width_um: float, sem_width_px: int, afm_width_px: int) -> tuple[list[float], float]:
    expected_sem_width_px = sem_width_px * scan_size_um / max(ebsd_width_um, 1e-6)
    expected_scale = expected_sem_width_px / max(afm_width_px, 1)
    multipliers = [0.55, 0.75, 0.90, 1.0, 1.15, 1.35, 1.65, 2.10]
    scales = [expected_scale * value for value in multipliers]
    scales.extend([0.16, 0.22, 0.30, 0.40, 0.55])
    unique = sorted({round(float(np.clip(scale, 0.10, 1.0)), 4) for scale in scales})
    return unique, float(expected_scale)


def scaled_afm_feature_images(
    channels: dict[str, np.ndarray],
    scales: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    base = afm_feature_images(channels)
    keep = (
        "HeightRetrace_hp",
        "HeightRetrace_sobel",
        "AmplitudeRetrace_hp",
        "AmplitudeRetrace_sobel",
        "ZSensorRetrace_hp",
        "PhaseRetrace_hp",
    )
    output: dict[str, np.ndarray] = {}
    scale_by_name: dict[str, float] = {}
    for name in keep:
        if name not in base:
            continue
        image = base[name]
        height, width = image.shape
        for scale in scales:
            out_w = max(64, int(round(width * scale)))
            out_h = max(64, int(round(height * scale)))
            resized = cv2.resize(
                image,
                (out_w, out_h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            ).astype(np.float32)
            key = f"{name}_scale{scale:.4f}"
            output[key] = resized
            scale_by_name[key] = scale
    return output, scale_by_name


def convert_candidate_to_original_afm(candidate: Candidate, scale: float) -> Candidate:
    converted = Candidate(
        afm_name=candidate.afm_name,
        sem_name=candidate.sem_name,
        matches=candidate.matches,
        inliers=candidate.inliers,
        inlier_ratio=candidate.inlier_ratio,
        rmse=candidate.rmse,
        affine_afm_to_sem=candidate.affine_afm_to_sem.copy(),
        inlier_mask=candidate.inlier_mask.copy(),
        key_sem=candidate.key_sem.copy(),
        key_afm=candidate.key_afm.copy(),
    )
    converted.affine_afm_to_sem[:, :2] *= scale
    return converted


def rank_candidates_with_scale(candidates: list[Candidate], expected_scale: float) -> list[Candidate]:
    def score(candidate: Candidate) -> tuple[int, float, int, float, float]:
        geom_scale = math.sqrt(max(abs(candidate.det), 1e-12))
        scale_error = abs(math.log(max(geom_scale, 1e-9) / max(expected_scale, 1e-9)))
        plausible = int(0.35 * expected_scale <= geom_scale <= 2.80 * expected_scale and candidate.det > 0)
        return (plausible, candidate.inliers, candidate.inlier_ratio, -candidate.rmse, -scale_error)

    return sorted(candidates, key=score, reverse=True)


def save_rgb(path: Path, image: np.ndarray, title: str | None = None, mask: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220, constrained_layout=True)
    if mask is None:
        ax.imshow(np.clip(image, 0.0, 1.0))
    else:
        ax.imshow(np.clip(image, 0.0, 1.0), alpha=np.clip(mask, 0.0, 1.0))
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_gray(path: Path, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220, constrained_layout=True)
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def warp_rgb_to_sem(image: np.ndarray, affine_afm_to_sem: np.ndarray, sem_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = sem_shape
    warped = cv2.warpAffine(
        image.astype(np.float32),
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones(image.shape[:2], dtype=np.float32),
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(warped, 0.0, 1.0).astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32)


def overlay_gray_with_rgba(gray: np.ndarray, rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    base = np.dstack([gray] * 3)
    a = alpha * mask[..., None]
    return np.clip(base * (1.0 - a) + rgb * a, 0.0, 1.0).astype(np.float32)


def sem_grain_boundary_feature(sem: np.ndarray) -> np.ndarray:
    """Extract broad SEM grain-boundary contrast while suppressing fine scan/channeling stripes."""
    normalized = robust_rescale(sem)
    equalized = exposure.equalize_adapthist(normalized, clip_limit=0.018).astype(np.float32)
    smooth = filters.gaussian(equalized, sigma=3.2, preserve_range=True).astype(np.float32)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    edge = robust_rescale(np.sqrt(gx * gx + gy * gy), 55.0, 99.8)
    canny = cv2.Canny((smooth * 255).astype(np.uint8), 22, 85).astype(np.float32) / 255.0
    combined = np.maximum(edge, 0.72 * canny)
    combined = filters.gaussian(combined, sigma=0.8, preserve_range=True).astype(np.float32)
    return robust_rescale(combined, 1.0, 99.5)


def afm_grain_boundary_feature(channels: dict[str, np.ndarray], height_channel: str) -> np.ndarray:
    """AFM height-edge feature used for physical grain-boundary alignment."""
    height = channels[height_channel].astype(np.float32)
    height = height - np.nanmedian(height)
    height = robust_rescale(height, 1.0, 99.0)
    smooth = filters.gaussian(height, sigma=4.0, preserve_range=True).astype(np.float32)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    edge = robust_rescale(np.sqrt(gx * gx + gy * gy), 55.0, 99.8)
    edge = exposure.equalize_adapthist(edge, clip_limit=0.015).astype(np.float32)
    return filters.gaussian(edge, sigma=1.1, preserve_range=True).astype(np.float32)


def build_similarity_affine(
    afm_shape: tuple[int, int],
    target_center_xy: np.ndarray,
    scale: float,
    angle_deg: float,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    afm_h, afm_w = afm_shape
    afm_center = np.array([afm_w / 2.0, afm_h / 2.0], dtype=np.float64)
    theta = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    flip = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
    linear = scale * (rotation @ flip)
    offset = target_center_xy - linear @ afm_center
    return np.column_stack([linear, offset]).astype(np.float64)


def center_error_px(candidate: Candidate, target_shape: tuple[int, int], afm_shape: tuple[int, int]) -> float:
    target_h, target_w = target_shape
    afm_h, afm_w = afm_shape
    mapped = candidate.affine_afm_to_sem @ np.array([afm_w / 2.0, afm_h / 2.0, 1.0], dtype=np.float64)
    expected = np.array([target_w / 2.0, target_h / 2.0], dtype=np.float64)
    return float(np.linalg.norm(mapped - expected))


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    if int(valid.sum()) < 256:
        return -1.0
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    av -= float(av.mean())
    bv -= float(bv.mean())
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(av.dot(bv) / denom) if denom > 1e-12 else -1.0


def constrained_center_grain_alignment(
    sem_feature: np.ndarray,
    afm_feature: np.ndarray,
    expected_scale: float,
    search_px: float,
    coarse_angle_step_deg: float,
    refine_angle_step_deg: float,
    scale_span: float,
) -> tuple[Candidate, list[Candidate], dict[str, float]]:
    """Find AFM->flipped-SEM affine using grain boundaries, known scale, and near-center prior."""
    target_h, target_w = sem_feature.shape
    afm_h, afm_w = afm_feature.shape
    target_center = np.array([target_w / 2.0, target_h / 2.0], dtype=np.float64)
    sem = robust_rescale(sem_feature, 1.0, 99.5).astype(np.float32)
    afm = robust_rescale(afm_feature, 1.0, 99.5).astype(np.float32)

    def evaluate(
        scale: float,
        angle: float,
        flip_x: bool,
        flip_y: bool,
        center_hint: np.ndarray | None,
        local_search_px: float,
    ) -> tuple[float, Candidate]:
        center = target_center if center_hint is None else center_hint
        affine = build_similarity_affine((afm_h, afm_w), center, scale, angle, flip_x, flip_y)
        warped = cv2.warpAffine(
            afm,
            affine,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            np.ones((afm_h, afm_w), dtype=np.float32),
            affine,
            (target_w, target_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        x0 = max(0, int(round(center[0] - scale * afm_w / 2.0 - local_search_px)))
        x1 = min(target_w, int(round(center[0] + scale * afm_w / 2.0 + local_search_px)))
        y0 = max(0, int(round(center[1] - scale * afm_h / 2.0 - local_search_px)))
        y1 = min(target_h, int(round(center[1] + scale * afm_h / 2.0 + local_search_px)))
        valid = (mask[y0:y1, x0:x1] > 0.5) & np.isfinite(sem[y0:y1, x0:x1])
        score = normalized_cross_correlation(warped[y0:y1, x0:x1], sem[y0:y1, x0:x1], valid)
        candidate = Candidate(
            afm_name=f"afm_height_grain_boundary_scale{scale:.4f}_rot{angle:+.2f}_fx{int(flip_x)}_fy{int(flip_y)}",
            sem_name="flipped_sem_grain_boundary_center_prior",
            matches=0,
            inliers=int(round(max(score, 0.0) * 10000.0)),
            inlier_ratio=max(score, 0.0),
            rmse=1.0 - score,
            affine_afm_to_sem=affine,
            inlier_mask=np.zeros(0, dtype=bool),
            key_sem=np.zeros((0, 2), dtype=np.float32),
            key_afm=np.zeros((0, 2), dtype=np.float32),
        )
        return score, candidate

    coarse_candidates: list[tuple[float, Candidate, float, bool, bool]] = []
    scale_values = np.linspace(1.0 - scale_span, 1.0 + scale_span, 7) * expected_scale
    angle_values = np.arange(-180.0, 180.0 + 0.1 * coarse_angle_step_deg, coarse_angle_step_deg)
    for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
        for scale in scale_values:
            for angle in angle_values:
                score, candidate = evaluate(float(scale), float(angle), flip_x, flip_y, None, search_px)
                coarse_candidates.append((score, candidate, float(angle), flip_x, flip_y))
    coarse_candidates.sort(key=lambda item: item[0], reverse=True)

    refined: list[tuple[float, Candidate]] = []
    for _, coarse, angle0, flip_x, flip_y in coarse_candidates[:12]:
        coarse_center = coarse.affine_afm_to_sem @ np.array([afm_w / 2.0, afm_h / 2.0, 1.0])
        base_scale = math.sqrt(max(abs(coarse.det), 1e-12))
        for scale in np.linspace(base_scale * 0.98, base_scale * 1.02, 7):
            for angle in np.arange(angle0 - 5.0, angle0 + 5.0 + 0.1 * refine_angle_step_deg, refine_angle_step_deg):
                for dx in np.linspace(-search_px * 0.35, search_px * 0.35, 7):
                    for dy in np.linspace(-search_px * 0.35, search_px * 0.35, 7):
                        center = coarse_center + np.array([dx, dy], dtype=np.float64)
                        score, candidate = evaluate(float(scale), float(angle), flip_x, flip_y, center, search_px * 0.55)
                        refined.append((score, candidate))
    refined.sort(key=lambda item: item[0], reverse=True)
    best_score, best = refined[0] if refined else (coarse_candidates[0][0], coarse_candidates[0][1])
    diagnostic_candidates = [item[1] for item in refined[:24]] + [item[1] for item in coarse_candidates[:24]]
    diag = {
        "grain_boundary_ncc": float(best_score),
        "expected_scale": float(expected_scale),
        "search_px": float(search_px),
        "coarse_angle_step_deg": float(coarse_angle_step_deg),
        "refine_angle_step_deg": float(refine_angle_step_deg),
        "scale_span": float(scale_span),
    }
    return best, diagnostic_candidates, diag


def save_sem_ipf_check(path: Path, sem_raw: np.ndarray, sem_flipped: np.ndarray, ipf_sem: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=220, constrained_layout=True)
    axes[0].imshow(sem_raw, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("H5 SEM raw row order")
    axes[1].imshow(sem_flipped, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("H5 SEM flipud: IPF frame")
    axes[2].imshow(np.clip(0.44 * np.dstack([sem_flipped] * 3) + 0.72 * ipf_sem, 0.0, 1.0))
    axes[2].set_title("Flipped SEM + EDAX IPF-Z")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_grain_edge_overlay(path: Path, sem_feature: np.ndarray, afm_feature: np.ndarray, candidate: Candidate) -> None:
    afm_rgb = np.dstack([afm_feature] * 3)
    warped, mask = warp_rgb_to_sem(afm_rgb, candidate.affine_afm_to_sem, sem_feature.shape)
    sem_gray = robust_rescale(sem_feature, 1.0, 99.5)
    overlay = np.dstack(
        [
            np.maximum(sem_gray, warped[..., 0] * mask),
            sem_gray * (1.0 - 0.42 * mask),
            np.maximum(sem_gray, warped[..., 2] * mask),
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=220, constrained_layout=True)
    axes[0].imshow(sem_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Flipped SEM grain-boundary feature")
    axes[1].imshow(warped[..., 0], cmap="gray", vmin=0.0, vmax=1.0, alpha=mask)
    axes[1].set_title("AFM height-edge warped to SEM")
    axes[2].imshow(np.clip(overlay, 0.0, 1.0))
    axes[2].set_title(f"Boundary overlay, NCC={candidate.inlier_ratio:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_candidate_overlay_previews(
    path: Path,
    sem: np.ndarray,
    normal_rgb: np.ndarray,
    candidates: list[Candidate],
    max_candidates: int = 8,
) -> None:
    count = min(max_candidates, len(candidates))
    cols = min(4, count)
    rows_n = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.6 * rows_n), dpi=180, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, candidate in zip(axes_arr, candidates[:count]):
        warped, mask = warp_rgb_to_sem(normal_rgb, candidate.affine_afm_to_sem, sem.shape)
        image = overlay_gray_with_rgba(sem, warped, mask, 0.72)
        ax.imshow(image)
        ax.set_title(
            f"inliers={candidate.inliers}/{candidate.matches}, rmse={candidate.rmse:.2f}\n"
            f"{candidate.afm_name[:30]}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes_arr[count:]:
        ax.axis("off")
    fig.suptitle("Top LightGlue AFM -> 60 deg EBSD SEM candidate overlays", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_alignment_outputs(
    args: argparse.Namespace,
    ebsd: dict[str, Any],
    channels: dict[str, np.ndarray],
    afm_meta: dict[str, Any],
    sem_features: dict[str, np.ndarray],
    sem_grain_feature: np.ndarray,
    afm_grain_feature: np.ndarray,
    afm_features_scaled: dict[str, np.ndarray],
    candidates: list[Candidate],
    lightglue_candidates: list[Candidate],
    expected_scale: float,
    constrained_diag: dict[str, float],
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = candidates[0]
    sem = ebsd["sem"]
    ipf_sem = ebsd["ipf_sem"]
    height_channel = args.height_channel
    if height_channel not in channels:
        raise KeyError(f"AFM channel {height_channel!r} not found. Available: {list(channels)}")
    height_um = channels[height_channel].astype(np.float32) * 1e6
    amp = channels.get("AmplitudeRetrace", next(iter(channels.values())))
    height_norm_warped, afm_mask = warp_to_sem(height_um, best.affine_afm_to_sem, sem.shape)
    amp_warped, _ = warp_to_sem(amp, best.affine_afm_to_sem, sem.shape)

    normal_data = height_to_normals(
        height_um=height_um,
        scan_size_um=float(afm_meta["scan_size_um"]),
        affine_afm_to_sem=best.affine_afm_to_sem,
        smooth_sigma_px=args.normal_smooth_sigma_px,
        level=not args.no_plane_level,
    )
    normal_rgb = normal_direction_rgb(normal_data["normals_sample"], args.tilt_color_ref_deg)
    normal_rgb_warped, normal_mask = warp_rgb_to_sem(normal_rgb, best.affine_afm_to_sem, sem.shape)

    paths = {
        "ebsd60_sem_raw": args.output_dir / "ebsd60_sem_h5_raw_row_order.png",
        "ebsd60_sem": args.output_dir / "ebsd60_sem_h5_flipud_ipf_frame.png",
        "ebsd60_ipf": args.output_dir / "ebsd60_ipf_edax_style_sem_frame.png",
        "sem_ipf_check": args.output_dir / "ebsd60_sem_flipud_ipf_check.png",
        "sem_grain_boundary": args.output_dir / "ebsd60_sem_flipud_grain_boundary_feature.png",
        "afm_grain_boundary": args.output_dir / "afm_height_grain_boundary_feature.png",
        "grain_boundary_overlay": args.output_dir / "afm_sem_grain_boundary_overlay_corrected.png",
        "afm_height": args.output_dir / "afm_height_nm.png",
        "afm_amplitude": args.output_dir / "afm_amplitude.png",
        "afm_normalmap": args.output_dir / "afm_scharr_normalmap.png",
        "afm_normalmap_colorbar": args.output_dir / "afm_scharr_normalmap_with_colorbar.png",
        "afm_normal_tilt": args.output_dir / "afm_normal_tilt_deg.png",
        "afm_normal_azimuth": args.output_dir / "afm_normal_azimuth_deg.png",
        "afm_scharr_dx": args.output_dir / "afm_scharr_dz_dx_um_per_um.png",
        "afm_scharr_dy": args.output_dir / "afm_scharr_dz_dy_um_per_um.png",
        "lightglue_matches": args.output_dir / "lightglue_afm_ebsd60_matches.png",
        "candidate_table": args.output_dir / "lightglue_afm_ebsd60_candidates.csv",
        "afm_height_warped": args.output_dir / "afm_height_warped_to_ebsd60_sem.png",
        "afm_amplitude_warped": args.output_dir / "afm_amplitude_warped_to_ebsd60_sem.png",
        "afm_normalmap_warped": args.output_dir / "afm_normalmap_warped_to_ebsd60_sem.png",
        "afm_height_on_sem": args.output_dir / "afm_height_overlay_on_ebsd60_sem.png",
        "afm_normal_on_sem": args.output_dir / "afm_normalmap_overlay_on_ebsd60_sem.png",
        "afm_normal_on_ipf": args.output_dir / "afm_normalmap_overlay_on_ebsd60_ipf.png",
        "candidate_previews": args.output_dir / "candidate_normalmap_overlay_previews.png",
        "data_npz": args.output_dir / "pt_highres60_afm_alignment_data.npz",
        "metadata": args.output_dir / "pt_highres60_afm_alignment_metadata.json",
    }

    save_gray(paths["ebsd60_sem_raw"], ebsd["sem_raw"], "60 deg EBSD SEM from H5, raw row order")
    save_gray(paths["ebsd60_sem"], sem, "60 deg EBSD SEM from H5")
    save_rgb(paths["ebsd60_ipf"], ipf_sem, "60 deg EDAX-style IPF-Z in SEM frame")
    save_sem_ipf_check(paths["sem_ipf_check"], ebsd["sem_raw"], sem, ipf_sem)
    save_gray(paths["sem_grain_boundary"], sem_grain_feature, "Flipped 60 deg SEM grain-boundary feature")
    save_gray(paths["afm_grain_boundary"], afm_grain_feature, "AFM height grain-boundary feature")
    save_scalar_image(paths["afm_height"], normal_data["height_um"] * 1000.0, "AFM height", "viridis", "height (nm)")
    save_gray(paths["afm_amplitude"], robust_rescale(amp), "AFM AmplitudeRetrace", cmap="magma")
    plt.imsave(paths["afm_normalmap"], normal_rgb)
    save_normalmap_with_legend(
        paths["afm_normalmap_colorbar"],
        normal_rgb,
        "AFM Scharr normalmap in EBSD top-view frame",
        args.tilt_color_ref_deg,
    )
    save_scalar_image(
        paths["afm_normal_tilt"],
        normal_data["tilt_deg"],
        "AFM normal tilt from sample Z",
        "magma",
        "tilt (deg)",
        vmin=0.0,
        vmax=float(np.nanpercentile(normal_data["tilt_deg"], 99.0)),
    )
    save_scalar_image(
        paths["afm_normal_azimuth"],
        normal_data["azimuth_deg"],
        "AFM normal azimuth in EBSD top-view frame",
        "twilight",
        "azimuth (deg)",
        vmin=-180.0,
        vmax=180.0,
    )
    save_scalar_image(
        paths["afm_scharr_dx"],
        normal_data["scharr_dz_dcol"],
        "Scharr dz/dx from AFM depthmap",
        "coolwarm",
        "dz/dx (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 99.0)),
    )
    save_scalar_image(
        paths["afm_scharr_dy"],
        normal_data["scharr_dz_drow"],
        "Scharr dz/dy from AFM depthmap",
        "coolwarm",
        "dz/dy (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_drow"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_drow"], 99.0)),
    )
    if lightglue_candidates:
        diagnostic = lightglue_candidates[0]
        save_match_figure(paths["lightglue_matches"], sem_features[diagnostic.sem_name], afm_features_scaled[diagnostic.afm_name], diagnostic)
    elif paths["lightglue_matches"].exists():
        paths["lightglue_matches"].unlink()
    write_candidates(paths["candidate_table"], candidates)
    save_grain_edge_overlay(paths["grain_boundary_overlay"], sem_grain_feature, afm_grain_feature, best)
    save_gray(paths["afm_height_warped"], height_norm_warped, "AFM height warped to 60 deg EBSD SEM", cmap="viridis")
    save_gray(paths["afm_amplitude_warped"], amp_warped, "AFM amplitude warped to 60 deg EBSD SEM", cmap="magma")
    save_rgb(paths["afm_normalmap_warped"], normal_rgb_warped, "AFM normalmap warped to 60 deg EBSD SEM", normal_mask)
    save_rgb(paths["afm_height_on_sem"], overlay_gray_with_rgba(sem, np.dstack([height_norm_warped] * 3), afm_mask, 0.55), "AFM height overlay on 60 deg EBSD SEM")
    save_rgb(paths["afm_normal_on_sem"], overlay_gray_with_rgba(sem, normal_rgb_warped, normal_mask, 0.70), "AFM normalmap overlay on 60 deg EBSD SEM")
    save_rgb(paths["afm_normal_on_ipf"], overlay_gray_with_rgba(ipf_sem.mean(axis=2), normal_rgb_warped, normal_mask, 0.70), "AFM normalmap overlay on 60 deg IPF")
    save_candidate_overlay_previews(paths["candidate_previews"], sem, normal_rgb, candidates)

    np.savez_compressed(
        paths["data_npz"],
        affine_afm_to_ebsd60_sem=best.affine_afm_to_sem,
        height_um=normal_data["height_um"],
        height_smooth_um=normal_data["height_smooth_um"],
        normals_afm=normal_data["normals_afm"],
        normals_sample=normal_data["normals_sample"],
        normal_rgb=normal_rgb,
        normal_rgb_warped=normal_rgb_warped,
        afm_mask_in_ebsd60_sem=afm_mask,
        scharr_dz_dcol=normal_data["scharr_dz_dcol"],
        scharr_dz_drow=normal_data["scharr_dz_drow"],
        ebsd60_sem=sem,
        ebsd60_sem_raw=ebsd["sem_raw"],
        ebsd60_ipf_sem=ipf_sem,
        sem_grain_boundary=sem_grain_feature,
        afm_grain_boundary=afm_grain_feature,
    )

    metadata = {
        "afm": str(args.afm),
        "afm_metadata": afm_meta,
        "h5": str(args.h5),
        "h5_group": ebsd["spec"].h5_group,
        "angle_deg": args.angle,
        "edax_ipf_reference": str(args.edax_ipf),
        "sem_flip_y_to_match_ipf": ebsd["sem_flip_y_to_match_ipf"],
        "ebsd_grid": {"rows": ebsd["nrows"], "cols": ebsd["ncols"]},
        "ebsd_step_um": {"x": ebsd["step_x_um"], "y": ebsd["step_y_um"]},
        "ebsd_physical_size_um": {"width": ebsd["physical_width_um"], "height": ebsd["physical_height_um"]},
        "expected_afm_to_sem_scale": expected_scale,
        "final_alignment_method": "center-constrained AFM height grain-boundary to flipped SEM grain-boundary NCC",
        "constrained_grain_boundary_search": constrained_diag,
        "lightglue_diagnostic": None
        if not lightglue_candidates
        else {
            "afm_feature": lightglue_candidates[0].afm_name,
            "sem_feature": lightglue_candidates[0].sem_name,
            "matches": lightglue_candidates[0].matches,
            "inliers": lightglue_candidates[0].inliers,
            "inlier_ratio": lightglue_candidates[0].inlier_ratio,
            "rmse_px": lightglue_candidates[0].rmse,
            "center_error_px": center_error_px(lightglue_candidates[0], sem.shape, next(iter(channels.values())).shape),
            "det": lightglue_candidates[0].det,
            "sx": lightglue_candidates[0].sx,
            "sy": lightglue_candidates[0].sy,
        },
        "best_alignment": {
            "afm_feature": best.afm_name,
            "sem_feature": best.sem_name,
            "matches": best.matches,
            "inliers": best.inliers,
            "inlier_ratio": best.inlier_ratio,
            "rmse_px": best.rmse,
            "center_error_px": center_error_px(best, sem.shape, next(iter(channels.values())).shape),
            "det": best.det,
            "sx": best.sx,
            "sy": best.sy,
            "affine_afm_to_ebsd60_sem_2x3": best.affine_afm_to_sem.tolist(),
        },
        "normalmap": {
            "method": "Plane-level AFM height, Scharr dz/dx and dz/dy, normal=(-dz/dx,-dz/dy,1), then rotate XY by AFM->SEM polar rotation.",
            "height_channel": height_channel,
            "smooth_sigma_px": args.normal_smooth_sigma_px,
            "tilt_color_ref_deg": args.tilt_color_ref_deg,
        },
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if key != "metadata" and value.exists()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    ebsd = read_ebsd60(args)
    channels, afm_meta = read_afm_channels(args.afm)
    if args.height_channel not in channels:
        raise KeyError(f"AFM channel {args.height_channel!r} not found. Available: {list(channels)}")
    scan_size_um = float(afm_meta["scan_size_um"])
    first_channel = next(iter(channels.values()))
    scales, expected_scale = choose_match_scales(
        scan_size_um=scan_size_um,
        ebsd_width_um=ebsd["physical_width_um"],
        sem_width_px=ebsd["sem"].shape[1],
        afm_width_px=first_channel.shape[1],
    )
    sem_grain_feature = sem_grain_boundary_feature(ebsd["sem"])
    afm_grain_feature = afm_grain_boundary_feature(channels, args.height_channel)
    sem_features = sem_feature_images(ebsd["sem"])
    sem_features = {key: sem_features[key] for key in ("sem_norm", "sem_hp", "sem_sobel", "sem_canny") if key in sem_features}
    sem_features["sem_grain_boundary"] = sem_grain_feature
    afm_features_scaled, scale_by_name = scaled_afm_feature_images(channels, scales)
    lightglue_candidates: list[Candidate] = []
    if not args.skip_lightglue:
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        extractor, matcher = make_lightglue_models(device, args.max_keypoints)
        raw_candidates = find_best_alignment(
            sem_features=sem_features,
            afm_features=afm_features_scaled,
            extractor=extractor,
            matcher=matcher,
            device=device,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
        )
        candidates = [
            convert_candidate_to_original_afm(candidate, scale_by_name[candidate.afm_name])
            for candidate in raw_candidates
        ]
        lightglue_candidates = rank_candidates_with_scale(candidates, expected_scale)
    constrained_best, constrained_candidates, constrained_diag = constrained_center_grain_alignment(
        sem_feature=sem_grain_feature,
        afm_feature=afm_grain_feature,
        expected_scale=expected_scale,
        search_px=args.center_search_px,
        coarse_angle_step_deg=args.coarse_angle_step_deg,
        refine_angle_step_deg=args.refine_angle_step_deg,
        scale_span=args.scale_span,
    )
    candidates = [constrained_best] + constrained_candidates[:24] + lightglue_candidates
    metadata = save_alignment_outputs(
        args=args,
        ebsd=ebsd,
        channels=channels,
        afm_meta=afm_meta,
        sem_features=sem_features,
        sem_grain_feature=sem_grain_feature,
        afm_grain_feature=afm_grain_feature,
        afm_features_scaled=afm_features_scaled,
        candidates=candidates,
        lightglue_candidates=lightglue_candidates,
        expected_scale=expected_scale,
        constrained_diag=constrained_diag,
    )
    best = constrained_best
    print(f"Saved Pt high-resolution 60 deg AFM alignment to {args.output_dir}")
    print(
        f"Best constrained grain-boundary alignment: {best.afm_name} -> {best.sem_name}, "
        f"NCC={best.inlier_ratio:.3f}, center_error={center_error_px(best, ebsd['sem'].shape, first_channel.shape):.1f}px, "
        f"sx={best.sx:.4f}, sy={best.sy:.4f}, expected={expected_scale:.4f}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align Pt-2 high-resolution AFM to the 60-degree Pt high-resolution EBSD SEM/IPF frame and export AFM Scharr normalmap."
    )
    parser.add_argument("--afm", type=Path, default=DEFAULT_AFM)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--edax-ipf", type=Path, default=DEFAULT_EDAX_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-channel", default="HeightRetrace")
    parser.add_argument("--normal-smooth-sigma-px", type=float, default=1.2)
    parser.add_argument("--tilt-color-ref-deg", type=float, default=12.0)
    parser.add_argument("--no-plane-level", action="store_true")
    parser.add_argument("--flip-sem-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=7.0)
    parser.add_argument("--center-search-px", type=float, default=70.0)
    parser.add_argument("--coarse-angle-step-deg", type=float, default=5.0)
    parser.add_argument("--refine-angle-step-deg", type=float, default=0.5)
    parser.add_argument("--scale-span", type=float, default=0.06)
    parser.add_argument("--skip-lightglue", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force LightGlue/SuperPoint to run on CPU.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
