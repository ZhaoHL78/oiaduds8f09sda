"""Stable global-PC residual plus per-pattern orientation residual calibration.

This is the simplified Pt workflow after the PC/orientation compensation test:

1. Select several high-quality Kikuchi patterns from one EBSD mapping.
2. Use H5 PC/orientation as priors and scan-position PC as the deterministic
   geometry baseline.
3. Fit one shared PC residual for the whole mapping, keeping PC globally stable.
4. With that stable PC fixed per pattern, fit only a small orientation residual.

The objective is the existing detector-validated image score
(preprocessed intensity + band-enhanced NCC). No band-width/profile terms are
used here.
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
    orientation_finetune,
    projection_with_pc,
    scan_position_pc,
    score_orientation_matrix,
    write_orientation_finetune_trace,
)
from project_edax_oim_to_sphere import EdaxMapInputs, build_master_lon_colat, read_edax_inputs
from single_kikuchi_pc_finetune import (
    build_master_samplers,
    build_preprocessed_images,
    choose_orientation_matrix,
    project_crystal_patch,
)


DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_stable_global_pc_orientation"


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
    scan_score: tuple[float, float, float]


@dataclass(frozen=True)
class GlobalPcResult:
    delta_pc: tuple[float, float, float]
    mean_intensity_score: float
    mean_band_score: float
    mean_combined_score: float
    prior_penalty: float
    objective: float


@dataclass(frozen=True)
class OrientationResult:
    stable_pc: tuple[float, float, float]
    stable_score: tuple[float, float, float]
    orientation_matrix: np.ndarray
    orientation_best: dict[str, float]
    orientation_trace: list[dict[str, float]]


def safe_key(text: str) -> str:
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_"))


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


def select_mapping(args: argparse.Namespace) -> dict[str, Any]:
    matched = matched_pt_maps(args.h5, args.up2_root or [DEFAULT_UP2_ROOT], args.specimen)
    if args.area:
        matched = [row for row in matched if row["area"] == args.area]
    if args.map_group:
        matched = [row for row in matched if row["h5_path"].strip("/") == args.map_group.strip("/")]
    if not matched:
        raise RuntimeError("No matched Pt H5/UP2 mapping found.")
    return matched[0]


def build_case(args: argparse.Namespace, h5_row: dict[str, Any], selection: dict[str, Any], samplers) -> PatternCase:
    pattern_index = int(selection["pattern_index"])
    key = safe_key(f"{h5_row['specimen']}_{h5_row['area']}_{h5_row['map_name']}_idx{pattern_index}")
    raw_projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=Path(h5_row["up2_actual_path"]),
            map_group=h5_row["h5_path"],
            pattern_index=pattern_index,
        )
    )
    map_pc = raw_projection.pc_edax
    scan_pc, scan_meta = scan_position_pc(args, h5_row["h5_path"], pattern_index, map_pc)
    projection = projection_with_pc(raw_projection, scan_pc)
    mask, circle, mask_mode = detector_mask_for_pattern(projection, args)
    images = build_preprocessed_images(projection.pattern, mask)
    orientation_name, base_matrix, _rows = choose_orientation_matrix(
        projection=projection,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    scan_score = score_orientation_matrix(projection, scan_pc, base_matrix, mask, images, samplers, args)
    return PatternCase(
        key=key,
        h5_row=h5_row,
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
        scan_score=scan_score,
    )


def score_global_pc_delta(
    cases: list[PatternCase],
    delta_pc: tuple[float, float, float],
    args: argparse.Namespace,
    samplers,
) -> GlobalPcResult:
    intensity_scores: list[float] = []
    band_scores: list[float] = []
    combined_scores: list[float] = []
    for case in cases:
        pc = tuple(float(case.scan_pc[i] + delta_pc[i]) for i in range(3))
        if pc[2] <= 0.05:
            return GlobalPcResult(delta_pc, -np.inf, -np.inf, -np.inf, np.inf, -np.inf)
        intensity, band, combined = score_orientation_matrix(
            case.projection,
            pc,
            case.base_matrix,
            case.mask,
            case.images,
            samplers,
            args,
        )
        intensity_scores.append(intensity)
        band_scores.append(band)
        combined_scores.append(combined)
    scales = np.asarray(args.global_pc_prior_scale, dtype=np.float64)
    delta = np.asarray(delta_pc, dtype=np.float64)
    prior = float(np.mean((delta / (scales + 1e-12)) ** 2))
    mean_intensity = float(np.mean(intensity_scores))
    mean_band = float(np.mean(band_scores))
    mean_combined = float(np.mean(combined_scores))
    objective = mean_combined - args.global_pc_prior_weight * prior
    return GlobalPcResult(delta_pc, mean_intensity, mean_band, mean_combined, prior, objective)


def fit_global_pc_residual(cases: list[PatternCase], args: argparse.Namespace, samplers) -> tuple[GlobalPcResult, list[GlobalPcResult]]:
    ranges = np.asarray(args.global_pc_range, dtype=np.float64)
    steps = np.asarray(args.global_pc_step, dtype=np.float64)
    axes = [np.arange(-rng, rng + 0.5 * step, step, dtype=np.float64) for rng, step in zip(ranges, steps)]
    rows: list[GlobalPcResult] = []
    best: GlobalPcResult | None = None
    for dx in axes[0]:
        for dy in axes[1]:
            for dz in axes[2]:
                result = score_global_pc_delta(cases, (float(dx), float(dy), float(dz)), args, samplers)
                rows.append(result)
                if best is None or result.objective > best.objective:
                    best = result
    if best is None:
        raise RuntimeError("No valid global PC residual candidate.")
    return best, rows


def fit_orientation_after_global_pc(
    case: PatternCase,
    global_pc: GlobalPcResult,
    args: argparse.Namespace,
    samplers,
) -> OrientationResult:
    stable_pc = tuple(float(case.scan_pc[i] + global_pc.delta_pc[i]) for i in range(3))
    stable_score = score_orientation_matrix(
        case.projection,
        stable_pc,
        case.base_matrix,
        case.mask,
        case.images,
        samplers,
        args,
    )
    matrix, best, trace = orientation_finetune(
        projection=case.projection,
        pc=stable_pc,
        base_matrix=case.base_matrix,
        mask=case.mask,
        images=case.images,
        samplers=samplers,
        args=args,
    )
    return OrientationResult(
        stable_pc=stable_pc,
        stable_score=stable_score,
        orientation_matrix=matrix,
        orientation_best=best,
        orientation_trace=trace,
    )


def imshow_patch(ax, master_texture: np.ndarray, patch: tuple[np.ndarray, np.ndarray], title: str) -> None:
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
    global_pc: GlobalPcResult,
    orientation: OrientationResult,
    master_texture: np.ndarray,
) -> None:
    scan_patch = project_crystal_patch(case.projection, case.scan_pc, case.base_matrix, case.images["enhanced"], case.mask)
    stable_patch = project_crystal_patch(
        case.projection,
        orientation.stable_pc,
        case.base_matrix,
        case.images["enhanced"],
        case.mask,
    )
    final_patch = project_crystal_patch(
        case.projection,
        orientation.stable_pc,
        orientation.orientation_matrix,
        case.images["enhanced"],
        case.mask,
    )

    fig = plt.figure(figsize=(20, 12))
    axes = [fig.add_subplot(2, 5, i + 1) for i in range(10)]
    axes[0].imshow(case.projection.pattern, cmap="gray")
    axes[0].contour(case.mask, levels=[0.5], colors=["#ff3030"], linewidths=0.8)
    axes[0].set_title(f"Raw + full mask\ncircle=({case.circle[0]}, {case.circle[1]}, r={case.circle[2]})")
    axes[0].axis("off")

    axes[1].imshow(np.where(case.mask, case.images["enhanced"], np.nan), cmap="gray")
    axes[1].set_title("Preprocessed Kikuchi")
    axes[1].axis("off")

    axes[2].imshow(np.where(case.mask, case.images["band"], np.nan), cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Band-enhanced image for NCC")
    axes[2].axis("off")

    imshow_patch(axes[3], master_texture, scan_patch, f"Start: scan-position PC\nscore={case.scan_score[2]:+.5f}")
    imshow_patch(axes[4], master_texture, stable_patch, f"Stable global PC residual\nscore={orientation.stable_score[2]:+.5f}")
    imshow_patch(
        axes[5],
        master_texture,
        final_patch,
        "Final orientation residual\n"
        f"score={orientation.orientation_best['combined_score']:+.5f}",
    )

    overlay = np.zeros((*scan_patch[1].shape, 4), dtype=np.float32)
    overlay[..., 0] = np.where(scan_patch[1], 1.0, 0.0)
    overlay[..., 1] = np.where(final_patch[1], 0.85, 0.0)
    overlay[..., 2] = np.where(stable_patch[1], 1.0, 0.0)
    overlay[..., 3] = np.where(scan_patch[1] | stable_patch[1] | final_patch[1], 0.65, 0.0)
    axes[6].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[6].imshow(overlay, origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[6].set_title("Footprint shift\nred=start, cyan=global PC, yellow=final")
    axes[6].set_xlabel("longitude (deg)")
    axes[6].set_ylabel("colatitude (deg)")

    labels = ["scan PC", "global PC", "PC+ori"]
    scores = [case.scan_score[2], orientation.stable_score[2], orientation.orientation_best["combined_score"]]
    axes[7].bar(labels, scores, color=["#ffbf00", "#1f77b4", "#2ca02c"])
    axes[7].set_title("Stage-wise image NCC score")
    axes[7].tick_params(axis="x", rotation=15)

    trace = orientation.orientation_trace
    if trace:
        x = np.arange(len(trace))
        combined = np.array([row["combined_score"] for row in trace], dtype=np.float64)
        axes[8].plot(x, combined, color="#6a5acd", alpha=0.45)
        axes[8].plot(x, np.maximum.accumulate(combined), color="black", lw=1.1)
        axes[8].scatter([int(np.argmax(combined))], [float(np.max(combined))], c="red", s=35)
    axes[8].set_title("Orientation residual search trace")
    axes[8].set_xlabel("evaluation")
    axes[8].set_ylabel("combined score")

    axes[9].axis("off")
    scan_shift = np.asarray(case.scan_pc) - np.asarray(case.map_pc)
    pc_delta = np.asarray(global_pc.delta_pc)
    rot = (
        orientation.orientation_best["rot_x_deg"],
        orientation.orientation_best["rot_y_deg"],
        orientation.orientation_best["rot_z_deg"],
    )
    summary = (
        f"map: {case.h5_row['area']} / {case.h5_row['map_name']}\n"
        f"idx: {case.selection['pattern_index']}, row={case.selection['row']}, col={case.selection['col']}\n"
        f"orientation variant: {case.orientation_name}\n"
        f"map PC:  ({case.map_pc[0]:.6f}, {case.map_pc[1]:.6f}, {case.map_pc[2]:.6f})\n"
        f"scan PC: ({case.scan_pc[0]:.6f}, {case.scan_pc[1]:.6f}, {case.scan_pc[2]:.6f})\n"
        f"scan-position shift: ({scan_shift[0]:+.6f}, {scan_shift[1]:+.6f}, {scan_shift[2]:+.6f})\n"
        f"global PC residual: ({pc_delta[0]:+.6f}, {pc_delta[1]:+.6f}, {pc_delta[2]:+.6f}), |dPC|={norm3(pc_delta):.6f}\n"
        f"stable PC: ({orientation.stable_pc[0]:.6f}, {orientation.stable_pc[1]:.6f}, {orientation.stable_pc[2]:.6f})\n"
        f"orientation residual: ({rot[0]:+.3f}, {rot[1]:+.3f}, {rot[2]:+.3f}) deg, |dR|={norm3(rot):.3f} deg\n"
        f"score: scan={case.scan_score[2]:+.5f}, globalPC={orientation.stable_score[2]:+.5f}, final={orientation.orientation_best['combined_score']:+.5f}"
    )
    axes[9].text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=9.4)

    fig.suptitle("Stable global PC residual -> per-pattern orientation residual", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_global_visualization(
    path: Path,
    cases: list[PatternCase],
    global_pc: GlobalPcResult,
    grid_rows: list[GlobalPcResult],
    orientation_results: dict[str, OrientationResult],
) -> None:
    dz_values = np.array([row.delta_pc[2] for row in grid_rows], dtype=np.float64)
    best_dz = dz_values[np.argmin(np.abs(dz_values - global_pc.delta_pc[2]))]
    slab = [row for row in grid_rows if abs(row.delta_pc[2] - best_dz) < 1e-12]
    xs = sorted(set(row.delta_pc[0] for row in slab))
    ys = sorted(set(row.delta_pc[1] for row in slab))
    score_grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    for row in slab:
        ix = xs.index(row.delta_pc[0])
        iy = ys.index(row.delta_pc[1])
        score_grid[iy, ix] = row.objective

    fig = plt.figure(figsize=(17, 11))
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]
    im = axes[0].imshow(
        score_grid,
        origin="lower",
        extent=[min(xs), max(xs), min(ys), max(ys)],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].scatter([global_pc.delta_pc[0]], [global_pc.delta_pc[1]], marker="x", c="red", s=90, lw=2)
    axes[0].set_title(f"Global PC residual objective slice\nnear dPCz={best_dz:+.5f}")
    axes[0].set_xlabel("delta PCx")
    axes[0].set_ylabel("delta PCy")
    fig.colorbar(im, ax=axes[0], label="objective")

    rows = [case.selection["row"] for case in cases]
    cols = [case.selection["col"] for case in cases]
    axes[1].scatter(cols, rows, c=np.arange(len(cases)), cmap="tab10", s=55)
    axes[1].invert_yaxis()
    axes[1].set_title("Selected Kikuchi positions in mapping")
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("row")

    scan_pc = np.array([case.scan_pc for case in cases], dtype=np.float64)
    stable_pc = scan_pc + np.asarray(global_pc.delta_pc, dtype=np.float64)
    axes[2].scatter(scan_pc[:, 0], scan_pc[:, 1], label="scan-position PC", c="#ffbf00")
    axes[2].scatter(stable_pc[:, 0], stable_pc[:, 1], label="stable global PC", c="#1f77b4")
    for a, b in zip(scan_pc, stable_pc):
        axes[2].plot([a[0], b[0]], [a[1], b[1]], color="gray", lw=0.8, alpha=0.8)
    axes[2].set_title("PC stability across selected patterns")
    axes[2].set_xlabel("PCx")
    axes[2].set_ylabel("PCy")
    axes[2].legend()

    x = np.arange(len(cases))
    scan_scores = [case.scan_score[2] for case in cases]
    stable_scores = [orientation_results[case.key].stable_score[2] for case in cases]
    final_scores = [orientation_results[case.key].orientation_best["combined_score"] for case in cases]
    width = 0.24
    axes[3].bar(x - width, scan_scores, width, label="scan PC")
    axes[3].bar(x, stable_scores, width, label="global PC")
    axes[3].bar(x + width, final_scores, width, label="PC+ori")
    axes[3].set_xticks(x, [str(case.selection["pattern_index"]) for case in cases], rotation=25)
    axes[3].set_title("Per-pattern score progression")
    axes[3].set_ylabel("combined image NCC")
    axes[3].legend()

    rot = np.array(
        [
            [
                orientation_results[case.key].orientation_best["rot_x_deg"],
                orientation_results[case.key].orientation_best["rot_y_deg"],
                orientation_results[case.key].orientation_best["rot_z_deg"],
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    axes[4].plot(x, rot[:, 0], "o-", label="rx")
    axes[4].plot(x, rot[:, 1], "o-", label="ry")
    axes[4].plot(x, rot[:, 2], "o-", label="rz")
    axes[4].axhline(0, color="black", lw=0.8)
    axes[4].set_xticks(x, [str(case.selection["pattern_index"]) for case in cases], rotation=25)
    axes[4].set_title("Per-pattern orientation residual")
    axes[4].set_ylabel("degree")
    axes[4].legend()

    axes[5].axis("off")
    summary = (
        f"global PC residual: ({global_pc.delta_pc[0]:+.6f}, {global_pc.delta_pc[1]:+.6f}, {global_pc.delta_pc[2]:+.6f})\n"
        f"|dPC|={norm3(global_pc.delta_pc):.6f}\n"
        f"mean combined score: {global_pc.mean_combined_score:+.5f}\n"
        f"prior penalty: {global_pc.prior_penalty:.5f}\n"
        f"objective: {global_pc.objective:+.5f}\n\n"
        "Interpretation:\n"
        "PC residual is mapping-level and shared by all selected patterns.\n"
        "Orientation residual is per-pattern and only applied after PC is fixed.\n"
        "This keeps PC stable and prevents per-pattern PC/orientation compensation."
    )
    axes[5].text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle("Stable global PC residual diagnostic", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = min(2, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 8.0, rows_n * 4.8), dpi=160)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, row in zip(axes_arr.ravel(), rows):
        image = plt.imread(row["figure"])
        ax.imshow(image)
        ax.set_title(
            f"{row['area']} idx={row['pattern_index']} "
            f"score {float(row['scan_combined_score']):+.3f}->{float(row['final_combined_score']):+.3f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Stable global PC + orientation residual calibration", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samplers = build_master_samplers(args.master)
    master_texture = build_master_lon_colat(samplers.upper_corrected, samplers.lower_corrected)
    h5_row = select_mapping(args)
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
        raise RuntimeError("No selected Kikuchi patterns passed the filters.")

    cases: list[PatternCase] = []
    for idx, selection in enumerate(selections, start=1):
        print(f"[build {idx}/{len(selections)}] {h5_row['area']} idx={selection['pattern_index']}")
        cases.append(build_case(args, h5_row, selection, samplers))

    print("[fit] global PC residual shared across selected patterns")
    global_pc, pc_grid = fit_global_pc_residual(cases, args, samplers)
    write_csv(
        args.output_dir / "global_pc_residual_grid.csv",
        [
            {
                "delta_pcx": row.delta_pc[0],
                "delta_pcy": row.delta_pc[1],
                "delta_pcz": row.delta_pc[2],
                "mean_intensity_score": row.mean_intensity_score,
                "mean_band_score": row.mean_band_score,
                "mean_combined_score": row.mean_combined_score,
                "pc_prior_penalty": row.prior_penalty,
                "objective": row.objective,
            }
            for row in pc_grid
        ],
    )

    orientation_results: dict[str, OrientationResult] = {}
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[orientation {idx}/{len(cases)}] {case.key}")
        orientation = fit_orientation_after_global_pc(case, global_pc, args, samplers)
        orientation_results[case.key] = orientation
        case_dir = args.output_dir / "per_pattern" / case.key
        write_orientation_finetune_trace(case_dir / "orientation_residual_trace.csv", orientation.orientation_trace)
        figure = case_dir / f"{case.key}_stable_global_pc_orientation.png"
        save_case_visualization(figure, case, global_pc, orientation, master_texture)
        pc_delta = np.asarray(global_pc.delta_pc, dtype=np.float64)
        rot = (
            orientation.orientation_best["rot_x_deg"],
            orientation.orientation_best["rot_y_deg"],
            orientation.orientation_best["rot_z_deg"],
        )
        rows.append(
            {
                "specimen": h5_row["specimen"],
                "area": h5_row["area"],
                "map_name": h5_row["map_name"],
                "h5_path": h5_row["h5_path"],
                "up2_file": h5_row["up2_display_name"],
                "pattern_index": case.selection["pattern_index"],
                "row": case.selection["row"],
                "col": case.selection["col"],
                "IQ": case.selection["IQ"],
                "CI": case.selection["CI"],
                "orientation_variant": case.orientation_name,
                "map_pc_x": case.map_pc[0],
                "map_pc_y": case.map_pc[1],
                "map_pc_z": case.map_pc[2],
                "scan_pc_x": case.scan_pc[0],
                "scan_pc_y": case.scan_pc[1],
                "scan_pc_z": case.scan_pc[2],
                "global_delta_pcx": global_pc.delta_pc[0],
                "global_delta_pcy": global_pc.delta_pc[1],
                "global_delta_pcz": global_pc.delta_pc[2],
                "global_delta_pc_norm": norm3(pc_delta),
                "stable_pc_x": orientation.stable_pc[0],
                "stable_pc_y": orientation.stable_pc[1],
                "stable_pc_z": orientation.stable_pc[2],
                "scan_intensity_score": case.scan_score[0],
                "scan_band_score": case.scan_score[1],
                "scan_combined_score": case.scan_score[2],
                "stable_intensity_score": orientation.stable_score[0],
                "stable_band_score": orientation.stable_score[1],
                "stable_combined_score": orientation.stable_score[2],
                "orientation_rot_x_deg": rot[0],
                "orientation_rot_y_deg": rot[1],
                "orientation_rot_z_deg": rot[2],
                "orientation_rot_norm_deg": norm3(rot),
                "final_intensity_score": orientation.orientation_best["intensity_score"],
                "final_band_score": orientation.orientation_best["band_score"],
                "final_combined_score": orientation.orientation_best["combined_score"],
                "global_pc_score_gain": orientation.stable_score[2] - case.scan_score[2],
                "orientation_score_gain": orientation.orientation_best["combined_score"] - orientation.stable_score[2],
                "total_score_gain": orientation.orientation_best["combined_score"] - case.scan_score[2],
                "figure": str(figure),
            }
        )

    write_csv(args.output_dir / "stable_global_pc_orientation_summary.csv", rows)
    save_global_visualization(args.output_dir / "stable_global_pc_diagnostic.png", cases, global_pc, pc_grid, orientation_results)
    save_contact_sheet(args.output_dir / "stable_global_pc_orientation_contact_sheet.png", rows)
    print(f"Global PC residual: {global_pc.delta_pc}; objective={global_pc.objective:+.6f}")
    print(f"Summary: {args.output_dir / 'stable_global_pc_orientation_summary.csv'}")
    print(f"Contact sheet: {args.output_dir / 'stable_global_pc_orientation_contact_sheet.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one stable mapping-level PC residual, then per-pattern orientation residuals."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", action="append", type=Path, default=None)
    parser.add_argument("--specimen", default="Pt-3")
    parser.add_argument("--area", default="Area 3-90")
    parser.add_argument("--map-group", default="")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern-count", type=int, default=5)
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
    parser.add_argument("--global-pc-range", nargs=3, type=float, default=(0.006, 0.006, 0.016))
    parser.add_argument("--global-pc-step", nargs=3, type=float, default=(0.002, 0.002, 0.004))
    parser.add_argument("--global-pc-prior-scale", nargs=3, type=float, default=(0.010, 0.010, 0.020))
    parser.add_argument("--global-pc-prior-weight", type=float, default=0.030)
    parser.set_defaults(orientation_finetune=True)
    parser.add_argument("--orientation-finetune", dest="orientation_finetune", action="store_true")
    parser.add_argument("--no-orientation-finetune", dest="orientation_finetune", action="store_false")
    parser.add_argument("--orientation-bound-deg", type=float, default=0.5)
    parser.add_argument("--orientation-step-deg", type=float, default=0.05)
    parser.add_argument("--orientation-steps", type=int, default=7)
    parser.add_argument("--orientation-fine-steps", type=int, default=9)
    parser.add_argument("--orientation-regularization-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
