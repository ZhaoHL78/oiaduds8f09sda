"""Batch spherical calibration for selected Pt Kikuchi patterns.

This is the fixed, reusable version of the single-pattern workflow:

1. match Pt H5 EBSD maps to local UP2 stacks;
2. select high-IQ/high-CI Kikuchi patterns;
3. run full-disk preprocessing, scan-position PC correction, residual PC finetune,
   and residual orientation finetune;
4. save one detailed stage-wise visualization per pattern plus summary sheets.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from export_h5_mapping_sem_correspondence import (
    collect_h5_maps,
    collect_up2_candidates,
    match_h5_to_up2,
    safe_name,
)
from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    build_master_lon_colat,
    estimate_circular_detector_mask,
    read_edax_inputs,
)
from single_kikuchi_pc_finetune import (
    build_master_samplers,
    build_preprocessed_images,
    centered_circular_detector_mask,
    choose_orientation_matrix,
    detector_directions_with_pc,
    make_stride_indices,
    pc_finetune,
    project_crystal_patch,
    project_detector_patch,
    save_visualization,
    score_with_directions,
    write_orientation_scores,
    write_pc_scores,
    write_summary,
)
from visualize_scan_position_pc_correction import (
    adjusted_pc_from_scan_position,
    index_to_scan_offset_um,
    read_scan_geometry,
)


DEFAULT_H5 = Path(r"D:\EBSD-data\Pt-1\20251209Pt.edaxh5")
DEFAULT_UP2_ROOT = Path(r"D:\EBSD-data")
DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_batch_kikuchi_spherical_calibration"


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def scan_shape(map_group: h5py.Group) -> tuple[int, int]:
    nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
    ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
    return nrows, ncols


def circular_mask_from_circle(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def detector_mask_for_pattern(projection, args: argparse.Namespace) -> tuple[np.ndarray, tuple[int, int, int], str]:
    if args.mask_mode == "estimated":
        try:
            _mask, circle = estimate_circular_detector_mask(projection.pattern)
            circle = tuple(int(v) for v in circle)
            return circular_mask_from_circle(projection.pattern.shape, circle), circle, "estimated_hough_full_kikuchi_disk"
        except Exception as exc:
            print(f"Warning: Hough disk estimation failed for {projection.map_name}: {exc}; using centered fallback.")
    mask, circle = centered_circular_detector_mask(projection.pattern.shape, args.mask_radius_fraction)
    return mask, circle, f"centered_full_disk_fraction_{args.mask_radius_fraction:.3f}"


def projection_with_pc(projection, pc_edax: tuple[float, float, float]):
    return replace(
        projection,
        pc_edax=pc_edax,
        detector_directions=detector_directions_with_pc(projection, pc_edax),
    )


def scan_position_pc(
    args: argparse.Namespace,
    map_group: str,
    pattern_index: int,
    map_pc: tuple[float, float, float],
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    if args.pc_initial == "map":
        return map_pc, {
            "pc_initial_mode": "map",
            "scan_x_um": 0.0,
            "scan_y_um": 0.0,
            "scan_pc_available": False,
            "scan_pc_note": "Using map-level EDAX PC without scan-position correction.",
        }
    try:
        geometry = read_scan_geometry(args.h5, map_group)
        x_um, y_um = index_to_scan_offset_um(pattern_index, geometry)
        pc = adjusted_pc_from_scan_position(
            pattern_index,
            geometry,
            x_sign=args.pc_x_sign,
            y_sign=args.pc_y_sign,
            y_scale=args.pc_y_scale,
            z_sign=args.pc_z_sign,
            z_scale=args.pc_z_scale,
        )
        return pc, {
            "pc_initial_mode": "scan_position",
            "scan_x_um": x_um,
            "scan_y_um": y_um,
            "scan_pc_available": True,
            "scan_pc_note": "PC initial value corrected from EBSD scan position.",
            "step_x_um": geometry.step_x_um,
            "step_y_um": geometry.step_y_um,
            "detector_diameter_mm": geometry.detector_diameter_mm,
            "grid_type": geometry.grid_type,
        }
    except Exception as exc:
        return map_pc, {
            "pc_initial_mode": "map_fallback",
            "scan_x_um": 0.0,
            "scan_y_um": 0.0,
            "scan_pc_available": False,
            "scan_pc_note": f"Scan-position PC unavailable ({exc}); using map-level EDAX PC.",
        }


def rotation_matrix_from_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    return Rotation.from_rotvec(np.deg2rad([rx, ry, rz])).as_matrix()


def score_orientation_matrix(
    projection,
    pc: tuple[float, float, float],
    matrix: np.ndarray,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    indices = make_stride_indices(mask, args.stride)
    detector_directions = detector_directions_with_pc(projection, pc)
    return score_with_directions(
        detector_directions=detector_directions,
        matrix=matrix,
        indices=indices,
        exp_corrected_values=images["enhanced"].ravel()[indices],
        exp_band_values=images["band"].ravel()[indices],
        samplers=samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )


def orientation_finetune(
    projection,
    pc: tuple[float, float, float],
    base_matrix: np.ndarray,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]]]:
    zero_score = score_orientation_matrix(projection, pc, base_matrix, mask, images, samplers, args)
    trace: list[dict[str, float]] = [
        {
            "stage": "initial",
            "rot_x_deg": 0.0,
            "rot_y_deg": 0.0,
            "rot_z_deg": 0.0,
            "intensity_score": zero_score[0],
            "band_score": zero_score[1],
            "combined_score": zero_score[2],
            "objective": zero_score[2],
        }
    ]
    best = trace[0].copy()

    if not args.orientation_finetune:
        return base_matrix, best, trace

    def evaluate(stage: str, rx: float, ry: float, rz: float) -> dict[str, float]:
        delta = rotation_matrix_from_deg(rx, ry, rz)
        matrix = base_matrix @ delta.T
        intensity, band, combined = score_orientation_matrix(projection, pc, matrix, mask, images, samplers, args)
        if args.orientation_bound_deg > 0:
            penalty = args.orientation_regularization_weight * float(
                np.mean(np.square(np.array([rx, ry, rz], dtype=np.float64) / args.orientation_bound_deg))
            )
        else:
            penalty = 0.0
        return {
            "stage": stage,
            "rot_x_deg": float(rx),
            "rot_y_deg": float(ry),
            "rot_z_deg": float(rz),
            "intensity_score": intensity,
            "band_score": band,
            "combined_score": combined,
            "objective": combined - penalty,
        }

    coarse_offsets = np.linspace(-args.orientation_bound_deg, args.orientation_bound_deg, args.orientation_steps)
    for rx in coarse_offsets:
        for ry in coarse_offsets:
            for rz in coarse_offsets:
                if rx == 0 and ry == 0 and rz == 0:
                    continue
                row = evaluate("orientation_coarse", float(rx), float(ry), float(rz))
                trace.append(row)
                if row["objective"] > best["objective"]:
                    best = row.copy()

    if args.orientation_steps > 1:
        fine_span = 2.0 * args.orientation_bound_deg / (args.orientation_steps - 1)
    else:
        fine_span = args.orientation_bound_deg * 0.5
    fine_offsets = np.linspace(-fine_span, fine_span, args.orientation_fine_steps)
    center = np.array([best["rot_x_deg"], best["rot_y_deg"], best["rot_z_deg"]], dtype=np.float64)
    for dx in fine_offsets:
        for dy in fine_offsets:
            for dz in fine_offsets:
                rot = center + np.array([dx, dy, dz], dtype=np.float64)
                rot = np.clip(rot, -args.orientation_bound_deg, args.orientation_bound_deg)
                row = evaluate("orientation_fine", float(rot[0]), float(rot[1]), float(rot[2]))
                trace.append(row)
                if row["objective"] > best["objective"]:
                    best = row.copy()

    best_delta = rotation_matrix_from_deg(best["rot_x_deg"], best["rot_y_deg"], best["rot_z_deg"])
    return base_matrix @ best_delta.T, best, trace


def choose_quality_indices(
    map_group: h5py.Group,
    count: int,
    patterns_per_map: int,
    ci_min: float,
    min_scan_distance_fraction: float,
) -> list[dict[str, Any]]:
    data = map_group["EBSD/ANG/DATA/DATA"]
    iq = data["IQ"][:].astype(np.float64)
    ci = data["CI"][:].astype(np.float64)
    phase = data["Phase"][:].astype(np.int32) if "Phase" in data.dtype.names else np.zeros(count, dtype=np.int32)
    valid = data["Valid"][:].astype(bool) if "Valid" in data.dtype.names else np.ones(count, dtype=bool)
    fit = data["Fit"][:].astype(np.float64) if "Fit" in data.dtype.names else np.full(count, np.nan)

    score = 0.72 * normalize(iq) + 0.28 * np.clip(ci, 0.0, 1.0)
    score = np.where(valid & np.isfinite(iq) & np.isfinite(ci) & (ci >= ci_min), score, -np.inf)
    order = np.argsort(score)[::-1]
    nrows, ncols = scan_shape(map_group)
    min_distance = max(0.0, min_scan_distance_fraction * min(nrows, ncols))
    selected: list[dict[str, Any]] = []

    for index in order:
        index = int(index)
        if not np.isfinite(score[index]):
            continue
        row = index // ncols
        col = index % ncols
        if min_distance > 0 and any(math.hypot(row - item["row"], col - item["col"]) < min_distance for item in selected):
            continue
        selected.append(
            {
                "pattern_index": index,
                "row": row,
                "col": col,
                "IQ": float(iq[index]),
                "CI": float(ci[index]),
                "Fit": float(fit[index]) if np.isfinite(fit[index]) else "",
                "Phase": int(phase[index]),
                "selection_score": float(score[index]),
            }
        )
        if len(selected) >= patterns_per_map:
            break

    if len(selected) < patterns_per_map and min_distance > 0:
        used = {item["pattern_index"] for item in selected}
        for index in order:
            index = int(index)
            if index in used or not np.isfinite(score[index]):
                continue
            row = index // ncols
            col = index % ncols
            selected.append(
                {
                    "pattern_index": index,
                    "row": row,
                    "col": col,
                    "IQ": float(iq[index]),
                    "CI": float(ci[index]),
                    "Fit": float(fit[index]) if np.isfinite(fit[index]) else "",
                    "Phase": int(phase[index]),
                    "selection_score": float(score[index]),
                }
            )
            if len(selected) >= patterns_per_map:
                break
    return selected


def matched_pt_maps(h5_path: Path, up2_roots: list[Path], specimen: str | None) -> list[dict[str, Any]]:
    h5_rows = collect_h5_maps(h5_path, specimen_filter=specimen)
    up2_candidates = collect_up2_candidates(up2_roots)
    rows, _unmatched = match_h5_to_up2(h5_rows, up2_candidates)
    matched = [row for row in rows if row["match_status"].startswith("matched") and row["up2_actual_path"]]
    return sorted(matched, key=lambda row: (row["specimen"], row["scan_time"], row["h5_path"]))


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


def one_pattern_args(
    args: argparse.Namespace,
    up2_path: Path,
    map_group: str,
    pattern_index: int,
    output_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        h5=args.h5,
        up2=up2_path,
        map_group=map_group,
        pattern_index=pattern_index,
        master=args.master,
        output_dir=output_dir,
    )


def write_orientation_finetune_trace(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def imshow_master_patch(ax, master_texture: np.ndarray, patch: tuple[np.ndarray, np.ndarray], title: str) -> None:
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


def save_position_pc_orientation_visualization(
    path: Path,
    projection,
    images: dict[str, np.ndarray],
    mask: np.ndarray,
    circle: tuple[int, int, int],
    master_texture: np.ndarray,
    matrix: np.ndarray,
    orientation_matrix: np.ndarray,
    map_pc: tuple[float, float, float],
    scan_pc: tuple[float, float, float],
    refined_pc: tuple[float, float, float],
    map_score: tuple[float, float, float],
    scan_score: tuple[float, float, float],
    refined_score: tuple[float, float, float],
    orientation_best: dict[str, float],
    orientation_trace: list[dict[str, float]],
    scan_meta: dict[str, Any],
) -> None:
    map_patch = project_crystal_patch(projection, map_pc, matrix, images["enhanced"], mask)
    scan_patch = project_crystal_patch(projection, scan_pc, matrix, images["enhanced"], mask)
    pc_patch = project_crystal_patch(projection, refined_pc, matrix, images["enhanced"], mask)
    orientation_patch = project_crystal_patch(projection, refined_pc, orientation_matrix, images["enhanced"], mask)

    fig = plt.figure(figsize=(20, 15))
    axes = [fig.add_subplot(3, 4, index + 1) for index in range(12)]

    axes[0].imshow(projection.pattern, cmap="gray")
    axes[0].contour(mask, levels=[0.5], colors=["#ff3030"], linewidths=0.8)
    axes[0].set_title(f"Raw UP2 + full Kikuchi mask\ncircle=({circle[0]}, {circle[1]}, r={circle[2]})")
    axes[0].axis("off")

    for ax, image, title, cmap in [
        (axes[1], images["corrected"], "Background corrected", "gray"),
        (axes[2], images["enhanced"], "CLAHE contrast enhanced", "gray"),
        (axes[3], images["band"], "Band enhanced for scoring", "magma"),
    ]:
        ax.imshow(np.where(mask, image, np.nan), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    imshow_master_patch(axes[4], master_texture, map_patch, f"Map PC on master\nscore={map_score[2]:+.5f}")
    imshow_master_patch(axes[5], master_texture, scan_patch, f"Scan-position PC\nscore={scan_score[2]:+.5f}")
    imshow_master_patch(axes[6], master_texture, pc_patch, f"Residual PC finetune\nscore={refined_score[2]:+.5f}")
    imshow_master_patch(
        axes[7],
        master_texture,
        orientation_patch,
        "Residual orientation finetune\n"
        f"score={orientation_best['combined_score']:+.5f}",
    )

    overlay = np.zeros((*map_patch[1].shape, 4), dtype=np.float32)
    overlay[..., 0] = np.where(map_patch[1], 1.0, 0.0)
    overlay[..., 1] = np.where(scan_patch[1] | orientation_patch[1], 0.82, 0.0)
    overlay[..., 2] = np.where(pc_patch[1], 1.0, 0.0)
    overlay[..., 3] = np.where(map_patch[1] | scan_patch[1] | pc_patch[1] | orientation_patch[1], 0.68, 0.0)
    axes[8].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[8].imshow(overlay, origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[8].set_title("Footprint shift\nred=map, yellow=scan/orient, blue=residual PC")
    axes[8].set_xlabel("longitude (deg)")
    axes[8].set_ylabel("colatitude (deg)")

    labels = ["map", "scan PC", "PC residual", "PC+ori"]
    scores = [map_score[2], scan_score[2], refined_score[2], orientation_best["combined_score"]]
    axes[9].bar(labels, scores, color=["#d62728", "#ffbf00", "#1f77b4", "#2ca02c"])
    axes[9].axhline(0, color="black", linewidth=0.8)
    axes[9].set_title("Stage-wise combined NCC score")
    axes[9].set_ylabel("combined score")
    axes[9].tick_params(axis="x", rotation=20)

    if orientation_trace:
        evals = np.arange(len(orientation_trace))
        combined = np.array([row["combined_score"] for row in orientation_trace], dtype=np.float64)
        best_so_far = np.maximum.accumulate(combined)
        axes[10].plot(evals, combined, color="#6a5acd", alpha=0.35, label="trial")
        axes[10].plot(evals, best_so_far, color="#111111", linewidth=1.1, label="best so far")
        axes[10].scatter([int(np.argmax(combined))], [float(np.max(combined))], c="red", s=42, zorder=4)
        axes[10].legend(loc="lower right")
    axes[10].set_title("Orientation residual search trace")
    axes[10].set_xlabel("evaluation")
    axes[10].set_ylabel("combined score")

    axes[11].axis("off")
    pc_scan_shift = np.array(scan_pc) - np.array(map_pc)
    pc_residual_shift = np.array(refined_pc) - np.array(scan_pc)
    summary = (
        f"map: {projection.map_name}\n"
        f"mask mode: {scan_meta.get('mask_mode', 'unknown')}\n"
        f"scan offset: x={scan_meta.get('scan_x_um', 0.0):+.3f} um, "
        f"y={scan_meta.get('scan_y_um', 0.0):+.3f} um\n"
        f"PC initial mode: {scan_meta.get('pc_initial_mode')}\n"
        f"map PC:     ({map_pc[0]:.6f}, {map_pc[1]:.6f}, {map_pc[2]:.6f})\n"
        f"scan PC:    ({scan_pc[0]:.6f}, {scan_pc[1]:.6f}, {scan_pc[2]:.6f})\n"
        f"scan shift: ({pc_scan_shift[0]:+.6f}, {pc_scan_shift[1]:+.6f}, {pc_scan_shift[2]:+.6f})\n"
        f"refined PC: ({refined_pc[0]:.6f}, {refined_pc[1]:.6f}, {refined_pc[2]:.6f})\n"
        f"residual:   ({pc_residual_shift[0]:+.6f}, {pc_residual_shift[1]:+.6f}, {pc_residual_shift[2]:+.6f})\n"
        f"orientation dR: rx={orientation_best['rot_x_deg']:+.4f} deg, "
        f"ry={orientation_best['rot_y_deg']:+.4f} deg, "
        f"rz={orientation_best['rot_z_deg']:+.4f} deg\n"
        f"score: map={map_score[2]:+.5f}, scan={scan_score[2]:+.5f}, "
        f"PC={refined_score[2]:+.5f}, PC+ori={orientation_best['combined_score']:+.5f}"
    )
    axes[11].text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=9.4)

    fig.suptitle("Pt Kikuchi: full-mask PC position correction -> residual PC -> residual orientation finetune", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def calibrate_one(
    args: argparse.Namespace,
    h5_row: dict[str, Any],
    selection: dict[str, Any],
    samplers,
    master_texture: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    up2_path = Path(h5_row["up2_actual_path"])
    pattern_index = int(selection["pattern_index"])
    key = safe_name(
        f"{h5_row['specimen']}_{h5_row['area']}_{h5_row['map_name']}_idx{pattern_index}"
    )
    pattern_dir = output_root / "per_pattern" / key
    pattern_dir.mkdir(parents=True, exist_ok=True)

    raw_projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=up2_path,
            map_group=h5_row["h5_path"],
            pattern_index=pattern_index,
        )
    )
    map_pc = raw_projection.pc_edax
    scan_pc, scan_meta = scan_position_pc(args, h5_row["h5_path"], pattern_index, map_pc)
    projection = projection_with_pc(raw_projection, scan_pc)
    mask, circle, mask_mode = detector_mask_for_pattern(projection, args)
    images = build_preprocessed_images(projection.pattern, mask)
    orientation_name, matrix, orientation_rows = choose_orientation_matrix(
        projection=projection,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    refined, score_rows = pc_finetune(
        projection=projection,
        matrix=matrix,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        coarse_range=tuple(args.pc_range),
        coarse_steps=args.coarse_steps,
        fine_steps=args.fine_steps,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    original = next(row for row in score_rows if row.stage == "original")
    map_score = score_orientation_matrix(projection, map_pc, matrix, mask, images, samplers, args)
    scan_score = (original.intensity_score, original.band_score, original.combined_score)
    refined_score = (refined.intensity_score, refined.band_score, refined.combined_score)
    orientation_matrix, orientation_best, orientation_trace = orientation_finetune(
        projection=projection,
        pc=refined.pc,
        base_matrix=matrix,
        mask=mask,
        images=images,
        samplers=samplers,
        args=args,
    )
    detector_patch = project_detector_patch(projection, original.pc, images["enhanced"], mask)
    original_patch = project_crystal_patch(projection, original.pc, matrix, images["enhanced"], mask)
    refined_patch = project_crystal_patch(projection, refined.pc, matrix, images["enhanced"], mask)

    write_orientation_scores(pattern_dir / "orientation_scores.csv", orientation_rows)
    write_pc_scores(pattern_dir / "pc_finetune_scores.csv", score_rows)
    write_orientation_finetune_trace(pattern_dir / "orientation_finetune_trace.csv", orientation_trace)
    single_args = one_pattern_args(args, up2_path, h5_row["h5_path"], pattern_index, pattern_dir)
    write_summary(
        pattern_dir / "single_kikuchi_pc_finetune_summary.csv",
        args=single_args,
        projection=projection,
        orientation_name=orientation_name,
        circle=circle,
        original=original,
        refined=refined,
        stride=args.stride,
    )
    overview_path = pattern_dir / f"{key}_spherical_calibration_overview.png"
    save_visualization(
        overview_path,
        projection=projection,
        images=images,
        mask=mask,
        circle=circle,
        master_texture=master_texture,
        orientation_name=orientation_name,
        original=original,
        refined=refined,
        detector_patch=detector_patch,
        original_patch=original_patch,
        refined_patch=refined_patch,
        score_rows=score_rows,
    )
    position_orientation_path = pattern_dir / f"{key}_position_pc_orientation_finetune.png"
    save_position_pc_orientation_visualization(
        position_orientation_path,
        projection=projection,
        images=images,
        mask=mask,
        circle=circle,
        master_texture=master_texture,
        matrix=matrix,
        orientation_matrix=orientation_matrix,
        map_pc=map_pc,
        scan_pc=scan_pc,
        refined_pc=refined.pc,
        map_score=map_score,
        scan_score=scan_score,
        refined_score=refined_score,
        orientation_best=orientation_best,
        orientation_trace=orientation_trace,
        scan_meta={**scan_meta, "mask_mode": mask_mode},
    )

    return {
        "specimen": h5_row["specimen"],
        "area": h5_row["area"],
        "map_name": h5_row["map_name"],
        "h5_path": h5_row["h5_path"],
        "up2_file": h5_row["up2_display_name"],
        "up2_path": str(up2_path),
        "pattern_index": pattern_index,
        "row": selection["row"],
        "col": selection["col"],
        "IQ": selection["IQ"],
        "CI": selection["CI"],
        "Fit": selection["Fit"],
        "Phase": selection["Phase"],
        "selection_score": selection["selection_score"],
        "orientation_variant": orientation_name,
        "mask_mode": mask_mode,
        "mask_center_x": circle[0],
        "mask_center_y": circle[1],
        "mask_radius": circle[2],
        "pc_initial_mode": scan_meta["pc_initial_mode"],
        "scan_x_um": scan_meta["scan_x_um"],
        "scan_y_um": scan_meta["scan_y_um"],
        "map_pc_x": map_pc[0],
        "map_pc_y": map_pc[1],
        "map_pc_z": map_pc[2],
        "scan_pc_x": scan_pc[0],
        "scan_pc_y": scan_pc[1],
        "scan_pc_z": scan_pc[2],
        "pc_edax_x": original.pc[0],
        "pc_edax_y": original.pc[1],
        "pc_edax_z": original.pc[2],
        "pc_refined_x": refined.pc[0],
        "pc_refined_y": refined.pc[1],
        "pc_refined_z": refined.pc[2],
        "scan_pc_delta_pcx": scan_pc[0] - map_pc[0],
        "scan_pc_delta_pcy": scan_pc[1] - map_pc[1],
        "scan_pc_delta_pcz": scan_pc[2] - map_pc[2],
        "delta_pcx": refined.pc[0] - scan_pc[0],
        "delta_pcy": refined.pc[1] - scan_pc[1],
        "delta_pcz": refined.pc[2] - scan_pc[2],
        "map_intensity_score": map_score[0],
        "map_band_score": map_score[1],
        "map_combined_score": map_score[2],
        "original_intensity_score": original.intensity_score,
        "original_band_score": original.band_score,
        "original_combined_score": original.combined_score,
        "refined_intensity_score": refined.intensity_score,
        "refined_band_score": refined.band_score,
        "refined_combined_score": refined.combined_score,
        "score_gain": refined.combined_score - original.combined_score,
        "orientation_rot_x_deg": orientation_best["rot_x_deg"],
        "orientation_rot_y_deg": orientation_best["rot_y_deg"],
        "orientation_rot_z_deg": orientation_best["rot_z_deg"],
        "orientation_intensity_score": orientation_best["intensity_score"],
        "orientation_band_score": orientation_best["band_score"],
        "orientation_combined_score": orientation_best["combined_score"],
        "orientation_score_gain": orientation_best["combined_score"] - refined.combined_score,
        "sample_tilt_deg": projection.sample_tilt,
        "camera_elevation_deg": projection.camera_elevation,
        "camera_azimuthal_deg": projection.camera_azimuthal,
        "overview_png": str(overview_path),
        "position_orientation_png": str(position_orientation_path),
        "output_dir": str(pattern_dir),
    }


def save_contact_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    cols = min(3, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 6.2, rows_n * 4.8), dpi=180)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, row in zip(axes_arr.ravel(), rows):
        image = plt.imread(row.get("position_orientation_png") or row["overview_png"])
        ax.imshow(image)
        ax.set_title(
            f"{row['specimen']} {row['area']} idx={row['pattern_index']}\n"
            f"IQ={float(row['IQ']):.0f}, CI={float(row['CI']):.3f}, "
            f"score {float(row['map_combined_score']):+.3f}->{float(row['orientation_combined_score']):+.3f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle("Pt Kikuchi full-mask PC-position correction + orientation residual finetune", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pt Kikuchi Spherical Calibration Batch",
        "",
        f"- Calibrated patterns: {len(rows)}",
        "- Fixed workflow: full Kikuchi disk mask -> background correction/CLAHE -> scan-position PC correction -> residual PC finetune -> residual orientation finetune.",
        "- Cubic symmetry/axis placement is not applied here; this is the detector-validated single-pattern matching flow.",
        "",
        "| Specimen | Area | Index | IQ | CI | Orientation variant | Map score | Scan PC score | PC score | PC+ori score | Residual PC | dR deg | Overview |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        delta = f"({float(row['delta_pcx']):+.5f}, {float(row['delta_pcy']):+.5f}, {float(row['delta_pcz']):+.5f})"
        drot = f"({float(row['orientation_rot_x_deg']):+.3f}, {float(row['orientation_rot_y_deg']):+.3f}, {float(row['orientation_rot_z_deg']):+.3f})"
        lines.append(
            f"| {row['specimen']} | {row['area']} | {row['pattern_index']} | "
            f"{float(row['IQ']):.0f} | {float(row['CI']):.3f} | {row['orientation_variant']} | "
            f"{float(row['map_combined_score']):+.5f} | {float(row['original_combined_score']):+.5f} | "
            f"{float(row['refined_combined_score']):+.5f} | {float(row['orientation_combined_score']):+.5f} | "
            f"{delta} | {drot} | `{row['position_orientation_png']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    up2_roots = args.up2_root or [DEFAULT_UP2_ROOT]
    matched_rows = matched_pt_maps(args.h5, up2_roots, args.specimen)
    if not matched_rows:
        raise RuntimeError(f"No matched Pt H5/UP2 maps found for {args.h5}")
    if args.max_maps > 0:
        matched_rows = matched_rows[: args.max_maps]

    samplers = build_master_samplers(args.master)
    master_texture = build_master_lon_colat(samplers.upper_corrected, samplers.lower_corrected)
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    with h5py.File(args.h5, "r") as h5:
        for h5_row in matched_rows:
            map_group = h5[h5_row["h5_path"]]
            selections = choose_quality_indices(
                map_group=map_group,
                count=int(h5_row["point_count"]),
                patterns_per_map=args.patterns_per_map,
                ci_min=args.ci_min,
                min_scan_distance_fraction=args.min_scan_distance_fraction,
            )
            for selection in selections:
                selected_rows.append({**h5_row, **selection})

    if args.max_patterns > 0:
        selected_rows = selected_rows[: args.max_patterns]
    if not selected_rows:
        raise RuntimeError("No Kikuchi patterns passed the IQ/CI selection filters.")

    for ordinal, row in enumerate(selected_rows, start=1):
        h5_row = {key: row[key] for key in matched_rows[0].keys()}
        selection = {
            key: row[key]
            for key in ["pattern_index", "row", "col", "IQ", "CI", "Fit", "Phase", "selection_score"]
        }
        print(
            f"[{ordinal}/{len(selected_rows)}] {h5_row['specimen']} {h5_row['area']} "
            f"idx={selection['pattern_index']} IQ={selection['IQ']:.0f} CI={selection['CI']:.3f}"
        )
        summary_rows.append(calibrate_one(args, h5_row, selection, samplers, master_texture, args.output_dir))

    write_csv(args.output_dir / "pt_kikuchi_spherical_calibration_summary.csv", summary_rows)
    write_markdown(args.output_dir / "pt_kikuchi_spherical_calibration_summary.md", summary_rows)
    save_contact_sheet(summary_rows, args.output_dir / "pt_kikuchi_spherical_calibration_contact_sheet.png")
    print(f"Saved {len(summary_rows)} calibrated Kikuchi patterns to {args.output_dir}")
    print(f"Summary CSV: {args.output_dir / 'pt_kikuchi_spherical_calibration_summary.csv'}")
    print(f"Contact sheet: {args.output_dir / 'pt_kikuchi_spherical_calibration_contact_sheet.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high-quality Pt Kikuchi patterns and run the fixed spherical calibration + PC finetune workflow."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", action="append", type=Path, default=None)
    parser.add_argument("--specimen", default="Pt-3", help="H5 specimen filter, e.g. Pt-3. Use empty string for all.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patterns-per-map", type=int, default=1)
    parser.add_argument("--max-maps", type=int, default=0, help="0 means all matched maps.")
    parser.add_argument("--max-patterns", type=int, default=0, help="0 means all selected patterns.")
    parser.add_argument("--ci-min", type=float, default=0.30)
    parser.add_argument("--min-scan-distance-fraction", type=float, default=0.18)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--pc-range", nargs=3, type=float, default=(0.02, 0.02, 0.04), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-steps", type=int, default=7)
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
    parser.set_defaults(orientation_finetune=True)
    parser.add_argument("--orientation-finetune", dest="orientation_finetune", action="store_true")
    parser.add_argument("--no-orientation-finetune", dest="orientation_finetune", action="store_false")
    parser.add_argument("--orientation-bound-deg", type=float, default=1.2)
    parser.add_argument("--orientation-steps", type=int, default=5)
    parser.add_argument("--orientation-fine-steps", type=int, default=5)
    parser.add_argument("--orientation-regularization-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.specimen == "":
        args.specimen = None
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
