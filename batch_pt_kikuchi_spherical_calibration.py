"""Batch spherical calibration for selected Pt Kikuchi patterns.

This is the fixed, reusable version of the single-pattern workflow:

1. match Pt H5 EBSD maps to local UP2 stacks;
2. select high-IQ/high-CI Kikuchi patterns;
3. run circular-mask preprocessing, H5 orientation projection, and local PC finetune;
4. save one detailed nine-panel visualization per pattern plus summary sheets.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from export_h5_mapping_sem_correspondence import (
    collect_h5_maps,
    collect_up2_candidates,
    match_h5_to_up2,
    safe_name,
)
from project_edax_oim_to_sphere import EdaxMapInputs, build_master_lon_colat, read_edax_inputs
from single_kikuchi_pc_finetune import (
    build_master_samplers,
    build_preprocessed_images,
    centered_circular_detector_mask,
    choose_orientation_matrix,
    pc_finetune,
    project_crystal_patch,
    project_detector_patch,
    save_visualization,
    write_orientation_scores,
    write_pc_scores,
    write_summary,
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

    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=up2_path,
            map_group=h5_row["h5_path"],
            pattern_index=pattern_index,
        )
    )
    mask, circle = centered_circular_detector_mask(projection.pattern.shape, args.mask_radius_fraction)
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
    detector_patch = project_detector_patch(projection, original.pc, images["enhanced"], mask)
    original_patch = project_crystal_patch(projection, original.pc, matrix, images["enhanced"], mask)
    refined_patch = project_crystal_patch(projection, refined.pc, matrix, images["enhanced"], mask)

    write_orientation_scores(pattern_dir / "orientation_scores.csv", orientation_rows)
    write_pc_scores(pattern_dir / "pc_finetune_scores.csv", score_rows)
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
        "mask_center_x": circle[0],
        "mask_center_y": circle[1],
        "mask_radius": circle[2],
        "pc_edax_x": original.pc[0],
        "pc_edax_y": original.pc[1],
        "pc_edax_z": original.pc[2],
        "pc_refined_x": refined.pc[0],
        "pc_refined_y": refined.pc[1],
        "pc_refined_z": refined.pc[2],
        "delta_pcx": refined.delta[0],
        "delta_pcy": refined.delta[1],
        "delta_pcz": refined.delta[2],
        "original_intensity_score": original.intensity_score,
        "original_band_score": original.band_score,
        "original_combined_score": original.combined_score,
        "refined_intensity_score": refined.intensity_score,
        "refined_band_score": refined.band_score,
        "refined_combined_score": refined.combined_score,
        "score_gain": refined.combined_score - original.combined_score,
        "sample_tilt_deg": projection.sample_tilt,
        "camera_elevation_deg": projection.camera_elevation,
        "camera_azimuthal_deg": projection.camera_azimuthal,
        "overview_png": str(overview_path),
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
        image = plt.imread(row["overview_png"])
        ax.imshow(image)
        ax.set_title(
            f"{row['specimen']} {row['area']} idx={row['pattern_index']}\n"
            f"IQ={float(row['IQ']):.0f}, CI={float(row['CI']):.3f}, "
            f"score {float(row['original_combined_score']):+.3f}->{float(row['refined_combined_score']):+.3f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle("Pt Kikuchi spherical matching calibration batch", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pt Kikuchi Spherical Calibration Batch",
        "",
        f"- Calibrated patterns: {len(rows)}",
        "- Fixed workflow: circular mask -> background correction/CLAHE -> H5 orientation on master sphere -> local PC finetune.",
        "- Cubic symmetry/axis placement is not applied here; this is the detector-validated single-pattern matching flow.",
        "",
        "| Specimen | Area | Index | IQ | CI | Orientation variant | Original score | Refined score | Delta PC | Overview |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        delta = f"({float(row['delta_pcx']):+.5f}, {float(row['delta_pcy']):+.5f}, {float(row['delta_pcz']):+.5f})"
        lines.append(
            f"| {row['specimen']} | {row['area']} | {row['pattern_index']} | "
            f"{float(row['IQ']):.0f} | {float(row['CI']):.3f} | {row['orientation_variant']} | "
            f"{float(row['original_combined_score']):+.5f} | {float(row['refined_combined_score']):+.5f} | "
            f"{delta} | `{row['overview_png']}` |"
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
    parser.add_argument("--mask-radius-fraction", type=float, default=0.40)
    args = parser.parse_args()
    if args.specimen == "":
        args.specimen = None
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
