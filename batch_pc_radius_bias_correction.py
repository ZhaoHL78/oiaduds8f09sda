from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    default_map_specs,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
)
from pc_radius_bias_correction import (
    corrected_pc,
    deterministic_rotation_refine,
    parse_float_list,
    prepared_with_pc,
    save_score_landscape,
    write_search_csv,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    parse_refine_schedule,
    save_final_spatial_visualization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


def write_batch_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "index",
        "row",
        "col",
        "h5_score",
        "best_score",
        "score_gain",
        "score_gain_percent",
        "dx_px",
        "dy_px",
        "radius_scale",
        "pcx",
        "pcy",
        "effective_pcz",
        "convention_name",
        "line_variant",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_landscape_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 620, columns: int = 2) -> None:
    tiles = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        label_bar = np.full((40, thumb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(label_bar, path.parent.name, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(np.vstack([label_bar, thumb]))

    if not tiles:
        return

    rows = int(np.ceil(len(tiles) / columns))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    pad = 20
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, columns * tile_w + (columns + 1) * pad, 3),
        255,
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        y = pad + (i // columns) * (tile_h + pad)
        x = pad + (i % columns) * (tile_w + pad)
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    cv2.imwrite(str(out_path), canvas)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch try EBSD pattern-center and projection-radius bias correction.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--indices", default=None, help="Comma list or ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "pc_radius_bias_correction_batch")
    parser.add_argument("--pc-shifts-px", default="-18,-12,-6,0,6,12,18")
    parser.add_argument("--radius-scales", default="0.86,0.90,0.94,0.98,1.00,1.02")
    parser.add_argument("--local-steps-deg", default="1.5,0.5")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=5000)
    parser.add_argument("--coarse-rotations", type=int, default=160)
    parser.add_argument("--refine-schedule", default="8:100,3:140,1:140")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sphere-lon-count", type=int, default=420)
    parser.add_argument("--sphere-colat-count", type=int, default=210)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.45)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_spec = default_map_specs(args.data_dir)[args.map]
    total = read_up2_info(map_spec.up2_path).count
    indices = parse_indices(args.indices, total, args.count, args.strategy)
    if not indices:
        raise ValueError("No valid pattern indices selected")

    pc_shifts = parse_float_list(args.pc_shifts_px)
    radius_scales = parse_float_list(args.radius_scales)
    local_steps = parse_float_list(args.local_steps_deg)
    refine_schedule = parse_refine_schedule(args.refine_schedule)
    weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere once: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    h5_paths: list[Path] = []
    corrected_paths: list[Path] = []
    landscape_paths: list[Path] = []
    summary_rows: list[dict] = []
    full_summaries: list[dict] = []
    iterator = tqdm(indices, desc="PC/radius batch") if tqdm is not None else indices

    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        out_dir = batch_dir / f"idx_{index:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = read_pattern_bundle(args.h5, map_spec, index)
        products = build_preprocessing_products(
            bundle.pattern_u16,
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
        )
        prepared, variant_diagnostics = prepare_pattern(
            bundle=bundle,
            weights=weights,
            label="H5-band-enhanced",
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
            match_quantile=args.match_quantile,
            top_k_points=args.top_k_points,
            line_variant_name="auto",
        )
        initial_result = match_to_master(
            prepared,
            master,
            coarse_rotation_count=args.coarse_rotations,
            refine_schedule=refine_schedule,
            random_seed=args.seed + int(index),
        )

        h5_path = out_dir / "01_h5_pc_final_spatial.png"
        corrected_path = out_dir / "02_corrected_pc_radius_final_spatial.png"
        landscape_path = out_dir / "03_pc_radius_score_landscape.png"
        save_final_spatial_visualization(
            initial_result,
            master,
            products,
            h5_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )

        height, width = prepared.image.shape
        candidates = [(dx, dy, radius) for radius in radius_scales for dy in pc_shifts for dx in pc_shifts]
        records: list[dict] = []
        best_result: MatchResult | None = None
        best_record: dict | None = None
        search_iterator = tqdm(candidates, desc=f"idx_{index:05d}", leave=False) if tqdm is not None else candidates
        for dx_px, dy_px, radius_scale in search_iterator:
            pc = corrected_pc(bundle.pc, dx_px, dy_px, radius_scale, height, width)
            candidate_prepared = prepared_with_pc(prepared, pc)
            rotation, score = deterministic_rotation_refine(
                candidate_prepared,
                master,
                initial_result.detector_transform,
                initial_result.rotation,
                local_steps,
            )
            record = {
                "dx_px": float(dx_px),
                "dy_px": float(dy_px),
                "radius_scale": float(radius_scale),
                "pcx": float(pc[0]),
                "pcy": float(pc[1]),
                "effective_pcz": float(pc[2]),
                "score": float(score),
                "rotation_quat_xyzw": rotation.as_quat().tolist(),
            }
            records.append(record)
            if best_record is None or score > float(best_record["score"]):
                best_record = record
                best_result = MatchResult(
                    label="PC-radius-corrected",
                    score=float(score),
                    rotation=rotation,
                    convention_name=initial_result.convention_name,
                    detector_transform=initial_result.detector_transform,
                    prepared=candidate_prepared,
                )

        h5_local = next(
            (
                row
                for row in records
                if abs(float(row["dx_px"])) < 1e-9
                and abs(float(row["dy_px"])) < 1e-9
                and abs(float(row["radius_scale"]) - 1.0) < 1e-9
            ),
            None,
        )
        if h5_local is None:
            h5_prepared = prepared_with_pc(prepared, bundle.pc)
            h5_rotation, h5_score = deterministic_rotation_refine(
                h5_prepared,
                master,
                initial_result.detector_transform,
                initial_result.rotation,
                local_steps,
            )
            h5_local = {
                "dx_px": 0.0,
                "dy_px": 0.0,
                "radius_scale": 1.0,
                "pcx": float(bundle.pc[0]),
                "pcy": float(bundle.pc[1]),
                "effective_pcz": float(bundle.pc[2]),
                "score": float(h5_score),
                "rotation_quat_xyzw": h5_rotation.as_quat().tolist(),
            }
            records.append(h5_local)

        assert best_result is not None and best_record is not None
        save_final_spatial_visualization(
            best_result,
            master,
            products,
            corrected_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )
        save_score_landscape(records, landscape_path)
        write_search_csv(records, out_dir / "pc_radius_search_results.csv")

        h5_score = float(h5_local["score"])
        best_score = float(best_record["score"])
        row = {
            "index": int(bundle.index),
            "row": int(bundle.row),
            "col": int(bundle.col),
            "h5_score": h5_score,
            "best_score": best_score,
            "score_gain": best_score - h5_score,
            "score_gain_percent": (best_score - h5_score) / max(abs(h5_score), 1e-12) * 100.0,
            "dx_px": float(best_record["dx_px"]),
            "dy_px": float(best_record["dy_px"]),
            "radius_scale": float(best_record["radius_scale"]),
            "pcx": float(best_record["pcx"]),
            "pcy": float(best_record["pcy"]),
            "effective_pcz": float(best_record["effective_pcz"]),
            "convention_name": initial_result.convention_name,
            "line_variant": prepared.line_variant.name,
        }
        summary = {
            "map": map_spec.key,
            "map_label": map_spec.label,
            "index": bundle.index,
            "row": bundle.row,
            "col": bundle.col,
            "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
            "search_parameters": {
                "pc_shifts_px": pc_shifts,
                "radius_scales": radius_scales,
                "local_steps_deg": local_steps,
                "interpretation": "radius_scale multiplies H5 pcz, changing the detector-to-sphere angular radius while pcx/pcy shifts move the pattern center.",
            },
            "initial_global_match": initial_result.to_json_dict(),
            "h5_pc_local_refined": h5_local,
            "best_correction": best_record,
            "best_match": best_result.to_json_dict(),
            "line_variant": prepared.line_variant.name,
            "line_variant_score": prepared.line_variant_score,
            "variant_diagnostics": variant_diagnostics,
            "outputs": {
                "h5_pc_final_spatial": str(h5_path),
                "corrected_pc_radius_final_spatial": str(corrected_path),
                "score_landscape": str(landscape_path),
                "search_csv": str(out_dir / "pc_radius_search_results.csv"),
                "summary": str(out_dir / "summary.json"),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Best index={index}: dx={row['dx_px']:+.1f}px, dy={row['dy_px']:+.1f}px, "
            f"radius={row['radius_scale']:.3f}, score={best_score:.4f}, gain={row['score_gain']:.4f}"
        )
        h5_paths.append(h5_path)
        corrected_paths.append(corrected_path)
        landscape_paths.append(landscape_path)
        summary_rows.append(row)
        full_summaries.append(summary)

    write_batch_csv(summary_rows, batch_dir / "batch_pc_radius_summary.csv")
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(
            jsonable(
                {
                    "map": map_spec.key,
                    "total_patterns": total,
                    "indices": indices,
                    "search_parameters": {
                        "pc_shifts_px": pc_shifts,
                        "radius_scales": radius_scales,
                        "local_steps_deg": local_steps,
                    },
                    "rows": summary_rows,
                    "summaries": [str(batch_dir / f"idx_{row['index']:05d}" / "summary.json") for row in summary_rows],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    make_contact_sheet(h5_paths, batch_dir / "contact_sheet_h5_pc.png", thumb_width=880, columns=2)
    make_contact_sheet(corrected_paths, batch_dir / "contact_sheet_corrected_pc_radius.png", thumb_width=880, columns=2)
    make_landscape_contact_sheet(landscape_paths, batch_dir / "contact_sheet_score_landscape.png", thumb_width=620, columns=2)
    print(f"Saved {len(summary_rows)} PC/radius correction runs to: {batch_dir}")
    print(f"Summary CSV: {batch_dir / 'batch_pc_radius_summary.csv'}")
    print(f"Corrected contact sheet: {batch_dir / 'contact_sheet_corrected_pc_radius.png'}")


if __name__ == "__main__":
    main()
