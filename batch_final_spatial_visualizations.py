from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
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
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    parse_refine_schedule,
    save_final_spatial_visualization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


def parse_indices(text: str | None, total: int, count: int, strategy: str) -> list[int]:
    if text:
        indices: list[int] = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                parts = [int(part) for part in item.split(":")]
                if len(parts) == 2:
                    start, stop = parts
                    step = 1
                elif len(parts) == 3:
                    start, stop, step = parts
                else:
                    raise ValueError(f"Bad index range: {item}")
                indices.extend(range(start, stop, step))
            else:
                indices.append(int(item))
        return [idx for idx in indices if 0 <= idx < total]

    if strategy == "sequential":
        return list(range(min(count, total)))

    if count <= 1:
        return [0]
    return np.linspace(0, total - 1, count, dtype=int).tolist()


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 980, columns: int = 2) -> None:
    tiles = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        label = path.parent.name
        label_bar = np.full((44, thumb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(label_bar, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(np.vstack([label_bar, thumb]))

    if not tiles:
        return

    columns = max(1, columns)
    rows = int(np.ceil(len(tiles) / columns))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    pad = 24
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, columns * tile_w + (columns + 1) * pad, 3),
        255,
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        row = i // columns
        col = i % columns
        y = pad + row * (tile_h + pad)
        x = pad + col * (tile_w + pad)
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    cv2.imwrite(str(out_path), canvas)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch export final EBSD Kikuchi sphere spatial match visualizations.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--indices", default=None, help="Comma list or Python-like ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "final_spatial_batch")
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
    parser.add_argument("--sphere-lon-count", type=int, default=520)
    parser.add_argument("--sphere-colat-count", type=int, default=260)
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

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere once: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    refine_schedule = parse_refine_schedule(args.refine_schedule)
    weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    summaries = []
    iterator = tqdm(indices, desc="final spatial visualizations") if tqdm is not None else indices

    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        out_dir = batch_dir / f"idx_{index:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / "final_spatial_match.png"
        summary_path = out_dir / "summary.json"

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
        result = match_to_master(
            prepared,
            master,
            coarse_rotation_count=args.coarse_rotations,
            refine_schedule=refine_schedule,
            random_seed=args.seed + int(index),
        )
        save_final_spatial_visualization(
            result,
            master,
            products,
            final_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )
        summary = {
            "map": map_spec.key,
            "map_label": map_spec.label,
            "index": bundle.index,
            "row": bundle.row,
            "col": bundle.col,
            "pattern_shape": list(bundle.pattern_u16.shape),
            "pattern_center_from_h5": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
            "line_variant": prepared.line_variant.name,
            "line_variant_score": prepared.line_variant_score,
            "variant_diagnostics": variant_diagnostics,
            "match_result": result.to_json_dict(),
            "final_spatial": str(final_path),
        }
        summary_path.write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        final_paths.append(final_path)
        summaries.append(summary)

    batch_summary = {
        "map": map_spec.key,
        "total_patterns": total,
        "indices": indices,
        "outputs": [str(path) for path in final_paths],
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(batch_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(final_paths, batch_dir / "final_spatial_contact_sheet.png")
    print(f"Saved {len(final_paths)} final visualizations to: {batch_dir}")
    print(f"Contact sheet: {batch_dir / 'final_spatial_contact_sheet.png'}")


if __name__ == "__main__":
    main()
