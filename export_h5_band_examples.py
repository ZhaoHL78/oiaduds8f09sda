from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    SCRIPT_DIR,
    MatchWeights,
    default_map_specs,
    jsonable,
    prepare_pattern,
    read_pattern_bundle,
    save_detector_overlay,
)


DEFAULT_EXAMPLES = [
    "area1_high:0",
    "area1_high:659",
    "area1_high:10685",
    "area2_high:0",
    "area2_high:659",
    "area2_high:10685",
]


def parse_example(text: str) -> tuple[str, int]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"Expected map:index, got {text!r}")
    map_key, index_text = text.split(":", 1)
    return map_key, int(index_text)


def draw_overlay_axis(ax, prepared) -> None:
    ax.imshow(
        prepared.bundle.pattern_u16,
        cmap="gray",
        vmin=int(prepared.bundle.pattern_u16.min()),
        vmax=int(prepared.bundle.pattern_u16.max()),
    )
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, max(1, len(prepared.line_segments))))
    for color, segment in zip(colors, prepared.line_segments):
        ax.plot([segment.col0, segment.col1], [segment.row0, segment.row1], color=color, linewidth=1.35)
        mid_col = 0.5 * (segment.col0 + segment.col1)
        mid_row = 0.5 * (segment.row0 + segment.row1)
        ax.text(mid_col, mid_row, str(segment.band_index + 1), color="white", fontsize=6, ha="center", va="center")
    ax.set_title(
        f"{prepared.bundle.map_spec.label}  idx={prepared.bundle.index}\n"
        f"row={prepared.bundle.row}, col={prepared.bundle.col}, r={prepared.line_variant_score:.3f}",
        fontsize=9,
    )
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw EBSD patterns with their EDAX H5/OHP Kikuchi-band data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "outputs" / "h5_band_examples")
    parser.add_argument("--example", action="append", type=parse_example, help="Example as map:index. Can be repeated.")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    args = parser.parse_args()

    examples = args.example or [parse_example(item) for item in DEFAULT_EXAMPLES]
    map_specs = default_map_specs(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = MatchWeights(image_line=0.60, intensity=0.20, h5_band=0.20)
    prepared_items = []
    rows: list[dict] = []
    summaries: list[dict] = []

    for map_key, index in examples:
        if map_key not in map_specs:
            raise ValueError(f"Unknown map {map_key!r}. Valid maps: {', '.join(map_specs)}")
        bundle = read_pattern_bundle(args.h5, map_specs[map_key], index)
        prepared, variant_diagnostics = prepare_pattern(
            bundle=bundle,
            weights=weights,
            label="h5-band-export",
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
            match_quantile=0.82,
            top_k_points=5000,
            line_variant_name="auto",
        )
        prepared_items.append(prepared)

        stem = f"{map_key}_idx_{index:05d}"
        overlay_path = args.out_dir / f"{stem}_pattern_h5_bands.png"
        save_detector_overlay(prepared, overlay_path)

        for segment in prepared.line_segments:
            rows.append(
                {
                    "map": map_key,
                    "map_label": bundle.map_spec.label,
                    "index": bundle.index,
                    "row": bundle.row,
                    "col": bundle.col,
                    "height": bundle.pattern_u16.shape[0],
                    "width": bundle.pattern_u16.shape[1],
                    "pcx": bundle.pc[0],
                    "pcy": bundle.pc[1],
                    "pcz": bundle.pc[2],
                    "line_variant": prepared.line_variant.name,
                    "line_correlation": prepared.line_variant_score,
                    "band_order": segment.band_index + 1,
                    "rho_bin": segment.band.rho_bin,
                    "theta_deg": segment.band.theta_deg,
                    "band_width": segment.band.width,
                    "band_intensity": segment.band.intensity,
                    "line_row0": segment.row0,
                    "line_col0": segment.col0,
                    "line_row1": segment.row1,
                    "line_col1": segment.col1,
                }
            )

        summaries.append(
            {
                "map": map_key,
                "index": bundle.index,
                "row": bundle.row,
                "col": bundle.col,
                "pattern_shape": list(bundle.pattern_u16.shape),
                "pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
                "line_variant": asdict(prepared.line_variant),
                "line_correlation": prepared.line_variant_score,
                "overlay_path": str(overlay_path),
                "bands": [asdict(band) for band in bundle.bands],
                "variant_diagnostics": variant_diagnostics,
            }
        )

    csv_path = args.out_dir / "kikuchi_band_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "kikuchi_band_examples.json"
    json_path.write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(len(prepared_items), 2, figsize=(9.5, 4.3 * len(prepared_items)))
    if len(prepared_items) == 1:
        axes = np.asarray([axes])
    for ax_row, prepared in zip(axes, prepared_items):
        ax_row[0].imshow(
            prepared.bundle.pattern_u16,
            cmap="gray",
            vmin=int(prepared.bundle.pattern_u16.min()),
            vmax=int(prepared.bundle.pattern_u16.max()),
        )
        ax_row[0].set_title(f"{prepared.bundle.map_spec.label} raw idx={prepared.bundle.index}", fontsize=9)
        ax_row[0].axis("off")
        draw_overlay_axis(ax_row[1], prepared)
    fig.tight_layout()
    montage_path = args.out_dir / "kikuchi_band_examples_montage.png"
    fig.savefig(montage_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {len(prepared_items)} examples to {args.out_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Montage: {montage_path}")


if __name__ == "__main__":
    main()
