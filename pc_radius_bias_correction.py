from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    MasterSphere,
    PreparedPattern,
    default_map_specs,
    detector_to_sphere_grid,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    resolve_master_path,
    score_rotation,
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


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def corrected_pc(base_pc: tuple[float, float, float], dx_px: float, dy_px: float, radius_scale: float, height: int, width: int) -> tuple[float, float, float]:
    pcx, pcy, pcz = base_pc
    return (
        pcx + dx_px / float(width - 1),
        pcy + dy_px / float(height - 1),
        pcz * radius_scale,
    )


def prepared_with_pc(prepared: PreparedPattern, pc: tuple[float, float, float]) -> PreparedPattern:
    height, width = prepared.image.shape
    full_points_grid = detector_to_sphere_grid(height, width, pc)
    bundle = replace(prepared.bundle, pc=pc)
    return replace(
        prepared,
        bundle=bundle,
        full_points_grid=full_points_grid,
        exp_points=full_points_grid[prepared.match_mask],
    )


def deterministic_rotation_refine(
    prepared: PreparedPattern,
    master: MasterSphere,
    detector_transform: np.ndarray,
    initial_rotation: R,
    local_steps_deg: list[float],
) -> tuple[R, float]:
    points = prepared.exp_points @ detector_transform.T
    rotation = initial_rotation
    best_score = score_rotation(rotation, points, prepared, master)

    axes = np.array(
        [
            [dz, dy, dx]
            for dz in (-1.0, 0.0, 1.0)
            for dy in (-1.0, 0.0, 1.0)
            for dx in (-1.0, 0.0, 1.0)
            if not (dz == 0.0 and dy == 0.0 and dx == 0.0)
        ],
        dtype=np.float32,
    )
    for step_deg in local_steps_deg:
        improved = True
        while improved:
            improved = False
            best_candidate_rotation = rotation
            best_candidate_score = best_score
            for delta_zyx in axes * step_deg:
                candidate = R.from_euler("zyx", delta_zyx, degrees=True) * rotation
                candidate_score = score_rotation(candidate, points, prepared, master)
                if candidate_score > best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate_rotation = candidate
            if best_candidate_score > best_score + 1e-7:
                rotation = best_candidate_rotation
                best_score = best_candidate_score
                improved = True
    return rotation, float(best_score)


def save_score_landscape(records: list[dict], out_path: Path) -> None:
    dx_values = sorted({float(row["dx_px"]) for row in records})
    dy_values = sorted({float(row["dy_px"]) for row in records})
    radius_values = sorted({float(row["radius_scale"]) for row in records})
    best = max(records, key=lambda row: float(row["score"]))
    best_radius = float(best["radius_scale"])

    score_grid = np.full((len(dy_values), len(dx_values)), np.nan, dtype=np.float32)
    for row in records:
        if abs(float(row["radius_scale"]) - best_radius) > 1e-9:
            continue
        y = dy_values.index(float(row["dy_px"]))
        x = dx_values.index(float(row["dx_px"]))
        score_grid[y, x] = float(row["score"])

    radius_scores = []
    for radius in radius_values:
        radius_scores.append(max(float(row["score"]) for row in records if abs(float(row["radius_scale"]) - radius) < 1e-9))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    im = axes[0].imshow(
        score_grid,
        origin="lower",
        cmap="viridis",
        extent=[min(dx_values), max(dx_values), min(dy_values), max(dy_values)],
        aspect="auto",
    )
    axes[0].scatter([float(best["dx_px"])], [float(best["dy_px"])], color="red", marker="x", s=80, linewidths=2.2)
    axes[0].set_title(f"PC shift score at radius={best_radius:.4f}")
    axes[0].set_xlabel("PC x shift (pixels)")
    axes[0].set_ylabel("PC y shift (pixels)")
    fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03)

    axes[1].plot(radius_values, radius_scores, marker="o")
    axes[1].axvline(best_radius, color="red", linestyle="--", linewidth=1.1)
    axes[1].set_title("Best score per radius scale")
    axes[1].set_xlabel("Projection radius scale")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)

    labels = ["H5 PC", "Corrected"]
    scores = [
        float(row["score"])
        for row in records
        if abs(float(row["dx_px"])) < 1e-9
        and abs(float(row["dy_px"])) < 1e-9
        and abs(float(row["radius_scale"]) - 1.0) < 1e-9
    ]
    h5_score = scores[0] if scores else float("nan")
    axes[2].bar(labels, [h5_score, float(best["score"])], color=["#6b7280", "#4c9f70"])
    axes[2].set_title("Local correction improvement")
    axes[2].set_ylabel("Score")
    axes[2].grid(axis="y", alpha=0.25)
    for i, score in enumerate([h5_score, float(best["score"])]):
        axes[2].text(i, score, f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_search_csv(records: list[dict], out_path: Path) -> None:
    fieldnames = [
        "dx_px",
        "dy_px",
        "radius_scale",
        "pcx",
        "pcy",
        "effective_pcz",
        "score",
        "rotation_quat_xyzw",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row and key != "rotation_quat_xyzw"},
                    "rotation_quat_xyzw": json.dumps(row["rotation_quat_xyzw"]),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Try local EBSD pattern-center and projection-radius bias correction.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "pc_radius_bias_correction")
    parser.add_argument("--pc-shifts-px", default="-8,-4,0,4,8")
    parser.add_argument("--radius-scales", default="0.96,0.98,1.00,1.02,1.04")
    parser.add_argument("--local-steps-deg", default="2.0,0.75,0.25")
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
    master_h5 = resolve_master_path(args.master_h5)
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading pattern: {map_spec.label}, index={args.index}")
    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    products = build_preprocessing_products(
        bundle.pattern_u16,
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
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

    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    print("Initial global match with H5 PC...")
    initial_result = match_to_master(
        prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=parse_refine_schedule(args.refine_schedule),
        random_seed=args.seed,
    )
    save_final_spatial_visualization(
        initial_result,
        master,
        products,
        out_dir / "01_h5_pc_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    pc_shifts = parse_float_list(args.pc_shifts_px)
    radius_scales = parse_float_list(args.radius_scales)
    local_steps = parse_float_list(args.local_steps_deg)
    height, width = prepared.image.shape
    records: list[dict] = []
    best_result: MatchResult | None = None
    best_record: dict | None = None
    candidates = [(dx, dy, radius) for radius in radius_scales for dy in pc_shifts for dx in pc_shifts]
    iterator = tqdm(candidates, desc="PC/radius local search") if tqdm is not None else candidates

    for dx_px, dy_px, radius_scale in iterator:
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
        out_dir / "02_corrected_pc_radius_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_score_landscape(records, out_dir / "03_pc_radius_score_landscape.png")
    write_search_csv(records, out_dir / "pc_radius_search_results.csv")

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
            "h5_pc_final_spatial": str(out_dir / "01_h5_pc_final_spatial.png"),
            "corrected_pc_radius_final_spatial": str(out_dir / "02_corrected_pc_radius_final_spatial.png"),
            "score_landscape": str(out_dir / "03_pc_radius_score_landscape.png"),
            "search_csv": str(out_dir / "pc_radius_search_results.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PC/radius correction attempt to: {out_dir}")
    print(
        "Best correction: "
        f"dx={best_record['dx_px']:+.2f}px, dy={best_record['dy_px']:+.2f}px, "
        f"radius_scale={best_record['radius_scale']:.4f}, score={best_record['score']:.4f}"
    )


if __name__ == "__main__":
    main()
