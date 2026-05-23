from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
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
    read_up2_info,
    resolve_master_path,
)
from pc_radius_bias_correction import corrected_pc, deterministic_rotation_refine, prepared_with_pc
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    parse_refine_schedule,
    percentile_normalize,
    save_final_spatial_visualization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


@dataclass
class PatternContext:
    index: int
    row: int
    col: int
    prepared: PreparedPattern
    products: dict[str, np.ndarray]
    initial_result: MatchResult
    exp_dic: np.ndarray


@dataclass
class AverageField:
    field: np.ndarray
    counts: np.ndarray
    image_counts: np.ndarray


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def dic_uint8(image01: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    image01 = percentile_normalize(image01.astype(np.float32), mask, low=1.0, high=99.0)
    image01 = cv2.GaussianBlur(image01, (0, 0), sigmaX=0.55)
    return np.clip(image01 * 255.0, 0, 255).astype(np.uint8)


def experimental_dic_image(prepared: PreparedPattern, products: dict[str, np.ndarray]) -> np.ndarray:
    # The article tracks robust band intersections. In our data, line_response
    # supplies those features while contrast_enhanced keeps broader band context.
    exp = 0.70 * products["line_response"] + 0.30 * products["contrast_enhanced"]
    exp = percentile_normalize(exp, prepared.valid_mask, low=1.0, high=99.5)
    exp[~prepared.valid_mask] = 0.0
    return exp.astype(np.float32)


def render_master_detector_image(
    prepared: PreparedPattern,
    master: MasterSphere,
    result: MatchResult,
    pc: tuple[float, float, float],
    channel: str,
) -> np.ndarray:
    height, width = prepared.image.shape
    points_grid = detector_to_sphere_grid(height, width, pc)
    points = points_grid.reshape(-1, 3) @ result.detector_transform.T
    rotated = result.rotation.apply(points)
    if channel == "intensity":
        rendered = master.sample_intensity(rotated).reshape(height, width)
    elif channel == "band":
        rendered = master.sample_band(rotated).reshape(height, width)
    else:
        raise ValueError(f"Unknown render channel: {channel}")
    rendered = percentile_normalize(rendered.astype(np.float32), prepared.valid_mask, low=1.0, high=99.5)
    rendered[~prepared.valid_mask] = 0.0
    return rendered.astype(np.float32)


def compute_displacement_field(
    simulated: np.ndarray,
    experimental: np.ndarray,
    valid_mask: np.ndarray,
    roi_rows: int,
    roi_cols: int,
    max_features_per_roi: int,
    min_features_per_roi: int,
    max_flow_px: float,
    fb_error_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = simulated.shape
    sim_u8 = dic_uint8(simulated, valid_mask)
    exp_u8 = dic_uint8(experimental, valid_mask)
    field = np.full((roi_rows, roi_cols, 2), np.nan, dtype=np.float32)
    counts = np.zeros((roi_rows, roi_cols), dtype=np.float32)

    roi_h = height / roi_rows
    roi_w = width / roi_cols
    lk_params = dict(
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for rr in range(roi_rows):
        y0 = int(round(rr * roi_h))
        y1 = int(round((rr + 1) * roi_h))
        for cc in range(roi_cols):
            x0 = int(round(cc * roi_w))
            x1 = int(round((cc + 1) * roi_w))
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            roi_valid = valid_mask[y0:y1, x0:x1]
            if roi_valid.mean() < 0.25:
                continue
            roi_mask[y0:y1, x0:x1] = (roi_valid.astype(np.uint8) * 255)
            pts0 = cv2.goodFeaturesToTrack(
                sim_u8,
                maxCorners=max_features_per_roi,
                qualityLevel=0.01,
                minDistance=4,
                mask=roi_mask,
                blockSize=5,
                useHarrisDetector=False,
            )
            if pts0 is None or len(pts0) < min_features_per_roi:
                continue

            pts1, status_fwd, _ = cv2.calcOpticalFlowPyrLK(sim_u8, exp_u8, pts0, None, **lk_params)
            if pts1 is None or status_fwd is None:
                continue
            pts0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(exp_u8, sim_u8, pts1, None, **lk_params)
            if pts0_back is None or status_back is None:
                continue

            p0 = pts0.reshape(-1, 2)
            p1 = pts1.reshape(-1, 2)
            pb = pts0_back.reshape(-1, 2)
            status = (status_fwd.reshape(-1) > 0) & (status_back.reshape(-1) > 0)
            disp = p1 - p0
            fb = np.linalg.norm(pb - p0, axis=1)
            mag = np.linalg.norm(disp, axis=1)
            inside = (
                (p1[:, 0] >= 0)
                & (p1[:, 0] < width)
                & (p1[:, 1] >= 0)
                & (p1[:, 1] < height)
                & valid_mask[np.clip(np.round(p1[:, 1]).astype(int), 0, height - 1), np.clip(np.round(p1[:, 0]).astype(int), 0, width - 1)]
            )
            keep = status & inside & np.isfinite(disp).all(axis=1) & (mag <= max_flow_px) & (fb <= fb_error_px)
            if int(keep.sum()) < min_features_per_roi:
                continue
            good_disp = disp[keep]
            field[rr, cc] = np.median(good_disp, axis=0).astype(np.float32)
            counts[rr, cc] = float(keep.sum())

    return field, counts


def average_displacement_field(
    contexts: list[PatternContext],
    master: MasterSphere,
    dx_px: float,
    dy_px: float,
    radius_scale: float,
    args: argparse.Namespace,
) -> AverageField:
    sum_field = np.zeros((args.roi_rows, args.roi_cols, 2), dtype=np.float64)
    image_counts = np.zeros((args.roi_rows, args.roi_cols), dtype=np.float64)
    feature_counts = np.zeros((args.roi_rows, args.roi_cols), dtype=np.float64)
    iterator = tqdm(contexts, desc=f"DIC dx={dx_px:+.1f}, dy={dy_px:+.1f}, r={radius_scale:.4f}", leave=False) if tqdm is not None else contexts

    for context in iterator:
        height, width = context.prepared.image.shape
        pc = corrected_pc(context.prepared.bundle.pc, dx_px, dy_px, radius_scale, height, width)
        simulated = render_master_detector_image(context.prepared, master, context.initial_result, pc, args.render_channel)
        field, counts = compute_displacement_field(
            simulated,
            context.exp_dic,
            context.prepared.valid_mask,
            roi_rows=args.roi_rows,
            roi_cols=args.roi_cols,
            max_features_per_roi=args.max_features_per_roi,
            min_features_per_roi=args.min_features_per_roi,
            max_flow_px=args.max_flow_px,
            fb_error_px=args.fb_error_px,
        )
        valid = np.isfinite(field[..., 0]) & (counts >= args.min_features_per_roi)
        sum_field[valid] += field[valid]
        image_counts[valid] += 1.0
        feature_counts += counts

    avg_field = np.full_like(sum_field, np.nan, dtype=np.float32)
    valid_avg = image_counts > 0
    avg_field[valid_avg] = (sum_field[valid_avg] / image_counts[valid_avg][:, None]).astype(np.float32)
    return AverageField(avg_field, feature_counts.astype(np.float32), image_counts.astype(np.float32))


def weighted_rms(field: np.ndarray, counts: np.ndarray) -> float:
    valid = np.isfinite(field[..., 0]) & (counts > 0)
    if not np.any(valid):
        return float("nan")
    weights = counts[valid].astype(np.float64)
    vectors = field[valid].astype(np.float64)
    return float(np.sqrt(np.sum(weights * np.sum(vectors * vectors, axis=1)) / (np.sum(weights) + 1e-12)))


def solve_pc_update(
    base: AverageField,
    sensitivities: dict[str, np.ndarray],
    pc_step_px: float,
    radius_step: float,
    damping: float,
    max_pc_update_px: float,
    max_radius_update: float,
) -> dict:
    valid_roi = np.isfinite(base.field[..., 0]) & (base.counts > 0)
    for sensitivity in sensitivities.values():
        valid_roi &= np.isfinite(sensitivity[..., 0])
    if int(valid_roi.sum()) < 3:
        raise RuntimeError("Not enough valid DIC ROIs to solve a PC/radius update")

    b = base.field[valid_roi].reshape(-1).astype(np.float64)
    columns = [
        sensitivities["dx_px"][valid_roi].reshape(-1).astype(np.float64),
        sensitivities["dy_px"][valid_roi].reshape(-1).astype(np.float64),
        sensitivities["radius_scale"][valid_roi].reshape(-1).astype(np.float64),
    ]
    J = np.column_stack(columns)

    roi_weights = base.counts[valid_roi].astype(np.float64)
    nonzero = roi_weights[roi_weights > 0]
    c90 = float(np.percentile(nonzero, 90)) if nonzero.size else 1.0
    roi_weights = np.minimum(roi_weights / max(c90, 1e-8), 1.0)
    weights = np.repeat(np.sqrt(roi_weights), 2)
    Jw = J * weights[:, None]
    bw = b * weights

    column_norms = np.linalg.norm(Jw, axis=0)
    column_norms[column_norms < 1e-8] = 1.0
    Jn = Jw / column_norms[None, :]
    lhs = Jn.T @ Jn + damping * np.eye(Jn.shape[1])
    rhs = -Jn.T @ bw
    scaled_delta = np.linalg.solve(lhs, rhs)
    delta = scaled_delta / column_norms
    delta[0] = float(np.clip(delta[0], -max_pc_update_px, max_pc_update_px))
    delta[1] = float(np.clip(delta[1], -max_pc_update_px, max_pc_update_px))
    delta[2] = float(np.clip(delta[2], -max_radius_update, max_radius_update))

    return {
        "dx_px": float(delta[0]),
        "dy_px": float(delta[1]),
        "radius_delta": float(delta[2]),
        "radius_scale": float(1.0 + delta[2]),
        "valid_rois": int(valid_roi.sum()),
        "pc_step_px": float(pc_step_px),
        "radius_step": float(radius_step),
        "column_norms": column_norms.tolist(),
    }


def roi_centers(height: int, width: int, roi_rows: int, roi_cols: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(roi_cols, dtype=np.float32) + 0.5) * width / roi_cols
    ys = (np.arange(roi_rows, dtype=np.float32) + 0.5) * height / roi_rows
    return np.meshgrid(xs, ys)


def save_displacement_visualization(
    field: np.ndarray,
    counts: np.ndarray,
    background: np.ndarray,
    out_path: Path,
    title: str,
    vector_scale: float,
) -> None:
    height, width = background.shape
    x_grid, y_grid = roi_centers(height, width, field.shape[0], field.shape[1])
    valid = np.isfinite(field[..., 0]) & (counts > 0)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
    if np.any(valid):
        ax.quiver(
            x_grid[valid],
            y_grid[valid],
            field[..., 0][valid] * vector_scale,
            field[..., 1][valid] * vector_scale,
            counts[valid],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_visualization(
    sensitivities: dict[str, np.ndarray],
    counts: np.ndarray,
    background: np.ndarray,
    out_path: Path,
    vector_scale: float,
) -> None:
    height, width = background.shape
    x_grid, y_grid = roi_centers(height, width, next(iter(sensitivities.values())).shape[0], next(iter(sensitivities.values())).shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, (name, field) in zip(axes, sensitivities.items()):
        valid = np.isfinite(field[..., 0]) & (counts > 0)
        ax.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        if np.any(valid):
            ax.quiver(
                x_grid[valid],
                y_grid[valid],
                field[..., 0][valid] * vector_scale,
                field[..., 1][valid] * vector_scale,
                counts[valid],
                cmap="viridis",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
            )
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Numerical geometry sensitivity fields")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dic_pair_debug(
    context: PatternContext,
    master: MasterSphere,
    field: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    sim = render_master_detector_image(context.prepared, master, context.initial_result, context.prepared.bundle.pc, args.render_channel)
    height, width = sim.shape
    x_grid, y_grid = roi_centers(height, width, field.shape[0], field.shape[1])
    valid = np.isfinite(field[..., 0]) & (counts > 0)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    axes[0].imshow(context.exp_dic, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Experimental DIC image")
    axes[1].imshow(sim, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Simulated current-geometry image")
    axes[2].imshow(context.products["raw_percentile"], cmap="gray", vmin=0.0, vmax=1.0)
    if np.any(valid):
        axes[2].quiver(
            x_grid[valid],
            y_grid[valid],
            field[..., 0][valid] * args.vector_scale,
            field[..., 1][valid] * args.vector_scale,
            counts[valid],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )
    axes[2].set_title("ROI displacement field")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"DIC debug | idx={context.index}, score={context.initial_result.score:.4f}, convention={context.initial_result.convention_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_rows_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "index",
        "row",
        "col",
        "initial_score",
        "corrected_score",
        "score_gain",
        "convention",
        "line_variant",
        "corrected_pcx",
        "corrected_pcy",
        "corrected_pcz",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype global DIC-based PC/radius refinement over multiple EBSD patterns.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "global_dic_pc_refinement")
    parser.add_argument("--roi-rows", type=int, default=7)
    parser.add_argument("--roi-cols", type=int, default=7)
    parser.add_argument("--max-features-per-roi", type=int, default=70)
    parser.add_argument("--min-features-per-roi", type=int, default=4)
    parser.add_argument("--max-flow-px", type=float, default=24.0)
    parser.add_argument("--fb-error-px", type=float, default=2.5)
    parser.add_argument("--pc-step-px", type=float, default=4.0)
    parser.add_argument("--radius-step", type=float, default=0.025)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max-pc-update-px", type=float, default=18.0)
    parser.add_argument("--max-radius-update", type=float, default=0.12)
    parser.add_argument("--line-search-factors", default="1.0,0.75,0.5,0.25,0.0,-0.25,-0.5")
    parser.add_argument("--render-channel", choices=["band", "intensity"], default="band")
    parser.add_argument("--final-visualize-count", type=int, default=6)
    parser.add_argument("--sphere-lon-count", type=int, default=420)
    parser.add_argument("--sphere-colat-count", type=int, default=210)
    parser.add_argument("--vector-scale", type=float, default=5.0)
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=5000)
    parser.add_argument("--coarse-rotations", type=int, default=160)
    parser.add_argument("--refine-schedule", default="8:100,3:140,1:140")
    parser.add_argument("--local-steps-deg", default="1.5,0.5")
    parser.add_argument("--seed", type=int, default=0)
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

    out_dir = args.out_dir / args.map
    out_dir.mkdir(parents=True, exist_ok=True)
    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )
    refine_schedule = parse_refine_schedule(args.refine_schedule)
    local_steps = parse_float_list(args.local_steps_deg)

    contexts: list[PatternContext] = []
    iterator = tqdm(indices, desc="initial pattern matching") if tqdm is not None else indices
    for index in iterator:
        print(f"Preparing {map_spec.label} index={index}")
        bundle = read_pattern_bundle(args.h5, map_spec, index)
        products = build_preprocessing_products(
            bundle.pattern_u16,
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
        )
        prepared, _ = prepare_pattern(
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
        contexts.append(
            PatternContext(
                index=bundle.index,
                row=bundle.row,
                col=bundle.col,
                prepared=prepared,
                products=products,
                initial_result=result,
                exp_dic=experimental_dic_image(prepared, products),
            )
        )

    print("Computing average experimental-to-simulated DIC residual field...")
    base_field = average_displacement_field(contexts, master, 0.0, 0.0, 1.0, args)
    base_rms = weighted_rms(base_field.field, base_field.counts)

    print("Computing numerical PC/radius sensitivity fields...")
    dx_plus = average_displacement_field(contexts, master, args.pc_step_px, 0.0, 1.0, args)
    dx_minus = average_displacement_field(contexts, master, -args.pc_step_px, 0.0, 1.0, args)
    dy_plus = average_displacement_field(contexts, master, 0.0, args.pc_step_px, 1.0, args)
    dy_minus = average_displacement_field(contexts, master, 0.0, -args.pc_step_px, 1.0, args)
    r_plus = average_displacement_field(contexts, master, 0.0, 0.0, 1.0 + args.radius_step, args)
    r_minus = average_displacement_field(contexts, master, 0.0, 0.0, 1.0 - args.radius_step, args)
    sensitivities = {
        "dx_px": (dx_plus.field - dx_minus.field) / (2.0 * args.pc_step_px),
        "dy_px": (dy_plus.field - dy_minus.field) / (2.0 * args.pc_step_px),
        "radius_scale": (r_plus.field - r_minus.field) / (2.0 * args.radius_step),
    }
    proposed = solve_pc_update(
        base_field,
        sensitivities,
        pc_step_px=args.pc_step_px,
        radius_step=args.radius_step,
        damping=args.damping,
        max_pc_update_px=args.max_pc_update_px,
        max_radius_update=args.max_radius_update,
    )

    print(
        "Proposed update from average DIC field: "
        f"dx={proposed['dx_px']:+.2f}px, dy={proposed['dy_px']:+.2f}px, "
        f"radius_scale={proposed['radius_scale']:.4f}"
    )

    line_search_rows = []
    best_field = base_field
    best_update = {"dx_px": 0.0, "dy_px": 0.0, "radius_scale": 1.0, "factor": 0.0, "residual_rms_px": base_rms}
    for factor in parse_float_list(args.line_search_factors):
        dx = proposed["dx_px"] * factor
        dy = proposed["dy_px"] * factor
        radius_scale = 1.0 + proposed["radius_delta"] * factor
        candidate_field = average_displacement_field(contexts, master, dx, dy, radius_scale, args)
        rms = weighted_rms(candidate_field.field, candidate_field.counts)
        row = {
            "factor": float(factor),
            "dx_px": float(dx),
            "dy_px": float(dy),
            "radius_scale": float(radius_scale),
            "residual_rms_px": float(rms),
        }
        line_search_rows.append(row)
        if np.isfinite(rms) and rms < best_update["residual_rms_px"]:
            best_update = row
            best_field = candidate_field

    background = contexts[0].products["raw_percentile"]
    save_displacement_visualization(
        base_field.field,
        base_field.counts,
        background,
        out_dir / "01_average_dic_residual_before.png",
        f"Average DIC residual before correction | RMS={base_rms:.3f}px",
        args.vector_scale,
    )
    save_sensitivity_visualization(
        sensitivities,
        base_field.counts,
        background,
        out_dir / "02_pc_radius_sensitivity_fields.png",
        args.vector_scale,
    )
    save_displacement_visualization(
        best_field.field,
        best_field.counts,
        background,
        out_dir / "03_average_dic_residual_after.png",
        f"Average DIC residual after correction | RMS={best_update['residual_rms_px']:.3f}px",
        args.vector_scale,
    )
    save_dic_pair_debug(contexts[0], master, base_field.field, base_field.counts, out_dir / "04_first_pattern_dic_debug.png", args)

    corrected_paths: list[Path] = []
    initial_paths: list[Path] = []
    score_rows: list[dict] = []
    vis_contexts = contexts[: max(0, min(args.final_visualize_count, len(contexts)))]
    for context in tqdm(vis_contexts, desc="final corrected visualizations") if tqdm is not None else vis_contexts:
        pattern_dir = out_dir / f"idx_{context.index:05d}"
        pattern_dir.mkdir(parents=True, exist_ok=True)
        initial_path = pattern_dir / "01_initial_weighted_final_spatial.png"
        corrected_path = pattern_dir / "02_global_dic_pc_corrected_final_spatial.png"
        save_final_spatial_visualization(
            context.initial_result,
            master,
            context.products,
            initial_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )
        height, width = context.prepared.image.shape
        corrected_tuple = corrected_pc(
            context.prepared.bundle.pc,
            best_update["dx_px"],
            best_update["dy_px"],
            best_update["radius_scale"],
            height,
            width,
        )
        corrected_prepared = prepared_with_pc(context.prepared, corrected_tuple)
        corrected_rotation, corrected_score = deterministic_rotation_refine(
            corrected_prepared,
            master,
            context.initial_result.detector_transform,
            context.initial_result.rotation,
            local_steps,
        )
        corrected_result = MatchResult(
            label="Global-DIC-PC-corrected",
            score=float(corrected_score),
            rotation=corrected_rotation,
            convention_name=context.initial_result.convention_name,
            detector_transform=context.initial_result.detector_transform,
            prepared=corrected_prepared,
        )
        save_final_spatial_visualization(
            corrected_result,
            master,
            context.products,
            corrected_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )
        initial_paths.append(initial_path)
        corrected_paths.append(corrected_path)
        score_rows.append(
            {
                "index": context.index,
                "row": context.row,
                "col": context.col,
                "initial_score": float(context.initial_result.score),
                "corrected_score": float(corrected_score),
                "score_gain": float(corrected_score - context.initial_result.score),
                "convention": context.initial_result.convention_name,
                "line_variant": context.prepared.line_variant.name,
                "corrected_pcx": float(corrected_tuple[0]),
                "corrected_pcy": float(corrected_tuple[1]),
                "corrected_pcz": float(corrected_tuple[2]),
            }
        )

    if initial_paths:
        make_contact_sheet(initial_paths, out_dir / "contact_sheet_initial_weighted.png", thumb_width=880, columns=2)
    if corrected_paths:
        make_contact_sheet(corrected_paths, out_dir / "contact_sheet_global_dic_corrected.png", thumb_width=880, columns=2)
    write_rows_csv(score_rows, out_dir / "visualized_score_summary.csv")
    with (out_dir / "line_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["factor", "dx_px", "dy_px", "radius_scale", "residual_rms_px"])
        writer.writeheader()
        writer.writerows(line_search_rows)

    summary = {
        "map": map_spec.key,
        "indices": indices,
        "method": "Prototype global DIC PC/radius refinement. Local DIC residual fields are averaged across patterns; PCx/PCy/radius sensitivities are estimated by finite differences; a damped least-squares update is line-searched by average residual RMS.",
        "base_residual_rms_px": base_rms,
        "proposed_update": proposed,
        "best_update": best_update,
        "line_search": line_search_rows,
        "dic_parameters": {
            "roi_rows": args.roi_rows,
            "roi_cols": args.roi_cols,
            "max_features_per_roi": args.max_features_per_roi,
            "min_features_per_roi": args.min_features_per_roi,
            "max_flow_px": args.max_flow_px,
            "fb_error_px": args.fb_error_px,
            "render_channel": args.render_channel,
        },
        "outputs": {
            "average_before": str(out_dir / "01_average_dic_residual_before.png"),
            "sensitivities": str(out_dir / "02_pc_radius_sensitivity_fields.png"),
            "average_after": str(out_dir / "03_average_dic_residual_after.png"),
            "dic_debug": str(out_dir / "04_first_pattern_dic_debug.png"),
            "initial_contact_sheet": str(out_dir / "contact_sheet_initial_weighted.png"),
            "corrected_contact_sheet": str(out_dir / "contact_sheet_global_dic_corrected.png"),
            "line_search_csv": str(out_dir / "line_search.csv"),
            "score_summary_csv": str(out_dir / "visualized_score_summary.csv"),
        },
        "visualized_patterns": score_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved global DIC PC refinement prototype to: {out_dir}")
    print(
        f"Best line-search update: dx={best_update['dx_px']:+.2f}px, "
        f"dy={best_update['dy_px']:+.2f}px, radius_scale={best_update['radius_scale']:.4f}, "
        f"residual RMS {base_rms:.3f}px -> {best_update['residual_rms_px']:.3f}px"
    )


if __name__ == "__main__":
    main()
