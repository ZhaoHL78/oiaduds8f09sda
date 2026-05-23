from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    PreparedPattern,
    default_map_specs,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
    score_rotation,
    zscore_vector,
)
from labeled_band_radius_refinement import (
    HKLFamily,
    assign_labels,
    band_plane_normal,
    label_score,
    read_phase_hkl_families,
    save_detector_label_overlay,
    save_labeled_alignment,
)
from pc_radius_bias_correction import corrected_pc, prepared_with_pc
from visualize_calibration_pipeline import build_preprocessing_products, parse_refine_schedule, save_final_spatial_visualization

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def family_by_label(families: list[HKLFamily]) -> dict[str, HKLFamily]:
    return {family.label: family for family in families}


def transformed_band_normal(
    prepared: PreparedPattern,
    segment_index: int,
    detector_transform: np.ndarray,
    rotation: R,
) -> np.ndarray:
    segment = prepared.line_segments[segment_index]
    normal = band_plane_normal(prepared, segment)
    normal = normal @ detector_transform.T
    normal = rotation.apply(normal)
    normal = normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    return normal


def same_hkl_angle_deg(normal: np.ndarray, family: HKLFamily) -> float:
    dots = np.abs(family.normals @ normal.astype(np.float32))
    best = float(np.max(dots))
    return float(np.degrees(np.arccos(np.clip(best, -1.0, 1.0))))


def write_rows_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_point_indices(prepared: PreparedPattern, count: int, seed: int) -> np.ndarray:
    n_points = int(len(prepared.exp_points))
    if count <= 0 or count >= n_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    response = prepared.combined_response[prepared.match_mask].astype(np.float32)
    # Keep the strongest half deterministic, and sample the rest. This keeps
    # the residual tied to real Kikuchi bands while avoiding a single ridge
    # dominating every Jacobian evaluation.
    strong_count = min(count // 2, n_points)
    strong = np.argpartition(response, -strong_count)[-strong_count:]
    remaining = np.setdiff1d(np.arange(n_points), strong, assume_unique=False)
    sample_count = count - strong_count
    sampled = rng.choice(remaining, size=sample_count, replace=False) if sample_count > 0 and len(remaining) else np.array([], dtype=np.int64)
    return np.sort(np.concatenate([strong.astype(np.int64), sampled.astype(np.int64)]))


def parameters_to_state(
    params: np.ndarray,
    initial_rotation: R,
    base_pc: tuple[float, float, float],
    height: int,
    width: int,
    optimize_pc: bool,
) -> tuple[R, tuple[float, float, float], dict]:
    delta_rotation = R.from_rotvec(params[:3])
    rotation = delta_rotation * initial_rotation
    radius_scale = float(params[3])
    if optimize_pc:
        dx_px = float(params[4])
        dy_px = float(params[5])
    else:
        dx_px = 0.0
        dy_px = 0.0
    pc = corrected_pc(base_pc, dx_px, dy_px, radius_scale, height, width)
    state = {
        "rotvec_x_deg": float(np.degrees(params[0])),
        "rotvec_y_deg": float(np.degrees(params[1])),
        "rotvec_z_deg": float(np.degrees(params[2])),
        "delta_angle_deg": float(np.degrees(np.linalg.norm(params[:3]))),
        "radius_scale": radius_scale,
        "dx_px": dx_px,
        "dy_px": dy_px,
        "pcx": float(pc[0]),
        "pcy": float(pc[1]),
        "pcz": float(pc[2]),
    }
    return rotation, pc, state


def trace_row_to_params(row: dict, optimize_pc: bool) -> np.ndarray:
    params = [
        np.radians(float(row["rotvec_x_deg"])),
        np.radians(float(row["rotvec_y_deg"])),
        np.radians(float(row["rotvec_z_deg"])),
        float(row["radius_scale"]),
    ]
    if optimize_pc:
        params.extend([float(row["dx_px"]), float(row["dy_px"])])
    return np.asarray(params, dtype=np.float64)


def band_angle_rows(
    result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
) -> list[dict]:
    rows: list[dict] = []
    for segment_index, segment in enumerate(result.prepared.line_segments):
        hkl = fixed_hkl_by_band.get(segment.band_index)
        if hkl not in families_by_label:
            continue
        normal = transformed_band_normal(result.prepared, segment_index, result.detector_transform, result.rotation)
        angle = same_hkl_angle_deg(normal, families_by_label[hkl])
        rows.append(
            {
                "band_index": int(segment.band_index),
                "hkl": hkl,
                "angle_deg": float(angle),
                "band_intensity": float(segment.band.intensity),
                "rho_bin": float(segment.band.rho_bin),
                "theta_deg": float(segment.band.theta_deg),
                "width": float(segment.band.width),
            }
        )
    return rows


def save_band_residual_comparison(initial_rows: list[dict], final_rows: list[dict], out_path: Path) -> None:
    final_by_band = {row["band_index"]: row for row in final_rows}
    bands = [row["band_index"] for row in initial_rows if row["band_index"] in final_by_band]
    if not bands:
        return
    initial_angles = [row["angle_deg"] for row in initial_rows if row["band_index"] in final_by_band]
    final_angles = [final_by_band[band]["angle_deg"] for band in bands]
    labels = [f"{band}:{final_by_band[band]['hkl']}" for band in bands]

    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.bar(x - 0.18, initial_angles, width=0.36, label="initial", color="#9ca3af")
    ax.bar(x + 0.18, final_angles, width=0.36, label="refined", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Same-HKL band normal angle error (deg)")
    ax.set_title("Per-band geometric residuals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trace_plot(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    evals = [row["evaluation"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4))
    axes[0, 0].plot(evals, [row["loss"] for row in rows], color="#2563eb")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(evals, [row["mean_band_angle_deg"] for row in rows], color="#7c3aed")
    axes[0, 1].set_title("Mean band angle error")
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(evals, [row["match_score"] for row in rows], color="#16a34a")
    axes[1, 0].set_title("Image / band correlation score")
    axes[1, 0].set_ylabel("score")
    axes[1, 0].set_xlabel("function evaluation")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(evals, [row["delta_angle_deg"] for row in rows], label="rotation delta", color="#f97316")
    axes[1, 1].plot(evals, [row["radius_scale"] for row in rows], label="radius scale", color="#0f766e")
    axes[1, 1].set_title("Optimized transform parameters")
    axes[1, 1].set_xlabel("function evaluation")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def optimize_continuous_geometry(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    families: list[HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
    seed: int,
) -> tuple[MatchResult, dict, list[dict]]:
    height, width = prepared.image.shape
    point_indices = choose_point_indices(prepared, args.residual_points, seed)
    families_by_label = family_by_label(families)
    segment_indices = [
        i
        for i, segment in enumerate(prepared.line_segments)
        if segment.band_index in fixed_hkl_by_band and fixed_hkl_by_band[segment.band_index] in families_by_label
    ]
    if not segment_indices:
        raise ValueError("No H5 bands could be assigned to HKL families for continuous refinement")

    x0 = [0.0, 0.0, 0.0, 1.0]
    lower = [-np.radians(args.rotation_bound_deg)] * 3 + [args.radius_min]
    upper = [np.radians(args.rotation_bound_deg)] * 3 + [args.radius_max]
    if args.optimize_pc:
        x0.extend([0.0, 0.0])
        lower.extend([-args.pc_bound_px, -args.pc_bound_px])
        upper.extend([args.pc_bound_px, args.pc_bound_px])

    trace_rows: list[dict] = []
    eval_counter = {"value": 0}

    def build_result(params: np.ndarray) -> MatchResult:
        rotation, pc, _ = parameters_to_state(
            params,
            initial_result.rotation,
            prepared.bundle.pc,
            height,
            width,
            args.optimize_pc,
        )
        candidate_prepared = prepared_with_pc(prepared, pc)
        score = score_rotation(rotation, candidate_prepared.exp_points @ initial_result.detector_transform.T, candidate_prepared, master)
        return MatchResult(
            label="continuous-band-geometric-refined",
            score=float(score),
            rotation=rotation,
            convention_name=initial_result.convention_name,
            detector_transform=initial_result.detector_transform,
            prepared=candidate_prepared,
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        eval_counter["value"] += 1
        rotation, pc, state = parameters_to_state(
            params,
            initial_result.rotation,
            prepared.bundle.pc,
            height,
            width,
            args.optimize_pc,
        )
        candidate_prepared = prepared_with_pc(prepared, pc)
        points = candidate_prepared.exp_points[point_indices] @ initial_result.detector_transform.T
        rotated = rotation.apply(points)
        master_band_z = zscore_vector(master.sample_band(rotated))
        master_intensity_z = zscore_vector(master.sample_intensity(rotated))
        n_points = max(1, len(point_indices))

        residual_parts: list[np.ndarray] = []
        if args.image_line_weight > 0:
            residual_parts.append(
                np.sqrt(args.image_line_weight / n_points)
                * (master_band_z - prepared.exp_image_band_z[point_indices])
            )
        if args.h5_band_weight > 0:
            residual_parts.append(
                np.sqrt(args.h5_band_weight / n_points)
                * (master_band_z - prepared.exp_h5_band_z[point_indices])
            )
        if args.intensity_weight > 0:
            residual_parts.append(
                np.sqrt(args.intensity_weight / n_points)
                * (master_intensity_z - prepared.exp_intensity_z[point_indices])
            )

        band_residuals = []
        band_angles = []
        band_weights = []
        for segment_index in segment_indices:
            segment = candidate_prepared.line_segments[segment_index]
            hkl = fixed_hkl_by_band[segment.band_index]
            normal = transformed_band_normal(candidate_prepared, segment_index, initial_result.detector_transform, rotation)
            family = families_by_label[hkl]
            dots = np.abs(family.normals @ normal.astype(np.float32))
            best_dot = float(np.max(dots))
            angle_rad = float(np.arccos(np.clip(best_dot, -1.0, 1.0)))
            intensity_weight = max(1e-3, float(segment.band.intensity))
            band_weights.append(intensity_weight)
            band_angles.append(float(np.degrees(angle_rad)))
            band_residuals.append(angle_rad / np.radians(args.band_angle_scale_deg))

        band_weights_arr = np.asarray(band_weights, dtype=np.float32)
        band_weights_arr /= band_weights_arr.sum() + 1e-8
        residual_parts.append(
            np.sqrt(args.band_geometry_weight * band_weights_arr)
            * np.asarray(band_residuals, dtype=np.float32)
        )

        if args.radius_regularization_weight > 0:
            residual_parts.append(
                np.asarray(
                    [
                        np.sqrt(args.radius_regularization_weight)
                        * ((state["radius_scale"] - 1.0) / max(args.radius_regularization_sigma, 1e-8))
                    ],
                    dtype=np.float32,
                )
            )
        if args.optimize_pc and args.pc_regularization_weight > 0:
            residual_parts.append(
                np.sqrt(args.pc_regularization_weight)
                * np.asarray(
                    [
                        state["dx_px"] / max(args.pc_regularization_sigma_px, 1e-8),
                        state["dy_px"] / max(args.pc_regularization_sigma_px, 1e-8),
                    ],
                    dtype=np.float32,
                )
            )

        residual_vector = np.concatenate([part.astype(np.float32).ravel() for part in residual_parts])
        match_score = score_rotation(rotation, candidate_prepared.exp_points @ initial_result.detector_transform.T, candidate_prepared, master)
        trace_rows.append(
            {
                "evaluation": eval_counter["value"],
                **state,
                "match_score": float(match_score),
                "mean_band_angle_deg": float(np.mean(band_angles)),
                "max_band_angle_deg": float(np.max(band_angles)),
                "residual_rms": float(np.sqrt(np.mean(residual_vector**2))),
                "loss": float(np.sum(residual_vector**2)),
            }
        )
        return residual_vector

    optimizer_result = least_squares(
        residuals,
        np.asarray(x0, dtype=np.float64),
        bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        max_nfev=args.max_nfev,
        xtol=args.xtol,
        ftol=args.ftol,
        gtol=args.gtol,
        x_scale="jac",
        verbose=2 if args.verbose_optimizer else 0,
    )
    selected_params = optimizer_result.x
    selected_reason = "optimizer_result"
    min_allowed_score = float(initial_result.score) - float(args.max_match_score_drop)
    if trace_rows:
        feasible_rows = [row for row in trace_rows if float(row["match_score"]) >= min_allowed_score]
        if feasible_rows:
            selected_row = min(feasible_rows, key=lambda row: (float(row["mean_band_angle_deg"]), float(row["loss"])))
            selected_params = trace_row_to_params(selected_row, args.optimize_pc)
            selected_reason = "best_band_angle_with_match_score_guard"
        elif float(trace_rows[-1]["match_score"]) < min_allowed_score:
            selected_row = max(trace_rows, key=lambda row: float(row["match_score"]))
            selected_params = trace_row_to_params(selected_row, args.optimize_pc)
            selected_reason = "fallback_highest_match_score"

    final_result = build_result(selected_params)
    final_rotation, final_pc, final_state = parameters_to_state(
        selected_params,
        initial_result.rotation,
        prepared.bundle.pc,
        height,
        width,
        args.optimize_pc,
    )
    final_state.update(
        {
            "selected_reason": selected_reason,
            "max_match_score_drop": float(args.max_match_score_drop),
            "minimum_allowed_match_score": min_allowed_score,
            "raw_optimizer_match_score": float(build_result(optimizer_result.x).score),
            "optimizer_success": bool(optimizer_result.success),
            "optimizer_status": int(optimizer_result.status),
            "optimizer_message": str(optimizer_result.message),
            "optimizer_nfev": int(optimizer_result.nfev),
            "optimizer_cost": float(optimizer_result.cost),
            "optimizer_optimality": float(optimizer_result.optimality),
            "final_match_score": float(final_result.score),
            "final_rotation_quat_xyzw": final_rotation.as_quat().tolist(),
            "final_pc": {"pcx": final_pc[0], "pcy": final_pc[1], "pcz": final_pc[2]},
            "point_indices_count": int(len(point_indices)),
            "band_count": int(len(segment_indices)),
        }
    )
    return final_result, final_state, trace_rows


def process_one(args, map_spec, master, index: int, batch_dir: Path) -> dict:
    out_dir = batch_dir / f"idx_{index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = read_pattern_bundle(args.h5, map_spec, index)
    phase_id = int(bundle.ang_record.get("Phase", 1))
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)
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
    initial_result = match_to_master(
        prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=parse_refine_schedule(args.refine_schedule),
        random_seed=args.seed + int(index),
    )
    initial_assignments = assign_labels(initial_result, families)
    fixed_hkl_by_band = {item.band_index: item.hkl for item in initial_assignments}
    families_by_label = family_by_label(families)
    initial_label_score, initial_label_angle = label_score(initial_assignments)

    initial_rows = band_angle_rows(initial_result, families_by_label, fixed_hkl_by_band)
    write_rows_csv(initial_rows, out_dir / "02_initial_band_angle_residuals.csv")
    save_detector_label_overlay(initial_result, initial_assignments, out_dir / "01_detector_bands_initial_inferred_hkl.png")
    save_labeled_alignment(
        initial_result,
        master,
        initial_assignments,
        out_dir / "03_initial_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    final_result, final_state, trace_rows = optimize_continuous_geometry(
        prepared,
        master,
        initial_result,
        families,
        fixed_hkl_by_band,
        args,
        seed=args.seed + int(index),
    )
    final_assignments = assign_labels(final_result, families, fixed_hkl_by_band=fixed_hkl_by_band)
    final_label_score, final_label_angle = label_score(final_assignments)
    final_rows = band_angle_rows(final_result, families_by_label, fixed_hkl_by_band)
    write_rows_csv(final_rows, out_dir / "07_final_band_angle_residuals.csv")
    write_rows_csv(trace_rows, out_dir / "04_optimizer_evaluation_trace.csv")
    save_trace_plot(trace_rows, out_dir / "05_optimizer_trace.png")
    save_band_residual_comparison(initial_rows, final_rows, out_dir / "06_band_angle_residual_comparison.png")
    save_labeled_alignment(
        final_result,
        master,
        final_assignments,
        out_dir / "08_refined_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        final_result,
        master,
        products,
        out_dir / "09_refined_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    hyperparameters = {
        "initial_match": {
            "coarse_rotations": args.coarse_rotations,
            "refine_schedule": args.refine_schedule,
            "seed": args.seed + int(index),
            "detector_convention": initial_result.convention_name,
        },
        "preprocessing": {
            "mask_radius_frac": args.mask_radius_frac,
            "mask_erosion": args.mask_erosion,
            "background_sigma": args.background_sigma,
            "band_sigma_min": args.band_sigma_min,
            "band_sigma_max": args.band_sigma_max,
            "match_quantile": args.match_quantile,
            "top_k_points": args.top_k_points,
        },
            "least_squares": {
                "max_nfev": args.max_nfev,
                "xtol": args.xtol,
                "ftol": args.ftol,
                "gtol": args.gtol,
                "rotation_bound_deg": args.rotation_bound_deg,
                "radius_min": args.radius_min,
                "radius_max": args.radius_max,
                "optimize_pc": args.optimize_pc,
                "pc_bound_px": args.pc_bound_px,
                "max_match_score_drop": args.max_match_score_drop,
            },
        "loss_weights": {
            "image_line_weight": args.image_line_weight,
            "h5_band_weight": args.h5_band_weight,
            "intensity_weight": args.intensity_weight,
            "band_geometry_weight": args.band_geometry_weight,
            "band_angle_scale_deg": args.band_angle_scale_deg,
            "radius_regularization_weight": args.radius_regularization_weight,
            "radius_regularization_sigma": args.radius_regularization_sigma,
            "pc_regularization_weight": args.pc_regularization_weight,
            "pc_regularization_sigma_px": args.pc_regularization_sigma_px,
            "residual_points": args.residual_points,
        },
    }
    (out_dir / "hyperparameters.json").write_text(json.dumps(jsonable(hyperparameters), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "map": map_spec.key,
        "map_label": map_spec.label,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase": phase_info,
        "hkl_families": [
            {key: value for key, value in asdict(family).items() if key != "normals"} | {"label": family.label, "normal_count": int(len(family.normals))}
            for family in families
        ],
        "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "fixed_hkl_by_band": fixed_hkl_by_band,
        "initial_match": initial_result.to_json_dict(),
        "initial_label_score": initial_label_score,
        "initial_label_mean_angle_deg": initial_label_angle,
        "initial_mean_band_angle_deg": float(np.mean([row["angle_deg"] for row in initial_rows])) if initial_rows else float("nan"),
        "final_match": final_result.to_json_dict(),
        "final_label_score": final_label_score,
        "final_label_mean_angle_deg": final_label_angle,
        "final_mean_band_angle_deg": float(np.mean([row["angle_deg"] for row in final_rows])) if final_rows else float("nan"),
        "optimization": final_state,
        "hyperparameters": hyperparameters,
        "outputs": {
            "detector_labels": str(out_dir / "01_detector_bands_initial_inferred_hkl.png"),
            "initial_band_residuals_csv": str(out_dir / "02_initial_band_angle_residuals.csv"),
            "initial_labeled_alignment": str(out_dir / "03_initial_labeled_band_alignment.png"),
            "optimizer_trace_csv": str(out_dir / "04_optimizer_evaluation_trace.csv"),
            "optimizer_trace_plot": str(out_dir / "05_optimizer_trace.png"),
            "band_residual_comparison": str(out_dir / "06_band_angle_residual_comparison.png"),
            "final_band_residuals_csv": str(out_dir / "07_final_band_angle_residuals.csv"),
            "refined_labeled_alignment": str(out_dir / "08_refined_labeled_band_alignment.png"),
            "refined_final_spatial": str(out_dir / "09_refined_final_spatial.png"),
            "hyperparameters": str(out_dir / "hyperparameters.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_csv(summaries: list[dict], out_path: Path) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "index": summary["index"],
                "row": summary["row"],
                "col": summary["col"],
                "phase_id": summary["phase"]["phase_id"],
                "initial_match_score": summary["initial_match"]["score"],
                "final_match_score": summary["final_match"]["score"],
                "match_score_gain": summary["final_match"]["score"] - summary["initial_match"]["score"],
                "initial_label_mean_angle_deg": summary["initial_label_mean_angle_deg"],
                "final_label_mean_angle_deg": summary["final_label_mean_angle_deg"],
                "label_angle_gain_deg": summary["initial_label_mean_angle_deg"] - summary["final_label_mean_angle_deg"],
                "initial_mean_band_angle_deg": summary["initial_mean_band_angle_deg"],
                "final_mean_band_angle_deg": summary["final_mean_band_angle_deg"],
                "band_angle_gain_deg": summary["initial_mean_band_angle_deg"] - summary["final_mean_band_angle_deg"],
                "delta_angle_deg": summary["optimization"]["delta_angle_deg"],
                "radius_scale": summary["optimization"]["radius_scale"],
                "dx_px": summary["optimization"]["dx_px"],
                "dy_px": summary["optimization"]["dy_px"],
                "optimizer_nfev": summary["optimization"]["optimizer_nfev"],
                "optimizer_success": summary["optimization"]["optimizer_success"],
            }
        )
    write_rows_csv(rows, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous rotation/radius refinement using per-band same-HKL Kikuchi geometry residuals.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma list or Python-like ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "continuous_band_geometric_refinement")

    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=5000)

    parser.add_argument("--coarse-rotations", type=int, default=320)
    parser.add_argument("--refine-schedule", default="10:180,4:220,1.5:220")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--xtol", type=float, default=1e-7)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument("--rotation-bound-deg", type=float, default=6.0)
    parser.add_argument("--radius-min", type=float, default=0.98)
    parser.add_argument("--radius-max", type=float, default=1.02)
    parser.add_argument("--optimize-pc", action="store_true")
    parser.add_argument("--pc-bound-px", type=float, default=4.0)
    parser.add_argument("--verbose-optimizer", action="store_true")
    parser.add_argument("--max-match-score-drop", type=float, default=0.02)

    parser.add_argument("--residual-points", type=int, default=1600)
    parser.add_argument("--image-line-weight", type=float, default=1.0)
    parser.add_argument("--h5-band-weight", type=float, default=0.8)
    parser.add_argument("--intensity-weight", type=float, default=0.15)
    parser.add_argument("--band-geometry-weight", type=float, default=0.6)
    parser.add_argument("--band-angle-scale-deg", type=float, default=8.0)
    parser.add_argument("--radius-regularization-weight", type=float, default=0.08)
    parser.add_argument("--radius-regularization-sigma", type=float, default=0.02)
    parser.add_argument("--pc-regularization-weight", type=float, default=0.05)
    parser.add_argument("--pc-regularization-sigma-px", type=float, default=3.0)

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
    indices = parse_indices(args.indices, total, args.count, args.strategy) if args.indices or args.count > 1 else [args.index]
    if not indices:
        raise ValueError("No valid pattern indices selected")

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    iterator = tqdm(indices, desc="continuous band geometric refinement") if tqdm is not None else indices
    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        summaries.append(process_one(args, map_spec, master, int(index), batch_dir))

    write_batch_csv(summaries, batch_dir / "batch_continuous_band_refinement_summary.csv")
    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "06_band_angle_residual_comparison.png" for summary in summaries],
        batch_dir / "contact_sheet_band_angle_residuals.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "08_refined_labeled_band_alignment.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_labeled_band_alignment.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "09_refined_final_spatial.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_final_spatial.png",
        thumb_width=900,
        columns=2,
    )

    print(f"Saved continuous band geometric refinement results to: {batch_dir}")
    print(f"Batch CSV: {batch_dir / 'batch_continuous_band_refinement_summary.csv'}")
    print(f"Final contact sheet: {batch_dir / 'contact_sheet_refined_final_spatial.png'}")


if __name__ == "__main__":
    main()
