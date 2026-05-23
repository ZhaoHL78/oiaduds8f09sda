from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from continuous_band_geometric_refinement import band_angle_rows, family_by_label, write_rows_csv
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    LineSegment,
    MatchResult,
    MatchWeights,
    PreparedPattern,
    default_map_specs,
    detector_pixels_to_sphere,
    hough_rho_to_pixels,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
    score_rotation,
)
from labeled_band_radius_refinement import (
    HKLFamily,
    assign_labels,
    band_plane_normal,
    great_circle_from_normal,
    label_score,
    read_phase_hkl_families,
    save_detector_label_overlay,
    save_labeled_alignment,
)
from pc_radius_bias_correction import corrected_pc, prepared_with_pc
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    parse_refine_schedule,
    plot_master_sphere,
    save_final_spatial_visualization,
    set_3d_sphere_axes,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


def canonical_hough(theta_deg: float, rho_px: float) -> tuple[float, float]:
    theta = float(theta_deg) % 360.0
    rho = float(rho_px)
    if theta >= 180.0:
        theta -= 180.0
        rho = -rho
    if theta < 0.0:
        theta += 180.0
        rho = -rho
    return theta, rho


def hough_theta_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(((a_deg - b_deg + 90.0) % 180.0) - 90.0)


def observed_hough_point(prepared: PreparedPattern, segment: LineSegment) -> dict:
    height, width = prepared.image.shape
    variant = prepared.line_variant
    rho_px = variant.rho_sign * hough_rho_to_pixels(segment.band.rho_bin, prepared.bundle.ohp_header, height, width)
    theta_deg = segment.band.theta_deg + (90.0 if variant.theta_is_line_angle else 0.0)
    theta_deg, rho_px = canonical_hough(theta_deg, rho_px)
    return {
        "theta_deg": theta_deg,
        "rho_px": rho_px,
        "rho_norm": rho_px / max(1.0, 0.5 * min(height, width)),
    }


def predicted_hough_point_from_master_normal(
    master_normal: np.ndarray,
    rotation: R,
    detector_transform: np.ndarray,
    pc: tuple[float, float, float],
    height: int,
    width: int,
    y_axis: str,
) -> tuple[float, float] | None:
    normal_after_transform = rotation.inv().apply(master_normal.astype(np.float64))
    normal_local = normal_after_transform @ detector_transform
    normal_local = normal_local.astype(np.float64)
    norm = float(np.linalg.norm(normal_local))
    if norm <= 1e-10:
        return None
    normal_local /= norm

    pcx, pcy, pcz = pc
    center_col = (width - 1) / 2.0
    center_row = (height - 1) / 2.0
    pc_col = pcx * (width - 1)
    pc_row = pcy * (height - 1)
    detector_distance_px = pcz * height

    nx, ny, nz = normal_local
    coeff_x = nx
    coeff_y = ny if y_axis == "up" else -ny
    constant = nx * (center_col - pc_col) - ny * (center_row - pc_row) + nz * detector_distance_px
    coeff_norm = float(np.hypot(coeff_x, coeff_y))
    if coeff_norm <= 1e-10:
        return None

    theta_deg = float(np.degrees(np.arctan2(coeff_y / coeff_norm, coeff_x / coeff_norm)))
    rho_px = float(-constant / coeff_norm)
    return canonical_hough(theta_deg, rho_px)


def hough_point_match_for_segment(
    result: MatchResult,
    segment: LineSegment,
    family: HKLFamily,
    args,
) -> dict:
    prepared = result.prepared
    height, width = prepared.image.shape
    obs = observed_hough_point(prepared, segment)
    obs_normal = band_plane_normal(prepared, segment)
    obs_normal = obs_normal @ result.detector_transform.T
    obs_normal = result.rotation.apply(obs_normal)
    obs_normal = obs_normal.astype(np.float32)
    obs_normal /= np.linalg.norm(obs_normal) + 1e-8

    rho_scale_px = args.rho_scale_px if args.rho_scale_px > 0 else max(1.0, args.rho_scale_fraction * min(height, width))
    best: dict | None = None
    for normal in family.normals:
        pred = predicted_hough_point_from_master_normal(
            normal,
            result.rotation,
            result.detector_transform,
            prepared.bundle.pc,
            height,
            width,
            prepared.line_variant.y_axis,
        )
        if pred is None:
            continue
        pred_theta, pred_rho = pred
        theta_error = hough_theta_delta_deg(pred_theta, obs["theta_deg"])
        rho_error = pred_rho - obs["rho_px"]
        normal_dot = float(abs(np.dot(normal.astype(np.float32), obs_normal)))
        angle_deg = float(np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0))))
        cost = (
            args.hough_theta_weight * (theta_error / max(args.theta_scale_deg, 1e-8)) ** 2
            + args.hough_rho_weight * (rho_error / rho_scale_px) ** 2
            + args.band_normal_weight * (angle_deg / max(args.band_angle_scale_deg, 1e-8)) ** 2
        )
        row = {
            "band_index": int(segment.band_index),
            "hkl": family.label,
            "observed_theta_deg": float(obs["theta_deg"]),
            "observed_rho_px": float(obs["rho_px"]),
            "observed_rho_norm": float(obs["rho_norm"]),
            "predicted_theta_deg": float(pred_theta),
            "predicted_rho_px": float(pred_rho),
            "predicted_rho_norm": float(pred_rho / max(1.0, 0.5 * min(height, width))),
            "theta_error_deg": float(theta_error),
            "rho_error_px": float(rho_error),
            "rho_error_norm": float(rho_error / max(1.0, 0.5 * min(height, width))),
            "band_normal_angle_deg": float(angle_deg),
            "cost": float(cost),
            "band_intensity": float(segment.band.intensity),
            "master_normal_x": float(normal[0]),
            "master_normal_y": float(normal[1]),
            "master_normal_z": float(normal[2]),
        }
        if best is None or row["cost"] < best["cost"]:
            best = row
    if best is None:
        raise ValueError(f"Could not project HKL family {family.label} into detector Hough space")
    return best


def hough_point_match_rows(
    result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
) -> list[dict]:
    rows: list[dict] = []
    for segment in result.prepared.line_segments:
        hkl = fixed_hkl_by_band.get(segment.band_index)
        if hkl not in families_by_label:
            continue
        rows.append(hough_point_match_for_segment(result, segment, families_by_label[hkl], args))
    return rows


def hough_point_stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mean_abs_theta_error_deg": float("nan"),
            "mean_abs_rho_error_px": float("nan"),
            "mean_band_normal_angle_deg": float("nan"),
            "loss": float("nan"),
        }
    weights = np.asarray([max(1e-3, row["band_intensity"]) for row in rows], dtype=np.float64)
    weights /= weights.sum() + 1e-12
    theta = np.asarray([abs(row["theta_error_deg"]) for row in rows], dtype=np.float64)
    rho = np.asarray([abs(row["rho_error_px"]) for row in rows], dtype=np.float64)
    angle = np.asarray([row["band_normal_angle_deg"] for row in rows], dtype=np.float64)
    cost = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    return {
        "mean_abs_theta_error_deg": float(np.sum(weights * theta)),
        "mean_abs_rho_error_px": float(np.sum(weights * rho)),
        "mean_band_normal_angle_deg": float(np.sum(weights * angle)),
        "loss": float(np.sum(weights * cost)),
    }


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


def save_hough_match_visualization(
    prepared: PreparedPattern,
    rows: list[dict],
    out_path: Path,
    title: str,
) -> None:
    raw_display = detector_raw_display(prepared)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(prepared.line_segments))))
    color_by_band = {segment.band_index: colors[i] for i, segment in enumerate(prepared.line_segments)}

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4))
    axes[0].imshow(raw_display, cmap="gray", vmin=0.0, vmax=1.0)
    for segment in prepared.line_segments:
        color = color_by_band[segment.band_index]
        axes[0].plot([segment.col0, segment.col1], [segment.row0, segment.row1], color=color, linewidth=2.0)
        axes[0].text(
            0.5 * (segment.col0 + segment.col1),
            0.5 * (segment.row0 + segment.row1),
            str(segment.band_index),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.42, "boxstyle": "round,pad=0.2", "linewidth": 0},
        )
    axes[0].set_title("Detector Kikuchi bands")
    axes[0].axis("off")

    for row in rows:
        color = color_by_band.get(int(row["band_index"]), "tab:blue")
        obs = (row["observed_theta_deg"], row["observed_rho_norm"])
        pred = (row["predicted_theta_deg"], row["predicted_rho_norm"])
        axes[1].plot([obs[0], pred[0]], [obs[1], pred[1]], color=color, linewidth=1.0, alpha=0.65)
        axes[1].scatter(obs[0], obs[1], s=62, color=color, edgecolor="black", linewidth=0.5, marker="o")
        axes[1].scatter(pred[0], pred[1], s=70, color=color, linewidth=0.8, marker="x")
        axes[1].text(obs[0], obs[1], f"{row['band_index']}:{row['hkl']}", fontsize=8, color="black")
    axes[1].set_xlim(0, 180)
    axes[1].set_xlabel("Hough theta, detector normal convention (deg)")
    axes[1].set_ylabel("rho / detector radius")
    axes[1].set_title("Hough points: circle=observed, x=predicted from master")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title, y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trace_plot(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    evals = [row["evaluation"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.0))
    axes[0, 0].plot(evals, [row["hough_loss"] for row in rows], color="#2563eb")
    axes[0, 0].set_title("Hough point loss")
    axes[0, 1].plot(evals, [row["mean_abs_theta_error_deg"] for row in rows], color="#7c3aed")
    axes[0, 1].set_title("Mean |theta error|")
    axes[0, 2].plot(evals, [row["mean_abs_rho_error_px"] for row in rows], color="#db2777")
    axes[0, 2].set_title("Mean |rho error| (px)")
    axes[1, 0].plot(evals, [row["mean_band_normal_angle_deg"] for row in rows], color="#16a34a")
    axes[1, 0].set_title("Mean same-HKL normal angle")
    axes[1, 1].plot(evals, [row["delta_angle_deg"] for row in rows], label="rotation delta", color="#f97316")
    axes[1, 1].plot(evals, [row["radius_scale"] for row in rows], label="radius scale", color="#0f766e")
    axes[1, 1].set_title("Optimized parameters")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(evals, [row["match_score"] for row in rows], color="#475569")
    axes[1, 2].set_title("Full image/band match score guard")
    for ax in axes.ravel():
        ax.set_xlabel("function evaluation")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def reconstruct_band_points_3d(
    result: MatchResult,
    hough_rows: list[dict],
    samples_per_band: int,
) -> list[dict]:
    prepared = result.prepared
    height, width = prepared.image.shape
    hough_by_band = {int(row["band_index"]): row for row in hough_rows}
    rows_out: list[dict] = []
    for segment in prepared.line_segments:
        match = hough_by_band.get(segment.band_index)
        if match is None:
            continue
        sample_rows = np.linspace(segment.row0, segment.row1, samples_per_band, dtype=np.float32)
        sample_cols = np.linspace(segment.col0, segment.col1, samples_per_band, dtype=np.float32)
        rr = np.clip(np.round(sample_rows).astype(int), 0, height - 1)
        cc = np.clip(np.round(sample_cols).astype(int), 0, width - 1)
        keep = prepared.valid_mask[rr, cc]
        if not np.any(keep):
            continue
        sample_rows = sample_rows[keep]
        sample_cols = sample_cols[keep]
        vectors = detector_pixels_to_sphere(sample_rows, sample_cols, height, width, prepared.bundle.pc)
        vectors = vectors @ result.detector_transform.T
        vectors = result.rotation.apply(vectors).astype(np.float32)
        for sample_id, (row_px, col_px, vector) in enumerate(zip(sample_rows, sample_cols, vectors)):
            rows_out.append(
                {
                    "pattern_index": int(prepared.bundle.index),
                    "band_index": int(segment.band_index),
                    "hkl": match["hkl"],
                    "sample_id": int(sample_id),
                    "detector_row": float(row_px),
                    "detector_col": float(col_px),
                    "x": float(vector[0]),
                    "y": float(vector[1]),
                    "z": float(vector[2]),
                    "observed_theta_deg": float(match["observed_theta_deg"]),
                    "observed_rho_px": float(match["observed_rho_px"]),
                    "predicted_theta_deg": float(match["predicted_theta_deg"]),
                    "predicted_rho_px": float(match["predicted_rho_px"]),
                    "theta_error_deg": float(match["theta_error_deg"]),
                    "rho_error_px": float(match["rho_error_px"]),
                    "band_normal_angle_deg": float(match["band_normal_angle_deg"]),
                }
            )
    return rows_out


def save_reconstructed_3d_visualization(
    result: MatchResult,
    master,
    hough_rows: list[dict],
    point_rows: list[dict],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    plot_master_sphere(ax, master, lon_count, colat_count, alpha=0.36)

    if point_rows:
        points = np.asarray([[row["x"], row["y"], row["z"]] for row in point_rows], dtype=np.float32)
    else:
        points = np.empty((0, 3), dtype=np.float32)

    rows_by_band: dict[int, list[dict]] = {}
    for row in point_rows:
        rows_by_band.setdefault(int(row["band_index"]), []).append(row)
    hough_by_band = {int(row["band_index"]): row for row in hough_rows}
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(rows_by_band))))
    for color, (band_index, rows) in zip(colors, sorted(rows_by_band.items())):
        curve = np.asarray([[row["x"], row["y"], row["z"]] for row in rows], dtype=np.float32)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=3.0, alpha=0.98)
        match = hough_by_band.get(band_index)
        if match is not None:
            normal = np.asarray([match["master_normal_x"], match["master_normal_y"], match["master_normal_z"]], dtype=np.float32)
            standard = great_circle_from_normal(normal)
            ax.plot(standard[:, 0], standard[:, 1], standard[:, 2], color=color, linewidth=1.0, alpha=0.45)
            mid = curve[len(curve) // 2]
            ax.text(mid[0], mid[1], mid[2], f"{band_index}:{match['hkl']}", color="black", fontsize=8)

    title = (
        "3D reconstruction from Hough-point matched Kikuchi bands\n"
        "thick=observed H5 bands projected to master sphere, thin=same-HKL standard great circles"
    )
    set_3d_sphere_axes(ax, title, view_vectors=points if len(points) else None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def optimize_hough_points(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    families: list[HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
    seed: int,
) -> tuple[MatchResult, dict, list[dict], list[dict]]:
    height, width = prepared.image.shape
    families_by_label = family_by_label(families)
    segment_indices = [
        i
        for i, segment in enumerate(prepared.line_segments)
        if segment.band_index in fixed_hkl_by_band and fixed_hkl_by_band[segment.band_index] in families_by_label
    ]
    if not segment_indices:
        raise ValueError("No H5 bands could be assigned to HKL families")

    x0 = [0.0, 0.0, 0.0, 1.0]
    lower = [-np.radians(args.rotation_bound_deg)] * 3 + [args.radius_min]
    upper = [np.radians(args.rotation_bound_deg)] * 3 + [args.radius_max]
    if args.optimize_pc:
        x0.extend([0.0, 0.0])
        lower.extend([-args.pc_bound_px, -args.pc_bound_px])
        upper.extend([args.pc_bound_px, args.pc_bound_px])

    def build_result(params: np.ndarray, label: str = "hough-point-refined") -> MatchResult:
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
            label=label,
            score=float(score),
            rotation=rotation,
            convention_name=initial_result.convention_name,
            detector_transform=initial_result.detector_transform,
            prepared=candidate_prepared,
        )

    def evaluate_params(params: np.ndarray) -> tuple[MatchResult, dict, list[dict], np.ndarray]:
        result = build_result(params)
        _, _, state = parameters_to_state(
            params,
            initial_result.rotation,
            prepared.bundle.pc,
            height,
            width,
            args.optimize_pc,
        )
        rows = hough_point_match_rows(result, families_by_label, fixed_hkl_by_band, args)
        stats = hough_point_stats(rows)

        weights = np.asarray([max(1e-3, row["band_intensity"]) for row in rows], dtype=np.float64)
        weights /= weights.sum() + 1e-12
        rho_scale_px = args.rho_scale_px if args.rho_scale_px > 0 else max(1.0, args.rho_scale_fraction * min(height, width))
        residual_parts: list[float] = []
        for weight, row in zip(weights, rows):
            root = float(np.sqrt(weight))
            residual_parts.append(
                root * np.sqrt(args.hough_theta_weight) * row["theta_error_deg"] / max(args.theta_scale_deg, 1e-8)
            )
            residual_parts.append(root * np.sqrt(args.hough_rho_weight) * row["rho_error_px"] / rho_scale_px)
            residual_parts.append(
                root * np.sqrt(args.band_normal_weight) * row["band_normal_angle_deg"] / max(args.band_angle_scale_deg, 1e-8)
            )

        if args.radius_regularization_weight > 0:
            residual_parts.append(
                np.sqrt(args.radius_regularization_weight)
                * ((state["radius_scale"] - 1.0) / max(args.radius_regularization_sigma, 1e-8))
            )
        if args.optimize_pc and args.pc_regularization_weight > 0:
            residual_parts.append(
                np.sqrt(args.pc_regularization_weight) * state["dx_px"] / max(args.pc_regularization_sigma_px, 1e-8)
            )
            residual_parts.append(
                np.sqrt(args.pc_regularization_weight) * state["dy_px"] / max(args.pc_regularization_sigma_px, 1e-8)
            )
        if args.rotation_regularization_weight > 0:
            residual_parts.extend(
                (
                    np.sqrt(args.rotation_regularization_weight)
                    * params[:3]
                    / max(np.radians(args.rotation_regularization_sigma_deg), 1e-8)
                ).tolist()
            )

        residual_vector = np.asarray(residual_parts, dtype=np.float64)
        return result, {**state, "match_score": float(result.score), "hough_loss": float(np.sum(residual_vector**2)), **stats}, rows, residual_vector

    min_allowed_score = float(initial_result.score) - float(args.max_match_score_drop)
    rng = np.random.default_rng(seed)
    start_records: list[dict] = []
    start_candidates = [np.asarray(x0, dtype=np.float64)]
    for _ in range(max(0, args.hough_random_starts)):
        candidate = np.asarray(x0, dtype=np.float64).copy()
        candidate[:3] = rng.uniform(-np.radians(args.rotation_bound_deg), np.radians(args.rotation_bound_deg), size=3)
        candidate[3] = rng.uniform(args.radius_min, args.radius_max)
        if args.optimize_pc:
            candidate[4] = rng.uniform(-args.pc_bound_px, args.pc_bound_px)
            candidate[5] = rng.uniform(-args.pc_bound_px, args.pc_bound_px)
        start_candidates.append(candidate)

    best_start = start_candidates[0]
    best_start_objective = float("inf")
    for candidate_id, candidate in enumerate(start_candidates):
        result, metrics, _, residual_vector = evaluate_params(candidate)
        score_penalty = max(0.0, min_allowed_score - float(result.score))
        objective = float(np.sum(residual_vector**2) + args.start_score_penalty * score_penalty**2)
        start_records.append(
            {
                "candidate": int(candidate_id),
                **metrics,
                "score_penalty": float(score_penalty),
                "start_objective": objective,
            }
        )
        if objective < best_start_objective:
            best_start_objective = objective
            best_start = candidate

    trace_rows: list[dict] = []
    eval_counter = {"value": 0}

    def residuals(params: np.ndarray) -> np.ndarray:
        eval_counter["value"] += 1
        _, metrics, _, residual_vector = evaluate_params(params)
        trace_rows.append({"evaluation": int(eval_counter["value"]), **metrics})
        return residual_vector

    optimizer_result = least_squares(
        residuals,
        best_start,
        bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        max_nfev=args.max_nfev,
        xtol=args.xtol,
        ftol=args.ftol,
        gtol=args.gtol,
        x_scale="jac",
        verbose=2 if args.verbose_optimizer else 0,
    )

    selected_params = np.asarray(x0, dtype=np.float64)
    selected_reason = "fallback_initial_match_score_guard"
    candidate_pool = start_records + trace_rows
    if candidate_pool:
        feasible_rows = [row for row in candidate_pool if float(row["match_score"]) >= min_allowed_score]
        if feasible_rows:
            selected_row = min(feasible_rows, key=lambda row: (float(row["hough_loss"]), float(row["mean_band_normal_angle_deg"])))
            selected_params = trace_row_to_params(selected_row, args.optimize_pc)
            selected_reason = "best_hough_point_loss_with_match_score_guard"
        elif args.allow_score_guard_fallback:
            selected_row = max(candidate_pool, key=lambda row: float(row["match_score"]))
            selected_params = trace_row_to_params(selected_row, args.optimize_pc)
            selected_reason = "fallback_highest_match_score"

    final_result = build_result(selected_params)
    _, final_pc, final_state = parameters_to_state(
        selected_params,
        initial_result.rotation,
        prepared.bundle.pc,
        height,
        width,
        args.optimize_pc,
    )
    final_rows = hough_point_match_rows(final_result, families_by_label, fixed_hkl_by_band, args)
    final_stats = hough_point_stats(final_rows)
    final_state.update(
        {
            "selected_reason": selected_reason,
            "final_match_score": float(final_result.score),
            "minimum_allowed_match_score": float(min_allowed_score),
            "max_match_score_drop": float(args.max_match_score_drop),
            "optimizer_success": bool(optimizer_result.success),
            "optimizer_status": int(optimizer_result.status),
            "optimizer_message": str(optimizer_result.message),
            "optimizer_nfev": int(optimizer_result.nfev),
            "optimizer_cost": float(optimizer_result.cost),
            "optimizer_optimality": float(optimizer_result.optimality),
            "hough_random_starts": int(args.hough_random_starts),
            "best_start_objective": float(best_start_objective),
            "final_pc": {"pcx": final_pc[0], "pcy": final_pc[1], "pcz": final_pc[2]},
            "final_rotation_quat_xyzw": final_result.rotation.as_quat().tolist(),
            "band_count": int(len(segment_indices)),
            **{f"final_{key}": value for key, value in final_stats.items()},
        }
    )
    return final_result, final_state, trace_rows, final_rows, start_records


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

    initial_hough_rows = hough_point_match_rows(initial_result, families_by_label, fixed_hkl_by_band, args)
    initial_hough_stats = hough_point_stats(initial_hough_rows)
    initial_label_score, initial_label_angle = label_score(initial_assignments)
    initial_band_rows = band_angle_rows(initial_result, families_by_label, fixed_hkl_by_band)

    write_rows_csv(initial_hough_rows, out_dir / "02_initial_hough_point_matches.csv")
    write_rows_csv(initial_band_rows, out_dir / "03_initial_band_angle_residuals.csv")
    save_detector_label_overlay(initial_result, initial_assignments, out_dir / "01_detector_bands_initial_inferred_hkl.png")
    save_hough_match_visualization(
        prepared,
        initial_hough_rows,
        out_dir / "04_initial_hough_point_match.png",
        f"Initial Hough-point match | idx={index}",
    )

    final_result, final_state, trace_rows, final_hough_rows, start_records = optimize_hough_points(
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
    final_band_rows = band_angle_rows(final_result, families_by_label, fixed_hkl_by_band)
    point_rows = reconstruct_band_points_3d(final_result, final_hough_rows, args.samples_per_band)

    write_rows_csv(trace_rows, out_dir / "05_hough_point_optimizer_trace.csv")
    write_rows_csv(start_records, out_dir / "05a_hough_point_start_candidates.csv")
    save_trace_plot(trace_rows, out_dir / "06_hough_point_optimizer_trace.png")
    write_rows_csv(final_hough_rows, out_dir / "07_final_hough_point_matches.csv")
    write_rows_csv(final_band_rows, out_dir / "08_final_band_angle_residuals.csv")
    write_rows_csv(point_rows, out_dir / "09_reconstructed_band_points_3d.csv")
    save_hough_match_visualization(
        final_result.prepared,
        final_hough_rows,
        out_dir / "10_refined_hough_point_match.png",
        f"Refined Hough-point match | idx={index}",
    )
    save_labeled_alignment(
        final_result,
        master,
        final_assignments,
        out_dir / "11_refined_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_reconstructed_3d_visualization(
        final_result,
        master,
        final_hough_rows,
        point_rows,
        out_dir / "12_reconstructed_3d_band_points.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        final_result,
        master,
        products,
        out_dir / "13_final_spatial_from_hough_points.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    final_hough_stats = hough_point_stats(final_hough_rows)
    summary = {
        "map": map_spec.key,
        "map_label": map_spec.label,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase": phase_info,
        "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "fixed_hkl_by_band": fixed_hkl_by_band,
        "initial_match": initial_result.to_json_dict(),
        "initial_label_score": float(initial_label_score),
        "initial_label_mean_angle_deg": float(initial_label_angle),
        "initial_hough_stats": initial_hough_stats,
        "final_match": final_result.to_json_dict(),
        "final_label_score": float(final_label_score),
        "final_label_mean_angle_deg": float(final_label_angle),
        "final_hough_stats": final_hough_stats,
        "hough_improvement": {
            "mean_abs_theta_error_deg": initial_hough_stats["mean_abs_theta_error_deg"]
            - final_hough_stats["mean_abs_theta_error_deg"],
            "mean_abs_rho_error_px": initial_hough_stats["mean_abs_rho_error_px"] - final_hough_stats["mean_abs_rho_error_px"],
            "mean_band_normal_angle_deg": initial_hough_stats["mean_band_normal_angle_deg"]
            - final_hough_stats["mean_band_normal_angle_deg"],
            "loss": initial_hough_stats["loss"] - final_hough_stats["loss"],
            "match_score": float(final_result.score) - float(initial_result.score),
        },
        "refined_state": final_state,
        "hyperparameters": {
            "coarse_rotations": args.coarse_rotations,
            "refine_schedule": args.refine_schedule,
            "rotation_bound_deg": args.rotation_bound_deg,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "optimize_pc": args.optimize_pc,
            "pc_bound_px": args.pc_bound_px,
            "hough_theta_weight": args.hough_theta_weight,
            "hough_rho_weight": args.hough_rho_weight,
            "band_normal_weight": args.band_normal_weight,
            "theta_scale_deg": args.theta_scale_deg,
            "rho_scale_px": args.rho_scale_px,
            "rho_scale_fraction": args.rho_scale_fraction,
            "band_angle_scale_deg": args.band_angle_scale_deg,
            "max_match_score_drop": args.max_match_score_drop,
            "max_nfev": args.max_nfev,
            "samples_per_band": args.samples_per_band,
        },
        "outputs": {
            "detector_labels": str(out_dir / "01_detector_bands_initial_inferred_hkl.png"),
            "initial_hough_matches_csv": str(out_dir / "02_initial_hough_point_matches.csv"),
            "initial_hough_match": str(out_dir / "04_initial_hough_point_match.png"),
            "trace_csv": str(out_dir / "05_hough_point_optimizer_trace.csv"),
            "start_candidates_csv": str(out_dir / "05a_hough_point_start_candidates.csv"),
            "trace_plot": str(out_dir / "06_hough_point_optimizer_trace.png"),
            "final_hough_matches_csv": str(out_dir / "07_final_hough_point_matches.csv"),
            "reconstructed_points_csv": str(out_dir / "09_reconstructed_band_points_3d.csv"),
            "refined_hough_match": str(out_dir / "10_refined_hough_point_match.png"),
            "refined_labeled_alignment": str(out_dir / "11_refined_labeled_band_alignment.png"),
            "reconstructed_3d_band_points": str(out_dir / "12_reconstructed_3d_band_points.png"),
            "final_spatial": str(out_dir / "13_final_spatial_from_hough_points.png"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_summary_csv(summaries: list[dict], out_path: Path) -> None:
    rows = []
    for summary in summaries:
        improvement = summary["hough_improvement"]
        state = summary["refined_state"]
        rows.append(
            {
                "index": summary["index"],
                "row": summary["row"],
                "col": summary["col"],
                "phase_id": summary["phase"]["phase_id"],
                "initial_match_score": summary["initial_match"]["score"],
                "final_match_score": summary["final_match"]["score"],
                "match_score_gain": improvement["match_score"],
                "initial_mean_abs_theta_error_deg": summary["initial_hough_stats"]["mean_abs_theta_error_deg"],
                "final_mean_abs_theta_error_deg": summary["final_hough_stats"]["mean_abs_theta_error_deg"],
                "theta_error_gain_deg": improvement["mean_abs_theta_error_deg"],
                "initial_mean_abs_rho_error_px": summary["initial_hough_stats"]["mean_abs_rho_error_px"],
                "final_mean_abs_rho_error_px": summary["final_hough_stats"]["mean_abs_rho_error_px"],
                "rho_error_gain_px": improvement["mean_abs_rho_error_px"],
                "initial_mean_band_normal_angle_deg": summary["initial_hough_stats"]["mean_band_normal_angle_deg"],
                "final_mean_band_normal_angle_deg": summary["final_hough_stats"]["mean_band_normal_angle_deg"],
                "band_normal_angle_gain_deg": improvement["mean_band_normal_angle_deg"],
                "hough_loss_gain": improvement["loss"],
                "delta_angle_deg": state["delta_angle_deg"],
                "radius_scale": state["radius_scale"],
                "dx_px": state["dx_px"],
                "dy_px": state["dy_px"],
                "selected_reason": state["selected_reason"],
                "optimizer_success": state["optimizer_success"],
            }
        )
    write_rows_csv(rows, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hough-point Kikuchi band matching and 3D spatial reconstruction.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma list or Python-like ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "hough_point_spatial_reconstruction")

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

    parser.add_argument("--rotation-bound-deg", type=float, default=6.0)
    parser.add_argument("--radius-min", type=float, default=0.98)
    parser.add_argument("--radius-max", type=float, default=1.02)
    parser.add_argument("--optimize-pc", action="store_true")
    parser.add_argument("--pc-bound-px", type=float, default=4.0)
    parser.add_argument("--max-match-score-drop", type=float, default=0.03)
    parser.add_argument("--allow-score-guard-fallback", action="store_true")

    parser.add_argument("--hough-theta-weight", type=float, default=1.0)
    parser.add_argument("--hough-rho-weight", type=float, default=1.0)
    parser.add_argument("--band-normal-weight", type=float, default=0.35)
    parser.add_argument("--theta-scale-deg", type=float, default=2.5)
    parser.add_argument("--rho-scale-px", type=float, default=0.0, help="If <=0, use rho-scale-fraction * min(height,width).")
    parser.add_argument("--rho-scale-fraction", type=float, default=0.025)
    parser.add_argument("--band-angle-scale-deg", type=float, default=8.0)
    parser.add_argument("--radius-regularization-weight", type=float, default=0.05)
    parser.add_argument("--radius-regularization-sigma", type=float, default=0.02)
    parser.add_argument("--pc-regularization-weight", type=float, default=0.04)
    parser.add_argument("--pc-regularization-sigma-px", type=float, default=3.0)
    parser.add_argument("--rotation-regularization-weight", type=float, default=0.015)
    parser.add_argument("--rotation-regularization-sigma-deg", type=float, default=3.0)

    parser.add_argument("--max-nfev", type=int, default=220)
    parser.add_argument("--hough-random-starts", type=int, default=180)
    parser.add_argument("--start-score-penalty", type=float, default=200.0)
    parser.add_argument("--xtol", type=float, default=1e-7)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument("--verbose-optimizer", action="store_true")

    parser.add_argument("--samples-per-band", type=int, default=180)
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
    iterator = tqdm(indices, desc="Hough-point reconstruction") if tqdm is not None else indices
    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        summaries.append(process_one(args, map_spec, master, int(index), batch_dir))

    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    write_batch_summary_csv(summaries, batch_dir / "batch_hough_point_summary.csv")
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "10_refined_hough_point_match.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_hough_point_match.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "12_reconstructed_3d_band_points.png" for summary in summaries],
        batch_dir / "contact_sheet_reconstructed_3d_band_points.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "13_final_spatial_from_hough_points.png" for summary in summaries],
        batch_dir / "contact_sheet_final_spatial_from_hough_points.png",
        thumb_width=900,
        columns=2,
    )
    print(f"Saved Hough-point reconstruction results to: {batch_dir}")
    print(f"Batch CSV: {batch_dir / 'batch_hough_point_summary.csv'}")
    print(f"Final contact sheet: {batch_dir / 'contact_sheet_final_spatial_from_hough_points.png'}")


if __name__ == "__main__":
    main()
