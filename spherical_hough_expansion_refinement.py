from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from continuous_band_geometric_refinement import family_by_label, write_rows_csv
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
from hough_point_spatial_reconstruction import hough_theta_delta_deg
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
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    parse_refine_schedule,
    plot_master_sphere,
    save_final_spatial_visualization,
    set_3d_sphere_axes,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def line_hough_normal_angle(prepared: PreparedPattern, segment: LineSegment) -> tuple[float, float]:
    variant = prepared.line_variant
    height, width = prepared.image.shape
    rho_px = variant.rho_sign * hough_rho_to_pixels(segment.band.rho_bin, prepared.bundle.ohp_header, height, width)
    theta_deg = segment.band.theta_deg + (90.0 if variant.theta_is_line_angle else 0.0)
    return theta_deg, rho_px


def detector_plane_normal_from_hough(prepared: PreparedPattern, segment: LineSegment) -> np.ndarray:
    """Analytic spherical-Hough point for a detector line.

    Detector Hough line, using image-center coordinates:

        x cos(theta) + y sin(theta) = rho

    Backprojection through the pattern center gives a plane through the beam
    source. On the detector plane z=D_eff:

        a X + b Y + c = 0

    and its spherical-Hough point is the unit plane normal

        n_h = normalize([a, b, c / D_eff]).

    The curvature expansion coefficient is implemented through
    D_eff = D / expansion, equivalently PCz_eff = PCz / expansion.
    """
    theta_deg, rho_px = line_hough_normal_angle(prepared, segment)
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    height, width = prepared.image.shape
    pcx, pcy, pcz = prepared.bundle.pc
    center_col = (width - 1) / 2.0
    center_row = (height - 1) / 2.0
    pc_col = pcx * (width - 1)
    pc_row = pcy * (height - 1)
    detector_distance_px = pcz * height

    if prepared.line_variant.y_axis == "up":
        coeff_x = cos_t
        coeff_y = sin_t
        constant = cos_t * (pc_col - center_col) + sin_t * (center_row - pc_row) - rho_px
    else:
        coeff_x = cos_t
        coeff_y = -sin_t
        constant = cos_t * (pc_col - center_col) + sin_t * (pc_row - center_row) - rho_px

    normal = np.asarray([coeff_x, coeff_y, constant / detector_distance_px], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    return normal.astype(np.float32)


def spherical_hough_point_master(result: MatchResult, segment: LineSegment) -> np.ndarray:
    point = detector_plane_normal_from_hough(result.prepared, segment)
    point = point @ result.detector_transform.T
    point = result.rotation.apply(point).astype(np.float32)
    point /= np.linalg.norm(point) + 1e-8
    return point


def endpoint_band_normal_master(result: MatchResult, segment: LineSegment) -> np.ndarray:
    normal = band_plane_normal(result.prepared, segment)
    normal = normal @ result.detector_transform.T
    normal = result.rotation.apply(normal).astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    return normal


def closest_family_point(point: np.ndarray, family: HKLFamily) -> tuple[np.ndarray, float]:
    dots = family.normals @ point.astype(np.float32)
    index = int(np.argmax(np.abs(dots)))
    chosen = family.normals[index].astype(np.float32).copy()
    if float(np.dot(chosen, point)) < 0:
        chosen *= -1.0
    confidence = float(np.dot(chosen, point))
    angle_deg = float(np.degrees(np.arccos(np.clip(confidence, -1.0, 1.0))))
    return chosen, angle_deg


def spherical_hough_rows(
    result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
) -> list[dict]:
    rows: list[dict] = []
    for segment in result.prepared.line_segments:
        hkl = fixed_hkl_by_band.get(segment.band_index)
        if hkl not in families_by_label:
            continue
        observed = spherical_hough_point_master(result, segment)
        predicted, angle_deg = closest_family_point(observed, families_by_label[hkl])
        theta_deg, rho_px = line_hough_normal_angle(result.prepared, segment)
        angular_radius = math.degrees(math.atan2(abs(detector_plane_normal_from_hough(result.prepared, segment)[2]), 1.0))
        rows.append(
            {
                "band_index": int(segment.band_index),
                "hkl": hkl,
                "angle_deg": float(angle_deg),
                "band_intensity": float(segment.band.intensity),
                "theta_deg": float(theta_deg),
                "rho_px": float(rho_px),
                "angular_radius_deg": float(angular_radius),
                "observed_x": float(observed[0]),
                "observed_y": float(observed[1]),
                "observed_z": float(observed[2]),
                "predicted_x": float(predicted[0]),
                "predicted_y": float(predicted[1]),
                "predicted_z": float(predicted[2]),
            }
        )
    return rows


def spherical_hough_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"mean_angle_deg": float("nan"), "max_angle_deg": float("nan"), "loss": float("nan")}
    weights = np.asarray([max(1e-3, row["band_intensity"]) for row in rows], dtype=np.float64)
    weights /= weights.sum() + 1e-12
    angles = np.asarray([row["angle_deg"] for row in rows], dtype=np.float64)
    return {
        "mean_angle_deg": float(np.sum(weights * angles)),
        "max_angle_deg": float(np.max(angles)),
        "loss": float(np.sum(weights * (angles**2))),
    }


def build_result_from_state(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    rotvec: np.ndarray,
    expansion: float,
    label: str,
) -> MatchResult:
    height, width = prepared.image.shape
    expansion = float(expansion)
    pc = corrected_pc(prepared.bundle.pc, 0.0, 0.0, 1.0 / expansion, height, width)
    candidate_prepared = prepared_with_pc(prepared, pc)
    rotation = R.from_rotvec(rotvec) * initial_result.rotation
    score = score_rotation(rotation, candidate_prepared.exp_points @ initial_result.detector_transform.T, candidate_prepared, master)
    return MatchResult(
        label=label,
        score=float(score),
        rotation=rotation,
        convention_name=initial_result.convention_name,
        detector_transform=initial_result.detector_transform,
        prepared=candidate_prepared,
    )


def state_metrics(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    rotvec: np.ndarray,
    expansion: float,
) -> tuple[MatchResult, list[dict], dict]:
    result = build_result_from_state(prepared, master, initial_result, rotvec, expansion, "spherical-hough-expansion-refined")
    rows = spherical_hough_rows(result, families_by_label, fixed_hkl_by_band)
    stats = spherical_hough_stats(rows)
    stats.update(
        {
            "match_score": float(result.score),
            "expansion": float(expansion),
            "effective_radius_scale": float(1.0 / expansion),
            "delta_angle_deg": float(np.degrees(np.linalg.norm(rotvec))),
            "rotvec_x_deg": float(np.degrees(rotvec[0])),
            "rotvec_y_deg": float(np.degrees(rotvec[1])),
            "rotvec_z_deg": float(np.degrees(rotvec[2])),
        }
    )
    return result, rows, stats


def residual_vector_from_rows(rows: list[dict], args) -> np.ndarray:
    if not rows:
        return np.zeros(1, dtype=np.float64)
    weights = np.asarray([max(1e-3, row["band_intensity"]) for row in rows], dtype=np.float64)
    weights /= weights.sum() + 1e-12
    angles = np.radians(np.asarray([row["angle_deg"] for row in rows], dtype=np.float64))
    return np.sqrt(weights) * angles / max(np.radians(args.band_angle_scale_deg), 1e-8)


def save_formula_verification(
    result: MatchResult,
    fixed_hkl_by_band: dict[int, str],
    out_path: Path,
) -> list[dict]:
    rows = []
    for segment in result.prepared.line_segments:
        if segment.band_index not in fixed_hkl_by_band:
            continue
        analytic = spherical_hough_point_master(result, segment)
        endpoint = endpoint_band_normal_master(result, segment)
        dot = float(abs(np.dot(analytic, endpoint)))
        angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        local = detector_plane_normal_from_hough(result.prepared, segment)
        u = abs(float(local[2])) / max(1e-12, float(np.linalg.norm(local[:2])))
        beta = math.atan(u)
        expansion_factor = (u / beta) if beta > 1e-12 else 1.0
        rows.append(
            {
                "band_index": int(segment.band_index),
                "hkl": fixed_hkl_by_band[segment.band_index],
                "analytic_vs_endpoint_angle_deg": angle,
                "u_rho_over_d": u,
                "beta_rad": beta,
                "beta_deg": math.degrees(beta),
                "tangent_over_arc_expansion": expansion_factor,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    if rows:
        x = np.arange(len(rows))
        axes[0].bar(x, [row["analytic_vs_endpoint_angle_deg"] for row in rows], color="#2563eb")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{row['band_index']}:{row['hkl']}" for row in rows], rotation=35, ha="right")
    axes[0].set_title("Analytic spherical-Hough normal vs endpoint cross-product")
    axes[0].set_ylabel("angle difference (deg)")
    axes[0].grid(axis="y", alpha=0.25)

    u_grid = np.linspace(0.001, 1.2, 500)
    beta_grid = np.arctan(u_grid)
    expansion_grid = u_grid / beta_grid
    axes[1].plot(u_grid, np.degrees(beta_grid), label="spherical arc beta=atan(rho/D)", color="#16a34a")
    axes[1].set_xlabel("u = |rho_pc| / D")
    axes[1].set_ylabel("beta (deg)", color="#16a34a")
    axes[1].tick_params(axis="y", labelcolor="#16a34a")
    axes_r = axes[1].twinx()
    axes_r.plot(u_grid, expansion_grid, label="flat/arc expansion u/atan(u)", color="#db2777")
    axes_r.set_ylabel("curvature expansion factor", color="#db2777")
    axes_r.tick_params(axis="y", labelcolor="#db2777")
    if rows:
        axes[1].scatter([row["u_rho_over_d"] for row in rows], [row["beta_deg"] for row in rows], color="#111827", s=32)
    axes[1].set_title("Known unit-sphere curvature expansion")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def save_trace_plot(trace_rows: list[dict], out_path: Path) -> None:
    if not trace_rows:
        return
    step = np.arange(len(trace_rows))
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    axes[0, 0].plot(step, [row["mean_angle_deg"] for row in trace_rows], color="#7c3aed")
    axes[0, 0].set_title("Mean spherical-Hough point angle")
    axes[0, 1].plot(step, [row["loss"] for row in trace_rows], color="#2563eb")
    axes[0, 1].set_title("Point matching loss")
    axes[0, 2].plot(step, [row["match_score"] for row in trace_rows], color="#475569")
    axes[0, 2].set_title("Full image/band match score")
    axes[1, 0].plot(step, [row["expansion"] for row in trace_rows], color="#db2777")
    axes[1, 0].set_title("Expansion coefficient")
    axes[1, 1].plot(step, [row["delta_angle_deg"] for row in trace_rows], color="#f97316")
    axes[1, 1].set_title("Rotation delta (deg)")
    axes[1, 2].plot(step, [row["lr_rotation_deg"] for row in trace_rows], label="rotation lr", color="#16a34a")
    axes[1, 2].plot(step, [row["lr_expansion"] for row in trace_rows], label="expansion lr", color="#0f766e")
    axes[1, 2].set_title("Annealed learning rates")
    axes[1, 2].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("alternating phase")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sphere_wire(ax, alpha: float = 0.16) -> None:
    u = np.linspace(0, 2 * np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#94a3b8", linewidth=0.45, alpha=alpha)


def save_spherical_hough_points(rows: list[dict], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(12.8, 6.4))
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122)
    sphere_wire(ax0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(rows))))
    for color, row in zip(colors, rows):
        observed = np.asarray([row["observed_x"], row["observed_y"], row["observed_z"]], dtype=np.float32)
        predicted = np.asarray([row["predicted_x"], row["predicted_y"], row["predicted_z"]], dtype=np.float32)
        ax0.scatter(observed[0], observed[1], observed[2], color=color, s=50, marker="o", edgecolor="black", linewidth=0.4)
        ax0.scatter(predicted[0], predicted[1], predicted[2], color=color, s=58, marker="x", linewidth=1.2)
        ax0.plot([observed[0], predicted[0]], [observed[1], predicted[1]], [observed[2], predicted[2]], color=color, linewidth=1.0)
        ax0.text(observed[0], observed[1], observed[2], f"{row['band_index']}:{row['hkl']}", fontsize=8)

        obs_lon = math.degrees(math.atan2(observed[1], observed[0]))
        obs_lat = math.degrees(math.asin(np.clip(observed[2], -1.0, 1.0)))
        pred_lon = math.degrees(math.atan2(predicted[1], predicted[0]))
        pred_lat = math.degrees(math.asin(np.clip(predicted[2], -1.0, 1.0)))
        ax1.plot([obs_lon, pred_lon], [obs_lat, pred_lat], color=color, linewidth=1.0)
        ax1.scatter(obs_lon, obs_lat, color=color, s=48, marker="o", edgecolor="black", linewidth=0.4)
        ax1.scatter(pred_lon, pred_lat, color=color, s=56, marker="x", linewidth=1.2)
        ax1.text(obs_lon, obs_lat, f"{row['band_index']}:{row['hkl']}", fontsize=8)

    set_3d_sphere_axes(ax0, "Spherical Hough points on dual sphere")
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel("dual-sphere longitude (deg)")
    ax1.set_ylabel("dual-sphere latitude (deg)")
    ax1.set_title("Equirectangular dual-sphere points")
    ax1.grid(alpha=0.25)
    fig.suptitle(title, y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def reconstruct_band_points(result: MatchResult, rows: list[dict], samples_per_band: int) -> list[dict]:
    prepared = result.prepared
    height, width = prepared.image.shape
    hkl_by_band = {int(row["band_index"]): row["hkl"] for row in rows}
    out: list[dict] = []
    for segment in prepared.line_segments:
        hkl = hkl_by_band.get(segment.band_index)
        if hkl is None:
            continue
        sample_rows = np.linspace(segment.row0, segment.row1, samples_per_band, dtype=np.float32)
        sample_cols = np.linspace(segment.col0, segment.col1, samples_per_band, dtype=np.float32)
        rr = np.clip(np.round(sample_rows).astype(int), 0, height - 1)
        cc = np.clip(np.round(sample_cols).astype(int), 0, width - 1)
        keep = prepared.valid_mask[rr, cc]
        sample_rows = sample_rows[keep]
        sample_cols = sample_cols[keep]
        vectors = detector_pixels_to_sphere(sample_rows, sample_cols, height, width, prepared.bundle.pc)
        vectors = vectors @ result.detector_transform.T
        vectors = result.rotation.apply(vectors).astype(np.float32)
        for sample_id, (row_px, col_px, vector) in enumerate(zip(sample_rows, sample_cols, vectors)):
            out.append(
                {
                    "pattern_index": int(prepared.bundle.index),
                    "band_index": int(segment.band_index),
                    "hkl": hkl,
                    "sample_id": int(sample_id),
                    "detector_row": float(row_px),
                    "detector_col": float(col_px),
                    "x": float(vector[0]),
                    "y": float(vector[1]),
                    "z": float(vector[2]),
                }
            )
    return out


def save_reconstructed_bands(result: MatchResult, master, rows: list[dict], point_rows: list[dict], out_path: Path, lon_count: int, colat_count: int) -> None:
    fig = plt.figure(figsize=(12.6, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    plot_master_sphere(ax, master, lon_count, colat_count, alpha=0.34)
    rows_by_band: dict[int, list[dict]] = {}
    for row in point_rows:
        rows_by_band.setdefault(int(row["band_index"]), []).append(row)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(rows_by_band))))
    for color, (band_index, items) in zip(colors, sorted(rows_by_band.items())):
        curve = np.asarray([[item["x"], item["y"], item["z"]] for item in items], dtype=np.float32)
        if len(curve) < 2:
            continue
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=3.0, alpha=0.98)
        mid = curve[len(curve) // 2]
        hkl = items[0]["hkl"]
        ax.text(mid[0], mid[1], mid[2], f"{band_index}:{hkl}", fontsize=8)
    points = np.asarray([[row["x"], row["y"], row["z"]] for row in point_rows], dtype=np.float32) if point_rows else None
    set_3d_sphere_axes(ax, "3D band reconstruction after spherical-Hough expansion refinement", view_vectors=points)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def alternating_expansion_refine(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
) -> tuple[MatchResult, list[dict], list[dict], dict]:
    rotvec = np.zeros(3, dtype=np.float64)
    expansion = 1.0
    trace_rows: list[dict] = []
    min_allowed_score = float(initial_result.score) - float(args.max_match_score_drop)
    rng = np.random.default_rng(args.seed + int(prepared.bundle.index))
    selected_result, selected_rows, selected_stats = state_metrics(
        prepared, master, initial_result, families_by_label, fixed_hkl_by_band, rotvec, expansion
    )
    selected_loss = selected_stats["loss"]

    def evaluate(rotvec_value: np.ndarray, expansion_value: float):
        return state_metrics(prepared, master, initial_result, families_by_label, fixed_hkl_by_band, rotvec_value, expansion_value)

    def objective(stats: dict) -> float:
        score_drop = max(0.0, min_allowed_score - float(stats["match_score"]))
        return float(stats["loss"]) + args.score_penalty_weight * (score_drop / max(args.score_penalty_scale, 1e-8)) ** 2

    for iteration in range(args.iterations):
        lr_rot = args.lr_rotation_deg * (args.lr_decay**iteration)
        lr_exp = args.lr_expansion * (args.lr_decay**iteration)

        rotation_candidates = [rotvec]
        basis = np.eye(3, dtype=np.float64)
        for axis in basis:
            rotation_candidates.append(rotvec + np.radians(lr_rot) * axis)
            rotation_candidates.append(rotvec - np.radians(lr_rot) * axis)
        for _ in range(max(0, args.rotation_candidates)):
            step = rng.uniform(-np.radians(lr_rot), np.radians(lr_rot), size=3)
            rotation_candidates.append(rotvec + step)
        best_rotvec = rotvec
        best_objective = float("inf")
        for candidate in rotation_candidates:
            candidate = np.clip(candidate, -np.radians(args.rotation_bound_deg), np.radians(args.rotation_bound_deg))
            _, _, stats = evaluate(candidate, expansion)
            value = objective(stats)
            if value < best_objective:
                best_objective = value
                best_rotvec = candidate
        rotvec = best_rotvec

        def rotation_residual(delta_rot: np.ndarray) -> np.ndarray:
            _, rows, _ = evaluate(delta_rot, expansion)
            return residual_vector_from_rows(rows, args)

        lower_rot = np.maximum(rotvec - np.radians(lr_rot), -np.radians(args.rotation_bound_deg))
        upper_rot = np.minimum(rotvec + np.radians(lr_rot), np.radians(args.rotation_bound_deg))
        opt_rot = least_squares(
            rotation_residual,
            rotvec,
            bounds=(lower_rot, upper_rot),
            max_nfev=args.phase_max_nfev,
            xtol=args.xtol,
            ftol=args.ftol,
            gtol=args.gtol,
            x_scale="jac",
        )
        rotvec = opt_rot.x
        result, rows, stats = evaluate(rotvec, expansion)
        stats.update(
            {
                "iteration": iteration,
                "phase": "match_rotation",
                "lr_rotation_deg": lr_rot,
                "lr_expansion": lr_exp,
                "candidate_objective": best_objective,
            }
        )
        trace_rows.append(stats)
        if stats["match_score"] >= min_allowed_score and stats["loss"] <= selected_loss:
            selected_result, selected_rows, selected_stats = result, rows, stats
            selected_loss = stats["loss"]

        expansion_candidates = np.linspace(
            max(args.expansion_min, expansion - lr_exp),
            min(args.expansion_max, expansion + lr_exp),
            max(3, args.expansion_candidates),
        )
        best_expansion = expansion
        best_objective = float("inf")
        for candidate in expansion_candidates:
            _, _, stats = evaluate(rotvec, float(candidate))
            value = objective(stats)
            if value < best_objective:
                best_objective = value
                best_expansion = float(candidate)
        expansion = best_expansion

        def expansion_residual(value: np.ndarray) -> np.ndarray:
            _, rows, _ = evaluate(rotvec, float(value[0]))
            return residual_vector_from_rows(rows, args)

        lower_exp = max(args.expansion_min, expansion - lr_exp)
        upper_exp = min(args.expansion_max, expansion + lr_exp)
        opt_exp = least_squares(
            expansion_residual,
            np.asarray([expansion], dtype=np.float64),
            bounds=(np.asarray([lower_exp]), np.asarray([upper_exp])),
            max_nfev=args.phase_max_nfev,
            xtol=args.xtol,
            ftol=args.ftol,
            gtol=args.gtol,
            x_scale="jac",
        )
        expansion = float(opt_exp.x[0])
        result, rows, stats = evaluate(rotvec, expansion)
        stats.update(
            {
                "iteration": iteration,
                "phase": "expand_contract",
                "lr_rotation_deg": lr_rot,
                "lr_expansion": lr_exp,
                "candidate_objective": best_objective,
            }
        )
        trace_rows.append(stats)
        if stats["match_score"] >= min_allowed_score and stats["loss"] <= selected_loss:
            selected_result, selected_rows, selected_stats = result, rows, stats
            selected_loss = stats["loss"]

        if lr_rot < args.min_lr_rotation_deg and lr_exp < args.min_lr_expansion:
            break

    final_state = {
        "selected_reason": "best_spherical_hough_loss_with_match_score_guard",
        "minimum_allowed_match_score": min_allowed_score,
        "max_match_score_drop": float(args.max_match_score_drop),
        **selected_stats,
    }
    return selected_result, selected_rows, trace_rows, final_state


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
    initial_rows = spherical_hough_rows(initial_result, families_by_label, fixed_hkl_by_band)
    initial_stats = spherical_hough_stats(initial_rows)
    initial_label_score, initial_label_angle = label_score(initial_assignments)

    verification_rows = save_formula_verification(initial_result, fixed_hkl_by_band, out_dir / "00_formula_verification.png")
    write_rows_csv(verification_rows, out_dir / "00_formula_verification.csv")
    write_rows_csv(initial_rows, out_dir / "01_initial_spherical_hough_points.csv")
    save_detector_label_overlay(initial_result, initial_assignments, out_dir / "01_detector_bands_initial_inferred_hkl.png")
    save_spherical_hough_points(initial_rows, out_dir / "02_initial_spherical_hough_points.png", f"Initial spherical-Hough points | idx={index}")

    final_result, final_rows, trace_rows, final_state = alternating_expansion_refine(
        prepared,
        master,
        initial_result,
        families_by_label,
        fixed_hkl_by_band,
        args,
    )
    final_assignments = assign_labels(final_result, families, fixed_hkl_by_band=fixed_hkl_by_band)
    final_label_score, final_label_angle = label_score(final_assignments)
    final_stats = spherical_hough_stats(final_rows)
    point_rows = reconstruct_band_points(final_result, final_rows, args.samples_per_band)

    write_rows_csv(trace_rows, out_dir / "03_alternating_expansion_trace.csv")
    save_trace_plot(trace_rows, out_dir / "04_alternating_expansion_trace.png")
    write_rows_csv(final_rows, out_dir / "05_final_spherical_hough_points.csv")
    save_spherical_hough_points(final_rows, out_dir / "06_final_spherical_hough_points.png", f"Final spherical-Hough points | idx={index}")
    save_labeled_alignment(
        final_result,
        master,
        final_assignments,
        out_dir / "07_final_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    write_rows_csv(point_rows, out_dir / "08_reconstructed_band_points_3d.csv")
    save_reconstructed_bands(
        final_result,
        master,
        final_rows,
        point_rows,
        out_dir / "09_reconstructed_band_points_3d.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        final_result,
        master,
        products,
        out_dir / "10_final_spatial_after_expansion.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

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
        "initial_spherical_hough_stats": initial_stats,
        "final_match": final_result.to_json_dict(),
        "final_label_score": float(final_label_score),
        "final_label_mean_angle_deg": float(final_label_angle),
        "final_spherical_hough_stats": final_stats,
        "improvement": {
            "mean_spherical_hough_angle_deg": initial_stats["mean_angle_deg"] - final_stats["mean_angle_deg"],
            "max_spherical_hough_angle_deg": initial_stats["max_angle_deg"] - final_stats["max_angle_deg"],
            "spherical_hough_loss": initial_stats["loss"] - final_stats["loss"],
            "match_score": float(final_result.score) - float(initial_result.score),
            "label_mean_angle_deg": float(initial_label_angle) - float(final_label_angle),
        },
        "final_state": final_state,
        "hyperparameters": vars(args) | {"data_dir": str(args.data_dir), "h5": str(args.h5), "master_h5": str(args.master_h5)},
        "outputs": {
            "formula_verification": str(out_dir / "00_formula_verification.png"),
            "initial_spherical_hough_points": str(out_dir / "02_initial_spherical_hough_points.png"),
            "trace_plot": str(out_dir / "04_alternating_expansion_trace.png"),
            "final_spherical_hough_points": str(out_dir / "06_final_spherical_hough_points.png"),
            "final_labeled_band_alignment": str(out_dir / "07_final_labeled_band_alignment.png"),
            "reconstructed_points_csv": str(out_dir / "08_reconstructed_band_points_3d.csv"),
            "reconstructed_points": str(out_dir / "09_reconstructed_band_points_3d.png"),
            "final_spatial": str(out_dir / "10_final_spatial_after_expansion.png"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_csv(summaries: list[dict], out_path: Path) -> None:
    rows = []
    for summary in summaries:
        state = summary["final_state"]
        improvement = summary["improvement"]
        rows.append(
            {
                "index": summary["index"],
                "row": summary["row"],
                "col": summary["col"],
                "phase_id": summary["phase"]["phase_id"],
                "initial_match_score": summary["initial_match"]["score"],
                "final_match_score": summary["final_match"]["score"],
                "match_score_gain": improvement["match_score"],
                "initial_mean_spherical_hough_angle_deg": summary["initial_spherical_hough_stats"]["mean_angle_deg"],
                "final_mean_spherical_hough_angle_deg": summary["final_spherical_hough_stats"]["mean_angle_deg"],
                "spherical_hough_angle_gain_deg": improvement["mean_spherical_hough_angle_deg"],
                "initial_label_mean_angle_deg": summary["initial_label_mean_angle_deg"],
                "final_label_mean_angle_deg": summary["final_label_mean_angle_deg"],
                "label_angle_gain_deg": improvement["label_mean_angle_deg"],
                "expansion": state["expansion"],
                "effective_radius_scale": state["effective_radius_scale"],
                "delta_angle_deg": state["delta_angle_deg"],
                "selected_reason": state["selected_reason"],
            }
        )
    write_rows_csv(rows, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alternating spherical-Hough expansion refinement for EBSD Kikuchi bands.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "spherical_hough_expansion_refinement")

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

    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--lr-rotation-deg", type=float, default=2.0)
    parser.add_argument("--lr-expansion", type=float, default=0.018)
    parser.add_argument("--lr-decay", type=float, default=0.72)
    parser.add_argument("--min-lr-rotation-deg", type=float, default=0.05)
    parser.add_argument("--min-lr-expansion", type=float, default=0.0005)
    parser.add_argument("--rotation-bound-deg", type=float, default=6.0)
    parser.add_argument("--expansion-min", type=float, default=0.94)
    parser.add_argument("--expansion-max", type=float, default=1.06)
    parser.add_argument("--band-angle-scale-deg", type=float, default=7.0)
    parser.add_argument("--phase-max-nfev", type=int, default=70)
    parser.add_argument("--rotation-candidates", type=int, default=80)
    parser.add_argument("--expansion-candidates", type=int, default=17)
    parser.add_argument("--max-match-score-drop", type=float, default=0.03)
    parser.add_argument("--score-penalty-weight", type=float, default=20.0)
    parser.add_argument("--score-penalty-scale", type=float, default=0.01)
    parser.add_argument("--xtol", type=float, default=1e-7)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument("--gtol", type=float, default=1e-7)

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
    args.master_h5 = master_h5
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
    iterator = tqdm(indices, desc="spherical-Hough expansion") if tqdm is not None else indices
    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        summaries.append(process_one(args, map_spec, master, int(index), batch_dir))

    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    write_batch_csv(summaries, batch_dir / "batch_spherical_hough_expansion_summary.csv")
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "00_formula_verification.png" for summary in summaries],
        batch_dir / "contact_sheet_formula_verification.png",
        thumb_width=850,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "06_final_spherical_hough_points.png" for summary in summaries],
        batch_dir / "contact_sheet_final_spherical_hough_points.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "09_reconstructed_band_points_3d.png" for summary in summaries],
        batch_dir / "contact_sheet_reconstructed_band_points_3d.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "10_final_spatial_after_expansion.png" for summary in summaries],
        batch_dir / "contact_sheet_final_spatial_after_expansion.png",
        thumb_width=900,
        columns=2,
    )
    print(f"Saved spherical-Hough expansion results to: {batch_dir}")
    print(f"Batch CSV: {batch_dir / 'batch_spherical_hough_expansion_summary.csv'}")


if __name__ == "__main__":
    main()
