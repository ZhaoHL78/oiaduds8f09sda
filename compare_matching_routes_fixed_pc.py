from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from scipy.optimize import Bounds, minimize
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
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    resolve_master_path,
    score_rotation,
    zscore_vector,
)
from spherical_radon_graph_pipeline import (
    build_experimental_transfer_points,
    build_peak_descriptors,
    build_software_line_peak_descriptors,
    descriptor_cost_matrix,
    evaluate_candidates_with_ot,
    fibonacci_sphere,
    final_matches_for_rotation,
    generate_triangle_candidates,
    greedy_peak_pick,
    merge_software_and_radon_peaks,
    parse_float_list,
    partial_optimal_transport,
    peak_lon_colat,
    peak_weights,
    sample_software_kikuchi_lines_on_sphere,
    save_peak_descriptors,
    save_peak_graphs,
    save_radon_maps,
    save_transport_matching,
    software_band_normals_for_pc,
    spherical_radon_transform,
    transport_edge_loss,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    parse_refine_schedule,
    plot_matched_patch_on_master,
    save_final_spatial_visualization,
)
from labeled_band_radius_refinement import read_phase_hkl_families
from continuous_band_geometric_refinement import write_rows_csv


IDENTITY_TRANSFORM = np.eye(3, dtype=np.float32)


def fixed_pc_orientation_refinement(
    initial_rotation: R,
    graph_matches,
    exp_peaks,
    std_peaks,
    prepared: PreparedPattern,
    master: MasterSphere,
    args,
) -> tuple[MatchResult, dict, list[dict]]:
    exp_by_id = {peak.peak_id: peak for peak in exp_peaks}
    std_by_id = {peak.peak_id: peak for peak in std_peaks}
    exp_normals = []
    std_normals = []
    weights = []
    for match in graph_matches:
        exp_normal = exp_by_id[match.exp_peak_id].normal.astype(np.float64)
        std_normal = std_by_id[match.std_peak_id].normal.astype(np.float64)
        if float(np.dot(initial_rotation.apply(exp_normal), std_normal)) < 0.0:
            std_normal = -std_normal
        exp_normals.append(exp_normal)
        std_normals.append(std_normal)
        weights.append(max(1e-5, float(match.mass)))
    exp_normals = np.asarray(exp_normals, dtype=np.float64)
    std_normals = np.asarray(std_normals, dtype=np.float64)
    match_weights = np.asarray(weights, dtype=np.float64)
    if len(match_weights):
        match_weights /= match_weights.sum() + 1e-12
    trace: list[dict] = []
    eval_count = 0

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count
        rotation = R.from_rotvec(params) * initial_rotation
        image_score = score_rotation(rotation, prepared.exp_points, prepared, master)
        peak_mean_angle = float("nan")
        peak_loss = 0.0
        if len(exp_normals):
            rotated = rotation.apply(exp_normals)
            dots = np.sum(rotated * std_normals, axis=1)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            peak_mean_angle = float(np.degrees(np.sum(match_weights * angles)))
            peak_loss = float(np.sum(match_weights * (angles / max(np.radians(args.refine_peak_angle_scale_deg), 1e-8)) ** 2))
        rot_regularization = (np.linalg.norm(params) / max(np.radians(args.rotation_regularization_deg), 1e-8)) ** 2
        loss = (
            -args.refine_image_score_weight * image_score
            + args.refine_peak_weight * peak_loss
            + args.refine_rotation_regularization_weight * rot_regularization
        )
        trace.append(
            {
                "evaluation": eval_count,
                "loss": float(loss),
                "image_score": float(image_score),
                "peak_mean_angle_deg": float(peak_mean_angle),
                "delta_angle_deg": float(np.degrees(np.linalg.norm(params))),
                "pcx": float(prepared.bundle.pc[0]),
                "pcy": float(prepared.bundle.pc[1]),
                "pcz": float(prepared.bundle.pc[2]),
            }
        )
        eval_count += 1
        return float(loss)

    bound = np.radians(args.refine_rotation_bound_deg)
    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=Bounds([-bound, -bound, -bound], [bound, bound, bound]),
        options={"maxiter": args.refine_maxiter, "xtol": args.refine_xtol, "ftol": args.refine_ftol, "disp": False},
    )
    rotation = R.from_rotvec(result.x) * initial_rotation
    final_score = score_rotation(rotation, prepared.exp_points, prepared, master)
    match_result = MatchResult(
        label="spherical-radon-graph-fixed-pc",
        score=float(final_score),
        rotation=rotation,
        convention_name="graph_SO3_identity_detector_fixed_pc",
        detector_transform=IDENTITY_TRANSFORM,
        prepared=prepared,
    )
    summary = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "optimizer_nfev": int(result.nfev),
        "delta_angle_deg": float(np.degrees(np.linalg.norm(result.x))),
        "pcx": float(prepared.bundle.pc[0]),
        "pcy": float(prepared.bundle.pc[1]),
        "pcz": float(prepared.bundle.pc[2]),
        "final_image_score": float(final_score),
    }
    return match_result, summary, trace


def route_angle_deg(a: MatchResult, b: MatchResult) -> float:
    relative = b.rotation * a.rotation.inv()
    return float(np.degrees(relative.magnitude()))


def save_route_comparison(
    old_result: MatchResult,
    graph_result: MatchResult,
    master: MasterSphere,
    products: dict[str, np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    fig = plt.figure(figsize=(17.0, 8.2))
    ax0 = fig.add_subplot(121, projection="3d")
    plot_matched_patch_on_master(
        ax0,
        old_result,
        master,
        products["raw_percentile"],
        f"Old weighted image/band route\nscore={old_result.score:.4f}, convention={old_result.convention_name}",
        lon_count,
        colat_count,
    )
    ax1 = fig.add_subplot(122, projection="3d")
    plot_matched_patch_on_master(
        ax1,
        graph_result,
        master,
        products["raw_percentile"],
        f"New peak-graph route, fixed PC\nscore={graph_result.score:.4f}",
        lon_count,
        colat_count,
    )
    fig.suptitle("Fixed-PC route comparison on the same Kikuchi master sphere", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_bar(old_result: MatchResult, graph_result: MatchResult, angle_deg: float, graph_match_count: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    names = ["weighted\nimage/band", "spherical Radon\npeak graph"]
    scores = [old_result.score, graph_result.score]
    axes[0].bar(names, scores, color=["#2563eb", "#f97316"])
    axes[0].set_ylabel("fixed-PC image/band sphere score")
    axes[0].set_title("Final score comparison")
    axes[0].grid(axis="y", alpha=0.25)
    for x, y in enumerate(scores):
        axes[0].text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=10)

    metrics = [angle_deg, graph_match_count]
    metric_names = ["orientation\ndifference deg", "graph OT\nmatches"]
    axes[1].bar(metric_names, metrics, color=["#64748b", "#16a34a"])
    axes[1].set_title("Route difference")
    axes[1].grid(axis="y", alpha=0.25)
    for x, y in enumerate(metrics):
        axes[1].text(x, y, f"{y:.2f}" if x == 0 else f"{int(y)}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def great_circle_from_normal(normal: np.ndarray, samples: int = 360) -> np.ndarray:
    normal = normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)
    t = np.linspace(0.0, 2.0 * np.pi, samples, dtype=np.float32)
    return np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :]


def save_hough_point_explanation(
    prepared: PreparedPattern,
    exp_radon,
    software_peaks,
    out_path: Path,
) -> None:
    software_normals = np.asarray([peak.normal for peak in software_peaks], dtype=np.float32)
    fig = plt.figure(figsize=(17.0, 9.2))
    ax0 = fig.add_subplot(221)
    ax0.imshow(detector_raw_display(prepared), cmap="gray", vmin=0.0, vmax=1.0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(prepared.line_segments)))
    for color, segment in zip(colors, prepared.line_segments):
        ax0.plot([segment.col0, segment.col1], [segment.row0, segment.row1], color=color, linewidth=2.0)
        ax0.text(0.5 * (segment.col0 + segment.col1), 0.5 * (segment.row0 + segment.row1), str(segment.band_index), color="white", fontsize=8)
    ax0.set_title("Detector H5/OHP Kikuchi bands: 8 lines")
    ax0.axis("off")

    ax1 = fig.add_subplot(222, projection="3d")
    for color, normal, peak in zip(colors, software_normals, software_peaks):
        circle = great_circle_from_normal(normal)
        ax1.plot(circle[:, 0], circle[:, 1], circle[:, 2], color=color, linewidth=1.2, alpha=0.9)
        ax1.scatter([normal[0]], [normal[1]], [normal[2]], color=color, edgecolor="black", s=70)
        ax1.text(normal[0], normal[1], normal[2], f"B{peak.software_band_id}", fontsize=8)
    ax1.set_title("Each spherical great circle has one plane-normal Hough point")
    ax1.set_box_aspect((1, 1, 1))
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(223)
    lon, colat = peak_lon_colat(software_normals)
    ax2.scatter(lon, colat, c=colors, edgecolors="black", s=110)
    for peak, x, y in zip(software_peaks, lon, colat):
        ax2.text(x, y, f"B{peak.software_band_id}", ha="center", va="center", fontsize=8)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(90, 0)
    ax2.set_xlabel("normal longitude (deg)")
    ax2.set_ylabel("normal colatitude (deg)")
    ax2.set_title("Actual software-band Hough points: 8 points")
    ax2.grid(alpha=0.22)

    ax3 = fig.add_subplot(224)
    grid_lon, grid_colat = peak_lon_colat(exp_radon.normals)
    sc = ax3.scatter(grid_lon, grid_colat, c=exp_radon.best_scores, s=6, cmap="viridis", linewidths=0)
    ax3.scatter(lon, colat, facecolors="none", edgecolors="red", s=120, linewidths=1.6, label="8 software-line normals")
    ax3.set_xlim(-180, 180)
    ax3.set_ylim(90, 0)
    ax3.set_xlabel("candidate normal longitude (deg)")
    ax3.set_ylabel("candidate normal colatitude (deg)")
    ax3.set_title("Experimental spherical Hough response: many sampled query normals")
    ax3.legend(fontsize=8)
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.03)
    fig.suptitle("Why 8 Kikuchi bands are 8 points, but the Hough/Radon response contains many dots", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare old weighted matching and new spherical Radon graph matching with fixed H5 PC.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "fixed_pc_route_comparison")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=6500)
    parser.add_argument("--line-variant", default="auto")
    parser.add_argument("--coarse-rotations", type=int, default=450)
    parser.add_argument("--refine-schedule", default="8:180,3:240,1:240")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sphere-lon-count", type=int, default=420)
    parser.add_argument("--sphere-colat-count", type=int, default=210)
    parser.add_argument("--radon-scales-deg", default="1.4,2.4,3.8")
    parser.add_argument("--radon-kernel", choices=["gaussian", "profiled"], default="gaussian")
    parser.add_argument("--radon-side-lobe-offset-factor", type=float, default=1.65)
    parser.add_argument("--radon-side-lobe-sigma-factor", type=float, default=0.55)
    parser.add_argument("--radon-side-lobe-weight", type=float, default=0.55)
    parser.add_argument("--normal-count", type=int, default=2200)
    parser.add_argument("--master-sample-count", type=int, default=8000)
    parser.add_argument("--experiment-sample-count", type=int, default=11000)
    parser.add_argument("--radon-chunk-size", type=int, default=192)
    parser.add_argument("--experimental-transfer-source", choices=["h5_lines", "image_band", "h5_raster", "combined"], default="h5_lines")
    parser.add_argument("--h5-line-samples-per-band", type=int, default=420)
    parser.add_argument("--h5-line-offset-count", type=int, default=5)
    parser.add_argument("--h5-line-width-scale", type=float, default=1.0)
    parser.add_argument("--h5-line-min-width-px", type=float, default=1.5)
    parser.add_argument("--experimental-peak-source", choices=["software_lines", "radon", "software_lines_plus_radon"], default="software_lines_plus_radon")
    parser.add_argument("--peak-count", type=int, default=24)
    parser.add_argument("--peak-min-separation-deg", type=float, default=5.5)
    parser.add_argument("--peak-min-score-quantile", type=float, default=0.68)
    parser.add_argument("--profile-width-deg", type=float, default=8.0)
    parser.add_argument("--profile-bins", type=int, default=17)
    parser.add_argument("--triangle-top-peaks", type=int, default=10)
    parser.add_argument("--max-triangle-pairs", type=int, default=120)
    parser.add_argument("--triangle-rms-max-deg", type=float, default=10.0)
    parser.add_argument("--triplet-residual-max-deg", type=float, default=8.0)
    parser.add_argument("--max-orientation-candidates", type=int, default=180)
    parser.add_argument("--ot-candidate-count", type=int, default=40)
    parser.add_argument("--partial-transport-mass", type=float, default=0.80)
    parser.add_argument("--min-transport-mass", type=float, default=0.006)
    parser.add_argument("--ot-match-max-angle-deg", type=float, default=24.0)
    parser.add_argument("--ot-angle-weight", type=float, default=1.0)
    parser.add_argument("--ot-angle-scale-deg", type=float, default=7.5)
    parser.add_argument("--ot-strength-weight", type=float, default=0.25)
    parser.add_argument("--ot-bandwidth-weight", type=float, default=0.20)
    parser.add_argument("--ot-profile-weight", type=float, default=0.45)
    parser.add_argument("--ot-asymmetry-weight", type=float, default=0.12)
    parser.add_argument("--ot-edge-weight", type=float, default=0.12)
    parser.add_argument("--candidate-image-score-weight", type=float, default=8.0)
    parser.add_argument("--candidate-match-bonus", type=float, default=0.02)
    parser.add_argument("--hkl-assign-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--peak-band-assign-max-angle-deg", type=float, default=12.0)
    parser.add_argument("--refine-rotation-bound-deg", type=float, default=5.0)
    parser.add_argument("--refine-maxiter", type=int, default=70)
    parser.add_argument("--refine-xtol", type=float, default=3e-4)
    parser.add_argument("--refine-ftol", type=float, default=3e-4)
    parser.add_argument("--refine-image-score-weight", type=float, default=1.0)
    parser.add_argument("--refine-peak-weight", type=float, default=0.08)
    parser.add_argument("--refine-peak-angle-scale-deg", type=float, default=8.0)
    parser.add_argument("--refine-rotation-regularization-weight", type=float, default=0.01)
    parser.add_argument("--rotation-regularization-deg", type=float, default=5.0)
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
        label="fixed-pc-weighted",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=args.line_variant,
    )

    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    print("Route A: weighted image/band matching, fixed H5 PC")
    old_result = match_to_master(
        prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=parse_refine_schedule(args.refine_schedule),
        random_seed=args.seed,
    )

    print("Route B: spherical Radon graph matching, fixed H5 PC")
    phase_id = int(bundle.ang_record.get("Phase", 1))
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)
    exp_vectors, exp_values, exp_transfer_band_ids, transfer_info = build_experimental_transfer_points(prepared, args)
    normal_grid = fibonacci_sphere(args.normal_count, hemisphere=True)
    scales = parse_float_list(args.radon_scales_deg)
    master_vectors = fibonacci_sphere(args.master_sample_count, hemisphere=False)
    master_values = master.sample_band(master_vectors)

    exp_radon = spherical_radon_transform(
        exp_vectors,
        exp_values,
        normal_grid,
        scales,
        chunk_size=args.radon_chunk_size,
        desc="experimental spherical Radon",
        kernel=args.radon_kernel,
        side_lobe_offset_factor=args.radon_side_lobe_offset_factor,
        side_lobe_sigma_factor=args.radon_side_lobe_sigma_factor,
        side_lobe_weight=args.radon_side_lobe_weight,
    )
    std_radon = spherical_radon_transform(
        master_vectors,
        master_values,
        normal_grid,
        scales,
        chunk_size=args.radon_chunk_size,
        desc="master spherical Radon",
        kernel=args.radon_kernel,
        side_lobe_offset_factor=args.radon_side_lobe_offset_factor,
        side_lobe_sigma_factor=args.radon_side_lobe_sigma_factor,
        side_lobe_weight=args.radon_side_lobe_weight,
    )
    exp_peak_indices = greedy_peak_pick(
        exp_radon,
        peak_count=args.peak_count,
        min_separation_deg=args.peak_min_separation_deg,
        min_score_quantile=args.peak_min_score_quantile,
    )
    std_peak_indices = greedy_peak_pick(
        std_radon,
        peak_count=args.peak_count + 8,
        min_separation_deg=args.peak_min_separation_deg,
        min_score_quantile=args.peak_min_score_quantile,
    )
    exp_radon_peaks = build_peak_descriptors("experimental_radon", exp_radon, exp_peak_indices, exp_vectors, exp_values, args)
    software_peaks = build_software_line_peak_descriptors(prepared, exp_vectors, exp_values, args)
    if args.experimental_peak_source == "software_lines":
        exp_peaks = software_peaks
    elif args.experimental_peak_source == "software_lines_plus_radon":
        exp_peaks = merge_software_and_radon_peaks(software_peaks, exp_radon_peaks, args.peak_count, args.peak_min_separation_deg)
    else:
        exp_peaks = exp_radon_peaks
    std_peaks = build_peak_descriptors("master", std_radon, std_peak_indices, master_vectors, master_values, args, families)

    save_hough_point_explanation(prepared, exp_radon, software_peaks, out_dir / "01_hough_line_to_sphere_point_explanation.png")
    save_radon_maps(exp_radon, std_radon, exp_peaks, std_peaks, out_dir / "02_multiscale_spherical_radon_and_peaks.png")
    save_peak_descriptors(exp_peaks, std_peaks, out_dir / "03_peak_descriptors.png")
    save_peak_graphs(exp_peaks, std_peaks, out_dir / "04_standard_and_experimental_peak_graphs.png")

    candidates = generate_triangle_candidates(exp_peaks, std_peaks, args)
    selected_candidate, initial_plan, initial_matches = evaluate_candidates_with_ot(candidates, exp_peaks, std_peaks, prepared, master, args)
    save_transport_matching(
        initial_plan,
        initial_matches,
        exp_peaks,
        std_peaks,
        selected_candidate.rotation,
        out_dir / "05_partial_ot_peak_matching_initial.png",
    )
    graph_result, graph_refinement, graph_trace = fixed_pc_orientation_refinement(
        selected_candidate.rotation,
        initial_matches,
        exp_peaks,
        std_peaks,
        prepared,
        master,
        args,
    )
    final_plan, final_matches, final_ot_cost = final_matches_for_rotation(graph_result.rotation, exp_peaks, std_peaks, args)
    final_edge_loss = transport_edge_loss(final_plan, exp_peaks, std_peaks)
    save_transport_matching(
        final_plan,
        final_matches,
        exp_peaks,
        std_peaks,
        graph_result.rotation,
        out_dir / "06_partial_ot_peak_matching_fixed_pc_refined.png",
    )

    save_final_spatial_visualization(
        old_result,
        master,
        products,
        out_dir / "07_route_a_weighted_fixed_pc_final.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        graph_result,
        master,
        products,
        out_dir / "08_route_b_graph_fixed_pc_final.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    orientation_difference = route_angle_deg(old_result, graph_result)
    save_route_comparison(
        old_result,
        graph_result,
        master,
        products,
        out_dir / "09_fixed_pc_route_comparison.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_score_bar(
        old_result,
        graph_result,
        orientation_difference,
        len(final_matches),
        out_dir / "10_fixed_pc_route_metrics.png",
    )

    write_rows_csv([peak.to_row() for peak in exp_peaks], out_dir / "experimental_peak_descriptors.csv")
    write_rows_csv([peak.to_row() for peak in std_peaks], out_dir / "master_peak_descriptors.csv")
    write_rows_csv([match.to_row() for match in final_matches], out_dir / "partial_ot_matches_fixed_pc.csv")
    write_rows_csv(graph_trace, out_dir / "graph_fixed_pc_orientation_refinement_trace.csv")
    summary = {
        "pipeline": "compare_matching_routes_fixed_pc",
        "map": map_spec.key,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "h5_pattern_center_fixed": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "phase": phase_info,
        "route_a_weighted": old_result.to_json_dict(),
        "route_b_spherical_radon_graph_fixed_pc": graph_result.to_json_dict(),
        "route_b_selected_candidate": selected_candidate.to_row(),
        "route_b_refinement": graph_refinement,
        "route_b_final_partial_ot_cost": float(final_ot_cost),
        "route_b_final_edge_loss": float(final_edge_loss),
        "route_b_final_partial_ot_match_count": int(len(final_matches)),
        "orientation_difference_deg": float(orientation_difference),
        "experimental_transfer": transfer_info
        | {
            "band_id_point_count": {
                str(int(band_id)): int(np.sum(exp_transfer_band_ids == band_id))
                for band_id in np.unique(exp_transfer_band_ids)
            },
        },
        "hyperparameters": {
            "pc_adjustment": "disabled; both routes use the H5 PC exactly",
            "experimental_transfer_source": args.experimental_transfer_source,
            "experimental_peak_source": args.experimental_peak_source,
            "radon_kernel": args.radon_kernel,
            "ot_edge_weight": float(args.ot_edge_weight),
            "radon_scales_deg": scales,
            "normal_count": int(args.normal_count),
            "coarse_rotations": int(args.coarse_rotations),
            "refine_schedule": args.refine_schedule,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fixed-PC route comparison to: {out_dir}")
    print(f"Route A score={old_result.score:.5f}; Route B score={graph_result.score:.5f}; orientation difference={orientation_difference:.3f} deg")
    print(f"Graph route final OT matches={len(final_matches)}; fixed PC={bundle.pc}")


if __name__ == "__main__":
    main()
