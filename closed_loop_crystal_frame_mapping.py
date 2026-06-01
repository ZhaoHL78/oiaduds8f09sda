from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direct_hkl_sphere_localization import orientation_matrix_from_record
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DETECTOR_CONVENTIONS,
    MatchWeights,
    default_map_specs,
    jsonable,
    load_master_sphere,
    prepare_pattern,
    project_to_equirect,
    read_pattern_bundle,
    resolve_master_path,
    sphere_texture,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    equirect_line,
    plot_master_sphere,
    plot_pattern_patch,
)


@dataclass(frozen=True)
class Score:
    combined: float
    intensity: float
    band: float


@dataclass(frozen=True)
class GeometryCandidate:
    detector_convention: str
    rsd_name: str
    score: Score


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - float(values.mean())) / (float(values.std()) + 1e-8)


def percentile01(values: np.ndarray, mask: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    sample = values[mask].astype(np.float32)
    if sample.size == 0:
        return out
    lo, hi = np.percentile(sample, [low, high])
    if hi <= lo:
        return out
    out[mask] = np.clip((values[mask].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return out


def ncc(exp_values: np.ndarray, sim_values: np.ndarray) -> float:
    if exp_values.size < 3 or sim_values.size != exp_values.size:
        return 0.0
    return float(np.mean(zscore(exp_values) * zscore(sim_values)))


def detector_rays_from_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
    pc_z_scale: str,
) -> np.ndarray:
    pcx, pcy, pcz = pc
    z_denominator = pcz * (width if pc_z_scale == "width" else height)
    x_g = (cols.astype(np.float32) + 0.5 - pcx * width) / z_denominator
    y_g = (pcy * height - (rows.astype(np.float32) + 0.5)) / z_denominator
    rays = np.column_stack([x_g, y_g, np.ones_like(x_g, dtype=np.float32)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8
    return rays.astype(np.float32)


def detector_ray_grid(height: int, width: int, pc: tuple[float, float, float], pc_z_scale: str) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.float32)
    rays = detector_rays_from_pixels(rows.ravel(), cols.ravel(), height, width, pc, pc_z_scale)
    return rays.reshape(height, width, 3)


def parse_angle_token(token: str, sample_tilt_deg: float) -> tuple[str, float]:
    axis = token[:2]
    value = token[2:].strip()
    if value in {"+sample", "+tilt"}:
        angle = sample_tilt_deg
    elif value in {"-sample", "-tilt"}:
        angle = -sample_tilt_deg
    else:
        angle = float(value)
    return axis[1], angle


def rsd_matrix(name: str, sample_tilt_deg: float) -> np.ndarray:
    name = name.strip()
    if name == "identity":
        return np.eye(3, dtype=np.float64)
    if len(name) < 3 or name[0] != "r" or name[1] not in "xyz":
        raise ValueError(f"Unknown R_sd candidate: {name}")
    axis, angle = parse_angle_token(name, sample_tilt_deg)
    return R.from_euler(axis, angle, degrees=True).as_matrix().astype(np.float64)


def composite_detector_to_crystal(
    orientation_g: np.ndarray,
    detector_convention: str,
    rsd: np.ndarray,
    delta_rotvec: np.ndarray | None = None,
    orientation_direction: str = "gT",
) -> np.ndarray:
    convention = DETECTOR_CONVENTIONS[detector_convention].astype(np.float64)
    delta = np.eye(3, dtype=np.float64)
    if delta_rotvec is not None:
        delta = R.from_rotvec(delta_rotvec).as_matrix().astype(np.float64)

    if orientation_direction == "gT":
        # Column-vector formula: r_c = g.T @ delta.T @ R_sd @ C @ r_d
        return orientation_g.T @ delta.T @ rsd @ convention
    if orientation_direction == "g":
        return orientation_g @ delta.T @ rsd @ convention
    raise ValueError(f"Unknown orientation direction: {orientation_direction}")


def apply_composite(rays: np.ndarray, composite: np.ndarray) -> np.ndarray:
    vectors = rays.astype(np.float64) @ composite.T
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors.astype(np.float32)


def evaluate_score(
    master,
    rays: np.ndarray,
    exp_intensity: np.ndarray,
    exp_band: np.ndarray,
    composite: np.ndarray,
    intensity_weight: float,
    band_weight: float,
) -> Score:
    crystal_vectors = apply_composite(rays, composite)
    sim_intensity = master.sample_intensity(crystal_vectors)
    sim_band = master.sample_band(crystal_vectors)
    intensity_score = ncc(exp_intensity, sim_intensity)
    band_score = ncc(exp_band, sim_band)
    combined = intensity_weight * intensity_score + band_weight * band_score
    return Score(combined=float(combined), intensity=float(intensity_score), band=float(band_score))


def choose_score_indices(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    flat = np.flatnonzero(mask.ravel())
    if max_points <= 0 or flat.size <= max_points:
        return flat
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(flat, size=max_points, replace=False))


def rows_cols_from_flat(indices: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    rows = (indices // width).astype(np.float32)
    cols = (indices % width).astype(np.float32)
    return rows, cols


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stage1_coordinate_search(
    master,
    orientation_g: np.ndarray,
    rays: np.ndarray,
    exp_intensity: np.ndarray,
    exp_band: np.ndarray,
    detector_conventions: list[str],
    rsd_names: list[str],
    sample_tilt_deg: float,
    orientation_direction: str,
    intensity_weight: float,
    band_weight: float,
) -> tuple[GeometryCandidate, list[GeometryCandidate]]:
    candidates: list[GeometryCandidate] = []
    for convention_name in detector_conventions:
        for rsd_name in rsd_names:
            rsd = rsd_matrix(rsd_name, sample_tilt_deg)
            composite = composite_detector_to_crystal(
                orientation_g,
                convention_name,
                rsd,
                orientation_direction=orientation_direction,
            )
            score = evaluate_score(master, rays, exp_intensity, exp_band, composite, intensity_weight, band_weight)
            candidates.append(GeometryCandidate(convention_name, rsd_name, score))
    candidates.sort(key=lambda item: item.score.combined, reverse=True)
    return candidates[0], candidates


def pc_grid_values(center: tuple[float, float, float], xy_range: float, z_range: float, steps: int):
    pcx0, pcy0, pcz0 = center
    offsets_xy = np.linspace(-xy_range, xy_range, steps, dtype=np.float64)
    offsets_z = np.linspace(-z_range, z_range, steps, dtype=np.float64)
    for dx in offsets_xy:
        for dy in offsets_xy:
            for dz in offsets_z:
                yield (float(pcx0 + dx), float(pcy0 + dy), float(pcz0 + dz)), float(dx), float(dy), float(dz)


def stage2_pc_grid_search(
    master,
    orientation_g: np.ndarray,
    candidate: GeometryCandidate,
    exp_intensity: np.ndarray,
    exp_band: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    initial_pc: tuple[float, float, float],
    pc_z_scale: str,
    sample_tilt_deg: float,
    orientation_direction: str,
    intensity_weight: float,
    band_weight: float,
    xy_range: float,
    z_range: float,
    steps: int,
) -> tuple[tuple[float, float, float], Score, list[dict]]:
    rsd = rsd_matrix(candidate.rsd_name, sample_tilt_deg)
    composite = composite_detector_to_crystal(
        orientation_g,
        candidate.detector_convention,
        rsd,
        orientation_direction=orientation_direction,
    )
    rows_out: list[dict] = []
    best_pc = initial_pc
    best_score = Score(-np.inf, -np.inf, -np.inf)
    for pc, dx, dy, dz in pc_grid_values(initial_pc, xy_range, z_range, steps):
        rays = detector_rays_from_pixels(rows, cols, height, width, pc, pc_z_scale)
        score = evaluate_score(master, rays, exp_intensity, exp_band, composite, intensity_weight, band_weight)
        rows_out.append(
            {
                "pcx": pc[0],
                "pcy": pc[1],
                "pcz": pc[2],
                "dpcx": dx,
                "dpcy": dy,
                "dpcz": dz,
                "combined_ncc": score.combined,
                "intensity_ncc": score.intensity,
                "band_ncc": score.band,
            }
        )
        if score.combined > best_score.combined:
            best_pc = pc
            best_score = score
    rows_out.sort(key=lambda row: row["combined_ncc"], reverse=True)
    return best_pc, best_score, rows_out


def stage3_orientation_refine(
    master,
    orientation_g: np.ndarray,
    candidate: GeometryCandidate,
    exp_intensity: np.ndarray,
    exp_band: np.ndarray,
    rays: np.ndarray,
    sample_tilt_deg: float,
    orientation_direction: str,
    intensity_weight: float,
    band_weight: float,
    bound_deg: float,
    maxiter: int,
) -> tuple[np.ndarray, Score, list[dict]]:
    rsd = rsd_matrix(candidate.rsd_name, sample_tilt_deg)
    trace: list[dict] = []
    bound = math.radians(bound_deg)

    def objective(rotvec: np.ndarray) -> float:
        composite = composite_detector_to_crystal(
            orientation_g,
            candidate.detector_convention,
            rsd,
            delta_rotvec=rotvec,
            orientation_direction=orientation_direction,
        )
        score = evaluate_score(master, rays, exp_intensity, exp_band, composite, intensity_weight, band_weight)
        trace.append(
            {
                "rotvec_x_deg": math.degrees(float(rotvec[0])),
                "rotvec_y_deg": math.degrees(float(rotvec[1])),
                "rotvec_z_deg": math.degrees(float(rotvec[2])),
                "combined_ncc": score.combined,
                "intensity_ncc": score.intensity,
                "band_ncc": score.band,
            }
        )
        return -score.combined

    result = minimize(
        objective,
        x0=np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=[(-bound, bound), (-bound, bound), (-bound, bound)],
        options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-4, "disp": False},
    )
    best_rotvec = np.asarray(result.x, dtype=np.float64)
    final_composite = composite_detector_to_crystal(
        orientation_g,
        candidate.detector_convention,
        rsd,
        delta_rotvec=best_rotvec,
        orientation_direction=orientation_direction,
    )
    final_score = evaluate_score(master, rays, exp_intensity, exp_band, final_composite, intensity_weight, band_weight)
    return best_rotvec, final_score, trace


def simulated_images(
    master,
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    mask: np.ndarray,
    pc: tuple[float, float, float],
    pc_z_scale: str,
    composite: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = detector_rays_from_pixels(rows, cols, height, width, pc, pc_z_scale)
    crystal_vectors = apply_composite(rays, composite)
    sim_intensity = np.zeros((height, width), dtype=np.float32)
    sim_band = np.zeros((height, width), dtype=np.float32)
    sim_intensity.ravel()[mask.ravel()] = master.sample_intensity(crystal_vectors)
    sim_band.ravel()[mask.ravel()] = master.sample_band(crystal_vectors)
    return percentile01(sim_intensity, mask), percentile01(sim_band, mask), crystal_vectors


def transformed_band_curves(prepared, composite: np.ndarray, pc: tuple[float, float, float], pc_z_scale: str, samples: int = 260) -> list[np.ndarray]:
    curves: list[np.ndarray] = []
    height, width = prepared.image.shape
    for segment in prepared.line_segments:
        rows = np.linspace(segment.row0, segment.row1, samples, dtype=np.float32)
        cols = np.linspace(segment.col0, segment.col1, samples, dtype=np.float32)
        rays = detector_rays_from_pixels(rows, cols, height, width, pc, pc_z_scale)
        curves.append(apply_composite(rays, composite))
    return curves


def save_detector_sphere_step(prepared, products: dict[str, np.ndarray], detector_grid: np.ndarray, curves: list[np.ndarray], out_path: Path) -> None:
    raw = detector_raw_display(prepared)
    fig = plt.figure(figsize=(15.0, 9.8))
    ax0 = fig.add_subplot(221)
    ax0.imshow(raw, cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Raw experimental pattern")
    ax0.axis("off")
    ax1 = fig.add_subplot(222)
    ax1.imshow(products["contrast_enhanced"], cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Preprocessed experimental pattern")
    ax1.axis("off")
    ax2 = fig.add_subplot(223, projection="3d")
    plot_pattern_patch(ax2, detector_grid, raw, prepared.valid_mask, "Detector-frame sphere from EDAX PC", curves=curves)
    ax3 = fig.add_subplot(224, projection="3d")
    plot_pattern_patch(ax3, detector_grid, products["contrast_enhanced"], prepared.valid_mask, "Detector-frame sphere, enhanced", curves=curves)
    fig.suptitle("Layer 1: experimental pattern -> detector-frame sphere, using PC only", y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_candidate_barplot(candidates: list[GeometryCandidate], out_path: Path, top_n: int = 18) -> None:
    top = candidates[:top_n]
    labels = [f"{item.detector_convention}\n{item.rsd_name}" for item in top]
    scores = [item.score.combined for item in top]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(np.arange(len(top)), scores, color="#4c78a8")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("combined NCC")
    ax.set_title("Stage 1: coordinate convention and detector-to-sample geometry search")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_forward_validation(
    exp_intensity_img: np.ndarray,
    exp_band_img: np.ndarray,
    initial_sim_intensity: np.ndarray,
    initial_sim_band: np.ndarray,
    refined_sim_intensity: np.ndarray,
    refined_sim_band: np.ndarray,
    mask: np.ndarray,
    initial_score: Score,
    refined_score: Score,
    out_path: Path,
) -> None:
    diff_initial = np.zeros_like(exp_intensity_img, dtype=np.float32)
    diff_refined = np.zeros_like(exp_intensity_img, dtype=np.float32)
    diff_initial[mask] = np.abs(zscore(exp_intensity_img[mask]) - zscore(initial_sim_intensity[mask]))
    diff_refined[mask] = np.abs(zscore(exp_intensity_img[mask]) - zscore(refined_sim_intensity[mask]))
    diff_initial = percentile01(diff_initial, mask)
    diff_refined = percentile01(diff_refined, mask)

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.8))
    panels = [
        (axes[0, 0], exp_intensity_img, "Experimental corrected intensity"),
        (axes[0, 1], initial_sim_intensity, f"Initial forward simulation\nNCC={initial_score.combined:.4f}"),
        (axes[0, 2], refined_sim_intensity, f"Refined forward simulation\nNCC={refined_score.combined:.4f}"),
        (axes[0, 3], diff_refined, "Refined |z(exp)-z(sim)|"),
        (axes[1, 0], exp_band_img, "Experimental band response"),
        (axes[1, 1], initial_sim_band, f"Initial master band\nband NCC={initial_score.band:.4f}"),
        (axes[1, 2], refined_sim_band, f"Refined master band\nband NCC={refined_score.band:.4f}"),
        (axes[1, 3], diff_initial, "Initial |z(exp)-z(sim)|"),
    ]
    for ax, image, title in panels:
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Forward validation: sample master sphere at r_c(u,v), compare in detector plane", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_crystal_frame_maps(
    master,
    raw_values: np.ndarray,
    enhanced_values: np.ndarray,
    initial_vectors: np.ndarray,
    refined_vectors: np.ndarray,
    curves_initial: list[np.ndarray],
    curves_refined: list[np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    master_texture, _, _, _ = sphere_texture(master, lon_count, colat_count)
    initial_raw, initial_mask = project_to_equirect(initial_vectors, raw_values, lon_count, colat_count)
    refined_raw, refined_raw_mask = project_to_equirect(refined_vectors, raw_values, lon_count, colat_count)
    refined_enhanced, refined_enhanced_mask = project_to_equirect(refined_vectors, enhanced_values, lon_count, colat_count)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    panels = [
        (axes[0], initial_raw, initial_mask, curves_initial, "EDAX PC + orientation crystal-frame map"),
        (axes[1], refined_raw, refined_raw_mask, curves_refined, "Refined raw crystal-frame map"),
        (axes[2], refined_enhanced, refined_enhanced_mask, curves_refined, "Refined enhanced crystal-frame map"),
    ]
    for ax, image, mask, curves, title in panels:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.62)
        ax.imshow(
            image,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(mask, 0.93, 0.0),
        )
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(curves))))
        for curve, color in zip(curves, colors):
            equirect_line(ax, curve, color=color, linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("longitude in crystal frame (deg)")
        ax.set_ylabel("colatitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(180, 0)
        ax.grid(alpha=0.16)
    fig.suptitle("S_exp(theta, phi): experimental intensity mapped into the standard crystal-frame Kikuchi sphere", y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_refined_3d(
    prepared,
    master,
    refined_grid: np.ndarray,
    raw_img: np.ndarray,
    enhanced_img: np.ndarray,
    curves: list[np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    fig = plt.figure(figsize=(15.8, 7.4))
    ax0 = fig.add_subplot(121, projection="3d")
    plot_master_sphere(ax0, master, lon_count, colat_count, alpha=0.56)
    plot_pattern_patch(ax0, refined_grid, raw_img, prepared.valid_mask, "Raw pattern in refined crystal-frame sphere", curves=curves, draw_reference_sphere=False, radius_scale=1.006)
    ax1 = fig.add_subplot(122, projection="3d")
    plot_master_sphere(ax1, master, lon_count, colat_count, alpha=0.56)
    plot_pattern_patch(ax1, refined_grid, enhanced_img, prepared.valid_mask, "Enhanced pattern in refined crystal-frame sphere", curves=curves, draw_reference_sphere=False, radius_scale=1.006)
    fig.suptitle("Closed loop result: detector -> sample -> crystal -> master sphere", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop EBSD mapping: experimental pattern -> detector sphere -> sample sphere -> crystal sphere -> master pattern.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "closed_loop_crystal_frame_mapping")
    parser.add_argument("--pc-z-scale", choices=["width", "height"], default="width")
    parser.add_argument("--orientation-direction", choices=["gT", "g"], default="gT")
    parser.add_argument("--sample-tilt-deg", type=float, default=70.0)
    parser.add_argument(
        "--rsd-candidates",
        default="identity,rx+70,rx-70,ry+70,ry-70,rz+70,rz-70",
        help="Comma separated detector-to-sample candidates. Examples: identity,rx+70,rx-70,ry+70.",
    )
    parser.add_argument(
        "--detector-conventions",
        default=",".join(DETECTOR_CONVENTIONS.keys()),
        help="Comma separated detector coordinate transforms to test.",
    )
    parser.add_argument("--score-points", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--pc-xy-range", type=float, default=0.02)
    parser.add_argument("--pc-z-range", type=float, default=0.05)
    parser.add_argument("--pc-steps", type=int, default=5)
    parser.add_argument("--orientation-bound-deg", type=float, default=2.0)
    parser.add_argument("--orientation-maxiter", type=int, default=70)
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=2)
    parser.add_argument("--background-sigma", type=float, default=20.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=6)
    parser.add_argument("--match-quantile", type=float, default=0.92)
    parser.add_argument("--top-k-points", type=int, default=6500)
    parser.add_argument("--line-variant", default="auto")
    parser.add_argument("--sphere-lon-count", type=int, default=540)
    parser.add_argument("--sphere-colat-count", type=int, default=270)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_spec = default_map_specs(args.data_dir)[args.map]
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    prepared, variant_diagnostics = prepare_pattern(
        bundle=bundle,
        weights=MatchWeights(0.45, 0.15, 0.40),
        label="closed-loop-crystal-frame",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=args.line_variant,
    )
    products = build_preprocessing_products(
        bundle.pattern_u16,
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    master_h5 = resolve_master_path(args.master_h5)
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    height, width = prepared.image.shape
    valid_indices = choose_score_indices(prepared.valid_mask, args.score_points, args.seed)
    score_rows, score_cols = rows_cols_from_flat(valid_indices, width)
    score_rays = detector_rays_from_pixels(score_rows, score_cols, height, width, bundle.pc, args.pc_z_scale)
    exp_intensity_values = products["corrected_norm"].ravel()[valid_indices].astype(np.float32)
    exp_band_values = products["line_response"].ravel()[valid_indices].astype(np.float32)
    orientation_g = orientation_matrix_from_record(bundle.ang_record)

    detector_conventions = parse_list(args.detector_conventions)
    rsd_names = parse_list(args.rsd_candidates)
    best_candidate, candidates = stage1_coordinate_search(
        master,
        orientation_g,
        score_rays,
        exp_intensity_values,
        exp_band_values,
        detector_conventions,
        rsd_names,
        args.sample_tilt_deg,
        args.orientation_direction,
        args.intensity_weight,
        args.band_weight,
    )
    write_csv(
        [
            {
                "rank": rank + 1,
                "detector_convention": item.detector_convention,
                "rsd_name": item.rsd_name,
                "combined_ncc": item.score.combined,
                "intensity_ncc": item.score.intensity,
                "band_ncc": item.score.band,
            }
            for rank, item in enumerate(candidates)
        ],
        out_dir / "stage1_coordinate_search.csv",
    )

    refined_pc, pc_score, pc_rows = stage2_pc_grid_search(
        master,
        orientation_g,
        best_candidate,
        exp_intensity_values,
        exp_band_values,
        score_rows,
        score_cols,
        height,
        width,
        bundle.pc,
        args.pc_z_scale,
        args.sample_tilt_deg,
        args.orientation_direction,
        args.intensity_weight,
        args.band_weight,
        args.pc_xy_range,
        args.pc_z_range,
        args.pc_steps,
    )
    write_csv(pc_rows, out_dir / "stage2_pc_grid_search.csv")

    refined_score_rays = detector_rays_from_pixels(score_rows, score_cols, height, width, refined_pc, args.pc_z_scale)
    refined_rotvec, refined_score, orientation_trace = stage3_orientation_refine(
        master,
        orientation_g,
        best_candidate,
        exp_intensity_values,
        exp_band_values,
        refined_score_rays,
        args.sample_tilt_deg,
        args.orientation_direction,
        args.intensity_weight,
        args.band_weight,
        args.orientation_bound_deg,
        args.orientation_maxiter,
    )
    write_csv(orientation_trace, out_dir / "stage3_orientation_refinement_trace.csv")

    rsd_best = rsd_matrix(best_candidate.rsd_name, args.sample_tilt_deg)
    initial_composite = composite_detector_to_crystal(
        orientation_g,
        best_candidate.detector_convention,
        rsd_best,
        orientation_direction=args.orientation_direction,
    )
    refined_composite = composite_detector_to_crystal(
        orientation_g,
        best_candidate.detector_convention,
        rsd_best,
        delta_rotvec=refined_rotvec,
        orientation_direction=args.orientation_direction,
    )

    full_rows, full_cols = rows_cols_from_flat(np.flatnonzero(prepared.valid_mask.ravel()), width)
    raw_display = detector_raw_display(prepared)
    initial_sim_intensity, initial_sim_band, initial_vectors = simulated_images(
        master,
        full_rows,
        full_cols,
        height,
        width,
        prepared.valid_mask,
        bundle.pc,
        args.pc_z_scale,
        initial_composite,
    )
    refined_sim_intensity, refined_sim_band, refined_vectors = simulated_images(
        master,
        full_rows,
        full_cols,
        height,
        width,
        prepared.valid_mask,
        refined_pc,
        args.pc_z_scale,
        refined_composite,
    )
    full_exp_intensity = products["corrected_norm"][prepared.valid_mask]
    full_exp_band = products["line_response"][prepared.valid_mask]
    full_initial_score = Score(
        combined=args.intensity_weight * ncc(full_exp_intensity, initial_sim_intensity[prepared.valid_mask])
        + args.band_weight * ncc(full_exp_band, initial_sim_band[prepared.valid_mask]),
        intensity=ncc(full_exp_intensity, initial_sim_intensity[prepared.valid_mask]),
        band=ncc(full_exp_band, initial_sim_band[prepared.valid_mask]),
    )
    full_refined_score = Score(
        combined=args.intensity_weight * ncc(full_exp_intensity, refined_sim_intensity[prepared.valid_mask])
        + args.band_weight * ncc(full_exp_band, refined_sim_band[prepared.valid_mask]),
        intensity=ncc(full_exp_intensity, refined_sim_intensity[prepared.valid_mask]),
        band=ncc(full_exp_band, refined_sim_band[prepared.valid_mask]),
    )

    detector_grid = detector_ray_grid(height, width, bundle.pc, args.pc_z_scale)
    detector_curves = transformed_band_curves(prepared, np.eye(3), bundle.pc, args.pc_z_scale)
    save_detector_sphere_step(prepared, products, detector_grid, detector_curves, out_dir / "01_detector_frame_sphere_from_pc.png")
    save_candidate_barplot(candidates, out_dir / "02_coordinate_geometry_search_scores.png")
    save_forward_validation(
        products["corrected_norm"],
        products["line_response"],
        initial_sim_intensity,
        initial_sim_band,
        refined_sim_intensity,
        refined_sim_band,
        prepared.valid_mask,
        full_initial_score,
        full_refined_score,
        out_dir / "03_forward_validation_exp_vs_sim.png",
    )

    initial_curves = transformed_band_curves(prepared, initial_composite, bundle.pc, args.pc_z_scale)
    refined_curves = transformed_band_curves(prepared, refined_composite, refined_pc, args.pc_z_scale)
    save_crystal_frame_maps(
        master,
        raw_display[prepared.valid_mask],
        products["contrast_enhanced"][prepared.valid_mask],
        initial_vectors,
        refined_vectors,
        initial_curves,
        refined_curves,
        out_dir / "04_crystal_frame_spherical_map.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
    )
    refined_grid = detector_ray_grid(height, width, refined_pc, args.pc_z_scale).reshape(-1, 3)
    refined_grid = apply_composite(refined_grid, refined_composite).reshape(height, width, 3)
    save_refined_3d(
        prepared,
        master,
        refined_grid,
        raw_display,
        products["contrast_enhanced"],
        refined_curves,
        out_dir / "05_refined_closed_loop_3d.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
    )

    summary = {
        "pipeline": "closed_loop_crystal_frame_mapping",
        "formula": "r_c = g^T * delta_g^T * R_sd * C_detector * r_d",
        "map": asdict(map_spec),
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase_id": int(bundle.ang_record["Phase"]),
        "initial_pc": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "refined_pc": {"pcx": refined_pc[0], "pcy": refined_pc[1], "pcz": refined_pc[2]},
        "pc_delta": {"dpcx": refined_pc[0] - bundle.pc[0], "dpcy": refined_pc[1] - bundle.pc[1], "dpcz": refined_pc[2] - bundle.pc[2]},
        "pc_z_scale": args.pc_z_scale,
        "software_orientation_matrix_g": orientation_g.tolist(),
        "orientation_direction": args.orientation_direction,
        "best_detector_convention": best_candidate.detector_convention,
        "best_rsd_name": best_candidate.rsd_name,
        "sample_tilt_deg": args.sample_tilt_deg,
        "initial_subset_score": asdict(best_candidate.score),
        "pc_refined_subset_score": asdict(pc_score),
        "orientation_refined_subset_score": asdict(refined_score),
        "initial_full_score": asdict(full_initial_score),
        "refined_full_score": asdict(full_refined_score),
        "orientation_delta_rotvec_deg": [math.degrees(float(item)) for item in refined_rotvec],
        "line_variant": asdict(prepared.line_variant),
        "line_variant_score": float(prepared.line_variant_score),
        "line_variant_diagnostics": variant_diagnostics,
        "hyperparameters": {
            "detector_conventions": detector_conventions,
            "rsd_candidates": rsd_names,
            "score_points": args.score_points,
            "intensity_weight": args.intensity_weight,
            "band_weight": args.band_weight,
            "pc_xy_range": args.pc_xy_range,
            "pc_z_range": args.pc_z_range,
            "pc_steps": args.pc_steps,
            "orientation_bound_deg": args.orientation_bound_deg,
            "orientation_maxiter": args.orientation_maxiter,
        },
        "outputs": {
            "detector_frame_sphere": str(out_dir / "01_detector_frame_sphere_from_pc.png"),
            "coordinate_search": str(out_dir / "02_coordinate_geometry_search_scores.png"),
            "forward_validation": str(out_dir / "03_forward_validation_exp_vs_sim.png"),
            "crystal_frame_map": str(out_dir / "04_crystal_frame_spherical_map.png"),
            "refined_closed_loop_3d": str(out_dir / "05_refined_closed_loop_3d.png"),
            "stage1_csv": str(out_dir / "stage1_coordinate_search.csv"),
            "stage2_csv": str(out_dir / "stage2_pc_grid_search.csv"),
            "stage3_csv": str(out_dir / "stage3_orientation_refinement_trace.csv"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)

    print(f"Saved closed-loop crystal-frame mapping to: {out_dir}")
    print(
        "Best discrete geometry: "
        f"detector={best_candidate.detector_convention}, R_sd={best_candidate.rsd_name}, "
        f"subset NCC={best_candidate.score.combined:.4f}"
    )
    print(
        "Refined: "
        f"PC={tuple(round(v, 6) for v in refined_pc)}, "
        f"drot(deg)={[round(math.degrees(float(v)), 4) for v in refined_rotvec]}, "
        f"full NCC {full_initial_score.combined:.4f} -> {full_refined_score.combined:.4f}"
    )


if __name__ == "__main__":
    main()
