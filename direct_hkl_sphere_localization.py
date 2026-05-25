from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DETECTOR_CONVENTIONS,
    MatchWeights,
    default_map_specs,
    detector_pixels_to_sphere,
    detector_to_sphere_grid,
    jsonable,
    load_master_sphere,
    prepare_pattern,
    project_to_equirect,
    read_pattern_bundle,
    resolve_master_path,
    sphere_texture,
)
from labeled_band_radius_refinement import (
    HKLFamily,
    band_plane_normal,
    great_circle_from_normal,
    read_phase_hkl_families,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    equirect_line,
)


@dataclass(frozen=True)
class PredictedPlane:
    plane_id: int
    hkl: str
    family_intensity: float
    crystal_normal: np.ndarray
    detector_normal: np.ndarray
    line: tuple[float, float, float, float] | None
    visible_length_px: float


@dataclass(frozen=True)
class DirectConventionScore:
    orientation_op: str
    detector_convention: str
    weighted_mean_angle_deg: float
    max_angle_deg: float
    matched_count: int


def canonical_plane_normal(normal: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    if normal[2] < -1e-7 or (abs(float(normal[2])) <= 1e-7 and normal[1] < -1e-7):
        normal = -normal
    if abs(float(normal[2])) <= 1e-7 and abs(float(normal[1])) <= 1e-7 and normal[0] < 0:
        normal = -normal
    return normal.astype(np.float32)


def unique_antipodal_normals(normals: np.ndarray) -> np.ndarray:
    unique: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for normal in normals:
        canonical = canonical_plane_normal(normal)
        key = tuple(np.round(canonical, 6).astype(float))
        if key in seen:
            continue
        seen.add(key)
        unique.append(canonical)
    return np.asarray(unique, dtype=np.float32)


def orientation_matrix_from_record(record: dict) -> np.ndarray:
    matrix = np.asarray(record["Orientations"], dtype=np.float64).reshape(3, 3)
    det = float(np.linalg.det(matrix))
    if det < 0:
        matrix = matrix.copy()
        matrix[:, -1] *= -1.0
    return matrix


def crystal_to_detector_matrix(orientation: np.ndarray, orientation_op: str, detector_convention: str) -> np.ndarray:
    if orientation_op == "G":
        base = orientation
    elif orientation_op == "G_T":
        base = orientation.T
    else:
        raise ValueError(f"Unknown orientation op: {orientation_op}")
    convention = DETECTOR_CONVENTIONS[detector_convention].astype(np.float64)
    matrix = convention @ base
    return matrix.astype(np.float64)


def predicted_line_from_detector_normal(
    normal: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    normal = normal.astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    pcx, pcy, pcz = pc
    cx = pcx * (width - 1)
    cy = pcy * (height - 1)
    scale = pcz * height

    # n . [(col-cx)/scale, -(row-cy)/scale, 1] = 0
    a = normal[0]
    b = -normal[1]
    c = -normal[0] * cx + normal[1] * cy + normal[2] * scale

    points: list[tuple[float, float]] = []
    if abs(b) > 1e-10:
        for col in (0.0, float(width - 1)):
            row = -(a * col + c) / b
            if -1e-5 <= row <= height - 1 + 1e-5:
                points.append((row, col))
    if abs(a) > 1e-10:
        for row in (0.0, float(height - 1)):
            col = -(b * row + c) / a
            if -1e-5 <= col <= width - 1 + 1e-5:
                points.append((row, col))

    unique: list[tuple[float, float]] = []
    for row, col in points:
        if all((row - old_row) ** 2 + (col - old_col) ** 2 > 1e-4 for old_row, old_col in unique):
            unique.append((float(row), float(col)))
    if len(unique) < 2:
        return None

    best_pair = (unique[0], unique[1])
    best_dist = -1.0
    for i, first in enumerate(unique):
        for second in unique[i + 1 :]:
            dist = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = (first, second)
    (row0, col0), (row1, col1) = best_pair
    return row0, col0, row1, col1


def line_length(line: tuple[float, float, float, float] | None) -> float:
    if line is None:
        return 0.0
    row0, col0, row1, col1 = line
    return float(math.hypot(row1 - row0, col1 - col0))


def build_predicted_planes(
    families: list[HKLFamily],
    crystal_to_detector: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
    min_visible_length_px: float,
) -> list[PredictedPlane]:
    planes: list[PredictedPlane] = []
    plane_id = 0
    for family in families:
        for crystal_normal in unique_antipodal_normals(family.normals):
            detector_normal = crystal_to_detector @ crystal_normal.astype(np.float64)
            detector_normal = detector_normal / (np.linalg.norm(detector_normal) + 1e-12)
            line = predicted_line_from_detector_normal(detector_normal, height, width, pc)
            visible_length = line_length(line)
            if visible_length < min_visible_length_px:
                continue
            planes.append(
                PredictedPlane(
                    plane_id=plane_id,
                    hkl=family.label,
                    family_intensity=float(family.diffraction_intensity),
                    crystal_normal=crystal_normal.astype(np.float32),
                    detector_normal=canonical_plane_normal(detector_normal.astype(np.float32)),
                    line=line,
                    visible_length_px=visible_length,
                )
            )
            plane_id += 1
    return planes


def observed_band_normals(prepared) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    normals: list[np.ndarray] = []
    weights: list[float] = []
    for segment in prepared.line_segments:
        normal = band_plane_normal(prepared, segment)
        if not np.isfinite(normal).all():
            continue
        normal = canonical_plane_normal(normal)
        rows.append(
            {
                "band_index": int(segment.band_index),
                "rho_bin": float(segment.band.rho_bin),
                "theta_deg": float(segment.band.theta_deg),
                "width": float(segment.band.width),
                "intensity": float(segment.band.intensity),
                "line": (float(segment.row0), float(segment.col0), float(segment.row1), float(segment.col1)),
            }
        )
        normals.append(normal)
        weights.append(max(1e-3, float(segment.band.intensity)))
    return rows, np.asarray(normals, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def match_observed_to_planes(
    observed_normals: np.ndarray,
    observed_weights: np.ndarray,
    planes: list[PredictedPlane],
) -> tuple[list[dict], float, float]:
    if len(observed_normals) == 0 or len(planes) == 0:
        return [], float("nan"), float("nan")
    predicted = np.asarray([plane.detector_normal for plane in planes], dtype=np.float32)
    dots = np.clip(np.abs(observed_normals @ predicted.T), -1.0, 1.0)
    cost = np.degrees(np.arccos(dots))
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_rows: list[dict] = []
    angles: list[float] = []
    weights: list[float] = []
    for row, col in zip(row_ind, col_ind):
        angle = float(cost[row, col])
        plane = planes[int(col)]
        matched_rows.append(
            {
                "observed_order": int(row),
                "plane_id": int(plane.plane_id),
                "hkl": plane.hkl,
                "angle_deg": angle,
                "family_intensity": float(plane.family_intensity),
                "visible_length_px": float(plane.visible_length_px),
                "crystal_normal": plane.crystal_normal.astype(float).tolist(),
                "detector_normal": plane.detector_normal.astype(float).tolist(),
            }
        )
        angles.append(angle)
        weights.append(float(observed_weights[row]))
    weights_array = np.asarray(weights, dtype=np.float32)
    angles_array = np.asarray(angles, dtype=np.float32)
    weighted_mean = float(np.sum(weights_array * angles_array) / (np.sum(weights_array) + 1e-8))
    return matched_rows, weighted_mean, float(np.max(angles_array))


def select_direct_convention(
    orientation: np.ndarray,
    families: list[HKLFamily],
    observed_normals: np.ndarray,
    observed_weights: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
    min_visible_length_px: float,
    requested_orientation_op: str,
    requested_detector_convention: str,
) -> tuple[str, str, np.ndarray, list[PredictedPlane], list[dict], list[DirectConventionScore]]:
    orientation_ops = ["G_T", "G"] if requested_orientation_op == "auto" else [requested_orientation_op]
    detector_conventions = list(DETECTOR_CONVENTIONS) if requested_detector_convention == "auto" else [requested_detector_convention]

    diagnostics: list[DirectConventionScore] = []
    best: tuple[float, str, str, np.ndarray, list[PredictedPlane], list[dict], float] | None = None
    for op in orientation_ops:
        for convention in detector_conventions:
            matrix = crystal_to_detector_matrix(orientation, op, convention)
            planes = build_predicted_planes(families, matrix, height, width, pc, min_visible_length_px)
            matches, weighted_mean, max_angle = match_observed_to_planes(observed_normals, observed_weights, planes)
            score = weighted_mean if np.isfinite(weighted_mean) else float("inf")
            diagnostics.append(
                DirectConventionScore(
                    orientation_op=op,
                    detector_convention=convention,
                    weighted_mean_angle_deg=float(score),
                    max_angle_deg=float(max_angle),
                    matched_count=len(matches),
                )
            )
            if best is None or score < best[0]:
                best = (score, op, convention, matrix, planes, matches, max_angle)

    if best is None:
        raise RuntimeError("No direct HKL convention was evaluated")
    _, op, convention, matrix, planes, matches, _ = best
    return op, convention, matrix, planes, matches, sorted(diagnostics, key=lambda item: item.weighted_mean_angle_deg)


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_detector_overlay(prepared, observed_rows: list[dict], planes: list[PredictedPlane], matches: list[dict], out_path: Path) -> None:
    raw_display = detector_raw_display(prepared)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(10, len(matches))))
    plane_by_id = {plane.plane_id: plane for plane in planes}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax in axes:
        ax.imshow(raw_display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    axes[0].set_title(f"Software OHP bands on detector | {prepared.line_variant.name}")
    for row, color in zip(observed_rows, colors):
        row0, col0, row1, col1 = row["line"]
        axes[0].plot([col0, col1], [row0, row1], color=color, linewidth=1.8)
        axes[0].text((col0 + col1) / 2.0, (row0 + row1) / 2.0, str(row["band_index"]), color="white", fontsize=8)

    axes[1].set_title("Direct software orientation + HKL prediction")
    for match, color in zip(matches, colors):
        obs = observed_rows[match["observed_order"]]
        row0, col0, row1, col1 = obs["line"]
        axes[1].plot([col0, col1], [row0, row1], color=color, linewidth=2.0, alpha=0.85)
        plane = plane_by_id[match["plane_id"]]
        if plane.line is not None:
            prow0, pcol0, prow1, pcol1 = plane.line
            axes[1].plot([pcol0, pcol1], [prow0, prow1], color=color, linewidth=1.3, linestyle="--")
            axes[1].text(
                (pcol0 + pcol1) / 2.0,
                (prow0 + prow1) / 2.0,
                f"{obs['band_index']}:{match['hkl']}\n{match['angle_deg']:.1f} deg",
                color="white",
                fontsize=7,
                ha="center",
                va="center",
            )

    fig.suptitle("Solid = H5/OHP detected band, dashed = HKL line predicted from ANG orientation", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def normal_lon_colat(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    colat = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))
    return lon, colat


def save_normal_space_visualization(
    observed_normals: np.ndarray,
    planes: list[PredictedPlane],
    matches: list[dict],
    crystal_to_detector: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    plane_by_id = {plane.plane_id: plane for plane in planes}
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(10, len(matches))))

    predicted = np.asarray([plane.detector_normal for plane in planes], dtype=np.float32)
    pred_lon, pred_colat = normal_lon_colat(predicted)
    obs_lon, obs_colat = normal_lon_colat(observed_normals)
    axes[0].scatter(pred_lon, pred_colat, s=18, c="#777777", alpha=0.28, label="all visible HKL normals")
    axes[0].scatter(obs_lon, obs_colat, s=62, c="#e24a33", edgecolors="white", linewidths=0.6, label="OHP band normals")
    for match, color in zip(matches, colors):
        plane = plane_by_id[match["plane_id"]]
        p_lon, p_colat = normal_lon_colat(np.asarray([plane.detector_normal]))
        o_lon, o_colat = normal_lon_colat(np.asarray([observed_normals[match["observed_order"]]]))
        axes[0].plot([o_lon[0], p_lon[0]], [o_colat[0], p_colat[0]], color=color, linewidth=1.2)
        axes[0].text(p_lon[0], p_colat[0], match["hkl"], fontsize=7, color=color)
    axes[0].set_title("Detector plane-normal Hough points")

    observed_crystal = observed_normals.astype(np.float64) @ crystal_to_detector
    observed_crystal /= np.linalg.norm(observed_crystal, axis=1, keepdims=True) + 1e-12
    observed_crystal = np.asarray([canonical_plane_normal(item) for item in observed_crystal], dtype=np.float32)
    crystal_candidates = np.asarray([plane.crystal_normal for plane in planes], dtype=np.float32)
    cand_lon, cand_colat = normal_lon_colat(crystal_candidates)
    obs_c_lon, obs_c_colat = normal_lon_colat(observed_crystal)
    axes[1].scatter(cand_lon, cand_colat, s=18, c="#777777", alpha=0.25, label="HKL family normals")
    axes[1].scatter(obs_c_lon, obs_c_colat, s=62, c="#348abd", edgecolors="white", linewidths=0.6, label="OHP normals transformed to crystal frame")
    for match, color in zip(matches, colors):
        plane = plane_by_id[match["plane_id"]]
        p_lon, p_colat = normal_lon_colat(np.asarray([plane.crystal_normal]))
        o_lon, o_colat = normal_lon_colat(np.asarray([observed_crystal[match["observed_order"]]]))
        axes[1].plot([o_lon[0], p_lon[0]], [o_colat[0], p_colat[0]], color=color, linewidth=1.2)
        axes[1].text(p_lon[0], p_colat[0], match["hkl"], fontsize=7, color=color)
    axes[1].set_title("Crystal/master-sphere plane normals")

    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(180, 0)
        ax.set_xlabel("normal longitude (deg)")
        ax.set_ylabel("normal colatitude (deg)")
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Direct HKL localization: each software band becomes a plane-normal point", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pattern_on_sphere_visualization(
    prepared,
    products: dict[str, np.ndarray],
    master,
    crystal_to_detector: np.ndarray,
    planes: list[PredictedPlane],
    matches: list[dict],
    observed_rows: list[dict],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    height, width = prepared.image.shape
    detector_grid = detector_to_sphere_grid(height, width, prepared.bundle.pc)
    crystal_grid = detector_grid.reshape(-1, 3).astype(np.float64) @ crystal_to_detector
    crystal_grid = crystal_grid.reshape(detector_grid.shape).astype(np.float32)
    crystal_grid /= np.linalg.norm(crystal_grid, axis=-1, keepdims=True) + 1e-8

    raw_display = detector_raw_display(prepared)
    raw_projection, raw_mask = project_to_equirect(crystal_grid[prepared.valid_mask], raw_display[prepared.valid_mask], lon_count, colat_count)
    corrected_projection, corrected_mask = project_to_equirect(
        crystal_grid[prepared.valid_mask],
        products["contrast_enhanced"][prepared.valid_mask],
        lon_count,
        colat_count,
    )
    master_texture, _, _, _ = sphere_texture(master, lon_count, colat_count)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    panels = [
        (axes[0], raw_projection, raw_mask, "Raw pattern placed by software orientation"),
        (axes[1], corrected_projection, corrected_mask, "Contrast-enhanced pattern at the same position"),
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(10, len(matches))))
    plane_by_id = {plane.plane_id: plane for plane in planes}

    for ax, projection, mask, title in panels:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.72)
        ax.imshow(
            projection,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(mask, 0.92, 0.0),
        )
        for match, color in zip(matches, colors):
            row0, col0, row1, col1 = observed_rows[match["observed_order"]]["line"]
            rows = np.linspace(row0, row1, 260, dtype=np.float32)
            cols = np.linspace(col0, col1, 260, dtype=np.float32)
            observed_curve = detector_pixels_to_sphere(rows, cols, height, width, prepared.bundle.pc).astype(np.float64) @ crystal_to_detector
            observed_curve /= np.linalg.norm(observed_curve, axis=1, keepdims=True) + 1e-12
            equirect_line(ax, observed_curve.astype(np.float32), color=color, linewidth=1.7)

            plane = plane_by_id[match["plane_id"]]
            standard_curve = great_circle_from_normal(plane.crystal_normal, samples=361)
            equirect_line(ax, standard_curve, color=color, linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("longitude on master/crystal sphere (deg)")
        ax.set_ylabel("colatitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(180, 0)
        ax.grid(alpha=0.18)

    fig.suptitle("Direct software ANG orientation + phase HKL families, no orientation search", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def enrich_match_rows(matches: list[dict], observed_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for match in matches:
        obs = observed_rows[match["observed_order"]]
        row = {
            "band_index": obs["band_index"],
            "rho_bin": obs["rho_bin"],
            "theta_deg": obs["theta_deg"],
            "width": obs["width"],
            "intensity": obs["intensity"],
            "matched_hkl": match["hkl"],
            "angle_deg": match["angle_deg"],
            "plane_id": match["plane_id"],
            "family_intensity": match["family_intensity"],
            "visible_length_px": match["visible_length_px"],
            "crystal_normal": json.dumps(match["crystal_normal"]),
            "detector_normal": json.dumps(match["detector_normal"]),
        }
        out.append(row)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use EDAX ANG orientation plus phase HKL families to directly localize a pattern on the Kikuchi sphere.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "direct_hkl_sphere_localization")
    parser.add_argument("--orientation-op", choices=["auto", "G", "G_T"], default="auto")
    parser.add_argument("--detector-convention", choices=["auto", *DETECTOR_CONVENTIONS.keys()], default="auto")
    parser.add_argument("--min-visible-length-px", type=float, default=80.0)
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
        label="direct-software-hkl",
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
    phase_id = int(bundle.ang_record["Phase"])
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)
    orientation = orientation_matrix_from_record(bundle.ang_record)
    observed_rows, observed_normals, observed_weights = observed_band_normals(prepared)

    op, convention, crystal_to_detector, planes, matches, diagnostics = select_direct_convention(
        orientation,
        families,
        observed_normals,
        observed_weights,
        prepared.image.shape[0],
        prepared.image.shape[1],
        bundle.pc,
        args.min_visible_length_px,
        args.orientation_op,
        args.detector_convention,
    )

    match_rows = enrich_match_rows(matches, observed_rows)
    write_csv(match_rows, out_dir / "direct_hkl_band_matches.csv")
    write_csv([asdict(item) for item in diagnostics], out_dir / "direct_hkl_convention_diagnostics.csv")

    save_detector_overlay(prepared, observed_rows, planes, matches, out_dir / "01_detector_ohp_vs_direct_hkl_prediction.png")
    save_normal_space_visualization(
        observed_normals,
        planes,
        matches,
        crystal_to_detector,
        out_dir / "02_sphere_hough_points_direct_hkl.png",
    )

    master_h5 = resolve_master_path(args.master_h5)
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    save_pattern_on_sphere_visualization(
        prepared,
        products,
        master,
        crystal_to_detector,
        planes,
        matches,
        observed_rows,
        out_dir / "03_pattern_on_master_sphere_by_software_hkl.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
    )

    angle_values = np.asarray([row["angle_deg"] for row in match_rows], dtype=np.float32)
    summary = {
        "pipeline": "direct_hkl_sphere_localization",
        "map": asdict(map_spec),
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "phase": phase_info,
        "h5_per_band_hkl_status": "not present in OHP/DATA; direct localization uses ANG/DATA orientation plus phase HKL families",
        "ang_orientation_matrix": orientation.tolist(),
        "chosen_orientation_op": op,
        "chosen_detector_convention": convention,
        "chosen_crystal_to_detector_matrix": crystal_to_detector.tolist(),
        "line_variant": asdict(prepared.line_variant),
        "line_variant_score": float(prepared.line_variant_score),
        "line_variant_diagnostics": variant_diagnostics,
        "matched_band_count": len(match_rows),
        "mean_angle_deg": float(np.mean(angle_values)) if angle_values.size else float("nan"),
        "max_angle_deg": float(np.max(angle_values)) if angle_values.size else float("nan"),
        "weighted_mean_angle_deg": diagnostics[0].weighted_mean_angle_deg if diagnostics else float("nan"),
        "families": [
            {
                "label": family.label,
                "diffraction_intensity": float(family.diffraction_intensity),
                "normal_count": int(len(unique_antipodal_normals(family.normals))),
            }
            for family in families
        ],
        "outputs": {
            "detector_overlay": str(out_dir / "01_detector_ohp_vs_direct_hkl_prediction.png"),
            "normal_points": str(out_dir / "02_sphere_hough_points_direct_hkl.png"),
            "pattern_on_sphere": str(out_dir / "03_pattern_on_master_sphere_by_software_hkl.png"),
            "matches_csv": str(out_dir / "direct_hkl_band_matches.csv"),
            "convention_diagnostics_csv": str(out_dir / "direct_hkl_convention_diagnostics.csv"),
        },
        "hyperparameters": {
            "orientation_op": args.orientation_op,
            "detector_convention": args.detector_convention,
            "min_visible_length_px": args.min_visible_length_px,
            "sphere_lon_count": args.sphere_lon_count,
            "sphere_colat_count": args.sphere_colat_count,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)

    print(f"Saved direct HKL localization to: {out_dir}")
    print(f"Chosen convention: orientation_op={op}, detector_convention={convention}")
    print(
        "Matched bands: "
        f"{len(match_rows)}, weighted mean angle={summary['weighted_mean_angle_deg']:.3f} deg, "
        f"max angle={summary['max_angle_deg']:.3f} deg"
    )


if __name__ == "__main__":
    main()
