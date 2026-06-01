from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    default_map_specs,
    jsonable,
    load_master_sphere,
    project_to_equirect,
    read_pattern_bundle,
    resolve_master_path,
    sphere_texture,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    equirect_line,
    percentile_normalize,
    plot_master_sphere,
    plot_pattern_patch,
)


@dataclass(frozen=True)
class FileGeometry:
    sample_tilt_deg: float
    pre_tilt_deg: float
    camera_elevation_deg: float
    camera_azimuth_deg: float
    camera_diameter: float
    camera_model: str
    stage_tilt_deg: float
    stage_rotation_deg: float


def h5_scalar(group: h5py.Group, path: str, default: float = 0.0) -> float:
    if path not in group:
        return float(default)
    return float(np.asarray(group[path][()]).reshape(-1)[0])


def h5_string(group: h5py.Group, path: str, default: str = "") -> str:
    if path not in group:
        return default
    value = np.asarray(group[path][()]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_file_geometry(h5_path: Path, h5_group: str) -> FileGeometry:
    with h5py.File(h5_path, "r") as f:
        group = f[h5_group]
        return FileGeometry(
            sample_tilt_deg=h5_scalar(group, "Sample/Sample Tilt", 70.0),
            pre_tilt_deg=h5_scalar(group, "Sample/Pre Tilt", 70.0),
            camera_elevation_deg=h5_scalar(group, "Camera/Elevation Angle", 0.0),
            camera_azimuth_deg=h5_scalar(group, "Camera/Azimuthal Angle", 0.0),
            camera_diameter=h5_scalar(group, "Camera/Diameter", 0.0),
            camera_model=h5_string(group, "Camera/Model", ""),
            stage_tilt_deg=h5_scalar(group, "Stage/TiltAngle", 0.0),
            stage_rotation_deg=h5_scalar(group, "Stage/RotationAngle", 0.0),
        )


def orientation_matrix_from_record(record: dict) -> np.ndarray:
    matrix = np.asarray(record["Orientations"], dtype=np.float64).reshape(3, 3)
    if np.linalg.det(matrix) < 0:
        matrix = matrix.copy()
        matrix[:, -1] *= -1.0
    return matrix


def bunge_like_euler_deg(matrix: np.ndarray) -> list[float]:
    try:
        return [float(v) for v in R.from_matrix(matrix).as_euler("ZXZ", degrees=True)]
    except ValueError:
        return [float("nan"), float("nan"), float("nan")]


def edax_pc_detector_rays_from_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
) -> np.ndarray:
    """EDAX TSL detector-frame rays using the user's explicit convention.

    x_g = (u + 0.5 - PC_x W) / (PC_z W)
    y_g = (PC_y H - (v + 0.5)) / (PC_z W)
    r_d = normalize([x_g, y_g, 1])
    """

    pcx, pcy, pcz = pc
    denominator = pcz * width
    x_g = (cols.astype(np.float64) + 0.5 - pcx * width) / denominator
    y_g = (pcy * height - (rows.astype(np.float64) + 0.5)) / denominator
    rays = np.column_stack([x_g, y_g, np.ones_like(x_g, dtype=np.float64)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
    return rays.astype(np.float32)


def edax_pc_detector_ray_grid(height: int, width: int, pc: tuple[float, float, float]) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.float32)
    rays = edax_pc_detector_rays_from_pixels(rows.ravel(), cols.ravel(), height, width, pc)
    return rays.reshape(height, width, 3)


def detector_to_sample_matrix(geometry: FileGeometry, model: str) -> np.ndarray:
    """Return the fixed detector -> sample rotation read from file metadata.

    This function does not fit or search. It only chooses how to interpret the
    H5 geometry values. The default, sample_tilt, uses the sample tilt value
    directly as R_sd = Rx(-sample_tilt).
    """

    if model == "identity":
        return np.eye(3, dtype=np.float64)
    if model == "sample_tilt":
        return R.from_euler("x", -geometry.sample_tilt_deg, degrees=True).as_matrix()
    if model == "sample_tilt_plus_camera":
        rz = R.from_euler("z", geometry.camera_azimuth_deg, degrees=True).as_matrix()
        rx = R.from_euler("x", -(geometry.sample_tilt_deg - geometry.camera_elevation_deg), degrees=True).as_matrix()
        return rz @ rx
    raise ValueError(f"Unknown geometry model: {model}")


def kikuchipy_edax_sample_rays_from_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    pc_edax: tuple[float, float, float],
    geometry: FileGeometry,
) -> np.ndarray:
    """Sample-frame direction cosines following kikuchipy's EDAX reader.

    This is a direct transcription of kikuchipy's fixed-PC detector geometry:
    EDAX PC -> Bruker PC, then detector tilt/elevation, azimuthal angle, and
    sample tilt are applied. It is deterministic file geometry, not fitted.
    """

    pcx_tsl, pcy_tsl, pcz_tsl = pc_edax
    pcx = float(pcx_tsl)
    pcy = 1.0 - float(pcy_tsl)
    pcz = float(pcz_tsl) * min(width, height) / float(height)

    xpc = width * (0.5 - pcx)
    ypc = height * (0.5 - pcy)
    zpc = height * pcz

    det_x = xpc + (1.0 - width) * 0.5 + cols.astype(np.float64)
    nrows_from_bottom = (height - 1.0) - rows.astype(np.float64)
    det_y = ypc - (1.0 - height) * 0.5 - nrows_from_bottom

    alpha = np.pi / 2.0 - np.deg2rad(geometry.sample_tilt_deg) + np.deg2rad(geometry.camera_elevation_deg)
    azimuthal = np.deg2rad(geometry.camera_azimuth_deg)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cw = np.cos(azimuthal)
    sw = np.sin(azimuthal)

    ls = -sw * det_x + zpc * cw
    lc = cw * det_x + zpc * sw

    rays = np.column_stack(
        [
            det_y * ca + sa * ls,
            lc,
            -sa * det_y + ca * ls,
        ]
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
    return rays.astype(np.float32)


def kikuchipy_edax_sample_ray_grid(
    height: int,
    width: int,
    pc_edax: tuple[float, float, float],
    geometry: FileGeometry,
) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.float32)
    rays = kikuchipy_edax_sample_rays_from_pixels(rows.ravel(), cols.ravel(), height, width, pc_edax, geometry)
    return rays.reshape(height, width, 3)


def apply_column_rotation_grid(points_grid: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply column-vector rotation matrix to a row-vector grid."""

    out = points_grid.reshape(-1, 3).astype(np.float64) @ matrix.T
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out.reshape(points_grid.shape).astype(np.float32)


def detector_to_crystal_grid(
    detector_grid: np.ndarray,
    detector_to_sample: np.ndarray | None,
    orientation_g: np.ndarray,
    *,
    sample_grid_direct: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_grid_direct is None:
        if detector_to_sample is None:
            raise ValueError("detector_to_sample matrix is required unless sample_grid_direct is provided")
        sample_grid = apply_column_rotation_grid(detector_grid, detector_to_sample)
    else:
        sample_grid = sample_grid_direct
    # Column-vector formula: r_c = g.T @ r_s.
    # Row-vector equivalent: r_c(row) = r_s(row) @ g.
    crystal_grid = sample_grid.reshape(-1, 3).astype(np.float64) @ orientation_g
    crystal_grid /= np.linalg.norm(crystal_grid, axis=1, keepdims=True) + 1e-12
    return sample_grid, crystal_grid.reshape(detector_grid.shape).astype(np.float32)


def sample_grid_from_geometry_model(
    detector_grid: np.ndarray,
    pc: tuple[float, float, float],
    geometry: FileGeometry,
    model: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = detector_grid.shape[:2]
    if model == "kikuchipy_edax":
        return kikuchipy_edax_sample_ray_grid(height, width, pc, geometry), None
    rsd = detector_to_sample_matrix(geometry, model)
    return apply_column_rotation_grid(detector_grid, rsd), rsd


def sample_rays_from_geometry_model(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
    pc: tuple[float, float, float],
    geometry: FileGeometry,
    model: str,
    rsd: np.ndarray | None,
) -> np.ndarray:
    if model == "kikuchipy_edax":
        return kikuchipy_edax_sample_rays_from_pixels(rows, cols, height, width, pc, geometry)
    if rsd is None:
        raise ValueError("R_sd matrix is required for this geometry model")
    detector_rays = edax_pc_detector_rays_from_pixels(rows, cols, height, width, pc)
    sample = detector_rays.astype(np.float64) @ rsd.T
    sample /= np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12
    return sample.astype(np.float32)


def percentile_raw_display(pattern_u16: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return percentile_normalize(pattern_u16.astype(np.float32), mask, low=0.2, high=99.8)


def save_pc_only_detector_sphere(
    raw_display: np.ndarray,
    enhanced: np.ndarray,
    valid_mask: np.ndarray,
    detector_grid: np.ndarray,
    pc: tuple[float, float, float],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14.8, 10.6))
    ax0 = fig.add_subplot(221)
    ax0.imshow(raw_display, cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Raw pattern")
    ax0.axis("off")

    ax1 = fig.add_subplot(222)
    ax1.imshow(enhanced, cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Contrast-normalized pattern")
    ax1.axis("off")

    ax2 = fig.add_subplot(223, projection="3d")
    plot_pattern_patch(
        ax2,
        detector_grid,
        raw_display,
        valid_mask,
        "Raw pattern corrected to detector-frame sphere by PC only",
        curves=None,
    )

    ax3 = fig.add_subplot(224, projection="3d")
    plot_pattern_patch(
        ax3,
        detector_grid,
        enhanced,
        valid_mask,
        "Contrast-normalized pattern corrected to detector-frame sphere by PC only",
        curves=None,
    )

    fig.suptitle(
        f"Layer 1: EDAX PC-only spherical correction | PC=({pc[0]:.6f}, {pc[1]:.6f}, {pc[2]:.6f})",
        y=0.985,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_frame_chain_3d(
    raw_display: np.ndarray,
    valid_mask: np.ndarray,
    detector_grid: np.ndarray,
    sample_grid: np.ndarray,
    crystal_grid: np.ndarray,
    out_path: Path,
    geometry_model: str,
) -> None:
    fig = plt.figure(figsize=(17.8, 5.8))
    panels = [
        (detector_grid, "Detector-frame sphere\nPC only"),
        (sample_grid, f"Sample-frame sphere\nR_sd from H5 ({geometry_model})"),
        (crystal_grid, "Crystal-frame sphere\ng^T from H5 orientation"),
    ]
    for i, (points, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plot_pattern_patch(
            ax,
            points,
            raw_display,
            valid_mask,
            title,
            curves=None,
            draw_reference_sphere=True,
        )
    fig.suptitle("Geometry-only coordinate chain: no matching, no PC/orientation adjustment", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_master_overlap_3d(
    master,
    raw_display: np.ndarray,
    enhanced: np.ndarray,
    valid_mask: np.ndarray,
    crystal_grid: np.ndarray,
    out_path: Path,
    lon_count: int,
    colat_count: int,
    geometry_model: str,
) -> None:
    fig = plt.figure(figsize=(15.8, 7.4))
    ax0 = fig.add_subplot(121, projection="3d")
    plot_master_sphere(ax0, master, lon_count, colat_count, alpha=0.58)
    plot_pattern_patch(
        ax0,
        crystal_grid,
        raw_display,
        valid_mask,
        "Raw pattern on standard Kikuchi sphere",
        curves=None,
        draw_reference_sphere=False,
        radius_scale=1.006,
    )

    ax1 = fig.add_subplot(122, projection="3d")
    plot_master_sphere(ax1, master, lon_count, colat_count, alpha=0.58)
    plot_pattern_patch(
        ax1,
        crystal_grid,
        enhanced,
        valid_mask,
        "Contrast-normalized pattern at same position",
        curves=None,
        draw_reference_sphere=False,
        radius_scale=1.006,
    )
    fig.suptitle(
        f"Final geometry-only relative position on master sphere | R_sd={geometry_model}, no matching/refinement",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_master_overlap_equirect(
    master,
    raw_display: np.ndarray,
    enhanced: np.ndarray,
    valid_mask: np.ndarray,
    crystal_grid: np.ndarray,
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    master_texture, _, _, _ = sphere_texture(master, lon_count, colat_count)
    raw_projection, raw_mask = project_to_equirect(
        crystal_grid[valid_mask],
        raw_display[valid_mask],
        lon_count,
        colat_count,
    )
    enhanced_projection, enhanced_mask = project_to_equirect(
        crystal_grid[valid_mask],
        enhanced[valid_mask],
        lon_count,
        colat_count,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.5))
    panels = [
        (axes[0], raw_projection, raw_mask, "Raw experimental footprint in crystal-frame sphere"),
        (axes[1], enhanced_projection, enhanced_mask, "Contrast-normalized experimental footprint in crystal-frame sphere"),
    ]
    for ax, projection, mask, title in panels:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.66)
        ax.imshow(
            projection,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(mask, 0.94, 0.0),
        )
        ax.set_title(title)
        ax.set_xlabel("longitude in standard crystal frame (deg)")
        ax.set_ylabel("colatitude in standard crystal frame (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(180, 0)
        ax.grid(alpha=0.16)
    fig.suptitle("Experimental pattern written into the same coordinate system as the standard master sphere", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_forward_master_overlay(
    master,
    raw_display: np.ndarray,
    enhanced: np.ndarray,
    valid_mask: np.ndarray,
    crystal_grid: np.ndarray,
    out_path: Path,
) -> None:
    simulated = np.zeros(valid_mask.shape, dtype=np.float32)
    simulated[valid_mask] = master.sample_intensity(crystal_grid[valid_mask])
    simulated = percentile_normalize(simulated, valid_mask, low=0.5, high=99.5)

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.0))
    panels = [
        (axes[0, 0], raw_display, "Raw experimental pattern"),
        (axes[0, 1], simulated, "Master sphere sampled at H5 PC + orientation"),
        (axes[0, 2], np.abs(raw_display - simulated), "Absolute visual difference"),
        (axes[1, 0], enhanced, "Contrast-normalized experimental pattern"),
        (axes[1, 1], simulated, "Same master projection"),
        (axes[1, 2], np.abs(enhanced - simulated), "Absolute visual difference"),
    ]
    for ax, image, title in panels:
        shown = np.array(image, copy=True)
        shown[~valid_mask] = 0.0
        ax.imshow(shown, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Forward check only: query the standard sphere at r_c(u,v), no optimization", y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def transform_band_segments_to_crystal(
    bundle,
    geometry: FileGeometry,
    geometry_model: str,
    detector_to_sample: np.ndarray | None,
    orientation_g: np.ndarray,
    samples_per_segment: int = 260,
) -> list[np.ndarray]:
    """Optional visual helper for H5/OHP bands, using only PC and orientation."""

    from h5_band_enhanced_match import MatchWeights, choose_line_variant, normalize_u16, preprocess_pattern
    from h5_band_enhanced_match import circular_mask

    image = normalize_u16(bundle.pattern_u16)
    height, width = image.shape
    valid_mask = circular_mask(height, width, 0.49)
    valid_mask, _, _, image_band_score = preprocess_pattern(
        image,
        valid_mask,
        mask_erosion=2,
        background_sigma=20.0,
        band_sigma_min=1,
        band_sigma_max=6,
    )
    line_variant, _, _, segments, _ = choose_line_variant(
        bundle.bands,
        bundle.ohp_header,
        height,
        width,
        image_band_score,
        valid_mask,
        "auto",
    )
    _ = MatchWeights  # Keep the import local and explicit; no matching is performed here.
    curves: list[np.ndarray] = []
    for segment in segments:
        rows = np.linspace(segment.row0, segment.row1, samples_per_segment, dtype=np.float32)
        cols = np.linspace(segment.col0, segment.col1, samples_per_segment, dtype=np.float32)
        sample = sample_rays_from_geometry_model(
            rows,
            cols,
            height,
            width,
            bundle.pc,
            geometry,
            geometry_model,
            detector_to_sample,
        ).astype(np.float64)
        crystal = sample @ orientation_g
        crystal /= np.linalg.norm(crystal, axis=1, keepdims=True) + 1e-12
        curves.append(crystal.astype(np.float32))
    return curves


def save_h5_band_curves_on_master_map(
    master,
    crystal_grid: np.ndarray,
    valid_mask: np.ndarray,
    curves: list[np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    master_texture, _, _, _ = sphere_texture(master, lon_count, colat_count)
    footprint, footprint_mask = project_to_equirect(
        crystal_grid[valid_mask],
        np.ones(int(valid_mask.sum()), dtype=np.float32),
        lon_count,
        colat_count,
    )
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(curves))))

    fig, ax = plt.subplots(1, 1, figsize=(11.8, 6.2))
    ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.78)
    ax.imshow(
        footprint,
        cmap="magma",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=np.where(footprint_mask, 0.28, 0.0),
    )
    for curve, color in zip(curves, colors):
        equirect_line(ax, curve, color=color, linewidth=1.35)
    ax.set_title("H5/OHP band curves carried through the same PC + orientation chain")
    ax.set_xlabel("longitude in standard crystal frame (deg)")
    ax.set_ylabel("colatitude in standard crystal frame (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_file_geometry_comparison(
    master,
    raw_display: np.ndarray,
    valid_mask: np.ndarray,
    detector_grid: np.ndarray,
    pc: tuple[float, float, float],
    orientation_g: np.ndarray,
    geometry: FileGeometry,
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> dict[str, dict[str, list[float]]]:
    models = ["identity", "sample_tilt", "sample_tilt_plus_camera", "kikuchipy_edax"]
    fig = plt.figure(figsize=(20.0, 5.8))
    footprint_summary: dict[str, dict[str, list[float]]] = {}
    for i, model in enumerate(models, start=1):
        sample_grid, rsd = sample_grid_from_geometry_model(detector_grid, pc, geometry, model)
        _, crystal_grid = detector_to_crystal_grid(detector_grid, rsd, orientation_g, sample_grid_direct=sample_grid)
        visible = crystal_grid[valid_mask]
        lon = np.degrees(np.arctan2(visible[:, 1], visible[:, 0]))
        colat = np.degrees(np.arccos(np.clip(visible[:, 2], -1.0, 1.0)))
        footprint_summary[model] = {
            "lon_deg": [float(np.min(lon)), float(np.max(lon))],
            "colat_deg": [float(np.min(colat)), float(np.max(colat))],
        }
        ax = fig.add_subplot(1, 4, i, projection="3d")
        plot_master_sphere(ax, master, lon_count, colat_count, alpha=0.50)
        plot_pattern_patch(
            ax,
            crystal_grid,
            raw_display,
            valid_mask,
            model,
            curves=None,
            draw_reference_sphere=False,
            radius_scale=1.006,
        )
    fig.suptitle("Only file-parameter geometry variants, shown for convention diagnosis; no variant is fitted or optimized", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return footprint_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Geometry-only EBSD projection: raw pattern + H5 PC -> detector sphere, "
            "then H5 orientation and fixed file geometry -> standard crystal/master sphere."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "geometry_only_pc_orientation_projection")
    parser.add_argument(
        "--geometry-model",
        choices=["identity", "sample_tilt", "sample_tilt_plus_camera", "kikuchipy_edax"],
        default="kikuchipy_edax",
        help="Fixed detector-to-sample interpretation from H5 metadata. This is not optimized.",
    )
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=2)
    parser.add_argument("--background-sigma", type=float, default=20.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=6)
    parser.add_argument("--sphere-lon-count", type=int, default=540)
    parser.add_argument("--sphere-colat-count", type=int, default=270)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_spec = default_map_specs(args.data_dir)[args.map]
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    geometry = read_file_geometry(args.h5, map_spec.h5_group)
    products = build_preprocessing_products(
        bundle.pattern_u16,
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    valid_mask = products["valid_mask"].astype(bool)
    raw_display = percentile_raw_display(bundle.pattern_u16, valid_mask)
    enhanced = products["contrast_enhanced"]

    height, width = bundle.pattern_u16.shape
    detector_grid = edax_pc_detector_ray_grid(height, width, bundle.pc)
    orientation_g = orientation_matrix_from_record(bundle.ang_record)
    sample_grid, rsd = sample_grid_from_geometry_model(detector_grid, bundle.pc, geometry, args.geometry_model)
    _, crystal_grid = detector_to_crystal_grid(detector_grid, rsd, orientation_g, sample_grid_direct=sample_grid)

    master_h5 = resolve_master_path(args.master_h5)
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    save_pc_only_detector_sphere(
        raw_display,
        enhanced,
        valid_mask,
        detector_grid,
        bundle.pc,
        out_dir / "01_pc_only_detector_frame_sphere.png",
    )
    save_frame_chain_3d(
        raw_display,
        valid_mask,
        detector_grid,
        sample_grid,
        crystal_grid,
        out_dir / "02_detector_sample_crystal_frame_chain.png",
        args.geometry_model,
    )
    save_master_overlap_3d(
        master,
        raw_display,
        enhanced,
        valid_mask,
        crystal_grid,
        out_dir / "03_final_geometry_only_overlap_on_master_sphere_3d.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
        args.geometry_model,
    )
    save_master_overlap_equirect(
        master,
        raw_display,
        enhanced,
        valid_mask,
        crystal_grid,
        out_dir / "04_crystal_frame_master_sphere_map.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
    )
    save_forward_master_overlay(
        master,
        raw_display,
        enhanced,
        valid_mask,
        crystal_grid,
        out_dir / "05_forward_master_projection_check.png",
    )

    try:
        curves = transform_band_segments_to_crystal(bundle, geometry, args.geometry_model, rsd, orientation_g)
        save_h5_band_curves_on_master_map(
            master,
            crystal_grid,
            valid_mask,
            curves,
            out_dir / "06_h5_band_curves_in_crystal_frame.png",
            args.sphere_lon_count,
            args.sphere_colat_count,
        )
    except Exception as exc:
        curves = []
        (out_dir / "06_h5_band_curves_in_crystal_frame_error.txt").write_text(str(exc), encoding="utf-8")

    geometry_footprints = save_file_geometry_comparison(
        master,
        raw_display,
        valid_mask,
        detector_grid,
        bundle.pc,
        orientation_g,
        geometry,
        out_dir / "07_file_geometry_variants_no_fitting.png",
        min(args.sphere_lon_count, 420),
        min(args.sphere_colat_count, 210),
    )

    visible = crystal_grid[valid_mask]
    lon = np.degrees(np.arctan2(visible[:, 1], visible[:, 0]))
    colat = np.degrees(np.arccos(np.clip(visible[:, 2], -1.0, 1.0)))
    summary = {
        "pipeline": "geometry_only_pc_orientation_projection",
        "description": (
            "No matching, no coordinate search, no PC/orientation refinement. "
            "The experimental pixels are back-projected with the H5 EDAX PC, "
            "then carried through R_sd from file geometry and g^T from H5 orientation."
        ),
        "formula_column_vectors": {
            "detector_ray": "r_d = normalize([(u+0.5-PCx*W)/(PCz*W), (PCy*H-(v+0.5))/(PCz*W), 1])",
            "sample_direction": "r_s = R_sd * r_d, or the equivalent kikuchipy EDAX detector geometry when geometry_model=kikuchipy_edax",
            "crystal_direction": "r_c = g^T * r_s",
        },
        "row_vector_implementation": "matrix models: r_c(row) = r_d(row) @ R_sd.T @ g; kikuchipy_edax: r_s(row) is computed directly, then r_c(row)=r_s(row) @ g",
        "map": asdict(map_spec),
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase_id": int(bundle.ang_record["Phase"]),
        "pattern_shape_hw": [int(height), int(width)],
        "pattern_center_edax_tsl_from_h5": {
            "pcx": float(bundle.pc[0]),
            "pcy": float(bundle.pc[1]),
            "pcz": float(bundle.pc[2]),
        },
        "file_geometry_from_h5": asdict(geometry),
        "geometry_model_used": args.geometry_model,
        "detector_to_sample_R_sd": None if rsd is None else rsd.tolist(),
        "detector_to_sample_note": (
            "For geometry_model=kikuchipy_edax, sample-frame rays are computed directly from the EDAX PC plus "
            "Camera Elevation/Azimuth and Sample Tilt, following kikuchipy's EDAX H5 reader. No fitting is used."
        ),
        "software_orientation_matrix_g_from_h5": orientation_g.tolist(),
        "software_orientation_interpretation": "g is treated as crystal -> sample, so g.T is used for sample -> crystal.",
        "software_orientation_euler_zxz_deg_for_reference": bunge_like_euler_deg(orientation_g),
        "crystal_frame_footprint_lon_deg": [float(np.min(lon)), float(np.max(lon))],
        "crystal_frame_footprint_colat_deg": [float(np.min(colat)), float(np.max(colat))],
        "h5_band_curves_written": int(len(curves)),
        "geometry_variant_footprints_no_fitting": geometry_footprints,
        "outputs": {
            "pc_only_detector_frame_sphere": str(out_dir / "01_pc_only_detector_frame_sphere.png"),
            "detector_sample_crystal_frame_chain": str(out_dir / "02_detector_sample_crystal_frame_chain.png"),
            "final_overlap_3d": str(out_dir / "03_final_geometry_only_overlap_on_master_sphere_3d.png"),
            "crystal_frame_master_sphere_map": str(out_dir / "04_crystal_frame_master_sphere_map.png"),
            "forward_master_projection_check": str(out_dir / "05_forward_master_projection_check.png"),
            "h5_band_curves_in_crystal_frame": str(out_dir / "06_h5_band_curves_in_crystal_frame.png"),
            "file_geometry_variants_no_fitting": str(out_dir / "07_file_geometry_variants_no_fitting.png"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)

    print(f"Saved geometry-only PC/orientation projection to: {out_dir}")
    print(f"PC from H5: {tuple(round(float(v), 6) for v in bundle.pc)}")
    print(f"R_sd model: {args.geometry_model}; sample tilt={geometry.sample_tilt_deg:.3f} deg; camera elevation={geometry.camera_elevation_deg:.3f} deg")
    print(f"Orientation g from H5 det={np.linalg.det(orientation_g):.6f}; ZXZ reference={tuple(round(v, 3) for v in bunge_like_euler_deg(orientation_g))}")
    print(f"Crystal-frame footprint lon=[{summary['crystal_frame_footprint_lon_deg'][0]:.1f}, {summary['crystal_frame_footprint_lon_deg'][1]:.1f}] deg, colat=[{summary['crystal_frame_footprint_colat_deg'][0]:.1f}, {summary['crystal_frame_footprint_colat_deg'][1]:.1f}] deg")


if __name__ == "__main__":
    main()
