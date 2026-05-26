from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direct_hkl_sphere_localization import (
    crystal_to_detector_matrix,
    orientation_matrix_from_record,
)
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
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    equirect_line,
    patch_surface_step,
    plot_master_sphere,
    plot_pattern_patch,
    set_3d_sphere_axes,
)


def detector_grid_to_master_grid(detector_grid: np.ndarray, crystal_to_detector: np.ndarray) -> np.ndarray:
    master_grid = detector_grid.reshape(-1, 3).astype(np.float64) @ crystal_to_detector
    master_grid = master_grid.reshape(detector_grid.shape).astype(np.float32)
    master_grid /= np.linalg.norm(master_grid, axis=-1, keepdims=True) + 1e-8
    return master_grid


def transformed_band_curves(prepared, crystal_to_detector: np.ndarray, samples: int = 260) -> list[np.ndarray]:
    curves: list[np.ndarray] = []
    height, width = prepared.image.shape
    for segment in prepared.line_segments:
        rows = np.linspace(segment.row0, segment.row1, samples, dtype=np.float32)
        cols = np.linspace(segment.col0, segment.col1, samples, dtype=np.float32)
        curve = detector_pixels_to_sphere(rows, cols, height, width, prepared.bundle.pc).astype(np.float64)
        curve = curve @ crystal_to_detector
        curve /= np.linalg.norm(curve, axis=1, keepdims=True) + 1e-12
        curves.append(curve.astype(np.float32))
    return curves


def save_pc_backprojection_visualization(prepared, products: dict[str, np.ndarray], out_path: Path) -> None:
    raw_display = detector_raw_display(prepared)
    detector_curves = []
    height, width = prepared.image.shape
    for segment in prepared.line_segments:
        rows = np.linspace(segment.row0, segment.row1, 260, dtype=np.float32)
        cols = np.linspace(segment.col0, segment.col1, 260, dtype=np.float32)
        detector_curves.append(detector_pixels_to_sphere(rows, cols, height, width, prepared.bundle.pc))

    fig = plt.figure(figsize=(15.0, 10.0))
    ax0 = fig.add_subplot(221)
    ax0.imshow(raw_display, cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Raw detector pattern")
    ax0.axis("off")

    ax1 = fig.add_subplot(222)
    ax1.imshow(products["contrast_enhanced"], cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Contrast-enhanced detector pattern")
    ax1.axis("off")

    ax2 = fig.add_subplot(223, projection="3d")
    plot_pattern_patch(
        ax2,
        prepared.full_points_grid,
        raw_display,
        prepared.valid_mask,
        "Raw pattern back-projected to detector sphere by PC",
        curves=detector_curves,
    )

    ax3 = fig.add_subplot(224, projection="3d")
    plot_pattern_patch(
        ax3,
        prepared.full_points_grid,
        products["contrast_enhanced"],
        prepared.valid_mask,
        "Enhanced pattern back-projected to detector sphere by PC",
        curves=detector_curves,
    )

    fig.suptitle(
        f"Step 1: PC back-projection only | idx={prepared.bundle.index} "
        f"PC=({prepared.bundle.pc[0]:.5f}, {prepared.bundle.pc[1]:.5f}, {prepared.bundle.pc[2]:.5f})",
        y=0.985,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pattern_on_master(
    ax,
    master,
    master_grid: np.ndarray,
    texture01: np.ndarray,
    mask: np.ndarray,
    curves: list[np.ndarray],
    title: str,
    lon_count: int,
    colat_count: int,
) -> None:
    plot_master_sphere(ax, master, lon_count, colat_count, alpha=0.56)
    plot_pattern_patch(
        ax,
        master_grid,
        texture01,
        mask,
        title,
        curves=curves,
        draw_reference_sphere=False,
        radius_scale=1.006,
    )


def save_master_position_3d(
    prepared,
    products: dict[str, np.ndarray],
    master,
    master_grid: np.ndarray,
    curves: list[np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
    orientation_op: str,
    detector_convention: str,
) -> None:
    raw_display = detector_raw_display(prepared)
    fig = plt.figure(figsize=(16.2, 7.6))

    ax0 = fig.add_subplot(121, projection="3d")
    plot_pattern_on_master(
        ax0,
        master,
        master_grid,
        raw_display,
        prepared.valid_mask,
        curves,
        "Raw pattern placed on standard Kikuchi sphere",
        lon_count,
        colat_count,
    )

    ax1 = fig.add_subplot(122, projection="3d")
    plot_pattern_on_master(
        ax1,
        master,
        master_grid,
        products["contrast_enhanced"],
        prepared.valid_mask,
        curves,
        "Enhanced pattern at the same software-orientation position",
        lon_count,
        colat_count,
    )

    fig.suptitle(
        "Step 2: PC sphere rotated onto master sphere by software orientation | "
        f"orientation_op={orientation_op}, detector_convention={detector_convention}",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_master_position_equirect(
    prepared,
    products: dict[str, np.ndarray],
    master,
    master_grid: np.ndarray,
    curves: list[np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    raw_display = detector_raw_display(prepared)
    master_texture, _, _, _ = sphere_texture(master, lon_count, colat_count)
    raw_projection, raw_mask = project_to_equirect(master_grid[prepared.valid_mask], raw_display[prepared.valid_mask], lon_count, colat_count)
    enhanced_projection, enhanced_mask = project_to_equirect(
        master_grid[prepared.valid_mask],
        products["contrast_enhanced"][prepared.valid_mask],
        lon_count,
        colat_count,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.5))
    panels = [
        (axes[0], raw_projection, raw_mask, "Raw pattern footprint on master sphere"),
        (axes[1], enhanced_projection, enhanced_mask, "Enhanced pattern footprint on master sphere"),
    ]
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(curves))))
    for ax, projection, mask, title in panels:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.68)
        ax.imshow(
            projection,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(mask, 0.94, 0.0),
        )
        for curve, color in zip(curves, colors):
            equirect_line(ax, curve, color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("longitude on master sphere (deg)")
        ax.set_ylabel("colatitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(180, 0)
        ax.grid(alpha=0.18)
    fig.suptitle("Software orientation places the PC-corrected experimental sphere on the standard Kikuchi sphere", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_orientation_frame_visualization(
    master_grid: np.ndarray,
    mask: np.ndarray,
    crystal_to_detector: np.ndarray,
    out_path: Path,
) -> None:
    detector_axes = np.eye(3, dtype=np.float64)
    crystal_axes = detector_axes @ crystal_to_detector
    crystal_axes = crystal_axes / (np.linalg.norm(crystal_axes, axis=1, keepdims=True) + 1e-12)

    fig = plt.figure(figsize=(6.8, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    visible = master_grid[mask]
    ax.scatter(visible[:: max(1, len(visible) // 2500), 0], visible[:: max(1, len(visible) // 2500), 1], visible[:: max(1, len(visible) // 2500), 2], s=3, alpha=0.08)
    axis_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    axis_labels = ["detector +X after orientation", "detector +Y after orientation", "detector +Z after orientation"]
    for vec, color, label in zip(crystal_axes, axis_colors, axis_labels):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=2.2, length=1.0, normalize=True)
        ax.text(vec[0] * 1.08, vec[1] * 1.08, vec[2] * 1.08, label, color=color, fontsize=8)
    set_3d_sphere_axes(ax, "Software orientation frame on master sphere", view_vectors=visible)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def bunge_like_euler_deg(orientation_matrix: np.ndarray) -> list[float]:
    try:
        angles = R.from_matrix(orientation_matrix).as_euler("ZXZ", degrees=True)
    except ValueError:
        angles = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return [float(item) for item in angles]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project an experimental EBSD pattern to the detector sphere by PC, then place it on the master Kikuchi sphere using the software orientation.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "software_orientation_sphere_projection")
    parser.add_argument("--orientation-op", choices=["G_T", "G"], default="G_T")
    parser.add_argument("--detector-convention", choices=list(DETECTOR_CONVENTIONS.keys()), default="flip_xy")
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
        label="software-orientation-projection",
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

    software_orientation = orientation_matrix_from_record(bundle.ang_record)
    crystal_to_detector = crystal_to_detector_matrix(
        software_orientation,
        args.orientation_op,
        args.detector_convention,
    )
    detector_grid = detector_to_sphere_grid(prepared.image.shape[0], prepared.image.shape[1], bundle.pc)
    master_grid = detector_grid_to_master_grid(detector_grid, crystal_to_detector)
    curves = transformed_band_curves(prepared, crystal_to_detector)

    master_h5 = resolve_master_path(args.master_h5)
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    save_pc_backprojection_visualization(prepared, products, out_dir / "01_pc_backprojection_to_detector_sphere.png")
    save_master_position_3d(
        prepared,
        products,
        master,
        master_grid,
        curves,
        out_dir / "02_software_orientation_position_on_master_sphere_3d.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
        args.orientation_op,
        args.detector_convention,
    )
    save_master_position_equirect(
        prepared,
        products,
        master,
        master_grid,
        curves,
        out_dir / "03_software_orientation_position_on_master_sphere_map.png",
        args.sphere_lon_count,
        args.sphere_colat_count,
    )
    save_orientation_frame_visualization(
        master_grid,
        prepared.valid_mask,
        crystal_to_detector,
        out_dir / "04_software_orientation_frame_on_master_sphere.png",
    )

    valid_vectors = master_grid[prepared.valid_mask]
    lon = np.degrees(np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0]))
    colat = np.degrees(np.arccos(np.clip(valid_vectors[:, 2], -1.0, 1.0)))
    summary = {
        "pipeline": "software_orientation_sphere_projection",
        "description": "PC back-projects the detector pattern to a sphere; the H5 software orientation places that sphere on the standard Kikuchi master sphere. No orientation matching is performed.",
        "map": asdict(map_spec),
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "phase_id": int(bundle.ang_record["Phase"]),
        "software_orientation_matrix_from_h5": software_orientation.tolist(),
        "software_orientation_euler_zxz_deg_for_reference": bunge_like_euler_deg(software_orientation),
        "orientation_op": args.orientation_op,
        "detector_convention": args.detector_convention,
        "crystal_to_detector_matrix": crystal_to_detector.tolist(),
        "detector_to_master_row_vector_formula": "master_vector = detector_vector @ crystal_to_detector_matrix",
        "master_footprint_lon_deg": [float(np.min(lon)), float(np.max(lon))],
        "master_footprint_colat_deg": [float(np.min(colat)), float(np.max(colat))],
        "line_variant": asdict(prepared.line_variant),
        "line_variant_score": float(prepared.line_variant_score),
        "line_variant_diagnostics": variant_diagnostics,
        "outputs": {
            "pc_backprojection": str(out_dir / "01_pc_backprojection_to_detector_sphere.png"),
            "software_orientation_3d": str(out_dir / "02_software_orientation_position_on_master_sphere_3d.png"),
            "software_orientation_map": str(out_dir / "03_software_orientation_position_on_master_sphere_map.png"),
            "orientation_frame": str(out_dir / "04_software_orientation_frame_on_master_sphere.png"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)

    print(f"Saved software-orientation sphere projection to: {out_dir}")
    print(
        "Used H5 PC and H5 ANG orientation only: "
        f"PC={bundle.pc}, orientation_op={args.orientation_op}, detector_convention={args.detector_convention}"
    )
    print(
        "Master-sphere footprint: "
        f"lon=[{summary['master_footprint_lon_deg'][0]:.1f}, {summary['master_footprint_lon_deg'][1]:.1f}] deg, "
        f"colat=[{summary['master_footprint_colat_deg'][0]:.1f}, {summary['master_footprint_colat_deg'][1]:.1f}] deg"
    )


if __name__ == "__main__":
    main()
