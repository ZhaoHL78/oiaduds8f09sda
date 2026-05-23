from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import cm
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    MasterSphere,
    circular_mask,
    default_map_specs,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    resolve_master_path,
    save_score_comparison,
    segment_curve_vectors,
    sphere_texture,
)


def percentile_normalize(values: np.ndarray, mask: np.ndarray | None = None, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = values.astype(np.float32)
    sample = values[mask] if mask is not None else values.ravel()
    lo, hi = np.percentile(sample, [low, high])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0.0
    return out.astype(np.float32)


def apply_clahe(image01: np.ndarray, mask: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    image_u8 = np.clip(image01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(image_u8).astype(np.float32) / 255.0
    enhanced[~mask] = 0.0
    return enhanced


def build_preprocessing_products(
    raw_u16: np.ndarray,
    mask_radius_fraction: float,
    mask_erosion: int,
    background_sigma: float,
    band_sigma_min: int,
    band_sigma_max: int,
) -> dict[str, np.ndarray]:
    raw01 = raw_u16.astype(np.float32) / 65535.0
    mask0 = circular_mask(raw_u16.shape[0], raw_u16.shape[1], mask_radius_fraction)
    valid_mask = ndi.binary_erosion(mask0, iterations=mask_erosion)
    valid_mask = morphology.remove_small_holes(valid_mask, area_threshold=64)
    valid_mask = morphology.remove_small_objects(valid_mask, min_size=256)

    raw_percentile = percentile_normalize(raw01, valid_mask)
    background = filters.gaussian(raw01, sigma=background_sigma)
    corrected = raw01 - background
    corrected[~valid_mask] = 0.0
    corrected_norm = percentile_normalize(corrected, valid_mask, low=1.0, high=99.5)
    contrast_enhanced = apply_clahe(corrected_norm, valid_mask)

    line_response = filters.meijering(
        corrected_norm,
        sigmas=range(band_sigma_min, band_sigma_max + 1),
        black_ridges=False,
    )
    line_response = exposure.rescale_intensity(line_response, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    line_response[~valid_mask] = 0.0

    return {
        "raw01": raw01,
        "mask0": mask0.astype(np.float32),
        "valid_mask": valid_mask,
        "raw_percentile": raw_percentile,
        "background": percentile_normalize(background, mask0.astype(bool)),
        "corrected_norm": corrected_norm,
        "contrast_enhanced": contrast_enhanced,
        "line_response": line_response,
    }


def save_preprocessing_visualization(prepared, products: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 11.2))
    panels = [
        ("Raw UP2 uint16", prepared.bundle.pattern_u16, "gray", None, None),
        ("Raw normalized", products["raw01"], "gray", 0.0, 1.0),
        ("Valid circular mask", products["valid_mask"].astype(np.float32), "gray", 0.0, 1.0),
        ("Percentile normalized", products["raw_percentile"], "gray", 0.0, 1.0),
        ("Gaussian background", products["background"], "gray", 0.0, 1.0),
        ("Background-corrected", products["corrected_norm"], "gray", 0.0, 1.0),
        ("CLAHE contrast enhanced", products["contrast_enhanced"], "gray", 0.0, 1.0),
        ("Line response", products["line_response"], "gray", 0.0, 1.0),
        ("H5/OHP band raster", prepared.h5_band_score, "gray", 0.0, 1.0),
    ]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if title == "Raw UP2 uint16":
            ax.imshow(image, cmap=cmap, vmin=int(image.min()), vmax=int(image.max()))
        else:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Preprocessing and Contrast Pipeline | {prepared.bundle.map_spec.label} idx={prepared.bundle.index} "
        f"PC=({prepared.bundle.pc[0]:.4f}, {prepared.bundle.pc[1]:.4f}, {prepared.bundle.pc[2]:.4f})",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_sphere(ax, alpha: float = 0.08) -> None:
    lon = np.linspace(-np.pi, np.pi, 48)
    colat = np.linspace(0.0, np.pi, 24)
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)
    ax.plot_surface(
        x,
        y,
        z,
        color="#b8c2cc",
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        alpha=alpha,
        shade=False,
    )


def view_angles_from_vectors(vectors: np.ndarray) -> tuple[float, float]:
    center = np.nanmean(vectors.reshape(-1, 3), axis=0)
    norm = float(np.linalg.norm(center))
    if not np.isfinite(norm) or norm <= 1e-8:
        return 20.0, -58.0
    center = center / norm
    elev = float(np.degrees(np.arcsin(np.clip(center[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(center[1], center[0])))
    return elev, azim


def set_3d_sphere_axes(ax, title: str, view_vectors: np.ndarray | None = None) -> None:
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    elev, azim = view_angles_from_vectors(view_vectors) if view_vectors is not None else (20.0, -58.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def detector_raw_display(prepared) -> np.ndarray:
    return percentile_normalize(
        prepared.bundle.pattern_u16.astype(np.float32),
        prepared.valid_mask,
        low=0.2,
        high=99.8,
    )


def gray_facecolors(image01: np.ndarray, mask: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    colors = cm.gray(np.clip(image01, 0.0, 1.0))
    colors[..., 3] = np.where(mask, alpha, 0.0).astype(np.float32)
    return colors


def patch_surface_step(shape: tuple[int, int], target_samples: int = 320) -> int:
    return max(1, int(math.ceil(max(shape) / float(target_samples))))


def matched_points_grid(result: MatchResult) -> np.ndarray:
    prepared = result.prepared
    full_points = prepared.full_points_grid.reshape(-1, 3) @ result.detector_transform.T
    return result.rotation.apply(full_points).reshape(prepared.full_points_grid.shape)


def transformed_band_curves(result: MatchResult) -> list[np.ndarray]:
    prepared = result.prepared
    curves = []
    for segment in prepared.line_segments:
        curve = segment_curve_vectors(prepared, segment)
        curve = curve @ result.detector_transform.T
        curves.append(result.rotation.apply(curve))
    return curves


def detector_band_curves(prepared) -> list[np.ndarray]:
    return [segment_curve_vectors(prepared, segment) for segment in prepared.line_segments]


def plot_pattern_patch(
    ax,
    points_grid: np.ndarray,
    texture01: np.ndarray,
    mask: np.ndarray,
    title: str,
    curves: list[np.ndarray] | None = None,
    draw_reference_sphere: bool = True,
    radius_scale: float = 1.0,
) -> None:
    if draw_reference_sphere:
        plot_reference_sphere(ax, alpha=0.055)
    plot_points = points_grid.copy()
    plot_points[~mask] = np.nan
    plot_points *= radius_scale
    step = patch_surface_step(mask.shape)
    ax.plot_surface(
        plot_points[::step, ::step, 0],
        plot_points[::step, ::step, 1],
        plot_points[::step, ::step, 2],
        facecolors=gray_facecolors(texture01, mask)[::step, ::step],
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )
    if curves:
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(curves)))
        for color, curve in zip(colors, curves):
            curve = curve * (radius_scale * 1.055)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=2.7, alpha=0.98)
    set_3d_sphere_axes(ax, title, view_vectors=points_grid[mask])


def plot_master_sphere(ax, master: MasterSphere, lon_count: int, colat_count: int, alpha: float = 0.58) -> None:
    master_map, lon_grid, colat_grid, _ = sphere_texture(master, lon_count, colat_count)
    sphere_x = np.sin(colat_grid) * np.cos(lon_grid)
    sphere_y = np.sin(colat_grid) * np.sin(lon_grid)
    sphere_z = np.cos(colat_grid)
    sphere_colors = cm.gray(master_map)
    sphere_colors[..., 3] = alpha
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        facecolors=sphere_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )


def plot_matched_patch_on_master(
    ax,
    result: MatchResult,
    master: MasterSphere,
    texture01: np.ndarray,
    title: str,
    lon_count: int,
    colat_count: int,
) -> None:
    prepared = result.prepared
    plot_master_sphere(ax, master, lon_count, colat_count)
    plot_pattern_patch(
        ax,
        matched_points_grid(result),
        texture01,
        prepared.valid_mask,
        title,
        curves=transformed_band_curves(result),
        draw_reference_sphere=False,
        radius_scale=1.018,
    )


def save_sphere_correction_visualization(prepared, products: dict[str, np.ndarray], out_path: Path) -> None:
    raw_display = detector_raw_display(prepared)
    processed = products["contrast_enhanced"]
    curves = detector_band_curves(prepared)

    fig = plt.figure(figsize=(14.8, 11.2))
    ax0 = fig.add_subplot(221)
    ax0.imshow(raw_display, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Raw detector pattern")
    ax0.axis("off")

    ax1 = fig.add_subplot(222)
    ax1.imshow(processed, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Preprocessed detector pattern")
    ax1.axis("off")

    ax2 = fig.add_subplot(223, projection="3d")
    plot_pattern_patch(
        ax2,
        prepared.full_points_grid,
        raw_display,
        prepared.valid_mask,
        "Raw pattern corrected to detector sphere",
        curves=curves,
    )

    ax3 = fig.add_subplot(224, projection="3d")
    plot_pattern_patch(
        ax3,
        prepared.full_points_grid,
        processed,
        prepared.valid_mask,
        "Preprocessed pattern corrected to detector sphere",
        curves=curves,
    )
    fig.suptitle(
        f"Pattern corrected to sphere | {prepared.bundle.map_spec.label} idx={prepared.bundle.index} "
        f"PC=({prepared.bundle.pc[0]:.4f}, {prepared.bundle.pc[1]:.4f}, {prepared.bundle.pc[2]:.4f})",
        y=0.985,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def equirect_line(ax, curve: np.ndarray, color, linewidth: float = 1.35) -> None:
    lon = np.degrees(np.arctan2(curve[:, 1], curve[:, 0]))
    colat = np.degrees(np.arccos(np.clip(curve[:, 2], -1.0, 1.0)))
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for part in np.split(np.arange(len(lon)), jumps):
        if len(part) >= 2:
            ax.plot(lon[part], colat[part], color=color, linewidth=linewidth)


def save_final_spatial_visualization(
    result: MatchResult,
    master: MasterSphere,
    products: dict[str, np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    prepared = result.prepared
    raw_display = detector_raw_display(prepared)

    fig = plt.figure(figsize=(15.8, 7.4))
    ax0 = fig.add_subplot(121, projection="3d")
    plot_matched_patch_on_master(
        ax0,
        result,
        master,
        raw_display,
        "Raw pattern at final matched position",
        lon_count,
        colat_count,
    )

    ax1 = fig.add_subplot(122, projection="3d")
    plot_matched_patch_on_master(
        ax1,
        result,
        master,
        products["contrast_enhanced"],
        "Preprocessed pattern at the same matched position",
        lon_count,
        colat_count,
    )
    fig.suptitle(
        f"Final single spatial match on high-resolution Kikuchi sphere | "
        f"{result.label}, score={result.score:.4f}, convention={result.convention_name}",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_single_match_visualization(
    result: MatchResult,
    master: MasterSphere,
    products: dict[str, np.ndarray],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    prepared = result.prepared
    raw_display = detector_raw_display(prepared)
    fig = plt.figure(figsize=(18.0, 6.8))

    ax0 = fig.add_subplot(131)
    ax0.imshow(products["combined_response"] if "combined_response" in products else prepared.combined_response, cmap="gray", vmin=0, vmax=1)
    ax0.contour(prepared.match_mask.astype(np.float32), levels=[0.5], colors=["#ffcc33"], linewidths=0.7)
    ax0.set_title("Detector match response and selected points")
    ax0.axis("off")

    ax1 = fig.add_subplot(132, projection="3d")
    plot_matched_patch_on_master(
        ax1,
        result,
        master,
        raw_display,
        "Raw pattern projected after matching",
        lon_count,
        colat_count,
    )

    ax2 = fig.add_subplot(133, projection="3d")
    plot_matched_patch_on_master(
        ax2,
        result,
        master,
        products["contrast_enhanced"],
        "Preprocessed pattern projected after matching",
        lon_count,
        colat_count,
    )

    fig.suptitle(
        f"Spherical matching visualization | one H5-band-enhanced match, score={result.score:.4f}",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_refine_schedule(text: str) -> list[tuple[float, int]]:
    schedule: list[tuple[float, int]] = []
    if not text.strip():
        return schedule
    for item in text.split(","):
        step, attempts = item.split(":")
        schedule.append((float(step), int(attempts)))
    return schedule


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step-by-step EBSD pattern-center and Kikuchi-sphere calibration visualization.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "spherical_calibration_steps")
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
    parser.add_argument("--sphere-lon-count", type=int, default=640)
    parser.add_argument("--sphere-colat-count", type=int, default=320)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.45)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_specs = default_map_specs(args.data_dir)
    map_spec = map_specs[args.map]
    master_h5 = resolve_master_path(args.master_h5)
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading pattern and H5 metadata: {map_spec.label}, index={args.index}")
    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    products = build_preprocessing_products(
        bundle.pattern_u16,
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    baseline_weights = MatchWeights(image_line=0.75, intensity=0.25, h5_band=0.0)
    enhanced_weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )
    base_prepared, _ = prepare_pattern(
        bundle=bundle,
        weights=baseline_weights,
        label="image-only",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name="auto",
    )
    enhanced_prepared, variant_diagnostics = prepare_pattern(
        bundle=bundle,
        weights=enhanced_weights,
        label="H5-band-enhanced",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=base_prepared.line_variant.name,
    )

    paths = {
        "preprocessing": out_dir / "01_preprocessing_contrast.png",
        "sphere_correction": out_dir / "02_pattern_corrected_to_sphere.png",
        "sphere_matching": out_dir / "03_sphere_matching_comparison.png",
        "score_comparison": out_dir / "04_matching_score_comparison.png",
        "final_spatial": out_dir / "05_final_pattern_and_sphere_spatial_position.png",
        "summary": out_dir / "summary.json",
    }

    save_preprocessing_visualization(enhanced_prepared, products, paths["preprocessing"])
    save_sphere_correction_visualization(
        enhanced_prepared,
        products,
        paths["sphere_correction"],
    )

    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    refine_schedule = parse_refine_schedule(args.refine_schedule)
    print("Matching image-only baseline...")
    base_result = match_to_master(
        base_prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=refine_schedule,
        random_seed=args.seed,
    )
    print("Matching H5-band-enhanced...")
    enhanced_result = match_to_master(
        enhanced_prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=refine_schedule,
        random_seed=args.seed,
    )
    results = [base_result, enhanced_result]

    save_single_match_visualization(
        enhanced_result,
        master,
        products,
        paths["sphere_matching"],
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_score_comparison(results, paths["score_comparison"])
    save_final_spatial_visualization(
        enhanced_result,
        master,
        products,
        paths["final_spatial"],
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    summary = {
        "map": map_spec.key,
        "map_label": map_spec.label,
        "index": bundle.index,
        "row": bundle.row,
        "col": bundle.col,
        "pattern_shape": list(bundle.pattern_u16.shape),
        "pattern_center_from_h5": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": enhanced_prepared.line_variant.name,
        "line_variant_score": enhanced_prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "match_results": [result.to_json_dict() for result in results],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved step-by-step visualizations to: {out_dir}")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
