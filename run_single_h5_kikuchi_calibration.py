from __future__ import annotations

import argparse
import json
from dataclasses import replace
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import ndimage as ndi

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    ProjectionInputs,
    build_master_lon_colat,
    estimate_circular_detector_mask,
    project_patch_to_lon_colat,
    read_edax_inputs,
)
from spherical_finetune_pipeline import (
    ScoringSamplers,
    Score,
    build_parameter_layout,
    build_scoring_samplers,
    crystal_vectors_for_params,
    fit_parameters,
    full_resolution_vectors,
    prepare_images,
    sample_refined_master_on_detector,
    save_candidate_score_figure,
    save_detector_residual_figure,
    save_trace_csv,
    save_trace_figure,
    score_vectors,
    select_initial_candidate,
    unpack_params,
)
from visualize_scan_position_pc_correction import (
    ScanGeometry,
    adjusted_pc_from_scan_position,
    index_to_scan_offset_um,
    read_scan_geometry,
)


DEFAULT_H5 = Path(r"F:\kikuchi-super resolution\20260512Cu resolution-contrast.edaxh5")
DEFAULT_UP2 = Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2")
DEFAULT_MAP_GROUP = "/20260512/Cu/Area 1/OIM Map 1HighR"
DEFAULT_PATTERN_INDEX = 28676
DEFAULT_MASTER = (
    Path(__file__).resolve().parents[1]
    / ".venv"
    / "Lib"
    / "site-packages"
    / "kikuchipy"
    / "data"
    / "emsoft_ebsd_master_pattern"
    / "ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)


def normalize01(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    if mask is not None and np.any(mask):
        lo, hi = np.nanpercentile(values[mask], [1.0, 99.0])
    else:
        lo, hi = np.nanpercentile(values, [1.0, 99.0])
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def scalar(group: h5py.Group, path: str) -> Any:
    value = group[path][()]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_ang_metadata(h5_path: Path, map_group: str, index: int) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        group = h5[map_group]
        data = group["EBSD/ANG/DATA/DATA"]
        record = data[index]
        rows = int(scalar(group, "Sample/Number Of Rows"))
        cols = int(scalar(group, "Sample/Number Of Columns"))
        out: dict[str, Any] = {
            "rows": rows,
            "cols": cols,
            "row": int(index // cols),
            "col": int(index % cols),
        }
        for key in ("IQ", "CI", "Phase", "Fit", "SEM Signal", "Valid"):
            if key in (record.dtype.names or ()):
                value = record[key]
                if np.issubdtype(np.asarray(value).dtype, np.integer):
                    out[key] = int(value)
                else:
                    out[key] = float(value)
        return out


def detector_directions_and_internal_pc(
    projection: ProjectionInputs,
    pc_edax: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    detector = EBSDDetector(
        shape=projection.shape,
        pc=pc_edax,
        convention="edax",
        tilt=projection.camera_elevation,
        azimuthal=projection.camera_azimuthal,
        sample_tilt=projection.sample_tilt,
    )
    directions = _get_direction_cosines_from_detector(detector)
    return directions, tuple(float(v) for v in detector.pc.squeeze())


def projection_with_pc(
    projection: ProjectionInputs,
    pc_edax: tuple[float, float, float],
) -> ProjectionInputs:
    directions, pc_internal = detector_directions_and_internal_pc(projection, pc_edax)
    return replace(
        projection,
        pc_edax=pc_edax,
        pc_internal=pc_internal,
        detector_directions=directions,
    )


def circular_mask_from_circle(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def save_preprocess_figure(
    projection: ProjectionInputs,
    mask: np.ndarray,
    circle: tuple[int, int, int],
    prepared,
    output_path: Path,
) -> None:
    raw_norm = normalize01(projection.pattern, mask)
    corrected_norm = normalize01(prepared.corrected, mask)
    masked_corrected = np.where(mask, corrected_norm, np.nan)
    band_norm = normalize01(prepared.band, mask)

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.4))
    panels = [
        (raw_norm, "Raw pattern normalized", "gray"),
        (raw_norm, f"Circular detector mask\ncx={circle[0]}, cy={circle[1]}, r={circle[2]}", "gray"),
        (masked_corrected, "Background normalize + mask", "gray"),
        (np.where(mask, band_norm, np.nan), "Band-enhanced for scoring", "magma"),
    ]
    for ax, (image, title, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        if "mask" in title.lower() and "Background" not in title:
            ax.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
        ax.set_title(title)
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight", transparent=True)
    plt.close(fig)


def build_lon_colat_vectors(lon_count: int, colat_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.linspace(-np.pi, np.pi, lon_count)
    colat = np.linspace(0.0, np.pi, colat_count)
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    vectors = np.column_stack(
        [
            np.sin(colat_grid).ravel() * np.cos(lon_grid).ravel(),
            np.sin(colat_grid).ravel() * np.sin(lon_grid).ravel(),
            np.cos(colat_grid).ravel(),
        ]
    )
    return lon_grid, colat_grid, vectors


def save_software_projection_figure(
    master_texture: np.ndarray,
    vectors: np.ndarray,
    values: np.ndarray,
    score: Score,
    pc_edax: tuple[float, float, float],
    matrix_name: str,
    output_path: Path,
) -> None:
    patch, patch_mask, _ = project_patch_to_lon_colat(vectors, values)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2), sharex=True, sharey=True)
    axes[0].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[0].set_title("Standard master Kikuchi sphere")
    axes[1].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[1].imshow(
        patch,
        cmap="magma",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=np.where(patch_mask, 0.88, 0.0),
    )
    axes[1].set_title(
        f"Software orientation projection\n{matrix_name}, score={score.combined:+.4f}"
    )
    for ax in axes:
        ax.set_xlabel("longitude on crystal sphere (deg)")
        ax.set_ylabel("colatitude (deg)")
    fig.text(
        0.01,
        0.01,
        f"scan-position PC used as initial geometry: ({pc_edax[0]:.6f}, {pc_edax[1]:.6f}, {pc_edax[2]:.6f})",
        fontsize=8.2,
        family="monospace",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_refined_sphere_overlay_figure(
    master_texture: np.ndarray,
    initial_vectors: np.ndarray,
    refined_vectors: np.ndarray,
    values: np.ndarray,
    initial_score: Score,
    refined_score: Score,
    output_path: Path,
) -> None:
    initial_patch, initial_mask, _ = project_patch_to_lon_colat(initial_vectors, values)
    refined_patch, refined_mask, _ = project_patch_to_lon_colat(refined_vectors, values)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4), sharex=True, sharey=True)
    for ax, patch, patch_mask, title in [
        (axes[0], initial_patch, initial_mask, f"Initial\nscore={initial_score.combined:+.4f}"),
        (axes[1], refined_patch, refined_mask, f"Fine-tuned\nscore={refined_score.combined:+.4f}"),
    ]:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        ax.imshow(
            patch,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(patch_mask, 0.88, 0.0),
        )
        ax.set_title(title)
        ax.set_xlabel("longitude on crystal sphere (deg)")
        ax.set_ylabel("colatitude (deg)")

    overlay = np.zeros((*initial_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = np.where(initial_mask, 1.0, 0.0)
    overlay[..., 2] = np.where(refined_mask, 1.0, 0.0)
    overlay[..., 3] = np.where(initial_mask | refined_mask, 0.72, 0.0)
    axes[2].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[2].imshow(overlay, origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[2].set_title("Patch footprint shift\ninitial red, refined blue")
    axes[2].set_xlabel("longitude on crystal sphere (deg)")
    axes[2].set_ylabel("colatitude (deg)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def direction_to_lon_colat(direction: np.ndarray) -> tuple[float, float]:
    d = direction / (np.linalg.norm(direction) + 1e-12)
    lon = degrees(atan2(float(d[1]), float(d[0])))
    colat = degrees(np.arccos(np.clip(float(d[2]), -1.0, 1.0)))
    return lon, colat


def save_geometry_inversion_figure(
    geometry: ScanGeometry,
    map_pc: tuple[float, float, float],
    scan_pc: tuple[float, float, float],
    refined_pc: tuple[float, float, float],
    best_params: dict[str, float],
    matrix_name: str,
    final_matrix: np.ndarray,
    initial_score: Score,
    refined_score: Score,
    x_um: float,
    y_um: float,
    metadata: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    sample_axes = {
        "RD_sample": np.array([1.0, 0.0, 0.0]),
        "TD_sample": np.array([0.0, 1.0, 0.0]),
        "ND_sample": np.array([0.0, 0.0, 1.0]),
    }
    crystal_axes = {name: axis @ final_matrix for name, axis in sample_axes.items()}
    crystal_axes = {
        name: vec / (np.linalg.norm(vec) + 1e-12)
        for name, vec in crystal_axes.items()
    }
    axis_rows = {
        name: {
            "xyz": [float(v) for v in vec],
            "lon_colat_deg": [float(v) for v in direction_to_lon_colat(vec)],
        }
        for name, vec in crystal_axes.items()
    }

    fig = plt.figure(figsize=(14.0, 7.6))
    ax_text = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_text.axis("off")

    pc_shift = np.array(scan_pc) - np.array(map_pc)
    refine_shift = np.array(refined_pc) - np.array(scan_pc)
    lines = [
        "Detector -> sample -> crystal inverse geometry",
        "",
        f"map row,col: ({metadata['row']}, {metadata['col']}) / index {metadata.get('index', '')}",
        f"scan offset: x={x_um:+.3f} um, y={y_um:+.3f} um",
        f"step: x={geometry.step_x_um:.4f} um, y={geometry.step_y_um:.4f} um",
        f"detector diameter: {geometry.detector_diameter_mm:.4f} mm",
        "",
        f"map PC:      ({map_pc[0]:.7f}, {map_pc[1]:.7f}, {map_pc[2]:.7f})",
        f"scan PC:     ({scan_pc[0]:.7f}, {scan_pc[1]:.7f}, {scan_pc[2]:.7f})",
        f"scan shift:  ({pc_shift[0]:+.7f}, {pc_shift[1]:+.7f}, {pc_shift[2]:+.7f})",
        f"refined PC:  ({refined_pc[0]:.7f}, {refined_pc[1]:.7f}, {refined_pc[2]:.7f})",
        f"fine shift:  ({refine_shift[0]:+.7f}, {refine_shift[1]:+.7f}, {refine_shift[2]:+.7f})",
        "",
        f"matrix convention: {matrix_name}",
        (
            "delta rotation: "
            f"rx={best_params['rot_x_deg']:+.5f} deg, "
            f"ry={best_params['rot_y_deg']:+.5f} deg, "
            f"rz={best_params['rot_z_deg']:+.5f} deg"
        ),
        (
            "PC delta params: "
            f"dpcx={best_params['dpcx']:+.7f}, "
            f"dpcy={best_params['dpcy']:+.7f}, "
            f"dpcz={best_params['dpcz']:+.7f}"
        ),
        f"score: {initial_score.combined:+.5f} -> {refined_score.combined:+.5f}",
        "",
        "Sample axes expressed on the crystal/master sphere:",
    ]
    for name, item in axis_rows.items():
        lon, colat = item["lon_colat_deg"]
        xyz = item["xyz"]
        lines.append(
            f"{name}: xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}), "
            f"lon={lon:+.2f}, colat={colat:.2f}"
        )
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.3)

    u = np.linspace(0, 2 * np.pi, 52)
    v = np.linspace(0, np.pi, 26)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax_3d.plot_wireframe(x, y, z, color="#c8c8c8", linewidth=0.35, alpha=0.45)
    axis_colors = {"RD_sample": "#d62728", "TD_sample": "#2ca02c", "ND_sample": "#1f77b4"}
    for name, vec in crystal_axes.items():
        ax_3d.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=0.95, color=axis_colors[name], linewidth=2.4)
        ax_3d.text(vec[0] * 1.05, vec[1] * 1.05, vec[2] * 1.05, name[:2], color=axis_colors[name])
    ax_3d.set_title("Recovered sample axes on crystal sphere")
    ax_3d.set_box_aspect((1, 1, 1))
    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlim(-1, 1)
    ax_3d.set_axis_off()
    ax_3d.view_init(elev=22, azim=42)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return axis_rows


def save_3d_master_patch_figure(
    samplers: ScoringSamplers,
    refined_vectors: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    lon_count, colat_count = 240, 120
    lon_grid, colat_grid, _vectors = build_lon_colat_vectors(lon_count, colat_count)
    master_texture = build_master_lon_colat(
        samplers.upper_visual,
        samplers.lower_visual,
        lon_count=lon_count,
        colat_count=colat_count,
    )
    master_norm = normalize01(master_texture)
    master_rgba = cm.gray(master_norm)
    master_rgba[..., 3] = 0.42

    patch, patch_mask, _ = project_patch_to_lon_colat(
        refined_vectors,
        normalize01(values),
        lon_count=lon_count,
        colat_count=colat_count,
    )
    if np.any(patch_mask):
        filled = patch.copy()
        fill_values = ndi.grey_dilation(filled, size=(3, 3))
        filled[~patch_mask] = fill_values[~patch_mask]
    else:
        filled = patch
    patch_rgba = cm.magma(normalize01(filled, patch_mask if np.any(patch_mask) else None))
    patch_rgba[..., 3] = np.where(patch_mask, 0.90, 0.0)

    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)

    centroid = refined_vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    elev = degrees(np.arcsin(float(centroid[2])))
    azim = degrees(np.arctan2(float(centroid[1]), float(centroid[0])))

    if len(values) > max_points:
        rng = np.random.default_rng(11)
        keep = np.sort(rng.choice(len(values), size=max_points, replace=False))
    else:
        keep = np.arange(len(values))

    fig = plt.figure(figsize=(10.6, 9.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x, y, z, facecolors=master_rgba, rstride=1, cstride=1, linewidth=0, shade=False)
    ax.plot_surface(
        x * 1.012,
        y * 1.012,
        z * 1.012,
        facecolors=patch_rgba,
        rstride=1,
        cstride=1,
        linewidth=0,
        shade=False,
    )
    ax.scatter(
        refined_vectors[keep, 0] * 1.02,
        refined_vectors[keep, 1] * 1.02,
        refined_vectors[keep, 2] * 1.02,
        c=values[keep],
        cmap="magma",
        s=0.55,
        alpha=0.45,
        depthshade=False,
    )
    ax.set_title("Fine-tuned experimental Kikuchi pattern on master sphere")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight", transparent=True)
    plt.close(fig)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    geometry: ScanGeometry,
    map_pc: tuple[float, float, float],
    scan_pc: tuple[float, float, float],
    refined_pc: tuple[float, float, float],
    x_um: float,
    y_um: float,
    matrix_name: str,
    initial_score: Score,
    refined_score: Score,
    best_params: dict[str, float],
    axis_rows: dict[str, Any],
) -> None:
    summary = {
        "inputs": {
            "h5": str(args.h5),
            "up2": str(args.up2),
            "map_group": args.map_group,
            "pattern_index": args.pattern_index,
            "master": str(args.master),
            "raw_pattern_note": "The EDAX H5 stores the map metadata and index; the raw pattern bytes are read from the matched UP2 sidecar file.",
        },
        "pattern_metadata": metadata,
        "scan_geometry": {
            "ncols": geometry.ncols,
            "nrows": geometry.nrows,
            "step_x_um": geometry.step_x_um,
            "step_y_um": geometry.step_y_um,
            "detector_diameter_mm": geometry.detector_diameter_mm,
            "grid_type": geometry.grid_type,
            "scan_offset_um": [x_um, y_um],
        },
        "pc": {
            "map_pc_edax": list(map_pc),
            "scan_position_pc_edax": list(scan_pc),
            "refined_pc_edax": list(refined_pc),
            "scan_pc_model": {
                "x_sign": args.pc_x_sign,
                "y_sign": args.pc_y_sign,
                "y_scale": args.pc_y_scale,
                "z_sign": args.pc_z_sign,
                "z_scale": args.pc_z_scale,
            },
        },
        "fit": {
            "matrix_name": matrix_name,
            "params": best_params,
            "initial_score": initial_score.__dict__,
            "refined_score": refined_score.__dict__,
            "score_gain": refined_score.combined - initial_score.combined,
            "settings": {
                "stride": args.stride,
                "rotation_bound_deg": args.rotation_bound_deg,
                "pc_xy_bound": args.pc_xy_bound,
                "pc_z_bound": args.pc_z_bound,
                "global_iter": args.global_iter,
                "population": args.population,
                "local_maxiter": args.local_maxiter,
            },
        },
        "inverted_geometry": {
            "sample_axes_expressed_on_crystal_sphere": axis_rows,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_ang_metadata(args.h5, args.map_group, args.pattern_index)
    metadata["index"] = args.pattern_index

    raw_projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        )
    )
    geometry = read_scan_geometry(args.h5, args.map_group)
    map_pc = geometry.pc_edax_map
    x_um, y_um = index_to_scan_offset_um(args.pattern_index, geometry)
    scan_pc = adjusted_pc_from_scan_position(
        args.pattern_index,
        geometry,
        x_sign=args.pc_x_sign,
        y_sign=args.pc_y_sign,
        y_scale=args.pc_y_scale,
        z_sign=args.pc_z_sign,
        z_scale=args.pc_z_scale,
    )
    projection = projection_with_pc(raw_projection, scan_pc)

    mask, circle = estimate_circular_detector_mask(projection.pattern)
    mask = circular_mask_from_circle(projection.shape, circle)
    prepared = prepare_images(projection.pattern, mask, stride=args.stride)
    samplers = build_scoring_samplers(args.master, args.master_display)
    master_texture = build_master_lon_colat(samplers.upper_visual, samplers.lower_visual)

    save_preprocess_figure(
        projection=projection,
        mask=mask,
        circle=circle,
        prepared=prepared,
        output_path=args.output_dir / "00_preprocess_normalize_circular_mask.png",
    )

    matrix_name, base_matrix, candidate_rows = select_initial_candidate(
        projection=projection,
        prepared=prepared,
        samplers=samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
        requested_name=args.matrix,
    )
    save_candidate_score_figure(candidate_rows, args.output_dir / "01_orientation_matrix_candidate_scores.png")

    names, bounds, scales = build_parameter_layout(args)
    best_x, refined_score, trace = fit_parameters(
        projection=projection,
        prepared=prepared,
        samplers=samplers,
        base_matrix=base_matrix,
        names=names,
        bounds=bounds,
        scales=scales,
        args=args,
    )
    best_params = unpack_params(names, best_x)
    initial_params = unpack_params(names, np.zeros(len(names), dtype=np.float64))

    initial_eval_vectors, _ = crystal_vectors_for_params(
        projection=projection,
        indices=prepared.indices,
        base_matrix=base_matrix,
        params=initial_params,
    )
    initial_score = score_vectors(
        vectors=initial_eval_vectors,
        prepared=prepared,
        samplers=samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    initial_full_vectors, full_indices, _initial_pc = full_resolution_vectors(
        projection=projection,
        mask=mask,
        base_matrix=base_matrix,
        params=initial_params,
    )
    refined_full_vectors, refined_indices, refined_pc = full_resolution_vectors(
        projection=projection,
        mask=mask,
        base_matrix=base_matrix,
        params=best_params,
    )
    if not np.array_equal(full_indices, refined_indices):
        raise RuntimeError("Internal error: full-resolution index set changed during refinement")

    values = prepared.corrected[mask]
    save_software_projection_figure(
        master_texture=master_texture,
        vectors=initial_full_vectors,
        values=values,
        score=initial_score,
        pc_edax=scan_pc,
        matrix_name=matrix_name,
        output_path=args.output_dir / "02_software_orientation_to_master_sphere.png",
    )

    save_trace_csv(trace, args.output_dir / "03_finetune_trace.csv")
    save_trace_figure(trace, args.output_dir / "03_finetune_trace.png")
    save_refined_sphere_overlay_figure(
        master_texture=master_texture,
        initial_vectors=initial_full_vectors,
        refined_vectors=refined_full_vectors,
        values=values,
        initial_score=initial_score,
        refined_score=refined_score,
        output_path=args.output_dir / "04_finetuned_master_sphere_overlay.png",
    )

    refined_master_corr, refined_master_band = sample_refined_master_on_detector(
        shape=projection.shape,
        indices=full_indices,
        vectors=refined_full_vectors,
        samplers=samplers,
    )
    save_detector_residual_figure(
        prepared=prepared,
        refined_master_corr=refined_master_corr,
        refined_master_band=refined_master_band,
        mask=mask,
        output_path=args.output_dir / "05_forward_detector_residual.png",
    )

    final_delta = np.eye(3)
    if any(abs(best_params[name]) > 0 for name in ("rot_x_deg", "rot_y_deg", "rot_z_deg")):
        from spherical_finetune_pipeline import params_to_rotation

        final_delta = params_to_rotation(best_params)
    final_matrix = base_matrix @ final_delta.T
    axis_rows = save_geometry_inversion_figure(
        geometry=geometry,
        map_pc=map_pc,
        scan_pc=scan_pc,
        refined_pc=refined_pc,
        best_params=best_params,
        matrix_name=matrix_name,
        final_matrix=final_matrix,
        initial_score=initial_score,
        refined_score=refined_score,
        x_um=x_um,
        y_um=y_um,
        metadata=metadata,
        output_path=args.output_dir / "06_inverted_geometry_relationship.png",
    )

    save_3d_master_patch_figure(
        samplers=samplers,
        refined_vectors=refined_full_vectors,
        values=values,
        output_path=args.output_dir / "07_3d_kikuchi_on_master_sphere.png",
        max_points=args.max_3d_points,
    )

    np.savez_compressed(
        args.output_dir / "refined_single_pattern_mapping.npz",
        valid_mask=mask,
        initial_vectors=initial_full_vectors.astype(np.float32),
        refined_vectors=refined_full_vectors.astype(np.float32),
        corrected_values=values.astype(np.float32),
        full_indices=full_indices.astype(np.int64),
        refined_master_corrected=refined_master_corr.astype(np.float32),
        refined_master_band=refined_master_band.astype(np.float32),
    )
    write_summary(
        output_path=args.output_dir / "summary.json",
        args=args,
        metadata=metadata,
        geometry=geometry,
        map_pc=map_pc,
        scan_pc=scan_pc,
        refined_pc=refined_pc,
        x_um=x_um,
        y_um=y_um,
        matrix_name=matrix_name,
        initial_score=initial_score,
        refined_score=refined_score,
        best_params=best_params,
        axis_rows=axis_rows,
    )

    print(f"Saved single-pattern calibration outputs: {args.output_dir}")
    print(f"Pattern index: {args.pattern_index}, row={metadata['row']}, col={metadata['col']}")
    print(f"IQ={metadata.get('IQ')}, CI={metadata.get('CI')}, Phase={metadata.get('Phase')}")
    print(f"Scan offset: x={x_um:+.3f} um, y={y_um:+.3f} um")
    print(f"Map PC: {tuple(round(v, 7) for v in map_pc)}")
    print(f"Scan-position PC: {tuple(round(v, 7) for v in scan_pc)}")
    print(f"Refined PC: {tuple(round(v, 7) for v in refined_pc)}")
    print(f"Selected matrix: {matrix_name}")
    print(f"Initial combined score: {initial_score.combined:+.6f}")
    print(f"Refined combined score: {refined_score.combined:+.6f}")
    print(f"Score gain: {refined_score.combined - initial_score.combined:+.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one EDAX Kikuchi pattern through PC-adjusted spherical calibration and visualization."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2", type=Path, default=DEFAULT_UP2)
    parser.add_argument("--map-group", default=DEFAULT_MAP_GROUP)
    parser.add_argument("--pattern-index", type=int, default=DEFAULT_PATTERN_INDEX)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "single_h5_kikuchi_calibration" / "area1_high_idx28676",
    )
    parser.add_argument("--matrix", default="auto")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--regularization-weight", type=float, default=0.012)
    parser.add_argument("--rotation-bound-deg", type=float, default=1.5)
    parser.add_argument("--pc-xy-bound", type=float, default=0.004)
    parser.add_argument("--pc-z-bound", type=float, default=0.008)
    parser.add_argument("--radius-bound", type=float, default=0.0)
    parser.add_argument("--global-iter", type=int, default=5)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--global-tol", type=float, default=1e-3)
    parser.add_argument("--local-maxiter", type=int, default=120)
    parser.add_argument("--local-ftol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-3d-points", type=int, default=22000)
    parser.add_argument("--master-display", choices=("raw", "corrected", "band"), default="band")

    parser.add_argument("--pc-x-sign", type=float, default=-1.0)
    parser.add_argument("--pc-y-sign", type=float, default=1.0)
    parser.add_argument("--pc-y-scale", type=float, default=cos(radians(70.0)))
    parser.add_argument("--pc-z-sign", type=float, default=1.0)
    parser.add_argument("--pc-z-scale", type=float, default=sin(radians(70.0)))

    parser.set_defaults(fit_rotation=True, fit_pcxy=True, fit_pcz=True, fit_radius=False, write_csv=False)
    parser.add_argument("--fit-rotation", dest="fit_rotation", action="store_true")
    parser.add_argument("--no-fit-rotation", dest="fit_rotation", action="store_false")
    parser.add_argument("--fit-pcxy", dest="fit_pcxy", action="store_true")
    parser.add_argument("--no-fit-pcxy", dest="fit_pcxy", action="store_false")
    parser.add_argument("--fit-pcz", dest="fit_pcz", action="store_true")
    parser.add_argument("--no-fit-pcz", dest="fit_pcz", action="store_false")
    parser.add_argument("--fit-radius", dest="fit_radius", action="store_true")
    parser.add_argument("--no-fit-radius", dest="fit_radius", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    total_weight = args.intensity_weight + args.band_weight
    if total_weight <= 0:
        raise ValueError("At least one score weight must be positive")
    args.intensity_weight /= total_weight
    args.band_weight /= total_weight
    run(args)


if __name__ == "__main__":
    main()
