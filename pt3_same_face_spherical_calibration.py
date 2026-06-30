from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from skimage import exposure

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    build_master_lon_colat,
    read_edax_inputs,
)
from single_kikuchi_pc_finetune import (
    build_master_samplers,
    build_preprocessed_images,
    centered_circular_detector_mask,
    choose_orientation_matrix,
    detector_directions_with_pc,
    imshow_sphere,
    make_stride_indices,
    pc_finetune,
    project_crystal_patch,
    project_detector_patch,
    score_with_directions,
    write_pc_scores,
)


DEFAULT_H5 = Path(r"E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5")
DEFAULT_UP2_ROOT = Path(r"E:\ZHL\EBSD-RAW\20251209Pt")
DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt3_same_face_spherical_calibration"

# A conservative inner polygon on the same facet after rotating each SEM into
# the Area 3-360 in-plane frame.  Coordinates are SEM pixels in a 512 x 400 image.
DEFAULT_FACE_POLYGON_ALIGNED = np.array(
    [
        (295.0, 225.0),
        (430.0, 230.0),
        (430.0, 345.0),
        (335.0, 350.0),
        (280.0, 285.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Pt3MapSpec:
    label: str
    area: str
    inplane_angle_deg: float
    rotation_to_reference_deg: float
    up2_name: str
    color: str

    @property
    def h5_group(self) -> str:
        return f"20251209/Pt-3/{self.area}/OIM Map 1"


MAP_SPECS = [
    Pt3MapSpec(
        label="Pt-3 360 deg",
        area="Area 3-360",
        inplane_angle_deg=360.0,
        rotation_to_reference_deg=0.0,
        up2_name="20251209_Pt-3_Area 9_OIM Map 1.up2",
        color="#00e5ff",
    ),
    Pt3MapSpec(
        label="Pt-3 90 deg",
        area="Area 3-90",
        inplane_angle_deg=90.0,
        rotation_to_reference_deg=-90.0,
        up2_name="20251209_Pt-3_Area 4_OIM Map 1.up2",
        color="#ff3b30",
    ),
    Pt3MapSpec(
        label="Pt-3 180 deg",
        area="Area 3-180",
        inplane_angle_deg=180.0,
        rotation_to_reference_deg=-180.0,
        up2_name="20251209_Pt-3_Area 5_OIM Map 1.up2",
        color="#34c759",
    ),
    Pt3MapSpec(
        label="Pt-3 270 deg",
        area="Area 3-270",
        inplane_angle_deg=270.0,
        rotation_to_reference_deg=-270.0,
        up2_name="20251209_Pt-3_Area 7_OIM Map 1.up2",
        color="#ffcc00",
    ),
]


@dataclass
class ProcessedMap:
    spec: Pt3MapSpec
    selected_index: int
    selected_row: int
    selected_col: int
    selected_sem_xy: tuple[float, float]
    selected_aligned_xy: tuple[float, float]
    selected_iq: float
    selected_ci: float
    selected_phase: int
    candidate_count: int
    pc_original: tuple[float, float, float]
    pc_refined: tuple[float, float, float]
    pc_delta: tuple[float, float, float]
    orientation_variant: str
    orientation_matrix: np.ndarray
    original_score: float
    refined_score: float
    score_gain: float
    sem_gray: np.ndarray
    aligned_sem_gray: np.ndarray
    raw_poly: np.ndarray
    aligned_poly: np.ndarray
    raw_pattern_display: np.ndarray
    corrected_pattern: np.ndarray
    enhanced_pattern: np.ndarray
    band_pattern: np.ndarray
    detector_mask: np.ndarray
    detector_patch: tuple[np.ndarray, np.ndarray]
    original_patch: tuple[np.ndarray, np.ndarray]
    refined_patch: tuple[np.ndarray, np.ndarray]
    crystal_vectors: np.ndarray
    crystal_values: np.ndarray


def safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("__", "_")
    )


def normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def sem_rotation_matrix(shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    height, width = shape
    return cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0).astype(np.float64)


def scan_to_sem_xy(indices: np.ndarray, nrows: int, ncols: int, sem_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows = indices // ncols
    cols = indices % ncols
    sem_h, sem_w = sem_shape
    x = (cols + 0.5) / max(ncols, 1) * sem_w
    y = (rows + 0.5) / max(nrows, 1) * sem_h
    return x.astype(np.float64), y.astype(np.float64)


def row_rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    # Row-vector rotation about +z.
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def choose_inplane_prior_matrix(
    projection,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    master_samplers,
    reference_matrix: np.ndarray,
    rotation_to_reference_deg: float,
    args: argparse.Namespace,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    indices = make_stride_indices(mask, args.stride)
    detector_directions = detector_directions_with_pc(projection, projection.pc_edax)
    exp_corrected = images["enhanced"].ravel()[indices]
    exp_band = images["band"].ravel()[indices]

    angle = rotation_to_reference_deg
    row_rot = row_rotation_z(angle)
    all_candidates: dict[str, np.ndarray] = {
        f"inplane_prior_sample_pre_{angle:+.0f}deg": row_rot @ reference_matrix,
        f"inplane_prior_sample_pre_inv_{angle:+.0f}deg": row_rot.T @ reference_matrix,
        f"inplane_prior_crystal_post_{angle:+.0f}deg": reference_matrix @ row_rot,
        f"inplane_prior_crystal_post_inv_{angle:+.0f}deg": reference_matrix @ row_rot.T,
    }
    if args.inplane_prior_family == "auto_per_map":
        candidates = all_candidates
    else:
        wanted = f"inplane_prior_{args.inplane_prior_family}_{angle:+.0f}deg"
        candidates = {wanted: all_candidates[wanted]}

    rows: list[dict[str, object]] = []
    for name, matrix in candidates.items():
        intensity, band, combined = score_with_directions(
            detector_directions=detector_directions,
            matrix=matrix,
            indices=indices,
            exp_corrected_values=exp_corrected,
            exp_band_values=exp_band,
            samplers=master_samplers,
            intensity_weight=args.intensity_weight,
            band_weight=args.band_weight,
        )
        rows.append(
            {
                "orientation_variant": name,
                "intensity_score": intensity,
                "band_score": band,
                "combined_score": combined,
            }
        )

    best = max(rows, key=lambda row: float(row["combined_score"]))
    best_name = str(best["orientation_variant"])
    return best_name, candidates[best_name], rows


def select_same_face_high_quality_point(
    map_group: h5py.Group,
    rotation_to_reference_deg: float,
    face_polygon_aligned: np.ndarray,
    ci_min: float,
) -> dict[str, Any]:
    sem = normalize_gray(np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
    nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
    ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
    data = map_group["EBSD/ANG/DATA/DATA"][:]

    indices = np.arange(data.shape[0], dtype=np.int64)
    raw_x, raw_y = scan_to_sem_xy(indices, nrows, ncols, sem.shape)
    matrix = sem_rotation_matrix(sem.shape, rotation_to_reference_deg)
    aligned_xy = (matrix @ np.vstack([raw_x, raw_y, np.ones_like(raw_x)])).T
    inside = MplPath(face_polygon_aligned).contains_points(aligned_xy)

    valid = (
        inside
        & data["Valid"].astype(bool)
        & (data["Phase"] == 1)
        & np.isfinite(data["IQ"])
        & np.isfinite(data["CI"])
    )
    candidates = np.flatnonzero(valid & (data["CI"].astype(np.float64) >= ci_min))
    if candidates.size == 0:
        candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        raise RuntimeError("No valid EBSD points found inside the same-face ROI")

    iq = data["IQ"].astype(np.float64)
    ci = data["CI"].astype(np.float64)
    local_iq = iq[candidates]
    lo, hi = np.percentile(local_iq, [5, 99])
    iq_norm = np.clip((local_iq - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    centroid = face_polygon_aligned.mean(axis=0)
    distance = np.linalg.norm(aligned_xy[candidates] - centroid, axis=1)
    distance_norm = distance / max(float(distance.max()), 1e-6)
    score = 1.8 * iq_norm + 0.8 * np.clip(ci[candidates], 0.0, 1.0) - 0.25 * distance_norm
    selected_index = int(candidates[int(np.argmax(score))])

    row = selected_index // ncols
    col = selected_index % ncols
    selected_x, selected_y = scan_to_sem_xy(np.array([selected_index]), nrows, ncols, sem.shape)
    selected_xy = (float(selected_x[0]), float(selected_y[0]))
    selected_aligned = matrix @ np.array([selected_xy[0], selected_xy[1], 1.0], dtype=np.float64)
    inv_matrix = cv2.invertAffineTransform(matrix)
    raw_poly = (inv_matrix @ np.vstack([face_polygon_aligned.T, np.ones(len(face_polygon_aligned))])).T

    return {
        "sem_gray": sem,
        "aligned_sem_gray": cv2.warpAffine(sem, matrix, (sem.shape[1], sem.shape[0]), flags=cv2.INTER_LINEAR),
        "raw_poly": raw_poly,
        "aligned_poly": face_polygon_aligned.copy(),
        "selected_index": selected_index,
        "selected_row": int(row),
        "selected_col": int(col),
        "selected_sem_xy": selected_xy,
        "selected_aligned_xy": (float(selected_aligned[0]), float(selected_aligned[1])),
        "selected_iq": float(iq[selected_index]),
        "selected_ci": float(ci[selected_index]),
        "selected_phase": int(data["Phase"][selected_index]),
        "candidate_count": int(candidates.size),
    }


def crystal_vectors_for_patch(projection, pc: tuple[float, float, float], matrix: np.ndarray, values: np.ndarray, mask: np.ndarray, max_points: int):
    indices = np.flatnonzero(mask.ravel())
    step = max(1, int(math.ceil(indices.size / max_points)))
    indices = indices[::step]
    detector_directions = detector_directions_with_pc(projection, pc)
    vectors = detector_directions[indices] @ matrix
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors.astype(np.float32), values.ravel()[indices].astype(np.float32)


def process_one_map(
    spec: Pt3MapSpec,
    h5_path: Path,
    up2_root: Path,
    face_polygon_aligned: np.ndarray,
    master_samplers,
    output_dir: Path,
    args: argparse.Namespace,
    reference_matrix: np.ndarray | None = None,
) -> ProcessedMap:
    with h5py.File(h5_path, "r") as h5:
        selection = select_same_face_high_quality_point(
            map_group=h5[spec.h5_group],
            rotation_to_reference_deg=spec.rotation_to_reference_deg,
            face_polygon_aligned=face_polygon_aligned,
            ci_min=args.ci_min,
        )

    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=h5_path,
            up2_path=up2_root / spec.up2_name,
            map_group=spec.h5_group,
            pattern_index=selection["selected_index"],
        )
    )
    mask, _circle = centered_circular_detector_mask(projection.pattern.shape, args.mask_radius_fraction)
    images = build_preprocessed_images(projection.pattern, mask)
    if reference_matrix is None:
        orientation_name, orientation_matrix, orientation_rows = choose_orientation_matrix(
            projection=projection,
            mask=mask,
            images=images,
            samplers=master_samplers,
            stride=args.stride,
            intensity_weight=args.intensity_weight,
            band_weight=args.band_weight,
        )
        orientation_name = f"reference_h5_{orientation_name}"
    else:
        orientation_name, orientation_matrix, orientation_rows = choose_inplane_prior_matrix(
            projection=projection,
            mask=mask,
            images=images,
            master_samplers=master_samplers,
            reference_matrix=reference_matrix,
            rotation_to_reference_deg=spec.rotation_to_reference_deg,
            args=args,
        )
    with (output_dir / f"orientation_scores_{safe_name(spec.area)}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(orientation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(orientation_rows)
    refined, pc_score_rows = pc_finetune(
        projection=projection,
        matrix=orientation_matrix,
        mask=mask,
        images=images,
        samplers=master_samplers,
        stride=args.stride,
        coarse_range=tuple(args.pc_range),
        coarse_steps=args.coarse_steps,
        fine_steps=args.fine_steps,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    original = next(row for row in pc_score_rows if row.stage == "original")
    write_pc_scores(output_dir / f"pc_scores_{safe_name(spec.area)}.csv", pc_score_rows)

    detector_patch = project_detector_patch(projection, original.pc, images["enhanced"], mask)
    original_patch = project_crystal_patch(projection, original.pc, orientation_matrix, images["enhanced"], mask)
    refined_patch = project_crystal_patch(projection, refined.pc, orientation_matrix, images["enhanced"], mask)
    vectors, values = crystal_vectors_for_patch(
        projection=projection,
        pc=refined.pc,
        matrix=orientation_matrix,
        values=images["enhanced"],
        mask=mask,
        max_points=args.max_3d_points_per_pattern,
    )

    return ProcessedMap(
        spec=spec,
        selected_index=selection["selected_index"],
        selected_row=selection["selected_row"],
        selected_col=selection["selected_col"],
        selected_sem_xy=selection["selected_sem_xy"],
        selected_aligned_xy=selection["selected_aligned_xy"],
        selected_iq=selection["selected_iq"],
        selected_ci=selection["selected_ci"],
        selected_phase=selection["selected_phase"],
        candidate_count=selection["candidate_count"],
        pc_original=original.pc,
        pc_refined=refined.pc,
        pc_delta=refined.delta,
        orientation_variant=orientation_name,
        orientation_matrix=orientation_matrix,
        original_score=original.combined_score,
        refined_score=refined.combined_score,
        score_gain=refined.combined_score - original.combined_score,
        sem_gray=selection["sem_gray"],
        aligned_sem_gray=selection["aligned_sem_gray"],
        raw_poly=selection["raw_poly"],
        aligned_poly=selection["aligned_poly"],
        raw_pattern_display=images["raw_normalized"],
        corrected_pattern=images["corrected"],
        enhanced_pattern=images["enhanced"],
        band_pattern=images["band"],
        detector_mask=mask,
        detector_patch=detector_patch,
        original_patch=original_patch,
        refined_patch=refined_patch,
        crystal_vectors=vectors,
        crystal_values=values,
    )


def write_summary_csv(path: Path, results: list[ProcessedMap]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "label": result.spec.label,
                "h5_group": result.spec.h5_group,
                "up2_name": result.spec.up2_name,
                "selected_index": result.selected_index,
                "row": result.selected_row,
                "col": result.selected_col,
                "selected_sem_x": result.selected_sem_xy[0],
                "selected_sem_y": result.selected_sem_xy[1],
                "selected_aligned_x": result.selected_aligned_xy[0],
                "selected_aligned_y": result.selected_aligned_xy[1],
                "IQ": result.selected_iq,
                "CI": result.selected_ci,
                "phase": result.selected_phase,
                "candidate_count": result.candidate_count,
                "orientation_variant": result.orientation_variant,
                "pc_original_x": result.pc_original[0],
                "pc_original_y": result.pc_original[1],
                "pc_original_z": result.pc_original[2],
                "pc_refined_x": result.pc_refined[0],
                "pc_refined_y": result.pc_refined[1],
                "pc_refined_z": result.pc_refined[2],
                "delta_pcx": result.pc_delta[0],
                "delta_pcy": result.pc_delta[1],
                "delta_pcz": result.pc_delta[2],
                "original_score": result.original_score,
                "refined_score": result.refined_score,
                "score_gain": result.score_gain,
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_polygon(ax, polygon: np.ndarray, color: str = "cyan") -> None:
    ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]], color=color, lw=1.1)


def masked_image(image: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~mask, image)


def save_process_overview(path: Path, results: list[ProcessedMap], master_texture: np.ndarray) -> None:
    fig, axes = plt.subplots(len(results), 6, figsize=(24, 3.9 * len(results)), dpi=170)
    for row, result in enumerate(results):
        ax = axes[row, 0]
        ax.imshow(result.sem_gray, cmap="gray", vmin=0, vmax=1)
        draw_polygon(ax, result.raw_poly)
        ax.scatter([result.selected_sem_xy[0]], [result.selected_sem_xy[1]], s=68, facecolors="none", edgecolors="red", linewidths=1.5)
        ax.set_title(f"{result.spec.area} SEM same-face ROI\nidx={result.selected_index}, IQ={result.selected_iq:.0f}, CI={result.selected_ci:.3f}")
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(masked_image(result.raw_pattern_display, result.detector_mask), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Raw UP2 Kikuchi\nforced circular mask")
        ax.axis("off")

        ax = axes[row, 2]
        ax.imshow(masked_image(result.enhanced_pattern, result.detector_mask), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Preprocessed\nbackground removed + contrast enhanced")
        ax.axis("off")

        imshow_sphere(axes[row, 3], result.detector_patch[0], result.detector_patch[1], "Detector-frame sphere\nEDAX PC")

        ax = axes[row, 4]
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(
            ax,
            result.original_patch[0],
            result.original_patch[1],
            f"Crystal/master sphere\noriginal PC score={result.original_score:+.3f}",
        )

        ax = axes[row, 5]
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(
            ax,
            result.refined_patch[0],
            result.refined_patch[1],
            f"Crystal/master sphere\nrefined PC score={result.refined_score:+.3f}",
        )

    fig.suptitle("Pt-3 same physical facet: four Kikuchi spherical calibration workflows", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def master_surface(master_samplers, lon_count: int = 180, colat_count: int = 90):
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
    from project_edax_oim_to_sphere import sample_master

    values = sample_master(vectors, master_samplers.upper_corrected, master_samplers.lower_corrected).reshape(colat_count, lon_count)
    values = (values - np.percentile(values, 1)) / max(np.percentile(values, 99) - np.percentile(values, 1), 1e-6)
    values = np.clip(values, 0.0, 1.0)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)
    return x, y, z, values


def setup_3d_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-55)


def draw_master_surface(ax, surface_data) -> None:
    x, y, z, values = surface_data
    facecolors = plt.cm.gray(values)
    facecolors[..., 3] = 0.28
    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)


def draw_inplane_axis(ax, reference_matrix: np.ndarray) -> None:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ reference_matrix
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    points = np.vstack([-axis, axis])
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#ff2d55", linewidth=2.2, alpha=0.92)


def scatter_patch(ax, result: ProcessedMap, alpha: float = 0.35, size: float = 2.0, textured: bool = False) -> None:
    vectors = result.crystal_vectors
    if textured:
        values = result.crystal_values.astype(np.float64)
        lo, hi = np.percentile(values, [1, 99])
        values = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        base = np.array(to_rgb(result.spec.color), dtype=np.float64)
        colors = (0.15 + 0.85 * values[:, None]) * base[None, :]
        rgba = np.column_stack([np.clip(colors, 0.0, 1.0), np.full(values.shape, alpha)])
    else:
        rgba = result.spec.color
    ax.scatter(
        vectors[:, 0] * 1.012,
        vectors[:, 1] * 1.012,
        vectors[:, 2] * 1.012,
        s=size,
        c=rgba,
        alpha=alpha,
        depthshade=False,
        label=result.spec.area,
    )


def save_3d_sphere(path: Path, results: list[ProcessedMap], master_samplers) -> None:
    surface_data = master_surface(master_samplers)
    fig = plt.figure(figsize=(18, 10), dpi=170)

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw_master_surface(ax, surface_data)
    for result in results:
        scatter_patch(ax, result, alpha=0.34, size=1.7)
    draw_inplane_axis(ax, results[0].orientation_matrix)
    setup_3d_axis(ax, "Combined refined crystal-frame patches\nsame physical facet, four EBSD mappings")
    ax.legend(loc="lower left", fontsize=7)

    for i, result in enumerate(results, start=2):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        draw_master_surface(ax, surface_data)
        scatter_patch(ax, result, alpha=0.62, size=2.3, textured=True)
        draw_inplane_axis(ax, results[0].orientation_matrix)
        setup_3d_axis(
            ax,
            f"{result.spec.area}\nidx={result.selected_index}, refined PC=({result.pc_refined[0]:.3f}, {result.pc_refined[1]:.3f}, {result.pc_refined[2]:.3f})",
        )

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = [
        "PC finetune summary",
        "",
    ]
    for result in results:
        lines.append(
            f"{result.spec.area}: score {result.original_score:+.3f} -> {result.refined_score:+.3f}, "
            f"dPC=({result.pc_delta[0]:+.4f}, {result.pc_delta[1]:+.4f}, {result.pc_delta[2]:+.4f})"
        )
    lines.append("")
    lines.append("Pink line: enforced common in-plane rotation axis")
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Pt-3 same-facet Kikuchi patterns projected onto 3D master Kikuchi sphere", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_sem_selection_preview(path: Path, results: list[ProcessedMap]) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(15, 7.4), dpi=180)
    for i, result in enumerate(results):
        ax = axes[0, i]
        ax.imshow(result.sem_gray, cmap="gray", vmin=0, vmax=1)
        draw_polygon(ax, result.raw_poly)
        ax.scatter([result.selected_sem_xy[0]], [result.selected_sem_xy[1]], s=70, facecolors="none", edgecolors="red", linewidths=1.6)
        ax.set_title(f"{result.spec.area} raw\nidx={result.selected_index}, IQ={result.selected_iq:.0f}, CI={result.selected_ci:.3f}")
        ax.axis("off")

        ax = axes[1, i]
        ax.imshow(result.aligned_sem_gray, cmap="gray", vmin=0, vmax=1)
        draw_polygon(ax, result.aligned_poly)
        ax.scatter([result.selected_aligned_xy[0]], [result.selected_aligned_xy[1]], s=70, facecolors="none", edgecolors="red", linewidths=1.6)
        ax.set_title(f"aligned {result.spec.rotation_to_reference_deg:+g} deg to Area 3-360")
        ax.axis("off")
    fig.suptitle("Pt-3 same-facet ROI and selected high-quality Kikuchi points")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_samplers = build_master_samplers(args.master)
    master_texture = build_master_lon_colat(master_samplers.upper_corrected, master_samplers.lower_corrected)

    results: list[ProcessedMap] = []
    reference_result = process_one_map(
        spec=MAP_SPECS[0],
        h5_path=args.h5,
        up2_root=args.up2_root,
        face_polygon_aligned=DEFAULT_FACE_POLYGON_ALIGNED,
        master_samplers=master_samplers,
        output_dir=args.output_dir,
        args=args,
        reference_matrix=None,
    )
    results.append(reference_result)
    for spec in MAP_SPECS[1:]:
        results.append(
            process_one_map(
                spec=spec,
                h5_path=args.h5,
                up2_root=args.up2_root,
                face_polygon_aligned=DEFAULT_FACE_POLYGON_ALIGNED,
                master_samplers=master_samplers,
                output_dir=args.output_dir,
                args=args,
                reference_matrix=reference_result.orientation_matrix,
            )
        )

    write_summary_csv(args.output_dir / "pt3_same_face_spherical_calibration_summary.csv", results)
    save_sem_selection_preview(args.output_dir / "pt3_same_face_roi_selection.png", results)
    save_process_overview(args.output_dir / "pt3_same_face_spherical_calibration_workflow.png", results, master_texture)
    save_3d_sphere(args.output_dir / "pt3_same_face_3d_kikuchi_sphere.png", results, master_samplers)

    print(f"Saved outputs to {args.output_dir}")
    for result in results:
        print(
            f"{result.spec.area}: idx={result.selected_index}, IQ={result.selected_iq:.1f}, CI={result.selected_ci:.3f}, "
            f"PC {tuple(round(x, 6) for x in result.pc_original)} -> {tuple(round(x, 6) for x in result.pc_refined)}, "
            f"score {result.original_score:+.4f}->{result.refined_score:+.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the same Pt-3 SEM facet in four in-plane EBSD mappings and run Kikuchi spherical calibration with PC finetune."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", type=Path, default=DEFAULT_UP2_ROOT)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ci-min", type=float, default=0.30)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--pc-range", nargs=3, type=float, default=(0.01, 0.01, 0.02), metavar=("DX", "DY", "DZ"))
    parser.add_argument(
        "--mask-radius-fraction",
        type=float,
        default=0.40,
        help="Conservative circular detector mask radius as a fraction of min(pattern height, width).",
    )
    parser.add_argument(
        "--inplane-prior-family",
        choices=("sample_pre", "sample_pre_inv", "crystal_post", "crystal_post_inv", "auto_per_map"),
        default="crystal_post",
        help=(
            "How to apply the known in-plane rotation prior. The default uses one global crystal-sphere axis "
            "rotation family for all non-reference Pt-3 maps. auto_per_map is diagnostic only."
        ),
    )
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-steps", type=int, default=7)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--max-3d-points-per-pattern", type=int, default=18000)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
