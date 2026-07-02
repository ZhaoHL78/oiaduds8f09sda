from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from skimage import exposure, filters

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    build_master_lon_colat,
    orientation_candidates,
    project_patch_to_lon_colat,
    read_edax_inputs,
    sample_master,
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
    base_orientation_variant: str
    base_orientation_matrix: np.ndarray
    orientation_variant: str
    orientation_matrix: np.ndarray
    symmetry_name: str
    symmetry_matrix: np.ndarray
    axis_prior_score: float
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


def orthonormalize_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(matrix.astype(np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def cubic_proper_symmetry_matrices() -> list[tuple[str, np.ndarray]]:
    axes = ("x", "y", "z")
    operations: list[tuple[str, np.ndarray]] = []
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            name_parts: list[str] = []
            for row, axis_index in enumerate(perm):
                matrix[row, axis_index] = signs[row]
                name_parts.append(("-" if signs[row] < 0 else "+") + axes[axis_index])
            if np.linalg.det(matrix) > 0.5:
                operations.append(("".join(name_parts), matrix))
    operations.sort(key=lambda item: item[0])
    identity_index = next(i for i, (_name, matrix) in enumerate(operations) if np.allclose(matrix, np.eye(3)))
    operations.insert(0, operations.pop(identity_index))
    return operations


def rotation_angle_deg(matrix: np.ndarray) -> float:
    trace = float(np.trace(matrix))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def rotation_axis(matrix: np.ndarray) -> np.ndarray:
    angle = math.radians(rotation_angle_deg(matrix))
    if abs(math.sin(angle)) > 1e-6:
        axis = np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        )
        axis /= 2.0 * math.sin(angle)
    else:
        values, vectors = np.linalg.eig(matrix)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / norm
    return axis.astype(np.float64)


def choose_fixed_orientation_matrix(
    projection,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers,
    variant_name: str,
    args: argparse.Namespace,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    candidates = orientation_candidates(projection.orientation_flat)
    if variant_name not in candidates:
        raise KeyError(f"Orientation variant {variant_name!r} not present in H5 candidates")
    indices = make_stride_indices(mask, args.stride)
    detector_directions = detector_directions_with_pc(projection, projection.pc_edax)
    exp_corrected = images["enhanced"].ravel()[indices]
    exp_band = images["band"].ravel()[indices]
    rows: list[dict[str, object]] = []
    for name, matrix in candidates.items():
        intensity, band, combined = score_with_directions(
            detector_directions=detector_directions,
            matrix=matrix,
            indices=indices,
            exp_corrected_values=exp_corrected,
            exp_band_values=exp_band,
            samplers=samplers,
            intensity_weight=args.intensity_weight,
            band_weight=args.band_weight,
        )
        rows.append(
            {
                "orientation_variant": name,
                "intensity_score": intensity,
                "band_score": band,
                "combined_score": combined,
                "selected": name == variant_name,
            }
        )
    return variant_name, candidates[variant_name], rows


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


def project_vectors_patch(vectors: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected, patch_mask, _angles = project_patch_to_lon_colat(vectors, values)
    return projected, patch_mask


def apply_cubic_symmetry_axis_prior(results: list[ProcessedMap]) -> dict[str, Any]:
    if len(results) != 4:
        raise ValueError("The cubic symmetry axis prior currently expects four Pt-3 in-plane maps")

    symmetries = cubic_proper_symmetry_matrices()
    base_matrices = [orthonormalize_rotation(result.base_orientation_matrix) for result in results]
    final_candidates = [
        [orthonormalize_rotation(base_matrix @ symmetry_matrix) for _name, symmetry_matrix in symmetries]
        for base_matrix in base_matrices
    ]

    best: dict[str, Any] | None = None
    n_sym = len(symmetries)
    for s0 in range(n_sym):
        f0 = final_candidates[0][s0]
        for s1 in range(n_sym):
            q90 = f0.T @ final_candidates[1][s1]
            q90_sq = q90 @ q90
            q90_cube = q90_sq @ q90
            angle90 = rotation_angle_deg(q90)
            for s2 in range(n_sym):
                q180 = f0.T @ final_candidates[2][s2]
                closure180 = np.linalg.norm(q180 - q90_sq, ord="fro")
                angle180 = rotation_angle_deg(q180)
                for s3 in range(n_sym):
                    q270 = f0.T @ final_candidates[3][s3]
                    closure270 = np.linalg.norm(q270 - q90_cube, ord="fro")
                    angle270 = rotation_angle_deg(q270)
                    angle_penalty = (
                        abs(angle90 - 90.0) / 90.0
                        + abs(angle180 - 180.0) / 180.0
                        + abs(angle270 - 90.0) / 90.0
                    )
                    score = float(closure180 + closure270 + 2.0 * angle_penalty)
                    if best is None or score < best["score"]:
                        best = {
                            "score": score,
                            "symmetry_indices": (s0, s1, s2, s3),
                            "closure180": float(closure180),
                            "closure270": float(closure270),
                            "angle90_deg": float(angle90),
                            "angle180_deg": float(angle180),
                            "angle270_deg": float(angle270),
                            "q90": q90,
                        }

    if best is None:
        raise RuntimeError("No cubic symmetry axis-prior fit was found")

    common_axis = rotation_axis(best["q90"])
    best["common_axis"] = common_axis
    for result, symmetry_index in zip(results, best["symmetry_indices"]):
        symmetry_name, symmetry_matrix = symmetries[int(symmetry_index)]
        result.original_patch = project_vectors_patch(result.crystal_vectors, result.crystal_values)
        result.symmetry_name = symmetry_name
        result.symmetry_matrix = symmetry_matrix
        result.axis_prior_score = float(best["score"])
        result.orientation_matrix = result.base_orientation_matrix @ symmetry_matrix
        result.orientation_variant = f"h5_{result.base_orientation_variant} @ cubic_sym({symmetry_name})"
        result.crystal_vectors = (result.crystal_vectors @ symmetry_matrix).astype(np.float32)
        result.refined_patch = project_vectors_patch(result.crystal_vectors, result.crystal_values)

    return best


def write_symmetry_axis_summary(path: Path, results: list[ProcessedMap], fit: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    axis = fit["common_axis"]
    for result, symmetry_index in zip(results, fit["symmetry_indices"]):
        rows.append(
            {
                "label": result.spec.label,
                "area": result.spec.area,
                "base_orientation_variant": result.base_orientation_variant,
                "selected_cubic_symmetry_index": int(symmetry_index),
                "selected_cubic_symmetry": result.symmetry_name,
                "axis_prior_score": fit["score"],
                "closure180": fit["closure180"],
                "closure270": fit["closure270"],
                "angle90_deg": fit["angle90_deg"],
                "angle180_deg": fit["angle180_deg"],
                "angle270_deg": fit["angle270_deg"],
                "common_axis_x": axis[0],
                "common_axis_y": axis[1],
                "common_axis_z": axis[2],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_one_map(
    spec: Pt3MapSpec,
    h5_path: Path,
    up2_root: Path,
    face_polygon_aligned: np.ndarray,
    master_samplers,
    output_dir: Path,
    args: argparse.Namespace,
    fixed_orientation_variant: str | None = None,
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
    if fixed_orientation_variant is None:
        orientation_name, orientation_matrix, orientation_rows = choose_orientation_matrix(
            projection=projection,
            mask=mask,
            images=images,
            samplers=master_samplers,
            stride=args.stride,
            intensity_weight=args.intensity_weight,
            band_weight=args.band_weight,
        )
    else:
        orientation_name, orientation_matrix, orientation_rows = choose_fixed_orientation_matrix(
            projection=projection,
            mask=mask,
            images=images,
            samplers=master_samplers,
            variant_name=fixed_orientation_variant,
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
        base_orientation_variant=orientation_name,
        base_orientation_matrix=orientation_matrix,
        orientation_variant=f"h5_{orientation_name}",
        orientation_matrix=orientation_matrix.copy(),
        symmetry_name="identity_before_axis_fit",
        symmetry_matrix=np.eye(3, dtype=np.float64),
        axis_prior_score=float("nan"),
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
                "base_orientation_variant": result.base_orientation_variant,
                "orientation_variant": result.orientation_variant,
                "selected_cubic_symmetry": result.symmetry_name,
                "axis_prior_score": result.axis_prior_score,
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
            f"H5 orientation on master\nrefined PC score={result.refined_score:+.3f}",
        )

        ax = axes[row, 5]
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(
            ax,
            result.refined_patch[0],
            result.refined_patch[1],
            f"Cubic-symmetry equivalent\n{result.symmetry_name}",
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


def draw_inplane_axis(ax, axis: np.ndarray) -> None:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    points = np.vstack([-axis, axis])
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#ff2d55", linewidth=2.2, alpha=0.92)


def normalize_values(values: np.ndarray, mask: np.ndarray | None = None, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    selected = values[valid]
    if selected.size == 0:
        return output
    lo, hi = np.percentile(selected, [low, high])
    if hi <= lo:
        lo, hi = float(selected.min()), float(selected.max())
    if hi <= lo:
        return output
    output[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    return output


def smooth_masked_values(values: np.ndarray, mask: np.ndarray, sigma: float = 0.75) -> np.ndarray:
    if sigma <= 0:
        return values.astype(np.float32)
    values_f = values.astype(np.float32)
    mask_f = mask.astype(np.float32)
    blurred_values = filters.gaussian(values_f * mask_f, sigma=sigma, preserve_range=True)
    blurred_mask = filters.gaussian(mask_f, sigma=sigma, preserve_range=True)
    output = np.zeros_like(values_f, dtype=np.float32)
    valid = blurred_mask > 1e-4
    output[valid] = blurred_values[valid] / blurred_mask[valid]
    output[~mask] = 0.0
    return output


def patch_center_vector(result: ProcessedMap) -> np.ndarray:
    center = np.mean(result.crystal_vectors.astype(np.float64), axis=0)
    norm = np.linalg.norm(center)
    if norm < 1e-12:
        center = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        center = center / norm
    return center


def camera_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = center / max(np.linalg.norm(center), 1e-12)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_guess))) > 0.92:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up_guess, forward)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-12)
    return right, up, forward


def front_sphere_master_image(master_samplers, center: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    right, up, forward = camera_basis(center)
    grid = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(grid, grid)
    disk = x_grid**2 + y_grid**2 <= 1.0
    z_grid = np.zeros_like(x_grid)
    z_grid[disk] = np.sqrt(np.clip(1.0 - x_grid[disk] ** 2 - y_grid[disk] ** 2, 0.0, 1.0))
    vectors = (
        x_grid[disk, None] * right[None, :]
        + y_grid[disk, None] * up[None, :]
        + z_grid[disk, None] * forward[None, :]
    )
    master = np.ones((size, size), dtype=np.float32)
    sampled = sample_master(vectors, master_samplers.upper_corrected, master_samplers.lower_corrected)
    sampled = normalize_values(sampled.reshape(-1), low=0.5, high=99.5)
    master[disk] = sampled
    shade = np.ones_like(master, dtype=np.float32)
    shade[disk] = (0.62 + 0.38 * z_grid[disk]).astype(np.float32)
    master = np.clip(master * shade, 0.0, 1.0)
    image = np.ones((size, size, 4), dtype=np.float32)
    image[..., :3] = 1.0
    image[..., 3] = 0.0
    image[disk, :3] = np.repeat(master[disk, None], 3, axis=1)
    image[disk, 3] = 1.0
    return image, disk, z_grid.astype(np.float32), (right, up, forward)


def rasterize_patch_front_view(
    result: ProcessedMap,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = basis
    vectors = result.crystal_vectors.astype(np.float64)
    values = result.crystal_values.astype(np.float32)
    cam_x = vectors @ right
    cam_y = vectors @ up
    cam_z = vectors @ forward
    visible = cam_z > 0.0
    px = np.rint((cam_x[visible] + 1.0) * 0.5 * (size - 1)).astype(np.int64)
    py = np.rint((1.0 - (cam_y[visible] + 1.0) * 0.5) * (size - 1)).astype(np.int64)
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px = px[inside]
    py = py[inside]
    vals = values[visible][inside]

    patch_sum = np.zeros((size, size), dtype=np.float32)
    patch_count = np.zeros((size, size), dtype=np.float32)
    np.add.at(patch_sum, (py, px), vals)
    np.add.at(patch_count, (py, px), 1.0)
    patch_mask = patch_count > 0
    patch = np.zeros_like(patch_sum)
    patch[patch_mask] = patch_sum[patch_mask] / patch_count[patch_mask]
    patch = normalize_values(patch, patch_mask, low=0.8, high=99.2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = patch.copy()
    filled_mask = patch_mask.copy()
    for _ in range(2):
        dilated_values = cv2.dilate(filled, kernel)
        dilated_mask = cv2.dilate(filled_mask.astype(np.uint8), kernel).astype(bool)
        new_pixels = dilated_mask & ~filled_mask
        filled[new_pixels] = dilated_values[new_pixels]
        filled_mask = dilated_mask
    return filled, filled_mask


def render_front_sphere_view(
    result: ProcessedMap,
    master_samplers,
    size: int = 1500,
) -> np.ndarray:
    center = patch_center_vector(result)
    image, disk, z_grid, basis = front_sphere_master_image(master_samplers, center, size)
    patch, patch_mask = rasterize_patch_front_view(result, basis, size)
    patch_mask &= disk
    shade = 0.70 + 0.30 * z_grid
    patch_rgb = np.repeat(np.clip(patch * shade, 0.0, 1.0)[..., None], 3, axis=2)
    alpha = np.where(patch_mask, 0.94, 0.0).astype(np.float32)
    image[..., :3] = image[..., :3] * (1.0 - alpha[..., None]) + patch_rgb * alpha[..., None]
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edge = cv2.dilate(patch_mask.astype(np.uint8), edge_kernel).astype(bool) & ~cv2.erode(
        patch_mask.astype(np.uint8), edge_kernel
    ).astype(bool)
    color = np.array(to_rgb(result.spec.color), dtype=np.float32)
    image[edge, :3] = color
    image[edge, 3] = 1.0
    sphere_edge = cv2.dilate(disk.astype(np.uint8), edge_kernel).astype(bool) & ~cv2.erode(
        disk.astype(np.uint8), edge_kernel
    ).astype(bool)
    image[sphere_edge, :3] = 0.08
    image[sphere_edge, 3] = 1.0
    return np.clip(image, 0.0, 1.0)


def render_front_pattern_only(result: ProcessedMap, size: int = 1300) -> np.ndarray:
    center = patch_center_vector(result)
    grid = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(grid, grid)
    disk = x_grid**2 + y_grid**2 <= 1.0
    z_grid = np.zeros_like(x_grid, dtype=np.float32)
    z_grid[disk] = np.sqrt(np.clip(1.0 - x_grid[disk] ** 2 - y_grid[disk] ** 2, 0.0, 1.0)).astype(np.float32)
    basis = camera_basis(center)
    patch, patch_mask = rasterize_patch_front_view(result, basis, size)
    patch_mask &= disk

    image = np.ones((size, size, 4), dtype=np.float32)
    image[..., :3] = 1.0
    image[..., 3] = 0.0
    sphere_shade = 0.88 + 0.10 * z_grid
    image[disk, :3] = np.repeat(sphere_shade[disk, None], 3, axis=1)
    image[disk, 3] = 1.0

    shade = 0.76 + 0.24 * z_grid
    patch_rgb = np.repeat(np.clip(patch * shade, 0.0, 1.0)[..., None], 3, axis=2)
    image[patch_mask, :3] = patch_rgb[patch_mask]
    image[patch_mask, 3] = 1.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edge = cv2.dilate(patch_mask.astype(np.uint8), kernel).astype(bool) & ~cv2.erode(
        patch_mask.astype(np.uint8), kernel
    ).astype(bool)
    image[edge, :3] = np.array(to_rgb(result.spec.color), dtype=np.float32)
    image[edge, 3] = 1.0
    sphere_edge = cv2.dilate(disk.astype(np.uint8), kernel).astype(bool) & ~cv2.erode(
        disk.astype(np.uint8), kernel
    ).astype(bool)
    image[sphere_edge, :3] = 0.12
    image[sphere_edge, 3] = 1.0
    return np.clip(image, 0.0, 1.0)


def save_clear_spherical_maps(path: Path, results: list[ProcessedMap], master_texture: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.5), dpi=230)
    axes_flat = axes.ravel()
    for ax, result in zip(axes_flat, results):
        image = render_front_pattern_only(result, size=1300)
        ax.imshow(image)
        ax.set_title(f"{result.spec.area} sphere-corrected Kikuchi pattern\nsym={result.symmetry_name}, score={result.refined_score:+.3f}")
        ax.axis("off")
    fig.suptitle("Clear sphere-corrected Kikuchi patterns, front-facing in crystal/master-sphere coordinates", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_clear_front_sphere_views(path: Path, results: list[ProcessedMap], master_samplers, output_dir: Path) -> None:
    master_display = normalize_values(
        build_master_lon_colat(master_samplers.upper_corrected, master_samplers.lower_corrected),
        low=0.5,
        high=99.5,
    )
    lon = np.linspace(-np.pi, np.pi, master_display.shape[1])
    colat = np.linspace(0.0, np.pi, master_display.shape[0])
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)
    def facecolors_for_result(result: ProcessedMap) -> np.ndarray:
        patch = normalize_values(result.refined_patch[0], result.refined_patch[1], low=0.8, high=99.2)
        patch = smooth_masked_values(patch, result.refined_patch[1], sigma=0.65)
        base = np.repeat(master_display[..., None], 3, axis=2)
        base = 0.56 * base + 0.12
        patch_rgb = np.repeat(patch[..., None], 3, axis=2)
        mask = result.refined_patch[1]
        base[mask] = 0.88 * patch_rgb[mask] + 0.06
        rgba = np.ones((*base.shape[:2], 4), dtype=np.float32)
        rgba[..., :3] = np.clip(base, 0.0, 1.0)
        rgba[..., 3] = 1.0
        return rgba

    fig = plt.figure(figsize=(14, 14), dpi=220)
    for index, result in enumerate(results, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        colors = facecolors_for_result(result)
        stride = 1
        ax.plot_surface(
            x[::stride, ::stride],
            y[::stride, ::stride],
            z[::stride, ::stride],
            facecolors=colors[::stride, ::stride],
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        center = patch_center_vector(result)
        elev = math.degrees(math.asin(float(np.clip(center[2], -1.0, 1.0))))
        azim = math.degrees(math.atan2(float(center[1]), float(center[0])))
        setup_3d_axis(
            ax,
            f"{result.spec.area}: true 3D surface texture, view normal to patch\nsym={result.symmetry_name}",
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")

        individual_path = output_dir / f"pt3_front_view_{safe_name(result.spec.area)}.png"
        one_fig = plt.figure(figsize=(8, 8), dpi=240)
        one_ax = one_fig.add_subplot(1, 1, 1, projection="3d")
        one_ax.plot_surface(
            x[::stride, ::stride],
            y[::stride, ::stride],
            z[::stride, ::stride],
            facecolors=colors[::stride, ::stride],
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        setup_3d_axis(one_ax, f"{result.spec.area}: pattern attached at final master-sphere position")
        one_ax.view_init(elev=elev, azim=azim)
        one_ax.set_proj_type("ortho")
        one_fig.tight_layout()
        one_fig.savefig(individual_path, bbox_inches="tight", transparent=True)
        plt.close(one_fig)

    fig.suptitle("Final true 3D Kikuchi sphere: pattern texture attached at the cubic-symmetry-selected master-sphere position", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def same_sphere_composite_surface(results: list[ProcessedMap], master_samplers):
    master_display = normalize_values(
        build_master_lon_colat(master_samplers.upper_corrected, master_samplers.lower_corrected),
        low=0.5,
        high=99.5,
    )
    lon = np.linspace(-np.pi, np.pi, master_display.shape[1])
    colat = np.linspace(0.0, np.pi, master_display.shape[0])
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)

    rgb_sum = np.zeros((*master_display.shape, 3), dtype=np.float32)
    weight_sum = np.zeros(master_display.shape, dtype=np.float32)
    for result in results:
        patch = normalize_values(result.refined_patch[0], result.refined_patch[1], low=0.8, high=99.2)
        patch = smooth_masked_values(patch, result.refined_patch[1], sigma=0.55)
        color = np.array(to_rgb(result.spec.color), dtype=np.float32)
        patch_rgb = 0.82 * np.repeat(patch[..., None], 3, axis=2) + 0.18 * color[None, None, :]
        mask = result.refined_patch[1]
        rgb_sum[mask] += patch_rgb[mask]
        weight_sum[mask] += 1.0

    base = 0.42 * np.repeat(master_display[..., None], 3, axis=2) + 0.11
    combined = base.copy()
    occupied = weight_sum > 0
    combined[occupied] = rgb_sum[occupied] / weight_sum[occupied, None]

    rgba = np.ones((*master_display.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(combined, 0.0, 1.0)
    rgba[..., 3] = 1.0
    return x, y, z, rgba


def view_angles_from_vector(vector: np.ndarray) -> tuple[float, float]:
    vector = vector.astype(np.float64)
    vector /= max(np.linalg.norm(vector), 1e-12)
    elev = math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return elev, azim


def render_same_sphere_view(
    ax,
    surface_data,
    title: str,
    view_vector: np.ndarray,
    axis: np.ndarray | None = None,
) -> None:
    x, y, z, rgba = surface_data
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=rgba,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    if axis is not None:
        draw_inplane_axis(ax, axis)
    setup_3d_axis(ax, title)
    elev, azim = view_angles_from_vector(view_vector)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")


def save_same_sphere_axis_aligned_views(
    path: Path,
    results: list[ProcessedMap],
    master_samplers,
    symmetry_fit: dict[str, Any],
    output_dir: Path,
) -> None:
    surface_data = same_sphere_composite_surface(results, master_samplers)
    axis = symmetry_fit["common_axis"].astype(np.float64)
    mean_center = np.mean([patch_center_vector(result) for result in results], axis=0)
    if float(np.dot(axis, mean_center)) < 0:
        axis = -axis
    reference_center = patch_center_vector(results[0])
    oblique = axis + 0.85 * reference_center
    if np.linalg.norm(oblique) < 1e-12:
        oblique = reference_center

    views = [
        ("View along fitted common rotation axis", axis),
        ("Opposite side of the same common axis", -axis),
        ("View normal to Area 3-360 final patch", reference_center),
        ("Oblique view showing shared sphere and axis", oblique),
    ]

    fig = plt.figure(figsize=(15, 15), dpi=220)
    for index, (title, view_vector) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_same_sphere_view(
            ax=ax,
            surface_data=surface_data,
            title=title,
            view_vector=view_vector,
            axis=axis,
        )
    fig.suptitle("All Pt-3 Kikuchi patterns corrected onto one standard Kikuchi sphere by cubic symmetry", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    for suffix, view_vector in [
        ("axis_view", axis),
        ("reference_patch_view", reference_center),
        ("oblique_axis_view", oblique),
    ]:
        one_fig = plt.figure(figsize=(8.5, 8.5), dpi=240)
        one_ax = one_fig.add_subplot(1, 1, 1, projection="3d")
        render_same_sphere_view(
            ax=one_ax,
            surface_data=surface_data,
            title=f"Same Kikuchi sphere: {suffix.replace('_', ' ')}",
            view_vector=view_vector,
            axis=axis,
        )
        one_fig.tight_layout()
        one_fig.savefig(output_dir / f"pt3_same_sphere_{suffix}.png", bbox_inches="tight", transparent=True)
        plt.close(one_fig)


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


def save_3d_sphere(path: Path, results: list[ProcessedMap], master_samplers, symmetry_fit: dict[str, Any]) -> None:
    surface_data = master_surface(master_samplers)
    fig = plt.figure(figsize=(18, 10), dpi=170)
    common_axis = symmetry_fit["common_axis"]

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw_master_surface(ax, surface_data)
    for result in results:
        scatter_patch(ax, result, alpha=0.34, size=1.7)
    draw_inplane_axis(ax, common_axis)
    setup_3d_axis(ax, "Combined refined crystal-frame patches\nsame physical facet, four EBSD mappings")
    ax.legend(loc="lower left", fontsize=7)

    for i, result in enumerate(results, start=2):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        draw_master_surface(ax, surface_data)
        scatter_patch(ax, result, alpha=0.62, size=2.3, textured=True)
        draw_inplane_axis(ax, common_axis)
        setup_3d_axis(
            ax,
            f"{result.spec.area}\nidx={result.selected_index}, refined PC=({result.pc_refined[0]:.3f}, {result.pc_refined[1]:.3f}, {result.pc_refined[2]:.3f})",
        )

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = [
        "Cubic symmetry axis-prior summary",
        f"score={symmetry_fit['score']:.5f}",
        f"angles: 90={symmetry_fit['angle90_deg']:.2f}, 180={symmetry_fit['angle180_deg']:.2f}, 270={symmetry_fit['angle270_deg']:.2f}",
        f"axis=({common_axis[0]:+.3f}, {common_axis[1]:+.3f}, {common_axis[2]:+.3f})",
        "",
    ]
    for result in results:
        lines.append(
            f"{result.spec.area}: sym={result.symmetry_name}, "
            f"PC=({result.pc_refined[0]:.3f}, {result.pc_refined[1]:.3f}, {result.pc_refined[2]:.3f})"
        )
    lines.append("")
    lines.append("Pink line: common axis after cubic-symmetry placement")
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Pt-3 H5-orientation Kikuchi patterns placed by cubic symmetry onto one rotation axis", fontsize=15)
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
        fixed_orientation_variant=None,
    )
    results.append(reference_result)
    fixed_orientation_variant = (
        reference_result.base_orientation_variant if args.orientation_mode == "reference_variant" else None
    )
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
                fixed_orientation_variant=fixed_orientation_variant,
            )
        )

    symmetry_fit = apply_cubic_symmetry_axis_prior(results)
    write_summary_csv(args.output_dir / "pt3_same_face_spherical_calibration_summary.csv", results)
    write_symmetry_axis_summary(args.output_dir / "pt3_cubic_symmetry_axis_prior_summary.csv", results, symmetry_fit)
    save_sem_selection_preview(args.output_dir / "pt3_same_face_roi_selection.png", results)
    save_process_overview(args.output_dir / "pt3_same_face_spherical_calibration_workflow.png", results, master_texture)
    save_3d_sphere(args.output_dir / "pt3_same_face_3d_kikuchi_sphere.png", results, master_samplers, symmetry_fit)
    save_clear_spherical_maps(args.output_dir / "pt3_clear_final_spherical_kikuchi_maps.png", results, master_texture)
    save_clear_front_sphere_views(args.output_dir / "pt3_clear_3d_front_facing_kikuchi_spheres.png", results, master_samplers, args.output_dir)
    save_same_sphere_axis_aligned_views(
        args.output_dir / "pt3_same_sphere_axis_aligned_kikuchi_patterns.png",
        results,
        master_samplers,
        symmetry_fit,
        args.output_dir,
    )

    print(f"Saved outputs to {args.output_dir}")
    print(
        "Cubic symmetry axis prior: "
        f"score={symmetry_fit['score']:.5f}, "
        f"angles=({symmetry_fit['angle90_deg']:.2f}, {symmetry_fit['angle180_deg']:.2f}, {symmetry_fit['angle270_deg']:.2f}), "
        f"axis={tuple(round(float(x), 5) for x in symmetry_fit['common_axis'])}"
    )
    for result in results:
        print(
            f"{result.spec.area}: idx={result.selected_index}, IQ={result.selected_iq:.1f}, CI={result.selected_ci:.3f}, "
            f"sym={result.symmetry_name}, "
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
        "--orientation-mode",
        choices=("reference_variant", "auto_per_map"),
        default="reference_variant",
        help=(
            "reference_variant uses the best H5 orientation-matrix convention from Area 3-360 for all maps, "
            "while auto_per_map lets each map choose its own best H5 convention. Neither mode overwrites the "
            "software orientation with an in-plane prior."
        ),
    )
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-steps", type=int, default=7)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--max-3d-points-per-pattern", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
