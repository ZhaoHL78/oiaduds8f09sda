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
import torch
from matplotlib.colors import to_rgb
from scipy import optimize
from skimage import exposure, filters

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
    pc_finetune,
    project_crystal_patch,
    project_detector_patch,
    make_stride_indices,
    score_with_directions,
    write_pc_scores,
)
from pt3_same_face_spherical_calibration import (
    ProcessedMap,
    bilinear_direction_at_xy,
    camera_basis,
    crystal_vectors_for_patch,
    cubic_proper_symmetry_matrices,
    detector_directions_with_pc,
    draw_inplane_axis,
    imshow_sphere,
    masked_image,
    master_surface,
    normalize_values,
    orthonormalize_rotation,
    pc_crystal_vector,
    project_vectors_patch,
    render_same_sphere_view,
    rotation_angle_deg,
    rotation_axis,
    safe_name,
    same_sphere_composite_surface,
    scan_to_sem_xy,
    setup_3d_axis,
    smooth_masked_values,
    vector_to_lon_colat_deg,
    view_angles_from_vector,
)


DEFAULT_H5 = Path(r"E:\ZHL\EBSD-RAW\20251217Pt-high resolution\20251217.edaxh5")
DEFAULT_UP2_ROOT = Path(r"E:\ZHL\EBSD-RAW\20251217Pt-high resolution")
DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_highres_30deg_lightglue_calibration"
BASE_GROUP = "20251217/Pt foil-high resolution"


@dataclass(frozen=True)
class HighResMapSpec:
    label: str
    area: str
    angle_deg: int
    up2_name: str
    color: str

    @property
    def h5_group(self) -> str:
        return f"{BASE_GROUP}/{self.area}/OIM Map 1"


@dataclass
class MapSemData:
    spec: HighResMapSpec
    sem_gray: np.ndarray
    sem_match: np.ndarray
    nrows: int
    ncols: int
    iq: np.ndarray
    ci: np.ndarray
    valid: np.ndarray
    phase: np.ndarray


@dataclass
class PairAlignment:
    moving_angle: int
    fixed_angle: int
    initial_rotation_deg: float
    method: str
    matches: int
    inliers: int
    inlier_ratio: float
    residual_rmse: float
    transform_moving_to_fixed: np.ndarray


@dataclass
class SequenceScoringItem:
    result: ProcessedMap
    detector_directions: np.ndarray
    indices: np.ndarray
    exp_corrected_values: np.ndarray
    exp_band_values: np.ndarray


def build_map_specs() -> list[HighResMapSpec]:
    cmap = plt.get_cmap("turbo")
    specs: list[HighResMapSpec] = []
    for index, angle in enumerate(range(0, 360, 30)):
        area_number = 3 + index
        specs.append(
            HighResMapSpec(
                label=f"Pt high-res {angle} deg",
                area=f"Area 8-{angle}",
                angle_deg=angle,
                up2_name=f"20251217_Pt foil-high resolution_Area {area_number}_OIM Map 1.up2",
                color=to_hex(cmap(index / 11.0)),
            )
        )
    return specs


def to_hex(rgba) -> str:
    r, g, b = [int(round(255 * float(v))) for v in rgba[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def preprocess_sem_for_matching(image: np.ndarray) -> np.ndarray:
    image = normalize_gray(image)
    image = exposure.equalize_adapthist(image, clip_limit=0.02).astype(np.float32)
    high_pass = image - filters.gaussian(image, sigma=6.0, preserve_range=True)
    return exposure.rescale_intensity(high_pass, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)


def sem_rotation_matrix(shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    height, width = shape
    return cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0).astype(np.float64)


def affine23_to33(matrix: np.ndarray) -> np.ndarray:
    output = np.eye(3, dtype=np.float64)
    output[:2, :] = matrix[:2, :]
    return output


def affine33_to23(matrix: np.ndarray) -> np.ndarray:
    return matrix[:2, :].astype(np.float64)


def warp_sem(image: np.ndarray, transform_raw_to_ref: np.ndarray) -> np.ndarray:
    height, width = image.shape
    return cv2.warpAffine(
        image,
        affine33_to23(transform_raw_to_ref),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def read_highres_sem_data(h5_path: Path, specs: list[HighResMapSpec]) -> list[MapSemData]:
    maps: list[MapSemData] = []
    with h5py.File(h5_path, "r") as h5:
        for spec in specs:
            group = h5[spec.h5_group]
            data = group["EBSD/ANG/DATA/DATA"][:]
            sem = normalize_gray(np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
            nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
            ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
            valid = (
                data["Valid"].astype(bool)
                & np.isfinite(data["IQ"])
                & np.isfinite(data["CI"])
                & (data["Phase"] == 1)
            )
            maps.append(
                MapSemData(
                    spec=spec,
                    sem_gray=sem,
                    sem_match=preprocess_sem_for_matching(sem),
                    nrows=nrows,
                    ncols=ncols,
                    iq=data["IQ"].astype(np.float64),
                    ci=data["CI"].astype(np.float64),
                    valid=valid,
                    phase=data["Phase"].astype(np.int16),
                )
            )
    return maps


def make_lightglue_models(max_keypoints: int, device: str):
    from lightglue import LightGlue, SuperPoint

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def lightglue_affine_for_rotation(
    fixed: np.ndarray,
    moving: np.ndarray,
    rotation_deg: float,
    extractor,
    matcher,
    device: str,
    ransac_reproj_threshold: float,
) -> dict[str, Any]:
    from lightglue.utils import rbd

    initial = sem_rotation_matrix(moving.shape, rotation_deg)
    moving_rot = cv2.warpAffine(
        moving,
        initial,
        (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    with torch.inference_mode():
        tensor_fixed = torch.from_numpy(fixed.astype(np.float32))[None, None].to(device)
        tensor_moving = torch.from_numpy(moving_rot.astype(np.float32))[None, None].to(device)
        feats_fixed = extractor.extract(tensor_fixed)
        feats_moving = extractor.extract(tensor_moving)
        matches_out = matcher({"image0": feats_fixed, "image1": feats_moving})
        feats_fixed = rbd(feats_fixed)
        feats_moving = rbd(feats_moving)
        matches_out = rbd(matches_out)
        matches = matches_out["matches"]
        key_fixed = feats_fixed["keypoints"][matches[:, 0]].detach().cpu().numpy()
        key_moving = feats_moving["keypoints"][matches[:, 1]].detach().cpu().numpy()

    if key_fixed.shape[0] < 6:
        raise RuntimeError(f"Too few LightGlue matches: {key_fixed.shape[0]}")

    residual, inliers = cv2.estimateAffinePartial2D(
        key_moving,
        key_fixed,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=8000,
        confidence=0.999,
    )
    if residual is None or inliers is None:
        raise RuntimeError("LightGlue matches could not produce an affine transform")
    inlier_mask = inliers.reshape(-1).astype(bool)
    pred = (residual @ np.vstack([key_moving.T, np.ones(key_moving.shape[0])])).T
    err = np.linalg.norm(pred - key_fixed, axis=1)
    rmse = float(np.sqrt(np.mean(err[inlier_mask] ** 2))) if np.any(inlier_mask) else float("inf")
    full_transform = affine23_to33(residual) @ affine23_to33(initial)
    return {
        "method": "lightglue_superpoint",
        "matches": int(key_fixed.shape[0]),
        "inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "residual_rmse": rmse,
        "transform": full_transform,
    }


def cv2_fallback_affine_for_rotation(
    fixed: np.ndarray,
    moving: np.ndarray,
    rotation_deg: float,
    ransac_reproj_threshold: float,
) -> dict[str, Any]:
    initial = sem_rotation_matrix(moving.shape, rotation_deg)
    moving_rot = cv2.warpAffine(
        moving,
        initial,
        (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    fixed_u8 = np.clip(fixed * 255, 0, 255).astype(np.uint8)
    moving_u8 = np.clip(moving_rot * 255, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=2500)
    k0, d0 = sift.detectAndCompute(fixed_u8, None)
    k1, d1 = sift.detectAndCompute(moving_u8, None)
    if d0 is None or d1 is None or len(k0) < 6 or len(k1) < 6:
        raise RuntimeError("SIFT fallback found too few features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(d0, d1, k=2)
    good = [m for m, n in raw if m.distance < 0.78 * n.distance]
    if len(good) < 6:
        raise RuntimeError(f"SIFT fallback found too few matches: {len(good)}")
    key_fixed = np.float32([k0[m.queryIdx].pt for m in good])
    key_moving = np.float32([k1[m.trainIdx].pt for m in good])
    residual, inliers = cv2.estimateAffinePartial2D(
        key_moving,
        key_fixed,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=8000,
        confidence=0.999,
    )
    if residual is None or inliers is None:
        raise RuntimeError("SIFT fallback could not produce an affine transform")
    inlier_mask = inliers.reshape(-1).astype(bool)
    pred = (residual @ np.vstack([key_moving.T, np.ones(key_moving.shape[0])])).T
    err = np.linalg.norm(pred - key_fixed, axis=1)
    rmse = float(np.sqrt(np.mean(err[inlier_mask] ** 2))) if np.any(inlier_mask) else float("inf")
    return {
        "method": "sift_fallback",
        "matches": int(len(good)),
        "inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "residual_rmse": rmse,
        "transform": affine23_to33(residual) @ affine23_to33(initial),
    }


def estimate_adjacent_alignment(
    fixed: MapSemData,
    moving: MapSemData,
    extractor,
    matcher,
    device: str,
    args: argparse.Namespace,
) -> PairAlignment:
    delta = float(moving.spec.angle_deg - fixed.spec.angle_deg)
    candidates = [-delta, delta]
    trial_rows: list[dict[str, Any]] = []
    for rotation_deg in candidates:
        try:
            row = lightglue_affine_for_rotation(
                fixed=fixed.sem_match,
                moving=moving.sem_match,
                rotation_deg=rotation_deg,
                extractor=extractor,
                matcher=matcher,
                device=device,
                ransac_reproj_threshold=args.ransac_reproj_threshold,
            )
            row["rotation_deg"] = rotation_deg
            trial_rows.append(row)
        except Exception as exc:
            if not args.allow_cv2_fallback:
                raise
            row = cv2_fallback_affine_for_rotation(
                fixed=fixed.sem_match,
                moving=moving.sem_match,
                rotation_deg=rotation_deg,
                ransac_reproj_threshold=args.ransac_reproj_threshold,
            )
            row["rotation_deg"] = rotation_deg
            row["method"] = f"{row['method']} after LightGlue failure: {type(exc).__name__}"
            trial_rows.append(row)

    best = max(
        trial_rows,
        key=lambda row: (row["inliers"], row["inlier_ratio"], -row["residual_rmse"], row["matches"]),
    )
    return PairAlignment(
        moving_angle=moving.spec.angle_deg,
        fixed_angle=fixed.spec.angle_deg,
        initial_rotation_deg=float(best["rotation_deg"]),
        method=str(best["method"]),
        matches=int(best["matches"]),
        inliers=int(best["inliers"]),
        inlier_ratio=float(best["inlier_ratio"]),
        residual_rmse=float(best["residual_rmse"]),
        transform_moving_to_fixed=best["transform"].astype(np.float64),
    )


def build_chained_transforms(
    maps: list[MapSemData],
    extractor,
    matcher,
    device: str,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[PairAlignment]]:
    transforms = [np.eye(3, dtype=np.float64)]
    pair_rows: list[PairAlignment] = []
    for index in range(1, len(maps)):
        pair = estimate_adjacent_alignment(maps[index - 1], maps[index], extractor, matcher, device, args)
        pair_rows.append(pair)
        transforms.append(transforms[index - 1] @ pair.transform_moving_to_fixed)
    return transforms, pair_rows


def write_pair_alignment_csv(path: Path, pairs: list[PairAlignment]) -> None:
    rows = []
    for pair in pairs:
        transform = pair.transform_moving_to_fixed
        rows.append(
            {
                "moving_angle": pair.moving_angle,
                "fixed_angle": pair.fixed_angle,
                "initial_rotation_deg": pair.initial_rotation_deg,
                "method": pair.method,
                "matches": pair.matches,
                "inliers": pair.inliers,
                "inlier_ratio": pair.inlier_ratio,
                "residual_rmse": pair.residual_rmse,
                "m00": transform[0, 0],
                "m01": transform[0, 1],
                "m02": transform[0, 2],
                "m10": transform[1, 0],
                "m11": transform[1, 1],
                "m12": transform[1, 2],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transform_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return (matrix @ np.vstack([xy.T, np.ones(len(xy))])).T[:, :2]


def sem_xy_to_index(xy: np.ndarray, nrows: int, ncols: int, sem_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sem_h, sem_w = sem_shape
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x >= 0) & (x < sem_w) & (y >= 0) & (y < sem_h)
    cols = np.floor(x / sem_w * ncols).astype(np.int64)
    rows = np.floor(y / sem_h * nrows).astype(np.int64)
    cols = np.clip(cols, 0, ncols - 1)
    rows = np.clip(rows, 0, nrows - 1)
    return rows * ncols + cols, rows, cols, inside


def sample_image_at_xy(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return image[y, x]


def select_common_high_quality_point(
    maps: list[MapSemData],
    transforms_raw_to_ref: list[np.ndarray],
    ci_min: float,
    selection_stride: int,
) -> dict[str, Any]:
    reference = maps[0]
    count = reference.nrows * reference.ncols
    candidate_indices = np.arange(0, count, max(1, selection_stride), dtype=np.int64)
    ref_x, ref_y = scan_to_sem_xy(candidate_indices, reference.nrows, reference.ncols, reference.sem_gray.shape)
    ref_xy = np.column_stack([ref_x, ref_y])
    common = np.ones(candidate_indices.shape, dtype=bool)
    all_indices: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_raw_xy: list[np.ndarray] = []
    quality_terms = np.zeros(candidate_indices.shape, dtype=np.float64)
    ci_terms = np.zeros(candidate_indices.shape, dtype=np.float64)

    aligned_edges = []
    for sem_map, transform in zip(maps, transforms_raw_to_ref):
        aligned = warp_sem(sem_map.sem_gray, transform)
        edge = filters.sobel(aligned).astype(np.float32)
        aligned_edges.append(edge)

        raw_xy = transform_points(np.linalg.inv(transform), ref_xy)
        indices, rows, cols, inside = sem_xy_to_index(raw_xy, sem_map.nrows, sem_map.ncols, sem_map.sem_gray.shape)
        valid = inside & sem_map.valid[indices] & (sem_map.ci[indices] >= ci_min)
        common &= valid

        valid_iq = sem_map.iq[sem_map.valid]
        lo, hi = np.percentile(valid_iq, [5.0, 99.0]) if valid_iq.size else (0.0, 1.0)
        iq_norm = np.clip((sem_map.iq[indices] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        quality_terms += iq_norm
        ci_terms += np.clip(sem_map.ci[indices], 0.0, 1.0)
        all_indices.append(indices)
        all_rows.append(rows)
        all_cols.append(cols)
        all_raw_xy.append(raw_xy)

    if not np.any(common):
        common = np.ones(candidate_indices.shape, dtype=bool)
        for sem_map, transform in zip(maps, transforms_raw_to_ref):
            raw_xy = transform_points(np.linalg.inv(transform), ref_xy)
            indices, _rows, _cols, inside = sem_xy_to_index(raw_xy, sem_map.nrows, sem_map.ncols, sem_map.sem_gray.shape)
            common &= inside & sem_map.valid[indices]
    if not np.any(common):
        raise RuntimeError("No common valid point exists across the 12 aligned high-resolution maps")

    mean_edge = np.mean(np.stack(aligned_edges, axis=0), axis=0)
    edge_values = sample_image_at_xy(mean_edge, ref_xy)
    edge_norm = np.clip(edge_values / max(float(np.percentile(edge_values, 95)), 1e-6), 0.0, 1.0)
    center = np.array([reference.sem_gray.shape[1] * 0.5, reference.sem_gray.shape[0] * 0.5])
    distance = np.linalg.norm(ref_xy - center[None, :], axis=1)
    distance_norm = distance / max(float(distance.max()), 1e-6)
    score = quality_terms / len(maps) + 0.55 * ci_terms / len(maps) - 0.35 * edge_norm - 0.10 * distance_norm
    score[~common] = -np.inf
    selected_pos = int(np.argmax(score))
    selected_ref_xy = ref_xy[selected_pos]

    selected_per_map: list[dict[str, Any]] = []
    for map_index, sem_map in enumerate(maps):
        raw_xy = all_raw_xy[map_index][selected_pos]
        aligned_xy = transform_points(transforms_raw_to_ref[map_index], raw_xy[None, :])[0]
        selected_per_map.append(
            {
                "selected_index": int(all_indices[map_index][selected_pos]),
                "selected_row": int(all_rows[map_index][selected_pos]),
                "selected_col": int(all_cols[map_index][selected_pos]),
                "selected_sem_xy": (float(raw_xy[0]), float(raw_xy[1])),
                "selected_aligned_xy": (float(aligned_xy[0]), float(aligned_xy[1])),
                "selected_iq": float(sem_map.iq[all_indices[map_index][selected_pos]]),
                "selected_ci": float(sem_map.ci[all_indices[map_index][selected_pos]]),
                "selected_phase": int(sem_map.phase[all_indices[map_index][selected_pos]]),
            }
        )

    return {
        "selected_ref_xy": (float(selected_ref_xy[0]), float(selected_ref_xy[1])),
        "candidate_count": int(common.sum()),
        "selected_score": float(score[selected_pos]),
        "selected": selected_per_map,
        "mean_edge": mean_edge,
    }


def square_polygon(center_xy: tuple[float, float], half_size: float = 18.0) -> np.ndarray:
    x, y = center_xy
    return np.array(
        [
            (x - half_size, y - half_size),
            (x + half_size, y - half_size),
            (x + half_size, y + half_size),
            (x - half_size, y + half_size),
        ],
        dtype=np.float64,
    )


def choose_fixed_orientation_matrix(
    projection,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers,
    variant_name: str,
    args: argparse.Namespace,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    from project_edax_oim_to_sphere import orientation_candidates
    from single_kikuchi_pc_finetune import make_stride_indices, score_with_directions

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


def process_highres_map(
    spec: HighResMapSpec,
    sem_map: MapSemData,
    selection: dict[str, Any],
    transform_raw_to_ref: np.ndarray,
    h5_path: Path,
    up2_root: Path,
    master_samplers,
    output_dir: Path,
    args: argparse.Namespace,
    fixed_orientation_variant: str | None,
) -> ProcessedMap:
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
    pc_original_xy, pc_original_vector = pc_crystal_vector(projection, original.pc, orientation_matrix)
    pc_refined_xy, pc_refined_vector = pc_crystal_vector(projection, refined.pc, orientation_matrix)
    vectors, values = crystal_vectors_for_patch(
        projection=projection,
        pc=refined.pc,
        matrix=orientation_matrix,
        values=images["enhanced"],
        mask=mask,
        max_points=args.max_3d_points_per_pattern,
    )

    aligned_sem = warp_sem(sem_map.sem_gray, transform_raw_to_ref)
    aligned_poly = square_polygon(selection["selected_aligned_xy"], args.selection_marker_half_size)
    raw_poly = transform_points(np.linalg.inv(transform_raw_to_ref), aligned_poly)

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
        pc_original_xy=pc_original_xy,
        pc_refined_xy=pc_refined_xy,
        pc_original_crystal_vector=pc_original_vector,
        pc_refined_crystal_vector=pc_refined_vector,
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
        sem_gray=sem_map.sem_gray,
        aligned_sem_gray=aligned_sem,
        raw_poly=raw_poly,
        aligned_poly=aligned_poly,
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


def apply_cubic_symmetry_axis_prior_30deg(results: list[ProcessedMap]) -> dict[str, Any]:
    if len(results) < 2:
        raise ValueError("At least two high-resolution maps are required for a 30-degree axis prior")
    symmetries = cubic_proper_symmetry_matrices()
    base_matrices = [orthonormalize_rotation(result.base_orientation_matrix) for result in results]
    final_candidates = [
        [orthonormalize_rotation(base_matrix @ symmetry_matrix) for _name, symmetry_matrix in symmetries]
        for base_matrix in base_matrices
    ]
    n_sym = len(symmetries)
    best: dict[str, Any] | None = None
    for s0 in range(n_sym):
        f0 = final_candidates[0][s0]
        for s1 in range(n_sym):
            q30 = f0.T @ final_candidates[1][s1]
            angle30 = rotation_angle_deg(q30)
            score = 4.0 * abs(angle30 - 30.0) / 30.0
            selected = [s0, s1]
            residuals = [0.0, 0.0]
            for step in range(2, len(results)):
                expected = np.linalg.matrix_power(q30, step)
                options = []
                for sym_index in range(n_sym):
                    relative = f0.T @ final_candidates[step][sym_index]
                    closure = float(np.linalg.norm(relative - expected, ord="fro"))
                    options.append((closure, sym_index))
                closure, sym_index = min(options, key=lambda item: item[0])
                selected.append(int(sym_index))
                residuals.append(float(closure))
                score += closure
            if best is None or score < best["score"]:
                best = {
                    "score": float(score),
                    "symmetry_indices": tuple(selected),
                    "q30": q30,
                    "common_axis": rotation_axis(q30),
                    "angle30_deg": float(angle30),
                    "residuals": residuals,
                }
    if best is None:
        raise RuntimeError("Could not fit cubic symmetry axis prior for high-resolution 30-degree maps")

    best["angles_by_step_deg"] = [
        float(rotation_angle_deg(np.linalg.matrix_power(best["q30"], step))) for step in range(len(results))
    ]
    for result, symmetry_index in zip(results, best["symmetry_indices"]):
        symmetry_name, symmetry_matrix = symmetries[int(symmetry_index)]
        result.original_patch = project_vectors_patch(result.crystal_vectors, result.crystal_values)
        result.symmetry_name = symmetry_name
        result.symmetry_matrix = symmetry_matrix
        result.axis_prior_score = float(best["score"])
        result.orientation_matrix = result.base_orientation_matrix @ symmetry_matrix
        result.orientation_variant = f"h5_{result.base_orientation_variant} @ cubic_sym({symmetry_name})"
        result.crystal_vectors = (result.crystal_vectors @ symmetry_matrix).astype(np.float32)
        result.pc_original_crystal_vector = (result.pc_original_crystal_vector @ symmetry_matrix).astype(np.float32)
        result.pc_original_crystal_vector /= max(np.linalg.norm(result.pc_original_crystal_vector), 1e-12)
        result.pc_refined_crystal_vector = (result.pc_refined_crystal_vector @ symmetry_matrix).astype(np.float32)
        result.pc_refined_crystal_vector /= max(np.linalg.norm(result.pc_refined_crystal_vector), 1e-12)
        result.refined_patch = project_vectors_patch(result.crystal_vectors, result.crystal_values)
    return best


def axis_angle_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    angle = math.radians(float(angle_deg))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def fit_exact_axis_sequence(
    matrices: list[np.ndarray],
    angles_deg: list[float],
    initial_axis: np.ndarray,
) -> dict[str, Any]:
    relative_angles = [float(angle - angles_deg[0]) for angle in angles_deg]

    def solve_for_axis(axis_guess: np.ndarray) -> dict[str, Any]:
        axis = axis_guess.astype(np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = initial_axis.astype(np.float64)
            norm = np.linalg.norm(axis)
        axis /= max(norm, 1e-12)
        rotations = [axis_angle_rotation(axis, angle) for angle in relative_angles]
        reference_votes = [matrix @ rotation.T for matrix, rotation in zip(matrices, rotations)]
        reference = orthonormalize_rotation(np.mean(reference_votes, axis=0))
        fitted = [orthonormalize_rotation(reference @ rotation) for rotation in rotations]
        residuals = [float(np.linalg.norm(fit - matrix, ord="fro")) for fit, matrix in zip(fitted, matrices)]
        return {
            "axis": axis,
            "reference_matrix": reference,
            "fitted_matrices": fitted,
            "residuals": residuals,
            "score": float(np.mean(np.square(residuals))),
        }

    x0 = initial_axis.astype(np.float64)
    x0 /= max(np.linalg.norm(x0), 1e-12)

    def objective(x: np.ndarray) -> float:
        return solve_for_axis(x)["score"]

    best_result = solve_for_axis(x0)
    for start in [
        x0,
        -x0,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]:
        result = optimize.minimize(objective, start, method="Nelder-Mead", options={"maxiter": 900, "xatol": 1e-8, "fatol": 1e-10})
        candidate = solve_for_axis(result.x)
        if candidate["score"] < best_result["score"]:
            best_result = candidate

    fitted = best_result["fitted_matrices"]
    q30 = fitted[0].T @ fitted[1]
    best_result["angle30_deg"] = float(rotation_angle_deg(q30))
    best_result["angles_by_step_deg"] = [
        float(rotation_angle_deg(fitted[0].T @ matrix)) for matrix in fitted
    ]
    return best_result


def clone_result_for_axis_locked(
    result: ProcessedMap,
    locked_matrix: np.ndarray,
    locked_label: str,
    original_score_override: float | None = None,
    refined_score_override: float | None = None,
) -> ProcessedMap:
    old_matrix = result.orientation_matrix.astype(np.float64)
    correction = old_matrix.T @ locked_matrix
    new_vectors = (result.crystal_vectors.astype(np.float64) @ correction).astype(np.float32)
    new_vectors /= np.linalg.norm(new_vectors, axis=1, keepdims=True) + 1e-12

    original_pc = (result.pc_original_crystal_vector.astype(np.float64) @ correction).astype(np.float32)
    original_pc /= max(np.linalg.norm(original_pc), 1e-12)
    refined_pc = (result.pc_refined_crystal_vector.astype(np.float64) @ correction).astype(np.float32)
    refined_pc /= max(np.linalg.norm(refined_pc), 1e-12)
    refined_patch = project_vectors_patch(new_vectors, result.crystal_values)

    original_score = result.original_score if original_score_override is None else float(original_score_override)
    refined_score = result.refined_score if refined_score_override is None else float(refined_score_override)
    return ProcessedMap(
        spec=result.spec,
        selected_index=result.selected_index,
        selected_row=result.selected_row,
        selected_col=result.selected_col,
        selected_sem_xy=result.selected_sem_xy,
        selected_aligned_xy=result.selected_aligned_xy,
        selected_iq=result.selected_iq,
        selected_ci=result.selected_ci,
        selected_phase=result.selected_phase,
        candidate_count=result.candidate_count,
        pc_original=result.pc_original,
        pc_refined=result.pc_refined,
        pc_delta=result.pc_delta,
        pc_original_xy=result.pc_original_xy,
        pc_refined_xy=result.pc_refined_xy,
        pc_original_crystal_vector=original_pc,
        pc_refined_crystal_vector=refined_pc,
        base_orientation_variant=result.base_orientation_variant,
        base_orientation_matrix=result.base_orientation_matrix,
        orientation_variant=locked_label,
        orientation_matrix=locked_matrix.astype(np.float64),
        symmetry_name=result.symmetry_name,
        symmetry_matrix=result.symmetry_matrix,
        axis_prior_score=result.axis_prior_score,
        original_score=original_score,
        refined_score=refined_score,
        score_gain=refined_score - original_score,
        sem_gray=result.sem_gray,
        aligned_sem_gray=result.aligned_sem_gray,
        raw_poly=result.raw_poly,
        aligned_poly=result.aligned_poly,
        raw_pattern_display=result.raw_pattern_display,
        corrected_pattern=result.corrected_pattern,
        enhanced_pattern=result.enhanced_pattern,
        band_pattern=result.band_pattern,
        detector_mask=result.detector_mask,
        detector_patch=result.detector_patch,
        original_patch=result.original_patch,
        refined_patch=refined_patch,
        crystal_vectors=new_vectors,
        crystal_values=result.crystal_values,
    )


def build_axis_locked_results(results: list[ProcessedMap], rough_fit: dict[str, Any]) -> tuple[list[ProcessedMap], dict[str, Any]]:
    matrices = [orthonormalize_rotation(result.orientation_matrix) for result in results]
    angles = [float(result.spec.angle_deg) for result in results]
    fit = fit_exact_axis_sequence(matrices, angles, rough_fit["common_axis"])
    locked_results: list[ProcessedMap] = []
    for result, matrix in zip(results, fit["fitted_matrices"]):
        locked_results.append(
            clone_result_for_axis_locked(
                result=result,
                locked_matrix=matrix,
                locked_label=f"axis_locked_exact_30deg_from_{result.orientation_variant}",
            )
        )
    fit["common_axis"] = fit["axis"]
    return locked_results, fit


def write_axis_locked_summary(path: Path, results: list[ProcessedMap], fit: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    axis = fit["common_axis"]
    for index, result in enumerate(results):
        rows.append(
            {
                "angle_deg": result.spec.angle_deg,
                "area": result.spec.area,
                "orientation_variant": result.orientation_variant,
                "axis_locked_score": fit["score"],
                "fit_residual": fit["residuals"][index],
                "angle_from_reference_deg": fit["angles_by_step_deg"][index],
                "angle30_deg": fit["angle30_deg"],
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "pc_refined_sphere_x": result.pc_refined_crystal_vector[0],
                "pc_refined_sphere_y": result.pc_refined_crystal_vector[1],
                "pc_refined_sphere_z": result.pc_refined_crystal_vector[2],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = rotvec.astype(np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    return axis_angle_rotation(rotvec / angle, math.degrees(angle))


def build_sequence_scoring_items(
    results: list[ProcessedMap],
    h5_path: Path,
    up2_root: Path,
    args: argparse.Namespace,
) -> list[SequenceScoringItem]:
    items: list[SequenceScoringItem] = []
    for result in results:
        projection = read_edax_inputs(
            EdaxMapInputs(
                h5_path=h5_path,
                up2_path=up2_root / result.spec.up2_name,
                map_group=result.spec.h5_group,
                pattern_index=result.selected_index,
            )
        )
        mask, _circle = centered_circular_detector_mask(projection.pattern.shape, args.mask_radius_fraction)
        images = build_preprocessed_images(projection.pattern, mask)
        indices = make_stride_indices(mask, args.sequence_stride)
        detector_directions = detector_directions_with_pc(projection, result.pc_refined)
        items.append(
            SequenceScoringItem(
                result=result,
                detector_directions=detector_directions,
                indices=indices,
                exp_corrected_values=images["enhanced"].ravel()[indices],
                exp_band_values=images["band"].ravel()[indices],
            )
        )
    return items


def score_sequence_matrix(
    item: SequenceScoringItem,
    matrix: np.ndarray,
    master_samplers,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    return score_with_directions(
        detector_directions=item.detector_directions,
        matrix=matrix,
        indices=item.indices,
        exp_corrected_values=item.exp_corrected_values,
        exp_band_values=item.exp_band_values,
        samplers=master_samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )


def geodesic_angle_rad(matrix: np.ndarray) -> float:
    return math.radians(rotation_angle_deg(matrix))


def physical_sequence_matrices(
    reference_matrix: np.ndarray,
    axis: np.ndarray,
    angles_deg: list[float],
    side: str,
) -> list[np.ndarray]:
    axis = axis.astype(np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    relative_angles = [float(angle - angles_deg[0]) for angle in angles_deg]
    matrices: list[np.ndarray] = []
    for angle in relative_angles:
        rotation = axis_angle_rotation(axis, angle)
        if side == "right":
            matrix = reference_matrix @ rotation
        elif side == "left":
            matrix = rotation @ reference_matrix
        else:
            raise ValueError(f"Unknown sequence side {side!r}")
        matrices.append(orthonormalize_rotation(matrix))
    return matrices


def optimize_match_preserving_axis_sequence(
    results: list[ProcessedMap],
    geometric_fit: dict[str, Any],
    scoring_items: list[SequenceScoringItem],
    master_samplers,
    args: argparse.Namespace,
) -> tuple[list[ProcessedMap], dict[str, Any]]:
    angles = [float(result.spec.angle_deg) for result in results]
    free_matrices = [orthonormalize_rotation(result.orientation_matrix) for result in results]
    free_scores = [
        score_sequence_matrix(item, matrix, master_samplers, args)[2]
        for item, matrix in zip(scoring_items, free_matrices)
    ]

    starts: list[tuple[str, np.ndarray, np.ndarray]] = []
    starts.append(("right", geometric_fit["reference_matrix"], geometric_fit["common_axis"]))
    starts.append(("right", free_matrices[0], geometric_fit["common_axis"]))
    starts.append(("right", geometric_fit["reference_matrix"], -geometric_fit["common_axis"]))
    starts.append(("left", geometric_fit["reference_matrix"], geometric_fit["common_axis"]))
    starts.append(("left", free_matrices[0], geometric_fit["common_axis"]))
    starts.append(("left", geometric_fit["reference_matrix"], -geometric_fit["common_axis"]))

    best: dict[str, Any] | None = None
    for side, start_reference, start_axis in starts:
        start_reference = orthonormalize_rotation(start_reference)
        start_axis = start_axis.astype(np.float64)
        start_axis /= max(np.linalg.norm(start_axis), 1e-12)

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
            reference = orthonormalize_rotation(start_reference @ rotvec_to_matrix(params[:3]))
            axis = params[3:6].astype(np.float64)
            if np.linalg.norm(axis) < 1e-10:
                axis = start_axis.copy()
            axis /= max(np.linalg.norm(axis), 1e-12)
            matrices = physical_sequence_matrices(reference, axis, angles, side)
            return reference, axis, matrices

        def evaluate(params: np.ndarray) -> dict[str, Any]:
            reference, axis, matrices = unpack(params)
            scores = [
                score_sequence_matrix(item, matrix, master_samplers, args)
                for item, matrix in zip(scoring_items, matrices)
            ]
            combined = np.array([score[2] for score in scores], dtype=np.float64)
            prior = 0.0
            if args.sequence_prior_weight > 0:
                distances = [
                    geodesic_angle_rad(matrix.T @ free_matrix) ** 2
                    for matrix, free_matrix in zip(matrices, free_matrices)
                ]
                prior = float(args.sequence_prior_weight * np.mean(distances))
            objective = float(-combined.mean() + prior)
            return {
                "objective": objective,
                "reference_matrix": reference,
                "axis": axis,
                "matrices": matrices,
                "scores": scores,
                "combined_scores": combined,
                "prior": prior,
                "side": side,
            }

        x0 = np.zeros(6, dtype=np.float64)
        x0[3:6] = start_axis
        initial = evaluate(x0)
        result = optimize.minimize(
            lambda params: evaluate(params)["objective"],
            x0,
            method="Powell",
            options={"maxiter": args.sequence_maxiter, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
        )
        candidate = evaluate(result.x)
        candidate["optimizer_success"] = bool(result.success)
        candidate["optimizer_message"] = str(result.message)
        candidate["initial_objective"] = initial["objective"]
        if best is None or candidate["objective"] < best["objective"]:
            best = candidate

    if best is None:
        raise RuntimeError("No match-preserving physical 30-degree sequence could be optimized")

    sequence_scores = best["combined_scores"]
    mean_free_score = float(np.mean(free_scores))
    mean_sequence_score = float(np.mean(sequence_scores))
    score_drop = mean_free_score - mean_sequence_score
    accepted = bool(score_drop <= args.sequence_max_score_drop)
    q30 = best["matrices"][0].T @ best["matrices"][1]
    best.update(
        {
            "free_scores": free_scores,
            "mean_free_score": mean_free_score,
            "mean_sequence_score": mean_sequence_score,
            "score_drop": float(score_drop),
            "accepted": accepted,
            "angle30_deg": float(rotation_angle_deg(q30)),
            "common_axis": best["axis"],
            "angles_by_step_deg": [
                float(rotation_angle_deg(best["matrices"][0].T @ matrix)) for matrix in best["matrices"]
            ],
        }
    )

    sequence_results: list[ProcessedMap] = []
    for result, matrix, free_score, sequence_score in zip(results, best["matrices"], free_scores, sequence_scores):
        sequence_results.append(
            clone_result_for_axis_locked(
                result=result,
                locked_matrix=matrix,
                locked_label=f"match_preserving_30deg_{best['side']}_sequence",
                original_score_override=float(free_score),
                refined_score_override=float(sequence_score),
            )
        )
    return sequence_results, best


def write_match_preserving_axis_summary(path: Path, results: list[ProcessedMap], fit: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    axis = fit["common_axis"]
    for index, result in enumerate(results):
        rows.append(
            {
                "angle_deg": result.spec.angle_deg,
                "area": result.spec.area,
                "accepted": fit["accepted"],
                "sequence_side": fit["side"],
                "free_score": fit["free_scores"][index],
                "sequence_score": fit["combined_scores"][index],
                "score_drop": fit["free_scores"][index] - fit["combined_scores"][index],
                "mean_free_score": fit["mean_free_score"],
                "mean_sequence_score": fit["mean_sequence_score"],
                "mean_score_drop": fit["score_drop"],
                "angle_from_reference_deg": fit["angles_by_step_deg"][index],
                "angle30_deg": fit["angle30_deg"],
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "pc_refined_sphere_x": result.pc_refined_crystal_vector[0],
                "pc_refined_sphere_y": result.pc_refined_crystal_vector[1],
                "pc_refined_sphere_z": result.pc_refined_crystal_vector[2],
                "orientation_variant": result.orientation_variant,
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_symmetry_summary(path: Path, results: list[ProcessedMap], fit: dict[str, Any]) -> None:
    axis = fit["common_axis"]
    rows: list[dict[str, Any]] = []
    for step, result in enumerate(results):
        rows.append(
            {
                "angle_deg": result.spec.angle_deg,
                "area": result.spec.area,
                "selected_cubic_symmetry": result.symmetry_name,
                "axis_prior_score": fit["score"],
                "step_angle_deg": fit["angles_by_step_deg"][step],
                "closure_residual": fit["residuals"][step],
                "angle30_deg": fit["angle30_deg"],
                "common_axis_x": axis[0],
                "common_axis_y": axis[1],
                "common_axis_z": axis[2],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, results: list[ProcessedMap], selected_ref_xy: tuple[float, float]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "angle_deg": result.spec.angle_deg,
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
                "common_ref_x": selected_ref_xy[0],
                "common_ref_y": selected_ref_xy[1],
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
                "pc_original_pixel_x": result.pc_original_xy[0],
                "pc_original_pixel_y": result.pc_original_xy[1],
                "pc_refined_pixel_x": result.pc_refined_xy[0],
                "pc_refined_pixel_y": result.pc_refined_xy[1],
                "pc_original_sphere_x": result.pc_original_crystal_vector[0],
                "pc_original_sphere_y": result.pc_original_crystal_vector[1],
                "pc_original_sphere_z": result.pc_original_crystal_vector[2],
                "pc_refined_sphere_x": result.pc_refined_crystal_vector[0],
                "pc_refined_sphere_y": result.pc_refined_crystal_vector[1],
                "pc_refined_sphere_z": result.pc_refined_crystal_vector[2],
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


def draw_polygon(ax, polygon: np.ndarray, color: str = "cyan", lw: float = 1.0) -> None:
    ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]], color=color, lw=lw)


def draw_pc_marker_on_pattern(ax, result: ProcessedMap) -> None:
    ax.scatter([result.pc_original_xy[0]], [result.pc_original_xy[1]], marker="+", s=130, c="white", linewidths=2.0)
    ax.scatter(
        [result.pc_refined_xy[0]],
        [result.pc_refined_xy[1]],
        marker="o",
        s=72,
        facecolors="none",
        edgecolors=result.spec.color,
        linewidths=1.8,
    )


def save_alignment_overview(path: Path, maps: list[MapSemData], transforms: list[np.ndarray], pairs: list[PairAlignment], selected_ref_xy: tuple[float, float]) -> None:
    fig, axes = plt.subplots(3, len(maps), figsize=(2.6 * len(maps), 7.6), dpi=180)
    for index, (sem_map, transform) in enumerate(zip(maps, transforms)):
        axes[0, index].imshow(sem_map.sem_gray, cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(f"{sem_map.spec.angle_deg} deg raw")
        axes[0, index].axis("off")

        aligned = warp_sem(sem_map.sem_gray, transform)
        raw_xy = transform_points(np.linalg.inv(transform), np.array([selected_ref_xy], dtype=np.float64))[0]
        axes[1, index].imshow(aligned, cmap="gray", vmin=0, vmax=1)
        axes[1, index].scatter([selected_ref_xy[0]], [selected_ref_xy[1]], s=38, facecolors="none", edgecolors="red", linewidths=1.3)
        axes[1, index].set_title("aligned to 0 deg")
        axes[1, index].axis("off")

        axes[2, index].imshow(sem_map.sem_gray, cmap="gray", vmin=0, vmax=1)
        axes[2, index].scatter([raw_xy[0]], [raw_xy[1]], s=38, facecolors="none", edgecolors="red", linewidths=1.3)
        if index == 0:
            subtitle = "reference"
        else:
            pair = pairs[index - 1]
            subtitle = f"{pair.inliers}/{pair.matches} inl, rmse={pair.residual_rmse:.2f}"
        axes[2, index].set_title(subtitle, fontsize=8)
        axes[2, index].axis("off")
    fig.suptitle("Pt high-resolution 12-map SEM alignment: adjacent 30-degree LightGlue/SuperPoint chain")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_selection_preview(path: Path, results: list[ProcessedMap], selected_ref_xy: tuple[float, float]) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(2.6 * len(results), 5.2), dpi=180)
    for index, result in enumerate(results):
        ax = axes[0, index]
        ax.imshow(result.sem_gray, cmap="gray", vmin=0, vmax=1)
        draw_polygon(ax, result.raw_poly)
        ax.scatter([result.selected_sem_xy[0]], [result.selected_sem_xy[1]], s=42, facecolors="none", edgecolors="red", linewidths=1.3)
        ax.set_title(f"{result.spec.angle_deg} deg raw\nidx={result.selected_index}", fontsize=8)
        ax.axis("off")

        ax = axes[1, index]
        ax.imshow(result.aligned_sem_gray, cmap="gray", vmin=0, vmax=1)
        draw_polygon(ax, result.aligned_poly)
        ax.scatter([selected_ref_xy[0]], [selected_ref_xy[1]], s=42, facecolors="none", edgecolors="red", linewidths=1.3)
        ax.set_title(f"IQ={result.selected_iq:.0f}, CI={result.selected_ci:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Same physical SEM point selected after chained alignment")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_kikuchi_pc_patterns(path: Path, results: list[ProcessedMap]) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(2.45 * len(results), 5.0), dpi=190)
    for index, result in enumerate(results):
        ax = axes[0, index]
        ax.imshow(masked_image(result.raw_pattern_display, result.detector_mask), cmap="gray", vmin=0, vmax=1)
        draw_pc_marker_on_pattern(ax, result)
        ax.set_title(f"{result.spec.angle_deg} deg raw", fontsize=8)
        ax.axis("off")

        ax = axes[1, index]
        ax.imshow(masked_image(result.enhanced_pattern, result.detector_mask), cmap="gray", vmin=0, vmax=1)
        draw_pc_marker_on_pattern(ax, result)
        ax.set_title("preprocessed", fontsize=8)
        ax.axis("off")
    fig.suptitle("Selected high-resolution Kikuchi patterns with PC markers: white +=H5 PC, colored circle=refined PC")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_process_overview(path: Path, results: list[ProcessedMap], master_texture: np.ndarray) -> None:
    fig, axes = plt.subplots(len(results), 5, figsize=(18.0, 2.55 * len(results)), dpi=150)
    for row, result in enumerate(results):
        axes[row, 0].imshow(result.aligned_sem_gray, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].scatter([result.selected_aligned_xy[0]], [result.selected_aligned_xy[1]], s=34, facecolors="none", edgecolors="red", linewidths=1.2)
        axes[row, 0].set_title(f"{result.spec.angle_deg} deg aligned SEM\nidx={result.selected_index}, CI={result.selected_ci:.3f}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(masked_image(result.enhanced_pattern, result.detector_mask), cmap="gray", vmin=0, vmax=1)
        draw_pc_marker_on_pattern(axes[row, 1], result)
        axes[row, 1].set_title("Kikuchi preprocessed + PC")
        axes[row, 1].axis("off")

        imshow_sphere(axes[row, 2], result.detector_patch[0], result.detector_patch[1], "Detector sphere\nEDAX PC")
        axes[row, 3].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(axes[row, 3], result.original_patch[0], result.original_patch[1], "H5 orientation\nbefore symmetry")
        axes[row, 4].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(axes[row, 4], result.refined_patch[0], result.refined_patch[1], f"same sphere\nsym={result.symmetry_name}")
    fig.suptitle("Pt high-resolution 30-degree sequence: SEM alignment, Kikuchi preprocessing, sphere projection")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_same_sphere_lon_colat(path: Path, results: list[ProcessedMap], master_texture: np.ndarray) -> None:
    master_display = normalize_values(master_texture, np.isfinite(master_texture), low=0.5, high=99.5)
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=230)
    ax.imshow(master_display, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.72)
    for result in results:
        nrows, ncols = result.refined_patch[1].shape
        lon_grid = np.linspace(-180.0, 180.0, ncols)
        colat_grid = np.linspace(180.0, 0.0, nrows)
        ax.contour(lon_grid, colat_grid, result.refined_patch[1].astype(float), levels=[0.5], colors=[result.spec.color], linewidths=0.95)
        lon, colat = vector_to_lon_colat_deg(result.pc_refined_crystal_vector)
        ax.scatter([lon], [colat], s=42, facecolors=result.spec.color, edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_title("Twelve selected Kikuchi patterns corrected onto one cubic-symmetry-selected master sphere")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("colatitude (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pc_anchor_lon_colat(path: Path, results: list[ProcessedMap], master_texture: np.ndarray) -> None:
    master_display = normalize_values(master_texture, np.isfinite(master_texture), low=0.5, high=99.5)
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=240)
    ax.imshow(master_display, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto", alpha=0.74)
    for result in results:
        lon_original, colat_original = vector_to_lon_colat_deg(result.pc_original_crystal_vector)
        lon_refined, colat_refined = vector_to_lon_colat_deg(result.pc_refined_crystal_vector)
        ax.scatter([lon_original], [colat_original], marker="+", s=130, c=result.spec.color, linewidths=2.0, zorder=5)
        ax.scatter([lon_refined], [colat_refined], marker="o", s=70, facecolors=result.spec.color, edgecolors="white", linewidths=1.0, zorder=6)
        ax.plot([lon_original, lon_refined], [colat_original, colat_refined], color=result.spec.color, lw=0.8, alpha=0.75, zorder=4)
        ax.text(lon_refined + 2.0, colat_refined - 1.8, f"{result.spec.angle_deg}", color=result.spec.color, fontsize=8, weight="bold")
    ax.set_title("PC crystallographic anchors on the same master sphere: +=H5/EDAX PC, filled dot=refined PC")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("colatitude (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_pc_markers_on_sphere(ax, results: list[ProcessedMap], radius: float = 1.055) -> None:
    for result in results:
        vector = result.pc_refined_crystal_vector.astype(np.float64)
        vector /= max(np.linalg.norm(vector), 1e-12)
        ax.scatter(
            [vector[0] * radius],
            [vector[1] * radius],
            [vector[2] * radius],
            s=66,
            c=[result.spec.color],
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
        ax.text(
            vector[0] * (radius + 0.045),
            vector[1] * (radius + 0.045),
            vector[2] * (radius + 0.045),
            f"{result.spec.angle_deg}",
            color=result.spec.color,
            fontsize=7,
            weight="bold",
        )


def save_pc_anchor_3d(path: Path, results: list[ProcessedMap], master_samplers, symmetry_fit: dict[str, Any]) -> None:
    surface_data = master_surface(master_samplers, lon_count=180, colat_count=90)
    axis = symmetry_fit["common_axis"].astype(np.float64)
    pc_mean = np.mean([result.pc_refined_crystal_vector.astype(np.float64) for result in results], axis=0)
    if float(np.dot(axis, pc_mean)) < 0:
        axis = -axis
    views = [
        ("PC anchors viewed along fitted 30-degree axis", axis),
        ("PC anchors viewed from their mean direction", pc_mean),
        ("Opposite side of fitted axis", -axis),
        ("Oblique PC anchor view", axis + 0.8 * pc_mean),
    ]
    fig = plt.figure(figsize=(14, 14), dpi=210)
    x, y, z, values = surface_data
    facecolors = plt.cm.gray(values)
    facecolors[..., 3] = 0.26
    for index, (title, view_vector) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        draw_inplane_axis(ax, axis)
        scatter_pc_markers_on_sphere(ax, results)
        setup_3d_axis(ax, title)
        elev, azim = view_angles_from_vector(view_vector)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
    fig.suptitle("Standalone PC crystallographic anchors on the standard Kikuchi sphere")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_same_sphere_3d(path: Path, results: list[ProcessedMap], master_samplers, symmetry_fit: dict[str, Any]) -> None:
    surface_data = same_sphere_composite_surface(results, master_samplers)
    axis = symmetry_fit["common_axis"].astype(np.float64)
    reference_center = np.mean([np.mean(result.crystal_vectors.astype(np.float64), axis=0) for result in results], axis=0)
    if float(np.dot(axis, reference_center)) < 0:
        axis = -axis
    views = [
        ("All 12 patches viewed along fitted 30-degree axis", axis),
        ("Opposite fitted axis", -axis),
        ("Mean patch-normal view", reference_center),
        ("Oblique same-sphere view", axis + 0.8 * reference_center),
    ]
    fig = plt.figure(figsize=(14, 14), dpi=210)
    for index, (title, view_vector) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_same_sphere_view(ax, surface_data, title=title, view_vector=view_vector, axis=axis)
    fig.suptitle("All high-resolution 30-degree Kikuchi patterns on one master sphere")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_transforms_npz(path: Path, transforms: list[np.ndarray]) -> None:
    np.savez(path, **{f"angle_{angle:03d}": transform for angle, transform in zip(range(0, 360, 30), transforms)})


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_map_specs()
    maps = read_highres_sem_data(args.h5, specs)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    extractor, matcher = make_lightglue_models(args.max_keypoints, device)
    transforms, pair_rows = build_chained_transforms(maps, extractor, matcher, device, args)
    write_pair_alignment_csv(args.output_dir / "pt_highres_pair_alignments.csv", pair_rows)
    save_transforms_npz(args.output_dir / "pt_highres_sem_transforms_raw_to_angle0.npz", transforms)

    selection = select_common_high_quality_point(
        maps=maps,
        transforms_raw_to_ref=transforms,
        ci_min=args.ci_min,
        selection_stride=args.selection_stride,
    )
    save_alignment_overview(
        args.output_dir / "pt_highres_sem_lightglue_alignment_overview.png",
        maps,
        transforms,
        pair_rows,
        selection["selected_ref_xy"],
    )

    master_samplers = build_master_samplers(args.master)
    master_texture = build_master_lon_colat(master_samplers.upper_corrected, master_samplers.lower_corrected)
    results: list[ProcessedMap] = []
    fixed_orientation_variant: str | None = None
    for index, (spec, sem_map, transform) in enumerate(zip(specs, maps, transforms)):
        result = process_highres_map(
            spec=spec,
            sem_map=sem_map,
            selection={**selection["selected"][index], "candidate_count": selection["candidate_count"]},
            transform_raw_to_ref=transform,
            h5_path=args.h5,
            up2_root=args.up2_root,
            master_samplers=master_samplers,
            output_dir=args.output_dir,
            args=args,
            fixed_orientation_variant=fixed_orientation_variant,
        )
        results.append(result)
        if index == 0 and args.orientation_mode == "reference_variant":
            fixed_orientation_variant = result.base_orientation_variant

    symmetry_fit = apply_cubic_symmetry_axis_prior_30deg(results)
    write_summary_csv(args.output_dir / "pt_highres_30deg_spherical_calibration_summary.csv", results, selection["selected_ref_xy"])
    write_symmetry_summary(args.output_dir / "pt_highres_30deg_cubic_symmetry_axis_prior_summary.csv", results, symmetry_fit)

    axis_locked_results, axis_locked_fit = build_axis_locked_results(results, symmetry_fit)
    if args.save_geometric_axis_locked:
        write_summary_csv(
            args.output_dir / "pt_highres_geometric_axis_locked_30deg_spherical_calibration_summary.csv",
            axis_locked_results,
            selection["selected_ref_xy"],
        )
        write_axis_locked_summary(
            args.output_dir / "pt_highres_geometric_axis_locked_30deg_axis_summary.csv",
            axis_locked_results,
            axis_locked_fit,
        )

    scoring_items = build_sequence_scoring_items(results, args.h5, args.up2_root, args)
    match_axis_results, match_axis_fit = optimize_match_preserving_axis_sequence(
        results=results,
        geometric_fit=axis_locked_fit,
        scoring_items=scoring_items,
        master_samplers=master_samplers,
        args=args,
    )
    write_summary_csv(
        args.output_dir / "pt_highres_match_preserving_30deg_spherical_calibration_summary.csv",
        match_axis_results,
        selection["selected_ref_xy"],
    )
    write_match_preserving_axis_summary(
        args.output_dir / "pt_highres_match_preserving_30deg_axis_summary.csv",
        match_axis_results,
        match_axis_fit,
    )

    save_selection_preview(args.output_dir / "pt_highres_same_point_selection.png", results, selection["selected_ref_xy"])
    save_kikuchi_pc_patterns(args.output_dir / "pt_highres_selected_kikuchi_pc_patterns.png", results)
    save_process_overview(args.output_dir / "pt_highres_spherical_calibration_workflow.png", results, master_texture)
    save_same_sphere_lon_colat(args.output_dir / "pt_highres_same_sphere_lon_colat.png", results, master_texture)
    save_same_sphere_3d(args.output_dir / "pt_highres_same_sphere_3d.png", results, master_samplers, symmetry_fit)
    save_pc_anchor_lon_colat(args.output_dir / "pt_highres_pc_anchor_lon_colat.png", results, master_texture)
    save_pc_anchor_3d(args.output_dir / "pt_highres_pc_anchor_3d.png", results, master_samplers, symmetry_fit)
    if args.save_geometric_axis_locked:
        save_same_sphere_lon_colat(
            args.output_dir / "pt_highres_geometric_axis_locked_same_sphere_lon_colat.png",
            axis_locked_results,
            master_texture,
        )
        save_same_sphere_3d(
            args.output_dir / "pt_highres_geometric_axis_locked_same_sphere_3d.png",
            axis_locked_results,
            master_samplers,
            axis_locked_fit,
        )
        save_pc_anchor_lon_colat(
            args.output_dir / "pt_highres_geometric_axis_locked_pc_anchor_lon_colat.png",
            axis_locked_results,
            master_texture,
        )
        save_pc_anchor_3d(
            args.output_dir / "pt_highres_geometric_axis_locked_pc_anchor_3d.png",
            axis_locked_results,
            master_samplers,
            axis_locked_fit,
        )
    save_same_sphere_lon_colat(
        args.output_dir / "pt_highres_match_preserving_same_sphere_lon_colat.png",
        match_axis_results,
        master_texture,
    )
    save_same_sphere_3d(
        args.output_dir / "pt_highres_match_preserving_same_sphere_3d.png",
        match_axis_results,
        master_samplers,
        match_axis_fit,
    )
    save_pc_anchor_lon_colat(
        args.output_dir / "pt_highres_match_preserving_pc_anchor_lon_colat.png",
        match_axis_results,
        master_texture,
    )
    save_pc_anchor_3d(
        args.output_dir / "pt_highres_match_preserving_pc_anchor_3d.png",
        match_axis_results,
        master_samplers,
        match_axis_fit,
    )

    print(f"Saved outputs to {args.output_dir}")
    print(
        "High-res 30-degree cubic axis prior: "
        f"score={symmetry_fit['score']:.5f}, angle30={symmetry_fit['angle30_deg']:.2f}, "
        f"axis={tuple(round(float(x), 5) for x in symmetry_fit['common_axis'])}"
    )
    print(
        "Geometric exact 30-degree initializer: "
        f"score={axis_locked_fit['score']:.5f}, angle30={axis_locked_fit['angle30_deg']:.2f}, "
        f"axis={tuple(round(float(x), 5) for x in axis_locked_fit['common_axis'])}, "
        f"max_residual={max(axis_locked_fit['residuals']):.4f}"
    )
    print(
        "Match-preserving 30-degree sequence: "
        f"accepted={match_axis_fit['accepted']}, side={match_axis_fit['side']}, "
        f"mean_score={match_axis_fit['mean_sequence_score']:+.5f}, "
        f"free_mean={match_axis_fit['mean_free_score']:+.5f}, "
        f"drop={match_axis_fit['score_drop']:+.5f}, "
        f"angle30={match_axis_fit['angle30_deg']:.2f}, "
        f"axis={tuple(round(float(x), 5) for x in match_axis_fit['common_axis'])}"
    )
    print(f"Selected common reference SEM point: {selection['selected_ref_xy']}, candidates={selection['candidate_count']}")
    for pair in pair_rows:
        print(
            f"align {pair.moving_angle:03d}->{pair.fixed_angle:03d}: "
            f"rot={pair.initial_rotation_deg:+.1f}, {pair.inliers}/{pair.matches} inliers, rmse={pair.residual_rmse:.2f}, {pair.method}"
        )
    for result in results:
        print(
            f"{result.spec.angle_deg:03d}: idx={result.selected_index}, IQ={result.selected_iq:.1f}, CI={result.selected_ci:.3f}, "
            f"sym={result.symmetry_name}, PC {tuple(round(x, 6) for x in result.pc_original)} -> "
            f"{tuple(round(x, 6) for x in result.pc_refined)}, score {result.original_score:+.4f}->{result.refined_score:+.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align twelve 30-degree Pt high-resolution EBSD SEM maps with adjacent LightGlue/SuperPoint, "
            "select the same physical point, and project its Kikuchi patterns plus PC anchors onto one master sphere."
        )
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", type=Path, default=DEFAULT_UP2_ROOT)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force LightGlue/SuperPoint to run on CPU.")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=4.0)
    parser.add_argument("--allow-cv2-fallback", action="store_true", default=True)
    parser.add_argument("--ci-min", type=float, default=0.30)
    parser.add_argument("--selection-stride", type=int, default=2)
    parser.add_argument("--selection-marker-half-size", type=float, default=18.0)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--pc-range", nargs=3, type=float, default=(0.01, 0.01, 0.02), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--mask-radius-fraction", type=float, default=0.40)
    parser.add_argument(
        "--orientation-mode",
        choices=("reference_variant", "auto_per_map"),
        default="reference_variant",
    )
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-steps", type=int, default=7)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--max-3d-points-per-pattern", type=int, default=140000)
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=10,
        help="Pixel stride used when optimizing the match-preserving physical 30-degree sequence.",
    )
    parser.add_argument("--sequence-maxiter", type=int, default=90)
    parser.add_argument(
        "--sequence-prior-weight",
        type=float,
        default=0.004,
        help="Small prior keeping the physical sequence near the H5+cubic match-preserving orientation candidates.",
    )
    parser.add_argument(
        "--sequence-max-score-drop",
        type=float,
        default=0.008,
        help="Reject the physical-axis solution if its mean match score drops by more than this relative to free H5+cubic placement.",
    )
    parser.add_argument(
        "--save-geometric-axis-locked",
        action="store_true",
        help="Also save the old geometric exact-30-degree projection used only as an initializer/diagnostic.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
