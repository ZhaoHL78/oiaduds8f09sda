from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from skimage.filters import sato
from skimage.morphology import remove_small_objects, skeletonize

from afm_sem_ebsd_standard_pipeline import (
    build_transforms,
    ebsd_ipf_z_rgb,
    fold_cubic,
    homography_xy,
    map_afm_to_sem_ebsd,
    read_afm_height,
    read_json,
    read_sem_ebsd,
    robust_rescale,
    sample_linear,
    scalar_dataset,
    write_json,
)


DEFAULT_CONFIG = Path("configs") / "afm_ebsd_boundary_refine_pt_highres60.json"
_CUBIC_SYMMETRY: np.ndarray | None = None


def ensure_dirs(base: Path) -> tuple[Path, Path, Path]:
    figures = base / "figures"
    data = base / "data"
    controls = base / "control_points"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    controls.mkdir(parents=True, exist_ok=True)
    return figures, data, controls


def save_gray(path: Path, image: np.ndarray, title: str, cmap: str = "gray", dpi: int = 220) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_rgb(path: Path, image: np.ndarray, title: str, dpi: int = 220) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_scalar(path: Path, values: np.ndarray, title: str, cmap: str, label: str, dpi: int = 220, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label=label, shrink=0.82)
    fig.savefig(path)
    plt.close(fig)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1).astype(bool)


def extract_afm_boundary(height_um: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray | float]:
    method = str(cfg.get("method", "gradient")).lower()
    if method == "scharr_valley":
        scharr_sigma = float(cfg.get("scharr_smooth_sigma_px", 2.0))
        scharr_height = ndimage.gaussian_filter(height_um.astype(np.float32), sigma=scharr_sigma, mode="nearest")
        gx = cv2.Scharr(scharr_height, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
        gy = cv2.Scharr(scharr_height, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
        slope = np.hypot(gx, gy).astype(np.float32)
        finite_slope = slope[np.isfinite(slope)]
        slope_thr = float(np.percentile(finite_slope, float(cfg.get("scharr_percentile", 94.5)))) if finite_slope.size else float("inf")
        high_tilt = slope >= slope_thr
        high_tilt = close_mask(high_tilt, int(cfg.get("scharr_close_px", 3)))
        high_tilt = remove_small_objects(high_tilt.astype(bool), min_size=max(80, int(cfg.get("remove_small_objects_px", 450)) // 3))
        boundary_band = dilate_mask(high_tilt, int(cfg.get("scharr_band_dilate_px", 9)))

        background_sigma = float(cfg.get("valley_background_sigma_px", 20.0))
        local_background = ndimage.gaussian_filter(scharr_height, sigma=background_sigma, mode="nearest")
        valley_residual = (scharr_height - local_background).astype(np.float32)
        min_size = int(cfg.get("valley_min_filter_px", 15))
        if min_size % 2 == 0:
            min_size += 1
        local_minimum = scharr_height <= (
            ndimage.minimum_filter(scharr_height, size=max(3, min_size), mode="nearest")
            + float(cfg.get("valley_min_tolerance_um", 0.018))
        )

        labels, nlabels = ndimage.label(boundary_band, structure=np.ones((3, 3), dtype=np.uint8))
        valley = np.zeros_like(boundary_band, dtype=bool)
        percentile = float(cfg.get("valley_percentile_within_band", 32.0))
        min_component = max(30, int(cfg.get("remove_small_objects_px", 450)) // 4)
        for label in range(1, nlabels + 1):
            comp = labels == label
            if int(comp.sum()) < min_component:
                continue
            values = valley_residual[comp]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            thr = float(np.percentile(values, percentile))
            comp_valley = comp & (valley_residual <= thr)
            # Enforce that the selected pixels are either local minima or
            # adjacent to a local minimum, so the skeleton follows the groove
            # floor instead of the high-tilt side wall.
            near_minimum = dilate_mask(local_minimum & comp, 2)
            comp_valley &= near_minimum
            if comp_valley.sum() < 8:
                comp_valley = comp & (valley_residual <= thr)
            valley |= comp_valley

        valley = close_mask(valley, int(cfg.get("valley_close_px", 2)))
        valley = remove_small_objects(valley.astype(bool), min_size=max(30, int(cfg.get("remove_small_objects_px", 450)) // 5))
        skel = skeletonize(valley) if bool(cfg.get("skeletonize", True)) else valley
        return {
            "smooth_height": scharr_height.astype(np.float32),
            "gradient": slope,
            "threshold": slope_thr,
            "mask": valley.astype(bool),
            "skeleton": skel.astype(bool),
        }

    smooth_sigma = float(cfg.get("smooth_sigma_px", 12.0))
    smooth = ndimage.gaussian_filter(height_um.astype(np.float32), sigma=smooth_sigma, mode="nearest")
    gx = ndimage.sobel(smooth, axis=1, mode="nearest")
    gy = ndimage.sobel(smooth, axis=0, mode="nearest")
    grad = np.hypot(gx, gy).astype(np.float32)
    finite = grad[np.isfinite(grad)]
    threshold = float(np.percentile(finite, float(cfg.get("gradient_percentile", 95.5)))) if finite.size else float("inf")
    mask = grad >= threshold
    mask = close_mask(mask, int(cfg.get("morph_close_px", 5)))
    mask = dilate_mask(mask, int(cfg.get("dilate_px", 2)))
    mask = remove_small_objects(mask.astype(bool), min_size=int(cfg.get("remove_small_objects_px", 450)))
    if method == "hybrid_sato":
        sato_sigma = float(cfg.get("sato_smooth_sigma_px", 5.0))
        sato_height = ndimage.gaussian_filter(height_um.astype(np.float32), sigma=sato_sigma, mode="nearest")
        sato_input = robust_rescale(sato_height, 1.0, 99.0)
        sigmas = [float(x) for x in cfg.get("sato_sigmas_px", [2, 4, 6, 8])]
        sato_response = sato(sato_input, sigmas=sigmas, black_ridges=True, mode="reflect").astype(np.float32)
        finite_sato = sato_response[np.isfinite(sato_response)]
        sato_thr = float(np.percentile(finite_sato, float(cfg.get("sato_percentile", 98.5)))) if finite_sato.size else float("inf")
        sato_mask = sato_response >= sato_thr
        sato_mask = close_mask(sato_mask, max(1, int(cfg.get("morph_close_px", 5)) // 2))
        sato_mask = remove_small_objects(sato_mask.astype(bool), min_size=max(50, int(cfg.get("remove_small_objects_px", 450)) // 2))
        # Use Sato for the centerline; retain only high-confidence gradient pixels
        # as support so isolated scan stripes are less likely to dominate.
        support = dilate_mask(mask, max(2, int(cfg.get("dilate_px", 2)) + 2))
        center_mask = sato_mask & support
        if center_mask.sum() < 100:
            center_mask = sato_mask
        skel = skeletonize(center_mask) if bool(cfg.get("skeletonize", True)) else center_mask
        mask = center_mask
        response = sato_response
        threshold_out = sato_thr
    else:
        skel = skeletonize(mask) if bool(cfg.get("skeletonize", True)) else mask
        response = grad
        threshold_out = threshold
    return {
        "smooth_height": smooth.astype(np.float32),
        "gradient": response,
        "threshold": threshold_out,
        "mask": mask.astype(bool),
        "skeleton": skel.astype(bool),
    }


def cubic_symmetry_matrices() -> np.ndarray:
    global _CUBIC_SYMMETRY
    if _CUBIC_SYMMETRY is not None:
        return _CUBIC_SYMMETRY
    mats: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        base[np.arange(3), perm] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = base * np.asarray(signs, dtype=np.float64)[None, :]
            if round(np.linalg.det(mat)) == 1:
                mats.append(mat)
    _CUBIC_SYMMETRY = np.stack(mats, axis=0)
    return _CUBIC_SYMMETRY


def orientation_misorientation_deg(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    # EDAX/OIM orientation matrices for cubic Pt can jump between symmetry
    # equivalent representatives. Use cubic disorientation to avoid marking
    # equivalent 120/180 degree changes as grain boundaries.
    rel = np.einsum("...ij,...kj->...ik", g1, g2)
    best = np.full(rel.shape[:-2], np.inf, dtype=np.float64)
    for sym in cubic_symmetry_matrices():
        candidate = np.einsum("...ij,jk->...ik", rel, sym)
        trace = candidate[..., 0, 0] + candidate[..., 1, 1] + candidate[..., 2, 2]
        cosang = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        best = np.minimum(best, np.degrees(np.arccos(cosang)))
    return best.astype(np.float32)


def extract_ebsd_boundary(ebsd: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    nrows, ncols = int(ebsd["nrows"]), int(ebsd["ncols"])
    orientations = ebsd["orientations"].reshape(nrows, ncols, 3, 3)
    phase = ebsd["phase"]
    valid = ebsd["valid"]
    method = str(cfg.get("method", "orientation")).lower()
    boundary = np.zeros((nrows, ncols), dtype=bool)

    if method in {"ipf", "ipf_color", "ipf-map", "ipf_map"}:
        ipf = ebsd_ipf_z_rgb(ebsd).astype(np.float32)
        smooth_sigma = float(cfg.get("ipf_smooth_sigma_px", 0.0))
        ipf_for_diff = ndimage.gaussian_filter(ipf, sigma=(smooth_sigma, smooth_sigma, 0.0), mode="nearest") if smooth_sigma > 0 else ipf

        right_valid = valid[:, :-1] & valid[:, 1:]
        right_phase = phase[:, :-1] != phase[:, 1:]
        right_delta = np.linalg.norm(ipf_for_diff[:, 1:] - ipf_for_diff[:, :-1], axis=2)
        down_valid = valid[:-1, :] & valid[1:, :]
        down_phase = phase[:-1, :] != phase[1:, :]
        down_delta = np.linalg.norm(ipf_for_diff[1:, :] - ipf_for_diff[:-1, :], axis=2)

        deltas = np.concatenate([right_delta[right_valid].reshape(-1), down_delta[down_valid].reshape(-1)])
        finite = deltas[np.isfinite(deltas)]
        percentile = float(cfg.get("ipf_color_percentile", 97.0))
        percentile_threshold = float(np.percentile(finite, percentile)) if finite.size else float("inf")
        threshold = max(float(cfg.get("ipf_min_color_delta", 0.08)), percentile_threshold)
        right_edge = right_valid & (right_phase | (right_delta >= threshold))
        down_edge = down_valid & (down_phase | (down_delta >= threshold))
        boundary[:, :-1] |= right_edge
        boundary[:, 1:] |= right_edge
        boundary[:-1, :] |= down_edge
        boundary[1:, :] |= down_edge
        response = np.zeros((nrows, ncols), dtype=np.float32)
        response[:, :-1] = np.maximum(response[:, :-1], right_delta)
        response[:, 1:] = np.maximum(response[:, 1:], right_delta)
        response[:-1, :] = np.maximum(response[:-1, :], down_delta)
        response[1:, :] = np.maximum(response[1:, :], down_delta)
        source = "IPF-Z color discontinuity generated from H5 orientations"
        extra = {
            "ipf_rgb": ipf,
            "response": response,
            "ipf_color_threshold": float(threshold),
            "ipf_color_percentile": percentile,
        }
    else:
        threshold = float(cfg.get("misorientation_threshold_deg", 5.0))
        right_valid = valid[:, :-1] & valid[:, 1:]
        right_phase = phase[:, :-1] != phase[:, 1:]
        right_ang = orientation_misorientation_deg(orientations[:, :-1], orientations[:, 1:])
        right_edge = right_valid & (right_phase | (right_ang >= threshold))
        boundary[:, :-1] |= right_edge
        boundary[:, 1:] |= right_edge

        down_valid = valid[:-1, :] & valid[1:, :]
        down_phase = phase[:-1, :] != phase[1:, :]
        down_ang = orientation_misorientation_deg(orientations[:-1, :], orientations[1:, :])
        down_edge = down_valid & (down_phase | (down_ang >= threshold))
        boundary[:-1, :] |= down_edge
        boundary[1:, :] |= down_edge
        response = np.zeros((nrows, ncols), dtype=np.float32)
        response[:, :-1] = np.maximum(response[:, :-1], right_ang)
        response[:, 1:] = np.maximum(response[:, 1:], right_ang)
        response[:-1, :] = np.maximum(response[:-1, :], down_ang)
        response[1:, :] = np.maximum(response[1:, :], down_ang)
        source = "phase discontinuity or cubic-symmetry orientation misorientation from H5 matrices"
        extra = {
            "response": response,
            "misorientation_threshold_deg": threshold,
        }

    boundary &= valid
    boundary = dilate_mask(boundary, int(cfg.get("dilate_px", 1)))
    min_size = int(cfg.get("remove_small_objects_px", 0))
    if min_size > 0:
        boundary = remove_small_objects(boundary.astype(bool), min_size=min_size)
    skel = skeletonize(boundary)
    result = {
        "mask": boundary.astype(bool),
        "skeleton": skel.astype(bool),
        "method": method,
        "source": source,
    }
    result.update(extra)
    return result


def connected_grain_map_from_boundary(ebsd: dict[str, Any], boundary: np.ndarray) -> np.ndarray:
    phase = ebsd["phase"]
    valid = ebsd["valid"] & (phase > 0)
    labels = np.zeros(valid.shape, dtype=np.int32)
    current = 1
    for phase_id in np.unique(phase[valid]):
        core = valid & (phase == phase_id) & (~boundary)
        n, lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
        sel = lab > 0
        labels[sel] = lab[sel] + current - 1
        current += max(n - 1, 0)
    labels[valid & (labels == 0)] = current
    return labels


def transform_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    flat = arr.reshape(-1, 2)
    homo = np.column_stack([flat, np.ones(flat.shape[0])])
    mapped = homo @ matrix.T
    mapped = mapped[:, :2] / (mapped[:, 2:3] + 1e-12)
    return mapped.reshape(arr.shape)


def centered_affine(params: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    tx, ty, theta_deg, log_sx, log_sy, shear = [float(x) for x in params[:6]]
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    sx, sy = math.exp(log_sx), math.exp(log_sy)
    linear = np.array([[c, -s], [s, c]], dtype=np.float64) @ np.array([[sx, shear], [0.0, sy]], dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    offset = np.array([tx, ty], dtype=np.float64) + center - linear @ center
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = linear
    out[:2, 2] = offset
    return out


def centered_homography(params: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    affine = centered_affine(params[:6], center_xy)
    p, q = float(params[6]), float(params[7])
    cx, cy = center_xy
    c = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    ci = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    proj = c @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [p, q, 1.0]], dtype=np.float64) @ ci
    return proj @ affine


def sample_image_at_xy(image: np.ndarray, xy: np.ndarray, fill: float) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x >= 0) & (x <= image.shape[1] - 1) & (y >= 0) & (y <= image.shape[0] - 1)
    vals = np.full(x.shape, fill, dtype=np.float32)
    if np.any(inside):
        vals[inside] = ndimage.map_coordinates(
            image.astype(np.float32),
            [y[inside], x[inside]],
            order=1,
            mode="nearest",
        )
    return vals


def sample_points(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    y, x = np.nonzero(mask)
    pts = np.column_stack([x, y]).astype(np.float64)
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]


def residuals_for_model(
    params: np.ndarray,
    model: str,
    t_coarse: np.ndarray,
    center_ebsd: tuple[float, float],
    afm_pts: np.ndarray,
    ebsd_pts: np.ndarray,
    afm_dt: np.ndarray,
    ebsd_dt: np.ndarray,
    ebsd_to_afm_px_scale: float,
) -> np.ndarray:
    residual = centered_affine(params, center_ebsd) if model == "affine" else centered_homography(params, center_ebsd)
    t_final = residual @ t_coarse
    t_inv = np.linalg.inv(t_final)
    afm_to_ebsd = transform_points(t_final, afm_pts)
    ebsd_to_afm = transform_points(t_inv, ebsd_pts)
    fill_afm = float(max(afm_dt.shape) * 0.25)
    fill_ebsd = float(max(ebsd_dt.shape) * 0.25)
    d1 = sample_image_at_xy(ebsd_dt, afm_to_ebsd, fill_ebsd) * ebsd_to_afm_px_scale
    d2 = sample_image_at_xy(afm_dt, ebsd_to_afm, fill_afm)
    return np.concatenate([d1, d2]).astype(np.float64)


def optimize_residual(
    model: str,
    t_coarse: np.ndarray,
    center_ebsd: tuple[float, float],
    afm_pts: np.ndarray,
    ebsd_pts: np.ndarray,
    afm_dt: np.ndarray,
    ebsd_dt: np.ndarray,
    cfg: dict[str, Any],
    ebsd_to_afm_px_scale: float,
) -> dict[str, Any]:
    opt = cfg["optimization"]
    tx = float(opt.get("residual_translation_bound_ebsd_px", 24.0))
    rot = float(opt.get("residual_rotation_bound_deg", 3.0))
    log_s = float(opt.get("residual_log_scale_bound", 0.035))
    shear = float(opt.get("residual_shear_bound", 0.04))
    if model == "affine":
        x0 = np.zeros(6, dtype=np.float64)
        lower = np.array([-tx, -tx, -rot, -log_s, -log_s, -shear], dtype=np.float64)
        upper = np.array([tx, tx, rot, log_s, log_s, shear], dtype=np.float64)
        max_nfev = int(opt.get("max_nfev_affine", 180))
    else:
        proj = float(opt.get("residual_projective_bound", 0.00008))
        x0 = np.zeros(8, dtype=np.float64)
        lower = np.array([-tx, -tx, -rot, -log_s, -log_s, -shear, -proj, -proj], dtype=np.float64)
        upper = np.array([tx, tx, rot, log_s, log_s, shear, proj, proj], dtype=np.float64)
        max_nfev = int(opt.get("max_nfev_homography", 240))
    result = least_squares(
        residuals_for_model,
        x0,
        bounds=(lower, upper),
        args=(model, t_coarse, center_ebsd, afm_pts, ebsd_pts, afm_dt, ebsd_dt, ebsd_to_afm_px_scale),
        loss="huber",
        f_scale=float(opt.get("huber_f_scale_px", 4.0)),
        max_nfev=max_nfev,
        verbose=0,
    )
    residual = centered_affine(result.x, center_ebsd) if model == "affine" else centered_homography(result.x, center_ebsd)
    return {
        "model": model,
        "params": result.x.astype(float).tolist(),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "T_residual": residual,
        "T_final": residual @ t_coarse,
    }


def warp_ebsd_boundary_to_afm(ebsd_boundary: np.ndarray, afm_shape: tuple[int, int], t_afm_to_ebsd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ah, aw = afm_shape
    row, col = np.indices((ah, aw), dtype=np.float64)
    xy = np.dstack([col, row])
    ebsd_xy = homography_xy(t_afm_to_ebsd, xy)
    vals = ndimage.map_coordinates(
        ebsd_boundary.astype(np.float32),
        [ebsd_xy[..., 1].reshape(-1), ebsd_xy[..., 0].reshape(-1)],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(afm_shape)
    inside = (
        (ebsd_xy[..., 0] >= 0)
        & (ebsd_xy[..., 0] <= ebsd_boundary.shape[1] - 1)
        & (ebsd_xy[..., 1] >= 0)
        & (ebsd_xy[..., 1] <= ebsd_boundary.shape[0] - 1)
    )
    mask = (vals > 0.08) & inside
    return mask.astype(bool), inside.astype(bool)


def rasterize_afm_boundary_to_ebsd(
    afm_boundary: np.ndarray,
    ebsd_shape: tuple[int, int],
    t_afm_to_ebsd: np.ndarray,
    dilate_px: int = 1,
) -> np.ndarray:
    """Forward-rasterize AFM boundary pixels into the EBSD row/column grid."""
    ys, xs = np.nonzero(afm_boundary.astype(bool))
    out = np.zeros(ebsd_shape, dtype=bool)
    if xs.size == 0:
        return out
    xy = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mapped = homography_xy(t_afm_to_ebsd, xy)
    cols = np.rint(mapped[:, 0]).astype(np.int32)
    rows = np.rint(mapped[:, 1]).astype(np.int32)
    inside = (cols >= 0) & (cols < ebsd_shape[1]) & (rows >= 0) & (rows < ebsd_shape[0])
    out[rows[inside], cols[inside]] = True
    out = dilate_mask(out, max(0, int(dilate_px)))
    return skeletonize(out).astype(bool)


def boundary_metrics(
    afm_skeleton: np.ndarray,
    ebsd_on_afm_mask: np.ndarray,
    common_mask: np.ndarray,
    tolerances: list[int],
    pixel_size_um: float,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    afm = afm_skeleton.astype(bool) & common_mask
    ebsd = skeletonize(ebsd_on_afm_mask.astype(bool)) & common_mask
    if roi is not None:
        x0, y0, x1, y1 = roi
        roi_mask = np.zeros_like(common_mask, dtype=bool)
        roi_mask[y0:y1, x0:x1] = True
        afm &= roi_mask
        ebsd &= roi_mask
    dt_afm = ndimage.distance_transform_edt(~afm)
    dt_ebsd = ndimage.distance_transform_edt(~ebsd)
    d_ebsd_to_afm = dt_afm[ebsd]
    d_afm_to_ebsd = dt_ebsd[afm]
    both = np.concatenate([d_ebsd_to_afm, d_afm_to_ebsd]) if d_ebsd_to_afm.size and d_afm_to_ebsd.size else np.array([], dtype=float)
    out: dict[str, Any] = {
        "afm_boundary_pixels": int(afm.sum()),
        "ebsd_boundary_pixels": int(ebsd.sum()),
        "symmetric_mean_px": float(np.mean(both)) if both.size else None,
        "symmetric_median_px": float(np.median(both)) if both.size else None,
        "symmetric_p95_px": float(np.percentile(both, 95)) if both.size else None,
        "hausdorff95_px": float(np.percentile(both, 95)) if both.size else None,
        "hausdorff_max_px": float(np.max(both)) if both.size else None,
    }
    for key in list(out):
        if key.endswith("_px") and out[key] is not None:
            out[key.replace("_px", "_um")] = float(out[key]) * pixel_size_um
    for tol in tolerances:
        precision = float(np.mean(d_ebsd_to_afm <= tol)) if d_ebsd_to_afm.size else None
        recall = float(np.mean(d_afm_to_ebsd <= tol)) if d_afm_to_ebsd.size else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else None
        out[f"precision_at_{tol}px"] = precision
        out[f"recall_at_{tol}px"] = recall
        out[f"f1_at_{tol}px"] = f1
    return out


def overlay_boundaries(base: np.ndarray, afm_boundary: np.ndarray, ebsd_boundary: np.ndarray, radius: int) -> np.ndarray:
    if base.ndim == 3:
        rgb = np.clip(base.astype(np.float32), 0.0, 1.0).copy()
    else:
        gray = robust_rescale(base)
        rgb = np.dstack([gray, gray, gray])
    afm = dilate_mask(afm_boundary, radius)
    ebsd = dilate_mask(ebsd_boundary, radius)
    overlap = afm & ebsd
    rgb[afm] = np.array([0.0, 0.95, 1.0])
    rgb[ebsd] = np.array([1.0, 0.0, 0.85])
    rgb[overlap] = np.array([1.0, 0.95, 0.0])
    return rgb.astype(np.float32)


def overlay_one_boundary(base: np.ndarray, boundary: np.ndarray, color: tuple[float, float, float], radius: int) -> np.ndarray:
    if base.ndim == 3:
        rgb = np.clip(base.astype(np.float32), 0.0, 1.0).copy()
    else:
        gray = robust_rescale(base)
        rgb = np.dstack([gray, gray, gray])
    mask = dilate_mask(boundary.astype(bool), radius)
    rgb[mask] = np.array(color, dtype=np.float32)
    return rgb.astype(np.float32)


def save_zoom_overlays(fig_dir: Path, base: np.ndarray, afm_skel: np.ndarray, masks: dict[str, np.ndarray], rois: dict[str, list[float]], radius: int, dpi: int) -> None:
    h, w = base.shape
    for name, frac in rois.items():
        x0 = int(round(frac[0] * w))
        y0 = int(round(frac[1] * h))
        x1 = int(round(frac[2] * w))
        y1 = int(round(frac[3] * h))
        cols = len(masks)
        fig, axes = plt.subplots(1, cols, figsize=(5.2 * cols, 4.6), dpi=dpi, constrained_layout=True)
        if cols == 1:
            axes = [axes]
        for ax, (label, mask) in zip(axes, masks.items()):
            rgb = overlay_boundaries(base, afm_skel, mask, radius)
            ax.imshow(rgb[y0:y1, x0:x1])
            ax.set_title(f"{name}: {label}")
            ax.axis("off")
        fig.savefig(fig_dir / f"zoom_{name}_boundary_overlay.png")
        plt.close(fig)


def save_model_comparison(path: Path, base: np.ndarray, afm_skel: np.ndarray, masks: dict[str, np.ndarray], radius: int, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(masks), figsize=(6.0 * len(masks), 5.6), dpi=dpi, constrained_layout=True)
    if len(masks) == 1:
        axes = [axes]
    for ax, (label, mask) in zip(axes, masks.items()):
        ax.imshow(overlay_boundaries(base, afm_skel, mask, radius))
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_ebsd_frame_model_comparison(
    path: Path,
    base: np.ndarray,
    afm_on_ebsd_masks: dict[str, np.ndarray],
    ebsd_skel: np.ndarray,
    radius: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(afm_on_ebsd_masks), figsize=(6.0 * len(afm_on_ebsd_masks), 5.6), dpi=dpi, constrained_layout=True)
    if len(afm_on_ebsd_masks) == 1:
        axes = [axes]
    for ax, (label, afm_mask) in zip(axes, afm_on_ebsd_masks.items()):
        ax.imshow(overlay_boundaries(base, afm_mask, ebsd_skel, radius))
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def write_metrics_summary(path: Path, metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model, scopes in metrics.items():
        for scope, values in scopes.items():
            row = {
                "model": model,
                "scope": scope,
                "mean_px": values.get("symmetric_mean_px"),
                "median_px": values.get("symmetric_median_px"),
                "p95_px": values.get("symmetric_p95_px"),
                "hausdorff95_px": values.get("hausdorff95_px"),
                "mean_um": values.get("symmetric_mean_um"),
                "median_um": values.get("symmetric_median_um"),
                "p95_um": values.get("symmetric_p95_um"),
                "hausdorff95_um": values.get("hausdorff95_um"),
            }
            for key, value in values.items():
                if key.startswith(("precision_at_", "recall_at_", "f1_at_")):
                    row[key] = value
            rows.append(row)
    fieldnames = ["model", "scope", "mean_px", "median_px", "p95_px", "hausdorff95_px", "mean_um", "median_um", "p95_um", "hausdorff95_um"]
    fieldnames += sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_ipf_on_afm(ebsd: dict[str, Any], t_afm_to_ebsd: np.ndarray, afm_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    row, col = np.indices(afm_shape, dtype=np.float64)
    xy = np.dstack([col, row])
    ebsd_xy = homography_xy(t_afm_to_ebsd, xy)
    inside = (
        (ebsd_xy[..., 0] >= 0)
        & (ebsd_xy[..., 0] <= int(ebsd["ncols"]) - 1)
        & (ebsd_xy[..., 1] >= 0)
        & (ebsd_xy[..., 1] <= int(ebsd["nrows"]) - 1)
    )
    ipf_grid = ebsd_ipf_z_rgb(ebsd)
    ipf = sample_linear(ipf_grid, ebsd_xy[..., 1].astype(np.float32), ebsd_xy[..., 0].astype(np.float32), inside)
    return ipf, inside


def write_major_boundary_errors(path: Path, afm_skeleton: np.ndarray, ebsd_on_afm: np.ndarray, common: np.ndarray, pixel_size_um: float) -> None:
    labels, n = ndimage.label(afm_skeleton & common, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:12]
    dt_ebsd = ndimage.distance_transform_edt(~(skeletonize(ebsd_on_afm) & common))
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        lab = int(idx + 1)
        comp = labels == lab
        d = dt_ebsd[comp]
        if d.size == 0:
            continue
        ys, xs = np.nonzero(comp)
        rows.append({
            "rank": rank,
            "component_label": lab,
            "pixels": int(comp.sum()),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "mean_error_px": float(np.mean(d)),
            "median_error_px": float(np.median(d)),
            "p95_error_px": float(np.percentile(d, 95)),
            "mean_error_um": float(np.mean(d) * pixel_size_um),
            "median_error_um": float(np.median(d) * pixel_size_um),
            "p95_error_um": float(np.percentile(d, 95) * pixel_size_um),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)


def create_control_template(path: Path) -> None:
    if path.exists():
        return
    template = {
        "coordinate_frame": "AFM points are original AFM reference pixels. EBSD points are EBSD grid pixels as x=column, y=row. Residual transform is fitted in EBSD grid and composed as T_final = T_residual @ T_coarse_AFM_to_EBSD.",
        "pairs": [],
        "instructions": [
            "Add 8-15 AFM/EBSD boundary correspondences, especially triple junctions, boundary bends, image-edge intersections, and several points on the left-bottom boundary.",
            "Use figures/01_afm_boundary_mask_skeleton.png and figures/03_coarse_boundary_overlay.png for picking.",
            "After editing this JSON, rerun refine_afm_ebsd_boundaries.py with the same config. Manual point fitting is intentionally not used until points are present."
        ],
    }
    write_json(path, template)


def run(config_path: Path) -> None:
    cfg = read_json(config_path)
    source_cfg = read_json(Path(cfg["source_config"]))
    output_dir = Path(cfg["output_dir"])
    fig_dir, data_dir, controls_dir = ensure_dirs(output_dir)
    dpi = int(cfg.get("visualization", {}).get("dpi", 220))
    pixel_size_um = float(cfg.get("metrics", {}).get("pixel_size_um", 1.0))

    afm_height, afm_meta = read_afm_height(source_cfg)
    ebsd = read_sem_ebsd(source_cfg)
    sem_display = ebsd["sem_display"]
    registration = read_json(Path(source_cfg["afm_sem_alignment"]["registration_report_path"]))
    h_afm_resized_to_sem = np.asarray(registration["matrix_afm_to_sem"], dtype=np.float64)
    transforms = build_transforms(
        afm_height.shape,
        sem_display.shape,
        (int(ebsd["nrows"]), int(ebsd["ncols"])),
        h_afm_resized_to_sem,
        ebsd["sem_display_to_ebsd_grid"],
    )
    t_coarse = np.asarray(transforms["T_afm_reference_to_ebsd_grid"], dtype=np.float64)

    afm_boundary = extract_afm_boundary(afm_height, cfg["afm_boundary"])
    ebsd_boundary = extract_ebsd_boundary(ebsd, cfg["ebsd_boundary"])
    ebsd_ipf = np.asarray(ebsd_boundary.get("ipf_rgb", ebsd_ipf_z_rgb(ebsd)), dtype=np.float32)
    ebsd_grain_labels = connected_grain_map_from_boundary(ebsd, ebsd_boundary["mask"])

    save_gray(fig_dir / "00_afm_height_reference.png", afm_height, "AFM height reference", "copper", dpi)
    save_scalar(fig_dir / "01_afm_boundary_response.png", afm_boundary["gradient"], "AFM coarse height-gradient response for major boundaries", "magma", "gradient", dpi)
    save_rgb(fig_dir / "01_afm_boundary_mask_skeleton.png", overlay_boundaries(afm_height, afm_boundary["skeleton"], afm_boundary["mask"], 1), "AFM major boundary mask and skeleton", dpi)
    save_rgb(fig_dir / "01a_afm_boundary_on_afm_height.png", overlay_one_boundary(afm_height, afm_boundary["skeleton"], (0.0, 0.95, 1.0), overlay_radius := int(cfg.get("visualization", {}).get("boundary_overlay_dilate_px", 2))), "AFM Scharr-valley boundary on AFM height", dpi)
    save_gray(fig_dir / "02_ebsd_grain_components.png", ebsd_grain_labels, "EBSD grain components separated by IPF boundary", "nipy_spectral", dpi)
    save_rgb(fig_dir / "02_ebsd_ipf_z_map_for_boundary.png", ebsd_ipf, "EBSD IPF-Z map used for boundary extraction", dpi)
    save_scalar(fig_dir / "02a_ebsd_ipf_boundary_response.png", ebsd_boundary["response"], "EBSD IPF color-discontinuity response", "magma", "IPF color delta", dpi)
    save_rgb(fig_dir / "02b_ebsd_boundary_on_ebsd_ipf.png", overlay_one_boundary(ebsd_ipf, ebsd_boundary["skeleton"], (1.0, 0.0, 0.85), overlay_radius), "EBSD boundary extracted from IPF-Z map", dpi)

    afm_skel_pts = sample_points(afm_boundary["skeleton"], int(cfg["optimization"].get("max_points_per_direction", 4500)), int(cfg["optimization"].get("random_seed", 0)))
    ebsd_skel_pts = sample_points(ebsd_boundary["skeleton"], int(cfg["optimization"].get("max_points_per_direction", 4500)), int(cfg["optimization"].get("random_seed", 0)) + 17)
    afm_dt = ndimage.distance_transform_edt(~afm_boundary["skeleton"])
    ebsd_dt = ndimage.distance_transform_edt(~ebsd_boundary["skeleton"])
    # Local scale converts EBSD-pixel residuals into AFM-pixel-like weights.
    jac = t_coarse[:2, :2]
    sv = np.linalg.svd(jac, compute_uv=False)
    ebsd_to_afm_px_scale = float(1.0 / max(np.mean(sv), 1e-6))
    center_ebsd = ((int(ebsd["ncols"]) - 1) / 2.0, (int(ebsd["nrows"]) - 1) / 2.0)

    models: dict[str, dict[str, Any]] = {
        "coarse": {
            "model": "coarse",
            "params": [],
            "cost": None,
            "success": True,
            "message": "Existing AFM->SEM->EBSD transform without AFM-EBSD residual refinement.",
            "nfev": 0,
            "T_residual": np.eye(3, dtype=np.float64),
            "T_final": t_coarse,
        }
    }
    for model_name in ("affine", "homography"):
        models[model_name] = optimize_residual(
            model_name,
            t_coarse,
            center_ebsd,
            afm_skel_pts,
            ebsd_skel_pts,
            afm_dt,
            ebsd_dt,
            cfg,
            ebsd_to_afm_px_scale,
        )

    tolerances = [int(x) for x in cfg["metrics"].get("tolerance_px", [3, 5, 8])]
    roi_frac = cfg["metrics"].get("left_bottom_roi_xyxy_fraction", [0.0, 0.55, 0.38, 1.0])
    ah, aw = afm_height.shape
    left_bottom_roi = (
        int(round(roi_frac[0] * aw)),
        int(round(roi_frac[1] * ah)),
        int(round(roi_frac[2] * aw)),
        int(round(roi_frac[3] * ah)),
    )
    metrics: dict[str, Any] = {}
    warped_masks: dict[str, np.ndarray] = {}
    afm_on_ebsd_masks: dict[str, np.ndarray] = {}
    common_masks: dict[str, np.ndarray] = {}
    for name, model in models.items():
        warped, inside = warp_ebsd_boundary_to_afm(ebsd_boundary["mask"], afm_height.shape, model["T_final"])
        warped = skeletonize(warped)
        warped_masks[name] = warped
        afm_on_ebsd = rasterize_afm_boundary_to_ebsd(
            afm_boundary["skeleton"],
            (int(ebsd["nrows"]), int(ebsd["ncols"])),
            model["T_final"],
            max(1, overlay_radius),
        )
        afm_on_ebsd_masks[name] = afm_on_ebsd
        common_masks[name] = inside
        metrics[name] = {
            "global": boundary_metrics(afm_boundary["skeleton"], warped, inside, tolerances, pixel_size_um),
            "left_bottom_roi": boundary_metrics(afm_boundary["skeleton"], warped, inside, tolerances, pixel_size_um, left_bottom_roi),
        }
        save_rgb(fig_dir / f"03_{name}_boundary_overlay.png", overlay_boundaries(afm_height, afm_boundary["skeleton"], warped, overlay_radius), f"{name}: AFM boundary cyan, EBSD boundary magenta, overlap yellow", dpi)
        save_scalar(fig_dir / f"04_{name}_ebsd_to_afm_boundary_distance.png", ndimage.distance_transform_edt(~afm_boundary["skeleton"]) * warped, f"{name}: EBSD boundary distance to AFM boundary", "magma", "px", dpi, vmax=30)
        ipf_on_afm, inside_ipf = sample_ipf_on_afm(ebsd, model["T_final"], afm_height.shape)
        save_rgb(fig_dir / f"05_{name}_ipf_z_on_afm.png", ipf_on_afm, f"{name}: EBSD IPF-Z mapped to AFM reference", dpi)
        save_rgb(fig_dir / f"06_{name}_boundary_overlay_on_ebsd_ipf.png", overlay_boundaries(ebsd_ipf, afm_on_ebsd, ebsd_boundary["skeleton"], overlay_radius), f"{name}: AFM boundary mapped to EBSD IPF grid; EBSD IPF boundary fixed", dpi)
        write_major_boundary_errors(data_dir / f"major_boundary_errors_{name}.csv", afm_boundary["skeleton"], warped, inside, pixel_size_um)

    affine_mean = metrics["affine"]["global"]["symmetric_mean_px"]
    hom_mean = metrics["homography"]["global"]["symmetric_mean_px"]
    improvement = 0.0 if affine_mean in (None, 0) or hom_mean is None else (float(affine_mean) - float(hom_mean)) / max(float(affine_mean), 1e-12)
    prefer_cutoff = float(cfg["optimization"].get("prefer_affine_if_homography_improvement_lt_fraction", 0.10))
    selected = "affine" if improvement < prefer_cutoff else "homography"

    save_zoom_overlays(
        fig_dir,
        afm_height,
        afm_boundary["skeleton"],
        {"coarse": warped_masks["coarse"], "affine": warped_masks["affine"], "homography": warped_masks["homography"]},
        cfg.get("visualization", {}).get("zoom_rois", {}),
        overlay_radius,
        dpi,
    )
    save_model_comparison(
        fig_dir / "03_boundary_overlay_model_comparison.png",
        afm_height,
        afm_boundary["skeleton"],
        {"coarse": warped_masks["coarse"], "residual affine": warped_masks["affine"], "residual homography": warped_masks["homography"]},
        overlay_radius,
        dpi,
    )
    save_ebsd_frame_model_comparison(
        fig_dir / "06_boundary_overlay_on_ebsd_model_comparison.png",
        ebsd_ipf,
        {"coarse": afm_on_ebsd_masks["coarse"], "residual affine": afm_on_ebsd_masks["affine"], "residual homography": afm_on_ebsd_masks["homography"]},
        ebsd_boundary["skeleton"],
        overlay_radius,
        dpi,
    )
    write_metrics_summary(data_dir / "boundary_metrics_summary.csv", metrics)

    control_path = Path(cfg["control_points_path"])
    create_control_template(control_path)

    serializable_models = {}
    for name, model in models.items():
        serializable_models[name] = {
            "model": model["model"],
            "params": model["params"],
            "cost": model["cost"],
            "success": model["success"],
            "message": model["message"],
            "nfev": model["nfev"],
            "T_residual_ebsd_grid": np.asarray(model["T_residual"]).tolist(),
            "T_final_afm_to_ebsd": np.asarray(model["T_final"]).tolist(),
        }
    report = {
        "status": "completed",
        "config": str(config_path),
        "source_config": cfg["source_config"],
        "matrix_convention": {
            "T_coarse_direction": "AFM reference pixels -> EBSD grid coordinates (x=column, y=row)",
            "residual_direction": "EBSD grid -> corrected EBSD grid",
            "composition": "T_final_afm_to_ebsd = T_residual_ebsd_grid @ T_coarse_afm_to_ebsd",
        },
        "afm": afm_meta,
        "ebsd": {
            "h5_path": ebsd["h5_path"],
            "h5_group": ebsd["h5_group"],
            "shape_rows_cols": [int(ebsd["nrows"]), int(ebsd["ncols"])],
            "has_raw_grain_id": False,
            "boundary_source": ebsd_boundary["source"],
            "boundary_method": ebsd_boundary["method"],
            "misorientation_threshold_deg": (
                float(ebsd_boundary["misorientation_threshold_deg"]) if "misorientation_threshold_deg" in ebsd_boundary else None
            ),
            "ipf_color_threshold": float(ebsd_boundary["ipf_color_threshold"]) if "ipf_color_threshold" in ebsd_boundary else None,
            "ipf_color_percentile": float(ebsd_boundary["ipf_color_percentile"]) if "ipf_color_percentile" in ebsd_boundary else None,
        },
        "coarse_registration_report": source_cfg["afm_sem_alignment"]["registration_report_path"],
        "boundary_extraction": {
            "afm_gradient_threshold": float(afm_boundary["threshold"]),
            "afm_boundary_pixels": int(np.sum(afm_boundary["skeleton"])),
            "ebsd_boundary_pixels": int(np.sum(ebsd_boundary["skeleton"])),
            "ebsd_to_afm_px_scale_used_in_loss": ebsd_to_afm_px_scale,
            "ebsd_boundary_source": ebsd_boundary["source"],
        },
        "metrics": metrics,
        "model_selection": {
            "homography_improvement_over_affine_fraction": improvement,
            "prefer_affine_if_homography_improvement_lt_fraction": prefer_cutoff,
            "selected_for_review": selected,
            "note": "The selected model is only proposed for review and is not written back to surface-index outputs.",
        },
        "models": serializable_models,
        "outputs": {
            "figures": str(fig_dir.resolve()),
            "data": str(data_dir.resolve()),
            "control_points_template": str(control_path.resolve()),
            "report": str((data_dir / "afm_ebsd_boundary_refinement_report.json").resolve()),
            "metrics_summary": str((data_dir / "boundary_metrics_summary.csv").resolve()),
        },
    }
    write_json(data_dir / "afm_ebsd_boundary_refinement_report.json", report)
    np.savez_compressed(
        data_dir / "afm_ebsd_boundary_refinement_arrays.npz",
        afm_height_um=afm_height.astype(np.float32),
        afm_boundary_mask=afm_boundary["mask"].astype(bool),
        afm_boundary_skeleton=afm_boundary["skeleton"].astype(bool),
        ebsd_boundary_mask=ebsd_boundary["mask"].astype(bool),
        ebsd_boundary_skeleton=ebsd_boundary["skeleton"].astype(bool),
        ebsd_ipf_z_rgb=ebsd_ipf.astype(np.float32),
        ebsd_ipf_boundary_response=ebsd_boundary["response"].astype(np.float32),
        ebsd_grain_components=ebsd_grain_labels.astype(np.int32),
        coarse_ebsd_boundary_on_afm=warped_masks["coarse"].astype(bool),
        affine_ebsd_boundary_on_afm=warped_masks["affine"].astype(bool),
        homography_ebsd_boundary_on_afm=warped_masks["homography"].astype(bool),
        coarse_afm_boundary_on_ebsd=afm_on_ebsd_masks["coarse"].astype(bool),
        affine_afm_boundary_on_ebsd=afm_on_ebsd_masks["affine"].astype(bool),
        homography_afm_boundary_on_ebsd=afm_on_ebsd_masks["homography"].astype(bool),
        T_coarse_afm_to_ebsd=t_coarse,
        T_affine_final_afm_to_ebsd=models["affine"]["T_final"],
        T_homography_final_afm_to_ebsd=models["homography"]["T_final"],
    )
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "selected_for_review": selected,
        "coarse_mean_px": metrics["coarse"]["global"]["symmetric_mean_px"],
        "affine_mean_px": metrics["affine"]["global"]["symmetric_mean_px"],
        "homography_mean_px": metrics["homography"]["global"]["symmetric_mean_px"],
        "left_bottom_affine_mean_px": metrics["affine"]["left_bottom_roi"]["symmetric_mean_px"],
    }, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine coarse AFM->EBSD alignment using AFM and EBSD grain-boundary geometry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
