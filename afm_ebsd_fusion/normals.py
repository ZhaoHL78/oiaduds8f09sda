from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def plane_level(height_um: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    y, x = np.indices(height_um.shape, dtype=np.float64)
    values = np.asarray(height_um, dtype=np.float64)
    finite = np.isfinite(values)
    design = np.column_stack([x[finite], y[finite], np.ones(np.count_nonzero(finite))])
    coeff, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    leveled = values - plane
    return leveled.astype(np.float32), {
        "plane_coeff_um_per_px_x": float(coeff[0]),
        "plane_coeff_um_per_px_y": float(coeff[1]),
        "plane_intercept_um": float(coeff[2]),
    }


def local_plane_slopes(height_um: np.ndarray, pitch_x_um: float, pitch_y_um: float, window_px: int) -> tuple[np.ndarray, np.ndarray]:
    if window_px % 2 == 0:
        window_px += 1
    radius = window_px // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kx = np.tile(offsets, (window_px, 1))
    ky = np.tile(offsets[:, None], (1, window_px))
    denom_x = float(np.sum(kx * kx))
    denom_y = float(np.sum(ky * ky))
    z = np.asarray(height_um, dtype=np.float32)
    dz_dcol = ndimage.convolve(z, kx / max(denom_x, 1e-12), mode="nearest")
    dz_drow = ndimage.convolve(z, ky / max(denom_y, 1e-12), mode="nearest")
    return dz_dcol / max(pitch_x_um, 1e-12), dz_drow / max(pitch_y_um, 1e-12)


def scharr_slopes(height_um: np.ndarray, pitch_x_um: float, pitch_y_um: float) -> tuple[np.ndarray, np.ndarray]:
    dz_dcol = cv2.Scharr(height_um.astype(np.float32), cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    dz_drow = cv2.Scharr(height_um.astype(np.float32), cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    return dz_dcol / max(pitch_x_um, 1e-12), dz_drow / max(pitch_y_um, 1e-12)


def gaussian_derivative_slopes(
    height_um: np.ndarray,
    pitch_x_um: float,
    pitch_y_um: float,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(float(sigma_px), 0.1)
    dz_drow = ndimage.gaussian_filter(height_um.astype(np.float32), sigma=sigma, order=(1, 0), mode="nearest")
    dz_dcol = ndimage.gaussian_filter(height_um.astype(np.float32), sigma=sigma, order=(0, 1), mode="nearest")
    return dz_dcol / max(pitch_x_um, 1e-12), dz_drow / max(pitch_y_um, 1e-12)


def slopes_to_normals(
    dz_dcol: np.ndarray,
    dz_drow: np.ndarray,
    *,
    image_y_to_sample_y: int,
    afm_to_sample_rotation_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dz_dx = dz_dcol.astype(np.float32)
    dz_dy_sample = dz_drow.astype(np.float32) * float(image_y_to_sample_y)
    normals_afm_sample_axes = np.dstack([-dz_dx, -dz_dy_sample, np.ones_like(dz_dx, dtype=np.float32)])
    normals_afm_sample_axes /= np.linalg.norm(normals_afm_sample_axes, axis=2, keepdims=True) + 1e-12

    rotation = np.asarray(afm_to_sample_rotation_2d, dtype=np.float64)
    xy = normals_afm_sample_axes[..., :2].reshape(-1, 2) @ rotation.T
    normals_sample = np.column_stack([xy, normals_afm_sample_axes[..., 2].reshape(-1)]).reshape(normals_afm_sample_axes.shape)
    normals_sample /= np.linalg.norm(normals_sample, axis=2, keepdims=True) + 1e-12
    normals_sample[normals_sample[..., 2] < 0] *= -1.0
    slope_deg = np.degrees(np.arccos(np.clip(normals_sample[..., 2], -1.0, 1.0))).astype(np.float32)
    aspect_deg = np.degrees(np.arctan2(normals_sample[..., 1], normals_sample[..., 0])).astype(np.float32)
    return normals_sample.astype(np.float32), slope_deg, aspect_deg


def compute_afm_normals(
    height_display_um: np.ndarray,
    scan_size_x_um: float,
    scan_size_y_um: float,
    *,
    plane_level_enabled: bool,
    smooth_sigma_px: float,
    local_plane_window_px: int,
    image_y_to_sample_y: int,
    afm_to_sample_rotation_2d: np.ndarray,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    raw = np.asarray(height_display_um, dtype=np.float32)
    if plane_level_enabled:
        leveled, plane_meta = plane_level(raw)
    else:
        leveled = raw.copy()
        plane_meta = {"plane_level": 0.0}
    smooth = ndimage.gaussian_filter(leveled, sigma=float(smooth_sigma_px), mode="nearest") if smooth_sigma_px > 0 else leveled
    pitch_x_um = float(scan_size_x_um) / max(raw.shape[1] - 1, 1)
    pitch_y_um = float(scan_size_y_um) / max(raw.shape[0] - 1, 1)
    lp_col, lp_row = local_plane_slopes(smooth, pitch_x_um, pitch_y_um, local_plane_window_px)
    sch_col, sch_row = scharr_slopes(smooth, pitch_x_um, pitch_y_um)
    gau_col, gau_row = gaussian_derivative_slopes(smooth, pitch_x_um, pitch_y_um, max(smooth_sigma_px, 1.0))
    normals_sample, slope_deg, aspect_deg = slopes_to_normals(
        lp_col,
        lp_row,
        image_y_to_sample_y=image_y_to_sample_y,
        afm_to_sample_rotation_2d=afm_to_sample_rotation_2d,
    )
    normals_scharr, slope_scharr_deg, _aspect_scharr = slopes_to_normals(
        sch_col,
        sch_row,
        image_y_to_sample_y=image_y_to_sample_y,
        afm_to_sample_rotation_2d=afm_to_sample_rotation_2d,
    )
    normals_gauss, slope_gauss_deg, _aspect_gauss = slopes_to_normals(
        gau_col,
        gau_row,
        image_y_to_sample_y=image_y_to_sample_y,
        afm_to_sample_rotation_2d=afm_to_sample_rotation_2d,
    )
    angular_uncertainty = np.degrees(
        np.arccos(np.clip(np.sum(normals_sample * normals_scharr, axis=2), -1.0, 1.0))
    ).astype(np.float32)
    return {
        "height_raw_um": raw,
        "height_leveled_um": leveled,
        "height_smoothed_um": smooth.astype(np.float32),
        "dz_dx": lp_col.astype(np.float32),
        "dz_drow": lp_row.astype(np.float32),
        "normals_sample": normals_sample,
        "normals_scharr_sample": normals_scharr,
        "normals_gaussian_sample": normals_gauss,
        "slope_deg": slope_deg,
        "slope_scharr_deg": slope_scharr_deg,
        "slope_gaussian_deg": slope_gauss_deg,
        "aspect_deg": aspect_deg,
        "normal_method_disagreement_deg": angular_uncertainty,
        "pitch_x_um": pitch_x_um,
        "pitch_y_um": pitch_y_um,
        "plane_level_metadata": plane_meta,
    }

