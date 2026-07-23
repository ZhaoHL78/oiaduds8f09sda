from __future__ import annotations

import numpy as np


def apply_homography_xy(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    mapped = homo @ np.asarray(matrix, dtype=np.float64).T
    mapped_xy = mapped[:, :2] / (mapped[:, 2:3] + 1e-12)
    return mapped_xy.reshape(np.asarray(xy).shape)


def display_to_raw_xy(xy_display: np.ndarray, raw_shape: tuple[int, int], orientation: str) -> np.ndarray:
    xy = np.asarray(xy_display, dtype=np.float64).reshape(-1, 2)
    h, w = raw_shape
    x = xy[:, 0]
    y = xy[:, 1]
    if orientation == "raw":
        raw_x, raw_y = x, y
    elif orientation == "flipud":
        raw_x, raw_y = x, (h - 1) - y
    elif orientation == "fliplr":
        raw_x, raw_y = (w - 1) - x, y
    elif orientation == "rot180":
        raw_x, raw_y = (w - 1) - x, (h - 1) - y
    elif orientation == "rot90":
        raw_x, raw_y = y, (h - 1) - x
    elif orientation == "rot270":
        raw_x, raw_y = (w - 1) - y, x
    else:
        raise ValueError(f"Coordinate inverse for orientation {orientation!r} is not implemented.")
    return np.column_stack([raw_x, raw_y]).reshape(np.asarray(xy_display).shape)


def map_afm_pixels_to_ebsd(
    afm_shape: tuple[int, int],
    sem_shape: tuple[int, int],
    ebsd_shape: tuple[int, int],
    matrix_afm_resized_to_sem_display: np.ndarray,
    *,
    sem_display_orientation: str,
) -> dict[str, np.ndarray]:
    """Backward map every AFM-display pixel to the original EBSD map grid."""
    afm_h, afm_w = afm_shape
    sem_h, sem_w = sem_shape
    ebsd_h, ebsd_w = ebsd_shape
    row, col = np.indices(afm_shape, dtype=np.float64)
    x_small = col * (sem_w - 1) / max(afm_w - 1, 1)
    y_small = row * (sem_h - 1) / max(afm_h - 1, 1)
    afm_resized_xy = np.dstack([x_small, y_small])
    sem_display_xy = apply_homography_xy(matrix_afm_resized_to_sem_display, afm_resized_xy)
    sem_raw_xy = display_to_raw_xy(sem_display_xy, sem_shape, sem_display_orientation)
    ebsd_col = sem_raw_xy[..., 0] * (ebsd_w - 1) / max(sem_w - 1, 1)
    ebsd_row = sem_raw_xy[..., 1] * (ebsd_h - 1) / max(sem_h - 1, 1)
    inside_sem = (
        (sem_display_xy[..., 0] >= 0)
        & (sem_display_xy[..., 0] <= sem_w - 1)
        & (sem_display_xy[..., 1] >= 0)
        & (sem_display_xy[..., 1] <= sem_h - 1)
    )
    inside_ebsd = (ebsd_col >= 0) & (ebsd_col <= ebsd_w - 1) & (ebsd_row >= 0) & (ebsd_row <= ebsd_h - 1)
    nearest_row = np.clip(np.rint(ebsd_row).astype(np.int32), 0, ebsd_h - 1)
    nearest_col = np.clip(np.rint(ebsd_col).astype(np.int32), 0, ebsd_w - 1)
    nearest_index = nearest_row * ebsd_w + nearest_col
    return {
        "afm_resized_xy": afm_resized_xy.astype(np.float32),
        "sem_display_xy": sem_display_xy.astype(np.float32),
        "sem_raw_xy": sem_raw_xy.astype(np.float32),
        "ebsd_row": ebsd_row.astype(np.float32),
        "ebsd_col": ebsd_col.astype(np.float32),
        "nearest_row": nearest_row,
        "nearest_col": nearest_col,
        "nearest_index": nearest_index,
        "inside": inside_sem & inside_ebsd,
    }


def homography_center_polar_rotation(matrix: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    """Return the in-plane rotation part of the homography Jacobian at center."""
    h = np.asarray(matrix, dtype=np.float64)
    x, y = center_xy
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    u_num = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v_num = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    j = np.empty((2, 2), dtype=np.float64)
    j[0, 0] = (h[0, 0] * den - u_num * h[2, 0]) / (den * den)
    j[0, 1] = (h[0, 1] * den - u_num * h[2, 1]) / (den * den)
    j[1, 0] = (h[1, 0] * den - v_num * h[2, 0]) / (den * den)
    j[1, 1] = (h[1, 1] * den - v_num * h[2, 1]) / (den * den)
    u, _s, vt = np.linalg.svd(j)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r.astype(np.float64)

