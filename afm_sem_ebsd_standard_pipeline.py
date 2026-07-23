from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage


LINE_COLORS = ("#fee400", "#c800ff", "#0032ff", "#00e020", "#ff4040", "#00bfff", "#ff7f27", "#9b30ff")
HKL_CANDIDATES = np.array(
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [0, 1, 2], [0, 1, 3]],
    dtype=np.float64,
)
HKL_LABELS = tuple(f"{{{int(h)}{int(k)}{int(l)}}}" for h, k, l in HKL_CANDIDATES)


@dataclass(frozen=True)
class Up2Info:
    version: int
    width: int
    height: int
    header_bytes: int
    count: int


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def scalar_dataset(dataset: Any) -> Any:
    if dataset is None:
        return None
    value = np.asarray(dataset[()]).reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def robust_rescale(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def orient_image(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "raw":
        return image
    if orientation == "flipud":
        return np.flipud(image)
    if orientation == "fliplr":
        return np.fliplr(image)
    if orientation == "rot90":
        return np.rot90(image, 1)
    if orientation == "rot180":
        return np.rot90(image, 2)
    if orientation == "rot270":
        return np.rot90(image, 3)
    raise ValueError(f"Unsupported orientation: {orientation}")


def parse_igor_note(note: bytes | str) -> dict[str, str]:
    text = note.decode("utf-8", "ignore") if isinstance(note, bytes) else str(note)
    out: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def read_afm_height(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from igor2.binarywave import load as load_ibw

    afm_cfg = config["afm"]
    path = Path(afm_cfg["path"])
    payload = load_ibw(path)
    wave = payload["wave"]
    data = np.asarray(wave["wData"], dtype=np.float32)
    note = parse_igor_note(wave.get("note", b""))
    labels_raw = wave.get("labels", [])
    labels: list[str] = []
    for label in labels_raw[-1] if labels_raw else []:
        if isinstance(label, bytes):
            text = label.decode("utf-8", "ignore").strip("\x00")
        else:
            text = str(label).strip("\x00")
        if text:
            labels.append(text)
    if data.ndim == 3:
        channel = afm_cfg["height_channel"]
        if channel in labels:
            height = data[:, :, labels.index(channel)]
        else:
            height = data[:, :, 0]
    elif data.ndim == 2:
        height = data
    else:
        raise ValueError(f"Unsupported AFM data shape: {data.shape}")
    height_um = height.astype(np.float32) * float(afm_cfg.get("height_unit_scale_to_um", 1.0))
    display = orient_image(height_um, afm_cfg.get("display_orientation", "raw")).astype(np.float32)
    scan_size_um = afm_cfg.get("scan_size_um")
    if scan_size_um is None:
        raw_scan = note.get("ScanSize") or note.get("FastScanSize") or note.get("Scan Size")
        scan_size_um = float(raw_scan) * 1e6 if raw_scan is not None else float(display.shape[1])
    meta = {
        "path": str(path),
        "raw_shape": list(height.shape),
        "display_shape": list(display.shape),
        "channels": labels,
        "height_channel": afm_cfg.get("height_channel"),
        "display_orientation": afm_cfg.get("display_orientation", "raw"),
        "scan_size_um": float(scan_size_um),
        "height_unit": "um",
        "height_min_um": float(np.nanmin(display)),
        "height_max_um": float(np.nanmax(display)),
        "note": "AFM display height is the final spatial reference grid.",
    }
    return display, meta


def read_sem_ebsd(config: dict[str, Any]) -> dict[str, Any]:
    ebsd_cfg = config["sem_ebsd"]
    h5_path = Path(ebsd_cfg["h5_path"])
    h5_group = ebsd_cfg["h5_group"]
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        step_x = float(np.asarray(group["Sample/Step X"][()]).reshape(-1)[0])
        step_y = float(np.asarray(group["Sample/Step Y"][()]).reshape(-1)[0])
        grid_type = scalar_dataset(group["Sample/Grid Type"])
        sample_tilt = scalar_dataset(group["Sample/Sample Tilt"]) if "Sample/Sample Tilt" in group else None
        sem_raw = np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32)
        data = group["EBSD/ANG/DATA/DATA"]
        orientations = np.asarray(data["Orientations"][:], dtype=np.float64).reshape(-1, 3, 3)
        iq = np.asarray(data["IQ"][:], dtype=np.float32).reshape(nrows, ncols)
        ci = np.asarray(data["CI"][:], dtype=np.float32).reshape(nrows, ncols)
        phase = np.asarray(data["Phase"][:], dtype=np.int16).reshape(nrows, ncols)
        valid = np.asarray(data["Valid"][:], dtype=bool).reshape(nrows, ncols)
        names = set(data.dtype.names or ())
        fit = np.asarray(data["Fit"][:], dtype=np.float32).reshape(nrows, ncols) if "Fit" in names else np.full((nrows, ncols), np.nan, np.float32)
        phase_meta: dict[str, Any] = {}
        phase_root = group.get("EBSD/ANG/HEADER/Phase")
        if phase_root is not None:
            for key, phase_group in phase_root.items():
                phase_meta[key] = {
                    "Material Name": scalar_dataset(phase_group.get("Material Name")),
                    "Point Group": scalar_dataset(phase_group.get("Point Group")),
                    "Laue Group": scalar_dataset(phase_group.get("Laue Group")),
                    "Lattice Constant A": scalar_dataset(phase_group.get("Lattice Constant A")),
                    "Lattice Constant B": scalar_dataset(phase_group.get("Lattice Constant B")),
                    "Lattice Constant C": scalar_dataset(phase_group.get("Lattice Constant C")),
                }
    sem_display = orient_image(sem_raw, ebsd_cfg.get("sem_display_orientation", "raw")).astype(np.float32)
    return {
        "h5_path": str(h5_path),
        "h5_group": h5_group,
        "nrows": nrows,
        "ncols": ncols,
        "step_x_um": step_x,
        "step_y_um": step_y,
        "grid_type": grid_type,
        "sample_tilt_deg": sample_tilt,
        "sem_raw": sem_raw,
        "sem_display": sem_display,
        "sem_display_orientation": ebsd_cfg.get("sem_display_orientation", "raw"),
        "sem_display_to_ebsd_grid": ebsd_cfg.get("sem_display_to_ebsd_grid", "raw"),
        "orientations": orientations,
        "iq": iq,
        "ci": ci,
        "phase": phase,
        "valid": valid,
        "fit": fit,
        "phase_metadata": phase_meta,
    }


def homography_xy(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    flat = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    homo = np.column_stack([flat, np.ones(flat.shape[0])])
    mapped = homo @ matrix.T
    out = mapped[:, :2] / (mapped[:, 2:3] + 1e-12)
    return out.reshape(np.asarray(xy).shape)


def homography_center_rotation(matrix: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    h = np.asarray(matrix, dtype=np.float64)
    x, y = center_xy
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    jac = np.array(
        [
            [(h[0, 0] * den - u * h[2, 0]) / den**2, (h[0, 1] * den - u * h[2, 1]) / den**2],
            [(h[1, 0] * den - v * h[2, 0]) / den**2, (h[1, 1] * den - v * h[2, 1]) / den**2],
        ],
        dtype=np.float64,
    )
    u_svd, _s, vt = np.linalg.svd(jac)
    rot = u_svd @ vt
    if np.linalg.det(rot) < 0:
        u_svd[:, -1] *= -1.0
        rot = u_svd @ vt
    return rot


def scale_matrix(src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
    sh, sw = src_shape
    dh, dw = dst_shape
    sx = (dw - 1) / max(sw - 1, 1)
    sy = (dh - 1) / max(sh - 1, 1)
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def orientation_matrix(mode: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mode == "raw":
        return np.eye(3, dtype=np.float64)
    if mode == "flipud":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if mode == "fliplr":
        return np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if mode == "rot180":
        return np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported SEM-display to EBSD-grid orientation: {mode}")


def build_transforms(
    afm_shape: tuple[int, int],
    sem_shape: tuple[int, int],
    ebsd_shape: tuple[int, int],
    matrix_afm_resized_to_sem: np.ndarray,
    sem_display_to_ebsd_grid: str,
) -> dict[str, np.ndarray]:
    t_afm_to_resized = scale_matrix(afm_shape, sem_shape)
    t_sem_display_orientation = orientation_matrix(sem_display_to_ebsd_grid, sem_shape)
    t_sem_to_ebsd = scale_matrix(sem_shape, ebsd_shape) @ t_sem_display_orientation
    t_afm_to_sem = matrix_afm_resized_to_sem @ t_afm_to_resized
    t_afm_to_ebsd = t_sem_to_ebsd @ matrix_afm_resized_to_sem @ t_afm_to_resized
    return {
        "T_afm_reference_to_afm_resized_for_alignment": t_afm_to_resized,
        "T_afm_resized_to_sem_display": matrix_afm_resized_to_sem,
        "T_sem_display_to_ebsd_grid": t_sem_to_ebsd,
        "T_afm_reference_to_sem_display": t_afm_to_sem,
        "T_afm_reference_to_ebsd_grid": t_afm_to_ebsd,
    }


def map_afm_to_sem_ebsd(
    afm_shape: tuple[int, int],
    sem_shape: tuple[int, int],
    ebsd_shape: tuple[int, int],
    transforms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    ah, aw = afm_shape
    sh, sw = sem_shape
    eh, ew = ebsd_shape
    row, col = np.indices((ah, aw), dtype=np.float64)
    xy = np.dstack([col, row])
    sem_xy = homography_xy(transforms["T_afm_reference_to_sem_display"], xy)
    ebsd_xy = homography_xy(transforms["T_afm_reference_to_ebsd_grid"], xy)
    sem_x = sem_xy[..., 0]
    sem_y = sem_xy[..., 1]
    ebsd_col = ebsd_xy[..., 0]
    ebsd_row = ebsd_xy[..., 1]
    inside_sem = (sem_x >= 0) & (sem_x <= sw - 1) & (sem_y >= 0) & (sem_y <= sh - 1)
    inside_ebsd = (ebsd_col >= 0) & (ebsd_col <= ew - 1) & (ebsd_row >= 0) & (ebsd_row <= eh - 1)
    rr = np.clip(np.rint(ebsd_row).astype(np.int32), 0, eh - 1)
    cc = np.clip(np.rint(ebsd_col).astype(np.int32), 0, ew - 1)
    return {
        "sem_display_x": sem_x.astype(np.float32),
        "sem_display_y": sem_y.astype(np.float32),
        "ebsd_col_float": ebsd_col.astype(np.float32),
        "ebsd_row_float": ebsd_row.astype(np.float32),
        "ebsd_row": rr,
        "ebsd_col": cc,
        "ebsd_index": (rr * ew + cc).astype(np.int32),
        "inside": (inside_sem & inside_ebsd),
    }


def sample_linear(image: np.ndarray, row: np.ndarray, col: np.ndarray, inside: np.ndarray) -> np.ndarray:
    coords = np.vstack([row.reshape(-1), col.reshape(-1)])
    if image.ndim == 2:
        out = ndimage.map_coordinates(image.astype(np.float32), coords, order=1, mode="nearest").reshape(row.shape)
        out[~inside] = np.nan
        return out.astype(np.float32)
    channels = []
    for channel in range(image.shape[2]):
        values = ndimage.map_coordinates(image[..., channel].astype(np.float32), coords, order=1, mode="nearest").reshape(row.shape)
        values[~inside] = 0.0
        channels.append(values)
    return np.stack(channels, axis=-1).astype(np.float32)


def sample_ebsd_to_afm(ebsd: dict[str, Any], mapping: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rr = mapping["ebsd_row"]
    cc = mapping["ebsd_col"]
    idx = mapping["ebsd_index"]
    inside = mapping["inside"]
    orientations = ebsd["orientations"][idx.reshape(-1)].reshape(*rr.shape, 3, 3).astype(np.float32)
    phase = ebsd["phase"][rr, cc].astype(np.int16)
    valid = ebsd["valid"][rr, cc].astype(bool) & inside
    iq = sample_linear(ebsd["iq"], mapping["ebsd_row_float"], mapping["ebsd_col_float"], inside)
    ci = sample_linear(ebsd["ci"], mapping["ebsd_row_float"], mapping["ebsd_col_float"], inside)
    fit = sample_linear(ebsd["fit"], mapping["ebsd_row_float"], mapping["ebsd_col_float"], inside)
    phase[~inside] = 0
    return {
        "orientation": orientations,
        "phase": phase,
        "valid": valid,
        "iq": iq,
        "ci": ci,
        "fit": fit,
        "ebsd_index": idx,
    }


def plane_level(height_um: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    y, x = np.indices(height_um.shape, dtype=np.float64)
    valid = np.isfinite(height_um)
    design = np.column_stack([x[valid], y[valid], np.ones(valid.sum())])
    coeff, *_ = np.linalg.lstsq(design, height_um[valid].astype(np.float64), rcond=None)
    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    return (height_um - plane).astype(np.float32), {"a_um_per_col": float(coeff[0]), "b_um_per_row": float(coeff[1]), "c_um": float(coeff[2])}


def compute_afm_normals(
    height_um: np.ndarray,
    scan_size_um: float,
    smooth_sigma_px: float,
    plane_level_enabled: bool,
    row_derivative_sign: int,
    afm_to_sem_rotation: np.ndarray,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    if plane_level_enabled:
        leveled, plane_meta = plane_level(height_um)
    else:
        leveled = height_um.astype(np.float32)
        plane_meta = {}
    smooth = ndimage.gaussian_filter(leveled, sigma=float(smooth_sigma_px), mode="nearest") if smooth_sigma_px > 0 else leveled
    pitch_x = scan_size_um / max(height_um.shape[1] - 1, 1)
    pitch_y = scan_size_um / max(height_um.shape[0] - 1, 1)
    dz_dx = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 1, 0, scale=1.0 / 32.0) / max(pitch_x, 1e-12)
    dz_drow = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 0, 1, scale=1.0 / 32.0) / max(pitch_y, 1e-12)
    normals_afm = np.dstack([-dz_dx, -float(row_derivative_sign) * dz_drow, np.ones_like(smooth, dtype=np.float32)])
    normals_afm /= np.linalg.norm(normals_afm, axis=2, keepdims=True) + 1e-12
    xy = normals_afm[..., :2].reshape(-1, 2) @ np.asarray(afm_to_sem_rotation, dtype=np.float64).T
    normals_sample = np.column_stack([xy, normals_afm[..., 2].reshape(-1)]).reshape(normals_afm.shape)
    normals_sample /= np.linalg.norm(normals_sample, axis=2, keepdims=True) + 1e-12
    normals_sample[normals_sample[..., 2] < 0] *= -1.0
    slope_deg = np.degrees(np.arccos(np.clip(normals_sample[..., 2], -1.0, 1.0))).astype(np.float32)
    aspect_deg = np.degrees(np.arctan2(normals_sample[..., 1], normals_sample[..., 0])).astype(np.float32)
    return {
        "height_leveled_um": leveled.astype(np.float32),
        "height_smoothed_um": smooth.astype(np.float32),
        "dz_dx": dz_dx.astype(np.float32),
        "dz_drow": dz_drow.astype(np.float32),
        "normals_afm": normals_afm.astype(np.float32),
        "normals_sample": normals_sample.astype(np.float32),
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "pitch_x_um": float(pitch_x),
        "pitch_y_um": float(pitch_y),
        "plane_level": plane_meta,
    }


def normal_direction_rgb(normals: np.ndarray, tilt_ref_deg: float) -> np.ndarray:
    azimuth = (np.arctan2(normals[..., 1], normals[..., 0]) + np.pi) / (2.0 * np.pi)
    tilt = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    saturation = np.clip(tilt / max(float(tilt_ref_deg), 1e-6), 0.0, 1.0)
    value = np.ones_like(saturation, dtype=np.float32) * 0.97
    return hsv_to_rgb(np.dstack([azimuth, saturation, value])).astype(np.float32)


def fold_cubic(vectors: np.ndarray) -> np.ndarray:
    folded = np.sort(np.abs(vectors), axis=-1)[..., ::-1]
    folded /= np.linalg.norm(folded, axis=-1, keepdims=True) + 1e-12
    return folded.astype(np.float32)


def facet_type_rgb(folded: np.ndarray) -> np.ndarray:
    basis = np.array(
        [
            [1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)],
            [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)],
            [0.0, 0.0, 1.0 / np.sqrt(3.0)],
        ],
        dtype=np.float64,
    )
    coeff = np.linalg.solve(basis, folded.reshape(-1, 3).T).T
    coeff = np.clip(coeff, 0.0, None)
    coeff /= np.maximum(coeff.max(axis=1, keepdims=True), 1e-12)
    return coeff.reshape(folded.shape).astype(np.float32)


def nearest_hkl(folded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = HKL_CANDIDATES / np.linalg.norm(HKL_CANDIDATES, axis=1, keepdims=True)
    flat = folded.reshape(-1, 3).astype(np.float64)
    dots = np.clip(flat @ candidates.T, -1.0, 1.0)
    best = np.argmax(dots, axis=1)
    angle = np.degrees(np.arccos(dots[np.arange(flat.shape[0]), best]))
    return best.reshape(folded.shape[:2]).astype(np.int16), angle.reshape(folded.shape[:2]).astype(np.float32)


def ebsd_ipf_z_rgb(ebsd: dict[str, Any]) -> np.ndarray:
    from export_h5_ipf_bse_maps import cubic_ipf_z_colors

    rgb = cubic_ipf_z_colors(
        ebsd["orientations"].reshape(-1, 9),
        ebsd["valid"].reshape(-1),
        ebsd["ci"].reshape(-1),
        ci_weight=False,
    )
    return rgb.reshape(ebsd["nrows"], ebsd["ncols"], 3)


def read_optional_rgb(path: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def draw_marker(ax: plt.Axes, xy: tuple[float, float], radius: float = 9.0) -> None:
    x, y = xy
    ax.scatter([x], [y], s=95, facecolors="none", edgecolors="black", linewidths=2.5, zorder=5)
    ax.scatter([x], [y], s=34, facecolors="none", edgecolors="red", linewidths=1.6, zorder=6)


def save_scalar(path: Path, values: np.ndarray, title: str, cmap: str, label: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    im = ax.imshow(values, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label=label, shrink=0.82)
    fig.savefig(path)
    plt.close(fig)


def save_rgb(path: Path, rgb: np.ndarray, title: str, dpi: int, mask: np.ndarray | None = None, marker: tuple[float, float] | None = None) -> None:
    image = np.clip(rgb, 0.0, 1.0).copy()
    if mask is not None:
        image[~mask] = 0.0
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    ax.imshow(image)
    if marker is not None:
        draw_marker(ax, marker)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def composite_over_gray(gray: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    base = np.dstack([robust_rescale(gray)] * 3)
    alpha = rgba[..., 3:4]
    return np.clip(base * (1.0 - alpha) + rgba[..., :3] * alpha, 0.0, 1.0)


def make_rgba(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    out[..., 3] = mask.astype(np.float32) * float(alpha)
    return out


def warp_resized_afm_to_sem(afm_display_height: np.ndarray, sem_shape: tuple[int, int], h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(robust_rescale(afm_display_height), (sem_shape[1], sem_shape[0]), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(resized.astype(np.float32), h, (sem_shape[1], sem_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.ones(sem_shape, dtype=np.float32), h, (sem_shape[1], sem_shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    return warped, mask


def save_overlay(path: Path, base_gray: np.ndarray, overlay_rgb: np.ndarray, mask: np.ndarray, alpha: float, title: str, marker: tuple[float, float] | None, dpi: int) -> None:
    comp = composite_over_gray(base_gray, make_rgba(overlay_rgb, mask, alpha))
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=dpi, constrained_layout=True)
    ax.imshow(comp)
    if marker is not None:
        draw_marker(ax, marker)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_sem_ebsd_correspondence(path: Path, sem_display: np.ndarray, ipf_on_sem: np.ndarray, software_ipf: np.ndarray | None, dpi: int) -> None:
    cols = 3 if software_ipf is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5.6 * cols, 5.0), dpi=dpi, constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].imshow(robust_rescale(sem_display), cmap="gray")
    axes[0].set_title("H5 SEM display")
    axes[1].imshow(ipf_on_sem)
    axes[1].set_title("H5 EBSD IPF-Z resampled to SEM display")
    if software_ipf is not None:
        axes[2].imshow(software_ipf)
        axes[2].set_title("Software IPF reference, resized")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_normal_color_wheel(path: Path, tilt_ref_deg: float, dpi: int) -> None:
    size = 500
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.sqrt(xx * xx + yy * yy)
    hue = (np.arctan2(yy, xx) + np.pi) / (2.0 * np.pi)
    sat = np.clip(radius, 0.0, 1.0)
    val = np.ones_like(sat) * 0.97
    rgb = hsv_to_rgb(np.dstack([hue, sat, val]))
    rgb[radius > 1] = 0.0
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=dpi, constrained_layout=True)
    ax.imshow(rgb, extent=(-1, 1, -1, 1))
    ax.text(0.96, 0.0, "0 deg", ha="right", va="center")
    ax.text(0.0, 0.92, "+90 deg", ha="center", va="top")
    ax.text(-0.96, 0.0, "180 deg", ha="left", va="center")
    ax.text(0.0, -0.92, "-90 deg", ha="center", va="bottom")
    ax.set_title(f"Normal azimuth color key; saturation reaches {tilt_ref_deg:g} deg tilt")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def read_up2_info(path: Path) -> Up2Info:
    with path.open("rb") as file:
        version, width, height, header_bytes = struct.unpack("<4I", file.read(16))
    count = (path.stat().st_size - header_bytes) // (width * height * 2)
    return Up2Info(version, width, height, header_bytes, count)


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, Up2Info]:
    info = read_up2_info(path)
    if not (0 <= index < info.count):
        raise IndexError(f"UP2 index {index} outside 0..{info.count - 1}")
    offset = info.header_bytes + index * info.width * info.height * 2
    with path.open("rb") as file:
        file.seek(offset)
        pattern = np.fromfile(file, dtype="<u2", count=info.width * info.height).reshape(info.height, info.width)
    return pattern, info


def circular_mask(shape: tuple[int, int], radius_fraction: float = 0.49) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = radius_fraction * min(h, w)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def read_ohp_segments(h5_path: Path, h5_group: str, index: int, pattern_shape: tuple[int, int]) -> list[dict[str, float]]:
    h, w = pattern_shape
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        header = group["EBSD/OHP/HEADER"]
        circle_size = float(np.asarray(header["Circle Size"][()]).reshape(-1)[0])
        raw = np.asarray(group["EBSD/OHP/DATA/DATA"][index], dtype=np.float32).reshape(-1, 4)
    segments = []
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x_min, x_max = -cx, cx
    y_min, y_max = -cy, cy
    for band_id, (rho_bin, theta_deg, width, intensity) in enumerate(raw):
        if not np.isfinite([rho_bin, theta_deg, width, intensity]).all() or width <= 0 or intensity <= 0:
            continue
        rho_px = (float(rho_bin) - circle_size / 2.0) * (min(h, w) / circle_size)
        theta = math.radians(float(theta_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        pts: list[tuple[float, float]] = []
        if abs(s) > 1e-8:
            for x in (x_min, x_max):
                y = (rho_px - x * c) / s
                if y_min <= y <= y_max:
                    pts.append((x, y))
        if abs(c) > 1e-8:
            for y in (y_min, y_max):
                x = (rho_px - y * s) / c
                if x_min <= x <= x_max:
                    pts.append((x, y))
        unique: list[tuple[float, float]] = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-4 and abs(p[1] - q[1]) < 1e-4 for q in unique):
                unique.append(p)
        if len(unique) < 2:
            continue
        (x0, y0), (x1, y1) = unique[:2]
        segments.append(
            {
                "band_id": band_id,
                "row0": cy - y0,
                "col0": x0 + cx,
                "row1": cy - y1,
                "col1": x1 + cx,
                "rho_bin": float(rho_bin),
                "theta_deg": float(theta_deg),
                "bandwidth": float(width),
                "intensity": float(intensity),
            }
        )
    return segments


def save_kikuchi_with_bands(path: Path, pattern: np.ndarray, segments: list[dict[str, float]], metadata: dict[str, Any], dpi: int) -> None:
    mask = circular_mask(pattern.shape)
    display = robust_rescale(np.where(mask, pattern, np.nan), 0.5, 99.6)
    display[~mask] = 0.0
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=dpi, constrained_layout=True)
    ax.imshow(display, cmap="gray")
    for seg in segments:
        color = LINE_COLORS[int(seg["band_id"]) % len(LINE_COLORS)]
        ax.plot([seg["col0"], seg["col1"]], [seg["row0"], seg["row1"]], color=color, lw=max(1.2, 0.35 * float(seg["bandwidth"])), alpha=0.86)
        ax.text(0.5 * (seg["col0"] + seg["col1"]), 0.5 * (seg["row0"] + seg["row1"]), str(int(seg["band_id"] + 1)), color="white", fontsize=7, ha="center", va="center")
    ax.set_title(
        f"Kikuchi + OHP bands\nidx={metadata['index']} row={metadata['row']} col={metadata['col']} "
        f"IQ={metadata['IQ']:.0f} CI={metadata['CI']:.3f}"
    )
    ax.axis("off")
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def save_coupling_montage(
    path: Path,
    kikuchi_path: Path,
    height_rgb: np.ndarray,
    ipf_afm: np.ndarray,
    normal_rgb: np.ndarray,
    surface_rgb: np.ndarray,
    color_key_path: Path,
    selected_xy: tuple[int, int],
    dpi: int,
) -> None:
    from PIL import Image

    kikuchi = np.asarray(Image.open(kikuchi_path).convert("RGB"), dtype=np.float32) / 255.0
    key = np.asarray(Image.open(color_key_path).convert("RGB"), dtype=np.float32) / 255.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=dpi, facecolor="black", constrained_layout=True)
    panels = [
        (kikuchi, "Kikuchi + OHP bands", None),
        (height_rgb, "AFM height reference", selected_xy),
        (ipf_afm, "EBSD IPF-Z aligned to AFM", selected_xy),
        (normal_rgb, "AFM sample normal", selected_xy),
        (surface_rgb, "Crystal surface index", selected_xy),
        (key, "Normal azimuth key", None),
    ]
    for ax, (image, title, marker) in zip(axes.ravel(), panels):
        ax.imshow(image)
        if marker is not None:
            draw_marker(ax, marker)
        ax.set_title(title, color="white")
        ax.axis("off")
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def save_3d_surface_png(path: Path, height_um: np.ndarray, rgb: np.ndarray, mask: np.ndarray, scan_size_um: float, stride: int, dpi: int) -> None:
    h = height_um[::stride, ::stride]
    c = rgb[::stride, ::stride].copy()
    m = mask[::stride, ::stride]
    c[~m] = 0.0
    yy, xx = np.indices(h.shape)
    x = xx * scan_size_um / max(h.shape[1] - 1, 1)
    y = yy * scan_size_um / max(h.shape[0] - 1, 1)
    fig = plt.figure(figsize=(8.5, 7.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, h, facecolors=np.clip(c, 0, 1), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_title("AFM 3D surface colored by crystal surface index")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("height (um)")
    ax.view_init(elev=52, azim=-62)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_3d_surface_html(path: Path, height_um: np.ndarray, rgb: np.ndarray, mask: np.ndarray, scan_size_um: float, stride: int) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    try:
        z = height_um[::stride, ::stride].astype(np.float32)
        colors = np.clip(rgb[::stride, ::stride], 0.0, 1.0)
        m = mask[::stride, ::stride]
        colors[~m] = 0.0
        rows, cols = z.shape
        yy, xx = np.indices((rows, cols))
        x = (xx * scan_size_um / max(cols - 1, 1)).reshape(-1)
        y = (yy * scan_size_um / max(rows - 1, 1)).reshape(-1)
        zz = z.reshape(-1)
        vertexcolor = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors.reshape(-1, 3)]
        i: list[int] = []
        j: list[int] = []
        k: list[int] = []
        for r in range(rows - 1):
            base = r * cols
            nxt = (r + 1) * cols
            for c in range(cols - 1):
                if not (m[r, c] or m[r + 1, c] or m[r, c + 1] or m[r + 1, c + 1]):
                    continue
                p00 = base + c
                p01 = base + c + 1
                p10 = nxt + c
                p11 = nxt + c + 1
                i.extend([p00, p01])
                j.extend([p10, p10])
                k.extend([p01, p11])
        mesh = go.Mesh3d(x=x, y=y, z=zz, i=i, j=j, k=k, vertexcolor=vertexcolor, opacity=1.0, flatshading=False)
        fig = go.Figure(mesh)
        fig.update_layout(
            title="AFM 3D surface colored by crystal surface index",
            scene=dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="height (um)", aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.write_html(path)
    except Exception:
        return False
    return True


def select_point(config: dict[str, Any], valid_mask: np.ndarray, ci: np.ndarray, iq: np.ndarray) -> tuple[int, int]:
    point_cfg = config.get("selected_point", {})
    x = int(point_cfg.get("afm_x_px", valid_mask.shape[1] // 2))
    y = int(point_cfg.get("afm_y_px", valid_mask.shape[0] // 2))
    if 0 <= y < valid_mask.shape[0] and 0 <= x < valid_mask.shape[1] and valid_mask[y, x]:
        return x, y
    if not point_cfg.get("fallback_to_quality_point_if_invalid", True):
        return x, y
    score = np.where(valid_mask, np.nan_to_num(ci, nan=0.0) * robust_rescale(iq), -np.inf)
    y2, x2 = np.unravel_index(int(np.nanargmax(score)), score.shape)
    return int(x2), int(y2)


def write_hkl_summary(path: Path, best: np.ndarray, angle: np.ndarray, valid: np.ndarray, threshold: float) -> None:
    rows: list[dict[str, Any]] = []
    total = int(valid.sum())
    for i, label in enumerate(HKL_LABELS):
        mask = valid & (best == i) & (angle <= threshold)
        rows.append({"label": label, "pixels": int(mask.sum()), "fraction_of_valid": float(mask.sum() / max(total, 1)), "threshold_deg": threshold})
    high = valid & (angle > threshold)
    rows.append({"label": "high-index/unassigned", "pixels": int(high.sum()), "fraction_of_valid": float(high.sum() / max(total, 1)), "threshold_deg": threshold})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "pixels", "fraction_of_valid", "threshold_deg"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard AFM->SEM->EBSD alignment and AFM-reference surface-index pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/afm_sem_ebsd_standard_pipeline_pt_highres60.json"))
    args = parser.parse_args()

    config = read_json(args.config)
    np.random.seed(int(config.get("random_seed", 0)))
    output_dir = Path(config["output_dir"])
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("visualization", {}).get("dpi", 220))
    alpha = float(config.get("visualization", {}).get("overlay_alpha", 0.62))

    afm_height, afm_meta = read_afm_height(config)
    ebsd = read_sem_ebsd(config)
    sem_display = ebsd["sem_display"]
    registration = read_json(Path(config["afm_sem_alignment"]["registration_report_path"]))
    h_afm_resized_to_sem = np.asarray(registration["matrix_afm_to_sem"], dtype=np.float64)

    transforms = build_transforms(
        afm_height.shape,
        sem_display.shape,
        (int(ebsd["nrows"]), int(ebsd["ncols"])),
        h_afm_resized_to_sem,
        ebsd["sem_display_to_ebsd_grid"],
    )
    mapping = map_afm_to_sem_ebsd(afm_height.shape, sem_display.shape, (int(ebsd["nrows"]), int(ebsd["ncols"])), transforms)
    ebsd_on_afm = sample_ebsd_to_afm(ebsd, mapping)
    ipf_grid = ebsd_ipf_z_rgb(ebsd)
    ipf_on_sem = cv2.resize(ipf_grid, (sem_display.shape[1], sem_display.shape[0]), interpolation=cv2.INTER_LINEAR)
    if ebsd["sem_display_to_ebsd_grid"] == "flipud":
        ipf_on_sem = np.flipud(ipf_on_sem)
    elif ebsd["sem_display_to_ebsd_grid"] == "fliplr":
        ipf_on_sem = np.fliplr(ipf_on_sem)
    elif ebsd["sem_display_to_ebsd_grid"] == "rot180":
        ipf_on_sem = np.rot90(ipf_on_sem, 2)
    ipf_on_afm = sample_linear(ipf_grid, mapping["ebsd_row_float"], mapping["ebsd_col_float"], mapping["inside"])

    valid = ebsd_on_afm["valid"] & (ebsd_on_afm["phase"] > 0)
    selected_x, selected_y = select_point(config, valid, ebsd_on_afm["ci"], ebsd_on_afm["iq"])
    marker_afm = (selected_x, selected_y)
    marker_sem = (float(mapping["sem_display_x"][selected_y, selected_x]), float(mapping["sem_display_y"][selected_y, selected_x]))

    rotation = homography_center_rotation(h_afm_resized_to_sem, (sem_display.shape[1] / 2.0, sem_display.shape[0] / 2.0))
    normal_cfg = config.get("surface_index", {})
    normals = compute_afm_normals(
        afm_height,
        float(afm_meta["scan_size_um"]),
        float(config["afm"].get("normal_smooth_sigma_px", 1.0)),
        bool(config["afm"].get("plane_level", True)),
        int(normal_cfg.get("image_row_derivative_sign", 1)),
        rotation,
    )
    g = ebsd_on_afm["orientation"].astype(np.float32)
    normals_sample = normals["normals_sample"].astype(np.float32)
    n_crystal = np.einsum("...ij,...j->...i", g, normals_sample).astype(np.float32)
    n_crystal /= np.linalg.norm(n_crystal, axis=2, keepdims=True) + 1e-12
    n_crystal[~valid] = 0.0
    folded = fold_cubic(n_crystal)
    surface_rgb = facet_type_rgb(folded)
    surface_rgb[~valid] = 0.0
    best_hkl, angle_hkl = nearest_hkl(folded)
    angle_hkl[~valid] = np.nan
    normal_rgb = normal_direction_rgb(normals_sample, float(normal_cfg.get("normal_tilt_ref_deg", 16.0)))
    height_rgb = plt.get_cmap("copper")(robust_rescale(afm_height))[..., :3].astype(np.float32)

    # Stage 1: AFM/SEM/EBSD spatial checks.
    save_rgb(fig_dir / "01_afm_height_reference.png", height_rgb, "AFM height reference grid", dpi, marker=marker_afm)
    save_scalar(fig_dir / "02_sem_display_used_for_alignment.png", sem_display, "H5 SEM display used for AFM alignment", "gray", "intensity", dpi)
    afm_warped, afm_warp_mask = warp_resized_afm_to_sem(afm_height, sem_display.shape, h_afm_resized_to_sem)
    afm_warp_rgb = plt.get_cmap("copper")(afm_warped)[..., :3].astype(np.float32)
    save_overlay(fig_dir / "03_afm_to_sem_alignment_check.png", sem_display, afm_warp_rgb, afm_warp_mask, alpha, "AFM height warped onto SEM display", marker_sem, dpi)
    software_ipf = read_optional_rgb(config["sem_ebsd"].get("software_ipf_reference_path"), sem_display.shape)
    save_sem_ebsd_correspondence(fig_dir / "04_sem_ebsd_correspondence_check.png", sem_display, ipf_on_sem, software_ipf, dpi)
    save_rgb(fig_dir / "05_ebsd_ipf_z_aligned_to_afm_reference.png", ipf_on_afm, "EBSD IPF-Z mapped to AFM reference", dpi, valid, marker_afm)
    save_scalar(fig_dir / "06_ebsd_iq_aligned_to_afm_reference.png", ebsd_on_afm["iq"], "EBSD IQ mapped to AFM reference", "gray", "IQ", dpi)
    save_scalar(fig_dir / "07_ebsd_ci_aligned_to_afm_reference.png", ebsd_on_afm["ci"], "EBSD CI mapped to AFM reference", "viridis", "CI", dpi)

    # Stage 2: AFM normals and surface crystallography.
    save_rgb(fig_dir / "08_afm_sample_normal_reference.png", normal_rgb, "AFM sample-normal map on AFM reference", dpi, marker=marker_afm)
    save_scalar(fig_dir / "09_afm_slope_deg_reference.png", normals["slope_deg"], "AFM local slope", "magma", "deg", dpi)
    save_rgb(fig_dir / "10_surface_index_afm_reference.png", surface_rgb, "AFM surface normal in EBSD crystal frame", dpi, valid, marker_afm)
    save_overlay(fig_dir / "11_surface_index_over_afm_height_reference.png", afm_height, surface_rgb, valid, float(config["visualization"].get("surface_overlay_alpha", 0.82)), "Surface index over AFM height reference", marker_afm, dpi)
    save_scalar(fig_dir / "12_nearest_hkl_angle_deg_reference.png", angle_hkl, "Angle to nearest low-index cubic plane", "magma", "deg", dpi)
    color_key_path = fig_dir / "13_normal_azimuth_color_key.png"
    save_normal_color_wheel(color_key_path, float(normal_cfg.get("normal_tilt_ref_deg", 16.0)), dpi)

    selected_idx = int(ebsd_on_afm["ebsd_index"][selected_y, selected_x])
    selected_row = int(mapping["ebsd_row"][selected_y, selected_x])
    selected_col = int(mapping["ebsd_col"][selected_y, selected_x])
    pattern, up2_info = read_up2_pattern(Path(config["sem_ebsd"]["up2_path"]), selected_idx)
    ohp_segments = read_ohp_segments(Path(config["sem_ebsd"]["h5_path"]), config["sem_ebsd"]["h5_group"], selected_idx, pattern.shape)
    selected_meta = {
        "index": selected_idx,
        "row": selected_row,
        "col": selected_col,
        "IQ": float(ebsd_on_afm["iq"][selected_y, selected_x]),
        "CI": float(ebsd_on_afm["ci"][selected_y, selected_x]),
    }
    kikuchi_path = fig_dir / "14_selected_kikuchi_ohp_bands.png"
    save_kikuchi_with_bands(kikuchi_path, pattern, ohp_segments, selected_meta, dpi)
    save_coupling_montage(fig_dir / "15_standard_pipeline_montage.png", kikuchi_path, height_rgb, ipf_on_afm, normal_rgb, surface_rgb, color_key_path, marker_afm, dpi)
    save_3d_surface_png(fig_dir / "16_surface_index_3d.png", normals["height_leveled_um"], surface_rgb, valid, float(afm_meta["scan_size_um"]), int(config["visualization"].get("plot_3d_stride", 5)), dpi)
    html_written = save_3d_surface_html(fig_dir / "16_surface_index_3d_interactive.html", normals["height_leveled_um"], surface_rgb, valid, float(afm_meta["scan_size_um"]), int(config["visualization"].get("plot_3d_stride", 5)))

    threshold = float(normal_cfg.get("hkl_angle_threshold_deg", 10.0))
    write_hkl_summary(data_dir / "nearest_hkl_summary.csv", best_hkl, angle_hkl, valid, threshold)
    np.savez_compressed(
        data_dir / "afm_reference_ebsd_mapping_and_surface_index.npz",
        afm_height_um=afm_height.astype(np.float32),
        afm_height_leveled_um=normals["height_leveled_um"],
        afm_height_smoothed_um=normals["height_smoothed_um"],
        sem_display_x=mapping["sem_display_x"],
        sem_display_y=mapping["sem_display_y"],
        ebsd_row_float=mapping["ebsd_row_float"],
        ebsd_col_float=mapping["ebsd_col_float"],
        ebsd_row=mapping["ebsd_row"],
        ebsd_col=mapping["ebsd_col"],
        ebsd_index=ebsd_on_afm["ebsd_index"],
        phase=ebsd_on_afm["phase"],
        valid=valid,
        iq=ebsd_on_afm["iq"],
        ci=ebsd_on_afm["ci"],
        fit=ebsd_on_afm["fit"],
        normals_sample=normals_sample,
        normals_crystal=n_crystal,
        folded_cubic_direction=folded,
        surface_rgb=surface_rgb,
        nearest_hkl_index=best_hkl,
        nearest_hkl_angle_deg=angle_hkl,
        ipf_z_rgb_on_afm=ipf_on_afm,
    )

    metadata = {
        "status": "completed",
        "method": "Standard AFM->SEM->EBSD pipeline: use existing AFM-SEM alignment, then use SEM-display to EBSD-grid correspondence before surface-index calculation.",
        "afm": afm_meta,
        "sem_ebsd": {
            "h5_path": ebsd["h5_path"],
            "h5_group": ebsd["h5_group"],
            "sem_display_orientation": ebsd["sem_display_orientation"],
            "sem_display_to_ebsd_grid": ebsd["sem_display_to_ebsd_grid"],
            "sem_shape": list(sem_display.shape),
            "ebsd_shape_rows_cols": [int(ebsd["nrows"]), int(ebsd["ncols"])],
            "step_x_um": float(ebsd["step_x_um"]),
            "step_y_um": float(ebsd["step_y_um"]),
            "grid_type": ebsd["grid_type"],
            "sample_tilt_deg": ebsd["sample_tilt_deg"],
            "phase_metadata": ebsd["phase_metadata"],
            "orientation_convention_used": config["sem_ebsd"].get("orientation_convention"),
        },
        "registration": {
            "report_path": config["afm_sem_alignment"]["registration_report_path"],
            "matrix_direction": "T_afm_resized_to_sem_display maps AFM display after resize-to-SEM into SEM display; T_afm_reference_to_ebsd_grid maps original AFM reference pixels into EBSD row/col grid.",
            "control_point_metrics": registration.get("control_point_metrics", {}),
            "transforms": {name: value.tolist() for name, value in transforms.items()},
            "normal_in_plane_rotation": rotation.tolist(),
        },
        "selected_point": {
            "afm_x_px": selected_x,
            "afm_y_px": selected_y,
            "sem_display_x_px": float(marker_sem[0]),
            "sem_display_y_px": float(marker_sem[1]),
            "ebsd_row": selected_row,
            "ebsd_col": selected_col,
            "ebsd_index": selected_idx,
            "IQ": float(ebsd_on_afm["iq"][selected_y, selected_x]),
            "CI": float(ebsd_on_afm["ci"][selected_y, selected_x]),
            "Fit": float(ebsd_on_afm["fit"][selected_y, selected_x]),
            "phase": int(ebsd_on_afm["phase"][selected_y, selected_x]),
            "n_sample": normals_sample[selected_y, selected_x].astype(float).tolist(),
            "n_crystal": n_crystal[selected_y, selected_x].astype(float).tolist(),
            "folded_cubic_direction": folded[selected_y, selected_x].astype(float).tolist(),
            "nearest_hkl": HKL_LABELS[int(best_hkl[selected_y, selected_x])],
            "nearest_hkl_angle_deg": float(angle_hkl[selected_y, selected_x]),
        },
        "up2": up2_info.__dict__,
        "valid_fraction": float(valid.mean()),
        "html_3d_written": bool(html_written),
        "outputs": {
            "figures": str(fig_dir.resolve()),
            "data_npz": str((data_dir / "afm_reference_ebsd_mapping_and_surface_index.npz").resolve()),
            "metadata": str((data_dir / "standard_pipeline_metadata.json").resolve()),
            "hkl_summary": str((data_dir / "nearest_hkl_summary.csv").resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
        },
    }
    write_json(data_dir / "standard_pipeline_metadata.json", metadata)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "selected_point": metadata["selected_point"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
