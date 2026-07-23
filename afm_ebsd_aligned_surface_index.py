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
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [0, 1, 2], [0, 1, 3]], dtype=np.float64
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_igor_note(note: bytes | str) -> dict[str, str]:
    text = note.decode("utf-8", "ignore") if isinstance(note, bytes) else str(note)
    out: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


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
    if orientation == "transpose":
        axes = (1, 0, *range(2, image.ndim))
        return np.transpose(image, axes)
    raise ValueError(f"Unsupported orientation: {orientation}")


def read_afm_height(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from igor2.binarywave import load as load_ibw

    path = Path(config["path"])
    wave = load_ibw(str(path))["wave"]
    data = np.asarray(wave["wData"], dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    labels_raw = wave.get("labels", [[], [], [], []])[2]
    labels = []
    for raw in labels_raw:
        label = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        label = label.strip("\x00").strip()
        if label:
            labels.append(label)
    if len(labels) < data.shape[2]:
        labels.extend(f"channel_{idx}" for idx in range(len(labels), data.shape[2]))
    channel = config["height_channel"]
    if channel not in labels:
        raise KeyError(f"AFM channel {channel!r} not found; available={labels}")
    raw_height = data[:, :, labels.index(channel)] * float(config["height_unit_scale_to_um"])
    display_height = orient_image(raw_height, config.get("display_orientation", "raw")).astype(np.float32)
    note = parse_igor_note(wave.get("note", b""))
    scan_size_um = float(note.get("ScanSize", config.get("scan_size_um", "nan"))) * (
        1e6 if "ScanSize" in note else 1.0
    )
    metadata = {
        "path": str(path),
        "format": "ibw",
        "raw_shape": list(raw_height.shape),
        "display_shape": list(display_height.shape),
        "channels": labels[: data.shape[2]],
        "height_channel": channel,
        "scan_size_um": scan_size_um,
        "scan_angle_deg": float(note.get("ScanAngle", "nan")) if "ScanAngle" in note else float("nan"),
        "display_orientation": config.get("display_orientation", "raw"),
        "height_min_um": float(np.nanmin(display_height)),
        "height_max_um": float(np.nanmax(display_height)),
        "note": "Height is kept as float data; color maps are display-only.",
    }
    return display_height, metadata


def robust_rescale(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return np.clip((arr - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)


def plane_level(height: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    y, x = np.indices(height.shape, dtype=np.float64)
    finite = np.isfinite(height)
    design = np.column_stack([x[finite], y[finite], np.ones(np.count_nonzero(finite))])
    coeff, *_ = np.linalg.lstsq(design, height[finite].astype(np.float64), rcond=None)
    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    return (height - plane).astype(np.float32), {
        "a_um_per_px": float(coeff[0]),
        "b_um_per_px": float(coeff[1]),
        "c_um": float(coeff[2]),
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


def read_ebsd(h5_path: Path, h5_group: str) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        step_x = float(np.asarray(group["Sample/Step X"][()]).reshape(-1)[0])
        step_y = float(np.asarray(group["Sample/Step Y"][()]).reshape(-1)[0])
        sem_raw = np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32)
        data = group["EBSD/ANG/DATA/DATA"]
        orientations = np.asarray(data["Orientations"][:], dtype=np.float64).reshape(-1, 3, 3)
        iq = np.asarray(data["IQ"][:], dtype=np.float32).reshape(nrows, ncols)
        ci = np.asarray(data["CI"][:], dtype=np.float32).reshape(nrows, ncols)
        phase = np.asarray(data["Phase"][:], dtype=np.int16).reshape(nrows, ncols)
        valid = np.asarray(data["Valid"][:], dtype=bool).reshape(nrows, ncols)
        fit = np.asarray(data["Fit"][:], dtype=np.float32).reshape(nrows, ncols)
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
    return {
        "nrows": nrows,
        "ncols": ncols,
        "step_x_um": step_x,
        "step_y_um": step_y,
        "sem_raw": sem_raw,
        "orientations": orientations,
        "iq": iq,
        "ci": ci,
        "phase": phase,
        "valid": valid,
        "fit": fit,
        "phase_metadata": phase_meta,
    }


def scalar_dataset(dataset: Any) -> Any:
    if dataset is None:
        return None
    value = np.asarray(dataset[()]).reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def map_afm_to_ebsd(
    afm_shape: tuple[int, int],
    sem_shape: tuple[int, int],
    ebsd_shape: tuple[int, int],
    matrix_afm_resized_to_sem: np.ndarray,
    sem_display_orientation: str,
) -> dict[str, np.ndarray]:
    ah, aw = afm_shape
    sh, sw = sem_shape
    eh, ew = ebsd_shape
    row, col = np.indices((ah, aw), dtype=np.float64)
    afm_small_xy = np.dstack([col * (sw - 1) / max(aw - 1, 1), row * (sh - 1) / max(ah - 1, 1)])
    sem_display_xy = homography_xy(matrix_afm_resized_to_sem, afm_small_xy)
    sem_x = sem_display_xy[..., 0]
    sem_y = sem_display_xy[..., 1]
    if sem_display_orientation == "flipud":
        sem_raw_x = sem_x
        sem_raw_y = (sh - 1) - sem_y
    elif sem_display_orientation == "raw":
        sem_raw_x = sem_x
        sem_raw_y = sem_y
    else:
        raise ValueError(f"Only raw/flipud SEM display orientation is supported here, got {sem_display_orientation}")
    ebsd_col = sem_raw_x * (ew - 1) / max(sw - 1, 1)
    ebsd_row = sem_raw_y * (eh - 1) / max(sh - 1, 1)
    inside = (sem_x >= 0) & (sem_x <= sw - 1) & (sem_y >= 0) & (sem_y <= sh - 1)
    inside &= (ebsd_col >= 0) & (ebsd_col <= ew - 1) & (ebsd_row >= 0) & (ebsd_row <= eh - 1)
    rr = np.clip(np.rint(ebsd_row).astype(np.int32), 0, eh - 1)
    cc = np.clip(np.rint(ebsd_col).astype(np.int32), 0, ew - 1)
    return {
        "sem_display_xy": sem_display_xy.astype(np.float32),
        "ebsd_row_float": ebsd_row.astype(np.float32),
        "ebsd_col_float": ebsd_col.astype(np.float32),
        "ebsd_row": rr,
        "ebsd_col": cc,
        "ebsd_index": rr * ew + cc,
        "inside": inside,
    }


def compute_normals(
    height_um: np.ndarray,
    scan_size_um: float,
    smooth_sigma_px: float,
    plane_level_enabled: bool,
    row_derivative_sign: int,
    afm_to_sem_rotation: np.ndarray,
) -> dict[str, np.ndarray | dict[str, float] | float]:
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
    normals_afm = np.dstack(
        [-dz_dx, -float(row_derivative_sign) * dz_drow, np.ones_like(smooth, dtype=np.float32)]
    )
    normals_afm /= np.linalg.norm(normals_afm, axis=2, keepdims=True) + 1e-12
    xy = normals_afm[..., :2].reshape(-1, 2) @ np.asarray(afm_to_sem_rotation, dtype=np.float64).T
    normals_sample = np.column_stack([xy, normals_afm[..., 2].reshape(-1)]).reshape(normals_afm.shape)
    normals_sample /= np.linalg.norm(normals_sample, axis=2, keepdims=True) + 1e-12
    normals_sample[normals_sample[..., 2] < 0] *= -1.0
    slope_deg = np.degrees(np.arccos(np.clip(normals_sample[..., 2], -1.0, 1.0))).astype(np.float32)
    aspect_deg = np.degrees(np.arctan2(normals_sample[..., 1], normals_sample[..., 0])).astype(np.float32)
    return {
        "height_leveled_um": leveled,
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
    saturation = np.clip(tilt / max(tilt_ref_deg, 1e-6), 0.0, 1.0)
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


def ebsd_ipf_z_rgb(orientations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from export_h5_ipf_bse_maps import cubic_ipf_z_colors

    return cubic_ipf_z_colors(orientations.reshape(-1, 9), valid.reshape(-1), None)


def rgba(rgb: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    out[..., 3] = mask.astype(np.float32) * float(alpha)
    return out


def save_rgb(path: Path, rgb: np.ndarray, title: str | None = None, mask: np.ndarray | None = None, dpi: int = 220) -> None:
    image = np.clip(rgb, 0.0, 1.0).copy()
    if mask is not None:
        image[~mask] = 0.0
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi, constrained_layout=True)
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_scalar(path: Path, values: np.ndarray, title: str, cmap: str, label: str, dpi: int = 220) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi, constrained_layout=True)
    im = ax.imshow(values, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label=label, shrink=0.82)
    fig.savefig(path)
    plt.close(fig)


def save_normal_color_wheel(path: Path, tilt_ref_deg: float, dpi: int = 220) -> None:
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
    fig.savefig(path, transparent=True)
    plt.close(fig)


def draw_marker(ax: plt.Axes, xy: tuple[float, float], color: str = "black") -> None:
    ax.plot(xy[0], xy[1], "o", ms=10, mfc="none", mec=color, mew=2.0)
    ax.plot(xy[0], xy[1], "o", ms=4, mfc="red", mec="white", mew=0.8)


def save_selected_map(path: Path, image: np.ndarray, xy: tuple[int, int], title: str, kind: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi, constrained_layout=True)
    if kind == "rgb":
        ax.imshow(np.clip(image, 0.0, 1.0))
    else:
        ax.imshow(image, cmap=kind)
    draw_marker(ax, xy)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def warp_afm_rgba_to_sem(rgba_image: np.ndarray, matrix_small_to_sem: np.ndarray, sem_shape: tuple[int, int]) -> np.ndarray:
    ah, aw = rgba_image.shape[:2]
    sh, sw = sem_shape
    scale = np.array([[(sw - 1) / max(aw - 1, 1), 0.0, 0.0], [0.0, (sh - 1) / max(ah - 1, 1), 0.0], [0.0, 0.0, 1.0]])
    matrix_full = matrix_small_to_sem @ scale
    return cv2.warpPerspective(
        rgba_image.astype(np.float32),
        matrix_full,
        (sw, sh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def composite_over_gray(gray: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base = np.dstack([robust_rescale(gray)] * 3)
    alpha = overlay[..., 3:4]
    return np.clip(base * (1.0 - alpha) + overlay[..., :3] * alpha, 0.0, 1.0)


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
        ax.plot(
            [seg["col0"], seg["col1"]],
            [seg["row0"], seg["row1"]],
            color=color,
            lw=max(1.2, 0.35 * float(seg["bandwidth"])),
            alpha=0.86,
        )
        mid_col = 0.5 * (seg["col0"] + seg["col1"])
        mid_row = 0.5 * (seg["row0"] + seg["row1"])
        ax.text(mid_col, mid_row, str(int(seg["band_id"] + 1)), color="white", fontsize=7, ha="center", va="center")
    ax.set_title(
        f"Kikuchi + OHP bands\nidx={metadata['index']} row={metadata['row']} col={metadata['col']} "
        f"IQ={metadata['IQ']:.0f} CI={metadata['CI']:.3f}"
    )
    ax.axis("off")
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def save_sem_overlay(path: Path, sem_display: np.ndarray, overlay_rgba: np.ndarray, title: str, marker_xy: tuple[float, float], dpi: int) -> None:
    composite = composite_over_gray(sem_display, overlay_rgba)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=dpi, constrained_layout=True)
    ax.imshow(composite)
    draw_marker(ax, marker_xy)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_coupling_montage(
    path: Path,
    kikuchi_path: Path,
    height_rgb: np.ndarray,
    normal_rgb: np.ndarray,
    surface_rgb: np.ndarray,
    color_key_path: Path,
    selected_xy: tuple[int, int],
    dpi: int,
) -> None:
    from PIL import Image

    kikuchi = np.asarray(Image.open(kikuchi_path).convert("RGB"), dtype=np.float32) / 255.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=dpi, facecolor="black", constrained_layout=True)
    axes[0, 0].imshow(kikuchi)
    axes[0, 0].set_title("Kikuchi + OHP bands", color="white")
    axes[0, 1].imshow(height_rgb)
    draw_marker(axes[0, 1], selected_xy)
    axes[0, 1].set_title("AFM height reference", color="white")
    axes[0, 2].imshow(normal_rgb)
    draw_marker(axes[0, 2], selected_xy)
    axes[0, 2].set_title("AFM sample normal", color="white")
    axes[1, 0].imshow(surface_rgb)
    draw_marker(axes[1, 0], selected_xy)
    axes[1, 0].set_title("Crystal surface index", color="white")
    key = np.asarray(Image.open(color_key_path).convert("RGB"), dtype=np.float32) / 255.0
    axes[1, 1].imshow(key)
    axes[1, 1].set_title("Normal azimuth key", color="white")
    axes[1, 2].axis("off")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def save_3d_surface_png(path: Path, height_um: np.ndarray, rgb: np.ndarray, scan_size_um: float, stride: int, dpi: int) -> None:
    h = height_um[::stride, ::stride]
    c = rgb[::stride, ::stride]
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


def save_3d_surface_html(path: Path, height_um: np.ndarray, rgb: np.ndarray, scan_size_um: float, stride: int) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    z = height_um[::stride, ::stride].astype(np.float32)
    colors = np.clip(rgb[::stride, ::stride], 0.0, 1.0)
    rows, cols = z.shape
    yy, xx = np.indices((rows, cols))
    x = (xx * scan_size_um / max(cols - 1, 1)).reshape(-1)
    y = (yy * scan_size_um / max(rows - 1, 1)).reshape(-1)
    zz = z.reshape(-1)
    vertexcolor = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors.reshape(-1, 3)
    ]
    i = []
    j = []
    k = []
    for r in range(rows - 1):
        base = r * cols
        nxt = (r + 1) * cols
        for c in range(cols - 1):
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
    return True


def choose_selected_point(config: dict[str, Any], valid: np.ndarray, ci_afm: np.ndarray, iq_afm: np.ndarray) -> tuple[int, int]:
    x = int(config["selected_point"]["afm_x_px"])
    y = int(config["selected_point"]["afm_y_px"])
    h, w = valid.shape
    if 0 <= x < w and 0 <= y < h and valid[y, x]:
        return x, y
    if not config["selected_point"].get("fallback_to_quality_point_if_invalid", True):
        return min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    yy, xx = np.indices(valid.shape)
    central = valid & (xx > 0.15 * w) & (xx < 0.85 * w) & (yy > 0.15 * h) & (yy < 0.85 * h)
    score = robust_rescale(ci_afm) + robust_rescale(iq_afm)
    score[~central] = -np.inf
    index = int(np.nanargmax(score))
    row, col = np.unravel_index(index, valid.shape)
    return int(col), int(row)


def run(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    out_dir = Path(config["output_dir"])
    figures = out_dir / "figures"
    data_dir = out_dir / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["visualization"]["dpi"])

    height_um, afm_meta = read_afm_height(config["afm"])
    reg = read_json(Path(config["registration"]["report_path"]))
    matrix = np.asarray(reg["matrix_afm_to_sem"], dtype=np.float64)
    ebsd = read_ebsd(Path(config["ebsd"]["h5_path"]), config["ebsd"]["h5_group"])
    sem_display = orient_image(ebsd["sem_raw"], config["registration"]["sem_display_orientation"])
    sem_display = robust_rescale(sem_display)
    mapping = map_afm_to_ebsd(
        height_um.shape,
        sem_display.shape,
        (ebsd["nrows"], ebsd["ncols"]),
        matrix,
        config["registration"]["sem_display_orientation"],
    )

    rr = mapping["ebsd_row"]
    cc = mapping["ebsd_col"]
    valid = mapping["inside"] & ebsd["valid"][rr, cc]
    orientations_afm = ebsd["orientations"][mapping["ebsd_index"].reshape(-1)].reshape(*height_um.shape, 3, 3)
    ci_afm = ebsd["ci"][rr, cc]
    iq_afm = ebsd["iq"][rr, cc]
    phase_afm = ebsd["phase"][rr, cc]

    center_xy = ((sem_display.shape[1] - 1) / 2.0, (sem_display.shape[0] - 1) / 2.0)
    rotation = homography_center_rotation(matrix, center_xy)
    normal_data = compute_normals(
        height_um,
        float(afm_meta["scan_size_um"]),
        float(config["afm"]["normal_smooth_sigma_px"]),
        bool(config["afm"]["plane_level"]),
        int(config["normal_frame"]["image_row_derivative_sign"]),
        rotation,
    )
    normals_sample = normal_data["normals_sample"]
    normals_crystal = np.einsum("...ij,...j->...i", orientations_afm, normals_sample)
    normals_crystal /= np.linalg.norm(normals_crystal, axis=2, keepdims=True) + 1e-12
    folded = fold_cubic(normals_crystal)
    surface_rgb = facet_type_rgb(folded)
    surface_rgb[~valid] = 0.0
    normal_rgb = normal_direction_rgb(normals_sample, float(config["visualization"]["normal_tilt_ref_deg"]))
    height_rgb = plt.get_cmap("copper")(robust_rescale(normal_data["height_leveled_um"]))[..., :3].astype(np.float32)
    ipf_z_grid = ebsd_ipf_z_rgb(ebsd["orientations"], ebsd["valid"]).reshape(ebsd["nrows"], ebsd["ncols"], 3)
    ipf_afm = ipf_z_grid[rr, cc]
    ipf_afm[~valid] = 0.0
    hkl_idx, hkl_angle = nearest_hkl(folded)

    selected_x, selected_y = choose_selected_point(config, valid, ci_afm, iq_afm)
    selected_index = int(mapping["ebsd_index"][selected_y, selected_x])
    selected_row = int(mapping["ebsd_row"][selected_y, selected_x])
    selected_col = int(mapping["ebsd_col"][selected_y, selected_x])
    selected_sem = tuple(float(v) for v in mapping["sem_display_xy"][selected_y, selected_x])

    paths = {
        "sem_overlay_height": figures / "01_sem_with_aligned_afm_height_overlay.png",
        "sem_overlay_surface": figures / "02_sem_with_aligned_surface_index_overlay.png",
        "afm_height": figures / "03_afm_height_reference_selected.png",
        "afm_normal": figures / "04_afm_normal_reference_selected.png",
        "ebsd_ipf": figures / "05_ebsd_ipf_z_mapped_to_afm_reference.png",
        "surface_index": figures / "06_surface_index_afm_reference_selected.png",
        "surface_on_height": figures / "07_surface_index_over_afm_height_reference.png",
        "hkl_angle": figures / "08_nearest_hkl_angle_reference.png",
        "normal_key": figures / "09_normal_azimuth_color_key.png",
        "kikuchi": figures / "10_selected_kikuchi_ohp_bands.png",
        "montage": figures / "11_coupling_montage_like_reference.png",
        "surface_3d_png": figures / "12_surface_index_3d.png",
        "surface_3d_html": figures / "12_surface_index_3d_interactive.html",
        "npz": data_dir / "aligned_surface_index_afm_reference_data.npz",
        "metadata": data_dir / "selected_point_and_surface_index_metadata.json",
        "hkl_csv": data_dir / "nearest_hkl_summary.csv",
    }

    height_overlay = warp_afm_rgba_to_sem(rgba(height_rgb, np.ones_like(valid), config["visualization"]["overlay_alpha"]), matrix, sem_display.shape)
    surface_overlay = warp_afm_rgba_to_sem(rgba(surface_rgb, valid, config["visualization"]["surface_overlay_alpha"]), matrix, sem_display.shape)
    save_sem_overlay(paths["sem_overlay_height"], sem_display, height_overlay, "Aligned AFM height over SEM display", selected_sem, dpi)
    save_sem_overlay(paths["sem_overlay_surface"], sem_display, surface_overlay, "Aligned crystal surface index over SEM display", selected_sem, dpi)
    save_selected_map(paths["afm_height"], height_rgb, (selected_x, selected_y), "AFM height, AFM reference", "rgb", dpi)
    save_selected_map(paths["afm_normal"], normal_rgb, (selected_x, selected_y), "AFM local surface normal, AFM reference", "rgb", dpi)
    save_selected_map(paths["ebsd_ipf"], ipf_afm, (selected_x, selected_y), "EBSD IPF-Z mapped onto AFM reference", "rgb", dpi)
    save_selected_map(paths["surface_index"], surface_rgb, (selected_x, selected_y), "Continuous crystal surface index, AFM reference", "rgb", dpi)
    surface_on_height = np.clip(0.45 * height_rgb + 0.75 * surface_rgb, 0.0, 1.0)
    save_selected_map(paths["surface_on_height"], surface_on_height, (selected_x, selected_y), "Crystal surface index over AFM height", "rgb", dpi)
    save_scalar(paths["hkl_angle"], np.where(valid, hkl_angle, np.nan), "Angle to nearest low-index surface family", "magma", "deg", dpi)
    save_normal_color_wheel(paths["normal_key"], float(config["visualization"]["normal_tilt_ref_deg"]), dpi)

    pattern, up2_info = read_up2_pattern(Path(config["ebsd"]["up2_path"]), selected_index)
    segments = read_ohp_segments(Path(config["ebsd"]["h5_path"]), config["ebsd"]["h5_group"], selected_index, pattern.shape)
    selected_meta = {
        "afm_x_px": selected_x,
        "afm_y_px": selected_y,
        "sem_display_x_px": selected_sem[0],
        "sem_display_y_px": selected_sem[1],
        "ebsd_index": selected_index,
        "row": selected_row,
        "col": selected_col,
        "IQ": float(ebsd["iq"][selected_row, selected_col]),
        "CI": float(ebsd["ci"][selected_row, selected_col]),
        "Fit": float(ebsd["fit"][selected_row, selected_col]),
        "phase": int(ebsd["phase"][selected_row, selected_col]),
        "n_sample": [float(v) for v in normals_sample[selected_y, selected_x]],
        "n_crystal": [float(v) for v in normals_crystal[selected_y, selected_x]],
        "folded_cubic_direction": [float(v) for v in folded[selected_y, selected_x]],
        "nearest_hkl": HKL_LABELS[int(hkl_idx[selected_y, selected_x])],
        "nearest_hkl_angle_deg": float(hkl_angle[selected_y, selected_x]),
    }
    save_kikuchi_with_bands(paths["kikuchi"], pattern, segments, {**selected_meta, "index": selected_index}, dpi)
    save_coupling_montage(paths["montage"], paths["kikuchi"], height_rgb, normal_rgb, surface_rgb, paths["normal_key"], (selected_x, selected_y), dpi)
    save_3d_surface_png(
        paths["surface_3d_png"],
        normal_data["height_leveled_um"],
        surface_rgb,
        float(afm_meta["scan_size_um"]),
        int(config["visualization"]["plot_3d_stride"]),
        dpi,
    )
    html_ok = save_3d_surface_html(
        paths["surface_3d_html"],
        normal_data["height_leveled_um"],
        surface_rgb,
        float(afm_meta["scan_size_um"]),
        max(2, int(config["visualization"]["plot_3d_stride"])),
    )

    np.savez_compressed(
        paths["npz"],
        height_um=height_um,
        height_leveled_um=normal_data["height_leveled_um"],
        normals_sample=normals_sample,
        normals_crystal=normals_crystal.astype(np.float32),
        folded_cubic_direction=folded,
        surface_index_rgb=surface_rgb,
        normal_rgb=normal_rgb,
        ebsd_ipf_z_on_afm=ipf_afm,
        nearest_hkl_index=hkl_idx,
        nearest_hkl_angle_deg=hkl_angle,
        valid=valid,
        ebsd_row=mapping["ebsd_row"],
        ebsd_col=mapping["ebsd_col"],
        phase=phase_afm,
        ci=ci_afm,
        iq=iq_afm,
    )

    hkl_rows = []
    for idx, label in enumerate(HKL_LABELS):
        mask = valid & (hkl_idx == idx)
        hkl_rows.append(
            {
                "nearest_hkl": label,
                "pixel_count": int(np.count_nonzero(mask)),
                "fraction_of_valid": float(np.count_nonzero(mask) / max(np.count_nonzero(valid), 1)),
                "median_angle_deg": float(np.nanmedian(hkl_angle[mask])) if np.any(mask) else float("nan"),
            }
        )
    with paths["hkl_csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hkl_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hkl_rows)

    report = {
        "status": "completed",
        "method": "Use existing AFM-SEM/EBSD alignment; keep AFM display grid as reference; combine AFM local normal with nearest EBSD orientation.",
        "afm": afm_meta,
        "ebsd": {
            "h5_path": config["ebsd"]["h5_path"],
            "h5_group": config["ebsd"]["h5_group"],
            "up2_path": config["ebsd"]["up2_path"],
            "shape_rows_cols": [ebsd["nrows"], ebsd["ncols"]],
            "step_x_um": ebsd["step_x_um"],
            "step_y_um": ebsd["step_y_um"],
            "phase_metadata": ebsd["phase_metadata"],
            "orientation_convention_used": config["ebsd"]["orientation_convention"],
        },
        "registration": {
            "report_path": config["registration"]["report_path"],
            "matrix_direction": reg.get("matrix_direction", config["registration"]["matrix_direction"]),
            "sem_display_orientation": config["registration"]["sem_display_orientation"],
            "matrix_afm_resized_to_sem_display": matrix.tolist(),
            "control_point_metrics": reg.get("control_point_metrics", {}),
            "in_plane_rotation_used_for_normals": rotation.tolist(),
        },
        "normal_frame": config["normal_frame"],
        "selected_point": selected_meta,
        "up2": up2_info.__dict__,
        "valid_fraction": float(np.mean(valid)),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "interactive_3d_written": bool(html_ok),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(paths["metadata"], report)
    print(json.dumps({"status": "completed", "output_dir": str(out_dir), "selected_point": selected_meta}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFM-reference surface-index visualization from aligned AFM and EBSD data.")
    parser.add_argument("--config", type=Path, default=Path("configs/afm_ebsd_aligned_surface_index_pt_highres60.json"))
    return parser.parse_args()


def main() -> None:
    run(parse_args().config)


if __name__ == "__main__":
    main()
