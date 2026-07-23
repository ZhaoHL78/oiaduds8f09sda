from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from igor2.binarywave import load as load_ibw
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage, optimize
from skimage import exposure, filters, morphology, transform


@dataclass
class RegistrationData:
    afm_height_um: np.ndarray
    afm_display_height_um: np.ndarray
    afm_display_gray: np.ndarray
    sem_raw: np.ndarray
    sem_display: np.ndarray
    ipf_display: np.ndarray | None
    afm_meta: dict[str, Any]
    config: dict[str, Any]


@dataclass
class FeatureSet:
    gradient: np.ndarray
    edge_response: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    distance: np.ndarray


@dataclass
class ModelResult:
    name: str
    dof: int
    matrix: np.ndarray
    inverse: np.ndarray
    inliers: np.ndarray
    residuals_px: np.ndarray
    train_metrics: dict[str, float]
    holdout_metrics: dict[str, float] | None


def parse_note(note: bytes | str) -> dict[str, str]:
    text = note.decode("utf-8", "ignore") if isinstance(note, bytes) else str(note)
    output: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        output[key.strip()] = value.strip()
    return output


def read_afm_channels(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = load_ibw(str(path))["wave"]
    data = np.asarray(payload["wData"], dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]

    labels_raw = payload.get("labels", [[], [], [], []])
    channel_labels = labels_raw[2] if len(labels_raw) > 2 else []
    labels: list[str] = []
    for raw in channel_labels:
        label = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        label = label.strip("\x00").strip()
        if label:
            labels.append(label)
    if len(labels) < data.shape[2]:
        labels.extend(f"channel_{index}" for index in range(len(labels), data.shape[2]))

    channels = {label: data[:, :, index] for index, label in enumerate(labels[: data.shape[2]])}
    note = parse_note(payload.get("note", b""))
    scan_size_um = float(note.get("ScanSize", "nan")) * 1e6 if "ScanSize" in note else float("nan")
    metadata = {
        "shape": list(data.shape[:2]),
        "channel_labels": labels[: data.shape[2]],
        "scan_size_um": scan_size_um,
        "scan_angle_deg": float(note.get("ScanAngle", "nan")) if "ScanAngle" in note else float("nan"),
        "imaging_mode": note.get("ImagingMode", ""),
        "points_lines": note.get("PointsLines", ""),
    }
    return channels, metadata


def robust_rescale(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def plane_level(height_um: np.ndarray) -> np.ndarray:
    y, x = np.indices(height_um.shape, dtype=np.float64)
    values = height_um.astype(np.float64)
    finite = np.isfinite(values)
    design = np.column_stack([x[finite], y[finite], np.ones(np.count_nonzero(finite))])
    coeff, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    return (values - plane).astype(np.float32)


def affine_rotation_2d(affine_afm_to_sem: np.ndarray) -> np.ndarray:
    linear = affine_afm_to_sem[:2, :2].astype(np.float64)
    u, _s, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation.astype(np.float32)


def height_to_normals(
    height_um: np.ndarray,
    scan_size_um: float,
    affine_afm_to_sem: np.ndarray,
    smooth_sigma_px: float,
    level: bool,
) -> dict[str, np.ndarray | float]:
    height = plane_level(height_um) if level else height_um.astype(np.float32)
    smooth = ndimage.gaussian_filter(height.astype(np.float32), sigma=smooth_sigma_px) if smooth_sigma_px > 0 else height

    pitch_x_um = scan_size_um / max(height.shape[1] - 1, 1)
    pitch_y_um = scan_size_um / max(height.shape[0] - 1, 1)
    scharr_col = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    scharr_row = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    dz_dcol_um = scharr_col / max(pitch_x_um, 1e-12)
    dz_drow_um = scharr_row / max(pitch_y_um, 1e-12)

    normals_afm = np.dstack(
        [
            -dz_dcol_um.astype(np.float32),
            -dz_drow_um.astype(np.float32),
            np.ones_like(smooth, dtype=np.float32),
        ]
    )
    normals_afm /= np.linalg.norm(normals_afm, axis=2, keepdims=True) + 1e-12

    rotation = affine_rotation_2d(affine_afm_to_sem)
    normals_xy = normals_afm[..., :2].reshape(-1, 2) @ rotation.T
    normals = np.column_stack([normals_xy, normals_afm[..., 2].reshape(-1)]).reshape(normals_afm.shape)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-12
    normals[normals[..., 2] < 0] *= -1.0

    tilt_deg = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0))).astype(np.float32)
    azimuth_deg = np.degrees(np.arctan2(normals[..., 1], normals[..., 0])).astype(np.float32)
    return {
        "height_um": height.astype(np.float32),
        "height_smooth_um": smooth.astype(np.float32),
        "normals_afm": normals_afm.astype(np.float32),
        "normals_sample": normals.astype(np.float32),
        "scharr_dz_dcol": dz_dcol_um.astype(np.float32),
        "scharr_dz_drow": dz_drow_um.astype(np.float32),
        "tilt_deg": tilt_deg,
        "azimuth_deg": azimuth_deg,
        "pitch_x_um": float(pitch_x_um),
        "pitch_y_um": float(pitch_y_um),
        "afm_to_sem_rotation_2d": rotation,
    }


def normal_direction_rgb(normals: np.ndarray, tilt_ref_deg: float) -> np.ndarray:
    azimuth = (np.arctan2(normals[..., 1], normals[..., 0]) + np.pi) / (2.0 * np.pi)
    tilt = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    saturation = np.clip(tilt / max(tilt_ref_deg, 1e-6), 0.0, 1.0)
    value = np.ones_like(saturation) * 0.96
    return hsv_to_rgb(np.dstack([azimuth, saturation, value])).astype(np.float32)


def normal_legend_rgb(size: int = 320) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.sqrt(xx**2 + yy**2)
    hue = (np.arctan2(yy, xx) + np.pi) / (2.0 * np.pi)
    saturation = np.clip(radius, 0.0, 1.0)
    rgb = hsv_to_rgb(np.dstack([hue, saturation, np.full_like(hue, 0.96)])).astype(np.float32)
    rgba_image = np.zeros((size, size, 4), dtype=np.float32)
    rgba_image[..., :3] = rgb
    rgba_image[..., 3] = (radius <= 1.0).astype(np.float32)
    return rgba_image


def save_normalmap_with_legend(path: Path, normal_rgb: np.ndarray, title: str, tilt_ref_deg: float) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.8),
        dpi=220,
        gridspec_kw={"width_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    axes[0].imshow(normal_rgb)
    axes[0].set_title(title)
    axes[0].axis("off")
    axes[1].imshow(normal_legend_rgb(), extent=(-1, 1, -1, 1))
    axes[1].set_title("Normal color key")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].text(0.94, 0.50, "0 deg", transform=axes[1].transAxes, ha="right", va="center", fontsize=8)
    axes[1].text(0.06, 0.50, "180 deg", transform=axes[1].transAxes, ha="left", va="center", fontsize=8)
    axes[1].text(0.50, 0.94, "+90 deg", transform=axes[1].transAxes, ha="center", va="top", fontsize=8)
    axes[1].text(0.50, 0.06, "-90 deg", transform=axes[1].transAxes, ha="center", va="bottom", fontsize=8)
    axes[1].text(
        0.50,
        -0.16,
        f"hue=azimuth, center=0 deg tilt, rim={tilt_ref_deg:g} deg",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        return image.T
    if orientation == "transpose_flipud":
        return np.flipud(image.T)
    raise ValueError(f"Unknown orientation: {orientation}")


def read_sem(config: dict[str, Any]) -> np.ndarray:
    sem_cfg = config["sem"]
    source = sem_cfg.get("source", "image")
    if source == "edax_h5":
        with h5py.File(sem_cfg["h5_path"], "r") as h5:
            sem = np.asarray(h5[sem_cfg["h5_group"]]["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32)
        return robust_rescale(sem)
    path = Path(sem_cfg["path"])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return robust_rescale(image.astype(np.float32))


def read_optional_ipf(config: dict[str, Any], sem_shape: tuple[int, int]) -> np.ndarray | None:
    ipf_cfg = config.get("ipf_reference", {})
    path = Path(ipf_cfg.get("path", ""))
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.resize(image, (sem_shape[1], sem_shape[0]), interpolation=cv2.INTER_LINEAR)


def load_data(config: dict[str, Any]) -> RegistrationData:
    afm_cfg = config["afm"]
    channels, afm_meta = read_afm_channels(Path(afm_cfg["path"]))
    channel = afm_cfg["height_channel"]
    if channel not in channels:
        raise KeyError(f"AFM channel {channel!r} missing. Available: {list(channels)}")
    height_um = channels[channel].astype(np.float32) * float(afm_cfg.get("height_unit_scale_to_um", 1.0))
    afm_display_height = orient_image(height_um, afm_cfg.get("display_orientation", "raw"))
    sem_raw = read_sem(config)
    sem_display = orient_image(sem_raw, config["sem"].get("display_orientation", "raw"))
    ipf = read_optional_ipf(config, sem_display.shape)
    afm_gray = robust_rescale(afm_display_height, 1.0, 99.0)
    afm_gray = cv2.resize(afm_gray, (sem_display.shape[1], sem_display.shape[0]), interpolation=cv2.INTER_AREA)
    return RegistrationData(
        afm_height_um=height_um,
        afm_display_height_um=afm_display_height,
        afm_display_gray=afm_gray.astype(np.float32),
        sem_raw=sem_raw,
        sem_display=sem_display,
        ipf_display=ipf,
        afm_meta=afm_meta,
        config=config,
    )


def image_stats(name: str, image: np.ndarray) -> dict[str, Any]:
    values = image[np.isfinite(image)]
    return {
        "name": name,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "mean": float(values.mean()) if values.size else float("nan"),
        "p1": float(np.percentile(values, 1)) if values.size else float("nan"),
        "p99": float(np.percentile(values, 99)) if values.size else float("nan"),
    }


def save_gray(path: Path, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=180, constrained_layout=True)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_binary(path: Path, image: np.ndarray, title: str) -> None:
    save_gray(path, image.astype(np.float32), title, "gray")


def local_variance(image: np.ndarray, radius: int = 5) -> np.ndarray:
    image = image.astype(np.float32)
    mean = ndimage.uniform_filter(image, size=2 * radius + 1)
    mean2 = ndimage.uniform_filter(image * image, size=2 * radius + 1)
    return np.maximum(mean2 - mean * mean, 0.0).astype(np.float32)


def clean_edge_mask(response: np.ndarray, percentile: float, remove_small_px: int, closing_radius: int) -> np.ndarray:
    threshold = np.percentile(response[np.isfinite(response)], percentile)
    mask = response >= threshold
    if closing_radius > 0:
        footprint = morphology.disk(closing_radius)
        mask = morphology.binary_closing(mask, footprint)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=remove_small_px)
    mask = morphology.remove_small_holes(mask, area_threshold=remove_small_px)
    return mask.astype(bool)


def extract_afm_features(image: np.ndarray, config: dict[str, Any]) -> FeatureSet:
    params = config["features"]
    smooth = filters.gaussian(robust_rescale(image, 1.0, 99.0), sigma=params["afm_smooth_sigma_px"], preserve_range=True)
    gx = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gy = cv2.Scharr(smooth.astype(np.float32), cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    gradient = robust_rescale(np.sqrt(gx * gx + gy * gy), 1.0, 99.7)
    variance = robust_rescale(local_variance(smooth, radius=7), 2.0, 99.5)
    valley = robust_rescale(1.0 - smooth, 80.0, 99.5)
    response = robust_rescale(np.maximum.reduce([gradient, 0.65 * variance, 0.75 * valley]), 1.0, 99.7)
    mask = clean_edge_mask(
        response,
        params["afm_edge_percentile"],
        int(params["remove_small_objects_px"]),
        int(params["closing_radius_px"]),
    )
    skeleton = morphology.skeletonize(mask)
    distance = ndimage.distance_transform_edt(~skeleton)
    return FeatureSet(gradient, response, mask, skeleton, distance.astype(np.float32))


def extract_sem_features(image: np.ndarray, config: dict[str, Any]) -> FeatureSet:
    params = config["features"]
    norm = robust_rescale(image, 0.5, 99.5)
    background = filters.gaussian(norm, sigma=18.0, preserve_range=True)
    corrected = robust_rescale(norm - background, 1.0, 99.0)
    smooth = filters.gaussian(corrected, sigma=params["sem_smooth_sigma_px"], preserve_range=True)
    equalized = exposure.equalize_adapthist(smooth, clip_limit=0.015).astype(np.float32)
    gx = cv2.Scharr(equalized, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gy = cv2.Scharr(equalized, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    gradient = robust_rescale(np.sqrt(gx * gx + gy * gy), 1.0, 99.7)
    # Repeated intragranular stripes are sharp but narrow; a broad low-pass
    # response keeps grain boundaries and suppresses much of that texture.
    broad = filters.gaussian(gradient, sigma=1.6, preserve_range=True)
    response = robust_rescale(broad, 1.0, 99.7)
    mask = clean_edge_mask(
        response,
        params["sem_edge_percentile"],
        int(params["remove_small_objects_px"]),
        int(params["closing_radius_px"]),
    )
    skeleton = morphology.skeletonize(mask)
    distance = ndimage.distance_transform_edt(~skeleton)
    return FeatureSet(gradient, response, mask, skeleton, distance.astype(np.float32))


def save_feature_set(output_dir: Path, prefix: str, features: FeatureSet) -> None:
    save_gray(output_dir / f"{prefix}_gradient.png", features.gradient, f"{prefix} gradient")
    save_gray(output_dir / f"{prefix}_edge_response.png", features.edge_response, f"{prefix} edge response")
    save_binary(output_dir / f"{prefix}_boundary_mask.png", features.mask, f"{prefix} cleaned boundary mask")
    save_binary(output_dir / f"{prefix}_skeleton.png", features.skeleton, f"{prefix} boundary skeleton")
    save_gray(output_dir / f"{prefix}_distance_transform.png", robust_rescale(features.distance, 0, 95), f"{prefix} distance transform")


def save_orientation_sheet(output_dir: Path, data: RegistrationData) -> None:
    sem_variants = ["raw", "flipud", "fliplr", "rot90", "rot180", "rot270"]
    afm_variants = ["raw", "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose"]
    sem_h, sem_w = data.sem_raw.shape
    fig, axes = plt.subplots(len(afm_variants), len(sem_variants), figsize=(3.0 * len(sem_variants), 2.4 * len(afm_variants)), dpi=150, constrained_layout=True)
    for row, afm_name in enumerate(afm_variants):
        afm = orient_image(data.afm_height_um, afm_name)
        afm = cv2.resize(robust_rescale(afm, 1, 99), (sem_w, sem_h), interpolation=cv2.INTER_AREA)
        for col, sem_name in enumerate(sem_variants):
            sem = orient_image(data.sem_raw, sem_name)
            if sem.shape != (sem_h, sem_w):
                sem = cv2.resize(sem, (sem_w, sem_h), interpolation=cv2.INTER_AREA)
            axes[row, col].imshow(sem, cmap="gray", vmin=0, vmax=1)
            axes[row, col].imshow(afm, cmap="magma", alpha=0.42)
            axes[row, col].set_title(f"AFM {afm_name}\nSEM {sem_name}", fontsize=7)
            axes[row, col].axis("off")
    fig.suptitle("Discrete AFM/SEM orientation check; use control points to choose the true geometry", fontsize=12)
    fig.savefig(output_dir / "orientation_discrete_candidates.png", bbox_inches="tight")
    plt.close(fig)


def control_point_template(path: Path, data: RegistrationData) -> None:
    payload = {
        "coordinate_frame": (
            "AFM points are in AFM_display pixels after config afm.display_orientation and resize to SEM_display size. "
            "SEM points are in SEM_display pixels after config sem.display_orientation."
        ),
        "afm_display_orientation": data.config["afm"].get("display_orientation", "raw"),
        "sem_display_orientation": data.config["sem"].get("display_orientation", "raw"),
        "pairs": [],
        "instructions": [
            "Pick 8-15 corresponding points on grain-boundary intersections, boundary bends, boundary-image-border intersections, or isolated defects.",
            "Avoid dense repeated intragranular stripes.",
            "Run: python afm_sem_geometric_registration.py --config <config> --pick-points",
            "After saving points, run the same command without --pick-points for global fit and distance-field refinement."
        ],
    }
    if not path.exists():
        write_json(path, payload)


def load_control_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    payload = read_json(path)
    pairs = payload.get("pairs", [])
    afm = []
    sem = []
    rows = []
    for index, pair in enumerate(pairs, start=1):
        if pair.get("disabled", False):
            continue
        a = pair.get("afm")
        s = pair.get("sem")
        if a is None or s is None:
            continue
        afm.append([float(a[0]), float(a[1])])
        sem.append([float(s[0]), float(s[1])])
        rows.append({"id": pair.get("id", index), "afm_x": a[0], "afm_y": a[1], "sem_x": s[0], "sem_y": s[1], "label": pair.get("label", "")})
    return np.asarray(afm, dtype=np.float64), np.asarray(sem, dtype=np.float64), rows


def save_control_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def launch_control_point_picker(data: RegistrationData, cp_path: Path) -> None:
    sem = data.sem_display
    afm = data.afm_display_gray
    payload = read_json(cp_path) if cp_path.exists() else {"pairs": []}
    pairs = payload.setdefault("pairs", [])
    pending: dict[str, list[float] | None] = {"afm": None, "sem": None}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=120)
    axes[0].imshow(afm, cmap="afmhot")
    axes[0].set_title("AFM display: click AFM point")
    axes[1].imshow(sem, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("SEM display: click corresponding SEM point")
    for ax in axes:
        ax.axis("off")

    def redraw() -> None:
        axes[0].cla()
        axes[1].cla()
        axes[0].imshow(afm, cmap="afmhot")
        axes[1].imshow(sem, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("AFM display coordinates")
        axes[1].set_title("SEM display coordinates")
        for idx, pair in enumerate(pairs, start=1):
            if pair.get("disabled", False):
                continue
            axp = pair["afm"]
            sxp = pair["sem"]
            axes[0].scatter([axp[0]], [axp[1]], s=35, facecolors="none", edgecolors="cyan", linewidths=1.5)
            axes[1].scatter([sxp[0]], [sxp[1]], s=35, facecolors="none", edgecolors="cyan", linewidths=1.5)
            axes[0].text(axp[0] + 4, axp[1] + 4, str(idx), color="yellow", fontsize=8)
            axes[1].text(sxp[0] + 4, sxp[1] + 4, str(idx), color="yellow", fontsize=8)
        if pending["afm"] is not None:
            axes[0].scatter([pending["afm"][0]], [pending["afm"][1]], s=40, c="lime")
        if pending["sem"] is not None:
            axes[1].scatter([pending["sem"][0]], [pending["sem"][1]], s=40, c="lime")
        axes[0].set_title("AFM: click point | keys: s save, u undo, c clear pending, q quit")
        axes[1].set_title("SEM: click corresponding point")
        for ax in axes:
            ax.axis("off")
        fig.canvas.draw_idle()

    def save() -> None:
        payload["pairs"] = pairs
        write_json(cp_path, payload)
        print(f"Saved {len(pairs)} control-point pairs to {cp_path}")

    def on_click(event) -> None:
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        key = "afm" if event.inaxes is axes[0] else "sem"
        pending[key] = [float(event.xdata), float(event.ydata)]
        if pending["afm"] is not None and pending["sem"] is not None:
            pairs.append({"id": len(pairs) + 1, "afm": pending["afm"], "sem": pending["sem"], "label": ""})
            pending["afm"] = None
            pending["sem"] = None
        redraw()

    def on_key(event) -> None:
        if event.key == "u" and pairs:
            pairs.pop()
            redraw()
        elif event.key == "c":
            pending["afm"] = None
            pending["sem"] = None
            redraw()
        elif event.key == "s":
            save()
        elif event.key == "q":
            save()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


def as_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points), dtype=np.float64)])


def transform_points(matrix: np.ndarray, points: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in {"similarity", "affine", "refined_affine"}:
        return (matrix @ as_homogeneous(points).T).T
    hom = (matrix @ as_homogeneous(points).T).T
    return hom[:, :2] / np.maximum(hom[:, 2:3], 1e-12)


def invert_matrix(matrix: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in {"similarity", "affine", "refined_affine"}:
        return cv2.invertAffineTransform(matrix.astype(np.float64))
    return np.linalg.inv(matrix)


def residual_metrics(residuals: np.ndarray, pixel_um: float | None = None) -> dict[str, float]:
    if residuals.size == 0:
        return {"rmse_px": float("nan"), "median_px": float("nan"), "p95_px": float("nan")}
    rmse = float(np.sqrt(np.mean(residuals**2)))
    metrics = {
        "rmse_px": rmse,
        "median_px": float(np.median(residuals)),
        "p95_px": float(np.percentile(residuals, 95)),
        "max_px": float(np.max(residuals)),
    }
    if pixel_um is not None:
        metrics.update({key.replace("_px", "_um"): value * pixel_um for key, value in metrics.items() if key.endswith("_px")})
    return metrics


def fit_single_model(name: str, afm: np.ndarray, sem: np.ndarray, threshold: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if name == "similarity":
        matrix, inliers = cv2.estimateAffinePartial2D(afm, sem, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=20000, confidence=0.995)
    elif name == "affine":
        matrix, inliers = cv2.estimateAffine2D(afm, sem, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=20000, confidence=0.995)
    elif name == "homography":
        matrix, inliers = cv2.findHomography(afm, sem, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=20000, confidence=0.995)
    else:
        raise ValueError(name)
    if matrix is None or inliers is None:
        return None, None
    return matrix.astype(np.float64), inliers.reshape(-1).astype(bool)


def fit_models(afm: np.ndarray, sem: np.ndarray, config: dict[str, Any], output_dir: Path) -> list[ModelResult]:
    n = len(afm)
    min_points = {"similarity": 2, "affine": 3, "homography": 4}
    rng = np.random.default_rng(int(config.get("random_seed", 0)))
    holdout_count = int(round(n * float(config["control_points"].get("holdout_fraction", 0.25)))) if n >= 8 else 0
    holdout_count = min(max(holdout_count, 0), max(0, n - 4))
    holdout_indices = np.sort(rng.choice(n, holdout_count, replace=False)) if holdout_count else np.zeros(0, dtype=int)
    train_mask = np.ones(n, dtype=bool)
    train_mask[holdout_indices] = False
    threshold = float(config["model_selection"]["ransac_reproj_threshold_px"])
    rows = []
    results: list[ModelResult] = []
    for name, dof in [("similarity", 4), ("affine", 6), ("homography", 8)]:
        if int(train_mask.sum()) < min_points[name]:
            continue
        matrix, inliers_train = fit_single_model(name, afm[train_mask], sem[train_mask], threshold)
        if matrix is None or inliers_train is None:
            continue
        pred_all = transform_points(matrix, afm, name)
        residuals = np.linalg.norm(pred_all - sem, axis=1)
        inliers_all = residuals <= threshold
        train_metrics = residual_metrics(residuals[train_mask])
        hold_metrics = residual_metrics(residuals[holdout_indices]) if holdout_indices.size else None
        inverse = invert_matrix(matrix, name)
        result = ModelResult(name, dof, matrix, inverse, inliers_all, residuals, train_metrics, hold_metrics)
        results.append(result)
        rows.append(
            {
                "model": name,
                "dof": dof,
                "inliers": int(inliers_all.sum()),
                "points": n,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **({f"holdout_{k}": v for k, v in hold_metrics.items()} if hold_metrics else {}),
            }
        )
    write_csv(output_dir / "global_model_comparison.csv", rows)
    return results


def choose_model(results: list[ModelResult], config: dict[str, Any]) -> ModelResult:
    if not results:
        raise RuntimeError("No global model could be fitted")
    best_error = min((r.holdout_metrics or r.train_metrics)["rmse_px"] for r in results)
    tolerance = 1.0 + float(config["model_selection"]["prefer_low_dof_within_fraction"])
    max_median = float(config["model_selection"]["max_acceptable_median_error_px"])
    for result in sorted(results, key=lambda r: r.dof):
        metrics = result.holdout_metrics or result.train_metrics
        if metrics["rmse_px"] <= best_error * tolerance and result.train_metrics["median_px"] <= max_median:
            return result
    return min(results, key=lambda r: (r.holdout_metrics or r.train_metrics)["rmse_px"])


def matrix_to33(matrix: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in {"similarity", "affine", "refined_affine"}:
        out = np.eye(3, dtype=np.float64)
        out[:2, :] = matrix
        return out
    return matrix


def matrix_from33(matrix: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in {"similarity", "affine", "refined_affine"}:
        return matrix[:2, :]
    return matrix


def scale_transform(matrix: np.ndarray, model_name: str, scale: float) -> np.ndarray:
    s = np.diag([scale, scale, 1.0])
    h = matrix_to33(matrix, model_name)
    return matrix_from33(s @ h @ np.linalg.inv(s), model_name)


def edge_points(skeleton: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.nonzero(skeleton)
    points = np.column_stack([cols, rows]).astype(np.float64)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def affine_refine_distance(initial: ModelResult, afm_feat: FeatureSet, sem_feat: FeatureSet, config: dict[str, Any], output_dir: Path) -> ModelResult:
    params = config["distance_refinement"]
    if initial.name == "homography":
        return initial
    rng = np.random.default_rng(int(config.get("random_seed", 0)))
    matrix = initial.matrix.astype(np.float64).copy()
    max_points = int(params["max_edge_points_per_direction"])
    loss_rows = []
    for scale in params["pyramid_scales"]:
        afm_mask = transform.resize(afm_feat.skeleton.astype(np.float32), (int(round(afm_feat.skeleton.shape[0] * scale)), int(round(afm_feat.skeleton.shape[1] * scale))), order=0, preserve_range=True, anti_aliasing=False) > 0.5
        sem_mask = transform.resize(sem_feat.skeleton.astype(np.float32), (int(round(sem_feat.skeleton.shape[0] * scale)), int(round(sem_feat.skeleton.shape[1] * scale))), order=0, preserve_range=True, anti_aliasing=False) > 0.5
        sem_dt = ndimage.distance_transform_edt(~sem_mask).astype(np.float32)
        afm_dt = ndimage.distance_transform_edt(~afm_mask).astype(np.float32)
        afm_pts = edge_points(afm_mask, max_points, rng)
        sem_pts = edge_points(sem_mask, max_points, rng)
        matrix_scaled = scale_transform(matrix, "affine", float(scale))

        def residual_vector(p: np.ndarray) -> np.ndarray:
            m = p.reshape(2, 3)
            inv = cv2.invertAffineTransform(m)
            aw = transform_points(m, afm_pts, "affine")
            sw = transform_points(inv, sem_pts, "affine")
            # Keep a fixed residual length for least_squares. Points outside the
            # image receive a large constant penalty instead of being dropped.
            da = ndimage.map_coordinates(sem_dt, [aw[:, 1], aw[:, 0]], order=1, mode="constant", cval=max(sem_dt.shape))
            ds = ndimage.map_coordinates(afm_dt, [sw[:, 1], sw[:, 0]], order=1, mode="constant", cval=max(afm_dt.shape))
            return np.concatenate([da, ds]).astype(np.float64)

        before = residual_vector(matrix_scaled.reshape(-1))
        opt = optimize.least_squares(
            residual_vector,
            matrix_scaled.reshape(-1),
            loss="huber",
            f_scale=float(params["huber_f_scale_px"]) * float(scale),
            max_nfev=int(params["max_nfev"]),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        after = residual_vector(opt.x)
        matrix = scale_transform(opt.x.reshape(2, 3), "affine", 1.0 / float(scale))
        loss_rows.append(
            {
                "scale": scale,
                "before_median_px": float(np.median(before) / scale),
                "after_median_px": float(np.median(after) / scale),
                "before_mean_px": float(np.mean(before) / scale),
                "after_mean_px": float(np.mean(after) / scale),
                "nfev": int(opt.nfev),
                "success": bool(opt.success),
            }
        )
    write_csv(output_dir / "distance_refinement_loss.csv", loss_rows)
    pred = transform_points(matrix, initial.residuals_px.reshape(-1, 1) * 0 + np.zeros((len(initial.residuals_px), 2)), "affine")
    # Control-point residuals are recomputed by caller for reporting; this field
    # is filled with NaNs here to keep matrix/inverse in one result object.
    residuals = np.full_like(initial.residuals_px, np.nan, dtype=np.float64)
    return ModelResult("refined_affine", 6, matrix, cv2.invertAffineTransform(matrix), initial.inliers, residuals, {}, None)


def warp_image(image: np.ndarray, matrix: np.ndarray, model_name: str, output_shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape
    if model_name in {"similarity", "affine", "refined_affine"}:
        warped = cv2.warpAffine(image.astype(np.float32), matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones(image.shape[:2], dtype=np.float32), matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = cv2.warpPerspective(image.astype(np.float32), matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.float32), matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32)


def boundary_metrics(afm_skel_warped: np.ndarray, sem_skel: np.ndarray, tolerance: float) -> dict[str, float]:
    afm_bool = afm_skel_warped > 0.5
    sem_bool = sem_skel > 0
    sem_dt = ndimage.distance_transform_edt(~sem_bool)
    afm_dt = ndimage.distance_transform_edt(~afm_bool)
    precision = float(np.mean(sem_dt[afm_bool] <= tolerance)) if afm_bool.any() else float("nan")
    recall = float(np.mean(afm_dt[sem_bool] <= tolerance)) if sem_bool.any() else float("nan")
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12)) if np.isfinite(precision) and np.isfinite(recall) else float("nan")
    sym = float(0.5 * ((np.mean(sem_dt[afm_bool]) if afm_bool.any() else np.nan) + (np.mean(afm_dt[sem_bool]) if sem_bool.any() else np.nan)))
    return {"precision": precision, "recall": recall, "f1": f1, "symmetric_boundary_distance_px": sym}


def save_registration_visuals(
    output_dir: Path,
    data: RegistrationData,
    afm_feat: FeatureSet,
    sem_feat: FeatureSet,
    model: ModelResult,
    afm_warped: np.ndarray,
    common_mask: np.ndarray,
    afm_skel_warped: np.ndarray,
    afm_points: np.ndarray,
    sem_points: np.ndarray,
) -> None:
    sem = data.sem_display
    alpha = float(data.config["visualization"].get("overlay_alpha", 0.5))
    block = int(data.config["visualization"].get("checker_block_px", 32))
    rows, cols = np.indices(sem.shape)
    checker = ((rows // block + cols // block) % 2).astype(bool)
    chess = np.dstack([sem] * 3)
    afm_rgb = plt.get_cmap("magma")(robust_rescale(afm_warped, 1, 99))[..., :3]
    chess[checker & (common_mask > 0.5)] = afm_rgb[checker & (common_mask > 0.5)]
    plt.imsave(output_dir / "final_checkerboard.png", np.clip(chess, 0, 1))
    overlay = np.clip((1 - alpha) * np.dstack([sem] * 3) + alpha * afm_rgb, 0, 1)
    overlay[common_mask < 0.5] = np.dstack([sem] * 3)[common_mask < 0.5]
    plt.imsave(output_dir / "final_alpha_overlay.png", overlay)
    boundary = np.dstack([sem, sem, sem])
    boundary[sem_feat.skeleton > 0] = [1.0, 0.0, 0.0]
    boundary[afm_skel_warped > 0.5] = [0.0, 1.0, 1.0]
    plt.imsave(output_dir / "final_boundary_two_color_overlay.png", np.clip(boundary, 0, 1))
    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=180, constrained_layout=True)
    ax.imshow(sem, cmap="gray", vmin=0, vmax=1)
    pred = transform_points(model.matrix, afm_points, model.name)
    ax.scatter(sem_points[:, 0], sem_points[:, 1], s=40, facecolors="none", edgecolors="lime", label="SEM CP")
    ax.scatter(pred[:, 0], pred[:, 1], s=20, c="cyan", label="AFM CP warped")
    for s, p in zip(sem_points, pred):
        ax.plot([s[0], p[0]], [s[1], p[1]], color="yellow", linewidth=0.9)
    ax.set_title("Control-point residual vectors")
    ax.legend(loc="lower right")
    ax.axis("off")
    fig.savefig(output_dir / "control_point_residual_vectors.png", bbox_inches="tight")
    plt.close(fig)
    # Zoom panels: center and four quadrants.
    h, w = sem.shape
    crops = [
        ("center", w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        ("top_left", 0, 0, w // 2, h // 2),
        ("top_right", w // 2, 0, w, h // 2),
        ("bottom_left", 0, h // 2, w // 2, h),
        ("bottom_right", w // 2, h // 2, w, h),
    ]
    fig, axes = plt.subplots(len(crops), 3, figsize=(12, 3.2 * len(crops)), dpi=160, constrained_layout=True)
    for row, (name, x0, y0, x1, y1) in enumerate(crops):
        axes[row, 0].imshow(sem[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name}: SEM")
        axes[row, 1].imshow(afm_warped[y0:y1, x0:x1], cmap="magma")
        axes[row, 1].set_title("AFM warped")
        axes[row, 2].imshow(overlay[y0:y1, x0:x1])
        axes[row, 2].set_title("overlay")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(output_dir / "local_zoom_panels.png", bbox_inches="tight")
    plt.close(fig)


def jacobian_map(matrix: np.ndarray, model_name: str, shape: tuple[int, int]) -> np.ndarray:
    if model_name in {"similarity", "affine", "refined_affine"}:
        det = float(np.linalg.det(matrix[:, :2]))
        return np.full(shape, det, dtype=np.float32)
    h, w = shape
    yy, xx = np.mgrid[0:h:complex(0, 80), 0:w:complex(0, 80)]
    eps = 1.0
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    px = transform_points(matrix, pts + [eps, 0], model_name) - transform_points(matrix, pts - [eps, 0], model_name)
    py = transform_points(matrix, pts + [0, eps], model_name) - transform_points(matrix, pts - [0, eps], model_name)
    det = (px[:, 0] * py[:, 1] - px[:, 1] * py[:, 0]) / (4 * eps * eps)
    return cv2.resize(det.reshape(80, 80).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def pixel_size_um_from_afm(data: RegistrationData) -> float | None:
    scan = data.afm_meta.get("scan_size_um")
    if scan is None or not np.isfinite(float(scan)):
        return None
    return float(scan) / max(data.sem_display.shape[1], data.sem_display.shape[0])


def run_pipeline(config_path: Path, pick_points: bool = False, prepare_only: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    np.random.seed(int(config.get("random_seed", 0)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(config)
    write_json(
        output_dir / "data_inspection.json",
        {
            "afm_height_original": image_stats("afm_height_original_um", data.afm_height_um),
            "afm_display_height": image_stats("afm_display_height_um", data.afm_display_height_um),
            "afm_display_gray_resized": image_stats("afm_display_gray_resized", data.afm_display_gray),
            "sem_raw": image_stats("sem_raw", data.sem_raw),
            "sem_display": image_stats("sem_display", data.sem_display),
            "afm_metadata": data.afm_meta,
            "matrix_direction": "All reported global matrices map AFM_display pixels to SEM_display pixels.",
        },
    )
    save_gray(output_dir / "afm_display_height.png", data.afm_display_gray, "AFM display height, resized to SEM frame", "afmhot")
    save_gray(output_dir / "sem_display.png", data.sem_display, "SEM display frame", "gray")
    if data.ipf_display is not None:
        plt.imsave(output_dir / "ipf_reference_resized.png", data.ipf_display)
    save_orientation_sheet(output_dir, data)
    afm_features = extract_afm_features(data.afm_display_gray, config)
    sem_features = extract_sem_features(data.sem_display, config)
    save_feature_set(output_dir, "afm", afm_features)
    save_feature_set(output_dir, "sem", sem_features)
    cp_path = Path(config["control_points"]["path"])
    control_point_template(cp_path, data)
    if pick_points:
        launch_control_point_picker(data, cp_path)
        return {"status": "control_point_picker_closed", "control_points": str(cp_path)}

    afm_points, sem_points, cp_rows = load_control_points(cp_path)
    save_control_points_csv(output_dir / "control_points_loaded.csv", cp_rows)
    min_points = int(config["control_points"].get("min_points", 8))
    if prepare_only or len(afm_points) < min_points:
        report = {
            "status": "needs_control_points",
            "reason": f"{len(afm_points)} valid control-point pairs found; need at least {min_points}.",
            "next_step": f"Run python afm_sem_geometric_registration.py --config {config_path} --pick-points",
            "outputs_available": "data inspection, orientation candidates, AFM/SEM feature maps, boundary masks, skeletons, distance transforms, control point template",
            "control_points_file": str(cp_path.resolve()),
        }
        write_json(output_dir / "registration_report.json", report)
        (output_dir / "README_next_steps.md").write_text(
            "# AFM-SEM Registration Prepared\n\n"
            "Automatic preprocessing finished, but reliable cross-modal registration requires manual control points.\n\n"
            f"Control point file: `{cp_path.resolve()}`\n\n"
            "Use the picker:\n\n"
            f"```bash\npython afm_sem_geometric_registration.py --config {config_path} --pick-points\n```\n\n"
            "Pick 8-15 grain-boundary intersections, boundary bends, boundary-image-border intersections, or isolated defects. "
            "Then rerun the same command without `--pick-points`.\n",
            encoding="utf-8",
        )
        return report

    models = fit_models(afm_points, sem_points, config, output_dir)
    chosen = choose_model(models, config)
    refined = chosen
    pixel_size_um = pixel_size_um_from_afm(data)
    chosen_all_metrics = residual_metrics(chosen.residuals_px, pixel_size_um)
    distance_refinement_note = {
        "attempted": False,
        "accepted": False,
        "reason": "Distance-field refinement disabled by config.",
        "initial_control_point_metrics": chosen_all_metrics,
    }
    if config.get("distance_refinement", {}).get("enabled", True):
        candidate = affine_refine_distance(chosen, afm_features, sem_features, config, output_dir)
        candidate.name = "refined_affine"
        candidate.residuals_px = np.linalg.norm(transform_points(candidate.matrix, afm_points, "refined_affine") - sem_points, axis=1)
        candidate.train_metrics = residual_metrics(candidate.residuals_px, pixel_size_um)
        candidate.holdout_metrics = None
        max_median = float(config["model_selection"]["max_acceptable_median_error_px"])
        cp_guard = (
            candidate.train_metrics["rmse_px"] <= max(chosen_all_metrics["rmse_px"] * 1.35, chosen_all_metrics["rmse_px"] + 2.0)
            and candidate.train_metrics["median_px"] <= max(chosen_all_metrics["median_px"] * 1.35, chosen_all_metrics["median_px"] + 2.0)
            and candidate.train_metrics["median_px"] <= max_median
        )
        distance_refinement_note = {
            "attempted": True,
            "accepted": bool(cp_guard),
            "reason": "Accepted because control-point residuals stayed within guard limits."
            if cp_guard
            else "Rejected because boundary-distance refinement degraded manually validated control-point residuals; automatic masks contain non-corresponding edges.",
            "initial_control_point_metrics": chosen_all_metrics,
            "candidate_control_point_metrics": candidate.train_metrics,
        }
        if cp_guard:
            refined = candidate
        else:
            refined = chosen
            refined.train_metrics = chosen_all_metrics
            refined.holdout_metrics = None

    afm_warped, common_mask = warp_image(data.afm_display_gray, refined.matrix, refined.name, data.sem_display.shape)
    afm_skel_warped, _ = warp_image(afm_features.skeleton.astype(np.float32), refined.matrix, refined.name, data.sem_display.shape, cv2.INTER_NEAREST)
    bmetrics = boundary_metrics(afm_skel_warped, sem_features.skeleton, float(config["distance_refinement"]["tolerance_px"]))
    height_display_um = cv2.resize(data.afm_display_height_um.astype(np.float32), (data.sem_display.shape[1], data.sem_display.shape[0]), interpolation=cv2.INTER_AREA)
    height_warped_um, height_mask = warp_image(height_display_um, refined.matrix, refined.name, data.sem_display.shape, cv2.INTER_LINEAR)
    normal_data = height_to_normals(
        height_um=height_warped_um,
        scan_size_um=float(data.afm_meta["scan_size_um"]) if np.isfinite(float(data.afm_meta.get("scan_size_um", np.nan))) else float(data.sem_display.shape[1]),
        affine_afm_to_sem=np.eye(2, 3, dtype=np.float64),
        smooth_sigma_px=float(config["normalmap"]["smooth_sigma_px"]),
        level=bool(config["normalmap"].get("plane_level", True)),
    )
    normal_rgb = normal_direction_rgb(normal_data["normals_sample"], float(config["normalmap"]["tilt_color_ref_deg"]))
    normal_rgb[height_mask < 0.5] = 1.0
    plt.imsave(output_dir / "final_afm_height_warped_to_sem_um.png", robust_rescale(height_warped_um, 1, 99))
    plt.imsave(output_dir / "final_common_valid_mask.png", common_mask, cmap="gray")
    plt.imsave(output_dir / "final_normalmap_recomputed_in_sem_frame.png", normal_rgb)
    save_normalmap_with_legend(output_dir / "final_normalmap_recomputed_in_sem_frame_with_colorbar.png", normal_rgb, "Recomputed AFM normalmap in SEM coordinates", float(config["normalmap"]["tilt_color_ref_deg"]))
    save_registration_visuals(output_dir, data, afm_features, sem_features, refined, afm_warped, common_mask, afm_skel_warped, afm_points, sem_points)
    jmap = jacobian_map(refined.matrix, refined.name, data.sem_display.shape)
    save_gray(output_dir / "global_transform_jacobian.png", robust_rescale(jmap, 1, 99), "Global transform Jacobian/local area change", "viridis")
    np.savez_compressed(
        output_dir / "registration_results_data.npz",
        afm_height_warped_um=height_warped_um,
        common_mask=common_mask,
        normal_rgb=normal_rgb,
        matrix_afm_to_sem=refined.matrix,
        matrix_sem_to_afm=refined.inverse,
        afm_points=afm_points,
        sem_points=sem_points,
        residuals_px=refined.residuals_px,
        jacobian=jmap,
    )
    cp_residual_rows = []
    pred = transform_points(refined.matrix, afm_points, refined.name)
    for row, p, s, pr, res in zip(cp_rows, afm_points, sem_points, pred, refined.residuals_px):
        cp_residual_rows.append({**row, "pred_sem_x": pr[0], "pred_sem_y": pr[1], "residual_px": res})
    write_csv(output_dir / "control_point_residuals.csv", cp_residual_rows)
    nonrigid_note = {
        "attempted": False,
        "reason": "Non-rigid registration is disabled by config or not justified until manually validated global residuals show systematic spatial bias.",
        "global_jacobian_min": float(np.nanmin(jmap)),
        "global_jacobian_median": float(np.nanmedian(jmap)),
        "global_jacobian_max": float(np.nanmax(jmap)),
    }
    write_json(output_dir / "nonrigid_decision.json", nonrigid_note)
    report = {
        "status": "completed_global_registration",
        "chosen_model_before_distance_refinement": chosen.name,
        "final_model": refined.name,
        "selection_reason": "Lowest-free-degree model within configured error tolerance is selected before optional affine distance-field refinement. Homography is not assumed better by default.",
        "matrix_direction": "AFM_display -> SEM_display",
        "matrix_afm_to_sem": refined.matrix.tolist(),
        "matrix_sem_to_afm": refined.inverse.tolist(),
        "control_point_metrics": residual_metrics(refined.residuals_px, pixel_size_um_from_afm(data)),
        "boundary_metrics": bmetrics,
        "jacobian": nonrigid_note,
        "pixel_size_um_note": "Micron errors use AFM scan size divided by display width/height as an approximate same-FOV scale. If this calibration is wrong, use pixel errors and update config.",
        "nonrigid": nonrigid_note,
        "distance_refinement": distance_refinement_note,
    }
    write_json(output_dir / "registration_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-structure-driven AFM-SEM cross-modal registration.")
    parser.add_argument("--config", type=Path, default=Path("configs") / "afm_sem_pt_highres60.json")
    parser.add_argument("--pick-points", action="store_true", help="Open interactive AFM/SEM control point picker.")
    parser.add_argument("--prepare-only", action="store_true", help="Only run inspection/feature extraction and write control-point template.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(args.config, pick_points=args.pick_points, prepare_only=args.prepare_only)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
