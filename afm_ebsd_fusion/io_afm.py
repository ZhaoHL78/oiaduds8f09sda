from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def parse_igor_note(note: bytes | str) -> dict[str, str]:
    text = note.decode("utf-8", "ignore") if isinstance(note, bytes) else str(note)
    result: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


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
    if orientation == "transpose_flipud":
        axes = (1, 0, *range(2, image.ndim))
        return np.flipud(np.transpose(image, axes))
    raise ValueError(f"Unknown image orientation: {orientation}")


def read_ibw_channels(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from igor2.binarywave import load as load_ibw

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

    note = parse_igor_note(payload.get("note", b""))
    scan_size_um = float(note.get("ScanSize", "nan")) * 1e6 if "ScanSize" in note else float("nan")
    metadata = {
        "format": "ibw",
        "shape": list(data.shape[:2]),
        "dtype": str(data.dtype),
        "channel_labels": labels[: data.shape[2]],
        "scan_size_um": scan_size_um,
        "scan_angle_deg": float(note.get("ScanAngle", "nan")) if "ScanAngle" in note else float("nan"),
        "imaging_mode": note.get("ImagingMode", ""),
        "points_lines": note.get("PointsLines", ""),
        "raw_note_keys": sorted(note.keys()),
        "plane_leveling_from_metadata": "not reliably encoded in IBW note",
        "line_flattening_from_metadata": "not reliably encoded in IBW note",
    }
    channels = {label: data[:, :, index] for index, label in enumerate(labels[: data.shape[2]])}
    return channels, metadata


def read_height_file(
    path: Path,
    *,
    height_channel: str | None,
    height_unit_scale_to_um: float,
    display_orientation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    metadata: dict[str, Any]
    if suffix == ".ibw":
        channels, metadata = read_ibw_channels(path)
        if height_channel is None:
            height_channel = next(iter(channels))
        if height_channel not in channels:
            raise KeyError(f"AFM channel {height_channel!r} not found; available={list(channels)}")
        raw = channels[height_channel].astype(np.float32)
        metadata["height_channel"] = height_channel
    elif suffix == ".npy":
        raw = np.load(path).astype(np.float32)
        metadata = {"format": "npy", "shape": list(raw.shape), "dtype": str(raw.dtype)}
    elif suffix == ".npz":
        payload = np.load(path)
        key = height_channel or next(iter(payload.files))
        raw = np.asarray(payload[key], dtype=np.float32)
        metadata = {
            "format": "npz",
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "height_channel": key,
            "available_keys": list(payload.files),
        }
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        raw = np.loadtxt(path, delimiter=delimiter).astype(np.float32)
        metadata = {"format": suffix.lstrip("."), "shape": list(raw.shape), "dtype": str(raw.dtype)}
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        raw = tifffile.imread(path).astype(np.float32)
        metadata = {"format": "tiff", "shape": list(raw.shape), "dtype": str(raw.dtype)}
    else:
        raise ValueError(f"Unsupported AFM height format: {path.suffix}")

    if raw.ndim != 2:
        raise ValueError(f"AFM height must be a 2D matrix; got shape {raw.shape}")
    height_um = raw.astype(np.float32) * float(height_unit_scale_to_um)
    display_height_um = orient_image(height_um, display_orientation).astype(np.float32)
    metadata.update(
        {
            "path": str(path),
            "height_unit_scale_to_um": float(height_unit_scale_to_um),
            "display_orientation": display_orientation,
            "raw_height_min_um": float(np.nanmin(height_um)),
            "raw_height_max_um": float(np.nanmax(height_um)),
            "display_shape": list(display_height_um.shape),
        }
    )
    return height_um, display_height_um, metadata


def robust_rescale(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return np.clip((arr - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
