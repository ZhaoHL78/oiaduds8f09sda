from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .io_afm import orient_image


def compare_ipf_reference(ipf_z_grid: np.ndarray, reference_path: Path, configured_orientation: str) -> dict[str, Any]:
    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float32) / 255.0
    candidates = ["raw", "flipud", "fliplr", "rot180", "transpose"]
    rows = []
    for orientation in candidates:
        try:
            img = orient_image(ipf_z_grid, orientation)
        except Exception:
            continue
        resized = cv2.resize(img.astype(np.float32), (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = resized - reference
        rows.append(
            {
                "orientation": orientation,
                "mean_abs_rgb": float(np.mean(np.abs(diff))),
                "rmse_rgb": float(np.sqrt(np.mean(diff * diff))),
                "gray_corr": float(np.corrcoef(resized.mean(axis=2).reshape(-1), reference.mean(axis=2).reshape(-1))[0, 1]),
            }
        )
    rows.sort(key=lambda row: row["mean_abs_rgb"])
    configured = next((row for row in rows if row["orientation"] == configured_orientation), None)
    best = rows[0] if rows else None
    passed = configured is not None and best is not None and configured["orientation"] == best["orientation"] and configured["mean_abs_rgb"] < 0.08
    return {
        "reference_path": str(reference_path),
        "reference_shape": list(reference.shape[:2]),
        "configured_orientation": configured_orientation,
        "candidate_metrics": rows,
        "best_orientation": best["orientation"] if best else None,
        "configured_metrics": configured,
        "passed": bool(passed),
        "criterion": "configured orientation must be best candidate and mean_abs_rgb < 0.08",
    }


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))).astype(np.float32)


def summarize_angles(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = values[mask & np.isfinite(values)]
    if vals.size == 0:
        return {"count": 0, "mean_deg": float("nan"), "median_deg": float("nan"), "p95_deg": float("nan")}
    return {
        "count": int(vals.size),
        "mean_deg": float(np.mean(vals)),
        "median_deg": float(np.median(vals)),
        "p95_deg": float(np.percentile(vals, 95)),
        "max_deg": float(np.max(vals)),
    }

