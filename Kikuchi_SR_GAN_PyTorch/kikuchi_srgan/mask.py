from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def circular_mask(
    height: int,
    width: int,
    radius_fraction: float = 0.49,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return a float32 circular mask with ones inside the valid detector area."""

    if not 0.0 < radius_fraction <= 0.5:
        raise ValueError("radius_fraction must be in (0, 0.5]")
    cy, cx = center if center is not None else ((height - 1) / 2.0, (width - 1) / 2.0)
    radius = min(height, width) * radius_fraction
    yy, xx = np.ogrid[:height, :width]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    return mask.astype(np.float32)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, mask, cmap="gray", vmin=0, vmax=1)
