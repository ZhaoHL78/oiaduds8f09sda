from __future__ import annotations

import numpy as np


def _stereo(v: np.ndarray) -> np.ndarray:
    return np.column_stack([v[:, 0] / (1.0 + v[:, 2]), v[:, 1] / (1.0 + v[:, 2])])


def cubic_reduce_to_standard_sector(directions: np.ndarray) -> np.ndarray:
    """Fold cubic directions into the [001]-[101]-[111] IPF sector.

    The returned sector uses coordinates [middle_abs, smallest_abs, largest_abs],
    matching the EDAX-style color reconstruction already validated in this repo.
    """
    flat = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    ordered = np.sort(np.abs(flat), axis=1)
    sector = np.column_stack([ordered[:, 1], ordered[:, 0], ordered[:, 2]])
    sector /= np.linalg.norm(sector, axis=1, keepdims=True) + 1e-12
    return sector.reshape(np.asarray(directions).shape).astype(np.float32)


def cubic_ipf_colors_from_crystal_directions(directions: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """EDAX/OIM-like cubic IPF colors for arbitrary crystal-frame directions.

    This is not ``RGB = abs(direction)``. Directions are first folded to the
    cubic fundamental sector and then colored by barycentric coordinates in the
    stereographic [001]-[101]-[111] triangle with EDAX-like gamma/saturation.
    """
    original_shape = np.asarray(directions).shape[:-1]
    sector = cubic_reduce_to_standard_sector(directions).reshape(-1, 3).astype(np.float64)

    vertices = np.array(
        [
            [0.0, 0.0, 1.0],  # [001]
            [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],  # [101]
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],  # [111]
        ],
        dtype=np.float64,
    )
    tri = _stereo(vertices)
    pts = _stereo(sector)

    a, b, c = tri
    mat = np.column_stack([a - c, b - c])
    inv = np.linalg.inv(mat)
    uv = (pts - c) @ inv.T
    w0 = uv[:, 0]
    w1 = uv[:, 1]
    w2 = 1.0 - w0 - w1
    weights = np.column_stack([w0, w1, w2])
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12

    rgb = weights[:, [0, 1, 2]]
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))
    rgb /= np.max(rgb, axis=1, keepdims=True) + 1e-12

    if valid is not None:
        valid_flat = np.asarray(valid, dtype=bool).reshape(-1)
        rgb[~valid_flat] = 0.0
    return rgb.reshape(*original_shape, 3).astype(np.float32)


def ipf_colors_from_sample_directions(
    sample_to_crystal: np.ndarray,
    sample_direction: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    matrices = np.asarray(sample_to_crystal, dtype=np.float64).reshape(-1, 3, 3)
    direction = np.asarray(sample_direction, dtype=np.float64).reshape(3)
    crystal = np.einsum("nij,j->ni", matrices, direction)
    crystal /= np.linalg.norm(crystal, axis=1, keepdims=True) + 1e-12
    return cubic_ipf_colors_from_crystal_directions(crystal, valid=valid)

