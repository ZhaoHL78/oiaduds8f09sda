from __future__ import annotations

import itertools

import numpy as np

from .ipf import cubic_ipf_colors_from_crystal_directions, cubic_reduce_to_standard_sector


def grain_average_orientations(orientations: np.ndarray, grain_id_grid: np.ndarray, valid_grid: np.ndarray) -> np.ndarray:
    matrices = np.asarray(orientations, dtype=np.float64).reshape(-1, 3, 3)
    grain_flat = grain_id_grid.reshape(-1)
    valid_flat = valid_grid.reshape(-1)
    max_grain = int(grain_flat.max())
    averaged = matrices.copy()
    grain_mean = np.repeat(np.eye(3)[None, :, :], max_grain + 1, axis=0)
    for grain_id in range(1, max_grain + 1):
        mask = (grain_flat == grain_id) & valid_flat
        if np.count_nonzero(mask) == 0:
            continue
        mean = matrices[mask].mean(axis=0)
        u, _s, vt = np.linalg.svd(mean)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        grain_mean[grain_id] = r
        averaged[mask] = r
    return averaged.astype(np.float64)


def crystal_normals_from_sample_normals(sample_to_crystal: np.ndarray, normals_sample: np.ndarray) -> np.ndarray:
    matrices = np.asarray(sample_to_crystal, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.asarray(normals_sample, dtype=np.float64).reshape(-1, 3)
    out = np.einsum("nij,nj->ni", matrices, normals)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out.reshape(np.asarray(normals_sample).shape).astype(np.float32)


def sample_normals_from_crystal_normals(sample_to_crystal: np.ndarray, normals_crystal: np.ndarray) -> np.ndarray:
    matrices = np.asarray(sample_to_crystal, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.asarray(normals_crystal, dtype=np.float64).reshape(-1, 3)
    out = np.einsum("nji,nj->ni", matrices, normals)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out.reshape(np.asarray(normals_crystal).shape).astype(np.float32)


def cubic_family_normals(hkl: tuple[int, int, int]) -> np.ndarray:
    vals = tuple(int(v) for v in hkl)
    normals: list[tuple[int, int, int]] = []
    for perm in set(itertools.permutations(vals)):
        for signs in itertools.product([-1, 1], repeat=3):
            v = tuple(int(a * s) for a, s in zip(perm, signs))
            if v != (0, 0, 0):
                normals.append(v)
    unique = np.unique(np.asarray(normals, dtype=np.float64), axis=0)
    unique /= np.linalg.norm(unique, axis=1, keepdims=True) + 1e-12
    return unique


def nearest_hkl_family(
    normals_crystal: np.ndarray,
    candidates: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(normals_crystal, dtype=np.float64).reshape(-1, 3)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    family_normals = [cubic_family_normals(tuple(hkl)) for hkl in candidates]
    best_dot = np.full(flat.shape[0], -1.0, dtype=np.float64)
    best_idx = np.full(flat.shape[0], -1, dtype=np.int16)
    for index, normals in enumerate(family_normals):
        dots = np.max(np.abs(flat @ normals.T), axis=1)
        update = dots > best_dot
        best_dot[update] = dots[update]
        best_idx[update] = index
    best_angle = np.degrees(np.arccos(np.clip(best_dot, -1.0, 1.0))).astype(np.float32)
    second_dot = np.full(flat.shape[0], -1.0, dtype=np.float64)
    for index, normals in enumerate(family_normals):
        dots = np.max(np.abs(flat @ normals.T), axis=1)
        dots[best_idx == index] = -1
        second_dot = np.maximum(second_dot, dots)
    margin_deg = np.degrees(
        np.arccos(np.clip(second_dot, -1.0, 1.0)) - np.arccos(np.clip(best_dot, -1.0, 1.0))
    ).astype(np.float32)
    return best_idx.reshape(normals_crystal.shape[:2]), best_angle.reshape(normals_crystal.shape[:2]), margin_deg.reshape(normals_crystal.shape[:2])


def surface_ipf_rgb(normals_crystal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return cubic_ipf_colors_from_crystal_directions(normals_crystal, valid=valid)


def reduced_direction(normals_crystal: np.ndarray) -> np.ndarray:
    return cubic_reduce_to_standard_sector(normals_crystal)

