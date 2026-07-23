from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb

from .io_afm import robust_rescale


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normal_rgb(normals: np.ndarray, tilt_ref_deg: float = 20.0) -> np.ndarray:
    azimuth = (np.arctan2(normals[..., 1], normals[..., 0]) + np.pi) / (2 * np.pi)
    tilt = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    sat = np.clip(tilt / max(tilt_ref_deg, 1e-6), 0.0, 1.0)
    val = np.ones_like(sat) * 0.97
    return hsv_to_rgb(np.dstack([azimuth, sat, val])).astype(np.float32)


def save_scalar(path: Path, image: np.ndarray, title: str, cmap: str, colorbar_label: str, *, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180, constrained_layout=True)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.86, label=colorbar_label)
    fig.savefig(path)
    plt.close(fig)


def save_rgb(path: Path, rgb: np.ndarray, title: str, mask: np.ndarray | None = None) -> None:
    image = np.clip(rgb, 0.0, 1.0).copy()
    if mask is not None:
        image = image.copy()
        image[~mask] = 0.0
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180, constrained_layout=True)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_mask(path: Path, mask: np.ndarray, title: str) -> None:
    save_scalar(path, mask.astype(np.float32), title, "gray", "mask")


def save_label_map(
    path: Path,
    labels: np.ndarray,
    title: str,
    label_names: Sequence[str],
    valid: np.ndarray,
    *,
    unassigned_value: int,
) -> None:
    colors = [
        (0.00, 0.00, 0.00, 1.0),
        (0.86, 0.13, 0.12, 1.0),
        (0.18, 0.60, 0.20, 1.0),
        (0.15, 0.32, 0.85, 1.0),
        (0.94, 0.70, 0.14, 1.0),
        (0.72, 0.18, 0.78, 1.0),
        (0.00, 0.65, 0.72, 1.0),
        (0.95, 0.45, 0.18, 1.0),
        (0.50, 0.50, 0.50, 1.0),
        (0.98, 0.98, 0.98, 1.0),
    ]
    cmap = ListedColormap(colors[: max(len(label_names), 1)])
    show = labels.copy()
    show[~valid] = unassigned_value
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=180, constrained_layout=True)
    im = ax.imshow(show, cmap=cmap, vmin=0, vmax=max(len(label_names) - 1, 1), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(label_names)), shrink=0.86)
    cbar.ax.set_yticklabels(label_names)
    fig.savefig(path)
    plt.close(fig)


def save_boundary_overlay(path: Path, height: np.ndarray, boundary: np.ndarray, surface_rgb: np.ndarray, title: str) -> None:
    base = np.dstack([robust_rescale(height)] * 3)
    overlay = np.clip(0.55 * base + 0.70 * surface_rgb, 0.0, 1.0)
    boundary_dil = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    overlay[boundary_dil] = np.array([1.0, 0.05, 0.05])
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=180, constrained_layout=True)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_ipf_triangle_scatter(path: Path, reduced_dirs: np.ndarray, valid: np.ndarray, rgb: np.ndarray, title: str, max_points: int = 25000) -> None:
    dirs = reduced_dirs[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3)
    if dirs.shape[0] > max_points:
        rng = np.random.default_rng(20260723)
        take = rng.choice(dirs.shape[0], size=max_points, replace=False)
        dirs = dirs[take]
        colors = colors[take]
    x = dirs[:, 0] / (1.0 + dirs[:, 2])
    y = dirs[:, 1] / (1.0 + dirs[:, 2])
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=180, constrained_layout=True)
    ax.scatter(x, y, s=2, c=np.clip(colors, 0, 1), alpha=0.55, linewidths=0)
    verts = np.array([[0, 0, 1], [1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])
    vx = verts[:, 0] / (1 + verts[:, 2])
    vy = verts[:, 1] / (1 + verts[:, 2])
    ax.plot([vx[0], vx[1], vx[2], vx[0]], [vy[0], vy[1], vy[2], vy[0]], "k-", lw=1.2)
    for label, px, py in zip(["001", "101", "111"], vx, vy):
        ax.text(px, py, label, ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def save_overview(path: Path, panels: list[tuple[str, np.ndarray, str]]) -> None:
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.4), dpi=150, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (title, image, kind) in zip(axes_arr, panels):
        if kind == "rgb":
            ax.imshow(np.clip(image, 0.0, 1.0))
        else:
            ax.imshow(image, cmap=kind)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes_arr[len(panels) :]:
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)

