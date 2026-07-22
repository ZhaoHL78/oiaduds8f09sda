from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from skimage import exposure

from align_pt_afm_sem_ipf import read_afm_channels, robust_rescale
from export_h5_ipf_bse_maps import cubic_ipf_z_colors


DEFAULT_AFM = Path(r"C:\Users\WHJ\OneDrive\xwechat_files\wxid_udhlesdsllnu22_8cd9\msg\file\2026-07\Pt-1(1).ibw")
DEFAULT_ALIGNMENT_METADATA = Path("outputs") / "pt_afm_sem_ipf_alignment" / "afm_sem_ipf_alignment_metadata.json"
DEFAULT_FINETUNED_IPF_METADATA = (
    Path("outputs") / "pt3_area90_finetuned_ipf_map" / "pt3_area90_finetuned_ipf_metadata.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_afm_ebsd_scharr_surface_index"
DEFAULT_H5 = Path(r"D:\EBSD project\EBSD-data\Pt-1\20251209Pt.edaxh5")
DEFAULT_H5_GROUP = "20251209/Pt-3/Area 3-90/OIM Map 1"
DEFAULT_UP2 = Path(r"D:\EBSD project\EBSD-data\Pt-1\20251209_Pt-3_Area 4_OIM Map 1.up2")

OHP_LINE_COLORS = (
    "#fee400",
    "#c800ff",
    "#0032ff",
    "#ff7040",
    "#ff448c",
    "#7100ff",
    "#00b6ff",
    "#76ff54",
)
OHP_LINE_VARIANT = "normal_theta_rho+_yup"

HKL_CANDIDATES = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [2, 1, 0],
        [2, 1, 1],
        [2, 2, 1],
        [3, 1, 0],
        [3, 1, 1],
        [3, 2, 0],
        [3, 2, 1],
        [3, 3, 1],
        [4, 1, 0],
        [4, 1, 1],
        [4, 2, 1],
        [4, 3, 1],
    ],
    dtype=np.float64,
)
HKL_LABELS = tuple(f"{{{int(h)}{int(k)}{int(l)}}}" for h, k, l in HKL_CANDIDATES)


@dataclass(frozen=True)
class EbsdMap:
    nrows: int
    ncols: int
    orientations: np.ndarray
    valid: np.ndarray
    phase: np.ndarray
    iq: np.ndarray
    ci: np.ndarray
    ipf_grid: np.ndarray


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plane_level(height_um: np.ndarray) -> np.ndarray:
    y, x = np.indices(height_um.shape, dtype=np.float64)
    values = height_um.astype(np.float64)
    finite = np.isfinite(values)
    design = np.column_stack([x[finite], y[finite], np.ones(np.count_nonzero(finite))])
    coeff, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    plane = coeff[0] * x + coeff[1] * y + coeff[2]
    return (values - plane).astype(np.float32)


def affine_rotation_2d(affine_afm_to_sem: np.ndarray) -> np.ndarray:
    linear = affine_afm_to_sem[:, :2].astype(np.float64)
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
    smooth = gaussian_filter(height.astype(np.float32), sigma=smooth_sigma_px) if smooth_sigma_px > 0 else height

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

    # Only the in-plane rotation part of AFM->SEM is used for normals. Scale and
    # shear are registration terms, not height-gradient terms.
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


def save_scalar_image(
    path: Path,
    values: np.ndarray,
    title: str,
    cmap: str,
    colorbar_label: str,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    display = np.asarray(values, dtype=np.float32).copy()
    if mask is not None:
        display = np.ma.array(display, mask=~mask.astype(bool))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220, constrained_layout=True)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.82, label=colorbar_label)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_normalmap_with_legend(
    path: Path,
    normal_rgb: np.ndarray,
    title: str,
    tilt_ref_deg: float,
) -> None:
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
    legend = normal_legend_rgb()
    axes[1].imshow(legend, extent=(-1, 1, -1, 1))
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


def save_rgb_image(path: Path, rgb: np.ndarray, title: str, mask: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220, constrained_layout=True)
    if mask is None:
        ax.imshow(np.clip(rgb, 0.0, 1.0))
    else:
        ax.imshow(rgba(rgb, mask, alpha=1.0))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def read_ebsd_map(h5_path: Path, h5_group: str, orientation_delta_deg: tuple[float, float, float] | None) -> EbsdMap:
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        data = group["EBSD/ANG/DATA/DATA"]
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        orientations = data["Orientations"][:].astype(np.float64).reshape(-1, 3, 3)
        if orientation_delta_deg is not None:
            delta = Rotation.from_euler("xyz", orientation_delta_deg, degrees=True).as_matrix()
            orientations = orientations @ delta.T
        valid = data["Valid"][:].astype(bool)
        phase = data["Phase"][:].astype(np.int16)
        iq = data["IQ"][:].astype(np.float32)
        ci = data["CI"][:].astype(np.float32)
    ipf = cubic_ipf_z_colors(orientations.reshape(-1, 9), valid, ci).reshape(nrows, ncols, 3)
    return EbsdMap(
        nrows=nrows,
        ncols=ncols,
        orientations=orientations,
        valid=valid,
        phase=phase,
        iq=iq,
        ci=ci,
        ipf_grid=ipf,
    )


def map_afm_to_ebsd(
    afm_shape: tuple[int, int],
    affine_afm_to_sem: np.ndarray,
    sem_shape: tuple[int, int],
    ebsd: EbsdMap,
) -> dict[str, np.ndarray]:
    height, width = afm_shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    coords = np.stack([cols.ravel(), rows.ravel(), np.ones(height * width, dtype=np.float32)], axis=0)
    sem_xy = (affine_afm_to_sem @ coords).T.astype(np.float32)
    sem_h, sem_w = sem_shape
    inside_sem = (
        (sem_xy[:, 0] >= 0)
        & (sem_xy[:, 0] < sem_w)
        & (sem_xy[:, 1] >= 0)
        & (sem_xy[:, 1] < sem_h)
    )
    ebsd_col = np.floor(sem_xy[:, 0] / sem_w * ebsd.ncols).astype(np.int64)
    ebsd_row = np.floor(sem_xy[:, 1] / sem_h * ebsd.nrows).astype(np.int64)
    ebsd_col = np.clip(ebsd_col, 0, ebsd.ncols - 1)
    ebsd_row = np.clip(ebsd_row, 0, ebsd.nrows - 1)
    ebsd_index = ebsd_row * ebsd.ncols + ebsd_col
    valid = inside_sem & ebsd.valid[ebsd_index] & (ebsd.phase[ebsd_index] == 1)
    return {
        "sem_xy": sem_xy.reshape(height, width, 2),
        "ebsd_row": ebsd_row.reshape(height, width),
        "ebsd_col": ebsd_col.reshape(height, width),
        "ebsd_index": ebsd_index.reshape(height, width),
        "valid": valid.reshape(height, width),
    }


def crystal_normals_from_ebsd(normals_sample: np.ndarray, ebsd: EbsdMap, ebsd_index: np.ndarray) -> np.ndarray:
    flat_normals = normals_sample.reshape(-1, 3).astype(np.float64)
    matrices = ebsd.orientations[ebsd_index.ravel()]
    normals_crystal = np.einsum("nij,nj->ni", matrices, flat_normals)
    normals_crystal /= np.linalg.norm(normals_crystal, axis=1, keepdims=True) + 1e-12
    return normals_crystal.reshape(normals_sample.shape).astype(np.float32)


def fold_cubic_direction(vectors: np.ndarray) -> np.ndarray:
    folded = np.sort(np.abs(vectors), axis=2)[..., ::-1]
    folded /= np.linalg.norm(folded, axis=2, keepdims=True) + 1e-12
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
    candidate_dirs = HKL_CANDIDATES / np.linalg.norm(HKL_CANDIDATES, axis=1, keepdims=True)
    flat = folded.reshape(-1, 3).astype(np.float64)
    dots = np.clip(flat @ candidate_dirs.T, -1.0, 1.0)
    best = np.argmax(dots, axis=1)
    angle = np.degrees(np.arccos(dots[np.arange(flat.shape[0]), best]))
    return best.reshape(folded.shape[:2]).astype(np.int16), angle.reshape(folded.shape[:2]).astype(np.float32)


def rgba(rgb: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    out[..., 3] = mask.astype(np.float32) * alpha
    return out


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def save_overview(
    path: Path,
    height_um: np.ndarray,
    normal_rgb: np.ndarray,
    tilt_deg: np.ndarray,
    facet_rgb_map: np.ndarray,
    valid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), dpi=220, constrained_layout=True)
    im0 = axes[0, 0].imshow(height_um * 1000.0, cmap="viridis")
    axes[0, 0].set_title("AFM height (nm)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.82)
    axes[0, 1].imshow(normal_rgb)
    axes[0, 1].set_title("Surface normal direction color")
    im2 = axes[1, 0].imshow(tilt_deg, cmap="magma", vmin=0, vmax=np.nanpercentile(tilt_deg, 99.0))
    axes[1, 0].set_title("Normal tilt from sample Z (deg)")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.82)
    axes[1, 1].imshow(rgba(facet_rgb_map, valid, alpha=1.0))
    axes[1, 1].set_title("Crystal-frame surface index color")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("AFM height field -> normals -> crystal surface index")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_top_view(
    path: Path,
    ipf_path: Path,
    facet_only_path: Path,
    sem_gray: np.ndarray,
    ipf_sem: np.ndarray,
    affine_afm_to_sem: np.ndarray,
    facet_rgb_map: np.ndarray,
    normal_rgb: np.ndarray,
    valid: np.ndarray,
) -> None:
    sem_rgb = np.dstack([robust_rescale(sem_gray)] * 3)
    ipf_overlay = np.clip(0.42 * sem_rgb + 0.72 * ipf_sem, 0.0, 1.0)
    sem_shape = sem_gray.shape
    warped_facet = cv2.warpAffine(
        rgba(facet_rgb_map, valid, alpha=0.82),
        affine_afm_to_sem,
        (sem_shape[1], sem_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped_normal = cv2.warpAffine(
        rgba(normal_rgb, valid, alpha=0.82),
        affine_afm_to_sem,
        (sem_shape[1], sem_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    plt.imsave(ipf_path, ipf_overlay)
    facet_only = np.zeros((sem_shape[0], sem_shape[1], 4), dtype=np.float32)
    facet_only[..., :3] = ipf_overlay
    facet_only[..., 3] = 1.0
    alpha = warped_facet[..., 3:4]
    facet_only[..., :3] = facet_only[..., :3] * (1.0 - alpha) + warped_facet[..., :3] * alpha
    plt.imsave(facet_only_path, facet_only)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), dpi=220, constrained_layout=True)
    axes[0].imshow(ipf_overlay)
    axes[0].set_title("EBSD/IPF top view in SEM frame")
    axes[1].imshow(ipf_overlay)
    axes[1].imshow(warped_normal)
    axes[1].set_title("AFM normal direction over EBSD/IPF")
    axes[2].imshow(ipf_overlay)
    axes[2].imshow(warped_facet)
    axes[2].set_title("Crystal surface index over EBSD/IPF")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("AFM-derived surface normals combined with Pt-3 Area 3-90 EBSD orientation")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_3d_surface(
    path: Path,
    height_um: np.ndarray,
    color_rgb: np.ndarray,
    title: str,
    scan_size_um: float,
    stride: int,
) -> None:
    h, w = height_um.shape
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    x_um = x.astype(np.float32) / max(w - 1, 1) * scan_size_um
    y_um = y.astype(np.float32) / max(h - 1, 1) * scan_size_um
    z_nm = height_um[::stride, ::stride] * 1000.0
    colors = color_rgb[::stride, ::stride]

    fig = plt.figure(figsize=(10.5, 8.5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_um,
        y_um,
        z_nm,
        facecolors=np.clip(colors, 0.0, 1.0),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_xlabel("AFM x (um)")
    ax.set_ylabel("AFM y (um)")
    ax.set_zlabel("height (nm)", labelpad=12)
    ax.set_title(title)
    ax.view_init(elev=58, azim=-70)
    ax.set_box_aspect((1, 1, 0.18))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25, transparent=True)
    plt.close(fig)


def save_interactive_surface_index(
    path: Path,
    height_um: np.ndarray,
    facet_rgb_map: np.ndarray,
    valid: np.ndarray,
    scan_size_um: float,
    stride: int,
) -> None:
    h, w = height_um.shape
    rows = np.arange(0, h, stride, dtype=np.int32)
    cols = np.arange(0, w, stride, dtype=np.int32)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    z = height_um[yy, xx].astype(np.float32) * 1000.0
    x = xx.astype(np.float32) / max(w - 1, 1) * scan_size_um
    y = yy.astype(np.float32) / max(h - 1, 1) * scan_size_um
    colors = np.clip(facet_rgb_map[yy, xx] * 255.0, 0, 255).astype(np.uint8)
    valid_ds = valid[yy, xx]

    vertex_index = -np.ones(x.shape, dtype=np.int32)
    flat_valid = valid_ds.ravel()
    vertex_index.ravel()[flat_valid] = np.arange(int(np.count_nonzero(flat_valid)), dtype=np.int32)
    vertices = np.column_stack([x.ravel()[flat_valid], y.ravel()[flat_valid], z.ravel()[flat_valid]])
    vertex_colors = [f"rgb({r},{g},{b})" for r, g, b in colors.reshape(-1, 3)[flat_valid]]

    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []
    for row in range(vertex_index.shape[0] - 1):
        for col in range(vertex_index.shape[1] - 1):
            a = int(vertex_index[row, col])
            b = int(vertex_index[row, col + 1])
            c = int(vertex_index[row + 1, col])
            d = int(vertex_index[row + 1, col + 1])
            if min(a, b, c) >= 0:
                faces_i.append(a)
                faces_j.append(c)
                faces_k.append(b)
            if min(b, c, d) >= 0:
                faces_i.append(b)
                faces_j.append(c)
                faces_k.append(d)

    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces_i,
        j=faces_j,
        k=faces_k,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=dict(ambient=0.78, diffuse=0.5, specular=0.05, roughness=0.9),
        lightposition=dict(x=0, y=-20, z=80),
        name="crystal surface index",
        showscale=False,
        hovertemplate="x=%{x:.2f} um<br>y=%{y:.2f} um<br>height=%{z:.1f} nm<extra></extra>",
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title="Interactive AFM 3D surface colored by crystal-frame surface index",
        scene=dict(
            xaxis_title="AFM x (um)",
            yaxis_title="AFM y (um)",
            zaxis_title="height (nm)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.16),
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.85)),
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        paper_bgcolor="white",
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def save_facet_legend(path: Path) -> None:
    size = 420
    xs = np.linspace(0, 1, size)
    ys = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(xs, ys)
    tri = np.array([[0.10, 0.88], [0.90, 0.88], [0.63, 0.12]], dtype=np.float64)
    denom = ((tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1]))
    w0 = ((tri[1, 1] - tri[2, 1]) * (xx - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (yy - tri[2, 1])) / denom
    w1 = ((tri[2, 1] - tri[0, 1]) * (xx - tri[2, 0]) + (tri[0, 0] - tri[2, 0]) * (yy - tri[2, 1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    dirs = (
        w0[..., None] * np.array([1.0, 0.0, 0.0])
        + w1[..., None] * np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
        + w2[..., None] * np.array([1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    )
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12
    image = np.ones((size, size, 4), dtype=np.float32)
    image[..., :3] = facet_type_rgb(dirs)
    image[..., 3] = inside.astype(np.float32)
    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=220)
    ax.imshow(image)
    ax.text(tri[0, 0] * size, tri[0, 1] * size + 20, "{100}", ha="center", va="top", fontsize=12)
    ax.text(tri[1, 0] * size, tri[1, 1] * size + 20, "{110}", ha="center", va="top", fontsize=12)
    ax.text(tri[2, 0] * size, tri[2, 1] * size - 20, "{111}", ha="center", va="bottom", fontsize=12)
    ax.set_title("Cubic surface index color key")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def write_counts(path: Path, hkl_indices: np.ndarray, angle_deg: np.ndarray, valid: np.ndarray) -> None:
    rows = []
    total = int(np.count_nonzero(valid))
    for idx, label in enumerate(HKL_LABELS):
        mask = valid & (hkl_indices == idx)
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "nearest_hkl": label,
                "pixel_count": count,
                "fraction": count / max(total, 1),
                "median_misorientation_deg": float(np.nanmedian(angle_deg[mask])) if count else float("nan"),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stride_csv(
    path: Path,
    height_um: np.ndarray,
    normals_sample: np.ndarray,
    normals_crystal: np.ndarray,
    folded: np.ndarray,
    facet_rgb_map: np.ndarray,
    hkl_indices: np.ndarray,
    hkl_angle_deg: np.ndarray,
    mapping: dict[str, np.ndarray],
    ebsd: EbsdMap,
    scan_size_um: float,
    stride: int,
) -> None:
    h, w = height_um.shape
    rows_out = []
    valid = mapping["valid"]
    for row in range(0, h, stride):
        for col in range(0, w, stride):
            if not bool(valid[row, col]):
                continue
            idx = int(mapping["ebsd_index"][row, col])
            rows_out.append(
                {
                    "afm_row": row,
                    "afm_col": col,
                    "afm_x_um": col / max(w - 1, 1) * scan_size_um,
                    "afm_y_um": row / max(h - 1, 1) * scan_size_um,
                    "height_um": float(height_um[row, col]),
                    "normal_sample_x": float(normals_sample[row, col, 0]),
                    "normal_sample_y": float(normals_sample[row, col, 1]),
                    "normal_sample_z": float(normals_sample[row, col, 2]),
                    "normal_crystal_x": float(normals_crystal[row, col, 0]),
                    "normal_crystal_y": float(normals_crystal[row, col, 1]),
                    "normal_crystal_z": float(normals_crystal[row, col, 2]),
                    "folded_h": float(folded[row, col, 0]),
                    "folded_k": float(folded[row, col, 1]),
                    "folded_l": float(folded[row, col, 2]),
                    "facet_rgb_r": float(facet_rgb_map[row, col, 0]),
                    "facet_rgb_g": float(facet_rgb_map[row, col, 1]),
                    "facet_rgb_b": float(facet_rgb_map[row, col, 2]),
                    "nearest_hkl": HKL_LABELS[int(hkl_indices[row, col])],
                    "nearest_hkl_angle_deg": float(hkl_angle_deg[row, col]),
                    "sem_x_px": float(mapping["sem_xy"][row, col, 0]),
                    "sem_y_px": float(mapping["sem_xy"][row, col, 1]),
                    "ebsd_row": int(mapping["ebsd_row"][row, col]),
                    "ebsd_col": int(mapping["ebsd_col"][row, col]),
                    "ebsd_index": idx,
                    "iq": float(ebsd.iq[idx]),
                    "ci": float(ebsd.ci[idx]),
                }
            )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def write_ply(
    path: Path,
    height_um: np.ndarray,
    facet_rgb_map: np.ndarray,
    valid: np.ndarray,
    scan_size_um: float,
    stride: int,
) -> None:
    h, w = height_um.shape
    points = []
    for row in range(0, h, stride):
        for col in range(0, w, stride):
            if not bool(valid[row, col]):
                continue
            rgb = np.clip(facet_rgb_map[row, col] * 255.0, 0, 255).astype(np.uint8)
            points.append(
                (
                    col / max(w - 1, 1) * scan_size_um,
                    row / max(h - 1, 1) * scan_size_um,
                    height_um[row, col],
                    int(rgb[0]),
                    int(rgb[1]),
                    int(rgb[2]),
                )
            )
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point in points:
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.9f} {point[3]} {point[4]} {point[5]}\n")


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, dict[str, int]]:
    with path.open("rb") as stream:
        version = int(np.fromfile(stream, np.uint32, 1)[0])
        width, height, header_bytes = np.fromfile(stream, np.uint32, 3).astype(np.int64)
        width = int(width)
        height = int(height)
        header_bytes = int(header_bytes)
        bytes_per_pattern = width * height * np.dtype("<u2").itemsize
        count = (path.stat().st_size - header_bytes) // bytes_per_pattern
        if index < 0 or index >= count:
            raise IndexError(f"UP2 index {index} is outside 0..{count - 1} for {path}")
        stream.seek(header_bytes + index * bytes_per_pattern)
        pattern = np.fromfile(stream, dtype="<u2", count=width * height).reshape(height, width)
    return pattern, {"version": version, "width": width, "height": height, "header_bytes": header_bytes, "count": int(count)}


def pattern_display(pattern: np.ndarray) -> np.ndarray:
    values = pattern.astype(np.float32)
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [0.4, 99.6])
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def ohp_lines_from_record(
    record: np.ndarray,
    circle_size: float,
    output_size: int,
    rho_sign: float = 1.0,
) -> list[dict[str, float]]:
    center = 0.5 * (output_size - 1)
    x_min = -center
    x_max = output_size - 1 - center
    y_min = -(output_size - 1 - center)
    y_max = center
    lines: list[dict[str, float]] = []
    for band_id, (rho_bin, theta_deg, width, peak) in enumerate(record.reshape(-1, 4)):
        # EDAX OHP stores rho in Hough bins. The pixel rho is the offset from
        # the Hough circle center scaled by the detector diameter. Do not apply
        # Maximum Rho Fraction again here; doing so shrinks the bands toward the
        # center and makes them visibly miss the Kikuchi pattern.
        rho = rho_sign * (float(rho_bin) - 0.5 * circle_size) * (output_size / circle_size)
        theta = np.deg2rad(float(theta_deg))
        c = np.cos(theta)
        s = np.sin(theta)
        points: list[tuple[float, float]] = []
        if abs(s) > 1e-8:
            for x in (x_min, x_max):
                y = (rho - x * c) / s
                if y_min - 1e-5 <= y <= y_max + 1e-5:
                    points.append((float(x), float(y)))
        if abs(c) > 1e-8:
            for y in (y_min, y_max):
                x = (rho - y * s) / c
                if x_min - 1e-5 <= x <= x_max + 1e-5:
                    points.append((float(x), float(y)))
        unique: list[tuple[float, float]] = []
        for point in points:
            if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 > 1e-4 for old in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        p0, p1 = max(
            ((a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]),
            key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
        )
        lines.append(
            {
                "band_id": float(band_id),
                "x0": float(p0[0] + center),
                "y0": float(center - p0[1]),
                "x1": float(p1[0] + center),
                "y1": float(center - p1[1]),
                "rho_bin": float(rho_bin),
                "theta_deg": float(theta_deg),
                "bandwidth": float(width),
                "peak": float(peak),
            }
        )
    return lines


def phase_hkl_families(h5_path: Path, h5_group: str, phase_id: int) -> list[str]:
    families: list[str] = []
    with h5py.File(h5_path, "r") as h5:
        dataset = h5[f"{h5_group}/EBSD/ANG/HEADER/Phase/{phase_id}/HKL Families"]
        for row in dataset[:]:
            if int(row["Use in Indexing"]) == 0:
                continue
            families.append(f"({int(row['H'])}{int(row['K'])}{int(row['L'])})")
    return families


def read_ohp_lines(h5_path: Path, h5_group: str, index: int, output_size: int) -> list[dict[str, float]]:
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        header = group["EBSD/OHP/HEADER"]
        circle_size = float(np.asarray(header["Circle Size"][()]).reshape(-1)[0])
        record = group["EBSD/OHP/DATA/DATA"][index].astype(np.float32)
    return ohp_lines_from_record(record, circle_size, output_size=output_size)


def choose_coupling_point(
    mapping: dict[str, np.ndarray],
    ebsd: EbsdMap,
    tilt_deg: np.ndarray,
) -> tuple[int, int]:
    valid = mapping["valid"].astype(bool)
    h, w = valid.shape
    rows, cols = np.indices((h, w), dtype=np.float32)
    ebsd_index = mapping["ebsd_index"]
    ci = ebsd.ci[ebsd_index]
    iq = ebsd.iq[ebsd_index]
    core = valid & (rows > 0.08 * h) & (rows < 0.92 * h) & (cols > 0.08 * w) & (cols < 0.92 * w)
    flat = core & (tilt_deg < 6.0) & (ci > 0.45)
    candidates = flat if np.count_nonzero(flat) > 0 else core if np.count_nonzero(core) > 0 else valid
    iq_scale = np.nanpercentile(iq[candidates], 99.0) if np.count_nonzero(candidates) else 1.0
    dist = np.sqrt(((rows - 0.5 * h) / h) ** 2 + ((cols - 0.5 * w) / w) ** 2)
    score = 1.8 * ci + 0.45 * np.clip(iq / max(iq_scale, 1e-6), 0.0, 1.5) - 0.65 * dist - 0.025 * tilt_deg
    score = np.where(candidates, score, -np.inf)
    flat_index = int(np.nanargmax(score))
    row, col = np.unravel_index(flat_index, valid.shape)
    return int(row), int(col)


def save_kikuchi_coupling_figure(
    path: Path,
    h5_path: Path,
    h5_group: str,
    up2_path: Path,
    selected: dict[str, Any],
    sem_gray: np.ndarray,
    ipf_sem: np.ndarray,
    normal_rgb: np.ndarray,
    facet_rgb_map: np.ndarray,
) -> None:
    pattern, up2_info = read_up2_pattern(up2_path, int(selected["ebsd_index"]))
    pattern_gray = pattern_display(pattern)
    lines = read_ohp_lines(h5_path, h5_group, int(selected["ebsd_index"]), output_size=pattern_gray.shape[1])

    sem_rgb = np.dstack([robust_rescale(sem_gray)] * 3)
    ipf_overlay = np.clip(0.42 * sem_rgb + 0.72 * ipf_sem, 0.0, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), dpi=220, constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(pattern_gray, cmap="gray", vmin=0, vmax=1)
    for line in lines:
        band_id = int(line["band_id"])
        ax.plot(
            [line["x0"], line["x1"]],
            [line["y0"], line["y1"]],
            color=OHP_LINE_COLORS[band_id % len(OHP_LINE_COLORS)],
            linewidth=max(1.1, 0.35 * float(line["bandwidth"])),
            alpha=0.86,
        )
        ax.text(
            0.5 * (line["x0"] + line["x1"]),
            0.5 * (line["y0"] + line["y1"]),
            str(band_id + 1),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
        )
    ax.set_title("Raw Kikuchi pattern + H5 OHP bands")
    ax.axis("off")

    sem_x, sem_y = selected["sem_xy"]
    axes[0, 1].imshow(ipf_overlay)
    axes[0, 1].scatter([sem_x], [sem_y], s=100, facecolors="none", edgecolors="black", linewidths=1.7)
    axes[0, 1].scatter([sem_x], [sem_y], s=16, c="red")
    axes[0, 1].set_title("Same EBSD index on IPF/SEM")
    axes[0, 1].axis("off")

    afm_row = int(selected["afm_row"])
    afm_col = int(selected["afm_col"])
    axes[0, 2].imshow(normal_rgb)
    axes[0, 2].scatter([afm_col], [afm_row], s=100, facecolors="none", edgecolors="black", linewidths=1.7)
    axes[0, 2].scatter([afm_col], [afm_row], s=16, c="red")
    axes[0, 2].set_title("AFM Scharr normal at the same point")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(rgba(facet_rgb_map, selected["valid_mask"], alpha=1.0))
    axes[1, 0].scatter([afm_col], [afm_row], s=100, facecolors="none", edgecolors="black", linewidths=1.7)
    axes[1, 0].scatter([afm_col], [afm_row], s=16, c="red")
    axes[1, 0].set_title("Crystal-frame surface index at the same point")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(normal_legend_rgb(), extent=(-1, 1, -1, 1))
    axes[1, 1].set_title("Normal color key")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.94, 0.50, "0 deg", transform=axes[1, 1].transAxes, ha="right", va="center", fontsize=8)
    axes[1, 1].text(0.06, 0.50, "180 deg", transform=axes[1, 1].transAxes, ha="left", va="center", fontsize=8)
    axes[1, 1].text(0.50, 0.94, "+90 deg", transform=axes[1, 1].transAxes, ha="center", va="top", fontsize=8)
    axes[1, 1].text(0.50, 0.06, "-90 deg", transform=axes[1, 1].transAxes, ha="center", va="bottom", fontsize=8)

    phase_families = ", ".join(selected["phase_hkl_families"]) or "not stored"
    text = (
        f"UP2: {up2_path.name}\n"
        f"OHP line variant: {OHP_LINE_VARIANT}\n"
        f"UP2 resolution/count: {up2_info['width']}x{up2_info['height']} / {up2_info['count']}\n"
        f"EBSD index: {selected['ebsd_index']}  row={selected['ebsd_row']} col={selected['ebsd_col']}\n"
        f"IQ={selected['iq']:.1f}, CI={selected['ci']:.3f}, phase={selected['phase']}\n"
        f"indexed phase HKL families: {phase_families}\n\n"
        f"n_sample = [{selected['normal_sample'][0]:+.4f}, {selected['normal_sample'][1]:+.4f}, {selected['normal_sample'][2]:+.4f}]\n"
        f"n_crystal = G @ n_sample\n"
        f"          = [{selected['normal_crystal'][0]:+.4f}, {selected['normal_crystal'][1]:+.4f}, {selected['normal_crystal'][2]:+.4f}]\n"
        f"folded cubic [h k l] = [{selected['folded'][0]:.4f}, {selected['folded'][1]:.4f}, {selected['folded'][2]:.4f}]\n"
        f"nearest surface index = {selected['nearest_hkl']}  angle={selected['nearest_hkl_angle_deg']:.2f} deg\n\n"
        "Interpretation:\n"
        "Kikuchi bands fix the local crystal orientation G at this EBSD point.\n"
        "AFM Scharr gives the local sample-frame surface normal.\n"
        "Multiplying G with that sample normal expresses the surface normal in\n"
        "the crystal frame, so its folded direction is the local {hkl} surface index."
    )
    axes[1, 2].text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=7.4)
    axes[1, 2].axis("off")

    fig.suptitle("Kikuchi-indexed EBSD orientation coupled with AFM Scharr normal")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def orientation_delta_from_metadata(path: Path | None) -> tuple[float, float, float] | None:
    if path is None or not path.exists():
        return None
    metadata = read_json(path)
    value = metadata.get("orientation_residual_median_deg_used")
    if value is None:
        return None
    return tuple(float(x) for x in value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment = read_json(args.alignment_metadata)
    affine = np.asarray(alignment["afm_to_sem_affine_2x3"], dtype=np.float64)
    sem_shape = tuple(int(x) for x in alignment["sem_content_shape"])
    sem_image = robust_rescale(np.asarray(Image.open(alignment["sem_file"]), dtype=np.float32))[: sem_shape[0], : sem_shape[1]]
    ipf_image = np.asarray(Image.open(alignment["ipf_file"]).convert("RGB"), dtype=np.float32) / 255.0
    ipf_sem = resize_rgb(ipf_image, sem_shape)

    channels, afm_meta = read_afm_channels(args.afm)
    if args.height_channel not in channels:
        raise KeyError(f"AFM channel {args.height_channel!r} not found; available={list(channels)}")
    scan_size_um = float(afm_meta["scan_size_um"])
    height_um_raw = channels[args.height_channel].astype(np.float32) * 1e6

    normal_data = height_to_normals(
        height_um=height_um_raw,
        scan_size_um=scan_size_um,
        affine_afm_to_sem=affine,
        smooth_sigma_px=args.normal_smooth_sigma_px,
        level=args.plane_level,
    )
    orientation_delta = None if args.no_finetuned_orientation else orientation_delta_from_metadata(args.finetuned_ipf_metadata)
    ebsd = read_ebsd_map(args.h5, args.h5_group, orientation_delta)
    mapping = map_afm_to_ebsd(height_um_raw.shape, affine, sem_shape, ebsd)
    normals_sample = normal_data["normals_sample"]
    normals_crystal = crystal_normals_from_ebsd(normals_sample, ebsd, mapping["ebsd_index"])
    folded = fold_cubic_direction(normals_crystal)
    facet_rgb_map = facet_type_rgb(folded)
    normal_rgb = normal_direction_rgb(normals_sample, args.tilt_color_ref_deg)
    hkl_indices, hkl_angle_deg = nearest_hkl(folded)
    valid = mapping["valid"]

    paths = {
        "overview": args.output_dir / "afm_normals_surface_index_overview.png",
        "height_nm": args.output_dir / "afm_height_nm.png",
        "scharr_normalmap": args.output_dir / "afm_scharr_normalmap.png",
        "scharr_normalmap_colorbar": args.output_dir / "afm_scharr_normalmap_with_colorbar.png",
        "scharr_dz_dcol": args.output_dir / "afm_scharr_dz_dcol_um_per_um.png",
        "scharr_dz_drow": args.output_dir / "afm_scharr_dz_drow_um_per_um.png",
        "normal_tilt": args.output_dir / "afm_normal_tilt_deg.png",
        "normal_azimuth": args.output_dir / "afm_normal_azimuth_deg.png",
        "surface_index_afm": args.output_dir / "afm_crystal_surface_index_color.png",
        "top_view": args.output_dir / "ebsd_afm_surface_index_top_view.png",
        "ebsd_top_view": args.output_dir / "ebsd_ipf_top_view_sem_frame.png",
        "surface_index_top_view": args.output_dir / "surface_index_top_view_ebsd_frame.png",
        "normal_3d": args.output_dir / "afm_surface_normals_3d.png",
        "surface_index_3d": args.output_dir / "afm_surface_index_3d.png",
        "surface_index_3d_interactive": args.output_dir / "afm_surface_index_3d_interactive.html",
        "kikuchi_coupling": args.output_dir / "kikuchi_ebsd_afm_surface_index_coupling.png",
        "facet_legend": args.output_dir / "facet_type_color_key.png",
        "data_npz": args.output_dir / "afm_ebsd_surface_index_data.npz",
        "point_csv": args.output_dir / f"afm_ebsd_surface_index_point_cloud_stride{args.export_stride}.csv",
        "point_ply": args.output_dir / f"afm_ebsd_surface_index_point_cloud_stride{args.export_stride}.ply",
        "hkl_counts": args.output_dir / "nearest_hkl_counts.csv",
        "metadata": args.output_dir / "afm_ebsd_surface_index_metadata.json",
    }
    plt.imsave(paths["scharr_normalmap"], normal_rgb)
    save_scalar_image(paths["height_nm"], normal_data["height_um"] * 1000.0, "AFM height", "viridis", "height (nm)")
    save_scalar_image(
        paths["scharr_dz_dcol"],
        normal_data["scharr_dz_dcol"],
        "Scharr dz/dx from AFM depthmap",
        "coolwarm",
        "dz/dx (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 99.0)),
    )
    save_scalar_image(
        paths["scharr_dz_drow"],
        normal_data["scharr_dz_drow"],
        "Scharr dz/dy from AFM depthmap",
        "coolwarm",
        "dz/dy (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_drow"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_drow"], 99.0)),
    )
    save_normalmap_with_legend(
        paths["scharr_normalmap_colorbar"],
        normal_rgb,
        "AFM Scharr normalmap",
        args.tilt_color_ref_deg,
    )
    save_scalar_image(
        paths["normal_tilt"],
        normal_data["tilt_deg"],
        "Surface normal tilt from sample Z",
        "magma",
        "tilt (deg)",
        vmin=0.0,
        vmax=float(np.nanpercentile(normal_data["tilt_deg"], 99.0)),
    )
    save_scalar_image(
        paths["normal_azimuth"],
        normal_data["azimuth_deg"],
        "Surface normal azimuth in sample top-view frame",
        "twilight",
        "azimuth (deg)",
        vmin=-180.0,
        vmax=180.0,
    )
    save_rgb_image(paths["surface_index_afm"], facet_rgb_map, "Crystal-frame surface index color in AFM frame", valid)
    save_overview(paths["overview"], normal_data["height_um"], normal_rgb, normal_data["tilt_deg"], facet_rgb_map, valid)
    save_top_view(
        paths["top_view"],
        paths["ebsd_top_view"],
        paths["surface_index_top_view"],
        sem_image,
        ipf_sem,
        affine,
        facet_rgb_map,
        normal_rgb,
        valid,
    )
    save_3d_surface(paths["normal_3d"], normal_data["height_um"], normal_rgb, "AFM 3D surface colored by sample-frame normal direction", scan_size_um, args.plot_stride)
    save_3d_surface(paths["surface_index_3d"], normal_data["height_um"], facet_rgb_map, "AFM 3D surface colored by crystal-frame surface index", scan_size_um, args.plot_stride)
    save_interactive_surface_index(paths["surface_index_3d_interactive"], normal_data["height_um"], facet_rgb_map, valid, scan_size_um, max(args.plot_stride, 6))
    save_facet_legend(paths["facet_legend"])

    selected_row, selected_col = choose_coupling_point(mapping, ebsd, normal_data["tilt_deg"])
    selected_index = int(mapping["ebsd_index"][selected_row, selected_col])
    selected_phase = int(ebsd.phase[selected_index])
    selected = {
        "afm_row": selected_row,
        "afm_col": selected_col,
        "sem_xy": [float(mapping["sem_xy"][selected_row, selected_col, 0]), float(mapping["sem_xy"][selected_row, selected_col, 1])],
        "ebsd_row": int(mapping["ebsd_row"][selected_row, selected_col]),
        "ebsd_col": int(mapping["ebsd_col"][selected_row, selected_col]),
        "ebsd_index": selected_index,
        "iq": float(ebsd.iq[selected_index]),
        "ci": float(ebsd.ci[selected_index]),
        "phase": selected_phase,
        "normal_sample": [float(x) for x in normals_sample[selected_row, selected_col]],
        "normal_crystal": [float(x) for x in normals_crystal[selected_row, selected_col]],
        "folded": [float(x) for x in folded[selected_row, selected_col]],
        "nearest_hkl": HKL_LABELS[int(hkl_indices[selected_row, selected_col])],
        "nearest_hkl_angle_deg": float(hkl_angle_deg[selected_row, selected_col]),
        "phase_hkl_families": phase_hkl_families(args.h5, args.h5_group, selected_phase),
        "valid_mask": valid,
    }
    save_kikuchi_coupling_figure(
        paths["kikuchi_coupling"],
        args.h5,
        args.h5_group,
        args.up2,
        selected,
        sem_image,
        ipf_sem,
        normal_rgb,
        facet_rgb_map,
    )
    selected_for_metadata = {key: value for key, value in selected.items() if key != "valid_mask"}

    write_counts(paths["hkl_counts"], hkl_indices, hkl_angle_deg, valid)
    write_stride_csv(
        paths["point_csv"],
        normal_data["height_um"],
        normals_sample,
        normals_crystal,
        folded,
        facet_rgb_map,
        hkl_indices,
        hkl_angle_deg,
        mapping,
        ebsd,
        scan_size_um,
        args.export_stride,
    )
    write_ply(paths["point_ply"], normal_data["height_um"], facet_rgb_map, valid, scan_size_um, args.export_stride)
    np.savez_compressed(
        paths["data_npz"],
        height_um=normal_data["height_um"],
        height_smooth_um=normal_data["height_smooth_um"],
        normals_afm=normal_data["normals_afm"],
        normals_sample=normals_sample,
        normals_crystal=normals_crystal,
        scharr_dz_dcol=normal_data["scharr_dz_dcol"],
        scharr_dz_drow=normal_data["scharr_dz_drow"],
        folded_surface_index=folded,
        facet_rgb=facet_rgb_map,
        normal_direction_rgb=normal_rgb,
        nearest_hkl_index=hkl_indices,
        nearest_hkl_angle_deg=hkl_angle_deg,
        valid=valid,
        ebsd_row=mapping["ebsd_row"],
        ebsd_col=mapping["ebsd_col"],
        ebsd_index=mapping["ebsd_index"],
        sem_xy=mapping["sem_xy"],
    )
    metadata = {
        "method_note": (
            "AFM depthmap normals are extracted with the Scharr operator. The AFM->SEM affine contributes "
            "only its in-plane rotation to orient the normalmap in the EBSD/IPF top-view frame; affine scale "
            "and shear are not used in the depth gradient."
        ),
        "paper_method_reference": "Brüning et al. 2023, Journal of Microscopy: combine AFM surface normals with EBSD orientation to map facet types.",
        "afm": str(args.afm),
        "height_channel": args.height_channel,
        "scan_size_um": scan_size_um,
        "h5": str(args.h5),
        "h5_group": args.h5_group,
        "up2": str(args.up2),
        "alignment_metadata": str(args.alignment_metadata),
        "finetuned_orientation_delta_deg": orientation_delta,
        "normal_operator": "cv2.Scharr(depthmap, scale=1/32), then physical pitch normalization",
        "ohp_line_variant_for_kikuchi_check": OHP_LINE_VARIANT,
        "afm_to_sem_rotation_2d": np.asarray(normal_data["afm_to_sem_rotation_2d"]).tolist(),
        "plane_level": args.plane_level,
        "normal_smooth_sigma_px": args.normal_smooth_sigma_px,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "valid_fraction_of_afm": float(np.mean(valid)),
        "normal_pitch_x_um": normal_data["pitch_x_um"],
        "normal_pitch_y_um": normal_data["pitch_y_um"],
        "selected_kikuchi_coupling_point": selected_for_metadata,
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved AFM/EBSD surface-index outputs to {args.output_dir}")
    print(f"Valid AFM pixels with EBSD orientation: {metadata['valid_pixel_count']} ({metadata['valid_fraction_of_afm']:.3f})")
    if orientation_delta is not None:
        print(f"Applied finetuned orientation residual: {tuple(round(float(x), 4) for x in orientation_delta)} deg")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a registered AFM height field into surface normals, map them to Pt-3 EBSD orientations, "
            "and output crystal-frame surface-index data plus top-view EBSD overlays."
        )
    )
    parser.add_argument("--afm", type=Path, default=DEFAULT_AFM)
    parser.add_argument("--alignment-metadata", type=Path, default=DEFAULT_ALIGNMENT_METADATA)
    parser.add_argument("--finetuned-ipf-metadata", type=Path, default=DEFAULT_FINETUNED_IPF_METADATA)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--h5-group", default=DEFAULT_H5_GROUP)
    parser.add_argument("--up2", type=Path, default=DEFAULT_UP2)
    parser.add_argument("--height-channel", default="HeightRetrace")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normal-smooth-sigma-px", type=float, default=0.0)
    parser.add_argument("--tilt-color-ref-deg", type=float, default=35.0)
    parser.add_argument("--plot-stride", type=int, default=4)
    parser.add_argument("--export-stride", type=int, default=4)
    parser.add_argument("--plane-level", action="store_true")
    parser.add_argument("--no-finetuned-orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
