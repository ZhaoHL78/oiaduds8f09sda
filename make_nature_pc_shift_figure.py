from __future__ import annotations

import csv
from dataclasses import dataclass
from math import acos, cos, degrees, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    ProjectionInputs,
    build_master_lon_colat,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    preprocess_master_hemisphere,
    preprocess_pattern,
    project_patch_to_lon_colat,
    read_edax_inputs,
)
from visualize_scan_position_pc_correction import (
    ScanGeometry,
    adjusted_pc_from_scan_position,
    index_to_scan_offset_um,
    read_scan_geometry,
)


H5_PATH = Path(r"F:\kikuchi-super resolution\20260512Cu resolution-contrast.edaxh5")
UP2_PATH = Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2")
MAP_GROUP = "/20260512/Cu/Area 1/OIM Map 1HighR"
MASTER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".venv"
    / "Lib"
    / "site-packages"
    / "kikuchipy"
    / "data"
    / "emsoft_ebsd_master_pattern"
    / "ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
OUTPUT_DIR = Path("outputs") / "nature_pc_shift_visualization"
INDICES = (34, 14441, 28676, 29111)
PANEL_IDS = ("i", "ii", "iii", "iv")

# Okabe-Ito inspired colors, chosen to survive grayscale and common color vision deficiencies.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
FIXED_COLOR = "#6f6f6f"


@dataclass(frozen=True)
class PatternView:
    panel_id: str
    index: int
    row: int
    col: int
    x_um: float
    y_um: float
    projection: ProjectionInputs
    corrected: np.ndarray
    mask: np.ndarray
    fixed_pc: tuple[float, float, float]
    adjusted_pc: tuple[float, float, float]
    fixed_vectors: np.ndarray
    adjusted_vectors: np.ndarray
    fixed_patch: np.ndarray
    fixed_patch_mask: np.ndarray
    adjusted_patch: np.ndarray
    adjusted_patch_mask: np.ndarray
    fixed_centroid: np.ndarray
    adjusted_centroid: np.ndarray
    sphere_shift_deg: float


def normalize01(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    if mask is not None and np.any(mask):
        lo, hi = np.nanpercentile(values[mask], [1.0, 99.0])
    else:
        lo, hi = np.nanpercentile(values, [1.0, 99.0])
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def circular_mask(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def detector_directions_with_pc(
    projection: ProjectionInputs,
    pc_edax: tuple[float, float, float],
) -> np.ndarray:
    detector = EBSDDetector(
        shape=projection.shape,
        pc=pc_edax,
        convention="edax",
        tilt=projection.camera_elevation,
        azimuthal=projection.camera_azimuthal,
        sample_tilt=projection.sample_tilt,
    )
    return _get_direction_cosines_from_detector(detector)


def crystal_vectors_for_pc(
    projection: ProjectionInputs,
    mask: np.ndarray,
    pc_edax: tuple[float, float, float],
) -> np.ndarray:
    directions = detector_directions_with_pc(projection, pc_edax)
    matrix = projection.orientation_flat.reshape(3, 3).T
    vectors = directions[np.flatnonzero(mask.ravel())] @ matrix
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def mean_direction(vectors: np.ndarray) -> np.ndarray:
    centroid = np.mean(vectors, axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    return degrees(acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def direction_to_lon_colat(direction: np.ndarray) -> tuple[float, float]:
    lon = degrees(np.arctan2(float(direction[1]), float(direction[0])))
    colat = degrees(np.arccos(np.clip(float(direction[2]), -1.0, 1.0)))
    return lon, colat


def read_pattern_row_col(index: int, geometry: ScanGeometry) -> tuple[int, int]:
    return int(index // geometry.ncols), int(index % geometry.ncols)


def build_master_texture() -> np.ndarray:
    upper, lower, _upper_raw, _lower_raw = load_master_samplers(MASTER_PATH)
    upper_band = make_master_sampler(preprocess_master_hemisphere(upper, "band"))
    lower_band = make_master_sampler(preprocess_master_hemisphere(lower, "band"))
    master = build_master_lon_colat(upper_band, lower_band, lon_count=720, colat_count=360)
    master = normalize01(master)
    # Dark Kikuchi bands on a light background reproduce better in print than a black field.
    return np.clip(1.0 - master**0.72, 0.10, 1.0)


def collect_views() -> tuple[ScanGeometry, tuple[int, int, int], list[PatternView]]:
    geometry = read_scan_geometry(H5_PATH, MAP_GROUP)
    projections = [
        read_edax_inputs(
            EdaxMapInputs(
                h5_path=H5_PATH,
                up2_path=UP2_PATH,
                map_group=MAP_GROUP,
                pattern_index=index,
            )
        )
        for index in INDICES
    ]
    circles = [estimate_circular_detector_mask(projection.pattern)[1] for projection in projections]
    fixed_circle = tuple(int(round(v)) for v in np.median(np.asarray(circles), axis=0))

    views: list[PatternView] = []
    for panel_id, index, projection in zip(PANEL_IDS, INDICES, projections):
        mask = circular_mask(projection.shape, fixed_circle)
        corrected = preprocess_pattern(projection.pattern)
        fixed_pc = geometry.pc_edax_map
        adjusted_pc = adjusted_pc_from_scan_position(
            index,
            geometry,
            x_sign=-1.0,
            y_sign=1.0,
            y_scale=cos(radians(70.0)),
            z_sign=1.0,
            z_scale=sin(radians(70.0)),
        )
        fixed_vectors = crystal_vectors_for_pc(projection, mask, fixed_pc)
        adjusted_vectors = crystal_vectors_for_pc(projection, mask, adjusted_pc)
        values = corrected[mask]
        fixed_patch, fixed_patch_mask, _ = project_patch_to_lon_colat(fixed_vectors, values)
        adjusted_patch, adjusted_patch_mask, _ = project_patch_to_lon_colat(adjusted_vectors, values)
        fixed_centroid = mean_direction(fixed_vectors)
        adjusted_centroid = mean_direction(adjusted_vectors)
        x_um, y_um = index_to_scan_offset_um(index, geometry)
        row, col = read_pattern_row_col(index, geometry)
        views.append(
            PatternView(
                panel_id=panel_id,
                index=index,
                row=row,
                col=col,
                x_um=x_um,
                y_um=y_um,
                projection=projection,
                corrected=corrected,
                mask=mask,
                fixed_pc=fixed_pc,
                adjusted_pc=adjusted_pc,
                fixed_vectors=fixed_vectors,
                adjusted_vectors=adjusted_vectors,
                fixed_patch=fixed_patch,
                fixed_patch_mask=fixed_patch_mask,
                adjusted_patch=adjusted_patch,
                adjusted_patch_mask=adjusted_patch_mask,
                fixed_centroid=fixed_centroid,
                adjusted_centroid=adjusted_centroid,
                sphere_shift_deg=angular_distance_deg(fixed_centroid, adjusted_centroid),
            )
        )
    return geometry, fixed_circle, views


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def draw_scan_position_panel(ax, geometry: ScanGeometry, views: list[PatternView]) -> None:
    half_width = (geometry.ncols - 1) * geometry.step_x_um / 2
    half_height = (geometry.nrows - 1) * geometry.step_y_um / 2
    ax.add_patch(
        plt.Rectangle(
            (-half_width, -half_height),
            2 * half_width,
            2 * half_height,
            fill=False,
            lw=0.75,
            ec="#b6b6b6",
        )
    )
    label_offsets = {
        "i": (-62, -70, "right"),
        "ii": (34, 34, "left"),
        "iii": (78, 62, "left"),
        "iv": (-78, 62, "right"),
    }
    for view, color in zip(views, COLORS):
        ax.scatter(view.x_um, view.y_um, s=34, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        dx, dy, ha = label_offsets[view.panel_id]
        ax.text(view.x_um + dx, view.y_um + dy, view.panel_id, color=color, fontsize=7.5, weight="bold", ha=ha)
    ax.plot(
        [-half_width + 90, -half_width + 590],
        [-half_height + 90, -half_height + 90],
        color="#222222",
        lw=1.2,
    )
    ax.text(-half_width + 340, -half_height + 125, "500 um", ha="center", va="bottom", fontsize=6.8)
    ax.set_xlim(-half_width - 80, half_width + 80)
    ax.set_ylim(-half_height - 80, half_height + 80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "a")
    ax.text(0.0, 1.02, "scan positions", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5)


def draw_raw_patterns(raw_axes, views: list[PatternView], circle: tuple[int, int, int]) -> None:
    for ax, view, color in zip(raw_axes, views, COLORS):
        image = normalize01(view.projection.pattern, view.mask)
        rgba = np.dstack([image, image, image, view.mask.astype(float)])
        ax.imshow(rgba)
        cx, cy, radius = circle
        ax.add_patch(plt.Circle((cx, cy), radius, fill=False, ec=color, lw=0.7))
        ax.text(0.02, 0.98, view.panel_id, transform=ax.transAxes, color=color, ha="left", va="top", fontsize=8.0, weight="bold")
        ax.text(
            0.02,
            -0.05,
            f"{view.index}\n({view.x_um:+.0f}, {view.y_um:+.0f}) um",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            color="#333333",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    add_panel_label(raw_axes[0], "b", x=-0.10, y=1.14)
    raw_axes[0].text(0.02, 1.14, "experimental Kikuchi patterns", transform=raw_axes[0].transAxes, ha="left", va="bottom", fontsize=7.5)


def draw_sphere_panel(ax, master_texture: np.ndarray, views: list[PatternView]) -> None:
    ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    lon = np.linspace(-180, 180, master_texture.shape[1])
    colat = np.linspace(0, 180, master_texture.shape[0])
    patch_lon = np.linspace(-180, 180, views[0].adjusted_patch_mask.shape[1])
    patch_colat = np.linspace(0, 180, views[0].adjusted_patch_mask.shape[0])

    for view, color in zip(views, COLORS):
        ax.contour(
            patch_lon,
            patch_colat,
            view.fixed_patch_mask.astype(float),
            levels=[0.5],
            colors=[FIXED_COLOR],
            linewidths=0.55,
            linestyles="--",
            alpha=0.72,
        )
        ax.contourf(
            patch_lon,
            patch_colat,
            view.adjusted_patch_mask.astype(float),
            levels=[0.5, 1.5],
            colors=[color],
            alpha=0.12,
        )
        ax.contour(
            patch_lon,
            patch_colat,
            view.adjusted_patch_mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=0.8,
        )

        lon0, colat0 = direction_to_lon_colat(view.fixed_centroid)
        lon1, colat1 = direction_to_lon_colat(view.adjusted_centroid)
        if lon1 - lon0 > 180:
            lon1 -= 360
        elif lon1 - lon0 < -180:
            lon1 += 360
        ax.annotate(
            "",
            xy=(lon1, colat1),
            xytext=(lon0, colat0),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, shrinkA=0, shrinkB=0, mutation_scale=7),
            zorder=7,
        )
        ax.scatter([lon1], [colat1], s=12, color=color, edgecolor="white", linewidth=0.35, zorder=8)
        ax.text(lon1 + 2.5, colat1 - 2.5, view.panel_id, color=color, fontsize=7.0, weight="bold", zorder=8)

    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    ax.set_xlabel("longitude on master sphere (deg)", fontsize=7.2)
    ax.set_ylabel("colatitude (deg)", fontsize=7.2)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    legend = [
        Line2D([0], [0], color=FIXED_COLOR, lw=0.75, ls="--", label="fixed map PC"),
        Line2D([0], [0], color="#222222", lw=0.9, label="scan-position PC"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=6.8, handlelength=2.4)
    add_panel_label(ax, "c", x=-0.055, y=1.05)
    ax.text(0.0, 1.02, "spherical footprints after PC correction", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5)


def draw_shift_panel(ax, views: list[PatternView]) -> None:
    y = np.arange(len(views))
    shifts = [view.sphere_shift_deg for view in views]
    ax.barh(y, shifts, color=COLORS, height=0.58)
    ax.set_xlim(0.0, max(shifts) * 1.32)
    ax.set_yticks(y, [view.panel_id for view in views])
    ax.invert_yaxis()
    ax.set_xlabel("centroid shift (deg)", fontsize=7.0)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.55)
    for item, shift in zip(y, shifts):
        ax.text(shift + 0.08, item, f"{shift:.2f}", va="center", fontsize=6.4)
    add_panel_label(ax, "d", x=-0.16, y=1.10)
    ax.text(0.02, 1.10, "sphere displacement", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5)


def write_summary(views: list[PatternView], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "panel_id",
                "index",
                "row",
                "col",
                "x_um",
                "y_um",
                "fixed_pcx",
                "fixed_pcy",
                "fixed_pcz",
                "adjusted_pcx",
                "adjusted_pcy",
                "adjusted_pcz",
                "sphere_shift_deg",
            ],
        )
        writer.writeheader()
        for view in views:
            writer.writerow(
                {
                    "panel_id": view.panel_id,
                    "index": view.index,
                    "row": view.row,
                    "col": view.col,
                    "x_um": view.x_um,
                    "y_um": view.y_um,
                    "fixed_pcx": view.fixed_pc[0],
                    "fixed_pcy": view.fixed_pc[1],
                    "fixed_pcz": view.fixed_pc[2],
                    "adjusted_pcx": view.adjusted_pc[0],
                    "adjusted_pcy": view.adjusted_pc[1],
                    "adjusted_pcz": view.adjusted_pc[2],
                    "sphere_shift_deg": view.sphere_shift_deg,
                }
            )


def make_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    geometry, circle, views = collect_views()
    master_texture = build_master_texture()

    fig = plt.figure(figsize=(7.2, 5.55), facecolor="white")
    outer = fig.add_gridspec(
        2,
        4,
        height_ratios=(1.18, 2.15),
        width_ratios=(1.0, 1.0, 1.0, 0.72),
        left=0.055,
        right=0.988,
        bottom=0.075,
        top=0.965,
        hspace=0.30,
        wspace=0.26,
    )

    ax_scan = fig.add_subplot(outer[0, 0])
    draw_scan_position_panel(ax_scan, geometry, views)

    raw_grid = outer[0, 1:4].subgridspec(1, 4, wspace=0.18)
    raw_axes = [fig.add_subplot(raw_grid[0, i]) for i in range(4)]
    draw_raw_patterns(raw_axes, views, circle)

    ax_sphere = fig.add_subplot(outer[1, 0:3])
    draw_sphere_panel(ax_sphere, master_texture, views)

    ax_shift = fig.add_subplot(outer[1, 3])
    draw_shift_panel(ax_shift, views)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "area1_pc_shift_nature_layout.png"
    pdf_path = OUTPUT_DIR / "area1_pc_shift_nature_layout.pdf"
    svg_path = OUTPUT_DIR / "area1_pc_shift_nature_layout.svg"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    write_summary(views, OUTPUT_DIR / "area1_pc_shift_nature_layout_summary.csv")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    make_figure()
