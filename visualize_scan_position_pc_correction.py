from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    preprocess_master_hemisphere,
    preprocess_pattern,
    project_patch_to_lon_colat,
    read_edax_inputs,
    sample_master,
)


H5_PATH = Path(r"F:\kikuchi-super resolution\20260512Cu resolution-contrast.edaxh5")
MASTER_PATH = Path(
    r"E:\EBSD-projiect\.venv\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
OUTPUT_DIR = Path(r"E:\EBSD-projiect\EBSD\outputs\scan_position_pc_correction")


@dataclass(frozen=True)
class MapConfig:
    name: str
    up2_path: Path
    map_group: str
    indices: tuple[int, ...]
    output_name: str


@dataclass(frozen=True)
class ScanGeometry:
    ncols: int
    nrows: int
    step_x_um: float
    step_y_um: float
    detector_diameter_mm: float
    pc_edax_map: tuple[float, float, float]
    grid_type: str


def _scalar(value) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return [_scalar(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_scan_geometry(h5_path: Path, map_group: str) -> ScanGeometry:
    with h5py.File(h5_path, "r") as h5:
        group = h5[map_group]
        calibration = group["EBSD/ANG/HEADER/Pattern Center Calibration"]
        pc_edax = (
            float(calibration["X-Star"][0]),
            float(calibration["Y-Star"][0]),
            float(calibration["Z-Star"][0]),
        )
        return ScanGeometry(
            ncols=int(group["Sample/Number Of Columns"][0]),
            nrows=int(group["Sample/Number Of Rows"][0]),
            step_x_um=float(group["Sample/Step X"][0]),
            step_y_um=float(group["Sample/Step Y"][0]),
            detector_diameter_mm=float(group["Camera/Diameter"][0]),
            pc_edax_map=pc_edax,
            grid_type=str(_scalar(group["Sample/Grid Type"][0])),
        )


def index_to_scan_offset_um(index: int, geometry: ScanGeometry) -> tuple[float, float]:
    row = index // geometry.ncols
    col = index % geometry.ncols

    # EDAX HexGrid rows have the same number of saved points here.  The half-step
    # row offset is kept relative to the map center so the average correction is
    # close to zero across the full map.
    row_parity = row % 2
    center_row_parity = ((geometry.nrows - 1) / 2) % 2
    hex_shift = 0.5 * (row_parity - center_row_parity) if "hex" in geometry.grid_type.lower() else 0.0

    x_um = (col - (geometry.ncols - 1) / 2 + hex_shift) * geometry.step_x_um
    y_um = (row - (geometry.nrows - 1) / 2) * geometry.step_y_um
    return float(x_um), float(y_um)


def adjusted_pc_from_scan_position(
    index: int,
    geometry: ScanGeometry,
    x_sign: float = 1.0,
    y_sign: float = 1.0,
    y_scale: float = 1.0,
    z_sign: float = 0.0,
    z_scale: float = 0.0,
) -> tuple[float, float, float]:
    x_um, y_um = index_to_scan_offset_um(index, geometry)
    dx = x_sign * (x_um / 1000.0) / geometry.detector_diameter_mm
    dy = y_sign * y_scale * (y_um / 1000.0) / geometry.detector_diameter_mm
    dz = z_sign * z_scale * (y_um / 1000.0) / geometry.detector_diameter_mm
    pcx, pcy, pcz = geometry.pc_edax_map
    return pcx + dx, pcy + dy, pcz + dz


def detector_directions_with_pc(projection, pc_edax: tuple[float, float, float]) -> np.ndarray:
    detector = EBSDDetector(
        shape=projection.shape,
        pc=pc_edax,
        convention="edax",
        tilt=projection.camera_elevation,
        azimuthal=projection.camera_azimuthal,
        sample_tilt=projection.sample_tilt,
    )
    return _get_direction_cosines_from_detector(detector)


def circular_mask(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def project_crystal_patch(projection, mask: np.ndarray, pc_edax: tuple[float, float, float]):
    corrected = preprocess_pattern(projection.pattern)
    detector_directions = detector_directions_with_pc(projection, pc_edax)
    indices = np.flatnonzero(mask.ravel())
    g_h5 = projection.orientation_flat.reshape(3, 3)
    crystal_vectors = detector_directions[indices] @ g_h5.T
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    return project_patch_to_lon_colat(
        crystal_vectors,
        corrected[mask],
        lon_count=720,
        colat_count=360,
    )[:2]


def build_master_texture() -> np.ndarray:
    upper, lower, _upper_sampler, _lower_sampler = load_master_samplers(MASTER_PATH)
    upper_sampler = make_master_sampler(preprocess_master_hemisphere(upper, "band"))
    lower_sampler = make_master_sampler(preprocess_master_hemisphere(lower, "band"))
    lon_count, colat_count = 720, 360
    lon = np.linspace(-np.pi, np.pi, lon_count)
    colat = np.linspace(0.0, np.pi, colat_count)
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    vectors = np.column_stack(
        [
            np.sin(colat_grid).ravel() * np.cos(lon_grid).ravel(),
            np.sin(colat_grid).ravel() * np.sin(lon_grid).ravel(),
            np.cos(colat_grid).ravel(),
        ]
    )
    return sample_master(vectors, upper_sampler, lower_sampler).reshape(colat_count, lon_count)


def format_pc(pc: tuple[float, float, float]) -> str:
    return f"({float(pc[0]):.5f}, {float(pc[1]):.5f}, {float(pc[2]):.5f})"


def summarize_pc_range(
    geometry: ScanGeometry,
    x_sign: float,
    y_sign: float,
    y_scale: float,
    z_sign: float,
    z_scale: float,
) -> str:
    corners = [0, geometry.ncols - 1, (geometry.nrows - 1) * geometry.ncols, geometry.nrows * geometry.ncols - 1]
    pcs = np.array(
        [adjusted_pc_from_scan_position(i, geometry, x_sign, y_sign, y_scale, z_sign, z_scale) for i in corners]
    )
    return (
        f"PC range from map corners, EDAX convention:\n"
        f"PCx {pcs[:, 0].min():.5f}..{pcs[:, 0].max():.5f}; "
        f"PCy {pcs[:, 1].min():.5f}..{pcs[:, 1].max():.5f}; "
        f"PCz {pcs[:, 2].min():.5f}..{pcs[:, 2].max():.5f}"
    )


def visualize_map(
    config: MapConfig,
    master_texture: np.ndarray,
    x_sign: float,
    y_sign: float,
    y_scale: float,
    z_sign: float = 0.0,
    z_scale: float = 0.0,
    model_name: str = "scan-adjusted",
) -> None:
    geometry = read_scan_geometry(H5_PATH, config.map_group)
    projections = [
        read_edax_inputs(
            EdaxMapInputs(
                h5_path=H5_PATH,
                up2_path=config.up2_path,
                map_group=config.map_group,
                pattern_index=index,
            )
        )
        for index in config.indices
    ]
    raw_circles = [estimate_circular_detector_mask(projection.pattern)[1] for projection in projections]
    fixed_circle = tuple(int(round(v)) for v in np.median(np.array(raw_circles), axis=0))

    n = len(config.indices)
    fig, axes = plt.subplots(n, 4, figsize=(19, 4.0 * n), squeeze=False)
    for row, (index, projection) in enumerate(zip(config.indices, projections)):
        mask = circular_mask(projection.pattern.shape, fixed_circle)
        fixed_patch, fixed_patch_mask = project_crystal_patch(projection, mask, geometry.pc_edax_map)
        pc_adjusted = adjusted_pc_from_scan_position(index, geometry, x_sign, y_sign, y_scale, z_sign, z_scale)
        adjusted_patch, adjusted_patch_mask = project_crystal_patch(projection, mask, pc_adjusted)
        x_um, y_um = index_to_scan_offset_um(index, geometry)

        ax = axes[row, 0]
        ax.imshow(projection.pattern, cmap="gray")
        ax.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
        ax.set_title(f"{config.name} index {index}\nscan offset=({x_um:.1f}, {y_um:.1f}) um")
        ax.axis("off")

        for ax, patch, patch_mask, title in [
            (
                axes[row, 1],
                fixed_patch,
                fixed_patch_mask,
                f"fixed map PC\n{format_pc(geometry.pc_edax_map)}",
            ),
            (
                axes[row, 2],
                adjusted_patch,
                adjusted_patch_mask,
                f"{model_name} PC\n{format_pc(pc_adjusted)}",
            ),
        ]:
            ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
            ax.imshow(
                patch,
                cmap="magma",
                origin="upper",
                extent=[-180, 180, 180, 0],
                aspect="auto",
                alpha=np.where(patch_mask, 0.9, 0.0),
            )
            ax.set_title(title)
            ax.set_xlabel("lon (deg)")
            ax.set_ylabel("colat (deg)")

        ax = axes[row, 3]
        overlay = np.zeros((*fixed_patch.shape, 4), dtype=np.float32)
        overlay[..., 0] = np.where(fixed_patch_mask, 1.0, 0.0)
        overlay[..., 2] = np.where(adjusted_patch_mask, 1.0, 0.0)
        overlay[..., 3] = np.where(fixed_patch_mask | adjusted_patch_mask, 0.75, 0.0)
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        ax.imshow(overlay, origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        ax.set_title("mask shift: fixed red, adjusted blue")
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("colat (deg)")

    fig.suptitle(
        f"{config.name}: per-pattern PC estimated from scan position; "
        f"x_sign={x_sign:g}, y_sign={y_sign:g}, y_scale={y_scale:g}, "
        f"z_sign={z_sign:g}, z_scale={z_scale:g}\n"
        f"{summarize_pc_range(geometry, x_sign, y_sign, y_scale, z_sign, z_scale)}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / config.output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    master_texture = build_master_texture()
    configs_area1 = [
        MapConfig(
            name="Area 1 HighR",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 1/OIM Map 1HighR",
            indices=(34, 14441, 28676, 29111),
            output_name="area1_highr_pcxyz_xneg_ypos_zpos_tilt70.png",
        ),
        MapConfig(
            name="Area 1 HighR representative map positions",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 1/OIM Map 1HighR",
            indices=(0, 217, 21255, 42292, 42509),
            output_name="area1_highr_representative_pcxyz_xneg_ypos_zpos_tilt70.png",
        ),
    ]
    configs_area2 = [
        MapConfig(
            name="Area 2 HighR",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
            indices=(19802, 21230, 22834, 19625),
            output_name="area2_highr_pcxyz_xneg_yneg_zpos_tilt_plus_elev78.png",
        ),
        MapConfig(
            name="Area 2 HighR representative map positions",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
            indices=(0, 177, 14151, 28124, 28301),
            output_name="area2_highr_representative_pcxyz_xneg_yneg_zpos_tilt_plus_elev78.png",
        ),
    ]
    for config in configs_area1:
        visualize_map(
            config,
            master_texture,
            x_sign=-1.0,
            y_sign=1.0,
            y_scale=np.cos(np.radians(70.0)),
            z_sign=1.0,
            z_scale=np.sin(np.radians(70.0)),
            model_name="PCXYZ x- y+ z+ tilt70",
        )
    for config in configs_area2:
        visualize_map(
            config,
            master_texture,
            x_sign=-1.0,
            y_sign=-1.0,
            y_scale=np.cos(np.radians(70.0 + 8.0)),
            z_sign=1.0,
            z_scale=np.sin(np.radians(70.0 + 8.0)),
            model_name="PCXYZ x- y- z+ tilt+elev78",
        )


if __name__ == "__main__":
    main()
