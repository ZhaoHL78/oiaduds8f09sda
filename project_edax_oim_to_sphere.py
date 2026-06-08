from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, transform

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)


@dataclass(frozen=True)
class EdaxMapInputs:
    h5_path: Path
    up2_path: Path
    map_group: str
    pattern_index: int


@dataclass(frozen=True)
class ProjectionInputs:
    pattern: np.ndarray
    detector_directions: np.ndarray
    orientation_flat: np.ndarray
    pc_edax: tuple[float, float, float]
    pc_internal: tuple[float, float, float]
    shape: tuple[int, int]
    map_name: str
    sem_kv: float
    sample_tilt: float
    camera_elevation: float
    camera_azimuthal: float


@dataclass(frozen=True)
class MasterHemisphereSampler:
    data: np.ndarray
    nrows: int
    ncols: int
    scale: float


def _scalar(value: np.ndarray | bytes | float | int | str) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return [_scalar(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, dict[str, int]]:
    with path.open("rb") as file:
        version = int(np.fromfile(file, np.uint32, 1)[0])
        sx, sy, pattern_offset = np.fromfile(file, np.uint32, 3)
        sx = int(sx)
        sy = int(sy)
        pattern_offset = int(pattern_offset)
        bytes_per_pattern = sx * sy * np.dtype(np.uint16).itemsize
        file.seek(pattern_offset + index * bytes_per_pattern)
        pattern = np.fromfile(file, np.uint16, sx * sy).reshape(sy, sx)

    return pattern, {
        "version": version,
        "sx": sx,
        "sy": sy,
        "pattern_offset": pattern_offset,
    }


def read_edax_inputs(inputs: EdaxMapInputs) -> ProjectionInputs:
    pattern, up2_header = read_up2_pattern(inputs.up2_path, inputs.pattern_index)

    with h5py.File(inputs.h5_path, "r") as h5:
        group = h5[inputs.map_group]
        calibration = group["EBSD/ANG/HEADER/Pattern Center Calibration"]
        x_star = float(calibration["X-Star"][0])
        y_star = float(calibration["Y-Star"][0])
        z_star = float(calibration["Z-Star"][0])
        pc_edax = (x_star, y_star, z_star)

        data = group["EBSD/ANG/DATA/DATA"]
        orientation_flat = data["Orientations"][inputs.pattern_index].astype(np.float64)

        sem_kv = float(group["Column/SemKV"][0])
        sample_tilt = float(group["Sample/Sample Tilt"][0])
        camera_elevation = float(group["Camera/Elevation Angle"][0])
        camera_azimuthal = float(group["Camera/Azimuthal Angle"][0])
        map_name = str(_scalar(group.attrs.get("Name", inputs.map_group)))

    shape = (up2_header["sy"], up2_header["sx"])
    detector = EBSDDetector(
        shape=shape,
        pc=pc_edax,
        convention="edax",
        tilt=camera_elevation,
        azimuthal=camera_azimuthal,
        sample_tilt=sample_tilt,
    )
    detector_directions = _get_direction_cosines_from_detector(detector)
    pc_internal = tuple(float(v) for v in detector.pc.squeeze())

    return ProjectionInputs(
        pattern=pattern,
        detector_directions=detector_directions,
        orientation_flat=orientation_flat,
        pc_edax=pc_edax,
        pc_internal=pc_internal,
        shape=shape,
        map_name=map_name,
        sem_kv=sem_kv,
        sample_tilt=sample_tilt,
        camera_elevation=camera_elevation,
        camera_azimuthal=camera_azimuthal,
    )


def preprocess_pattern(pattern: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    image = exposure.rescale_intensity(
        pattern.astype(np.float32), in_range="image", out_range=(0.0, 1.0)
    )
    corrected = image - filters.gaussian(image, sigma=sigma, preserve_range=True)
    return exposure.rescale_intensity(corrected, in_range="image", out_range=(0.0, 1.0))


def estimate_circular_detector_mask(pattern: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    image = exposure.rescale_intensity(
        pattern.astype(np.float32), in_range="image", out_range=(0.0, 1.0)
    )
    smooth = filters.gaussian(image, sigma=2.0, preserve_range=True)
    edges = feature.canny(smooth, sigma=3.0, low_threshold=0.02, high_threshold=0.08)

    h, w = pattern.shape
    # The phosphor screen occupies almost the full saved square raster.
    # A wider radius range can let strong Kikuchi bands win the Hough vote,
    # so keep the search focused on the outer detector rim.
    min_radius = int(0.48 * min(h, w))
    max_radius = int(0.525 * min(h, w))
    radii = np.arange(min_radius, max_radius + 1, 2)
    hough = transform.hough_circle(edges, radii)
    accums, center_x, center_y, radius = transform.hough_circle_peaks(
        hough, radii, total_num_peaks=8
    )
    if len(accums) == 0:
        center_x = np.array([w // 2])
        center_y = np.array([h // 2])
        radius = np.array([int(0.48 * min(h, w))])

    center_target = np.array([w / 2, h / 2])
    radius_target = 0.5 * min(h, w)
    penalties = (
        0.9 * np.hypot(center_x - center_target[0], center_y - center_target[1]) / min(h, w)
        + 0.4 * np.abs(radius - radius_target) / min(h, w)
        - accums
    )
    best = int(np.argmin(penalties))
    cx = int(center_x[best])
    cy = int(center_y[best])
    r = int(radius[best])
    yy, xx = np.indices(pattern.shape)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    return mask, (cx, cy, r)


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (values - values.mean()) / (values.std() + 1e-8)


def _as_last_2d_image(dataset: h5py.Dataset) -> np.ndarray:
    raw = dataset[()]
    while raw.ndim > 2:
        # kikuchipy uses the highest energy by default if an energy is
        # not explicitly requested.  For single-energy files this is a no-op.
        raw = raw[-1]

    if np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype(np.float32) / np.iinfo(raw.dtype).max
    else:
        raw = raw.astype(np.float32)
    return raw


def make_master_sampler(data: np.ndarray) -> MasterHemisphereSampler:
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(
            f"Expected a square Lambert hemisphere image, got shape {data.shape}"
        )
    nrows, ncols = data.shape
    return MasterHemisphereSampler(
        data=data.astype(np.float32, copy=False),
        nrows=nrows,
        ncols=ncols,
        scale=float((ncols - 1) / 2),
    )


def preprocess_master_hemisphere(data: np.ndarray, mode: str = "corrected") -> np.ndarray:
    if mode == "raw":
        return data.astype(np.float32, copy=False)

    corrected = exposure.rescale_intensity(
        data - filters.gaussian(data, sigma=9.0, preserve_range=True),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    if mode == "corrected":
        return corrected.astype(np.float32, copy=False)
    if mode == "band":
        return exposure.rescale_intensity(
            filters.meijering(corrected, sigmas=range(1, 6), black_ridges=False),
            in_range="image",
            out_range=(0.0, 1.0),
        ).astype(np.float32, copy=False)
    raise ValueError("mode must be one of: raw, corrected, band")


def load_master_samplers(master_h5_path: Path):
    with h5py.File(master_h5_path, "r") as h5:
        master_group = h5["EMData/EBSDmaster"]
        upper = _as_last_2d_image(master_group["mLPNH"])
        lower = _as_last_2d_image(master_group["mLPSH"])

    return upper, lower, make_master_sampler(upper), make_master_sampler(lower)


def _sample_lambert_hemisphere(
    vectors: np.ndarray,
    sampler: MasterHemisphereSampler,
) -> np.ndarray:
    if len(vectors) == 0:
        return np.empty(0, dtype=np.float32)

    (
        row,
        col,
        row_next,
        col_next,
        row_weight,
        col_weight,
        row_weight_inv,
        col_weight_inv,
    ) = _get_lambert_interpolation_parameters(
        v=vectors.astype(np.float64, copy=False),
        npx=sampler.nrows,
        npy=sampler.ncols,
        scale=sampler.scale,
    )

    data = sampler.data
    row = np.clip(row, 0, sampler.nrows - 1)
    col = np.clip(col, 0, sampler.ncols - 1)
    row_next = np.clip(row_next, 0, sampler.nrows - 1)
    col_next = np.clip(col_next, 0, sampler.ncols - 1)
    sampled = (
        data[row, col] * row_weight_inv * col_weight_inv
        + data[row_next, col] * row_weight * col_weight_inv
        + data[row, col_next] * row_weight_inv * col_weight
        + data[row_next, col_next] * row_weight * col_weight
    )
    return sampled.astype(np.float32, copy=False)


def sample_master(
    vectors: np.ndarray,
    upper_interp: MasterHemisphereSampler,
    lower_interp: MasterHemisphereSampler,
) -> np.ndarray:
    z = vectors[:, 2]
    sampled = np.zeros(len(vectors), dtype=np.float32)
    upper = z >= 0
    lower = ~upper

    if np.any(upper):
        sampled[upper] = _sample_lambert_hemisphere(vectors[upper], upper_interp)

    if np.any(lower):
        sampled[lower] = _sample_lambert_hemisphere(vectors[lower], lower_interp)

    return sampled


def orientation_candidates(orientation_flat: np.ndarray) -> dict[str, np.ndarray]:
    row_major = orientation_flat.reshape(3, 3)
    col_major = orientation_flat.reshape(3, 3, order="F")
    return {
        "edax_g_inverse_row_major": row_major,
        "edax_g_direct_row_major": row_major.T,
        "edax_g_inverse_col_major": col_major,
        "edax_g_direct_col_major": col_major.T,
    }


def score_orientation_candidates(
    projection: ProjectionInputs,
    upper_interp: MasterHemisphereSampler,
    lower_interp: MasterHemisphereSampler,
    detector_mask: np.ndarray,
    stride: int = 4,
) -> tuple[str, dict[str, float]]:
    image = preprocess_pattern(projection.pattern)
    stride_mask = np.zeros(detector_mask.shape, dtype=bool)
    stride_mask[::stride, ::stride] = True
    flat_indices = np.flatnonzero(detector_mask & stride_mask).astype(np.int64)
    experimental = zscore(image.ravel()[flat_indices])

    scores: dict[str, float] = {}
    for name, matrix in orientation_candidates(projection.orientation_flat).items():
        crystal_vectors = projection.detector_directions[flat_indices] @ matrix
        master_values = sample_master(crystal_vectors, upper_interp, lower_interp)
        scores[name] = float(np.mean(experimental * zscore(master_values)))

    best_name = max(scores, key=scores.get)
    return best_name, scores


def project_patch_to_lon_colat(
    crystal_vectors: np.ndarray,
    values: np.ndarray,
    lon_count: int = 720,
    colat_count: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.arctan2(crystal_vectors[:, 1], crystal_vectors[:, 0])
    phi = np.arccos(np.clip(crystal_vectors[:, 2], -1.0, 1.0))
    u = np.clip(
        np.round((theta + np.pi) / (2 * np.pi) * (lon_count - 1)).astype(int),
        0,
        lon_count - 1,
    )
    v = np.clip(
        np.round(phi / np.pi * (colat_count - 1)).astype(int),
        0,
        colat_count - 1,
    )

    projection_sum = np.zeros((colat_count, lon_count), dtype=np.float32)
    projection_count = np.zeros((colat_count, lon_count), dtype=np.float32)
    np.add.at(projection_sum, (v, u), values.astype(np.float32))
    np.add.at(projection_count, (v, u), 1.0)

    projected = np.zeros_like(projection_sum)
    mask = projection_count > 0
    projected[mask] = projection_sum[mask] / projection_count[mask]
    return projected, mask, np.vstack([theta, phi]).T


def build_master_lon_colat(
    upper_interp: MasterHemisphereSampler,
    lower_interp: MasterHemisphereSampler,
    lon_count: int = 720,
    colat_count: int = 360,
) -> np.ndarray:
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
    return sample_master(vectors, upper_interp, lower_interp).reshape(colat_count, lon_count)


def save_visualization(
    projection: ProjectionInputs,
    master_texture: np.ndarray,
    patch_projection: np.ndarray,
    patch_mask: np.ndarray,
    crystal_vectors: np.ndarray,
    detector_mask: np.ndarray,
    circle: tuple[int, int, int],
    candidate_name: str,
    candidate_scores: dict[str, float],
    output_path: Path,
) -> None:
    corrected = preprocess_pattern(projection.pattern)
    overlay_alpha = np.where(patch_mask, 0.92, 0.0)

    step = max(len(crystal_vectors) // 15000, 1)
    sample = crystal_vectors[::step]
    sample_values = corrected[detector_mask][::step]

    fig = plt.figure(figsize=(15.5, 10.0))
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.imshow(projection.pattern, cmap="gray")
    ax0.contour(detector_mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
    ax0.set_title("Raw EDAX UP2 pattern")
    ax0.axis("off")

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.imshow(corrected, cmap="gray")
    ax1.set_title("Background-corrected pattern")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.imshow(
        master_texture,
        cmap="gray",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
    )
    ax2.imshow(
        patch_projection,
        cmap="magma",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=overlay_alpha,
    )
    ax2.set_title("Experimental patch placed on master sphere")
    ax2.set_xlabel("Longitude on crystal sphere (deg)")
    ax2.set_ylabel("Colatitude (deg)")

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.scatter(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample_values,
        cmap="gray",
        s=1.0,
        depthshade=False,
    )
    ax3.set_title("Same patch as crystal-frame unit vectors")
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.set_axis_off()

    score_lines = "\n".join(
        f"{name}: {value:+.4f}" for name, value in sorted(candidate_scores.items())
    )
    fig.text(
        0.02,
        0.015,
        (
            f"Map: {projection.map_name}; PC EDAX={projection.pc_edax}; "
            f"PC internal={tuple(round(v, 6) for v in projection.pc_internal)}; circular mask=(cx={circle[0]}, cy={circle[1]}, r={circle[2]});\n"
            f"matrix interpretation={candidate_name}; "
            f"SEM={projection.sem_kv:g} kV; sample tilt={projection.sample_tilt:g} deg; "
            f"camera elevation={projection.camera_elevation:g} deg; scores: {score_lines}"
        ),
        fontsize=8,
        family="monospace",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(inputs: EdaxMapInputs, master_h5_path: Path, output_path: Path) -> None:
    projection = read_edax_inputs(inputs)
    _upper, _lower, upper_interp, lower_interp = load_master_samplers(master_h5_path)
    detector_mask, circle = estimate_circular_detector_mask(projection.pattern)

    best_name, scores = score_orientation_candidates(
        projection, upper_interp, lower_interp, detector_mask
    )
    matrix = orientation_candidates(projection.orientation_flat)[best_name]
    detector_indices = np.flatnonzero(detector_mask.ravel())
    crystal_vectors = projection.detector_directions[detector_indices] @ matrix
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12

    corrected = preprocess_pattern(projection.pattern)
    patch_projection, patch_mask, _ = project_patch_to_lon_colat(
        crystal_vectors, corrected[detector_mask]
    )
    master_texture = build_master_lon_colat(upper_interp, lower_interp)

    save_visualization(
        projection=projection,
        master_texture=master_texture,
        patch_projection=patch_projection,
        patch_mask=patch_mask,
        crystal_vectors=crystal_vectors,
        detector_mask=detector_mask,
        circle=circle,
        candidate_name=best_name,
        candidate_scores=scores,
        output_path=output_path,
    )

    print(f"Saved visualization: {output_path}")
    print(f"EDAX PC: {projection.pc_edax}")
    print(f"Internal PC used for detector geometry: {projection.pc_internal}")
    print(f"Circular detector mask: center=({circle[0]}, {circle[1]}), radius={circle[2]}, pixels={int(detector_mask.sum())}")
    print(f"Best orientation interpretation: {best_name}")
    for name, value in sorted(scores.items()):
        print(f"  {name}: {value:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project an EDAX UP2 pattern onto a master Kikuchi sphere using EDAX PC and OIM orientation."
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path(r"D:\project\EBSD2026\ebsd.edaxh5"),
    )
    parser.add_argument(
        "--up2",
        type=Path,
        default=Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2"),
    )
    parser.add_argument(
        "--map-group",
        default="/20260512/Cu/Area 1/OIM Map 1HighR",
    )
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path(
            r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"outputs\github_edax_visualizations\edax_area1_index0_software_projection.png"),
    )
    args = parser.parse_args()

    run(
        inputs=EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        ),
        master_h5_path=args.master,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
