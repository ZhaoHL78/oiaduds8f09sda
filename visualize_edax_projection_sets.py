from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from diagnose_edax_transform_chain import detector_local_vectors
from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    preprocess_pattern,
    preprocess_master_hemisphere,
    project_patch_to_lon_colat,
    read_edax_inputs,
    sample_master,
)


@dataclass(frozen=True)
class SampleSet:
    name: str
    h5_path: Path
    up2_path: Path
    map_group: str
    pattern_indices: tuple[int, ...]
    output_path: Path


H5_PATH = Path(r"D:\project\EBSD2026\ebsd.edaxh5")
MASTER_PATH = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
OUTPUT_DIR = Path(r"outputs\github_edax_visualizations\edax_projection_sets")


def build_master_texture(upper_interp, lower_interp, lon_count: int = 720, colat_count: int = 360) -> np.ndarray:
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


def detector_directions_for_projection(projection) -> np.ndarray:
    detector = EBSDDetector(
        shape=projection.shape,
        pc=projection.pc_edax,
        convention="edax",
        tilt=projection.camera_elevation,
        azimuthal=projection.camera_azimuthal,
        sample_tilt=projection.sample_tilt,
    )
    return _get_direction_cosines_from_detector(detector)


def project_detector_only(projection, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrected = preprocess_pattern(projection.pattern)
    local_vectors = detector_local_vectors(projection.shape, projection.pc_internal)
    patch, patch_mask, _ = project_patch_to_lon_colat(
        local_vectors[mask.ravel()],
        corrected[mask],
        lon_count=360,
        colat_count=180,
    )
    return patch, patch_mask


def project_crystal_position(projection, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrected = preprocess_pattern(projection.pattern)
    detector_directions = detector_directions_for_projection(projection)
    g_h5 = projection.orientation_flat.reshape(3, 3)

    # HDF5 Orientations behave as sample/lab -> crystal in this data.
    # With row vectors, column-vector v_crystal = G_h5 @ v_sample is v_sample_row @ G_h5.T.
    indices = np.flatnonzero(mask.ravel())
    crystal_vectors = detector_directions[indices] @ g_h5.T
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    patch, patch_mask, _ = project_patch_to_lon_colat(
        crystal_vectors,
        corrected[mask],
        lon_count=720,
        colat_count=360,
    )
    return patch, patch_mask


def circular_mask(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def visualize_sample_set(sample_set: SampleSet, master_texture: np.ndarray) -> None:
    n = len(sample_set.pattern_indices)
    fig, axes = plt.subplots(n, 3, figsize=(15.5, 4.2 * n), squeeze=False)
    projections = [
        read_edax_inputs(
            EdaxMapInputs(
                h5_path=sample_set.h5_path,
                up2_path=sample_set.up2_path,
                map_group=sample_set.map_group,
                pattern_index=pattern_index,
            )
        )
        for pattern_index in sample_set.pattern_indices
    ]
    raw_circles = [estimate_circular_detector_mask(projection.pattern)[1] for projection in projections]
    fixed_circle = tuple(int(round(v)) for v in np.median(np.array(raw_circles), axis=0))

    for row, (pattern_index, projection) in enumerate(zip(sample_set.pattern_indices, projections)):
        mask = circular_mask(projection.pattern.shape, fixed_circle)
        corrected = preprocess_pattern(projection.pattern)
        detector_patch, detector_patch_mask = project_detector_only(projection, mask)
        crystal_patch, crystal_patch_mask = project_crystal_position(projection, mask)

        ax = axes[row, 0]
        ax.imshow(projection.pattern, cmap="gray")
        ax.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
        ax.set_title(
            f"{sample_set.name} index {pattern_index}\n"
            f"fixed mask cx={fixed_circle[0]}, cy={fixed_circle[1]}, r={fixed_circle[2]}"
        )
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(
            detector_patch,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(detector_patch_mask, 1.0, 0.0),
        )
        ax.set_title("Detector-only sphere")
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("colat (deg)")

        ax = axes[row, 2]
        ax.imshow(
            master_texture,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
        )
        ax.imshow(
            crystal_patch,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(crystal_patch_mask, 0.9, 0.0),
        )
        ax.set_title(
            "Crystal sphere position\n"
            f"PC={tuple(round(v, 4) for v in projection.pc_internal)}"
        )
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("colat (deg)")

        # Keep the corrected pattern computation alive in the row workflow;
        # it also documents that both sphere projections use the same contrast.
        _ = corrected

    fig.suptitle(
        f"{sample_set.name}: fixed chain original image, PC converted from EDAX, +70/+8, v_crystal = G_h5 @ v_sample",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    sample_set.output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    sample_set.output_path.write_bytes(buffer.getvalue())
    plt.close(fig)
    print(f"Saved {sample_set.output_path}")


def main() -> None:
    upper, lower, _upper_interp, _lower_interp = load_master_samplers(MASTER_PATH)
    upper_interp = make_master_sampler(preprocess_master_hemisphere(upper, "band"))
    lower_interp = make_master_sampler(preprocess_master_hemisphere(lower, "band"))
    master_texture = build_master_texture(upper_interp, lower_interp)

    sample_sets = [
        SampleSet(
            name="Area 1 HighR",
            h5_path=H5_PATH,
            up2_path=Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 1/OIM Map 1HighR",
            pattern_indices=(34, 14441, 28676, 29111),
            output_path=OUTPUT_DIR / "area1_highr_fixed_chain_lambert_band.png",
        ),
        SampleSet(
            name="Area 2 HighR",
            h5_path=H5_PATH,
            up2_path=Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
            pattern_indices=(19802, 21230, 22834, 19625),
            output_path=OUTPUT_DIR / "area2_highr_fixed_chain_lambert_band.png",
        ),
    ]

    for sample_set in sample_sets:
        visualize_sample_set(sample_set, master_texture)


if __name__ == "__main__":
    main()
