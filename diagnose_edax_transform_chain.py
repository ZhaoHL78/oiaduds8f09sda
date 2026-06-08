from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, filters

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    MasterHemisphereSampler,
    preprocess_master_hemisphere,
    preprocess_pattern,
    project_patch_to_lon_colat,
    read_edax_inputs,
    sample_master,
)


@dataclass(frozen=True)
class ImageVariant:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TransformCandidate:
    image_variant: str
    matrix_variant: str
    sample_tilt: float
    camera_elevation: float
    intensity_score: float
    band_score: float
    combined_score: float
    abs_combined_score: float


def image_variants() -> list[ImageVariant]:
    return [
        ImageVariant("original", lambda x: x),
        ImageVariant("flip_ud", np.flipud),
        ImageVariant("flip_lr", np.fliplr),
        ImageVariant("transpose", lambda x: x.T),
        ImageVariant("transpose_flip_ud", lambda x: np.flipud(x.T)),
        ImageVariant("transpose_flip_lr", lambda x: np.fliplr(x.T)),
    ]


def matrix_variants(orientation_flat: np.ndarray) -> dict[str, np.ndarray]:
    g = orientation_flat.reshape(3, 3)
    inv_g = np.linalg.inv(g)
    return {
        "g": g,
        "g.T": g.T,
        "inv(g)": inv_g,
        "inv(g).T": inv_g.T,
    }


def zscore_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask].astype(np.float64)
    return (selected - selected.mean()) / (selected.std() + 1e-8)


def build_master_corrected_samplers(master_h5_path: Path):
    upper, lower, _upper_interp, _lower_interp = load_master_samplers(master_h5_path)
    upper_corr = preprocess_master_hemisphere(upper, "corrected")
    lower_corr = preprocess_master_hemisphere(lower, "corrected")
    upper_band = exposure.rescale_intensity(
        filters.meijering(upper_corr, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    lower_band = exposure.rescale_intensity(
        filters.meijering(lower_corr, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    samplers = {}
    for name, data in {
        "upper_corr": upper_corr,
        "lower_corr": lower_corr,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }.items():
        samplers[name] = make_master_sampler(data)
    return samplers


def detector_local_vectors(shape: tuple[int, int], pc_internal: tuple[float, float, float]) -> np.ndarray:
    h, w = shape
    pcx, pcy, pcz = pc_internal
    rows, cols = np.indices(shape)
    x = (cols + 0.5 - pcx * w) / (pcz * h)
    y = -(rows + 0.5 - pcy * h) / (pcz * h)
    z = np.ones_like(x, dtype=np.float64)
    vectors = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def save_detector_only_figure(
    pattern: np.ndarray,
    mask: np.ndarray,
    pc_internal: tuple[float, float, float],
    output_path: Path,
) -> None:
    corrected = preprocess_pattern(pattern)
    local_vectors = detector_local_vectors(pattern.shape, pc_internal)
    valid_vectors = local_vectors[mask.ravel()]
    patch, patch_mask, _ = project_patch_to_lon_colat(valid_vectors, corrected[mask])

    fig = plt.figure(figsize=(13.5, 7.5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(pattern, cmap="gray")
    ax0.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
    ax0.set_title("Raw pattern + circular mask")
    ax0.axis("off")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(
        patch,
        cmap="magma",
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=np.where(patch_mask, 1.0, 0.0),
    )
    ax1.set_title("Detector-only spherical patch")
    ax1.set_xlabel("Detector longitude (deg)")
    ax1.set_ylabel("Detector colatitude (deg)")

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    step = max(valid_vectors.shape[0] // 18000, 1)
    ax2.scatter(
        valid_vectors[::step, 0],
        valid_vectors[::step, 1],
        valid_vectors[::step, 2],
        c=corrected[mask][::step],
        cmap="gray",
        s=1,
        depthshade=False,
    )
    ax2.set_title("Detector local unit vectors")
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_detector_directions(
    shape: tuple[int, int],
    pc_edax: tuple[float, float, float],
    sample_tilt: float,
    camera_elevation: float,
    camera_azimuthal: float,
) -> np.ndarray:
    detector = EBSDDetector(
        shape=shape,
        pc=pc_edax,
        convention="edax",
        tilt=camera_elevation,
        azimuthal=camera_azimuthal,
        sample_tilt=sample_tilt,
    )
    return _get_direction_cosines_from_detector(detector)


def score_candidate(
    experimental_corr: np.ndarray,
    experimental_band: np.ndarray,
    mask: np.ndarray,
    detector_directions: np.ndarray,
    matrix: np.ndarray,
    samplers: dict[str, MasterHemisphereSampler],
    stride: int,
) -> tuple[float, float, float]:
    stride_mask = np.zeros(mask.shape, dtype=bool)
    stride_mask[::stride, ::stride] = True
    valid = mask & stride_mask
    indices = np.flatnonzero(valid.ravel())

    crystal_vectors = detector_directions[indices] @ matrix
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    master_corr = sample_master(crystal_vectors, samplers["upper_corr"], samplers["lower_corr"])
    master_band = sample_master(crystal_vectors, samplers["upper_band"], samplers["lower_band"])

    exp_corr = zscore_masked(experimental_corr, valid)
    exp_band = zscore_masked(experimental_band, valid)
    master_corr = (master_corr - master_corr.mean()) / (master_corr.std() + 1e-8)
    master_band = (master_band - master_band.mean()) / (master_band.std() + 1e-8)

    intensity_score = float(np.mean(exp_corr * master_corr))
    band_score = float(np.mean(exp_band * master_band))
    combined_score = 0.35 * intensity_score + 0.65 * band_score
    return intensity_score, band_score, combined_score


def run_exhaustive_diagnostic(
    projection,
    base_mask: np.ndarray,
    master_h5_path: Path,
    output_dir: Path,
    stride: int,
) -> list[TransformCandidate]:
    samplers = build_master_corrected_samplers(master_h5_path)
    matrices = matrix_variants(projection.orientation_flat)
    candidates: list[TransformCandidate] = []

    detector_cache: dict[tuple[float, float], np.ndarray] = {}
    for sample_tilt in (projection.sample_tilt, -projection.sample_tilt):
        for camera_elevation in (projection.camera_elevation, -projection.camera_elevation):
            detector_cache[(sample_tilt, camera_elevation)] = get_detector_directions(
                shape=projection.shape,
                pc_edax=projection.pc_edax,
                sample_tilt=sample_tilt,
                camera_elevation=camera_elevation,
                camera_azimuthal=projection.camera_azimuthal,
            )

    for variant in image_variants():
        image = variant.apply(projection.pattern)
        mask = variant.apply(base_mask)
        corr = preprocess_pattern(image)
        band = exposure.rescale_intensity(
            filters.meijering(corr, sigmas=range(1, 6), black_ridges=False),
            in_range="image",
            out_range=(0.0, 1.0),
        )
        for matrix_name, matrix in matrices.items():
            for (sample_tilt, camera_elevation), detector_directions in detector_cache.items():
                intensity_score, band_score, combined_score = score_candidate(
                    experimental_corr=corr,
                    experimental_band=band,
                    mask=mask,
                    detector_directions=detector_directions,
                    matrix=matrix,
                    samplers=samplers,
                    stride=stride,
                )
                candidates.append(
                    TransformCandidate(
                        image_variant=variant.name,
                        matrix_variant=matrix_name,
                        sample_tilt=sample_tilt,
                        camera_elevation=camera_elevation,
                        intensity_score=intensity_score,
                        band_score=band_score,
                        combined_score=combined_score,
                        abs_combined_score=abs(combined_score),
                    )
                )

    candidates.sort(key=lambda c: c.abs_combined_score, reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transform_chain_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(TransformCandidate.__dataclass_fields__))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.__dict__)
    return candidates


def save_top_candidate_figure(
    projection,
    base_mask: np.ndarray,
    master_h5_path: Path,
    candidates: list[TransformCandidate],
    output_path: Path,
    top_n: int = 8,
) -> None:
    _upper, _lower, upper_interp, lower_interp = load_master_samplers(master_h5_path)
    master_texture = np.zeros((360, 720), dtype=np.float32)
    lon = np.linspace(-np.pi, np.pi, master_texture.shape[1])
    colat = np.linspace(0.0, np.pi, master_texture.shape[0])
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    master_vectors = np.column_stack(
        [
            np.sin(colat_grid).ravel() * np.cos(lon_grid).ravel(),
            np.sin(colat_grid).ravel() * np.sin(lon_grid).ravel(),
            np.cos(colat_grid).ravel(),
        ]
    )
    master_texture[:] = sample_master(master_vectors, upper_interp, lower_interp).reshape(master_texture.shape)

    variants = {variant.name: variant for variant in image_variants()}
    matrices = matrix_variants(projection.orientation_flat)
    detector_cache: dict[tuple[float, float], np.ndarray] = {}

    rows = int(np.ceil(top_n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows), squeeze=False)
    for ax, candidate in zip(axes.ravel(), candidates[:top_n]):
        variant = variants[candidate.image_variant]
        image = variant.apply(projection.pattern)
        mask = variant.apply(base_mask)
        corrected = preprocess_pattern(image)
        key = (candidate.sample_tilt, candidate.camera_elevation)
        if key not in detector_cache:
            detector_cache[key] = get_detector_directions(
                shape=projection.shape,
                pc_edax=projection.pc_edax,
                sample_tilt=candidate.sample_tilt,
                camera_elevation=candidate.camera_elevation,
                camera_azimuthal=projection.camera_azimuthal,
            )
        detector_directions = detector_cache[key]
        matrix = matrices[candidate.matrix_variant]
        indices = np.flatnonzero(mask.ravel())
        crystal_vectors = detector_directions[indices] @ matrix
        crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
        patch, patch_mask, _ = project_patch_to_lon_colat(crystal_vectors, corrected[mask])

        ax.imshow(
            master_texture,
            cmap="gray",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
        )
        ax.imshow(
            patch,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(patch_mask, 0.88, 0.0),
        )
        ax.set_title(
            f"{candidate.image_variant}, {candidate.matrix_variant}, "
            f"tilt={candidate.sample_tilt:g}, elev={candidate.camera_elevation:g}, "
            f"score={candidate.combined_score:+.4f}"
        )
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Colatitude (deg)")

    for ax in axes.ravel()[len(candidates[:top_n]) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_orientation_and_pc_notes(h5_path: Path, map_group: str, pattern_index: int) -> None:
    with h5py.File(h5_path, "r") as h5:
        data = h5[map_group]["EBSD/ANG/DATA/DATA"]
        g = data["Orientations"][pattern_index].reshape(3, 3)
        phase = int(data["Phase"][pattern_index])
        ci = float(data["CI"][pattern_index])
        iq = float(data["IQ"][pattern_index])
    print("EDAX orientation matrix g from HDF5, row-major:")
    print(g)
    print(f"det(g)={np.linalg.det(g):+.8f}, max|g.T g - I|={np.max(np.abs(g.T @ g - np.eye(3))):.3e}")
    print(f"Pattern phase={phase}, CI={ci:.4f}, IQ={iq:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose EDAX image/detector/sample/crystal transform conventions.")
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
    parser.add_argument("--map-group", default="/20260512/Cu/Area 1/OIM Map 1HighR")
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path(
            r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"outputs\github_edax_visualizations\edax_transform_diagnostic"),
    )
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        )
    )
    base_mask, circle = estimate_circular_detector_mask(projection.pattern)
    print_orientation_and_pc_notes(args.h5, args.map_group, args.pattern_index)
    print(f"EDAX PC={projection.pc_edax}; internal PC={projection.pc_internal}")
    print(f"Estimated circular detector mask: center=({circle[0]}, {circle[1]}), radius={circle[2]}, pixels={int(base_mask.sum())}")

    detector_only_path = args.output_dir / "detector_only_spherical_geometry.png"
    save_detector_only_figure(projection.pattern, base_mask, projection.pc_internal, detector_only_path)
    print(f"Saved detector-only diagnostic: {detector_only_path}")

    candidates = run_exhaustive_diagnostic(
        projection=projection,
        base_mask=base_mask,
        master_h5_path=args.master,
        output_dir=args.output_dir,
        stride=args.stride,
    )
    top_path = args.output_dir / "top_transform_candidates.png"
    save_top_candidate_figure(projection, base_mask, args.master, candidates, top_path)
    print(f"Saved top candidate overlays: {top_path}")
    print(f"Saved score table: {args.output_dir / 'transform_chain_scores.csv'}")
    print("Top 12 candidates by absolute combined score:")
    for candidate in candidates[:12]:
        print(
            f"{candidate.image_variant:>18} | {candidate.matrix_variant:>8} | "
            f"tilt={candidate.sample_tilt:>6g} elev={candidate.camera_elevation:>5g} | "
            f"I={candidate.intensity_score:+.5f} B={candidate.band_score:+.5f} "
            f"C={candidate.combined_score:+.5f}"
        )


if __name__ == "__main__":
    main()
