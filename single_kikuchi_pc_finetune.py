from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, filters

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    MasterHemisphereSampler,
    build_master_lon_colat,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    orientation_candidates,
    preprocess_master_hemisphere,
    preprocess_pattern,
    project_patch_to_lon_colat,
    read_edax_inputs,
    sample_master,
)


DEFAULT_H5 = Path(r"D:\project\EBSD2026\ebsd.edaxh5")
DEFAULT_UP2 = Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2")
DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "single_kikuchi_pc_finetune"


@dataclass(frozen=True)
class MasterSamplers:
    upper_corrected: MasterHemisphereSampler
    lower_corrected: MasterHemisphereSampler
    upper_band: MasterHemisphereSampler
    lower_band: MasterHemisphereSampler


@dataclass(frozen=True)
class ScoreResult:
    stage: str
    pc: tuple[float, float, float]
    delta: tuple[float, float, float]
    intensity_score: float
    band_score: float
    combined_score: float


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return (values - values.mean()) / (values.std() + 1e-8)


def centered_circular_detector_mask(
    shape: tuple[int, int],
    radius_fraction: float = 0.40,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a conservative inner detector disk used for preprocessing/scoring.

    The saved EDAX raster can contain shadowed edge pixels near the phosphor rim.
    A smaller inner disk keeps every displayed and scored Kikuchi pattern circular.
    """
    height, width = shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = radius_fraction * min(height, width)
    yy, xx = np.indices(shape)
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    return mask, (int(round(center_x)), int(round(center_y)), int(round(radius)))


def rescale_masked(image: np.ndarray, mask: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    output = np.zeros_like(image, dtype=np.float32)
    values = image[mask & np.isfinite(image)]
    if values.size == 0:
        return output
    lo, hi = np.percentile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return output
    output[mask] = np.clip((image[mask] - lo) / (hi - lo), 0.0, 1.0)
    return output.astype(np.float32)


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


def make_stride_indices(mask: np.ndarray, stride: int) -> np.ndarray:
    stride_mask = np.zeros(mask.shape, dtype=bool)
    stride_mask[::stride, ::stride] = True
    return np.flatnonzero((mask & stride_mask).ravel()).astype(np.int64)


def build_preprocessed_images(pattern: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if mask is None:
        mask, _circle = centered_circular_detector_mask(pattern.shape)
    mask = mask.astype(bool)

    raw = pattern.astype(np.float32)
    raw_normalized = rescale_masked(raw, mask)

    fill_value = float(np.median(raw[mask])) if np.any(mask) else float(np.median(raw))
    filled = raw.copy()
    filled[~mask] = fill_value
    background = filters.gaussian(filled, sigma=18.0, preserve_range=True)
    corrected_float = filled - background
    corrected = rescale_masked(corrected_float, mask, low=0.5, high=99.5)

    enhanced = np.zeros_like(corrected, dtype=np.float32)
    if np.any(mask):
        enhanced_full = exposure.equalize_adapthist(corrected, clip_limit=0.025).astype(np.float32)
        enhanced[mask] = enhanced_full[mask]
    band = exposure.rescale_intensity(
        filters.meijering(enhanced, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    ).astype(np.float32)
    raw_normalized[~mask] = 0.0
    corrected[~mask] = 0.0
    enhanced[~mask] = 0.0
    band[~mask] = 0.0
    return {
        "raw_normalized": raw_normalized,
        "corrected": corrected.astype(np.float32),
        "enhanced": enhanced,
        "band": band,
        "mask": mask,
    }


def build_master_samplers(master_path: Path) -> MasterSamplers:
    upper, lower, _upper_raw, _lower_raw = load_master_samplers(master_path)
    upper_corrected = preprocess_master_hemisphere(upper, "corrected")
    lower_corrected = preprocess_master_hemisphere(lower, "corrected")
    upper_band = preprocess_master_hemisphere(upper, "band")
    lower_band = preprocess_master_hemisphere(lower, "band")
    return MasterSamplers(
        upper_corrected=make_master_sampler(upper_corrected),
        lower_corrected=make_master_sampler(lower_corrected),
        upper_band=make_master_sampler(upper_band),
        lower_band=make_master_sampler(lower_band),
    )


def score_with_directions(
    detector_directions: np.ndarray,
    matrix: np.ndarray,
    indices: np.ndarray,
    exp_corrected_values: np.ndarray,
    exp_band_values: np.ndarray,
    samplers: MasterSamplers,
    intensity_weight: float,
    band_weight: float,
) -> tuple[float, float, float]:
    crystal_vectors = detector_directions[indices] @ matrix
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12

    master_corrected = sample_master(crystal_vectors, samplers.upper_corrected, samplers.lower_corrected)
    master_band = sample_master(crystal_vectors, samplers.upper_band, samplers.lower_band)

    intensity_score = float(np.mean(zscore(exp_corrected_values) * zscore(master_corrected)))
    band_score = float(np.mean(zscore(exp_band_values) * zscore(master_band)))
    combined = float(intensity_weight * intensity_score + band_weight * band_score)
    return intensity_score, band_score, combined


def choose_orientation_matrix(
    projection,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers: MasterSamplers,
    stride: int,
    intensity_weight: float,
    band_weight: float,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    indices = make_stride_indices(mask, stride)
    detector_directions = detector_directions_with_pc(projection, projection.pc_edax)
    exp_corrected = images.get("enhanced", images["corrected"]).ravel()[indices]
    exp_band = images["band"].ravel()[indices]

    rows: list[dict[str, object]] = []
    for name, matrix in orientation_candidates(projection.orientation_flat).items():
        intensity, band, combined = score_with_directions(
            detector_directions=detector_directions,
            matrix=matrix,
            indices=indices,
            exp_corrected_values=exp_corrected,
            exp_band_values=exp_band,
            samplers=samplers,
            intensity_weight=intensity_weight,
            band_weight=band_weight,
        )
        rows.append(
            {
                "orientation_variant": name,
                "intensity_score": intensity,
                "band_score": band,
                "combined_score": combined,
            }
        )

    best = max(rows, key=lambda item: float(item["combined_score"]))
    best_name = str(best["orientation_variant"])
    return best_name, orientation_candidates(projection.orientation_flat)[best_name], rows


def score_pc(
    projection,
    pc: tuple[float, float, float],
    matrix: np.ndarray,
    indices: np.ndarray,
    exp_corrected_values: np.ndarray,
    exp_band_values: np.ndarray,
    samplers: MasterSamplers,
    intensity_weight: float,
    band_weight: float,
) -> tuple[float, float, float]:
    if pc[2] <= 0.05:
        return -np.inf, -np.inf, -np.inf
    detector_directions = detector_directions_with_pc(projection, pc)
    return score_with_directions(
        detector_directions=detector_directions,
        matrix=matrix,
        indices=indices,
        exp_corrected_values=exp_corrected_values,
        exp_band_values=exp_band_values,
        samplers=samplers,
        intensity_weight=intensity_weight,
        band_weight=band_weight,
    )


def evaluate_pc_grid(
    projection,
    matrix: np.ndarray,
    indices: np.ndarray,
    exp_corrected_values: np.ndarray,
    exp_band_values: np.ndarray,
    samplers: MasterSamplers,
    center_pc: tuple[float, float, float],
    reference_pc: tuple[float, float, float],
    pc_ranges: tuple[float, float, float],
    steps: int,
    stage: str,
    intensity_weight: float,
    band_weight: float,
) -> list[ScoreResult]:
    offsets = [np.linspace(-span, span, steps, dtype=np.float64) for span in pc_ranges]
    rows: list[ScoreResult] = []
    for dx, dy, dz in product(*offsets):
        pc = (center_pc[0] + float(dx), center_pc[1] + float(dy), center_pc[2] + float(dz))
        intensity, band, combined = score_pc(
            projection=projection,
            pc=pc,
            matrix=matrix,
            indices=indices,
            exp_corrected_values=exp_corrected_values,
            exp_band_values=exp_band_values,
            samplers=samplers,
            intensity_weight=intensity_weight,
            band_weight=band_weight,
        )
        rows.append(
            ScoreResult(
                stage=stage,
                pc=pc,
                delta=(
                    pc[0] - reference_pc[0],
                    pc[1] - reference_pc[1],
                    pc[2] - reference_pc[2],
                ),
                intensity_score=intensity,
                band_score=band,
                combined_score=combined,
            )
        )
    return rows


def pc_finetune(
    projection,
    matrix: np.ndarray,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    samplers: MasterSamplers,
    stride: int,
    coarse_range: tuple[float, float, float],
    coarse_steps: int,
    fine_steps: int,
    intensity_weight: float,
    band_weight: float,
) -> tuple[ScoreResult, list[ScoreResult]]:
    indices = make_stride_indices(mask, stride)
    exp_corrected = images.get("enhanced", images["corrected"]).ravel()[indices]
    exp_band = images["band"].ravel()[indices]
    pc0 = projection.pc_edax

    original_intensity, original_band, original_combined = score_pc(
        projection=projection,
        pc=pc0,
        matrix=matrix,
        indices=indices,
        exp_corrected_values=exp_corrected,
        exp_band_values=exp_band,
        samplers=samplers,
        intensity_weight=intensity_weight,
        band_weight=band_weight,
    )
    original = ScoreResult(
        stage="original",
        pc=pc0,
        delta=(0.0, 0.0, 0.0),
        intensity_score=original_intensity,
        band_score=original_band,
        combined_score=original_combined,
    )

    coarse_rows = evaluate_pc_grid(
        projection=projection,
        matrix=matrix,
        indices=indices,
        exp_corrected_values=exp_corrected,
        exp_band_values=exp_band,
        samplers=samplers,
        center_pc=pc0,
        reference_pc=pc0,
        pc_ranges=coarse_range,
        steps=coarse_steps,
        stage="coarse",
        intensity_weight=intensity_weight,
        band_weight=band_weight,
    )
    coarse_best = max(coarse_rows, key=lambda row: row.combined_score)

    if coarse_steps > 1:
        fine_range = tuple(2.0 * span / (coarse_steps - 1) for span in coarse_range)
    else:
        fine_range = tuple(span * 0.25 for span in coarse_range)
    fine_rows = evaluate_pc_grid(
        projection=projection,
        matrix=matrix,
        indices=indices,
        exp_corrected_values=exp_corrected,
        exp_band_values=exp_band,
        samplers=samplers,
        center_pc=coarse_best.pc,
        reference_pc=pc0,
        pc_ranges=fine_range,
        steps=fine_steps,
        stage="fine",
        intensity_weight=intensity_weight,
        band_weight=band_weight,
    )
    all_rows = [original] + coarse_rows + fine_rows
    return max(all_rows, key=lambda row: row.combined_score), all_rows


def project_detector_patch(projection, pc: tuple[float, float, float], values: np.ndarray, mask: np.ndarray):
    detector_directions = detector_directions_with_pc(projection, pc)
    indices = np.flatnonzero(mask.ravel())
    return project_patch_to_lon_colat(
        detector_directions[indices],
        values.ravel()[indices],
        lon_count=720,
        colat_count=360,
    )[:2]


def project_crystal_patch(
    projection,
    pc: tuple[float, float, float],
    matrix: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
):
    detector_directions = detector_directions_with_pc(projection, pc)
    indices = np.flatnonzero(mask.ravel())
    crystal_vectors = detector_directions[indices] @ matrix
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    return project_patch_to_lon_colat(
        crystal_vectors,
        values.ravel()[indices],
        lon_count=720,
        colat_count=360,
    )[:2]


def write_orientation_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_pc_scores(path: Path, rows: list[ScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "stage",
            "pcx",
            "pcy",
            "pcz",
            "delta_pcx",
            "delta_pcy",
            "delta_pcz",
            "intensity_score",
            "band_score",
            "combined_score",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row.stage,
                    "pcx": row.pc[0],
                    "pcy": row.pc[1],
                    "pcz": row.pc[2],
                    "delta_pcx": row.delta[0],
                    "delta_pcy": row.delta[1],
                    "delta_pcz": row.delta[2],
                    "intensity_score": row.intensity_score,
                    "band_score": row.band_score,
                    "combined_score": row.combined_score,
                }
            )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    projection,
    orientation_name: str,
    circle: tuple[int, int, int],
    original: ScoreResult,
    refined: ScoreResult,
    stride: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "h5": str(args.h5),
            "up2": str(args.up2),
            "map_group": args.map_group,
            "pattern_index": args.pattern_index,
            "map_name": projection.map_name,
            "orientation_variant": orientation_name,
            "mask_center_x": circle[0],
            "mask_center_y": circle[1],
            "mask_radius": circle[2],
            "stride": stride,
            "pc_edax_x": original.pc[0],
            "pc_edax_y": original.pc[1],
            "pc_edax_z": original.pc[2],
            "pc_refined_x": refined.pc[0],
            "pc_refined_y": refined.pc[1],
            "pc_refined_z": refined.pc[2],
            "delta_pcx": refined.delta[0],
            "delta_pcy": refined.delta[1],
            "delta_pcz": refined.delta[2],
            "original_combined_score": original.combined_score,
            "refined_combined_score": refined.combined_score,
            "score_gain": refined.combined_score - original.combined_score,
            "sample_tilt_deg": projection.sample_tilt,
            "camera_elevation_deg": projection.camera_elevation,
            "camera_azimuthal_deg": projection.camera_azimuthal,
            "sem_kv": projection.sem_kv,
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def imshow_sphere(ax, image: np.ndarray, mask: np.ndarray, title: str, cmap: str = "magma") -> None:
    ax.imshow(
        image,
        cmap=cmap,
        origin="upper",
        extent=[-180, 180, 180, 0],
        aspect="auto",
        alpha=np.where(mask, 0.92, 0.0),
    )
    ax.set_title(title)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("colatitude (deg)")


def save_visualization(
    path: Path,
    projection,
    images: dict[str, np.ndarray],
    mask: np.ndarray,
    circle: tuple[int, int, int],
    master_texture: np.ndarray,
    orientation_name: str,
    original: ScoreResult,
    refined: ScoreResult,
    detector_patch: tuple[np.ndarray, np.ndarray],
    original_patch: tuple[np.ndarray, np.ndarray],
    refined_patch: tuple[np.ndarray, np.ndarray],
    score_rows: list[ScoreResult],
) -> None:
    fig = plt.figure(figsize=(18, 13))
    axes = [fig.add_subplot(3, 3, i + 1) for i in range(9)]

    axes[0].imshow(projection.pattern, cmap="gray")
    axes[0].contour(mask, levels=[0.5], colors=["#ff3030"], linewidths=0.8)
    axes[0].set_title(f"Raw UP2 pattern\ncircle=({circle[0]}, {circle[1]}, r={circle[2]})")
    axes[0].axis("off")

    axes[1].imshow(images["corrected"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Preprocessed: circular mask + background corrected")
    axes[1].axis("off")

    axes[2].imshow(images["enhanced"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Contrast-enhanced image used in PC scoring")
    axes[2].axis("off")

    imshow_sphere(axes[3], detector_patch[0], detector_patch[1], "Detector-frame spherical patch\nEDAX PC, no orientation")

    for ax, patch, score, title in [
        (
            axes[4],
            original_patch,
            original,
            "Crystal-frame sphere on master\noriginal EDAX PC",
        ),
        (
            axes[5],
            refined_patch,
            refined,
            "Crystal-frame sphere on master\nrefined PC",
        ),
    ]:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        imshow_sphere(
            ax,
            patch[0],
            patch[1],
            f"{title}\ncombined score={score.combined_score:+.5f}",
        )

    axes[6].imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    overlay = np.zeros((*original_patch[1].shape, 4), dtype=np.float32)
    overlay[..., 0] = original_patch[1].astype(np.float32)
    overlay[..., 2] = refined_patch[1].astype(np.float32)
    overlay[..., 3] = np.where(original_patch[1] | refined_patch[1], 0.70, 0.0)
    axes[6].imshow(overlay, origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
    axes[6].set_title("Patch footprint shift on master\nred=original PC, blue=refined PC")
    axes[6].set_xlabel("longitude (deg)")
    axes[6].set_ylabel("colatitude (deg)")

    fine_rows = [row for row in score_rows if row.stage == "fine"]
    best_pcz = refined.pc[2]
    pcz_values = np.array([row.pc[2] for row in fine_rows], dtype=np.float64)
    if fine_rows and pcz_values.size:
        nearest_pcz = float(pcz_values[np.argmin(np.abs(pcz_values - best_pcz))])
        slab = [row for row in fine_rows if abs(row.pc[2] - nearest_pcz) < 1e-10]
    else:
        slab = score_rows
    scatter = axes[7].scatter(
        [row.delta[0] for row in slab],
        [row.delta[1] for row in slab],
        c=[row.combined_score for row in slab],
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[7].scatter([refined.delta[0]], [refined.delta[1]], marker="x", s=120, c="red", linewidths=2)
    axes[7].axhline(0, color="gray", linewidth=0.7)
    axes[7].axvline(0, color="gray", linewidth=0.7)
    axes[7].set_title(f"Fine PC score slice near PCz={refined.pc[2]:.5f}")
    axes[7].set_xlabel("delta PCx")
    axes[7].set_ylabel("delta PCy")
    fig.colorbar(scatter, ax=axes[7], fraction=0.046, pad=0.04, label="combined score")

    axes[8].axis("off")
    summary = (
        f"map: {projection.map_name}\n"
        f"orientation: {orientation_name}\n"
        f"original PC: ({original.pc[0]:.6f}, {original.pc[1]:.6f}, {original.pc[2]:.6f})\n"
        f"refined PC:  ({refined.pc[0]:.6f}, {refined.pc[1]:.6f}, {refined.pc[2]:.6f})\n"
        f"delta PC:    ({refined.delta[0]:+.6f}, {refined.delta[1]:+.6f}, {refined.delta[2]:+.6f})\n"
        f"original score: intensity={original.intensity_score:+.5f}, "
        f"band={original.band_score:+.5f}, combined={original.combined_score:+.5f}\n"
        f"refined score:  intensity={refined.intensity_score:+.5f}, "
        f"band={refined.band_score:+.5f}, combined={refined.combined_score:+.5f}\n"
        f"score gain: {refined.combined_score - original.combined_score:+.5f}\n"
        f"sample tilt={projection.sample_tilt:g} deg, "
        f"camera elevation={projection.camera_elevation:g} deg, "
        f"azimuthal={projection.camera_azimuthal:g} deg"
    )
    axes[8].text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("Single Kikuchi preprocessing -> spherical calibration -> PC finetune", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        )
    )
    mask, circle = centered_circular_detector_mask(projection.pattern.shape, args.mask_radius_fraction)
    images = build_preprocessed_images(projection.pattern, mask)
    samplers = build_master_samplers(args.master)

    orientation_name, matrix, orientation_rows = choose_orientation_matrix(
        projection=projection,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )

    refined, score_rows = pc_finetune(
        projection=projection,
        matrix=matrix,
        mask=mask,
        images=images,
        samplers=samplers,
        stride=args.stride,
        coarse_range=tuple(args.pc_range),
        coarse_steps=args.coarse_steps,
        fine_steps=args.fine_steps,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )
    original = next(row for row in score_rows if row.stage == "original")

    master_texture = build_master_lon_colat(samplers.upper_corrected, samplers.lower_corrected)
    detector_patch = project_detector_patch(projection, original.pc, images["enhanced"], mask)
    original_patch = project_crystal_patch(projection, original.pc, matrix, images["enhanced"], mask)
    refined_patch = project_crystal_patch(projection, refined.pc, matrix, images["enhanced"], mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_orientation_scores(args.output_dir / "orientation_scores.csv", orientation_rows)
    write_pc_scores(args.output_dir / "pc_finetune_scores.csv", score_rows)
    write_summary(
        args.output_dir / "single_kikuchi_pc_finetune_summary.csv",
        args=args,
        projection=projection,
        orientation_name=orientation_name,
        circle=circle,
        original=original,
        refined=refined,
        stride=args.stride,
    )
    save_visualization(
        args.output_dir / "single_kikuchi_pc_finetune_overview.png",
        projection=projection,
        images=images,
        mask=mask,
        circle=circle,
        master_texture=master_texture,
        orientation_name=orientation_name,
        original=original,
        refined=refined,
        detector_patch=detector_patch,
        original_patch=original_patch,
        refined_patch=refined_patch,
        score_rows=score_rows,
    )

    print(f"Saved outputs to {args.output_dir}")
    print(f"Map: {projection.map_name}")
    print(f"Pattern index: {args.pattern_index}")
    print(f"Orientation variant: {orientation_name}")
    print(f"Original PC: {original.pc}; combined score={original.combined_score:+.6f}")
    print(f"Refined  PC: {refined.pc}; combined score={refined.combined_score:+.6f}")
    print(f"Delta PC: {refined.delta}; gain={refined.combined_score - original.combined_score:+.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Kikuchi pattern through preprocessing, spherical projection, local PC finetune, and visualization."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2", type=Path, default=DEFAULT_UP2)
    parser.add_argument("--map-group", default="/20260512/Cu/Area 1/OIM Map 1HighR")
    parser.add_argument("--pattern-index", type=int, default=2661)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--pc-range", nargs=3, type=float, default=(0.02, 0.02, 0.04), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--coarse-steps", type=int, default=7)
    parser.add_argument("--fine-steps", type=int, default=7)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument(
        "--mask-radius-fraction",
        type=float,
        default=0.40,
        help="Conservative circular detector mask radius as a fraction of min(pattern height, width).",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
