from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation
from skimage import exposure, filters

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    MasterHemisphereSampler,
    ProjectionInputs,
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


@dataclass(frozen=True)
class Score:
    intensity: float
    band: float
    combined: float
    regularization: float
    objective: float


@dataclass(frozen=True)
class CandidateScore:
    name: str
    intensity: float
    band: float
    combined: float


@dataclass(frozen=True)
class PreparedImages:
    corrected: np.ndarray
    band: np.ndarray
    valid: np.ndarray
    indices: np.ndarray
    exp_corr_z: np.ndarray
    exp_band_z: np.ndarray


@dataclass(frozen=True)
class ScoringSamplers:
    upper_corr: MasterHemisphereSampler
    lower_corr: MasterHemisphereSampler
    upper_band: MasterHemisphereSampler
    lower_band: MasterHemisphereSampler
    upper_visual: MasterHemisphereSampler
    lower_visual: MasterHemisphereSampler


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return (values - values.mean()) / (values.std() + 1e-8)


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


def matrix_candidates(orientation_flat: np.ndarray) -> dict[str, np.ndarray]:
    candidates = dict(orientation_candidates(orientation_flat))
    g = orientation_flat.reshape(3, 3)
    inv_g = np.linalg.inv(g)
    candidates.update(
        {
            "g": g,
            "g.T": g.T,
            "inv(g)": inv_g,
            "inv(g).T": inv_g.T,
        }
    )
    return candidates


def build_scoring_samplers(master_h5_path: Path, visual_mode: str) -> ScoringSamplers:
    upper, lower, _upper_raw, _lower_raw = load_master_samplers(master_h5_path)

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
    upper_visual = preprocess_master_hemisphere(upper, visual_mode)
    lower_visual = preprocess_master_hemisphere(lower, visual_mode)

    return ScoringSamplers(
        upper_corr=make_master_sampler(upper_corr),
        lower_corr=make_master_sampler(lower_corr),
        upper_band=make_master_sampler(upper_band),
        lower_band=make_master_sampler(lower_band),
        upper_visual=make_master_sampler(upper_visual),
        lower_visual=make_master_sampler(lower_visual),
    )


def prepare_images(pattern: np.ndarray, mask: np.ndarray, stride: int) -> PreparedImages:
    corrected = preprocess_pattern(pattern)
    band = exposure.rescale_intensity(
        filters.meijering(corrected, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    stride_mask = np.zeros(mask.shape, dtype=bool)
    stride_mask[::stride, ::stride] = True
    valid = mask & stride_mask
    indices = np.flatnonzero(valid.ravel()).astype(np.int64)
    return PreparedImages(
        corrected=corrected,
        band=band,
        valid=valid,
        indices=indices,
        exp_corr_z=zscore(corrected.ravel()[indices]),
        exp_band_z=zscore(band.ravel()[indices]),
    )


def score_vectors(
    vectors: np.ndarray,
    prepared: PreparedImages,
    samplers: ScoringSamplers,
    intensity_weight: float,
    band_weight: float,
    regularization: float = 0.0,
) -> Score:
    master_corr = sample_master(vectors, samplers.upper_corr, samplers.lower_corr)
    master_band = sample_master(vectors, samplers.upper_band, samplers.lower_band)
    intensity = float(np.mean(prepared.exp_corr_z * zscore(master_corr)))
    band = float(np.mean(prepared.exp_band_z * zscore(master_band)))
    combined = intensity_weight * intensity + band_weight * band
    return Score(
        intensity=intensity,
        band=band,
        combined=combined,
        regularization=regularization,
        objective=combined - regularization,
    )


def select_initial_candidate(
    projection: ProjectionInputs,
    prepared: PreparedImages,
    samplers: ScoringSamplers,
    intensity_weight: float,
    band_weight: float,
    requested_name: str,
) -> tuple[str, np.ndarray, list[CandidateScore]]:
    detector_directions = projection.detector_directions
    candidates = matrix_candidates(projection.orientation_flat)
    rows: list[CandidateScore] = []

    for name, matrix in candidates.items():
        vectors = detector_directions[prepared.indices] @ matrix
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        score = score_vectors(
            vectors=vectors,
            prepared=prepared,
            samplers=samplers,
            intensity_weight=intensity_weight,
            band_weight=band_weight,
        )
        rows.append(
            CandidateScore(
                name=name,
                intensity=score.intensity,
                band=score.band,
                combined=score.combined,
            )
        )

    rows.sort(key=lambda row: row.combined, reverse=True)
    if requested_name != "auto":
        if requested_name not in candidates:
            names = ", ".join(sorted(candidates))
            raise ValueError(f"Unknown --matrix {requested_name!r}; choose one of: auto, {names}")
        return requested_name, candidates[requested_name], rows
    return rows[0].name, candidates[rows[0].name], rows


def build_parameter_layout(args: argparse.Namespace) -> tuple[list[str], list[tuple[float, float]], dict[str, float]]:
    names: list[str] = []
    bounds: list[tuple[float, float]] = []
    scales: dict[str, float] = {}

    if args.fit_rotation:
        for name in ("rot_x_deg", "rot_y_deg", "rot_z_deg"):
            names.append(name)
            bounds.append((-args.rotation_bound_deg, args.rotation_bound_deg))
            scales[name] = args.rotation_bound_deg
    if args.fit_pcxy:
        for name in ("dpcx", "dpcy"):
            names.append(name)
            bounds.append((-args.pc_xy_bound, args.pc_xy_bound))
            scales[name] = args.pc_xy_bound
    if args.fit_pcz:
        names.append("dpcz")
        bounds.append((-args.pc_z_bound, args.pc_z_bound))
        scales["dpcz"] = args.pc_z_bound
    if args.fit_radius:
        names.append("radius_delta")
        bounds.append((-args.radius_bound, args.radius_bound))
        scales["radius_delta"] = args.radius_bound

    return names, bounds, scales


def unpack_params(names: list[str], x: np.ndarray) -> dict[str, float]:
    values = {name: 0.0 for name in ("rot_x_deg", "rot_y_deg", "rot_z_deg", "dpcx", "dpcy", "dpcz", "radius_delta")}
    values.update({name: float(value) for name, value in zip(names, x)})
    return values


def params_to_rotation(params: dict[str, float]) -> np.ndarray:
    rotvec_rad = np.deg2rad(
        [params["rot_x_deg"], params["rot_y_deg"], params["rot_z_deg"]]
    )
    return Rotation.from_rotvec(rotvec_rad).as_matrix()


def effective_pc(
    base_pc: tuple[float, float, float],
    params: dict[str, float],
) -> tuple[float, float, float]:
    radius_scale = 1.0 + params["radius_delta"]
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must stay positive")
    pcx = base_pc[0] + params["dpcx"]
    pcy = base_pc[1] + params["dpcy"]
    pcz = (base_pc[2] + params["dpcz"]) / radius_scale
    return float(pcx), float(pcy), float(pcz)


def parameter_penalty(params: dict[str, float], scales: dict[str, float]) -> float:
    terms = []
    for name, scale in scales.items():
        if scale > 0.0:
            terms.append((params[name] / scale) ** 2)
    if not terms:
        return 0.0
    return float(np.mean(terms))


def crystal_vectors_for_params(
    projection: ProjectionInputs,
    indices: np.ndarray,
    base_matrix: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    pc_eff = effective_pc(projection.pc_edax, params)
    detector_directions = detector_directions_with_pc(projection, pc_eff)
    delta = params_to_rotation(params)
    vectors = detector_directions[indices] @ base_matrix @ delta.T
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors, pc_eff


def fit_parameters(
    projection: ProjectionInputs,
    prepared: PreparedImages,
    samplers: ScoringSamplers,
    base_matrix: np.ndarray,
    names: list[str],
    bounds: list[tuple[float, float]],
    scales: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Score, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    zero = np.zeros(len(names), dtype=np.float64)

    def evaluate(x: np.ndarray) -> Score:
        params = unpack_params(names, x)
        vectors, _pc_eff = crystal_vectors_for_params(
            projection=projection,
            indices=prepared.indices,
            base_matrix=base_matrix,
            params=params,
        )
        regularization = args.regularization_weight * parameter_penalty(params, scales)
        score = score_vectors(
            vectors=vectors,
            prepared=prepared,
            samplers=samplers,
            intensity_weight=args.intensity_weight,
            band_weight=args.band_weight,
            regularization=regularization,
        )
        row = {
            "eval": len(trace),
            "intensity": score.intensity,
            "band": score.band,
            "combined": score.combined,
            "regularization": score.regularization,
            "objective": score.objective,
        }
        row.update(params)
        row["radius_scale"] = 1.0 + params["radius_delta"]
        trace.append(row)
        return score

    initial_score = evaluate(zero)
    best_x = zero.copy()
    best_score = initial_score

    if len(names) > 0 and args.global_iter > 0:
        result_global = optimize.differential_evolution(
            lambda x: -evaluate(np.asarray(x, dtype=np.float64)).objective,
            bounds=bounds,
            maxiter=args.global_iter,
            popsize=args.population,
            polish=False,
            seed=args.seed,
            tol=args.global_tol,
            updating="immediate",
            workers=1,
        )
        global_score = evaluate(result_global.x)
        if global_score.objective > best_score.objective:
            best_x = result_global.x.astype(np.float64)
            best_score = global_score

    if len(names) > 0 and args.local_maxiter > 0:
        result_local = optimize.minimize(
            lambda x: -evaluate(np.asarray(x, dtype=np.float64)).objective,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": args.local_maxiter, "ftol": args.local_ftol},
        )
        local_score = evaluate(result_local.x)
        if local_score.objective > best_score.objective:
            best_x = result_local.x.astype(np.float64)
            best_score = local_score

    return best_x, best_score, trace


def full_resolution_vectors(
    projection: ProjectionInputs,
    mask: np.ndarray,
    base_matrix: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    indices = np.flatnonzero(mask.ravel()).astype(np.int64)
    vectors, pc_eff = crystal_vectors_for_params(
        projection=projection,
        indices=indices,
        base_matrix=base_matrix,
        params=params,
    )
    return vectors, indices, pc_eff


def sample_refined_master_on_detector(
    shape: tuple[int, int],
    indices: np.ndarray,
    vectors: np.ndarray,
    samplers: ScoringSamplers,
) -> tuple[np.ndarray, np.ndarray]:
    corr = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    band = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    corr[indices] = sample_master(vectors, samplers.upper_corr, samplers.lower_corr)
    band[indices] = sample_master(vectors, samplers.upper_band, samplers.lower_band)
    return corr.reshape(shape), band.reshape(shape)


def save_preprocess_figure(
    projection: ProjectionInputs,
    mask: np.ndarray,
    circle: tuple[int, int, int],
    prepared: PreparedImages,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].imshow(projection.pattern, cmap="gray")
    axes[0].contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
    axes[0].set_title(f"Raw pattern + mask\ncx={circle[0]}, cy={circle[1]}, r={circle[2]}")
    axes[1].imshow(prepared.corrected, cmap="gray")
    axes[1].set_title("Background corrected")
    axes[2].imshow(prepared.band, cmap="magma")
    axes[2].set_title("Band-enhanced image")
    for ax in axes:
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_candidate_score_figure(rows: list[CandidateScore], output_path: Path) -> None:
    top = rows[:8]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(y - 0.22, [row.intensity for row in top], height=0.22, label="intensity")
    ax.barh(y, [row.band for row in top], height=0.22, label="band")
    ax.barh(y + 0.22, [row.combined for row in top], height=0.22, label="combined")
    ax.set_yticks(y, [row.name for row in top])
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("NCC score")
    ax.set_title("Initial orientation/matrix candidates")
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trace_csv(trace: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not trace:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)


def save_trace_figure(trace: list[dict[str, Any]], output_path: Path) -> None:
    if not trace:
        return
    evals = [row["eval"] for row in trace]
    combined = [row["combined"] for row in trace]
    objective = [row["objective"] for row in trace]
    best_so_far = np.maximum.accumulate(objective)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(evals, combined, color="#6a5acd", alpha=0.45, label="combined")
    axes[0].plot(evals, objective, color="#008b8b", alpha=0.45, label="objective")
    axes[0].plot(evals, best_so_far, color="#111111", linewidth=1.2, label="best objective")
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Fine-tune score trace")

    for key, color in [
        ("rot_x_deg", "#d62728"),
        ("rot_y_deg", "#2ca02c"),
        ("rot_z_deg", "#1f77b4"),
        ("radius_scale", "#9467bd"),
    ]:
        if key in trace[0]:
            axes[1].plot(evals, [row[key] for row in trace], label=key, color=color)
    axes[1].set_xlabel("objective evaluation")
    axes[1].set_ylabel("parameter value")
    axes[1].legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sphere_overlay_figure(
    projection: ProjectionInputs,
    mask: np.ndarray,
    initial_vectors: np.ndarray,
    refined_vectors: np.ndarray,
    initial_score: Score,
    refined_score: Score,
    prepared: PreparedImages,
    samplers: ScoringSamplers,
    output_path: Path,
) -> None:
    master_texture = build_master_lon_colat(samplers.upper_visual, samplers.lower_visual)
    values = prepared.corrected[mask]
    initial_patch, initial_mask, _ = project_patch_to_lon_colat(initial_vectors, values)
    refined_patch, refined_mask, _ = project_patch_to_lon_colat(refined_vectors, values)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
    for ax, patch, patch_mask, title in [
        (
            axes[0],
            initial_patch,
            initial_mask,
            f"Initial patch, score={initial_score.combined:+.4f}",
        ),
        (
            axes[1],
            refined_patch,
            refined_mask,
            f"Refined patch, score={refined_score.combined:+.4f}",
        ),
    ]:
        ax.imshow(master_texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        ax.imshow(
            patch,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(patch_mask, 0.90, 0.0),
        )
        ax.set_title(title)
        ax.set_xlabel("longitude on crystal sphere (deg)")
        ax.set_ylabel("colatitude (deg)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_detector_residual_figure(
    prepared: PreparedImages,
    refined_master_corr: np.ndarray,
    refined_master_band: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    exp_corr = np.where(mask, prepared.corrected, np.nan)
    exp_band = np.where(mask, prepared.band, np.nan)
    residual = np.full(mask.shape, np.nan, dtype=np.float32)
    residual[mask] = zscore(exp_corr[mask]) - zscore(refined_master_corr[mask])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    panels = [
        (exp_corr, "Experimental corrected", "gray"),
        (refined_master_corr, "Master sampled on detector", "gray"),
        (residual, "Z-scored residual", "coolwarm"),
        (exp_band, "Experimental band image", "magma"),
        (refined_master_band, "Master band on detector", "magma"),
        (np.abs(residual), "Absolute residual", "inferno"),
    ]
    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_3d_patch_figure(
    initial_vectors: np.ndarray,
    refined_vectors: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    if len(values) > max_points:
        rng = np.random.default_rng(123)
        keep = np.sort(rng.choice(len(values), size=max_points, replace=False))
    else:
        keep = np.arange(len(values))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#d0d0d0", linewidth=0.35, alpha=0.45)
    ax.scatter(
        initial_vectors[keep, 0],
        initial_vectors[keep, 1],
        initial_vectors[keep, 2],
        c=values[keep],
        cmap="gray",
        s=1,
        alpha=0.22,
        depthshade=False,
        label="initial",
    )
    ax.scatter(
        refined_vectors[keep, 0],
        refined_vectors[keep, 1],
        refined_vectors[keep, 2],
        c=values[keep],
        cmap="magma",
        s=1,
        alpha=0.70,
        depthshade=False,
        label="refined",
    )
    ax.set_title("Experimental Kikuchi patch on the unit sphere")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.legend(loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_mapping_npz(
    output_path: Path,
    projection: ProjectionInputs,
    mask: np.ndarray,
    indices: np.ndarray,
    initial_vectors: np.ndarray,
    refined_vectors: np.ndarray,
    prepared: PreparedImages,
    refined_master_corr: np.ndarray,
    refined_master_band: np.ndarray,
) -> None:
    shape = projection.shape
    init_full = np.full((shape[0] * shape[1], 3), np.nan, dtype=np.float32)
    refined_full = np.full((shape[0] * shape[1], 3), np.nan, dtype=np.float32)
    init_full[indices] = initial_vectors.astype(np.float32)
    refined_full[indices] = refined_vectors.astype(np.float32)
    refined_lon = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    refined_colat = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    refined_lon[indices] = np.rad2deg(np.arctan2(refined_vectors[:, 1], refined_vectors[:, 0])).astype(np.float32)
    refined_colat[indices] = np.rad2deg(np.arccos(np.clip(refined_vectors[:, 2], -1.0, 1.0))).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        valid_mask=mask,
        initial_vectors=init_full.reshape(shape + (3,)),
        refined_vectors=refined_full.reshape(shape + (3,)),
        refined_lon_deg=refined_lon.reshape(shape),
        refined_colat_deg=refined_colat.reshape(shape),
        corrected_pattern=prepared.corrected.astype(np.float32),
        band_pattern=prepared.band.astype(np.float32),
        refined_master_corrected=refined_master_corr.astype(np.float32),
        refined_master_band=refined_master_band.astype(np.float32),
    )


def write_mapping_csv(
    output_path: Path,
    shape: tuple[int, int],
    indices: np.ndarray,
    refined_vectors: np.ndarray,
    prepared: PreparedImages,
    refined_master_corr: np.ndarray,
    refined_master_band: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = np.unravel_index(indices, shape)
    lon = np.rad2deg(np.arctan2(refined_vectors[:, 1], refined_vectors[:, 0]))
    colat = np.rad2deg(np.arccos(np.clip(refined_vectors[:, 2], -1.0, 1.0)))
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "row",
                "col",
                "x",
                "y",
                "z",
                "lon_deg",
                "colat_deg",
                "corrected",
                "band",
                "master_corrected",
                "master_band",
            ]
        )
        for i, row, col in zip(range(len(indices)), rows, cols):
            writer.writerow(
                [
                    int(row),
                    int(col),
                    float(refined_vectors[i, 0]),
                    float(refined_vectors[i, 1]),
                    float(refined_vectors[i, 2]),
                    float(lon[i]),
                    float(colat[i]),
                    float(prepared.corrected[row, col]),
                    float(prepared.band[row, col]),
                    float(refined_master_corr[row, col]),
                    float(refined_master_band[row, col]),
                ]
            )


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    projection: ProjectionInputs,
    circle: tuple[int, int, int],
    matrix_name: str,
    initial_score: Score,
    refined_score: Score,
    best_params: dict[str, float],
    refined_pc: tuple[float, float, float],
    candidate_rows: list[CandidateScore],
) -> None:
    summary = {
        "input": {
            "h5": str(args.h5),
            "up2": str(args.up2),
            "map_group": args.map_group,
            "pattern_index": args.pattern_index,
            "master": str(args.master),
        },
        "projection": {
            "map_name": projection.map_name,
            "shape": list(projection.shape),
            "pc_edax_initial": list(projection.pc_edax),
            "pc_edax_effective_refined": list(refined_pc),
            "sample_tilt": projection.sample_tilt,
            "camera_elevation": projection.camera_elevation,
            "camera_azimuthal": projection.camera_azimuthal,
            "sem_kv": projection.sem_kv,
            "circle": list(circle),
        },
        "fit": {
            "matrix_name": matrix_name,
            "params": best_params | {"radius_scale": 1.0 + best_params["radius_delta"]},
            "initial_score": initial_score.__dict__,
            "refined_score": refined_score.__dict__,
            "score_gain": refined_score.combined - initial_score.combined,
            "settings": {
                "stride": args.stride,
                "intensity_weight": args.intensity_weight,
                "band_weight": args.band_weight,
                "regularization_weight": args.regularization_weight,
                "rotation_bound_deg": args.rotation_bound_deg,
                "pc_xy_bound": args.pc_xy_bound,
                "pc_z_bound": args.pc_z_bound,
                "radius_bound": args.radius_bound,
                "fit_rotation": args.fit_rotation,
                "fit_pcxy": args.fit_pcxy,
                "fit_pcz": args.fit_pcz,
                "fit_radius": args.fit_radius,
            },
        },
        "candidate_scores": [row.__dict__ for row in candidate_rows],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        )
    )
    mask, circle = estimate_circular_detector_mask(projection.pattern)
    prepared = prepare_images(projection.pattern, mask, stride=args.stride)
    samplers = build_scoring_samplers(args.master, args.master_display)

    save_preprocess_figure(
        projection=projection,
        mask=mask,
        circle=circle,
        prepared=prepared,
        output_path=output_dir / "00_raw_preprocess_mask.png",
    )

    matrix_name, base_matrix, candidate_rows = select_initial_candidate(
        projection=projection,
        prepared=prepared,
        samplers=samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
        requested_name=args.matrix,
    )
    save_candidate_score_figure(candidate_rows, output_dir / "01_initial_candidate_scores.png")

    names, bounds, scales = build_parameter_layout(args)
    best_x, refined_score, trace = fit_parameters(
        projection=projection,
        prepared=prepared,
        samplers=samplers,
        base_matrix=base_matrix,
        names=names,
        bounds=bounds,
        scales=scales,
        args=args,
    )
    best_params = unpack_params(names, best_x)
    initial_params = unpack_params(names, np.zeros(len(names), dtype=np.float64))

    initial_full_vectors, full_indices, _initial_pc = full_resolution_vectors(
        projection=projection,
        mask=mask,
        base_matrix=base_matrix,
        params=initial_params,
    )
    refined_full_vectors, refined_indices, refined_pc = full_resolution_vectors(
        projection=projection,
        mask=mask,
        base_matrix=base_matrix,
        params=best_params,
    )
    if not np.array_equal(full_indices, refined_indices):
        raise RuntimeError("Internal error: full-resolution index sets changed during refinement")

    initial_eval_vectors, _initial_eval_pc = crystal_vectors_for_params(
        projection=projection,
        indices=prepared.indices,
        base_matrix=base_matrix,
        params=initial_params,
    )
    initial_score = score_vectors(
        vectors=initial_eval_vectors,
        prepared=prepared,
        samplers=samplers,
        intensity_weight=args.intensity_weight,
        band_weight=args.band_weight,
    )

    refined_master_corr, refined_master_band = sample_refined_master_on_detector(
        shape=projection.shape,
        indices=full_indices,
        vectors=refined_full_vectors,
        samplers=samplers,
    )

    save_trace_csv(trace, output_dir / "02_finetune_trace.csv")
    save_trace_figure(trace, output_dir / "02_finetune_trace.png")
    save_sphere_overlay_figure(
        projection=projection,
        mask=mask,
        initial_vectors=initial_full_vectors,
        refined_vectors=refined_full_vectors,
        initial_score=initial_score,
        refined_score=refined_score,
        prepared=prepared,
        samplers=samplers,
        output_path=output_dir / "03_sphere_overlay_initial_vs_refined.png",
    )
    save_detector_residual_figure(
        prepared=prepared,
        refined_master_corr=refined_master_corr,
        refined_master_band=refined_master_band,
        mask=mask,
        output_path=output_dir / "04_detector_forward_residual.png",
    )
    save_3d_patch_figure(
        initial_vectors=initial_full_vectors,
        refined_vectors=refined_full_vectors,
        values=prepared.corrected[mask],
        output_path=output_dir / "05_patch_on_unit_sphere_3d.png",
        max_points=args.max_3d_points,
    )
    write_mapping_npz(
        output_path=output_dir / "refined_pixel_to_sphere_mapping.npz",
        projection=projection,
        mask=mask,
        indices=full_indices,
        initial_vectors=initial_full_vectors,
        refined_vectors=refined_full_vectors,
        prepared=prepared,
        refined_master_corr=refined_master_corr,
        refined_master_band=refined_master_band,
    )
    if args.write_csv:
        write_mapping_csv(
            output_path=output_dir / "refined_pixel_to_sphere_mapping.csv",
            shape=projection.shape,
            indices=full_indices,
            refined_vectors=refined_full_vectors,
            prepared=prepared,
            refined_master_corr=refined_master_corr,
            refined_master_band=refined_master_band,
        )
    write_summary(
        output_path=output_dir / "summary.json",
        args=args,
        projection=projection,
        circle=circle,
        matrix_name=matrix_name,
        initial_score=initial_score,
        refined_score=refined_score,
        best_params=best_params,
        refined_pc=refined_pc,
        candidate_rows=candidate_rows,
    )

    print(f"Saved spherical fine-tune outputs: {output_dir}")
    print(f"Selected matrix: {matrix_name}")
    print(f"Initial combined score: {initial_score.combined:+.6f}")
    print(f"Refined combined score: {refined_score.combined:+.6f}")
    print(f"Score gain: {refined_score.combined - initial_score.combined:+.6f}")
    print(f"Initial EDAX PC: {projection.pc_edax}")
    print(f"Effective refined EDAX PC: {tuple(round(v, 7) for v in refined_pc)}")
    print(
        "Refined params: "
        + ", ".join(
            [
                f"{name}={best_params[name]:+.6g}"
                for name in ("rot_x_deg", "rot_y_deg", "rot_z_deg", "dpcx", "dpcy", "dpcz", "radius_delta")
            ]
        )
        + f", radius_scale={1.0 + best_params['radius_delta']:.7f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match one EDAX Kikuchi pattern to a master sphere and fine-tune a small "
            "spherical correction around the H5/OIM orientation."
        )
    )
    parser.add_argument("--h5", type=Path, default=Path("ebsd.edaxh5"))
    parser.add_argument("--up2", type=Path, required=True)
    parser.add_argument("--map-group", required=True)
    parser.add_argument("--pattern-index", type=int, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "spherical_finetune_pipeline",
    )
    parser.add_argument(
        "--matrix",
        default="auto",
        help="Initial orientation matrix interpretation. Use 'auto' or one candidate name.",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--intensity-weight", type=float, default=0.35)
    parser.add_argument("--band-weight", type=float, default=0.65)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--rotation-bound-deg", type=float, default=2.0)
    parser.add_argument("--pc-xy-bound", type=float, default=0.006)
    parser.add_argument("--pc-z-bound", type=float, default=0.012)
    parser.add_argument("--radius-bound", type=float, default=0.025)
    parser.add_argument("--global-iter", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--global-tol", type=float, default=1e-3)
    parser.add_argument("--local-maxiter", type=int, default=180)
    parser.add_argument("--local-ftol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-3d-points", type=int, default=18000)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument(
        "--master-display",
        choices=("raw", "corrected", "band"),
        default="band",
        help="Master sphere contrast used in overlay visualization.",
    )

    parser.set_defaults(fit_rotation=True, fit_radius=True)
    parser.add_argument("--fit-rotation", dest="fit_rotation", action="store_true")
    parser.add_argument("--no-fit-rotation", dest="fit_rotation", action="store_false")
    parser.add_argument("--fit-pcxy", action="store_true")
    parser.add_argument("--fit-pcz", action="store_true")
    parser.add_argument("--fit-radius", dest="fit_radius", action="store_true")
    parser.add_argument("--no-fit-radius", dest="fit_radius", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.intensity_weight < 0 or args.band_weight < 0:
        raise ValueError("Score weights must be non-negative")
    total_weight = args.intensity_weight + args.band_weight
    if total_weight <= 0:
        raise ValueError("At least one score weight must be positive")
    args.intensity_weight /= total_weight
    args.band_weight /= total_weight
    run(args)


if __name__ == "__main__":
    main()
