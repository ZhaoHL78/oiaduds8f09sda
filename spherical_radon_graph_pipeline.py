from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import Bounds, linprog, minimize
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from continuous_band_geometric_refinement import write_rows_csv
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    MasterSphere,
    PreparedPattern,
    default_map_specs,
    jsonable,
    load_master_sphere,
    prepare_pattern,
    project_to_equirect,
    read_pattern_bundle,
    resolve_master_path,
    score_rotation,
    sphere_texture,
    zscore_vector,
)
from labeled_band_radius_refinement import HKLFamily, read_phase_hkl_families
from pc_radius_bias_correction import corrected_pc, prepared_with_pc
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    parse_refine_schedule,
    plot_master_sphere,
    plot_pattern_patch,
    save_final_spatial_visualization,
    set_3d_sphere_axes,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None

warnings.filterwarnings("ignore", message=r"Optimal rotation is not uniquely or poorly defined.*")


IDENTITY_TRANSFORM = np.eye(3, dtype=np.float32)


@dataclass
class SphericalRadonResult:
    normals: np.ndarray
    scales_deg: list[float]
    scores_by_scale: np.ndarray
    best_scores: np.ndarray
    best_scale_index: np.ndarray


@dataclass
class PeakDescriptor:
    peak_id: int
    source: str
    normal: np.ndarray
    strength: float
    bandwidth_deg: float
    asymmetry: float
    profile: np.ndarray
    hkl: str = "unassigned"
    hkl_angle_deg: float = float("nan")

    def to_row(self) -> dict:
        return {
            "peak_id": int(self.peak_id),
            "source": self.source,
            "normal_x": float(self.normal[0]),
            "normal_y": float(self.normal[1]),
            "normal_z": float(self.normal[2]),
            "strength": float(self.strength),
            "bandwidth_deg": float(self.bandwidth_deg),
            "asymmetry": float(self.asymmetry),
            "hkl": self.hkl,
            "hkl_angle_deg": float(self.hkl_angle_deg),
            "profile": json.dumps(self.profile.astype(float).tolist()),
        }


@dataclass
class CandidateOrientation:
    candidate_id: int
    rotation: R
    triangle_rms_deg: float
    triplet_exp: tuple[int, int, int]
    triplet_std: tuple[int, int, int]
    triplet_residual_deg: float
    fast_score: float
    ot_cost: float = float("inf")
    transported_mass: float = 0.0
    match_count: int = 0
    image_score: float = float("nan")
    objective: float = float("inf")

    def to_row(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "triangle_rms_deg": float(self.triangle_rms_deg),
            "triplet_exp": "-".join(str(v) for v in self.triplet_exp),
            "triplet_std": "-".join(str(v) for v in self.triplet_std),
            "triplet_residual_deg": float(self.triplet_residual_deg),
            "fast_score": float(self.fast_score),
            "ot_cost": float(self.ot_cost),
            "transported_mass": float(self.transported_mass),
            "match_count": int(self.match_count),
            "image_score": float(self.image_score),
            "objective": float(self.objective),
            "rotation_quat_xyzw": json.dumps(self.rotation.as_quat().astype(float).tolist()),
        }


@dataclass
class TransportMatch:
    exp_peak_id: int
    std_peak_id: int
    mass: float
    cost: float
    angle_deg: float
    hkl: str

    def to_row(self) -> dict:
        return asdict(self)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def canonical_plane_normal(normal: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    if normal[2] < -1e-10 or (
        abs(float(normal[2])) <= 1e-10
        and (normal[1] < -1e-10 or (abs(float(normal[1])) <= 1e-10 and normal[0] < 0.0))
    ):
        normal *= -1.0
    return normal.astype(np.float32)


def plane_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(a.astype(np.float64), b.astype(np.float64))))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def fibonacci_sphere(count: int, hemisphere: bool = False) -> np.ndarray:
    count = max(4, int(count))
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    if hemisphere:
        z = (indices + 0.5) / count
    else:
        z = 1.0 - 2.0 * (indices + 0.5) / count
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = indices * golden
    vectors = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if hemisphere:
        vectors = np.asarray([canonical_plane_normal(vec) for vec in vectors], dtype=np.float32)
    return vectors.astype(np.float32)


def sample_rows(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    n = len(values)
    if count <= 0 or count >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(values[finite], [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def spherical_radon_transform(
    vectors: np.ndarray,
    values: np.ndarray,
    normals: np.ndarray,
    scales_deg: list[float],
    chunk_size: int = 256,
    desc: str = "spherical Radon",
) -> SphericalRadonResult:
    vectors = vectors.astype(np.float32)
    normals = normals.astype(np.float32)
    values = normalize_values(values)
    scores_by_scale = np.zeros((len(scales_deg), len(normals)), dtype=np.float32)
    iterator = range(0, len(normals), chunk_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=int(np.ceil(len(normals) / chunk_size)), desc=desc, leave=False)
    for start in iterator:
        stop = min(len(normals), start + chunk_size)
        dots = np.clip(vectors @ normals[start:stop].T, -1.0, 1.0)
        distances = np.abs(np.arcsin(dots)).astype(np.float32)
        for scale_index, scale_deg in enumerate(scales_deg):
            sigma = max(np.radians(scale_deg), 1e-5)
            weights = np.exp(-0.5 * (distances / sigma) ** 2).astype(np.float32)
            denom = weights.sum(axis=0) + 1e-8
            scores_by_scale[scale_index, start:stop] = (weights.T @ values) / denom
    best_scale_index = np.argmax(scores_by_scale, axis=0).astype(np.int32)
    best_scores = scores_by_scale[best_scale_index, np.arange(len(normals))]
    return SphericalRadonResult(
        normals=normals,
        scales_deg=list(scales_deg),
        scores_by_scale=scores_by_scale,
        best_scores=best_scores.astype(np.float32),
        best_scale_index=best_scale_index,
    )


def greedy_peak_pick(
    radon: SphericalRadonResult,
    peak_count: int,
    min_separation_deg: float,
    min_score_quantile: float,
) -> list[int]:
    scores = radon.best_scores
    threshold = float(np.quantile(scores, min_score_quantile))
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    min_sep_cos = math.cos(math.radians(min_separation_deg))
    for idx in order:
        if float(scores[idx]) < threshold:
            break
        normal = radon.normals[idx]
        if all(abs(float(np.dot(normal, radon.normals[old]))) < min_sep_cos for old in selected):
            selected.append(int(idx))
            if len(selected) >= peak_count:
                break
    return selected


def descriptor_profile(
    vectors: np.ndarray,
    values: np.ndarray,
    normal: np.ndarray,
    profile_width_deg: float,
    profile_bins: int,
) -> tuple[np.ndarray, float]:
    values = normalize_values(values)
    signed_distance_deg = np.degrees(np.arcsin(np.clip(vectors @ normal.astype(np.float32), -1.0, 1.0)))
    edges = np.linspace(-profile_width_deg, profile_width_deg, profile_bins + 1)
    weighted_sum, _ = np.histogram(signed_distance_deg, bins=edges, weights=values)
    counts, _ = np.histogram(signed_distance_deg, bins=edges)
    profile = np.zeros(profile_bins, dtype=np.float32)
    mask = counts > 0
    profile[mask] = (weighted_sum[mask] / counts[mask]).astype(np.float32)
    if np.std(profile) > 1e-8:
        profile = zscore_vector(profile)
    center = profile_bins // 2
    left = float(np.sum(profile[:center]))
    right = float(np.sum(profile[center + (profile_bins % 2) :]))
    asymmetry = (right - left) / (abs(right) + abs(left) + 1e-8)
    return profile.astype(np.float32), float(asymmetry)


def assign_peak_hkl(peak_normal: np.ndarray, families: list[HKLFamily], max_angle_deg: float) -> tuple[str, float]:
    best_label = "unassigned"
    best_angle = float("inf")
    for family in families:
        dots = np.abs(family.normals @ peak_normal.astype(np.float32))
        angle = float(np.degrees(np.arccos(np.clip(float(np.max(dots)), -1.0, 1.0))))
        if angle < best_angle:
            best_angle = angle
            best_label = family.label
    if best_angle > max_angle_deg:
        return "unassigned", best_angle
    return best_label, best_angle


def build_peak_descriptors(
    source: str,
    radon: SphericalRadonResult,
    peak_indices: list[int],
    vectors: np.ndarray,
    values: np.ndarray,
    args,
    families: list[HKLFamily] | None = None,
) -> list[PeakDescriptor]:
    peaks: list[PeakDescriptor] = []
    for peak_id, radon_index in enumerate(peak_indices):
        normal = canonical_plane_normal(radon.normals[radon_index])
        profile, asymmetry = descriptor_profile(
            vectors,
            values,
            normal,
            profile_width_deg=args.profile_width_deg,
            profile_bins=args.profile_bins,
        )
        hkl = "unassigned"
        hkl_angle = float("nan")
        if families:
            hkl, hkl_angle = assign_peak_hkl(normal, families, args.hkl_assign_max_angle_deg)
        peaks.append(
            PeakDescriptor(
                peak_id=peak_id,
                source=source,
                normal=normal,
                strength=float(radon.best_scores[radon_index]),
                bandwidth_deg=float(radon.scales_deg[int(radon.best_scale_index[radon_index])]),
                asymmetry=asymmetry,
                profile=profile,
                hkl=hkl,
                hkl_angle_deg=float(hkl_angle),
            )
        )
    return peaks


def peaks_to_arrays(peaks: list[PeakDescriptor]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray([peak.normal for peak in peaks], dtype=np.float32)
    strength = np.asarray([peak.strength for peak in peaks], dtype=np.float32)
    bandwidth = np.asarray([peak.bandwidth_deg for peak in peaks], dtype=np.float32)
    asymmetry = np.asarray([peak.asymmetry for peak in peaks], dtype=np.float32)
    profiles = np.asarray([peak.profile for peak in peaks], dtype=np.float32)
    return normals, strength, bandwidth, asymmetry, profiles


def peak_weights(peaks: list[PeakDescriptor]) -> np.ndarray:
    strengths = np.asarray([max(1e-4, peak.strength) for peak in peaks], dtype=np.float64)
    strengths = strengths - strengths.min() + 1e-3
    return strengths / (strengths.sum() + 1e-12)


def descriptor_cost_matrix(
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    rotation: R,
    args,
) -> tuple[np.ndarray, np.ndarray]:
    exp_normals, exp_strength, exp_bandwidth, exp_asymmetry, exp_profiles = peaks_to_arrays(exp_peaks)
    std_normals, std_strength, std_bandwidth, std_asymmetry, std_profiles = peaks_to_arrays(std_peaks)
    rotated = rotation.apply(exp_normals).astype(np.float32)
    dot = np.clip(np.abs(rotated @ std_normals.T), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot)).astype(np.float32)

    strength_cost = np.abs(exp_strength[:, None] - std_strength[None, :])
    bandwidth_scale = max(float(np.ptp(np.concatenate([exp_bandwidth, std_bandwidth]))), 1.0)
    bandwidth_cost = np.abs(exp_bandwidth[:, None] - std_bandwidth[None, :]) / bandwidth_scale
    asym_cost = np.abs(np.abs(exp_asymmetry[:, None]) - np.abs(std_asymmetry[None, :]))

    profile_direct = np.linalg.norm(exp_profiles[:, None, :] - std_profiles[None, :, :], axis=2)
    profile_reverse = np.linalg.norm(exp_profiles[:, None, ::-1] - std_profiles[None, :, :], axis=2)
    profile_cost = np.minimum(profile_direct, profile_reverse) / max(1.0, math.sqrt(exp_profiles.shape[1]))

    cost = (
        args.ot_angle_weight * (angle_deg / max(args.ot_angle_scale_deg, 1e-6)) ** 2
        + args.ot_strength_weight * strength_cost
        + args.ot_bandwidth_weight * bandwidth_cost
        + args.ot_profile_weight * profile_cost
        + args.ot_asymmetry_weight * asym_cost
    )
    return cost.astype(np.float64), angle_deg.astype(np.float32)


def partial_optimal_transport(
    cost: np.ndarray,
    exp_weights: np.ndarray,
    std_weights: np.ndarray,
    transported_mass: float,
) -> tuple[np.ndarray, float]:
    ne, ns = cost.shape
    mass = float(min(max(transported_mass, 1e-4), exp_weights.sum(), std_weights.sum()))
    c = cost.reshape(-1)

    a_ub = []
    b_ub = []
    for i in range(ne):
        row = np.zeros(ne * ns, dtype=np.float64)
        row[i * ns : (i + 1) * ns] = 1.0
        a_ub.append(row)
        b_ub.append(float(exp_weights[i]))
    for j in range(ns):
        col = np.zeros(ne * ns, dtype=np.float64)
        col[j::ns] = 1.0
        a_ub.append(col)
        b_ub.append(float(std_weights[j]))
    a_eq = np.ones((1, ne * ns), dtype=np.float64)
    b_eq = np.asarray([mass], dtype=np.float64)

    result = linprog(
        c,
        A_ub=np.asarray(a_ub, dtype=np.float64),
        b_ub=np.asarray(b_ub, dtype=np.float64),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        plan = np.zeros_like(cost, dtype=np.float64)
        return plan, float("inf")
    plan = result.x.reshape(ne, ns)
    return plan, float(result.fun / max(mass, 1e-12))


def extract_transport_matches(
    plan: np.ndarray,
    cost: np.ndarray,
    angle_deg: np.ndarray,
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    min_mass: float,
    max_angle_deg: float,
) -> list[TransportMatch]:
    matches: list[TransportMatch] = []
    for i, j in zip(*np.where(plan >= min_mass)):
        if float(angle_deg[i, j]) > max_angle_deg:
            continue
        matches.append(
            TransportMatch(
                exp_peak_id=int(exp_peaks[i].peak_id),
                std_peak_id=int(std_peaks[j].peak_id),
                mass=float(plan[i, j]),
                cost=float(cost[i, j]),
                angle_deg=float(angle_deg[i, j]),
                hkl=std_peaks[j].hkl,
            )
        )
    matches.sort(key=lambda row: (-row.mass, row.cost))
    return matches


def triangle_signature(normals: np.ndarray, triple: tuple[int, int, int]) -> np.ndarray:
    i, j, k = triple
    angles = [
        plane_angle_deg(normals[i], normals[j]),
        plane_angle_deg(normals[i], normals[k]),
        plane_angle_deg(normals[j], normals[k]),
    ]
    return np.sort(np.asarray(angles, dtype=np.float32))


def triplet_rotation(
    exp_normals: np.ndarray,
    std_normals: np.ndarray,
    exp_triple: tuple[int, int, int],
    std_ordered: tuple[int, int, int],
    signs: tuple[float, float, float],
) -> tuple[R, float]:
    source = exp_normals[list(exp_triple)].astype(np.float64)
    target = std_normals[list(std_ordered)].astype(np.float64) * np.asarray(signs, dtype=np.float64)[:, None]
    rotation, _ = R.align_vectors(target, source)
    residuals = [
        plane_angle_deg(rotation.apply(source[row]), target[row])
        for row in range(3)
    ]
    return rotation, float(np.mean(residuals))


def fast_candidate_score(rotation: R, exp_peaks: list[PeakDescriptor], std_peaks: list[PeakDescriptor]) -> float:
    exp_normals = np.asarray([peak.normal for peak in exp_peaks], dtype=np.float32)
    std_normals = np.asarray([peak.normal for peak in std_peaks], dtype=np.float32)
    rotated = rotation.apply(exp_normals).astype(np.float32)
    angle = np.degrees(np.arccos(np.clip(np.abs(rotated @ std_normals.T), -1.0, 1.0)))
    nearest = np.sort(angle.min(axis=1))[: max(3, min(8, len(exp_peaks)))]
    return float(np.mean(nearest))


def generate_triangle_candidates(
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    args,
) -> list[CandidateOrientation]:
    exp_normals = np.asarray([peak.normal for peak in exp_peaks], dtype=np.float32)
    std_normals = np.asarray([peak.normal for peak in std_peaks], dtype=np.float32)
    exp_top = list(range(min(args.triangle_top_peaks, len(exp_peaks))))
    std_top = list(range(min(args.triangle_top_peaks, len(std_peaks))))
    exp_triples = list(itertools.combinations(exp_top, 3))
    std_triples = list(itertools.combinations(std_top, 3))
    pair_rows = []
    for exp_triple in exp_triples:
        exp_sig = triangle_signature(exp_normals, exp_triple)
        for std_triple in std_triples:
            std_sig = triangle_signature(std_normals, std_triple)
            rms = float(np.sqrt(np.mean((exp_sig - std_sig) ** 2)))
            pair_rows.append((rms, exp_triple, std_triple))
    pair_rows.sort(key=lambda row: row[0])

    candidates: list[CandidateOrientation] = []
    seen: set[tuple[float, float, float, float]] = set()
    candidate_id = 0
    for triangle_rms, exp_triple, std_triple in pair_rows[: args.max_triangle_pairs]:
        if triangle_rms > args.triangle_rms_max_deg:
            continue
        for std_ordered in itertools.permutations(std_triple, 3):
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                try:
                    rotation, triplet_residual = triplet_rotation(exp_normals, std_normals, exp_triple, std_ordered, signs)
                except Exception:
                    continue
                if triplet_residual > args.triplet_residual_max_deg:
                    continue
                quat = rotation.as_quat()
                if quat[3] < 0:
                    quat *= -1.0
                key = tuple(np.round(quat, 3).astype(float))
                if key in seen:
                    continue
                seen.add(key)
                fast = fast_candidate_score(rotation, exp_peaks, std_peaks)
                candidates.append(
                    CandidateOrientation(
                        candidate_id=candidate_id,
                        rotation=rotation,
                        triangle_rms_deg=float(triangle_rms),
                        triplet_exp=tuple(int(v) for v in exp_triple),
                        triplet_std=tuple(int(v) for v in std_ordered),
                        triplet_residual_deg=float(triplet_residual),
                        fast_score=float(fast),
                    )
                )
                candidate_id += 1
    candidates.sort(key=lambda row: (row.fast_score, row.triangle_rms_deg, row.triplet_residual_deg))
    return candidates[: args.max_orientation_candidates]


def evaluate_candidates_with_ot(
    candidates: list[CandidateOrientation],
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    prepared: PreparedPattern,
    master: MasterSphere,
    args,
) -> tuple[CandidateOrientation, np.ndarray, list[TransportMatch]]:
    if not candidates:
        raise RuntimeError("No triangle-invariant orientation candidates were generated")
    exp_weights = peak_weights(exp_peaks)
    std_weights = peak_weights(std_peaks)
    best_candidate: CandidateOrientation | None = None
    best_plan: np.ndarray | None = None
    best_matches: list[TransportMatch] = []
    iterator = candidates[: args.ot_candidate_count]
    if tqdm is not None:
        iterator = tqdm(iterator, desc="partial OT candidates", leave=False)
    for candidate in iterator:
        cost, angle_deg = descriptor_cost_matrix(exp_peaks, std_peaks, candidate.rotation, args)
        plan, ot_cost = partial_optimal_transport(cost, exp_weights, std_weights, args.partial_transport_mass)
        matches = extract_transport_matches(
            plan,
            cost,
            angle_deg,
            exp_peaks,
            std_peaks,
            min_mass=args.min_transport_mass,
            max_angle_deg=args.ot_match_max_angle_deg,
        )
        image_score = score_rotation(candidate.rotation, prepared.exp_points, prepared, master)
        candidate.ot_cost = float(ot_cost)
        candidate.transported_mass = float(plan.sum())
        candidate.match_count = int(len(matches))
        candidate.image_score = float(image_score)
        candidate.objective = float(ot_cost - args.candidate_image_score_weight * image_score - args.candidate_match_bonus * len(matches))
        if best_candidate is None or candidate.objective < best_candidate.objective:
            best_candidate = candidate
            best_plan = plan
            best_matches = matches
    assert best_candidate is not None and best_plan is not None
    return best_candidate, best_plan, best_matches


def matched_peak_arrays(
    matches: list[TransportMatch],
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    rotation: R,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp_by_id = {peak.peak_id: peak for peak in exp_peaks}
    std_by_id = {peak.peak_id: peak for peak in std_peaks}
    exp_normals = []
    std_normals = []
    weights = []
    for match in matches:
        exp_normal = exp_by_id[match.exp_peak_id].normal.astype(np.float64)
        std_normal = std_by_id[match.std_peak_id].normal.astype(np.float64)
        if float(np.dot(rotation.apply(exp_normal), std_normal)) < 0.0:
            std_normal = -std_normal
        exp_normals.append(exp_normal)
        std_normals.append(std_normal)
        weights.append(max(1e-5, float(match.mass)))
    if not exp_normals:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum() + 1e-12
    return np.asarray(exp_normals), np.asarray(std_normals), weights_arr


def optimize_orientation_pc(
    initial_rotation: R,
    matches: list[TransportMatch],
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    prepared: PreparedPattern,
    master: MasterSphere,
    args,
) -> tuple[MatchResult, dict, list[dict]]:
    height, width = prepared.image.shape
    exp_normals, std_normals, match_weights = matched_peak_arrays(matches, exp_peaks, std_peaks, initial_rotation)
    trace: list[dict] = []
    eval_count = 0

    def build_state(params: np.ndarray) -> tuple[R, PreparedPattern, dict]:
        delta = R.from_rotvec(params[:3])
        rotation = delta * initial_rotation
        pc = corrected_pc(prepared.bundle.pc, float(params[3]), float(params[4]), float(params[5]), height, width)
        candidate_prepared = prepared_with_pc(prepared, pc)
        state = {
            "rotvec_x_deg": float(np.degrees(params[0])),
            "rotvec_y_deg": float(np.degrees(params[1])),
            "rotvec_z_deg": float(np.degrees(params[2])),
            "delta_angle_deg": float(np.degrees(np.linalg.norm(params[:3]))),
            "dx_px": float(params[3]),
            "dy_px": float(params[4]),
            "radius_scale": float(params[5]),
            "pcx": float(pc[0]),
            "pcy": float(pc[1]),
            "pcz": float(pc[2]),
        }
        return rotation, candidate_prepared, state

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count
        rotation, candidate_prepared, state = build_state(params)
        image_score = score_rotation(rotation, candidate_prepared.exp_points, candidate_prepared, master)
        peak_angle_deg = float("nan")
        peak_loss = 0.0
        if len(exp_normals):
            rotated = rotation.apply(exp_normals)
            dots = np.sum(rotated * std_normals, axis=1)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            peak_angle_deg = float(np.degrees(np.sum(match_weights * angles)))
            peak_loss = float(np.sum(match_weights * (angles / max(np.radians(args.refine_peak_angle_scale_deg), 1e-8)) ** 2))
        pc_regularization = (
            (state["dx_px"] / max(args.pc_regularization_px, 1e-6)) ** 2
            + (state["dy_px"] / max(args.pc_regularization_px, 1e-6)) ** 2
            + ((state["radius_scale"] - 1.0) / max(args.radius_regularization, 1e-6)) ** 2
        )
        rot_regularization = (np.linalg.norm(params[:3]) / max(np.radians(args.rotation_regularization_deg), 1e-8)) ** 2
        loss = (
            -args.refine_image_score_weight * image_score
            + args.refine_peak_weight * peak_loss
            + args.refine_pc_regularization_weight * pc_regularization
            + args.refine_rotation_regularization_weight * rot_regularization
        )
        trace.append(
            {
                "evaluation": eval_count,
                "loss": float(loss),
                "image_score": float(image_score),
                "peak_mean_angle_deg": float(peak_angle_deg),
                **state,
            }
        )
        eval_count += 1
        return float(loss)

    x0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    bounds = Bounds(
        [
            -np.radians(args.refine_rotation_bound_deg),
            -np.radians(args.refine_rotation_bound_deg),
            -np.radians(args.refine_rotation_bound_deg),
            -args.refine_pc_bound_px,
            -args.refine_pc_bound_px,
            args.refine_radius_min,
        ],
        [
            np.radians(args.refine_rotation_bound_deg),
            np.radians(args.refine_rotation_bound_deg),
            np.radians(args.refine_rotation_bound_deg),
            args.refine_pc_bound_px,
            args.refine_pc_bound_px,
            args.refine_radius_max,
        ],
    )
    result = minimize(
        objective,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": args.refine_maxiter, "xtol": args.refine_xtol, "ftol": args.refine_ftol, "disp": False},
    )
    rotation, final_prepared, state = build_state(result.x)
    final_score = score_rotation(rotation, final_prepared.exp_points, final_prepared, master)
    final = MatchResult(
        label="spherical-radon-graph-refined",
        score=float(final_score),
        rotation=rotation,
        convention_name="graph_SO3_identity_detector",
        detector_transform=IDENTITY_TRANSFORM,
        prepared=final_prepared,
    )
    summary = {
        **state,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "optimizer_nfev": int(result.nfev),
        "final_image_score": float(final_score),
    }
    return final, summary, trace


def final_matches_for_rotation(
    rotation: R,
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    args,
) -> tuple[np.ndarray, list[TransportMatch], float]:
    cost, angle_deg = descriptor_cost_matrix(exp_peaks, std_peaks, rotation, args)
    plan, ot_cost = partial_optimal_transport(cost, peak_weights(exp_peaks), peak_weights(std_peaks), args.partial_transport_mass)
    matches = extract_transport_matches(
        plan,
        cost,
        angle_deg,
        exp_peaks,
        std_peaks,
        min_mass=args.min_transport_mass,
        max_angle_deg=args.ot_match_max_angle_deg,
    )
    return plan, matches, ot_cost


def peak_lon_colat(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    colat = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))
    return lon, colat


def save_experiment_backprojection(prepared: PreparedPattern, products: dict[str, np.ndarray], out_path: Path, lon_count: int, colat_count: int) -> None:
    vectors = prepared.full_points_grid[prepared.valid_mask]
    raw_projection, raw_mask = project_to_equirect(vectors, detector_raw_display(prepared)[prepared.valid_mask], lon_count, colat_count)
    line_projection, line_mask = project_to_equirect(vectors, prepared.image_band_score[prepared.valid_mask], lon_count, colat_count)

    fig = plt.figure(figsize=(15.0, 9.0))
    ax0 = fig.add_subplot(221)
    ax0.imshow(detector_raw_display(prepared), cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Raw detector pattern")
    ax0.axis("off")

    ax1 = fig.add_subplot(222)
    ax1.imshow(prepared.image_band_score, cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Detector line response used for experimental spherical Radon")
    ax1.axis("off")

    ax2 = fig.add_subplot(223)
    extent = [-180, 180, 180, 0]
    ax2.imshow(raw_projection, cmap="gray", extent=extent, aspect="auto", alpha=np.where(raw_mask, 1.0, 0.0))
    ax2.set_title("Experimental pattern back-projected to sphere")
    ax2.set_xlabel("longitude (deg)")
    ax2.set_ylabel("colatitude (deg)")

    ax3 = fig.add_subplot(224, projection="3d")
    plot_pattern_patch(
        ax3,
        prepared.full_points_grid,
        products["raw_percentile"],
        prepared.valid_mask,
        "3D experimental sphere patch",
        curves=None,
        draw_reference_sphere=True,
    )
    fig.suptitle(
        f"Step 1: experimental pattern backprojection | idx={prepared.bundle.index} "
        f"PC=({prepared.bundle.pc[0]:.4f}, {prepared.bundle.pc[1]:.4f}, {prepared.bundle.pc[2]:.4f})",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    extra = out_path.with_name(out_path.stem + "_line_response_map.png")
    fig2, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.imshow(line_projection, cmap="magma", extent=extent, aspect="auto", alpha=np.where(line_mask, 1.0, 0.0))
    ax.set_title("Experimental line response on sphere")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("colatitude (deg)")
    fig2.tight_layout()
    fig2.savefig(extra, dpi=220, bbox_inches="tight")
    plt.close(fig2)


def save_radon_maps(
    exp_radon: SphericalRadonResult,
    std_radon: SphericalRadonResult,
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.2))
    panels = [
        ("Experimental spherical Hough/Radon response", exp_radon, exp_peaks),
        ("Master spherical Hough/Radon response", std_radon, std_peaks),
    ]
    for row, (title, radon, peaks) in enumerate(panels):
        lon, colat = peak_lon_colat(radon.normals)
        sc = axes[row, 0].scatter(lon, colat, c=radon.best_scores, s=5, cmap="viridis", linewidths=0)
        peak_normals = np.asarray([peak.normal for peak in peaks], dtype=np.float32)
        if len(peak_normals):
            peak_lon, peak_colat = peak_lon_colat(peak_normals)
            axes[row, 0].scatter(peak_lon, peak_colat, c="white", edgecolors="red", s=58, linewidths=1.2)
            for peak, x, y in zip(peaks, peak_lon, peak_colat):
                label = f"{peak.peak_id}" if peak.hkl == "unassigned" else f"{peak.peak_id}:{peak.hkl}"
                axes[row, 0].text(x, y, label, fontsize=7, color="black", ha="center", va="center")
        axes[row, 0].set_xlim(-180, 180)
        axes[row, 0].set_ylim(90, 0)
        axes[row, 0].set_xlabel("normal longitude (deg)")
        axes[row, 0].set_ylabel("normal colatitude, hemisphere (deg)")
        axes[row, 0].set_title(title)
        fig.colorbar(sc, ax=axes[row, 0], fraction=0.035, pad=0.02)

        for scale_index, scale in enumerate(radon.scales_deg):
            axes[row, 1].hist(radon.scores_by_scale[scale_index], bins=60, alpha=0.45, label=f"{scale:g} deg")
        axes[row, 1].set_title(f"Multi-scale response distribution: {title.split()[0]}")
        axes[row, 1].set_xlabel("Radon response")
        axes[row, 1].set_ylabel("normal count")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(alpha=0.2)
    fig.suptitle("Steps 2-3: multi-scale spherical Hough/Radon transform and peak extraction", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_peak_descriptors(exp_peaks: list[PeakDescriptor], std_peaks: list[PeakDescriptor], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8))
    for row, (name, peaks) in enumerate((("Experimental peaks", exp_peaks), ("Master peaks", std_peaks))):
        ids = [peak.peak_id for peak in peaks]
        strengths = [peak.strength for peak in peaks]
        widths = [peak.bandwidth_deg for peak in peaks]
        asym = [peak.asymmetry for peak in peaks]
        axes[row, 0].bar(ids, strengths, color="#2563eb")
        axes[row, 0].set_title(f"{name}: peak strength")
        axes[row, 0].set_xlabel("peak id")
        axes[row, 0].grid(axis="y", alpha=0.25)
        axes[row, 1].bar(ids, widths, color="#16a34a")
        axes[row, 1].set_title(f"{name}: selected bandwidth")
        axes[row, 1].set_xlabel("peak id")
        axes[row, 1].set_ylabel("deg")
        axes[row, 1].grid(axis="y", alpha=0.25)
        axes[row, 2].bar(ids, asym, color="#f97316")
        axes[row, 2].set_title(f"{name}: profile asymmetry")
        axes[row, 2].set_xlabel("peak id")
        axes[row, 2].grid(axis="y", alpha=0.25)

    fig.suptitle("Step 4: peak descriptors = normal + strength + bandwidth + profile + asymmetry", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    profile_path = out_path.with_name(out_path.stem + "_profiles.png")
    fig2, axes2 = plt.subplots(1, 2, figsize=(15.0, 5.2))
    for ax, name, peaks in ((axes2[0], "Experimental peak profiles", exp_peaks), (axes2[1], "Master peak profiles", std_peaks)):
        for peak in peaks[: min(12, len(peaks))]:
            ax.plot(peak.profile, linewidth=1.2, alpha=0.8, label=str(peak.peak_id))
        ax.set_title(name)
        ax.set_xlabel("signed normal-distance bin")
        ax.set_ylabel("normalized profile")
        ax.grid(alpha=0.22)
    axes2[0].legend(ncol=3, fontsize=7)
    axes2[1].legend(ncol=3, fontsize=7)
    fig2.tight_layout()
    fig2.savefig(profile_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)


def pairwise_angle_matrix(peaks: list[PeakDescriptor]) -> np.ndarray:
    normals = np.asarray([peak.normal for peak in peaks], dtype=np.float32)
    dot = np.clip(np.abs(normals @ normals.T), -1.0, 1.0)
    return np.degrees(np.arccos(dot)).astype(np.float32)


def save_peak_graphs(exp_peaks: list[PeakDescriptor], std_peaks: list[PeakDescriptor], out_path: Path) -> None:
    exp_matrix = pairwise_angle_matrix(exp_peaks)
    std_matrix = pairwise_angle_matrix(std_peaks)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, matrix, title in (
        (axes[0], exp_matrix, "Experimental peak graph: pairwise plane-normal angles"),
        (axes[1], std_matrix, "Master peak graph: pairwise plane-normal angles"),
    ):
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=90.0)
        ax.set_title(title)
        ax.set_xlabel("peak id")
        ax.set_ylabel("peak id")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Step 5: standard and experimental peak graphs", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_triangle_candidates(candidates: list[CandidateOrientation], selected: CandidateOrientation, out_path: Path) -> None:
    rows = candidates[: min(24, len(candidates))]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    x = np.arange(len(rows))
    axes[0].bar(x, [row.triangle_rms_deg for row in rows], color="#64748b")
    axes[0].set_title("Triangle invariant RMS")
    axes[0].set_xlabel("candidate rank")
    axes[0].set_ylabel("deg")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row.ot_cost for row in rows], color="#2563eb")
    axes[1].set_title("Partial OT cost")
    axes[1].set_xlabel("candidate rank")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, [row.image_score for row in rows], color="#16a34a")
    axes[2].set_title("Raw sphere image score")
    axes[2].set_xlabel("candidate rank")
    axes[2].grid(axis="y", alpha=0.25)
    selected_rank = next((i for i, row in enumerate(rows) if row.candidate_id == selected.candidate_id), None)
    if selected_rank is not None:
        for ax in axes:
            ax.axvline(selected_rank, color="red", linestyle="--", linewidth=1.2)
    fig.suptitle("Step 6: orientation candidates from triangle invariants", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_transport_matching(
    plan: np.ndarray,
    matches: list[TransportMatch],
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    rotation: R,
    out_path: Path,
) -> None:
    cost, angle_deg = descriptor_cost_matrix(exp_peaks, std_peaks, rotation, argparse.Namespace(
        ot_angle_weight=1.0,
        ot_angle_scale_deg=1.0,
        ot_strength_weight=0.0,
        ot_bandwidth_weight=0.0,
        ot_profile_weight=0.0,
        ot_asymmetry_weight=0.0,
    ))
    del cost
    fig = plt.figure(figsize=(16.0, 6.4))
    ax0 = fig.add_subplot(121)
    im = ax0.imshow(angle_deg, cmap="magma", vmin=0.0, vmax=min(35.0, float(np.nanmax(angle_deg))))
    ax0.set_title("Plane-normal angular cost after selected orientation")
    ax0.set_xlabel("master peak id")
    ax0.set_ylabel("experimental peak id")
    for match in matches:
        ax0.scatter([match.std_peak_id], [match.exp_peak_id], marker="s", facecolors="none", edgecolors="cyan", s=95, linewidths=1.5)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(122, projection="3d")
    exp_normals = np.asarray([peak.normal for peak in exp_peaks], dtype=np.float32)
    std_normals = np.asarray([peak.normal for peak in std_peaks], dtype=np.float32)
    exp_rot = rotation.apply(exp_normals).astype(np.float32)
    ax1.scatter(std_normals[:, 0], std_normals[:, 1], std_normals[:, 2], c="#1f77b4", s=52, label="master peaks")
    ax1.scatter(exp_rot[:, 0], exp_rot[:, 1], exp_rot[:, 2], c="#ff7f0e", s=42, marker="^", label="rotated experimental peaks")
    exp_by_id = {peak.peak_id: peak for peak in exp_peaks}
    std_by_id = {peak.peak_id: peak for peak in std_peaks}
    for match in matches:
        e = rotation.apply(exp_by_id[match.exp_peak_id].normal)
        s = std_by_id[match.std_peak_id].normal
        if np.dot(e, s) < 0:
            s = -s
        ax1.plot([e[0], s[0]], [e[1], s[1]], [e[2], s[2]], color="black", linewidth=1.1, alpha=0.62)
    set_3d_sphere_axes(ax1, "Partial OT peak matching on normal sphere", view_vectors=np.vstack([std_normals, exp_rot]))
    ax1.legend(loc="upper left", fontsize=8)
    fig.suptitle("Step 7: partial optimal transport global peak matching", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    plan_path = out_path.with_name(out_path.stem + "_transport_plan.png")
    fig2, ax = plt.subplots(figsize=(7.0, 5.8))
    im2 = ax.imshow(plan, cmap="viridis")
    ax.set_title("Partial OT transport mass")
    ax.set_xlabel("master peak id")
    ax.set_ylabel("experimental peak id")
    fig2.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
    fig2.tight_layout()
    fig2.savefig(plan_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)


def save_refinement_trace(trace: list[dict], out_path: Path) -> None:
    if not trace:
        return
    evals = [row["evaluation"] for row in trace]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.2))
    axes[0, 0].plot(evals, [row["loss"] for row in trace], color="#2563eb")
    axes[0, 0].set_title("Joint loss")
    axes[0, 1].plot(evals, [row["image_score"] for row in trace], color="#16a34a")
    axes[0, 1].set_title("Raw sphere image score")
    axes[0, 2].plot(evals, [row["peak_mean_angle_deg"] for row in trace], color="#7c3aed")
    axes[0, 2].set_title("Matched peak mean angle")
    axes[1, 0].plot(evals, [row["delta_angle_deg"] for row in trace], color="#f97316")
    axes[1, 0].set_title("Orientation delta")
    axes[1, 1].plot(evals, [row["dx_px"] for row in trace], label="dx", color="#0f766e")
    axes[1, 1].plot(evals, [row["dy_px"] for row in trace], label="dy", color="#dc2626")
    axes[1, 1].set_title("PC xy shift")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(evals, [row["radius_scale"] for row in trace], color="#334155")
    axes[1, 2].set_title("PCz / radius scale")
    for ax in axes.ravel():
        ax.set_xlabel("evaluation")
        ax.grid(alpha=0.25)
    fig.suptitle("Step 8: joint optimization of orientation + pattern center", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_hkl_explanation(
    result: MatchResult,
    master: MasterSphere,
    matches: list[TransportMatch],
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    texture, lon_grid, colat_grid, _ = sphere_texture(master, lon_count, colat_count)
    extent = [
        float(np.degrees(lon_grid.min())),
        float(np.degrees(lon_grid.max())),
        float(np.degrees(colat_grid.max())),
        float(np.degrees(colat_grid.min())),
    ]
    exp_by_id = {peak.peak_id: peak for peak in exp_peaks}
    std_by_id = {peak.peak_id: peak for peak in std_peaks}
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.0))
    axes[0].imshow(detector_raw_display(result.prepared), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original detector pattern")
    axes[0].axis("off")
    axes[1].imshow(texture, cmap="gray", extent=extent, aspect="auto", vmin=0.0, vmax=1.0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(matches))))
    for color, match in zip(colors, matches):
        exp_peak = exp_by_id[match.exp_peak_id]
        std_peak = std_by_id[match.std_peak_id]
        normal = std_peak.normal.astype(np.float32)
        if np.dot(result.rotation.apply(exp_peak.normal), normal) < 0:
            normal = -normal
        circle = great_circle_from_normal(normal, samples=720)
        lon = np.degrees(np.arctan2(circle[:, 1], circle[:, 0]))
        colat = np.degrees(np.arccos(np.clip(circle[:, 2], -1.0, 1.0)))
        jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
        for part in np.split(np.arange(len(lon)), jumps):
            if len(part) > 2:
                axes[1].plot(lon[part], colat[part], color=color, linewidth=1.3, alpha=0.9)
        marker = result.rotation.apply(exp_peak.normal)
        marker_lon = float(np.degrees(np.arctan2(marker[1], marker[0])))
        marker_colat = float(np.degrees(np.arccos(np.clip(marker[2], -1.0, 1.0))))
        axes[1].scatter([marker_lon], [marker_colat], color=color, edgecolor="black", s=52)
        axes[1].text(
            marker_lon,
            marker_colat,
            f"E{match.exp_peak_id}->S{match.std_peak_id} {match.hkl}",
            fontsize=7,
            color="white",
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.45, "boxstyle": "round,pad=0.2", "linewidth": 0},
        )
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(180, 0)
    axes[1].set_xlabel("longitude on master sphere (deg)")
    axes[1].set_ylabel("colatitude (deg)")
    axes[1].set_title("Matched peak normals and HKL interpretation")
    fig.suptitle("Step 10: output orientation, phase, PC and matched {hkl} explanation", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def great_circle_from_normal(normal: np.ndarray, samples: int = 361) -> np.ndarray:
    normal = normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)
    t = np.linspace(0.0, 2.0 * np.pi, samples, dtype=np.float32)
    return (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :]).astype(np.float32)


def save_summary_csvs(
    out_dir: Path,
    exp_peaks: list[PeakDescriptor],
    std_peaks: list[PeakDescriptor],
    candidates: list[CandidateOrientation],
    matches: list[TransportMatch],
    trace: list[dict],
) -> None:
    write_rows_csv([peak.to_row() for peak in exp_peaks], out_dir / "03a_experimental_peak_descriptors.csv")
    write_rows_csv([peak.to_row() for peak in std_peaks], out_dir / "03b_master_peak_descriptors.csv")
    write_rows_csv([candidate.to_row() for candidate in candidates], out_dir / "05_triangle_orientation_candidates.csv")
    write_rows_csv([match.to_row() for match in matches], out_dir / "07_partial_ot_matches.csv")
    write_rows_csv(trace, out_dir / "08_joint_orientation_pc_trace.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype pipeline: experimental sphere backprojection, multi-scale spherical Hough/Radon peaks, "
            "triangle-invariant orientation candidates, partial OT peak matching, and joint orientation/PC refinement."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=2661)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "spherical_radon_graph_pipeline")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=6500)
    parser.add_argument("--line-variant", default="auto")
    parser.add_argument("--radon-scales-deg", default="1.4,2.4,3.8")
    parser.add_argument("--normal-count", type=int, default=2600)
    parser.add_argument("--master-sample-count", type=int, default=9000)
    parser.add_argument("--experiment-sample-count", type=int, default=11000)
    parser.add_argument("--radon-chunk-size", type=int, default=192)
    parser.add_argument("--peak-count", type=int, default=24)
    parser.add_argument("--peak-min-separation-deg", type=float, default=5.5)
    parser.add_argument("--peak-min-score-quantile", type=float, default=0.68)
    parser.add_argument("--profile-width-deg", type=float, default=8.0)
    parser.add_argument("--profile-bins", type=int, default=17)
    parser.add_argument("--triangle-top-peaks", type=int, default=11)
    parser.add_argument("--max-triangle-pairs", type=int, default=90)
    parser.add_argument("--triangle-rms-max-deg", type=float, default=10.0)
    parser.add_argument("--triplet-residual-max-deg", type=float, default=8.0)
    parser.add_argument("--max-orientation-candidates", type=int, default=180)
    parser.add_argument("--ot-candidate-count", type=int, default=35)
    parser.add_argument("--partial-transport-mass", type=float, default=0.80)
    parser.add_argument("--min-transport-mass", type=float, default=0.006)
    parser.add_argument("--ot-match-max-angle-deg", type=float, default=24.0)
    parser.add_argument("--ot-angle-weight", type=float, default=1.0)
    parser.add_argument("--ot-angle-scale-deg", type=float, default=7.5)
    parser.add_argument("--ot-strength-weight", type=float, default=0.25)
    parser.add_argument("--ot-bandwidth-weight", type=float, default=0.20)
    parser.add_argument("--ot-profile-weight", type=float, default=0.45)
    parser.add_argument("--ot-asymmetry-weight", type=float, default=0.12)
    parser.add_argument("--candidate-image-score-weight", type=float, default=8.0)
    parser.add_argument("--candidate-match-bonus", type=float, default=0.02)
    parser.add_argument("--hkl-assign-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--refine-rotation-bound-deg", type=float, default=5.0)
    parser.add_argument("--refine-pc-bound-px", type=float, default=10.0)
    parser.add_argument("--refine-radius-min", type=float, default=0.96)
    parser.add_argument("--refine-radius-max", type=float, default=1.04)
    parser.add_argument("--refine-maxiter", type=int, default=90)
    parser.add_argument("--refine-xtol", type=float, default=3e-4)
    parser.add_argument("--refine-ftol", type=float, default=3e-4)
    parser.add_argument("--refine-image-score-weight", type=float, default=1.0)
    parser.add_argument("--refine-peak-weight", type=float, default=0.08)
    parser.add_argument("--refine-peak-angle-scale-deg", type=float, default=8.0)
    parser.add_argument("--refine-pc-regularization-weight", type=float, default=0.012)
    parser.add_argument("--refine-rotation-regularization-weight", type=float, default=0.01)
    parser.add_argument("--pc-regularization-px", type=float, default=10.0)
    parser.add_argument("--radius-regularization", type=float, default=0.04)
    parser.add_argument("--rotation-regularization-deg", type=float, default=5.0)
    parser.add_argument("--sphere-lon-count", type=int, default=520)
    parser.add_argument("--sphere-colat-count", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.50)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    map_spec = default_map_specs(args.data_dir)[args.map]
    master_h5 = resolve_master_path(args.master_h5)
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading pattern: {map_spec.label}, index={args.index}")
    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    products = build_preprocessing_products(
        bundle.pattern_u16,
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )
    prepared, variant_diagnostics = prepare_pattern(
        bundle=bundle,
        weights=weights,
        label="spherical-radon-graph",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=args.line_variant,
    )

    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    phase_id = int(bundle.ang_record.get("Phase", 1))
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)

    save_experiment_backprojection(
        prepared,
        products,
        out_dir / "01_experimental_pattern_backprojected_to_sphere.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    normal_grid = fibonacci_sphere(args.normal_count, hemisphere=True)
    scales = parse_float_list(args.radon_scales_deg)

    exp_all_vectors = prepared.full_points_grid[prepared.valid_mask]
    exp_all_values = prepared.image_band_score[prepared.valid_mask]
    exp_idx = sample_rows(exp_all_values, args.experiment_sample_count, args.seed + 11)
    exp_vectors = exp_all_vectors[exp_idx]
    exp_values = exp_all_values[exp_idx]

    master_vectors = fibonacci_sphere(args.master_sample_count, hemisphere=False)
    master_values = master.sample_band(master_vectors)

    exp_radon = spherical_radon_transform(
        exp_vectors,
        exp_values,
        normal_grid,
        scales,
        chunk_size=args.radon_chunk_size,
        desc="experimental spherical Radon",
    )
    std_radon = spherical_radon_transform(
        master_vectors,
        master_values,
        normal_grid,
        scales,
        chunk_size=args.radon_chunk_size,
        desc="master spherical Radon",
    )

    exp_peak_indices = greedy_peak_pick(
        exp_radon,
        peak_count=args.peak_count,
        min_separation_deg=args.peak_min_separation_deg,
        min_score_quantile=args.peak_min_score_quantile,
    )
    std_peak_indices = greedy_peak_pick(
        std_radon,
        peak_count=args.peak_count + 8,
        min_separation_deg=args.peak_min_separation_deg,
        min_score_quantile=args.peak_min_score_quantile,
    )
    exp_peaks = build_peak_descriptors("experimental", exp_radon, exp_peak_indices, exp_vectors, exp_values, args)
    std_peaks = build_peak_descriptors("master", std_radon, std_peak_indices, master_vectors, master_values, args, families)

    save_radon_maps(exp_radon, std_radon, exp_peaks, std_peaks, out_dir / "02_multiscale_spherical_radon_and_peaks.png")
    save_peak_descriptors(exp_peaks, std_peaks, out_dir / "03_peak_descriptors.png")
    save_peak_graphs(exp_peaks, std_peaks, out_dir / "04_standard_and_experimental_peak_graphs.png")

    print("Generating triangle-invariant orientation candidates...")
    candidates = generate_triangle_candidates(exp_peaks, std_peaks, args)
    print(f"Generated {len(candidates)} orientation candidates")
    selected_candidate, initial_plan, initial_matches = evaluate_candidates_with_ot(candidates, exp_peaks, std_peaks, prepared, master, args)
    candidates.sort(key=lambda row: row.objective)
    save_triangle_candidates(candidates, selected_candidate, out_dir / "05_triangle_orientation_candidates.png")
    save_transport_matching(
        initial_plan,
        initial_matches,
        exp_peaks,
        std_peaks,
        selected_candidate.rotation,
        out_dir / "06_partial_ot_peak_matching_initial.png",
    )

    print("Joint optimizing orientation + pattern center...")
    final_result, refinement_summary, trace = optimize_orientation_pc(
        selected_candidate.rotation,
        initial_matches,
        exp_peaks,
        std_peaks,
        prepared,
        master,
        args,
    )
    save_refinement_trace(trace, out_dir / "08_joint_orientation_pc_refinement_trace.png")

    final_plan, final_matches, final_ot_cost = final_matches_for_rotation(final_result.rotation, exp_peaks, std_peaks, args)
    save_transport_matching(
        final_plan,
        final_matches,
        exp_peaks,
        std_peaks,
        final_result.rotation,
        out_dir / "08b_partial_ot_peak_matching_refined.png",
    )

    save_final_spatial_visualization(
        final_result,
        master,
        products,
        out_dir / "09_final_raw_sphere_local_refinement.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_hkl_explanation(
        final_result,
        master,
        final_matches,
        exp_peaks,
        std_peaks,
        out_dir / "10_orientation_phase_pc_hkl_explanation.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_summary_csvs(out_dir, exp_peaks, std_peaks, candidates, final_matches, trace)

    hkl_rows = []
    exp_by_id = {peak.peak_id: peak for peak in exp_peaks}
    std_by_id = {peak.peak_id: peak for peak in std_peaks}
    for match in final_matches:
        exp_peak = exp_by_id[match.exp_peak_id]
        std_peak = std_by_id[match.std_peak_id]
        hkl_rows.append(
            {
                **match.to_row(),
                "exp_strength": float(exp_peak.strength),
                "std_strength": float(std_peak.strength),
                "std_hkl_angle_deg": float(std_peak.hkl_angle_deg),
                "exp_normal_x": float(exp_peak.normal[0]),
                "exp_normal_y": float(exp_peak.normal[1]),
                "exp_normal_z": float(exp_peak.normal[2]),
                "std_normal_x": float(std_peak.normal[0]),
                "std_normal_y": float(std_peak.normal[1]),
                "std_normal_z": float(std_peak.normal[2]),
            }
        )
    write_rows_csv(hkl_rows, out_dir / "10_matched_hkl_explanation.csv")

    summary = {
        "pipeline": "spherical_radon_graph_pipeline",
        "map": map_spec.key,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "pattern_shape": list(bundle.pattern_u16.shape),
        "pattern_center_from_h5": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "phase": phase_info,
        "families": [
            {
                "label": family.label,
                "diffraction_intensity": float(family.diffraction_intensity),
                "normal_count": int(len(family.normals)),
            }
            for family in families
        ],
        "selected_candidate": selected_candidate.to_row(),
        "initial_partial_ot_match_count": int(len(initial_matches)),
        "final_partial_ot_match_count": int(len(final_matches)),
        "final_partial_ot_cost": float(final_ot_cost),
        "refinement": refinement_summary,
        "final_match_result": final_result.to_json_dict(),
        "hyperparameters": {
            "radon_scales_deg": scales,
            "normal_count": int(args.normal_count),
            "master_sample_count": int(args.master_sample_count),
            "experiment_sample_count": int(args.experiment_sample_count),
            "peak_count": int(args.peak_count),
            "partial_transport_mass": float(args.partial_transport_mass),
            "refine_rotation_bound_deg": float(args.refine_rotation_bound_deg),
            "refine_pc_bound_px": float(args.refine_pc_bound_px),
            "refine_radius_min": float(args.refine_radius_min),
            "refine_radius_max": float(args.refine_radius_max),
        },
        "outputs": {
            "step1_backprojection": str(out_dir / "01_experimental_pattern_backprojected_to_sphere.png"),
            "step2_3_radon_peaks": str(out_dir / "02_multiscale_spherical_radon_and_peaks.png"),
            "step4_descriptors": str(out_dir / "03_peak_descriptors.png"),
            "step5_graphs": str(out_dir / "04_standard_and_experimental_peak_graphs.png"),
            "step6_candidates": str(out_dir / "05_triangle_orientation_candidates.png"),
            "step7_initial_ot": str(out_dir / "06_partial_ot_peak_matching_initial.png"),
            "step8_refinement": str(out_dir / "08_joint_orientation_pc_refinement_trace.png"),
            "step9_final_local_refinement": str(out_dir / "09_final_raw_sphere_local_refinement.png"),
            "step10_hkl_explanation": str(out_dir / "10_orientation_phase_pc_hkl_explanation.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved spherical Radon graph pipeline outputs to: {out_dir}")
    print(f"Final score: {final_result.score:.5f}; final PC={refinement_summary['pcx']:.5f}, {refinement_summary['pcy']:.5f}, {refinement_summary['pcz']:.5f}")
    print(f"Final matched HKL peaks: {len(final_matches)}")


if __name__ == "__main__":
    main()
