from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

# Windows Anaconda can load Intel OpenMP through both PyTorch and scientific
# packages. Keep this workaround local to this exploratory PyTorch runner.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from batch_pc_radius_bias_correction_gpu import sample_master_channel, to_torch_master
from continuous_band_geometric_refinement import (
    band_angle_rows,
    family_by_label,
    save_band_residual_comparison,
    write_rows_csv,
)
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    MatchResult,
    MatchWeights,
    PreparedPattern,
    default_map_specs,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
    score_rotation,
    zscore_vector,
)
from labeled_band_radius_refinement import (
    HKLFamily,
    assign_labels,
    label_score,
    read_phase_hkl_families,
    save_detector_label_overlay,
    save_labeled_alignment,
)
from pc_radius_bias_correction import corrected_pc, prepared_with_pc
from visualize_calibration_pipeline import build_preprocessing_products, parse_refine_schedule, save_final_spatial_visualization

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


class DIPParamNet(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, layers: int, out_dim: int):
        super().__init__()
        modules: list[nn.Module] = []
        in_dim = noise_dim
        for _ in range(max(1, layers)):
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(nn.SiLU())
            in_dim = hidden_dim
        final = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn.Sequential(*modules)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise).squeeze(0)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def choose_point_indices(prepared: PreparedPattern, count: int, seed: int) -> np.ndarray:
    n_points = int(len(prepared.exp_points))
    if count <= 0 or count >= n_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    response = prepared.combined_response[prepared.match_mask].astype(np.float32)
    strong_count = min(count // 2, n_points)
    strong = np.argpartition(response, -strong_count)[-strong_count:]
    remaining = np.setdiff1d(np.arange(n_points), strong, assume_unique=False)
    sample_count = count - strong_count
    sampled = rng.choice(remaining, size=sample_count, replace=False) if sample_count > 0 and len(remaining) else np.array([], dtype=np.int64)
    return np.sort(np.concatenate([strong.astype(np.int64), sampled.astype(np.int64)]))


def zscore_1d(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    zero = vector.new_tensor(0.0)
    x, y, z = vector[0], vector[1], vector[2]
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec)
    k = skew_symmetric(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    small = theta < 1e-6
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta) / (theta + 1e-12))
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(theta)) / (theta**2 + 1e-12))
    return eye + a * k + b * (k @ k)


def network_raw_to_params(raw: torch.Tensor, args) -> dict[str, torch.Tensor]:
    rot_bound = torch.tensor(np.radians(args.rotation_bound_deg), dtype=raw.dtype, device=raw.device)
    rotvec = torch.tanh(raw[:3]) * rot_bound
    radius_center = 0.5 * (args.radius_min + args.radius_max)
    radius_half = 0.5 * (args.radius_max - args.radius_min)
    radius_scale = raw.new_tensor(radius_center) + torch.tanh(raw[3]) * raw.new_tensor(radius_half)
    out = {
        "rotvec": rotvec,
        "radius_scale": radius_scale,
    }
    if args.optimize_pc:
        out["dx_px"] = torch.tanh(raw[4]) * raw.new_tensor(args.pc_bound_px)
        out["dy_px"] = torch.tanh(raw[5]) * raw.new_tensor(args.pc_bound_px)
    else:
        out["dx_px"] = raw.new_tensor(0.0)
        out["dy_px"] = raw.new_tensor(0.0)
    return out


def params_to_numpy(params: dict[str, torch.Tensor]) -> dict:
    rotvec = params["rotvec"].detach().cpu().numpy().astype(float)
    return {
        "rotvec_x_deg": float(np.degrees(rotvec[0])),
        "rotvec_y_deg": float(np.degrees(rotvec[1])),
        "rotvec_z_deg": float(np.degrees(rotvec[2])),
        "delta_angle_deg": float(np.degrees(np.linalg.norm(rotvec))),
        "radius_scale": float(params["radius_scale"].detach().cpu().item()),
        "dx_px": float(params["dx_px"].detach().cpu().item()),
        "dy_px": float(params["dy_px"].detach().cpu().item()),
        "rotvec": rotvec.tolist(),
    }


def numpy_params_to_result(
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    row: dict,
) -> MatchResult:
    height, width = prepared.image.shape
    pc = corrected_pc(
        prepared.bundle.pc,
        float(row["dx_px"]),
        float(row["dy_px"]),
        float(row["radius_scale"]),
        height,
        width,
    )
    candidate_prepared = prepared_with_pc(prepared, pc)
    rotvec = np.asarray(row["rotvec"], dtype=np.float64)
    rotation = R.from_rotvec(rotvec) * initial_result.rotation
    points = candidate_prepared.exp_points @ initial_result.detector_transform.T
    score = score_rotation(rotation, points, candidate_prepared, master)
    return MatchResult(
        label="DIP-parametric-refined",
        score=float(score),
        rotation=rotation,
        convention_name=initial_result.convention_name,
        detector_transform=initial_result.detector_transform,
        prepared=candidate_prepared,
    )


def detector_points_torch(
    rows: torch.Tensor,
    cols: torch.Tensor,
    height: int,
    width: int,
    base_pc: tuple[float, float, float],
    dx_px: torch.Tensor,
    dy_px: torch.Tensor,
    radius_scale: torch.Tensor,
    detector_transform: torch.Tensor,
) -> torch.Tensor:
    pcx = rows.new_tensor(base_pc[0]) + dx_px / float(width - 1)
    pcy = rows.new_tensor(base_pc[1]) + dy_px / float(height - 1)
    pcz = rows.new_tensor(base_pc[2]) * radius_scale
    cx = pcx * float(width - 1)
    cy = pcy * float(height - 1)
    x = (cols - cx) / (pcz * float(height))
    y = -(rows - cy) / (pcz * float(height))
    z = torch.ones_like(x)
    points = torch.stack((x, y, z), dim=1)
    points = points / (torch.linalg.norm(points, dim=1, keepdim=True) + 1e-8)
    return points @ detector_transform.T


def build_torch_arrays(
    prepared: PreparedPattern,
    initial_result: MatchResult,
    families: list[HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    point_indices: np.ndarray,
    device: torch.device,
) -> dict:
    rows_np, cols_np = np.nonzero(prepared.match_mask)
    rows = torch.as_tensor(rows_np[point_indices].astype(np.float32), dtype=torch.float32, device=device)
    cols = torch.as_tensor(cols_np[point_indices].astype(np.float32), dtype=torch.float32, device=device)
    detector_transform = torch.as_tensor(initial_result.detector_transform.astype(np.float32), dtype=torch.float32, device=device)
    initial_rotation = torch.as_tensor(initial_result.rotation.as_matrix().astype(np.float32), dtype=torch.float32, device=device)

    family_lookup = family_by_label(families)
    segment_rows = []
    segment_cols = []
    target_normals = []
    band_weights = []
    band_indices = []
    for segment in prepared.line_segments:
        hkl = fixed_hkl_by_band.get(segment.band_index)
        if hkl not in family_lookup:
            continue
        segment_rows.append([segment.row0, segment.row1])
        segment_cols.append([segment.col0, segment.col1])
        target_normals.append(torch.as_tensor(family_lookup[hkl].normals.astype(np.float32), dtype=torch.float32, device=device))
        band_weights.append(max(1e-3, float(segment.band.intensity)))
        band_indices.append(int(segment.band_index))

    weights = torch.as_tensor(np.asarray(band_weights, dtype=np.float32), dtype=torch.float32, device=device)
    weights = weights / (weights.sum() + 1e-8)
    return {
        "rows": rows,
        "cols": cols,
        "exp_image_band_z": torch.as_tensor(prepared.exp_image_band_z[point_indices], dtype=torch.float32, device=device),
        "exp_intensity_z": torch.as_tensor(prepared.exp_intensity_z[point_indices], dtype=torch.float32, device=device),
        "exp_h5_band_z": torch.as_tensor(prepared.exp_h5_band_z[point_indices], dtype=torch.float32, device=device),
        "detector_transform": detector_transform,
        "initial_rotation": initial_rotation,
        "segment_rows": torch.as_tensor(np.asarray(segment_rows, dtype=np.float32), dtype=torch.float32, device=device),
        "segment_cols": torch.as_tensor(np.asarray(segment_cols, dtype=np.float32), dtype=torch.float32, device=device),
        "target_normals": target_normals,
        "band_weights": weights,
        "band_indices": band_indices,
    }


def forward_loss(
    raw: torch.Tensor,
    args,
    prepared: PreparedPattern,
    torch_data: dict,
    torch_master,
) -> tuple[torch.Tensor, dict]:
    params = network_raw_to_params(raw, args)
    delta_rotation = rotvec_to_matrix(params["rotvec"])
    rotation = delta_rotation @ torch_data["initial_rotation"]

    points = detector_points_torch(
        torch_data["rows"],
        torch_data["cols"],
        prepared.image.shape[0],
        prepared.image.shape[1],
        prepared.bundle.pc,
        params["dx_px"],
        params["dy_px"],
        params["radius_scale"],
        torch_data["detector_transform"],
    )
    rotated = points @ rotation.T
    master_band = sample_master_channel(rotated, torch_master.upper_band, torch_master.lower_band).squeeze(0)
    master_intensity = sample_master_channel(rotated, torch_master.upper_intensity, torch_master.lower_intensity).squeeze(0)
    master_band_z = zscore_1d(master_band)
    master_intensity_z = zscore_1d(master_intensity)

    image_line_loss = F.mse_loss(master_band_z, torch_data["exp_image_band_z"])
    h5_band_loss = F.mse_loss(master_band_z, torch_data["exp_h5_band_z"])
    intensity_loss = F.mse_loss(master_intensity_z, torch_data["exp_intensity_z"])

    endpoint_rows = torch_data["segment_rows"].reshape(-1)
    endpoint_cols = torch_data["segment_cols"].reshape(-1)
    endpoint_points = detector_points_torch(
        endpoint_rows,
        endpoint_cols,
        prepared.image.shape[0],
        prepared.image.shape[1],
        prepared.bundle.pc,
        params["dx_px"],
        params["dy_px"],
        params["radius_scale"],
        torch_data["detector_transform"],
    ).reshape(-1, 2, 3)
    endpoint_rotated = torch.einsum("bij,kj->bik", endpoint_points, rotation)
    normals = torch.cross(endpoint_rotated[:, 0, :], endpoint_rotated[:, 1, :], dim=1)
    normals = normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-8)

    angle_terms = []
    angle_degrees = []
    for i, family_normals in enumerate(torch_data["target_normals"]):
        dots = torch.abs(family_normals @ normals[i])
        best_dot = torch.clamp(torch.max(dots), -1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(best_dot)
        angle_terms.append(angle)
        angle_degrees.append(torch.rad2deg(angle))
    band_angles = torch.stack(angle_terms)
    band_angle_degrees = torch.stack(angle_degrees)
    band_geometry_loss = torch.sum(torch_data["band_weights"] * (band_angles / np.radians(args.band_angle_scale_deg)) ** 2)

    radius_reg = ((params["radius_scale"] - 1.0) / max(args.radius_regularization_sigma, 1e-8)) ** 2
    pc_reg = (params["dx_px"] / max(args.pc_regularization_sigma_px, 1e-8)) ** 2 + (
        params["dy_px"] / max(args.pc_regularization_sigma_px, 1e-8)
    ) ** 2
    rotation_reg = torch.sum((params["rotvec"] / max(np.radians(args.rotation_regularization_sigma_deg), 1e-8)) ** 2)

    loss = (
        args.image_line_weight * image_line_loss
        + args.h5_band_weight * h5_band_loss
        + args.intensity_weight * intensity_loss
        + args.band_geometry_weight * band_geometry_loss
        + args.radius_regularization_weight * radius_reg
        + args.pc_regularization_weight * pc_reg
        + args.rotation_regularization_weight * rotation_reg
    )
    approx_match_score = (
        args.image_line_weight * torch.mean(torch_data["exp_image_band_z"] * master_band_z)
        + args.h5_band_weight * torch.mean(torch_data["exp_h5_band_z"] * master_band_z)
        + args.intensity_weight * torch.mean(torch_data["exp_intensity_z"] * master_intensity_z)
    )
    metrics = {
        **params_to_numpy(params),
        "loss": float(loss.detach().cpu().item()),
        "image_line_loss": float(image_line_loss.detach().cpu().item()),
        "h5_band_loss": float(h5_band_loss.detach().cpu().item()),
        "intensity_loss": float(intensity_loss.detach().cpu().item()),
        "band_geometry_loss": float(band_geometry_loss.detach().cpu().item()),
        "radius_regularization": float(radius_reg.detach().cpu().item()),
        "pc_regularization": float(pc_reg.detach().cpu().item()),
        "rotation_regularization": float(rotation_reg.detach().cpu().item()),
        "approx_match_score": float(approx_match_score.detach().cpu().item()),
        "mean_band_angle_deg": float(torch.mean(band_angle_degrees).detach().cpu().item()),
        "max_band_angle_deg": float(torch.max(band_angle_degrees).detach().cpu().item()),
    }
    return loss, metrics


def save_dip_trace_plot(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.0))
    axes[0, 0].plot(steps, [row["loss"] for row in rows], color="#2563eb")
    axes[0, 0].set_title("DIP loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(steps, [row["mean_band_angle_deg"] for row in rows], color="#7c3aed")
    axes[0, 1].set_title("Mean same-HKL band angle")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 2].plot(steps, [row["approx_match_score"] for row in rows], color="#16a34a")
    axes[0, 2].set_title("Approx. differentiable match score")
    axes[0, 2].grid(alpha=0.25)
    axes[1, 0].plot(steps, [row["delta_angle_deg"] for row in rows], color="#f97316")
    axes[1, 0].set_title("Rotation delta (deg)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(steps, [row["radius_scale"] for row in rows], color="#0f766e")
    axes[1, 1].set_title("Radius scale")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 2].plot(steps, [row["dx_px"] for row in rows], label="dx", color="#db2777")
    axes[1, 2].plot(steps, [row["dy_px"] for row in rows], label="dy", color="#0891b2")
    axes[1, 2].set_title("PC shift (px)")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_candidate_rows(
    trace_rows: list[dict],
    prepared: PreparedPattern,
    master,
    initial_result: MatchResult,
    families_by_label: dict[str, HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
    out_path: Path,
) -> tuple[MatchResult, dict, list[dict]]:
    candidates: dict[int, dict] = {}
    for row in trace_rows[:: max(1, args.candidate_stride)]:
        candidates[int(row["step"])] = row
    best_loss = min(trace_rows, key=lambda row: row["loss"])
    best_band = min(trace_rows, key=lambda row: row["mean_band_angle_deg"])
    last = trace_rows[-1]
    identity = {
        **trace_rows[0],
        "step": -1,
        "rotvec": [0.0, 0.0, 0.0],
        "rotvec_x_deg": 0.0,
        "rotvec_y_deg": 0.0,
        "rotvec_z_deg": 0.0,
        "delta_angle_deg": 0.0,
        "radius_scale": 1.0,
        "dx_px": 0.0,
        "dy_px": 0.0,
    }
    for row in (best_loss, best_band, last, identity):
        candidates[int(row["step"])] = row

    evaluated = []
    for row in candidates.values():
        result = numpy_params_to_result(prepared, master, initial_result, row)
        band_rows = band_angle_rows(result, families_by_label, fixed_hkl_by_band)
        mean_angle = float(np.mean([item["angle_deg"] for item in band_rows])) if band_rows else float("nan")
        evaluated.append(
            {
                "step": int(row["step"]),
                "loss": float(row["loss"]),
                "approx_match_score": float(row["approx_match_score"]),
                "full_match_score": float(result.score),
                "mean_band_angle_deg": mean_angle,
                "delta_angle_deg": float(row["delta_angle_deg"]),
                "radius_scale": float(row["radius_scale"]),
                "dx_px": float(row["dx_px"]),
                "dy_px": float(row["dy_px"]),
                "rotvec": row["rotvec"],
            }
        )

    min_allowed_score = float(initial_result.score) - float(args.max_match_score_drop)
    feasible = [row for row in evaluated if row["full_match_score"] >= min_allowed_score]
    if feasible:
        selected = min(feasible, key=lambda row: (row["mean_band_angle_deg"], row["loss"]))
        selected_reason = "best_band_angle_with_match_score_guard"
    else:
        selected = max(evaluated, key=lambda row: row["full_match_score"])
        selected_reason = "fallback_highest_full_match_score"

    for row in evaluated:
        row["selected"] = int(row is selected)
        row["selected_reason"] = selected_reason if row is selected else ""
    write_rows_csv(evaluated, out_path)
    return numpy_params_to_result(prepared, master, initial_result, selected), selected | {"selected_reason": selected_reason}, evaluated


def train_dip(
    prepared: PreparedPattern,
    master,
    torch_master,
    initial_result: MatchResult,
    families: list[HKLFamily],
    fixed_hkl_by_band: dict[int, str],
    args,
    seed: int,
    device: torch.device,
) -> tuple[MatchResult, dict, list[dict], list[dict]]:
    torch.manual_seed(seed)
    point_indices = choose_point_indices(prepared, args.residual_points, seed)
    torch_data = build_torch_arrays(prepared, initial_result, families, fixed_hkl_by_band, point_indices, device)
    out_dim = 6 if args.optimize_pc else 4
    model = DIPParamNet(args.noise_dim, args.hidden_dim, args.layers, out_dim).to(device)
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed)
    noise = torch.randn((1, args.noise_dim), generator=noise_generator, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace_rows: list[dict] = []

    iterator = range(args.steps + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="DIP gradient descent", leave=False)
    for step in iterator:
        optimizer.zero_grad(set_to_none=True)
        raw = model(noise)
        loss, metrics = forward_loss(raw, args, prepared, torch_data, torch_master)
        if step > 0:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if step % args.trace_interval == 0 or step == args.steps:
            trace_rows.append({"step": int(step), **metrics})

    families_by_label = family_by_label(families)
    final_result, selected, evaluated = evaluate_candidate_rows(
        trace_rows,
        prepared,
        master,
        initial_result,
        families_by_label,
        fixed_hkl_by_band,
        args,
        Path(args.current_out_dir) / "05_dip_candidate_selection.csv",
    )
    selected.update(
        {
            "point_indices_count": int(len(point_indices)),
            "device": str(device),
            "model_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    return final_result, selected, trace_rows, evaluated


def process_one(args, map_spec, master, torch_master, index: int, batch_dir: Path, device: torch.device) -> dict:
    out_dir = batch_dir / f"idx_{index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.current_out_dir = str(out_dir)

    bundle = read_pattern_bundle(args.h5, map_spec, index)
    phase_id = int(bundle.ang_record.get("Phase", 1))
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)
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
        label="H5-band-enhanced",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name="auto",
    )
    initial_result = match_to_master(
        prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=parse_refine_schedule(args.refine_schedule),
        random_seed=args.seed + int(index),
    )
    initial_assignments = assign_labels(initial_result, families)
    fixed_hkl_by_band = {item.band_index: item.hkl for item in initial_assignments}
    families_by_label = family_by_label(families)
    initial_label_score, initial_label_angle = label_score(initial_assignments)
    initial_rows = band_angle_rows(initial_result, families_by_label, fixed_hkl_by_band)
    write_rows_csv(initial_rows, out_dir / "02_initial_band_angle_residuals.csv")
    save_detector_label_overlay(initial_result, initial_assignments, out_dir / "01_detector_bands_initial_inferred_hkl.png")
    save_labeled_alignment(
        initial_result,
        master,
        initial_assignments,
        out_dir / "03_initial_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    final_result, selected, trace_rows, evaluated = train_dip(
        prepared,
        master,
        torch_master,
        initial_result,
        families,
        fixed_hkl_by_band,
        args,
        seed=args.seed + int(index),
        device=device,
    )
    final_assignments = assign_labels(final_result, families, fixed_hkl_by_band=fixed_hkl_by_band)
    final_label_score, final_label_angle = label_score(final_assignments)
    final_rows = band_angle_rows(final_result, families_by_label, fixed_hkl_by_band)
    write_rows_csv(trace_rows, out_dir / "04_dip_optimizer_trace.csv")
    save_dip_trace_plot(trace_rows, out_dir / "06_dip_optimizer_trace.png")
    write_rows_csv(final_rows, out_dir / "07_final_band_angle_residuals.csv")
    save_band_residual_comparison(initial_rows, final_rows, out_dir / "08_band_angle_residual_comparison.png")
    save_labeled_alignment(
        final_result,
        master,
        final_assignments,
        out_dir / "09_refined_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        final_result,
        master,
        products,
        out_dir / "10_refined_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    hyperparameters = {
        "dip_network": {
            "noise_dim": args.noise_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "trace_interval": args.trace_interval,
            "candidate_stride": args.candidate_stride,
        },
        "initial_match": {
            "coarse_rotations": args.coarse_rotations,
            "refine_schedule": args.refine_schedule,
            "seed": args.seed + int(index),
            "detector_convention": initial_result.convention_name,
        },
        "preprocessing": {
            "mask_radius_frac": args.mask_radius_frac,
            "mask_erosion": args.mask_erosion,
            "background_sigma": args.background_sigma,
            "band_sigma_min": args.band_sigma_min,
            "band_sigma_max": args.band_sigma_max,
            "match_quantile": args.match_quantile,
            "top_k_points": args.top_k_points,
        },
        "bounds": {
            "rotation_bound_deg": args.rotation_bound_deg,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "optimize_pc": args.optimize_pc,
            "pc_bound_px": args.pc_bound_px,
            "max_match_score_drop": args.max_match_score_drop,
        },
        "loss_weights": {
            "image_line_weight": args.image_line_weight,
            "h5_band_weight": args.h5_band_weight,
            "intensity_weight": args.intensity_weight,
            "band_geometry_weight": args.band_geometry_weight,
            "band_angle_scale_deg": args.band_angle_scale_deg,
            "radius_regularization_weight": args.radius_regularization_weight,
            "radius_regularization_sigma": args.radius_regularization_sigma,
            "pc_regularization_weight": args.pc_regularization_weight,
            "pc_regularization_sigma_px": args.pc_regularization_sigma_px,
            "rotation_regularization_weight": args.rotation_regularization_weight,
            "rotation_regularization_sigma_deg": args.rotation_regularization_sigma_deg,
            "residual_points": args.residual_points,
        },
    }
    (out_dir / "hyperparameters.json").write_text(json.dumps(jsonable(hyperparameters), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "map": map_spec.key,
        "map_label": map_spec.label,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase": phase_info,
        "hkl_families": [
            {key: value for key, value in asdict(family).items() if key != "normals"} | {"label": family.label, "normal_count": int(len(family.normals))}
            for family in families
        ],
        "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "fixed_hkl_by_band": fixed_hkl_by_band,
        "initial_match": initial_result.to_json_dict(),
        "initial_label_score": initial_label_score,
        "initial_label_mean_angle_deg": initial_label_angle,
        "initial_mean_band_angle_deg": float(np.mean([row["angle_deg"] for row in initial_rows])) if initial_rows else float("nan"),
        "final_match": final_result.to_json_dict(),
        "final_label_score": final_label_score,
        "final_label_mean_angle_deg": final_label_angle,
        "final_mean_band_angle_deg": float(np.mean([row["angle_deg"] for row in final_rows])) if final_rows else float("nan"),
        "selected_checkpoint": selected,
        "hyperparameters": hyperparameters,
        "outputs": {
            "detector_labels": str(out_dir / "01_detector_bands_initial_inferred_hkl.png"),
            "initial_band_residuals_csv": str(out_dir / "02_initial_band_angle_residuals.csv"),
            "initial_labeled_alignment": str(out_dir / "03_initial_labeled_band_alignment.png"),
            "dip_trace_csv": str(out_dir / "04_dip_optimizer_trace.csv"),
            "candidate_selection_csv": str(out_dir / "05_dip_candidate_selection.csv"),
            "dip_trace_plot": str(out_dir / "06_dip_optimizer_trace.png"),
            "final_band_residuals_csv": str(out_dir / "07_final_band_angle_residuals.csv"),
            "band_residual_comparison": str(out_dir / "08_band_angle_residual_comparison.png"),
            "refined_labeled_alignment": str(out_dir / "09_refined_labeled_band_alignment.png"),
            "refined_final_spatial": str(out_dir / "10_refined_final_spatial.png"),
            "hyperparameters": str(out_dir / "hyperparameters.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_csv(summaries: list[dict], out_path: Path) -> None:
    rows = []
    for summary in summaries:
        selected = summary["selected_checkpoint"]
        rows.append(
            {
                "index": summary["index"],
                "row": summary["row"],
                "col": summary["col"],
                "phase_id": summary["phase"]["phase_id"],
                "initial_match_score": summary["initial_match"]["score"],
                "final_match_score": summary["final_match"]["score"],
                "match_score_gain": summary["final_match"]["score"] - summary["initial_match"]["score"],
                "initial_label_mean_angle_deg": summary["initial_label_mean_angle_deg"],
                "final_label_mean_angle_deg": summary["final_label_mean_angle_deg"],
                "label_angle_gain_deg": summary["initial_label_mean_angle_deg"] - summary["final_label_mean_angle_deg"],
                "initial_mean_band_angle_deg": summary["initial_mean_band_angle_deg"],
                "final_mean_band_angle_deg": summary["final_mean_band_angle_deg"],
                "band_angle_gain_deg": summary["initial_mean_band_angle_deg"] - summary["final_mean_band_angle_deg"],
                "selected_step": selected["step"],
                "selected_reason": selected["selected_reason"],
                "delta_angle_deg": selected["delta_angle_deg"],
                "radius_scale": selected["radius_scale"],
                "dx_px": selected["dx_px"],
                "dy_px": selected["dy_px"],
                "device": selected["device"],
                "model_parameters": selected["model_parameters"],
            }
        )
    write_rows_csv(rows, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DIP-style neural parameter registration for Kikuchi band geometry.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma list or Python-like ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "dip_parametric_registration")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=5000)

    parser.add_argument("--coarse-rotations", type=int, default=320)
    parser.add_argument("--refine-schedule", default="10:180,4:220,1.5:220")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trace-interval", type=int, default=1)
    parser.add_argument("--candidate-stride", type=int, default=5)

    parser.add_argument("--rotation-bound-deg", type=float, default=6.0)
    parser.add_argument("--radius-min", type=float, default=0.98)
    parser.add_argument("--radius-max", type=float, default=1.02)
    parser.add_argument("--optimize-pc", action="store_true")
    parser.add_argument("--pc-bound-px", type=float, default=4.0)
    parser.add_argument("--max-match-score-drop", type=float, default=0.02)

    parser.add_argument("--residual-points", type=int, default=1600)
    parser.add_argument("--image-line-weight", type=float, default=1.0)
    parser.add_argument("--h5-band-weight", type=float, default=0.8)
    parser.add_argument("--intensity-weight", type=float, default=0.15)
    parser.add_argument("--band-geometry-weight", type=float, default=0.6)
    parser.add_argument("--band-angle-scale-deg", type=float, default=8.0)
    parser.add_argument("--radius-regularization-weight", type=float, default=0.08)
    parser.add_argument("--radius-regularization-sigma", type=float, default=0.02)
    parser.add_argument("--pc-regularization-weight", type=float, default=0.05)
    parser.add_argument("--pc-regularization-sigma-px", type=float, default=3.0)
    parser.add_argument("--rotation-regularization-weight", type=float, default=0.02)
    parser.add_argument("--rotation-regularization-sigma-deg", type=float, default=3.0)

    parser.add_argument("--sphere-lon-count", type=int, default=420)
    parser.add_argument("--sphere-colat-count", type=int, default=210)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.45)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.40)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    map_spec = default_map_specs(args.data_dir)[args.map]
    total = read_up2_info(map_spec.up2_path).count
    indices = parse_indices(args.indices, total, args.count, args.strategy) if args.indices or args.count > 1 else [args.index]
    if not indices:
        raise ValueError("No valid pattern indices selected")

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Using device: {device}")
    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    torch_master = to_torch_master(master, device)

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    iterator = tqdm(indices, desc="DIP parametric registration") if tqdm is not None else indices
    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        summaries.append(process_one(args, map_spec, master, torch_master, int(index), batch_dir, device))

    write_batch_csv(summaries, batch_dir / "batch_dip_parametric_summary.csv")
    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "06_dip_optimizer_trace.png" for summary in summaries],
        batch_dir / "contact_sheet_dip_optimizer_trace.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "08_band_angle_residual_comparison.png" for summary in summaries],
        batch_dir / "contact_sheet_band_angle_residuals.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "09_refined_labeled_band_alignment.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_labeled_band_alignment.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "10_refined_final_spatial.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_final_spatial.png",
        thumb_width=900,
        columns=2,
    )

    print(f"Saved DIP parametric registration results to: {batch_dir}")
    print(f"Batch CSV: {batch_dir / 'batch_dip_parametric_summary.csv'}")
    print(f"Final contact sheet: {batch_dir / 'contact_sheet_refined_final_spatial.png'}")


if __name__ == "__main__":
    main()
