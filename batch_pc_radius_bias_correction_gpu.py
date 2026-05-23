from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Windows Anaconda can load Intel OpenMP through both PyTorch and scientific
# packages. Keep this workaround local to this exploratory GPU runner.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    DETECTOR_CONVENTIONS,
    MatchResult,
    MatchWeights,
    MasterSphere,
    PreparedPattern,
    default_map_specs,
    jsonable,
    load_master_sphere,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
    score_rotation,
)
from pc_radius_bias_correction import (
    corrected_pc,
    parse_float_list,
    prepared_with_pc,
    save_score_landscape,
    write_search_csv,
)
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    parse_refine_schedule,
    save_final_spatial_visualization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


@dataclass
class TorchMasterSphere:
    upper_intensity: torch.Tensor
    lower_intensity: torch.Tensor
    upper_band: torch.Tensor
    lower_band: torch.Tensor


@dataclass
class TorchPreparedPattern:
    rows: torch.Tensor
    cols: torch.Tensor
    exp_image_band_z: torch.Tensor
    exp_intensity_z: torch.Tensor
    exp_h5_band_z: torch.Tensor
    exp_h5_band_weight: torch.Tensor
    weights: MatchWeights
    height: int
    width: int


def torch_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).contiguous()


def to_torch_master(master: MasterSphere, device: torch.device) -> TorchMasterSphere:
    return TorchMasterSphere(
        upper_intensity=torch_image(master.upper_intensity, device),
        lower_intensity=torch_image(master.lower_intensity, device),
        upper_band=torch_image(master.upper_band, device),
        lower_band=torch_image(master.lower_band, device),
    )


def to_torch_prepared(prepared: PreparedPattern, device: torch.device) -> TorchPreparedPattern:
    rows_np, cols_np = np.nonzero(prepared.match_mask)
    h5_weight = prepared.h5_line_mask[prepared.match_mask].astype(np.float32)
    if h5_weight.size == 0 or float(h5_weight.max()) <= 0.0:
        h5_weight = np.ones_like(prepared.exp_h5_band_z, dtype=np.float32)
    return TorchPreparedPattern(
        rows=torch.as_tensor(rows_np.astype(np.float32), dtype=torch.float32, device=device),
        cols=torch.as_tensor(cols_np.astype(np.float32), dtype=torch.float32, device=device),
        exp_image_band_z=torch.as_tensor(prepared.exp_image_band_z, dtype=torch.float32, device=device),
        exp_intensity_z=torch.as_tensor(prepared.exp_intensity_z, dtype=torch.float32, device=device),
        exp_h5_band_z=torch.as_tensor(prepared.exp_h5_band_z, dtype=torch.float32, device=device),
        exp_h5_band_weight=torch.as_tensor(h5_weight, dtype=torch.float32, device=device),
        weights=prepared.weights,
        height=prepared.image.shape[0],
        width=prepared.image.shape[1],
    )


def detector_points_from_pc_torch(
    prepared: TorchPreparedPattern,
    pc: tuple[float, float, float],
    detector_transform: torch.Tensor,
) -> torch.Tensor:
    pcx, pcy, pcz = pc
    cx = pcx * float(prepared.width - 1)
    cy = pcy * float(prepared.height - 1)
    x = (prepared.cols - cx) / (pcz * float(prepared.height))
    y = -(prepared.rows - cy) / (pcz * float(prepared.height))
    z = torch.ones_like(x)
    points = torch.stack((x, y, z), dim=1)
    points = points / (torch.linalg.norm(points, dim=1, keepdim=True) + 1e-8)
    return points @ detector_transform.T


def sample_master_channel(vectors: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    if vectors.ndim == 2:
        vectors = vectors.unsqueeze(0)
    batch, count, _ = vectors.shape
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    upper_grid = torch.stack((x / (1.0 + z + 1e-8), y / (1.0 + z + 1e-8)), dim=-1).view(batch, count, 1, 2)
    lower_grid = torch.stack((x / (1.0 - z + 1e-8), y / (1.0 - z + 1e-8)), dim=-1).view(batch, count, 1, 2)

    upper_sample = F.grid_sample(
        upper.expand(batch, -1, -1, -1),
        upper_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, count)
    lower_sample = F.grid_sample(
        lower.expand(batch, -1, -1, -1),
        lower_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, count)
    return torch.where(z >= 0.0, upper_sample, lower_sample)


def zscore_rows(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / (values.std(dim=1, keepdim=True, unbiased=False) + 1e-8)


def rotation_matrices_torch(rotations: R, device: torch.device) -> torch.Tensor:
    matrices = np.asarray(rotations.as_matrix(), dtype=np.float32)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    return torch.as_tensor(matrices, dtype=torch.float32, device=device)


def resolve_match_weights(args: argparse.Namespace) -> tuple[MatchWeights, str]:
    if args.match_mode == "band_only":
        return MatchWeights(image_line=0.0, intensity=0.0, h5_band=1.0), "H5-band-only"
    if args.match_mode == "band_dominant":
        return MatchWeights(image_line=0.15, intensity=0.0, h5_band=0.85), "H5-band-dominant"
    return (
        MatchWeights(
            image_line=args.enhanced_image_line_weight,
            intensity=args.enhanced_intensity_weight,
            h5_band=args.enhanced_h5_band_weight,
        ),
        "H5-band-enhanced",
    )


def match_to_master_constrained(
    prepared: PreparedPattern,
    master: MasterSphere,
    coarse_rotation_count: int,
    refine_schedule: list[tuple[float, int]],
    random_seed: int,
    force_convention: str,
    match_mode: str,
) -> MatchResult:
    rng = np.random.default_rng(random_seed)
    best_rotation: R | None = None
    best_score = -np.inf
    best_convention_name = ""
    best_transform = np.eye(3, dtype=np.float32)
    conventions = DETECTOR_CONVENTIONS.items() if force_convention == "auto" else [(force_convention, DETECTOR_CONVENTIONS[force_convention])]

    for convention_name, transform in conventions:
        convention_points = prepared.exp_points @ transform.T
        coarse_rotations = R.random(coarse_rotation_count, random_state=rng)
        coarse_scores = np.array([score_rotation_for_mode(rot, convention_points, prepared, master, match_mode) for rot in coarse_rotations])
        rotation = coarse_rotations[int(np.argmax(coarse_scores))]
        convention_score = float(coarse_scores.max())

        for step_deg, attempts in refine_schedule:
            for _ in range(attempts):
                delta = R.from_euler("zyx", rng.normal(scale=step_deg, size=3), degrees=True)
                candidate = delta * rotation
                candidate_score = score_rotation_for_mode(candidate, convention_points, prepared, master, match_mode)
                if candidate_score > convention_score:
                    rotation = candidate
                    convention_score = candidate_score

        if convention_score > best_score:
            best_score = convention_score
            best_rotation = rotation
            best_convention_name = convention_name
            best_transform = transform

    if best_rotation is None:
        raise RuntimeError("No rotation was evaluated")

    return MatchResult(
        label=prepared.label,
        score=best_score,
        rotation=best_rotation,
        convention_name=best_convention_name,
        detector_transform=best_transform,
        prepared=prepared,
    )


def score_rotation_for_mode(rotation: R, points: np.ndarray, prepared: PreparedPattern, master: MasterSphere, match_mode: str) -> float:
    if match_mode == "weighted":
        return score_rotation(rotation, points, prepared, master)

    rotated = rotation.apply(points)
    master_band = master.sample_band(rotated)
    h5_weight = prepared.h5_line_mask[prepared.match_mask].astype(np.float32)
    if h5_weight.size == 0 or float(h5_weight.max()) <= 0.0:
        h5_weight = np.ones_like(master_band, dtype=np.float32)
    h5_direct_score = float(np.sum(h5_weight * master_band) / (np.sum(h5_weight) + 1e-8))

    if match_mode == "band_only":
        return h5_direct_score

    score = prepared.weights.h5_band * h5_direct_score
    if prepared.weights.image_line or prepared.weights.intensity:
        score += score_rotation(rotation, points, prepared, master)
    return float(score)


@torch.no_grad()
def score_rotation_batch_torch(
    rotations: R,
    points: torch.Tensor,
    prepared: TorchPreparedPattern,
    master: TorchMasterSphere,
    device: torch.device,
    match_mode: str,
) -> torch.Tensor:
    matrices = rotation_matrices_torch(rotations, device)
    rotated = torch.einsum("nj,bkj->bnk", points, matrices)
    master_band = sample_master_channel(rotated, master.upper_band, master.lower_band)

    if match_mode != "weighted":
        weight = prepared.exp_h5_band_weight.unsqueeze(0)
        h5_direct_score = torch.sum(weight * master_band, dim=1) / (torch.sum(weight, dim=1) + 1e-8)
        if match_mode == "band_only":
            return h5_direct_score
    else:
        h5_direct_score = None

    master_band_z = zscore_rows(master_band)
    master_intensity_z = zscore_rows(sample_master_channel(rotated, master.upper_intensity, master.lower_intensity))

    score = torch.zeros(matrices.shape[0], dtype=torch.float32, device=device)
    if prepared.weights.image_line:
        score += prepared.weights.image_line * torch.mean(prepared.exp_image_band_z.unsqueeze(0) * master_band_z, dim=1)
    if prepared.weights.intensity:
        score += prepared.weights.intensity * torch.mean(prepared.exp_intensity_z.unsqueeze(0) * master_intensity_z, dim=1)
    if prepared.weights.h5_band:
        if match_mode == "weighted":
            score += prepared.weights.h5_band * torch.mean(prepared.exp_h5_band_z.unsqueeze(0) * master_band_z, dim=1)
        else:
            score += prepared.weights.h5_band * h5_direct_score
    return score


@torch.no_grad()
def deterministic_rotation_refine_torch(
    points: torch.Tensor,
    prepared: TorchPreparedPattern,
    master: TorchMasterSphere,
    initial_rotation: R,
    local_steps_deg: list[float],
    device: torch.device,
    match_mode: str,
) -> tuple[R, float]:
    rotation = initial_rotation
    best_score = float(score_rotation_batch_torch(rotation, points, prepared, master, device, match_mode)[0].item())
    axes = np.array(
        [
            [dz, dy, dx]
            for dz in (-1.0, 0.0, 1.0)
            for dy in (-1.0, 0.0, 1.0)
            for dx in (-1.0, 0.0, 1.0)
            if not (dz == 0.0 and dy == 0.0 and dx == 0.0)
        ],
        dtype=np.float32,
    )

    for step_deg in local_steps_deg:
        improved = True
        while improved:
            improved = False
            candidates = R.from_euler("zyx", axes * step_deg, degrees=True) * rotation
            candidate_scores = score_rotation_batch_torch(candidates, points, prepared, master, device, match_mode)
            best_idx = int(torch.argmax(candidate_scores).item())
            candidate_score = float(candidate_scores[best_idx].item())
            if candidate_score > best_score + 1e-7:
                rotation = candidates[best_idx]
                best_score = candidate_score
                improved = True
    return rotation, best_score


def write_batch_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "index",
        "row",
        "col",
        "h5_score",
        "best_score",
        "score_gain",
        "score_gain_percent",
        "dx_px",
        "dy_px",
        "radius_scale",
        "pcx",
        "pcy",
        "effective_pcz",
        "convention_name",
        "line_variant",
        "match_mode",
        "force_convention",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_landscape_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 620, columns: int = 2) -> None:
    tiles = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        label_bar = np.full((40, thumb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(label_bar, path.parent.name, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(np.vstack([label_bar, thumb]))
    if not tiles:
        return

    rows = int(np.ceil(len(tiles) / columns))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    pad = 20
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, columns * tile_w + (columns + 1) * pad, 3),
        255,
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        y = pad + (i // columns) * (tile_h + pad)
        x = pad + (i % columns) * (tile_w + pad)
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    cv2.imwrite(str(out_path), canvas)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU batch try EBSD pattern-center and projection-radius bias correction.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--indices", default=None, help="Comma list or ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "pc_radius_bias_correction_gpu_batch")
    parser.add_argument("--pc-shifts-px", default="-18,-12,-6,0,6,12,18")
    parser.add_argument("--radius-scales", default="0.86,0.90,0.94,0.98,1.00,1.02")
    parser.add_argument("--local-steps-deg", default="1.5,0.5")
    parser.add_argument("--match-mode", choices=["weighted", "band_dominant", "band_only"], default="weighted")
    parser.add_argument("--force-convention", choices=["auto", *DETECTOR_CONVENTIONS.keys()], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=5000)
    parser.add_argument("--coarse-rotations", type=int, default=160)
    parser.add_argument("--refine-schedule", default="8:100,3:140,1:140")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sphere-lon-count", type=int, default=420)
    parser.add_argument("--sphere-colat-count", type=int, default=210)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.45)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available in this Python environment; falling back to CPU torch.")
    else:
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

    map_spec = default_map_specs(args.data_dir)[args.map]
    total = read_up2_info(map_spec.up2_path).count
    indices = parse_indices(args.indices, total, args.count, args.strategy)
    if not indices:
        raise ValueError("No valid pattern indices selected")

    pc_shifts = parse_float_list(args.pc_shifts_px)
    radius_scales = parse_float_list(args.radius_scales)
    local_steps = parse_float_list(args.local_steps_deg)
    refine_schedule = parse_refine_schedule(args.refine_schedule)
    weights, match_label = resolve_match_weights(args)

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere once: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    torch_master = to_torch_master(master, device)

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    h5_paths: list[Path] = []
    corrected_paths: list[Path] = []
    landscape_paths: list[Path] = []
    summary_rows: list[dict] = []
    iterator = tqdm(indices, desc="PC/radius GPU batch") if tqdm is not None else indices

    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        out_dir = batch_dir / f"idx_{index:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = read_pattern_bundle(args.h5, map_spec, index)
        products = build_preprocessing_products(
            bundle.pattern_u16,
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
        )
        prepared, variant_diagnostics = prepare_pattern(
            bundle=bundle,
            weights=weights,
            label=match_label,
            mask_radius_fraction=args.mask_radius_frac,
            mask_erosion=args.mask_erosion,
            background_sigma=args.background_sigma,
            band_sigma_min=args.band_sigma_min,
            band_sigma_max=args.band_sigma_max,
            match_quantile=args.match_quantile,
            top_k_points=args.top_k_points,
            line_variant_name="auto",
        )
        initial_result = match_to_master_constrained(
            prepared,
            master,
            coarse_rotation_count=args.coarse_rotations,
            refine_schedule=refine_schedule,
            random_seed=args.seed + int(index),
            force_convention=args.force_convention,
            match_mode=args.match_mode,
        )

        h5_path = out_dir / "01_h5_pc_final_spatial.png"
        corrected_path = out_dir / "02_corrected_pc_radius_final_spatial.png"
        landscape_path = out_dir / "03_pc_radius_score_landscape.png"
        save_final_spatial_visualization(
            initial_result,
            master,
            products,
            h5_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )

        torch_prepared = to_torch_prepared(prepared, device)
        detector_transform = torch.as_tensor(initial_result.detector_transform, dtype=torch.float32, device=device)
        candidates = [(dx, dy, radius) for radius in radius_scales for dy in pc_shifts for dx in pc_shifts]
        records: list[dict] = []
        best_record: dict | None = None
        best_rotation: R | None = None
        search_iterator = tqdm(candidates, desc=f"idx_{index:05d} cuda", leave=False) if tqdm is not None else candidates
        for dx_px, dy_px, radius_scale in search_iterator:
            pc = corrected_pc(bundle.pc, dx_px, dy_px, radius_scale, torch_prepared.height, torch_prepared.width)
            points = detector_points_from_pc_torch(torch_prepared, pc, detector_transform)
            rotation, score = deterministic_rotation_refine_torch(
                points,
                torch_prepared,
                torch_master,
                initial_result.rotation,
                local_steps,
                device,
                args.match_mode,
            )
            record = {
                "dx_px": float(dx_px),
                "dy_px": float(dy_px),
                "radius_scale": float(radius_scale),
                "pcx": float(pc[0]),
                "pcy": float(pc[1]),
                "effective_pcz": float(pc[2]),
                "score": float(score),
                "rotation_quat_xyzw": rotation.as_quat().tolist(),
            }
            records.append(record)
            if best_record is None or score > float(best_record["score"]):
                best_record = record
                best_rotation = rotation

        h5_local = next(
            (
                row
                for row in records
                if abs(float(row["dx_px"])) < 1e-9
                and abs(float(row["dy_px"])) < 1e-9
                and abs(float(row["radius_scale"]) - 1.0) < 1e-9
            ),
            None,
        )
        if h5_local is None:
            h5_pc = bundle.pc
            h5_points = detector_points_from_pc_torch(torch_prepared, h5_pc, detector_transform)
            h5_rotation, h5_score = deterministic_rotation_refine_torch(
                h5_points,
                torch_prepared,
                torch_master,
                initial_result.rotation,
                local_steps,
                device,
                args.match_mode,
            )
            h5_local = {
                "dx_px": 0.0,
                "dy_px": 0.0,
                "radius_scale": 1.0,
                "pcx": float(bundle.pc[0]),
                "pcy": float(bundle.pc[1]),
                "effective_pcz": float(bundle.pc[2]),
                "score": float(h5_score),
                "rotation_quat_xyzw": h5_rotation.as_quat().tolist(),
            }
            records.append(h5_local)

        assert best_record is not None and best_rotation is not None
        best_prepared = prepared_with_pc(prepared, (best_record["pcx"], best_record["pcy"], best_record["effective_pcz"]))
        best_result = MatchResult(
            label=f"PC-radius-corrected GPU {match_label}",
            score=float(best_record["score"]),
            rotation=best_rotation,
            convention_name=initial_result.convention_name,
            detector_transform=initial_result.detector_transform,
            prepared=best_prepared,
        )
        save_final_spatial_visualization(
            best_result,
            master,
            products,
            corrected_path,
            lon_count=args.sphere_lon_count,
            colat_count=args.sphere_colat_count,
        )
        save_score_landscape(records, landscape_path)
        write_search_csv(records, out_dir / "pc_radius_search_results.csv")

        h5_score = float(h5_local["score"])
        best_score = float(best_record["score"])
        row = {
            "index": int(bundle.index),
            "row": int(bundle.row),
            "col": int(bundle.col),
            "h5_score": h5_score,
            "best_score": best_score,
            "score_gain": best_score - h5_score,
            "score_gain_percent": (best_score - h5_score) / max(abs(h5_score), 1e-12) * 100.0,
            "dx_px": float(best_record["dx_px"]),
            "dy_px": float(best_record["dy_px"]),
            "radius_scale": float(best_record["radius_scale"]),
            "pcx": float(best_record["pcx"]),
            "pcy": float(best_record["pcy"]),
            "effective_pcz": float(best_record["effective_pcz"]),
            "convention_name": initial_result.convention_name,
            "line_variant": prepared.line_variant.name,
            "match_mode": args.match_mode,
            "force_convention": args.force_convention,
        }
        summary = {
            "map": map_spec.key,
            "map_label": map_spec.label,
            "index": bundle.index,
            "row": bundle.row,
            "col": bundle.col,
            "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
            "search_parameters": {
                "pc_shifts_px": pc_shifts,
                "radius_scales": radius_scales,
                "local_steps_deg": local_steps,
                "match_mode": args.match_mode,
                "weights": {
                    "image_line": weights.image_line,
                    "intensity": weights.intensity,
                    "h5_band": weights.h5_band,
                },
                "force_convention": args.force_convention,
                "device": str(device),
                "interpretation": "radius_scale multiplies H5 pcz, changing the detector-to-sphere angular radius while pcx/pcy shifts move the pattern center.",
            },
            "initial_global_match_cpu": initial_result.to_json_dict(),
            "h5_pc_local_refined_gpu": h5_local,
            "best_correction_gpu": best_record,
            "best_match": best_result.to_json_dict(),
            "line_variant": prepared.line_variant.name,
            "line_variant_score": prepared.line_variant_score,
            "variant_diagnostics": variant_diagnostics,
            "outputs": {
                "h5_pc_final_spatial": str(h5_path),
                "corrected_pc_radius_final_spatial": str(corrected_path),
                "score_landscape": str(landscape_path),
                "search_csv": str(out_dir / "pc_radius_search_results.csv"),
                "summary": str(out_dir / "summary.json"),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Best index={index}: dx={row['dx_px']:+.1f}px, dy={row['dy_px']:+.1f}px, "
            f"radius={row['radius_scale']:.3f}, score={best_score:.4f}, gain={row['score_gain']:.4f}"
        )

        h5_paths.append(h5_path)
        corrected_paths.append(corrected_path)
        landscape_paths.append(landscape_path)
        summary_rows.append(row)

    write_batch_csv(summary_rows, batch_dir / "batch_pc_radius_summary.csv")
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(
            jsonable(
                {
                    "map": map_spec.key,
                    "total_patterns": total,
                    "indices": indices,
                    "search_parameters": {
                    "pc_shifts_px": pc_shifts,
                    "radius_scales": radius_scales,
                    "local_steps_deg": local_steps,
                    "match_mode": args.match_mode,
                    "weights": {
                        "image_line": weights.image_line,
                        "intensity": weights.intensity,
                        "h5_band": weights.h5_band,
                    },
                    "force_convention": args.force_convention,
                    "device": str(device),
                },
                    "rows": summary_rows,
                    "summaries": [str(batch_dir / f"idx_{row['index']:05d}" / "summary.json") for row in summary_rows],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    make_contact_sheet(h5_paths, batch_dir / "contact_sheet_h5_pc.png", thumb_width=880, columns=2)
    make_contact_sheet(corrected_paths, batch_dir / "contact_sheet_corrected_pc_radius.png", thumb_width=880, columns=2)
    make_landscape_contact_sheet(landscape_paths, batch_dir / "contact_sheet_score_landscape.png", thumb_width=620, columns=2)
    print(f"Saved {len(summary_rows)} GPU PC/radius correction runs to: {batch_dir}")
    print(f"Summary CSV: {batch_dir / 'batch_pc_radius_summary.csv'}")
    print(f"Corrected contact sheet: {batch_dir / 'contact_sheet_corrected_pc_radius.png'}")


if __name__ == "__main__":
    main()
