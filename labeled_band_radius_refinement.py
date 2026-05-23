from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import permutations, product
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_final_spatial_visualizations import make_contact_sheet, parse_indices
from h5_band_enhanced_match import (
    DEFAULT_DATA_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR,
    LineSegment,
    MatchResult,
    MatchWeights,
    MasterSphere,
    PreparedPattern,
    default_map_specs,
    detector_pixels_to_sphere,
    jsonable,
    load_master_sphere,
    match_to_master,
    prepare_pattern,
    read_pattern_bundle,
    read_up2_info,
    resolve_master_path,
    score_rotation,
    sphere_texture,
)
from pc_radius_bias_correction import corrected_pc, deterministic_rotation_refine, prepared_with_pc
from visualize_calibration_pipeline import (
    build_preprocessing_products,
    detector_raw_display,
    equirect_line,
    parse_refine_schedule,
    save_final_spatial_visualization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress display only.
    tqdm = None


@dataclass(frozen=True)
class HKLFamily:
    h: int
    k: int
    l: int
    diffraction_intensity: float
    use_in_indexing: int
    show_bands: int
    normals: np.ndarray

    @property
    def label(self) -> str:
        return f"({self.h}{self.k}{self.l})"


@dataclass
class LabelAssignment:
    band_index: int
    hkl: str
    confidence: float
    angle_deg: float
    band_intensity: float
    best_normal: np.ndarray

    def to_json_dict(self) -> dict:
        return {
            "band_index": int(self.band_index),
            "hkl": self.hkl,
            "confidence": float(self.confidence),
            "angle_deg": float(self.angle_deg),
            "band_intensity": float(self.band_intensity),
            "best_normal": self.best_normal.astype(float).tolist(),
        }


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def decode_h5_scalar(value) -> object:
    array = np.asarray(value)
    if array.shape:
        value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return value.item() if hasattr(value, "item") else value


def cubic_family_normals(h: int, k: int, l: int) -> np.ndarray:
    normals: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for permuted in set(permutations((h, k, l), 3)):
        base = np.asarray(permuted, dtype=np.float32)
        if np.linalg.norm(base) <= 1e-8:
            continue
        for signs in product((-1.0, 1.0), repeat=3):
            vec = base * np.asarray(signs, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-8:
                continue
            vec = vec / norm
            key = tuple(np.round(vec, 6).astype(float))
            if key in seen:
                continue
            seen.add(key)
            normals.append(vec.astype(np.float32))
    return np.asarray(normals, dtype=np.float32)


def read_phase_hkl_families(h5_path: Path, map_group_path: str, phase_id: int) -> tuple[dict, list[HKLFamily]]:
    with h5py.File(h5_path, "r") as f:
        phase_group = f[f"{map_group_path}/EBSD/ANG/HEADER/Phase/{int(phase_id)}"]
        phase_info = {
            "phase_id": int(phase_id),
            "material_name": decode_h5_scalar(phase_group["Material Name"][()]),
            "formula": decode_h5_scalar(phase_group["Formula"][()]),
            "laue_group": decode_h5_scalar(phase_group["Laue Group"][()]),
            "symmetry": int(decode_h5_scalar(phase_group["Symmetry"][()])),
        }
        rows = phase_group["HKL Families"][()]

    families: list[HKLFamily] = []
    for row in rows:
        h = int(row["H"])
        k = int(row["K"])
        l = int(row["L"])
        families.append(
            HKLFamily(
                h=h,
                k=k,
                l=l,
                diffraction_intensity=float(row["Diffraction Intensity"]),
                use_in_indexing=int(row["Use in Indexing"]),
                show_bands=int(row["Show Bands"]),
                normals=cubic_family_normals(h, k, l),
            )
        )
    return phase_info, [family for family in families if family.use_in_indexing and family.show_bands]


def band_plane_normal(prepared: PreparedPattern, segment: LineSegment) -> np.ndarray:
    height, width = prepared.image.shape
    rows = np.asarray([segment.row0, segment.row1], dtype=np.float32)
    cols = np.asarray([segment.col0, segment.col1], dtype=np.float32)
    rays = detector_pixels_to_sphere(rows, cols, height, width, prepared.bundle.pc)
    normal = np.cross(rays[0], rays[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return (normal / norm).astype(np.float32)


def transform_normal_to_master(normal: np.ndarray, result: MatchResult) -> np.ndarray:
    normal = normal @ result.detector_transform.T
    normal = result.rotation.apply(normal)
    normal = normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    return normal


def family_match(normal: np.ndarray, family: HKLFamily) -> tuple[float, np.ndarray]:
    dots = np.abs(family.normals @ normal.astype(np.float32))
    best_index = int(np.argmax(dots))
    return float(dots[best_index]), family.normals[best_index]


def assign_labels(
    result: MatchResult,
    families: list[HKLFamily],
    fixed_hkl_by_band: dict[int, str] | None = None,
) -> list[LabelAssignment]:
    assignments: list[LabelAssignment] = []
    family_by_label = {family.label: family for family in families}
    for segment in result.prepared.line_segments:
        normal = band_plane_normal(result.prepared, segment)
        if not np.isfinite(normal).all():
            continue
        normal = transform_normal_to_master(normal, result)

        if fixed_hkl_by_band and segment.band_index in fixed_hkl_by_band:
            label = fixed_hkl_by_band[segment.band_index]
            candidate_families = [family_by_label[label]] if label in family_by_label else families
        else:
            candidate_families = families

        best_family = None
        best_confidence = -np.inf
        best_normal = None
        for family in candidate_families:
            confidence, family_normal = family_match(normal, family)
            if confidence > best_confidence:
                best_family = family
                best_confidence = confidence
                best_normal = family_normal
        if best_family is None or best_normal is None:
            continue
        angle = float(np.degrees(np.arccos(np.clip(best_confidence, -1.0, 1.0))))
        assignments.append(
            LabelAssignment(
                band_index=int(segment.band_index),
                hkl=best_family.label,
                confidence=float(best_confidence),
                angle_deg=angle,
                band_intensity=float(segment.band.intensity),
                best_normal=best_normal.astype(np.float32),
            )
        )
    return assignments


def label_score(assignments: list[LabelAssignment]) -> tuple[float, float]:
    if not assignments:
        return 0.0, float("nan")
    weights = np.asarray([max(1e-3, item.band_intensity) for item in assignments], dtype=np.float32)
    confidences = np.asarray([item.confidence for item in assignments], dtype=np.float32)
    angles = np.asarray([item.angle_deg for item in assignments], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    return float(np.sum(weights * confidences)), float(np.sum(weights * angles))


def transformed_segment_curve(result: MatchResult, segment: LineSegment, samples: int = 260) -> np.ndarray:
    rows = np.linspace(segment.row0, segment.row1, samples, dtype=np.float32)
    cols = np.linspace(segment.col0, segment.col1, samples, dtype=np.float32)
    height, width = result.prepared.image.shape
    curve = detector_pixels_to_sphere(rows, cols, height, width, result.prepared.bundle.pc)
    curve = curve @ result.detector_transform.T
    return result.rotation.apply(curve).astype(np.float32)


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


def save_detector_label_overlay(result: MatchResult, assignments: list[LabelAssignment], out_path: Path) -> None:
    prepared = result.prepared
    raw_display = detector_raw_display(prepared)
    assignment_by_band = {item.band_index: item for item in assignments}

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.imshow(raw_display, cmap="gray", vmin=0.0, vmax=1.0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(prepared.line_segments))))
    for color, segment in zip(colors, prepared.line_segments):
        assignment = assignment_by_band.get(segment.band_index)
        label = assignment.hkl if assignment else "unlabeled"
        ax.plot([segment.col0, segment.col1], [segment.row0, segment.row1], color=color, linewidth=2.2)
        row_mid = 0.5 * (segment.row0 + segment.row1)
        col_mid = 0.5 * (segment.col0 + segment.col1)
        text = f"{segment.band_index}:{label}"
        if assignment:
            text += f" {assignment.angle_deg:.1f}deg"
        ax.text(
            col_mid,
            row_mid,
            text,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.45, "boxstyle": "round,pad=0.2", "linewidth": 0},
        )
    ax.set_title(
        f"H5 OHP bands with inferred HKL labels | idx={prepared.bundle.index} "
        f"phase={prepared.bundle.ang_record.get('Phase')}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_labeled_alignment(
    result: MatchResult,
    master: MasterSphere,
    assignments: list[LabelAssignment],
    out_path: Path,
    lon_count: int,
    colat_count: int,
) -> None:
    texture, lon_grid, colat_grid, _ = sphere_texture(master, lon_count, colat_count)
    assignment_by_band = {item.band_index: item for item in assignments}

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    extent = [
        float(np.degrees(lon_grid.min())),
        float(np.degrees(lon_grid.max())),
        float(np.degrees(colat_grid.max())),
        float(np.degrees(colat_grid.min())),
    ]
    ax.imshow(texture, cmap="gray", extent=extent, aspect="auto", vmin=0.0, vmax=1.0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(result.prepared.line_segments))))
    for color, segment in zip(colors, result.prepared.line_segments):
        assignment = assignment_by_band.get(segment.band_index)
        if assignment is None:
            continue
        observed = transformed_segment_curve(result, segment)
        standard = great_circle_from_normal(assignment.best_normal)
        equirect_line(ax, standard, color=color, linewidth=1.0)
        equirect_line(ax, observed, color=color, linewidth=2.4)
        midpoint = observed[len(observed) // 2]
        lon = float(np.degrees(np.arctan2(midpoint[1], midpoint[0])))
        colat = float(np.degrees(np.arccos(np.clip(midpoint[2], -1.0, 1.0))))
        ax.text(
            lon,
            colat,
            f"{segment.band_index}:{assignment.hkl}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.45, "boxstyle": "round,pad=0.2", "linewidth": 0},
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(180, 0)
    ax.set_xlabel("Longitude on master sphere (deg)")
    ax.set_ylabel("Colatitude on master sphere (deg)")
    ax.set_title(
        f"Labeled band alignment | solid=transformed H5 band, thin=same-HKL standard great circle | "
        f"score={result.score:.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_landscape(records: list[dict], out_path: Path) -> None:
    radius_values = sorted({float(row["radius_scale"]) for row in records})
    dx_values = sorted({float(row["dx_px"]) for row in records})
    dy_values = sorted({float(row["dy_px"]) for row in records})
    best = max(records, key=lambda row: float(row["composite_score"]))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    radius_composite = []
    radius_match = []
    radius_label = []
    for radius in radius_values:
        radius_rows = [row for row in records if abs(float(row["radius_scale"]) - radius) < 1e-9]
        row = max(radius_rows, key=lambda item: float(item["composite_score"]))
        radius_composite.append(float(row["composite_score"]))
        radius_match.append(float(row["match_score"]))
        radius_label.append(float(row["label_score"]))
    axes[0].plot(radius_values, radius_composite, marker="o", label="composite")
    axes[0].plot(radius_values, radius_match, marker=".", label="match")
    axes[0].plot(radius_values, radius_label, marker=".", label="label")
    axes[0].axvline(float(best["radius_scale"]), color="red", linestyle="--", linewidth=1.1)
    axes[0].set_title("Best score per radius scale")
    axes[0].set_xlabel("PCz / projection radius scale")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    if len(dx_values) > 1 or len(dy_values) > 1:
        score_grid = np.full((len(dy_values), len(dx_values)), np.nan, dtype=np.float32)
        best_radius = float(best["radius_scale"])
        for row in records:
            if abs(float(row["radius_scale"]) - best_radius) > 1e-9:
                continue
            y = dy_values.index(float(row["dy_px"]))
            x = dx_values.index(float(row["dx_px"]))
            score_grid[y, x] = float(row["composite_score"])
        im = axes[1].imshow(
            score_grid,
            origin="lower",
            cmap="viridis",
            extent=[min(dx_values), max(dx_values), min(dy_values), max(dy_values)],
            aspect="auto",
        )
        axes[1].scatter([float(best["dx_px"])], [float(best["dy_px"])], color="red", marker="x", s=70, linewidths=2)
        fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "PCx/PCy fixed\n(radius-only refinement)", ha="center", va="center")
    axes[1].set_title("PC shift composite score")
    axes[1].set_xlabel("PC x shift (px)")
    axes[1].set_ylabel("PC y shift (px)")

    h5_rows = [
        row
        for row in records
        if abs(float(row["dx_px"])) < 1e-9
        and abs(float(row["dy_px"])) < 1e-9
        and abs(float(row["radius_scale"]) - 1.0) < 1e-9
    ]
    h5_score = float(h5_rows[0]["composite_score"]) if h5_rows else float("nan")
    axes[2].bar(["H5 PC/radius", "Refined"], [h5_score, float(best["composite_score"])], color=["#6b7280", "#4c9f70"])
    axes[2].set_ylabel("Composite score")
    axes[2].set_title("Local labeled refinement")
    axes[2].grid(axis="y", alpha=0.25)
    for i, score in enumerate([h5_score, float(best["composite_score"])]):
        axes[2].text(i, score, f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_search_csv(records: list[dict], out_path: Path) -> None:
    fieldnames = [
        "dx_px",
        "dy_px",
        "radius_scale",
        "pcx",
        "pcy",
        "effective_pcz",
        "match_score",
        "label_score",
        "label_mean_angle_deg",
        "composite_score",
        "loss",
        "rotation_quat_xyzw",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row and key != "rotation_quat_xyzw"},
                    "rotation_quat_xyzw": json.dumps(row["rotation_quat_xyzw"]),
                }
            )


def process_one(
    args,
    map_spec,
    master: MasterSphere,
    index: int,
    batch_dir: Path,
) -> dict:
    out_dir = batch_dir / f"idx_{index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = read_pattern_bundle(args.h5, map_spec, index)
    phase_id = int(bundle.ang_record.get("Phase", 1))
    phase_info, families = read_phase_hkl_families(args.h5, map_spec.h5_group, phase_id)
    if not families:
        raise ValueError(f"No HKL families found for phase {phase_id}")

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
    initial_label_score, initial_label_angle = label_score(initial_assignments)

    save_detector_label_overlay(initial_result, initial_assignments, out_dir / "01_h5_detector_bands_with_inferred_hkl.png")
    save_final_spatial_visualization(
        initial_result,
        master,
        products,
        out_dir / "02_h5_pc_radius_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )

    pc_shifts = sorted(set(parse_float_list(args.pc_shifts_px) + [0.0]))
    radius_scales = sorted(set(parse_float_list(args.radius_scales) + [1.0]))
    local_steps = parse_float_list(args.local_steps_deg)

    records: list[dict] = []
    best_result: MatchResult | None = None
    best_record: dict | None = None
    best_assignments: list[LabelAssignment] = []
    height, width = prepared.image.shape
    candidates = [(dx, dy, radius) for radius in radius_scales for dy in pc_shifts for dx in pc_shifts]
    iterator = tqdm(candidates, desc=f"labeled radius idx_{index:05d}", leave=False) if tqdm is not None else candidates

    for dx_px, dy_px, radius_scale in iterator:
        pc = corrected_pc(bundle.pc, dx_px, dy_px, radius_scale, height, width)
        candidate_prepared = prepared_with_pc(prepared, pc)
        rotation, match_score = deterministic_rotation_refine(
            candidate_prepared,
            master,
            initial_result.detector_transform,
            initial_result.rotation,
            local_steps,
        )
        result = MatchResult(
            label="labeled-band-radius-refined",
            score=float(match_score),
            rotation=rotation,
            convention_name=initial_result.convention_name,
            detector_transform=initial_result.detector_transform,
            prepared=candidate_prepared,
        )
        assignments = assign_labels(result, families, fixed_hkl_by_band=fixed_hkl_by_band)
        current_label_score, current_label_angle = label_score(assignments)
        composite_score = args.match_score_weight * float(match_score) + args.label_score_weight * current_label_score
        record = {
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "radius_scale": float(radius_scale),
            "pcx": float(pc[0]),
            "pcy": float(pc[1]),
            "effective_pcz": float(pc[2]),
            "match_score": float(match_score),
            "label_score": float(current_label_score),
            "label_mean_angle_deg": float(current_label_angle),
            "composite_score": float(composite_score),
            "loss": float(-composite_score),
            "rotation_quat_xyzw": rotation.as_quat().tolist(),
        }
        records.append(record)
        if best_record is None or composite_score > float(best_record["composite_score"]):
            best_record = record
            best_result = result
            best_assignments = assignments

    assert best_result is not None and best_record is not None
    save_score_landscape(records, out_dir / "03_labeled_radius_score_landscape.png")
    save_labeled_alignment(
        best_result,
        master,
        best_assignments,
        out_dir / "04_refined_labeled_band_alignment.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    save_final_spatial_visualization(
        best_result,
        master,
        products,
        out_dir / "05_refined_radius_final_spatial.png",
        lon_count=args.sphere_lon_count,
        colat_count=args.sphere_colat_count,
    )
    write_search_csv(records, out_dir / "labeled_radius_search_results.csv")

    summary = {
        "map": map_spec.key,
        "map_label": map_spec.label,
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "phase": phase_info,
        "hkl_families": [
            {
                key: value
                for key, value in asdict(family).items()
                if key != "normals"
            }
            | {"label": family.label, "normal_count": int(len(family.normals))}
            for family in families
        ],
        "h5_pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "search_parameters": {
            "pc_shifts_px": pc_shifts,
            "radius_scales": radius_scales,
            "local_steps_deg": local_steps,
            "match_score_weight": args.match_score_weight,
            "label_score_weight": args.label_score_weight,
            "interpretation": "radius_scale multiplies H5 PCz; detector/sphere display radius remains a unit sphere except for a tiny visualization-only surface lift.",
            "label_source": "OHP bands do not store per-band HKL labels in this EDAX H5. Labels are inferred by matching transformed H5 band normals to the phase HKL family normals.",
        },
        "line_variant": prepared.line_variant.name,
        "line_variant_score": prepared.line_variant_score,
        "variant_diagnostics": variant_diagnostics,
        "initial_match": initial_result.to_json_dict(),
        "initial_label_score": initial_label_score,
        "initial_label_mean_angle_deg": initial_label_angle,
        "initial_assignments": [item.to_json_dict() for item in initial_assignments],
        "best_correction": best_record,
        "best_match": best_result.to_json_dict(),
        "best_assignments": [item.to_json_dict() for item in best_assignments],
        "outputs": {
            "detector_labels": str(out_dir / "01_h5_detector_bands_with_inferred_hkl.png"),
            "h5_pc_radius_final_spatial": str(out_dir / "02_h5_pc_radius_final_spatial.png"),
            "score_landscape": str(out_dir / "03_labeled_radius_score_landscape.png"),
            "labeled_alignment": str(out_dir / "04_refined_labeled_band_alignment.png"),
            "refined_final_spatial": str(out_dir / "05_refined_radius_final_spatial.png"),
            "search_csv": str(out_dir / "labeled_radius_search_results.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "index",
        "row",
        "col",
        "phase_id",
        "phase_name",
        "initial_match_score",
        "initial_label_score",
        "initial_label_mean_angle_deg",
        "best_dx_px",
        "best_dy_px",
        "best_radius_scale",
        "best_match_score",
        "best_label_score",
        "best_label_mean_angle_deg",
        "best_composite_score",
        "score_gain",
        "label_score_gain",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in rows:
            best = summary["best_correction"]
            writer.writerow(
                {
                    "index": summary["index"],
                    "row": summary["row"],
                    "col": summary["col"],
                    "phase_id": summary["phase"]["phase_id"],
                    "phase_name": summary["phase"]["material_name"],
                    "initial_match_score": summary["initial_match"]["score"],
                    "initial_label_score": summary["initial_label_score"],
                    "initial_label_mean_angle_deg": summary["initial_label_mean_angle_deg"],
                    "best_dx_px": best["dx_px"],
                    "best_dy_px": best["dy_px"],
                    "best_radius_scale": best["radius_scale"],
                    "best_match_score": best["match_score"],
                    "best_label_score": best["label_score"],
                    "best_label_mean_angle_deg": best["label_mean_angle_deg"],
                    "best_composite_score": best["composite_score"],
                    "score_gain": best["match_score"] - summary["initial_match"]["score"],
                    "label_score_gain": best["label_score"] - summary["initial_label_score"],
                }
            )


def save_global_radius_aggregate(batch_dir: Path, summaries: list[dict]) -> tuple[Path, Path, list[dict]]:
    by_radius: dict[float, dict[str, float]] = {}
    for summary in summaries:
        csv_path = batch_dir / f"idx_{summary['index']:05d}" / "labeled_radius_search_results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if abs(float(row["dx_px"])) > 1e-9 or abs(float(row["dy_px"])) > 1e-9:
                    continue
                radius = float(row["radius_scale"])
                bucket = by_radius.setdefault(
                    radius,
                    {
                        "patterns": 0.0,
                        "match": 0.0,
                        "label": 0.0,
                        "angle": 0.0,
                        "composite": 0.0,
                    },
                )
                bucket["patterns"] += 1.0
                bucket["match"] += float(row["match_score"])
                bucket["label"] += float(row["label_score"])
                bucket["angle"] += float(row["label_mean_angle_deg"])
                bucket["composite"] += float(row["composite_score"])

    rows: list[dict] = []
    for radius, bucket in sorted(by_radius.items()):
        count = max(1.0, bucket["patterns"])
        rows.append(
            {
                "radius_scale": float(radius),
                "patterns": int(bucket["patterns"]),
                "mean_match_score": float(bucket["match"] / count),
                "mean_label_score": float(bucket["label"] / count),
                "mean_label_angle_deg": float(bucket["angle"] / count),
                "mean_composite_score": float(bucket["composite"] / count),
                "mean_loss": float(-bucket["composite"] / count),
            }
        )

    csv_path = batch_dir / "global_radius_aggregate.csv"
    plot_path = batch_dir / "global_radius_aggregate.png"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        best = max(rows, key=lambda row: row["mean_composite_score"])
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
        radii = [row["radius_scale"] for row in rows]
        axes[0].plot(radii, [row["mean_composite_score"] for row in rows], marker="o", label="composite")
        axes[0].plot(radii, [row["mean_match_score"] for row in rows], marker=".", label="match")
        axes[0].plot(radii, [row["mean_label_score"] for row in rows], marker=".", label="label")
        axes[0].axvline(best["radius_scale"], color="red", linestyle="--", linewidth=1.1)
        axes[0].set_title(f"Global radius score, best={best['radius_scale']:.4f}")
        axes[0].set_xlabel("PCz / projection radius scale")
        axes[0].set_ylabel("Mean score")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].plot(radii, [row["mean_label_angle_deg"] for row in rows], marker="o", color="#7c3aed")
        axes[1].axvline(best["radius_scale"], color="red", linestyle="--", linewidth=1.1)
        axes[1].set_title("Mean same-HKL band angle error")
        axes[1].set_xlabel("PCz / projection radius scale")
        axes[1].set_ylabel("Mean angle (deg)")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return csv_path, plot_path, rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine PCz/radius registration with inferred HKL-labeled Kikuchi band consistency.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma list or Python-like ranges, for example 0,100,500:1000:100.")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strategy", choices=["linspace", "sequential"], default="linspace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "labeled_band_radius_refinement")
    parser.add_argument("--pc-shifts-px", default="0")
    parser.add_argument("--radius-scales", default="0.90,0.94,0.98,1.00,1.02,1.06,1.10")
    parser.add_argument("--local-steps-deg", default="0.75,0.25")
    parser.add_argument("--match-score-weight", type=float, default=1.0)
    parser.add_argument("--label-score-weight", type=float, default=0.25)
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
    map_spec = default_map_specs(args.data_dir)[args.map]
    total = read_up2_info(map_spec.up2_path).count
    indices = parse_indices(args.indices, total, args.count, args.strategy) if args.indices or args.count > 1 else [args.index]
    if not indices:
        raise ValueError("No valid pattern indices selected")

    master_h5 = resolve_master_path(args.master_h5)
    print(f"Loading master sphere: {master_h5}")
    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )

    batch_dir = args.out_dir / args.map
    batch_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    iterator = tqdm(indices, desc="labeled band radius refinement") if tqdm is not None else indices
    for index in iterator:
        print(f"Processing {map_spec.label} index={index}")
        summaries.append(process_one(args, map_spec, master, int(index), batch_dir))

    write_batch_csv(summaries, batch_dir / "batch_labeled_radius_summary.csv")
    global_radius_csv, global_radius_plot, global_radius_rows = save_global_radius_aggregate(batch_dir, summaries)
    (batch_dir / "batch_summary.json").write_text(json.dumps(jsonable(summaries), indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "05_refined_radius_final_spatial.png" for summary in summaries],
        batch_dir / "contact_sheet_refined_radius_final_spatial.png",
        thumb_width=900,
        columns=2,
    )
    make_contact_sheet(
        [batch_dir / f"idx_{summary['index']:05d}" / "04_refined_labeled_band_alignment.png" for summary in summaries],
        batch_dir / "contact_sheet_labeled_band_alignment.png",
        thumb_width=900,
        columns=2,
    )

    print(f"Saved labeled band radius refinement results to: {batch_dir}")
    print(f"Batch CSV: {batch_dir / 'batch_labeled_radius_summary.csv'}")
    if global_radius_rows:
        best_global = max(global_radius_rows, key=lambda row: row["mean_composite_score"])
        print(f"Global radius aggregate CSV: {global_radius_csv}")
        print(f"Global radius aggregate plot: {global_radius_plot}")
        print(f"Best global radius_scale={best_global['radius_scale']:.4f}, mean_loss={best_global['mean_loss']:.4f}")
    print(f"Final contact sheet: {batch_dir / 'contact_sheet_refined_radius_final_spatial.png'}")


if __name__ == "__main__":
    main()
