from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


DEFAULT_H5 = Path(r"D:\EBSD project\EBSD-data\Pt-1\20251209Pt.edaxh5")
DEFAULT_MAP_GROUP = "20251209/Pt-3/Area 3-90/OIM Map 1"
DEFAULT_EDAX_IPF = Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\90.bmp")
DEFAULT_FINETUNE_SUMMARY = (
    Path("outputs") / "pt3_area90_finetuned_ipf_map" / "finetune" / "stable_global_pc_orientation_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt3_area90_finetuned_ipf_map"


def fold_cubic(directions: np.ndarray) -> np.ndarray:
    directions = directions.astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    folded = np.sort(np.abs(directions), axis=1)
    folded = np.column_stack([folded[:, 1], folded[:, 0], folded[:, 2]])
    folded /= np.linalg.norm(folded, axis=1, keepdims=True) + 1e-12
    return folded


def stereo(vectors: np.ndarray) -> np.ndarray:
    return np.column_stack([vectors[:, 0] / (1.0 + vectors[:, 2]), vectors[:, 1] / (1.0 + vectors[:, 2])])


def cubic_ipf_rgb_from_directions(directions: np.ndarray) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        ],
        dtype=np.float64,
    )
    triangle = stereo(vertices)
    transform = np.array(
        [
            [triangle[0, 0], triangle[1, 0], triangle[2, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1]],
            [1.0, 1.0, 1.0],
        ]
    )
    points = stereo(fold_cubic(directions))
    rhs = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    barycentric = np.linalg.solve(transform, rhs.T).T
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= barycentric.sum(axis=1, keepdims=True) + 1e-12
    rgb = np.sqrt(np.clip(barycentric[:, [0, 1, 2]], 0.0, 1.0))
    rgb /= rgb.max(axis=1, keepdims=True) + 1e-12
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def read_finetune_rotations(summary_path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No finetune rows found in {summary_path}")
    rotations = np.array(
        [
            [
                float(row["orientation_rot_x_deg"]),
                float(row["orientation_rot_y_deg"]),
                float(row["orientation_rot_z_deg"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = np.array([max(0.0, float(row["final_combined_score"])) for row in rows], dtype=np.float64)
    return rows, rotations, weights


def choose_residual_rotation(rotations: np.ndarray, weights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "median":
        return np.median(rotations, axis=0)
    if mode == "weighted_mean":
        return np.average(rotations, axis=0, weights=weights / (weights.sum() + 1e-12))
    raise ValueError(f"Unknown residual mode: {mode}")


def copy_edax_scalebar(source: Image.Image, target: Image.Image) -> Image.Image:
    source_arr = np.asarray(source)
    width, height = source.size
    roi = source_arr[int(height * 0.70) :, : int(width * 0.45), :]
    white = np.all(roi > 238, axis=2)
    ys, xs = np.where(white)
    result = target.copy()
    if len(xs) > 500:
        x0 = max(0, int(xs.min()) - 6)
        x1 = min(width, int(xs.max()) + 7)
        y0 = max(0, int(ys.min() + int(height * 0.70)) - 6)
        y1 = min(height, int(ys.max() + int(height * 0.70)) + 7)
        result.paste(source.crop((x0, y0, x1, y1)), (x0, y0))
        return result

    draw = ImageDraw.Draw(result)
    x0, y0 = 18, height - 84
    draw.rectangle([x0, y0, x0 + 250, y0 + 58], fill=(255, 255, 255))
    draw.rectangle([x0 + 18, y0 + 37, x0 + 232, y0 + 48], fill=(0, 0, 0))
    draw.text((x0 + 125, y0 + 18), "20 um", fill=(0, 0, 0), anchor="mm")
    return result


def render_comparison(edax: Image.Image, base: Image.Image, refined: Image.Image, residual: np.ndarray, path: Path) -> None:
    width, height = edax.size
    label_height = 70
    canvas = Image.new("RGB", (width * 3, height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    panels = [
        ("EDAX original 90.bmp", edax),
        ("H5 original IPF: G @ ND", base),
        (f"Finetuned IPF: dR=({residual[0]:+.2f},{residual[1]:+.2f},{residual[2]:+.2f}) deg", refined),
    ]
    for index, (title, image) in enumerate(panels):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + width // 2, 18), title, fill=(0, 0, 0), anchor="ma", font=font)
        draw.text((x + width // 2, 45), f"{width} x {height} px", fill=(45, 45, 45), anchor="ma", font=small)
    canvas.save(path)


def export_finetuned_ipf(args: argparse.Namespace) -> dict[str, object]:
    rows, rotations, weights = read_finetune_rotations(args.finetune_summary)
    residual = choose_residual_rotation(rotations, weights, args.residual_mode)
    delta = Rotation.from_rotvec(np.deg2rad(residual)).as_matrix()

    with h5py.File(args.h5, "r") as h5:
        group = h5[args.map_group]
        nrows = int(group["Sample/Number Of Rows"][0])
        ncols = int(group["Sample/Number Of Columns"][0])
        step_x = float(group["Sample/Step X"][0])
        step_y = float(group["Sample/Step Y"][0])
        orientations = group["EBSD/ANG/DATA/DATA"]["Orientations"][:].astype(np.float64).reshape(-1, 3, 3)

    nd = np.array([0.0, 0.0, 1.0])
    base_dirs = np.einsum("nij,j->ni", orientations, nd)
    refined_orientations = np.einsum("nij,jk->nik", orientations, delta.T)
    refined_dirs = np.einsum("nij,j->ni", refined_orientations, nd)
    base_ipf = cubic_ipf_rgb_from_directions(base_dirs).reshape(nrows, ncols, 3)
    refined_ipf = cubic_ipf_rgb_from_directions(refined_dirs).reshape(nrows, ncols, 3)

    edax = Image.open(args.edax_ipf).convert("RGB")
    target_size = edax.size
    base_image = Image.fromarray((base_ipf * 255 + 0.5).astype(np.uint8), "RGB").resize(
        target_size, Image.Resampling.BICUBIC
    )
    refined_image = Image.fromarray((refined_ipf * 255 + 0.5).astype(np.uint8), "RGB").resize(
        target_size, Image.Resampling.BICUBIC
    )
    refined_style = copy_edax_scalebar(edax, refined_image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.output_dir / "pt3_area90_h5_ipf_original_convention_714x550.png"
    refined_clean_path = args.output_dir / "pt3_area90_finetuned_ipf_clean_714x550.png"
    refined_style_path = args.output_dir / "pt3_area90_finetuned_ipf_edax_style_714x550.png"
    comparison_path = args.output_dir / "pt3_area90_edax_h5_finetuned_ipf_comparison_714x550_panels.png"
    metadata_path = args.output_dir / "pt3_area90_finetuned_ipf_metadata.json"
    params_path = args.output_dir / "pt3_area90_finetuned_ipf_parameters.csv"

    base_image.save(base_path)
    refined_image.save(refined_clean_path)
    refined_style.save(refined_style_path)
    render_comparison(edax, base_image, refined_style, residual, comparison_path)

    metadata = {
        "h5": str(args.h5),
        "h5_group": args.map_group,
        "edax_reference_ipf": str(args.edax_ipf),
        "finetune_summary": str(args.finetune_summary),
        "residual_mode": args.residual_mode,
        "output_resolution_px": {"width": target_size[0], "height": target_size[1]},
        "h5_grid": {
            "rows": nrows,
            "cols": ncols,
            "step_x_um": step_x,
            "step_y_um": step_y,
            "physical_width_um": ncols * step_x,
            "physical_height_um": nrows * step_y,
        },
        "global_pc_residual": {
            "delta_pcx": float(rows[0]["global_delta_pcx"]),
            "delta_pcy": float(rows[0]["global_delta_pcy"]),
            "delta_pcz": float(rows[0]["global_delta_pcz"]),
        },
        "orientation_residuals_deg": rotations.tolist(),
        "orientation_residual_used_deg": residual.tolist(),
        "orientation_residual_weighted_mean_deg": choose_residual_rotation(rotations, weights, "weighted_mean").tolist(),
        "ipf_convention": "row-major G @ ND, cubic fold to [001]-[101]-[111], [001]=red [101]=green [111]=blue",
        "residual_application": "G_refined = G @ delta.T, matching stable_global_pc_orientation_calibration.py",
        "outputs": {
            "base_clean": str(base_path),
            "refined_clean": str(refined_clean_path),
            "refined_edax_style": str(refined_style_path),
            "comparison": str(comparison_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with params_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["parameter", "value"])
        writer.writeheader()
        for key, value in [
            ("output_width_px", target_size[0]),
            ("output_height_px", target_size[1]),
            ("h5_rows", nrows),
            ("h5_cols", ncols),
            ("step_x_um", step_x),
            ("step_y_um", step_y),
            ("global_delta_pcx", rows[0]["global_delta_pcx"]),
            ("global_delta_pcy", rows[0]["global_delta_pcy"]),
            ("global_delta_pcz", rows[0]["global_delta_pcz"]),
            ("orientation_rx_deg", residual[0]),
            ("orientation_ry_deg", residual[1]),
            ("orientation_rz_deg", residual[2]),
        ]:
            writer.writerow({"parameter": key, "value": value})
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Pt-3 Area 3-90 IPF map after stable PC/orientation finetune."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--map-group", default=DEFAULT_MAP_GROUP)
    parser.add_argument("--edax-ipf", type=Path, default=DEFAULT_EDAX_IPF)
    parser.add_argument("--finetune-summary", type=Path, default=DEFAULT_FINETUNE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--residual-mode", choices=("median", "weighted_mean"), default="median")
    return parser


def main() -> None:
    metadata = export_finetuned_ipf(build_arg_parser().parse_args())
    print(f"Finetuned IPF: {metadata['outputs']['refined_edax_style']}")
    print(f"Comparison: {metadata['outputs']['comparison']}")
    print(f"Orientation residual used: {metadata['orientation_residual_used_deg']} deg")


if __name__ == "__main__":
    main()
