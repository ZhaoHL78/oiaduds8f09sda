from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.restoration import inpaint_biharmonic

from annotate_simulated_ipf_points import direction_to_ipf_xy, nearest_integer_hkl
from project_edax_oim_to_sphere import (
    load_master_samplers,
    make_master_sampler,
    preprocess_master_hemisphere,
    sample_master,
)
from simulate_111_tilt_kikuchi_patterns import (
    detector_rays_edax,
    detector_to_zone_crystal_matrix,
    rotation_x_row,
    save_pattern_png,
)


DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_IPF = Path(
    r"C:\Users\WHJ\OneDrive\xwechat_files\wxid_udhlesdsllnu22_8cd9\temp\RWTemp\2026-06\9e20f478899dc29eb19741386f9343c8\c9771c10a60de75b54062862d72e2902.png"
)
DEFAULT_OUTPUT = Path(r"outputs\simulated_123_detector_sample_tilt_cases")
DEFAULT_IPF_OUTPUT = Path(r"outputs\ipf_annotations")
PRIOR_IPF_WITH_POINTS = DEFAULT_IPF_OUTPUT / "ipf_zone_123_tilt_points_transparent.png"
PRIOR_IPF_POINTS_CSV = DEFAULT_IPF_OUTPUT / "simulated_ipf_indexed_points.csv"
ZONE_123 = (1.0, 2.0, 3.0)
PC_STANDARD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    detector_tilt_deg: float
    sample_tilt_deg: float
    ring_color: tuple[int, int, int, int]


@dataclass(frozen=True)
class CaseResult:
    spec: CaseSpec
    pattern: np.ndarray
    direction: np.ndarray
    nearest_hkl: tuple[int, int, int]
    angular_error_deg: float
    ipf_xy: tuple[float, float]
    output_path: Path


CASES = (
    CaseSpec(
        key="default",
        label="default detector/sample",
        detector_tilt_deg=0.0,
        sample_tilt_deg=0.0,
        ring_color=(0, 0, 0, 255),
    ),
    CaseSpec(
        key="detector_up_5deg",
        label="detector up 5 deg",
        detector_tilt_deg=5.0,
        sample_tilt_deg=0.0,
        ring_color=(0, 0, 0, 255),
    ),
    CaseSpec(
        key="sample_up_2deg",
        label="sample up 2 deg",
        detector_tilt_deg=0.0,
        sample_tilt_deg=2.0,
        ring_color=(220, 0, 0, 255),
    ),
    CaseSpec(
        key="sample_down_2deg",
        label="sample down 2 deg",
        detector_tilt_deg=0.0,
        sample_tilt_deg=-2.0,
        ring_color=(220, 0, 0, 255),
    ),
)


def case_matrix(spec: CaseSpec) -> np.ndarray:
    # Detector tilt changes the outgoing ray direction. Sample tilt changes the
    # crystal frame relative to the detector, so its inverse is used for lookup.
    return rotation_x_row(spec.detector_tilt_deg) @ rotation_x_row(-spec.sample_tilt_deg)


def simulate_case(
    detector_rays: np.ndarray,
    spec: CaseSpec,
    upper_sampler,
    lower_sampler,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = detector_to_zone_crystal_matrix(ZONE_123)
    transform = case_matrix(spec) @ matrix
    crystal_vectors = detector_rays @ transform
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    values = sample_master(crystal_vectors, upper_sampler, lower_sampler)

    center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = center_ray @ transform
    direction /= np.linalg.norm(direction)
    return values.reshape(shape).astype(np.float32, copy=False), direction


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\calibri.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save_gray_preview(results: list[CaseResult], output_path: Path) -> None:
    tile = 350
    pad = 30
    label_h = 62
    width = len(results) * tile + (len(results) + 1) * pad
    height = tile + label_h + 2 * pad
    sheet = Image.new("RGBA", (width, height), (226, 226, 226, 255))
    draw = ImageDraw.Draw(sheet)
    font = load_font(23)
    for i, result in enumerate(results):
        x = pad + i * (tile + pad)
        label = result.spec.label.replace(" detector/sample", "\ndetector/sample")
        draw.multiline_text((x + 18, pad), label, fill=(15, 15, 15, 255), font=font, spacing=2)
        tile_image = Image.open(result.output_path).convert("RGBA")
        tile_image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        sheet.alpha_composite(tile_image, (x, pad + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_contact_sheet(results: list[CaseResult], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.45))
    for ax, result in zip(axes, results):
        ax.imshow(result.pattern, cmap="gray", interpolation="nearest")
        ax.set_title(result.spec.label)
        ax.axis("off")
    fig.suptitle("(1,2,3) simulated Kikuchi patterns | PC=(0.5,0.5,0.5)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_ipf(base_ipf: Path, results: list[CaseResult], output_path: Path) -> None:
    base = load_ipf_base(base_ipf)
    scale = 4
    canvas = base.resize((base.width * scale, base.height * scale), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = 8 * scale
    width = 4 * scale
    for result in results:
        x = result.ipf_xy[0] * scale
        y = result.ipf_xy[1] * scale
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=result.spec.ring_color,
            width=width,
        )
    output = Image.alpha_composite(canvas, overlay).resize(base.size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def load_ipf_base(base_ipf: Path) -> Image.Image:
    if base_ipf.exists():
        return Image.open(base_ipf).convert("RGBA")
    cleaned = DEFAULT_IPF_OUTPUT / "ipf_base_cleaned_from_previous.png"
    if cleaned.exists():
        return Image.open(cleaned).convert("RGBA")
    return reconstruct_ipf_base_from_previous(cleaned)


def reconstruct_ipf_base_from_previous(output_path: Path) -> Image.Image:
    if not PRIOR_IPF_WITH_POINTS.exists() or not PRIOR_IPF_POINTS_CSV.exists():
        raise FileNotFoundError(
            f"IPF base image is missing and no prior IPF output can be cleaned: {DEFAULT_IPF}"
        )

    image = Image.open(PRIOR_IPF_WITH_POINTS).convert("RGBA")
    arr = np.array(image)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    with PRIOR_IPF_POINTS_CSV.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("group") != "zone_123_tilt":
                continue
            x = float(row["ipf_x"])
            y = float(row["ipf_y"])
            yy, xx = np.indices(mask.shape)
            mask |= (xx - x) ** 2 + (yy - y) ** 2 <= 17**2

    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    alpha = arr[:, :, 3]
    repaired = inpaint_biharmonic(rgb, mask, channel_axis=-1)
    repaired_rgba = np.dstack(
        [
            np.clip(np.round(repaired * 255.0), 0, 255).astype(np.uint8),
            alpha.astype(np.uint8),
        ]
    )
    output = Image.fromarray(repaired_rgba, mode="RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return output


def write_metadata(results: list[CaseResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "label",
                "detector_tilt_deg",
                "sample_tilt_deg",
                "pc_x",
                "pc_y",
                "pc_z",
                "direction_h",
                "direction_k",
                "direction_l",
                "nearest_h",
                "nearest_k",
                "nearest_l",
                "angular_error_deg",
                "ipf_x",
                "ipf_y",
                "pattern_png",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case": result.spec.key,
                    "label": result.spec.label,
                    "detector_tilt_deg": result.spec.detector_tilt_deg,
                    "sample_tilt_deg": result.spec.sample_tilt_deg,
                    "pc_x": PC_STANDARD[0],
                    "pc_y": PC_STANDARD[1],
                    "pc_z": PC_STANDARD[2],
                    "direction_h": result.direction[0],
                    "direction_k": result.direction[1],
                    "direction_l": result.direction[2],
                    "nearest_h": result.nearest_hkl[0],
                    "nearest_k": result.nearest_hkl[1],
                    "nearest_l": result.nearest_hkl[2],
                    "angular_error_deg": result.angular_error_deg,
                    "ipf_x": result.ipf_xy[0],
                    "ipf_y": result.ipf_xy[1],
                    "pattern_png": str(result.output_path),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate four (1,2,3) detector/sample tilt cases and annotate them on IPF."
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--ipf", type=Path, default=DEFAULT_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ipf-output-dir", type=Path, default=DEFAULT_IPF_OUTPUT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--mode", choices=("raw", "corrected", "band"), default="corrected")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(0.5, 99.5))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = (int(args.height), int(args.width))
    upper, lower, _upper_raw, _lower_raw = load_master_samplers(args.master)
    upper = preprocess_master_hemisphere(upper, args.mode)
    lower = preprocess_master_hemisphere(lower, args.mode)
    upper_sampler = make_master_sampler(upper)
    lower_sampler = make_master_sampler(lower)
    detector_rays = detector_rays_edax(shape, PC_STANDARD)

    base_ipf = load_ipf_base(args.ipf)
    results: list[CaseResult] = []
    for spec in CASES:
        pattern, direction = simulate_case(detector_rays, spec, upper_sampler, lower_sampler, shape)
        hkl, error = nearest_integer_hkl(direction)
        output_path = args.output_dir / "individual" / f"sim_123_{spec.key}_pc050.png"
        save_pattern_png(
            pattern,
            output_path,
            tuple(args.percentiles),
            circular_transparent=True,
            circle_inset=2,
        )
        result = CaseResult(
            spec=spec,
            pattern=pattern,
            direction=direction,
            nearest_hkl=hkl,
            angular_error_deg=error,
            ipf_xy=direction_to_ipf_xy(direction, base_ipf.size),
            output_path=output_path,
        )
        results.append(result)
        print(
            f"{spec.key}: detector_tilt={spec.detector_tilt_deg:+g}, "
            f"sample_tilt={spec.sample_tilt_deg:+g}, "
            f"direction=[{direction[0]:.4f} {direction[1]:.4f} {direction[2]:.4f}], "
            f"nearest=[{hkl[0]} {hkl[1]} {hkl[2]}], err={error:.3f} deg"
        )

    save_contact_sheet(results, args.output_dir / "simulated_123_detector_sample_tilt_contact_sheet.png")
    save_gray_preview(results, args.output_dir / "transparent_circle_preview_on_gray.png")
    draw_ipf(
        args.ipf,
        results,
        args.ipf_output_dir / "ipf_zone_123_detector_sample_tilt_points_transparent.png",
    )
    write_metadata(results, args.output_dir / "simulated_123_detector_sample_tilt_metadata.csv")
    print(f"patterns: {args.output_dir / 'individual'}")
    print(
        "ipf: "
        f"{args.ipf_output_dir / 'ipf_zone_123_detector_sample_tilt_points_transparent.png'}"
    )


if __name__ == "__main__":
    main()
