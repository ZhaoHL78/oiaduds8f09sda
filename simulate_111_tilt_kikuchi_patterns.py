from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from project_edax_oim_to_sphere import (
    load_master_samplers,
    make_master_sampler,
    preprocess_master_hemisphere,
    sample_master,
)


DEFAULT_H5 = Path(r"D:\project\EBSD2026\ebsd.edaxh5")
DEFAULT_MASTER = Path(
    r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
DEFAULT_OUTPUT = Path(r"outputs\simulated_111_tilt_patterns")
DEFAULT_MAP_GROUP = "/20260512/Cu/Area 1/OIM Map 1HighR"


@dataclass(frozen=True)
class SimulationResult:
    label: str
    tilt_deg: float
    pc_edax: tuple[float, float, float]
    pattern: np.ndarray
    output_path: Path


def read_default_pc(h5_path: Path, map_group: str) -> tuple[float, float, float]:
    with h5py.File(h5_path, "r") as h5:
        calibration = h5[map_group]["EBSD/ANG/HEADER/Pattern Center Calibration"]
        return (
            float(calibration["X-Star"][0]),
            float(calibration["Y-Star"][0]),
            float(calibration["Z-Star"][0]),
        )


def detector_rays_edax(
    shape: tuple[int, int],
    pc_edax: tuple[float, float, float],
) -> np.ndarray:
    """Return detector-frame unit rays using the EDAX/TSL PC convention."""
    height, width = shape
    pcx, pcy, pcz = pc_edax
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = (xx + 0.5 - pcx * width) / (pcz * width)
    y = (pcy * height - (yy + 0.5)) / (pcz * width)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.reshape(-1, 3)


def rotation_x_row(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def detector_to_zone_crystal_matrix(zone: tuple[float, float, float]) -> np.ndarray:
    """Map detector x/y/z directions to a crystal frame with z_d || zone."""
    z_axis = np.array(zone, dtype=np.float64)
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-12:
        raise ValueError("zone axis cannot be zero")
    z_axis /= norm

    h, k, _l = z_axis
    if abs(h) + abs(k) > 1e-12:
        x_axis = np.array([k, -h, 0.0], dtype=np.float64)
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def normalize_to_uint8(pattern: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    lo, hi = np.percentile(pattern, percentiles)
    if hi <= lo:
        lo = float(pattern.min())
        hi = float(pattern.max())
    scaled = np.clip((pattern - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def circular_alpha(shape: tuple[int, int], inset: int = 2, supersample: int = 4) -> Image.Image:
    height, width = shape
    hi_h = height * supersample
    hi_w = width * supersample
    yy, xx = np.indices((hi_h, hi_w), dtype=np.float32)
    cx = (hi_w - 1) / 2.0
    cy = (hi_h - 1) / 2.0
    radius = 0.5 * min(hi_h, hi_w) - inset * supersample
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(np.uint8) * 255
    image = Image.fromarray(mask, mode="L")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def simulate_pattern(
    detector_rays: np.ndarray,
    tilt_deg: float,
    zone: tuple[float, float, float],
    upper_sampler,
    lower_sampler,
    shape: tuple[int, int],
) -> np.ndarray:
    matrix = detector_to_zone_crystal_matrix(zone)
    crystal_vectors = detector_rays @ rotation_x_row(tilt_deg) @ matrix
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12
    values = sample_master(crystal_vectors, upper_sampler, lower_sampler)
    return values.reshape(shape).astype(np.float32, copy=False)


def save_pattern_png(
    pattern: np.ndarray,
    path: Path,
    percentiles: tuple[float, float],
    circular_transparent: bool = False,
    circle_inset: int = 2,
) -> None:
    grayscale = Image.fromarray(normalize_to_uint8(pattern, percentiles), mode="L")
    if circular_transparent:
        alpha = circular_alpha(pattern.shape, inset=circle_inset)
        image = Image.merge("RGBA", (grayscale, grayscale, grayscale, alpha))
    else:
        image = grayscale
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_contact_sheet(
    results: list[SimulationResult],
    output_path: Path,
    pc_edax: tuple[float, float, float],
    zone: tuple[float, float, float],
    master_path: Path,
    mode: str,
    percentiles: tuple[float, float],
    circular_transparent: bool,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.6))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        ax.imshow(
            normalize_to_uint8(result.pattern, percentiles),
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        if circular_transparent:
            height, width = result.pattern.shape
            theta = np.linspace(0, 2 * np.pi, 720)
            radius = 0.5 * min(height, width) - 2
            ax.plot(
                (width - 1) / 2 + radius * np.cos(theta),
                (height - 1) / 2 + radius * np.sin(theta),
                color="white",
                linewidth=0.6,
                alpha=0.65,
            )
        ax.set_title(result.label)
        ax.axis("off")

    fig.suptitle(
        (
            f"[{zone[0]:g} {zone[1]:g} {zone[2]:g}] single-crystal Kikuchi simulation | "
            "EDAX PC="
            f"({pc_edax[0]:.6f}, {pc_edax[1]:.6f}, {pc_edax[2]:.6f}) | "
            f"master={master_path.name} | mode={mode}"
            f"{' | circular transparent PNGs' if circular_transparent else ''}"
        ),
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    results: list[SimulationResult],
    output_path: Path,
    pc_edax: tuple[float, float, float],
    zone: tuple[float, float, float],
    shape: tuple[int, int],
    master_path: Path,
    mode: str,
    circular_transparent: bool,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tilt_deg",
                "zone_h",
                "zone_k",
                "zone_l",
                "pc_x_edax",
                "pc_y_edax",
                "pc_z_edax",
                "height",
                "width",
                "master",
                "mode",
                "circular_transparent",
                "output_png",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "tilt_deg": result.tilt_deg,
                    "zone_h": zone[0],
                    "zone_k": zone[1],
                    "zone_l": zone[2],
                    "pc_x_edax": result.pc_edax[0],
                    "pc_y_edax": result.pc_edax[1],
                    "pc_z_edax": result.pc_edax[2],
                    "height": shape[0],
                    "width": shape[1],
                    "master": str(master_path),
                    "mode": mode,
                    "circular_transparent": circular_transparent,
                    "output_png": str(result.output_path),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate zone-axis single-crystal Kikuchi patterns with EDAX PC, "
            "vertical tilt, and optional horizontal PC drift."
        )
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--map-group", default=DEFAULT_MAP_GROUP)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument(
        "--zone",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("H", "K", "L"),
        help="Crystal direction aligned with the detector center at zero tilt.",
    )
    parser.add_argument(
        "--pc",
        type=float,
        nargs=3,
        default=None,
        metavar=("PCX", "PCY", "PCZ"),
        help="Fixed EDAX PC. If omitted, read X-Star/Y-Star/Z-Star from the H5 map group.",
    )
    parser.add_argument(
        "--tilts",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--pc-x-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional EDAX PCx sweep. PCy and PCz come from --pc or the H5 default PC.",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "corrected", "band"),
        default="corrected",
        help="Master sphere intensity used for simulation.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(0.5, 99.5),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--circular-transparent",
        action="store_true",
        help="Save each simulated pattern as an EDAX-like circular RGBA PNG.",
    )
    parser.add_argument(
        "--circle-inset",
        type=int,
        default=2,
        help="Inset of the circular alpha mask in pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = (int(args.height), int(args.width))
    pc_edax = tuple(args.pc) if args.pc is not None else read_default_pc(args.h5, args.map_group)
    zone = tuple(float(v) for v in args.zone)

    upper, lower, _upper_raw, _lower_raw = load_master_samplers(args.master)
    upper = preprocess_master_hemisphere(upper, args.mode)
    lower = preprocess_master_hemisphere(lower, args.mode)
    upper_sampler = make_master_sampler(upper)
    lower_sampler = make_master_sampler(lower)

    results: list[SimulationResult] = []
    pc_values = [pc_edax]
    if args.pc_x_values is not None:
        pc_values = [(float(pcx), pc_edax[1], pc_edax[2]) for pcx in args.pc_x_values]

    for current_pc in pc_values:
        detector_rays = detector_rays_edax(shape, current_pc)
        for tilt in args.tilts:
            pattern = simulate_pattern(
                detector_rays=detector_rays,
                tilt_deg=float(tilt),
                zone=zone,
                upper_sampler=upper_sampler,
                lower_sampler=lower_sampler,
                shape=shape,
            )
            zone_text = f"{int(zone[0])}{int(zone[1])}{int(zone[2])}"
            stem = (
                f"sim_{zone_text}_tilt_{float(tilt):+05.1f}deg_"
                f"pcx{current_pc[0]:.3f}_pcy{current_pc[1]:.3f}_pcz{current_pc[2]:.3f}"
            )
            stem = stem.replace("+", "p").replace("-", "m").replace(".", "p")
            output_path = args.output_dir / "individual" / f"{stem}.png"
            save_pattern_png(
                pattern,
                output_path,
                tuple(args.percentiles),
                circular_transparent=args.circular_transparent,
                circle_inset=args.circle_inset,
            )
            label = (
                f"tilt {float(tilt):+g} deg\n"
                f"PC=({current_pc[0]:.3f},{current_pc[1]:.3f},{current_pc[2]:.3f})"
            )
            results.append(SimulationResult(label, float(tilt), current_pc, pattern, output_path))
            print(
                f"saved zone=({zone[0]:g},{zone[1]:g},{zone[2]:g}) "
                f"tilt {float(tilt):+g} deg PC={current_pc}: {output_path}"
            )

    save_contact_sheet(
        results=results,
        output_path=args.output_dir / "simulated_zone_contact_sheet.png",
        pc_edax=pc_edax,
        zone=zone,
        master_path=args.master,
        mode=args.mode,
        percentiles=tuple(args.percentiles),
        circular_transparent=args.circular_transparent,
    )
    write_metadata(
        results=results,
        output_path=args.output_dir / "simulated_zone_metadata.csv",
        pc_edax=pc_edax,
        zone=zone,
        shape=shape,
        master_path=args.master,
        mode=args.mode,
        circular_transparent=args.circular_transparent,
    )
    print(f"base EDAX PC: {pc_edax}")
    print(f"zone axis: {zone}")
    print(f"contact sheet: {args.output_dir / 'simulated_zone_contact_sheet.png'}")
    print(f"metadata: {args.output_dir / 'simulated_zone_metadata.csv'}")


if __name__ == "__main__":
    main()
