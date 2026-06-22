from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_IPF = Path(
    r"C:\Users\WHJ\OneDrive\xwechat_files\wxid_udhlesdsllnu22_8cd9\temp\RWTemp\2026-06\9e20f478899dc29eb19741386f9343c8\c9771c10a60de75b54062862d72e2902.png"
)
DEFAULT_OUTPUT = Path(r"outputs\ipf_annotations")


@dataclass(frozen=True)
class IndexedPoint:
    group: str
    condition: str
    zone: tuple[float, float, float]
    tilt_deg: float
    pc: tuple[float, float, float]
    direction: np.ndarray
    nearest_hkl: tuple[int, int, int]
    angular_error_deg: float
    xy: tuple[float, float]


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
    z_axis = np.array(zone, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    h, k, _l = z_axis
    if abs(h) + abs(k) > 1e-12:
        x_axis = np.array([k, -h, 0.0], dtype=np.float64)
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def apparent_direction_with_standard_pc(
    zone: tuple[float, float, float],
    tilt_deg: float,
    actual_pc: tuple[float, float, float],
    standard_pc: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Direction indexed at the standard detector center for a simulated pattern."""
    pcx, pcy, pcz = actual_pc
    spcx, spcy, _spcz = standard_pc
    ray = np.array([(spcx - pcx) / pcz, (pcy - spcy) / pcz, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    direction = ray @ rotation_x_row(tilt_deg) @ detector_to_zone_crystal_matrix(zone)
    direction /= np.linalg.norm(direction)
    return direction


def nearest_integer_hkl(direction: np.ndarray, max_index: int = 20) -> tuple[tuple[int, int, int], float]:
    target = np.abs(direction).astype(np.float64)
    target /= np.linalg.norm(target)
    best_hkl = (0, 0, 1)
    best_angle = float("inf")
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                candidate = np.array([h, k, l], dtype=np.float64)
                candidate /= np.linalg.norm(candidate)
                dot = float(np.clip(np.dot(target, candidate), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(dot)))
                if angle < best_angle:
                    best_angle = angle
                    best_hkl = (h, k, l)
    return best_hkl, best_angle


def ipf_affine(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Affine map from stereographic IPF coordinates to the provided IPF image."""
    width, height = image_size
    # These are the [001], [101], [111] vertex positions in the user's IPF image.
    # They are scaled if the same image is resized.
    p001 = np.array([145.0 / 627.0 * width, 377.0 / 419.0 * height], dtype=np.float64)
    p101 = np.array([527.0 / 627.0 * width, 377.0 / 419.0 * height], dtype=np.float64)
    p111 = np.array([485.0 / 627.0 * width, 43.0 / 419.0 * height], dtype=np.float64)

    u101 = (1.0 / np.sqrt(2.0)) / (1.0 + 1.0 / np.sqrt(2.0))
    u111 = (1.0 / np.sqrt(3.0)) / (1.0 + 1.0 / np.sqrt(3.0))
    v111 = u111
    stereo = np.array([[0.0, 0.0, 1.0], [u101, 0.0, 1.0], [u111, v111, 1.0]])
    ax = np.linalg.solve(stereo, np.array([p001[0], p101[0], p111[0]]))
    ay = np.linalg.solve(stereo, np.array([p001[1], p101[1], p111[1]]))
    return ax, ay


def direction_to_ipf_xy(direction: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
    fundamental = np.sort(np.abs(direction).astype(np.float64))
    fundamental /= np.linalg.norm(fundamental)
    u = fundamental[1] / (1.0 + fundamental[2])
    v = fundamental[0] / (1.0 + fundamental[2])
    ax, ay = ipf_affine(image_size)
    x = float(np.dot(np.array([u, v, 1.0]), ax))
    y = float(np.dot(np.array([u, v, 1.0]), ay))
    return x, y


def make_points(image_size: tuple[int, int]) -> tuple[list[IndexedPoint], list[IndexedPoint]]:
    standard_pc = (0.5, 0.5, 0.5)
    group_123: list[IndexedPoint] = []
    for tilt in (0.0, -2.0, 2.0, 5.0):
        zone = (1.0, 2.0, 3.0)
        pc = standard_pc
        direction = apparent_direction_with_standard_pc(zone, tilt, pc, standard_pc)
        nearest, error = nearest_integer_hkl(direction)
        group_123.append(
            IndexedPoint(
                group="zone_123_tilt",
                condition=f"tilt={tilt:+g} deg",
                zone=zone,
                tilt_deg=tilt,
                pc=pc,
                direction=direction,
                nearest_hkl=nearest,
                angular_error_deg=error,
                xy=direction_to_ipf_xy(direction, image_size),
            )
        )

    group_135: list[IndexedPoint] = []
    for pcx in (0.4, 0.45, 0.5, 0.55):
        zone = (1.0, 3.0, 5.0)
        pc = (pcx, 0.5, 0.5)
        direction = apparent_direction_with_standard_pc(zone, 0.0, pc, standard_pc)
        nearest, error = nearest_integer_hkl(direction)
        group_135.append(
            IndexedPoint(
                group="zone_135_pcx_sweep",
                condition=f"PCx={pcx:.2f}",
                zone=zone,
                tilt_deg=0.0,
                pc=pc,
                direction=direction,
                nearest_hkl=nearest,
                angular_error_deg=error,
                xy=direction_to_ipf_xy(direction, image_size),
            )
        )
    return group_123, group_135


def draw_points(base: Image.Image, points: list[IndexedPoint], output_path: Path) -> None:
    scale = 4
    canvas = base.convert("RGBA").resize(
        (base.width * scale, base.height * scale), Image.Resampling.LANCZOS
    )
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = 8 * scale
    ring_width = 4 * scale
    for point in points:
        x = point.xy[0] * scale
        y = point.xy[1] * scale
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, outline=(0, 0, 0, 255), width=ring_width)
    canvas = Image.alpha_composite(canvas, overlay)
    canvas = canvas.resize(base.size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_csv(points: list[IndexedPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group",
                "condition",
                "zone_h",
                "zone_k",
                "zone_l",
                "tilt_deg",
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
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "group": point.group,
                    "condition": point.condition,
                    "zone_h": point.zone[0],
                    "zone_k": point.zone[1],
                    "zone_l": point.zone[2],
                    "tilt_deg": point.tilt_deg,
                    "pc_x": point.pc[0],
                    "pc_y": point.pc[1],
                    "pc_z": point.pc[2],
                    "direction_h": point.direction[0],
                    "direction_k": point.direction[1],
                    "direction_l": point.direction[2],
                    "nearest_h": point.nearest_hkl[0],
                    "nearest_k": point.nearest_hkl[1],
                    "nearest_l": point.nearest_hkl[2],
                    "angular_error_deg": point.angular_error_deg,
                    "ipf_x": point.xy[0],
                    "ipf_y": point.xy[1],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate simulated Kikuchi indexing results on a transparent cubic IPF."
    )
    parser.add_argument("--ipf", type=Path, default=DEFAULT_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Image.open(args.ipf).convert("RGBA")
    points_123, points_135 = make_points(base.size)
    out_123 = args.output_dir / "ipf_zone_123_tilt_points_transparent.png"
    out_135 = args.output_dir / "ipf_zone_135_pcx_points_transparent.png"
    draw_points(base, points_123, out_123)
    draw_points(base, points_135, out_135)
    write_csv(points_123 + points_135, args.output_dir / "simulated_ipf_indexed_points.csv")
    print(f"saved: {out_123}")
    print(f"saved: {out_135}")
    print(f"metadata: {args.output_dir / 'simulated_ipf_indexed_points.csv'}")
    for point in points_123 + points_135:
        h, k, l = point.nearest_hkl
        print(
            f"{point.group:20s} {point.condition:12s} "
            f"dir=[{point.direction[0]:.4f} {point.direction[1]:.4f} {point.direction[2]:.4f}] "
            f"nearest=[{h} {k} {l}] err={point.angular_error_deg:.3f} deg"
        )


if __name__ == "__main__":
    main()
