from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, exposure, morphology


DEFAULT_H5_PATH = Path("ebsd.edaxh5")
DEFAULT_OUT_DIR = Path("outputs") / "publication_h5_kikuchi_bands"
DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class MapSpec:
    key: str
    label: str
    h5_group: str
    up2_path: Path
    rows: int
    cols: int


@dataclass(frozen=True)
class Up2Info:
    path: Path
    version: int
    width: int
    height: int
    header_bytes: int
    count: int


@dataclass(frozen=True)
class OHPHeader:
    circle_size: int
    max_band_count: int
    max_rho_fraction: float
    max_band_width: float
    theta_step_size: float


@dataclass(frozen=True)
class Band:
    rho_bin: float
    theta_deg: float
    width: float
    intensity: float


@dataclass(frozen=True)
class LineVariant:
    name: str
    theta_is_line_angle: bool
    rho_sign: float
    y_axis: str


@dataclass(frozen=True)
class LineSegment:
    band_index: int
    band: Band
    row0: float
    col0: float
    row1: float
    col1: float


@dataclass(frozen=True)
class PatternBundle:
    map_spec: MapSpec
    index: int
    row: int
    col: int
    pattern_u16: np.ndarray
    pc: tuple[float, float, float]
    bands: list[Band]
    ohp_header: OHPHeader
    ang_record: dict[str, object]


PUBLICATION_VARIANT = LineVariant("normal_theta_rho+_yup", theta_is_line_angle=False, rho_sign=1.0, y_axis="up")


def default_map_specs(data_dir: Path) -> dict[str, MapSpec]:
    return {
        "area1_high": MapSpec(
            key="area1_high",
            label="Area 1 HighR",
            h5_group="20260512/Cu/Area 1/OIM Map 1HighR",
            up2_path=data_dir / "20260512_Cu_Area 1_OIM Map 1.up2",
            rows=195,
            cols=218,
        ),
        "area2_high": MapSpec(
            key="area2_high",
            label="Area 2 HighR",
            h5_group="20260512/Cu/Area 2/OIM Map 2HighR",
            up2_path=data_dir / "20260512_Cu_Area 2_OIM Map 1.up2",
            rows=159,
            cols=178,
        ),
    }


def scalar(group: h5py.Group, path: str) -> float:
    return float(np.asarray(group[path][()]).reshape(-1)[0])


def scalar_int(group: h5py.Group, path: str) -> int:
    return int(np.asarray(group[path][()]).reshape(-1)[0])


def read_up2_info(path: Path) -> Up2Info:
    with path.open("rb") as file:
        version, width, height, header_bytes = struct.unpack("<4I", file.read(16))
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    payload_bytes = path.stat().st_size - header_bytes
    return Up2Info(
        path=path,
        version=version,
        width=width,
        height=height,
        header_bytes=header_bytes,
        count=payload_bytes // pattern_bytes,
    )


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, Up2Info]:
    info = read_up2_info(path)
    offset = info.header_bytes + index * info.width * info.height * np.dtype("<u2").itemsize
    with path.open("rb") as file:
        file.seek(offset)
        pattern = np.fromfile(file, dtype="<u2", count=info.width * info.height).reshape(info.height, info.width)
    return pattern, info


def read_ohp_header(map_group: h5py.Group) -> OHPHeader:
    header = map_group["EBSD/OHP/HEADER"]
    return OHPHeader(
        circle_size=scalar_int(header, "Circle Size"),
        max_band_count=scalar_int(header, "Maximum Band Count"),
        max_rho_fraction=scalar(header, "Maximum Rho Fraction"),
        max_band_width=scalar(header, "Maximum Band Width"),
        theta_step_size=scalar(header, "Theta Step Size"),
    )


def read_pattern_center(map_group: h5py.Group) -> tuple[float, float, float]:
    pc_group = map_group["EBSD/ANG/HEADER/Pattern Center Calibration"]
    return (
        scalar(pc_group, "X-Star"),
        scalar(pc_group, "Y-Star"),
        scalar(pc_group, "Z-Star"),
    )


def read_ang_record(map_group: h5py.Group, index: int) -> dict[str, object]:
    record = map_group["EBSD/ANG/DATA/DATA"][index]
    out: dict[str, object] = {}
    for name in record.dtype.names or []:
        value = record[name]
        if isinstance(value, np.ndarray):
            out[name] = value.astype(float).tolist()
        elif np.issubdtype(np.asarray(value).dtype, np.integer):
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def read_bands(map_group: h5py.Group, index: int) -> list[Band]:
    raw = np.asarray(map_group["EBSD/OHP/DATA/DATA"][index], dtype=np.float32).reshape(-1, 4)
    bands = []
    for rho_bin, theta_deg, width, intensity in raw:
        if np.isfinite([rho_bin, theta_deg, width, intensity]).all() and intensity > 0 and width > 0:
            bands.append(Band(float(rho_bin), float(theta_deg), float(width), float(intensity)))
    bands.sort(key=lambda band: band.intensity, reverse=True)
    return bands


def read_pattern_bundle(h5_path: Path, map_spec: MapSpec, index: int) -> PatternBundle:
    pattern, info = read_up2_pattern(map_spec.up2_path, index)
    if map_spec.rows * map_spec.cols != info.count:
        raise ValueError(f"{map_spec.key}: rows*cols does not match UP2 pattern count")
    with h5py.File(h5_path, "r") as h5:
        map_group = h5[map_spec.h5_group]
        pc = read_pattern_center(map_group)
        ohp_header = read_ohp_header(map_group)
        bands = read_bands(map_group, index)
        ang_record = read_ang_record(map_group, index)
    return PatternBundle(
        map_spec=map_spec,
        index=index,
        row=index // map_spec.cols,
        col=index % map_spec.cols,
        pattern_u16=pattern,
        pc=pc,
        bands=bands,
        ohp_header=ohp_header,
        ang_record=ang_record,
    )


def circular_mask(height: int, width: int, radius_fraction: float = 0.49) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * radius_fraction
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def hough_rho_to_pixels(rho_bin: float, header: OHPHeader, height: int, width: int) -> float:
    center_bin = header.circle_size / 2.0
    return (rho_bin - center_bin) * (min(height, width) / header.circle_size)


def coordinate_bounds(height: int, width: int, y_axis: str) -> tuple[float, float, float, float]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_min = -cx
    x_max = (width - 1) - cx
    if y_axis == "down":
        y_min = -cy
        y_max = (height - 1) - cy
    else:
        y_min = -((height - 1) - cy)
        y_max = cy
    return x_min, x_max, y_min, y_max


def coord_to_pixel(x: float, y: float, height: int, width: int, y_axis: str) -> tuple[float, float]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    col = x + cx
    row = y + cy if y_axis == "down" else cy - y
    return row, col


def line_segment_from_band(
    band: Band,
    header: OHPHeader,
    height: int,
    width: int,
    variant: LineVariant,
    band_index: int,
) -> LineSegment | None:
    rho_px = variant.rho_sign * hough_rho_to_pixels(band.rho_bin, header, height, width)
    normal_angle = band.theta_deg + (90.0 if variant.theta_is_line_angle else 0.0)
    theta = math.radians(normal_angle)
    c = math.cos(theta)
    s = math.sin(theta)
    x_min, x_max, y_min, y_max = coordinate_bounds(height, width, variant.y_axis)
    points: list[tuple[float, float]] = []
    if abs(s) > 1e-8:
        for x in (x_min, x_max):
            y = (rho_px - x * c) / s
            if y_min - 1e-5 <= y <= y_max + 1e-5:
                points.append((x, y))
    if abs(c) > 1e-8:
        for y in (y_min, y_max):
            x = (rho_px - y * s) / c
            if x_min - 1e-5 <= x <= x_max + 1e-5:
                points.append((x, y))
    unique: list[tuple[float, float]] = []
    for pt in points:
        if all((pt[0] - old[0]) ** 2 + (pt[1] - old[1]) ** 2 > 1e-4 for old in unique):
            unique.append(pt)
    if len(unique) < 2:
        return None
    best_pair = (unique[0], unique[1])
    best_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = (unique[i], unique[j])
    p0, p1 = best_pair
    row0, col0 = coord_to_pixel(p0[0], p0[1], height, width, variant.y_axis)
    row1, col1 = coord_to_pixel(p1[0], p1[1], height, width, variant.y_axis)
    return LineSegment(band_index=band_index, band=band, row0=row0, col0=col0, row1=row1, col1=col1)


def alpha_circle(height: int, width: int, radius_fraction: float) -> np.ndarray:
    return circular_mask(height, width, radius_fraction).astype(np.float32)


def normalize_for_display(pattern: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = pattern.astype(np.float32)
    lo, hi = np.percentile(image[mask], [0.5, 99.6])
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def rgba_gray(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image.shape, 4), dtype=np.float32)
    rgba[..., :3] = image[..., None]
    rgba[..., 3] = alpha
    return rgba


def render_band_rgba(
    bundle: PatternBundle,
    alpha: np.ndarray,
    line_color: tuple[float, float, float] = (1.0, 0.12, 0.04),
    width_scale: float = 1.45,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, width = bundle.pattern_u16.shape
    band_rgba = np.zeros((height, width, 4), dtype=np.float32)
    rows: list[dict[str, object]] = []

    intensities = np.array([band.intensity for band in bundle.bands], dtype=np.float32)
    intensity_min = float(intensities.min()) if intensities.size else 0.0
    intensity_span = float(intensities.max() - intensity_min + 1e-8) if intensities.size else 1.0

    for band_order, band in enumerate(bundle.bands, start=1):
        segment = line_segment_from_band(
            band=band,
            header=bundle.ohp_header,
            height=height,
            width=width,
            variant=PUBLICATION_VARIANT,
            band_index=band_order - 1,
        )
        if segment is None:
            continue

        strength = 0.35 + 0.65 * ((band.intensity - intensity_min) / intensity_span)
        thickness = int(np.clip(round(abs(band.width) * width_scale), 3, 14))
        line_canvas = np.zeros((height, width), dtype=np.float32)
        rr, cc, values = draw.line_aa(
            int(round(segment.row0)),
            int(round(segment.col0)),
            int(round(segment.row1)),
            int(round(segment.col1)),
        )
        ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        line_canvas[rr[ok], cc[ok]] = np.maximum(line_canvas[rr[ok], cc[ok]], values[ok] * float(strength))
        radius = max(1, thickness // 2)
        if radius > 1:
            line_canvas = morphology.dilation(line_canvas, morphology.disk(radius))
        line_canvas *= alpha
        color = np.array(line_color, dtype=np.float32)
        band_rgba[..., :3] = np.maximum(band_rgba[..., :3], line_canvas[..., None] * color)
        band_rgba[..., 3] = np.maximum(band_rgba[..., 3], np.clip(line_canvas, 0.0, 1.0))

        rows.append(
            {
                "band_order_by_h5_intensity": band_order,
                "rho_bin": band.rho_bin,
                "theta_deg": band.theta_deg,
                "band_width": band.width,
                "band_intensity": band.intensity,
                "line_variant": PUBLICATION_VARIANT.name,
                "line_row0": segment.row0,
                "line_col0": segment.col0,
                "line_row1": segment.row1,
                "line_col1": segment.col1,
                "render_thickness_px": thickness,
            }
        )

    return band_rgba, rows


def composite_pattern_bands(pattern_rgba: np.ndarray, band_rgba: np.ndarray) -> np.ndarray:
    out = pattern_rgba.copy()
    alpha = band_rgba[..., 3:4]
    out[..., :3] = out[..., :3] * (1.0 - alpha) + band_rgba[..., :3] * alpha
    out[..., 3] = np.maximum(out[..., 3], band_rgba[..., 3])
    return out


def save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(rgba, 0.0, 1.0))


def choose_quality_examples(h5_path: Path, map_keys: list[str], map_specs: dict, per_map: int) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    with h5py.File(h5_path, "r") as h5:
        for key in map_keys:
            spec = map_specs[key]
            data = h5[f"{spec.h5_group}/EBSD/ANG/DATA/DATA"]
            iq = data["IQ"][:].astype(np.float64)
            ci = data["CI"][:].astype(np.float64)
            valid = data["Valid"][:].astype(bool)
            candidates = np.flatnonzero(valid & np.isfinite(iq) & np.isfinite(ci) & (ci >= 0.80))
            if candidates.size < per_map:
                candidates = np.flatnonzero(valid & np.isfinite(iq) & np.isfinite(ci))
            score = iq[candidates] * (0.65 + 0.35 * np.clip(ci[candidates], 0.0, 1.0))
            order = np.argsort(score)[::-1]
            selected: list[int] = []
            for idx in candidates[order]:
                row = int(idx) // spec.cols
                col = int(idx) % spec.cols
                if all(abs(row - old // spec.cols) + abs(col - old % spec.cols) > 8 for old in selected):
                    selected.append(int(idx))
                if len(selected) >= per_map:
                    break
            examples.extend((key, index) for index in selected)
    return examples


def export_one(bundle: PatternBundle, out_dir: Path, radius_fraction: float) -> dict[str, object]:
    height, width = bundle.pattern_u16.shape
    alpha = alpha_circle(height, width, radius_fraction)
    display = normalize_for_display(bundle.pattern_u16, alpha > 0)
    pattern_rgba = rgba_gray(display, alpha)
    band_rgba, band_rows = render_band_rgba(bundle, alpha)
    overlay_rgba = composite_pattern_bands(pattern_rgba, band_rgba)

    stem = f"{bundle.map_spec.key}_idx{bundle.index:05d}"
    pattern_path = out_dir / f"{stem}_pattern_circle_transparent.png"
    bands_path = out_dir / f"{stem}_h5_hough_bands_transparent.png"
    overlay_path = out_dir / f"{stem}_pattern_h5_bands_transparent.png"
    band_csv_path = out_dir / f"{stem}_h5_hough_bands.csv"

    save_rgba(pattern_path, pattern_rgba)
    save_rgba(bands_path, band_rgba)
    save_rgba(overlay_path, overlay_rgba)

    ang = bundle.ang_record
    metadata = {
        "map_key": bundle.map_spec.key,
        "map_label": bundle.map_spec.label,
        "h5_group": bundle.map_spec.h5_group,
        "pattern_index": bundle.index,
        "row": bundle.row,
        "col": bundle.col,
        "height": height,
        "width": width,
        "iq": float(ang["IQ"]),
        "ci": float(ang["CI"]),
        "phase": int(ang["Phase"]),
        "fit": float(ang["Fit"]),
        "valid": bool(ang["Valid"]),
        "pcx_edax": bundle.pc[0],
        "pcy_edax": bundle.pc[1],
        "pcz_edax": bundle.pc[2],
        "ohp_circle_size": bundle.ohp_header.circle_size,
        "ohp_max_band_count": bundle.ohp_header.max_band_count,
        "ohp_theta_step_size": bundle.ohp_header.theta_step_size,
        "line_variant": PUBLICATION_VARIANT.name,
        "pattern_png": str(pattern_path),
        "bands_png": str(bands_path),
        "overlay_png": str(overlay_path),
        "bands_csv": str(band_csv_path),
    }

    with band_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(metadata.keys()) + list(band_rows[0].keys() if band_rows else [])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in band_rows:
            writer.writerow({**metadata, **row})

    return metadata | {"band_count": len(band_rows)}


def parse_example(text: str) -> tuple[str, int]:
    key, index = text.split(":", 1)
    return key, int(index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export circular transparent EBSD patterns and their matched H5/OHP Hough-peak Kikuchi bands."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-map", type=int, default=3)
    parser.add_argument("--map", dest="map_keys", action="append", default=["area1_high", "area2_high"])
    parser.add_argument("--example", action="append", type=parse_example, help="Explicit example as map:index.")
    parser.add_argument("--radius-fraction", type=float, default=0.492)
    args = parser.parse_args()

    map_specs = default_map_specs(args.data_dir)
    examples = args.example or choose_quality_examples(args.h5, args.map_keys, map_specs, args.per_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for map_key, index in examples:
        bundle = read_pattern_bundle(args.h5, map_specs[map_key], index)
        summary_rows.append(export_one(bundle, args.out_dir, args.radius_fraction))

    summary_path = args.out_dir / "publication_h5_kikuchi_bands_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} circular transparent examples to {args.out_dir}")
    print(f"Summary: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['map_key']} idx={row['pattern_index']} IQ={row['iq']:.1f} CI={row['ci']:.3f} "
            f"bands={row['band_count']}"
        )


if __name__ == "__main__":
    main()
