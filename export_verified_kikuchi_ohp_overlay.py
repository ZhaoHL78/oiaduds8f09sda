from __future__ import annotations

import argparse
import csv
import json
import struct
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import ndimage
from skimage import exposure

from export_publication_h5_kikuchi_bands import (
    LineVariant,
    PUBLICATION_VARIANT,
    alpha_circle,
    composite_pattern_bands,
    line_segment_from_band,
    normalize_for_display,
    read_bands,
    read_ohp_header,
    render_band_rgba,
    rgba_gray,
    save_rgba,
)


VARIANTS = [
    PUBLICATION_VARIANT,
    LineVariant("normal_theta_rho-_yup", theta_is_line_angle=False, rho_sign=-1.0, y_axis="up"),
    LineVariant("normal_theta_rho+_ydown", theta_is_line_angle=False, rho_sign=1.0, y_axis="down"),
    LineVariant("normal_theta_rho-_ydown", theta_is_line_angle=False, rho_sign=-1.0, y_axis="down"),
    LineVariant("line_theta_rho+_yup", theta_is_line_angle=True, rho_sign=1.0, y_axis="up"),
    LineVariant("line_theta_rho-_yup", theta_is_line_angle=True, rho_sign=-1.0, y_axis="up"),
    LineVariant("line_theta_rho+_ydown", theta_is_line_angle=True, rho_sign=1.0, y_axis="down"),
    LineVariant("line_theta_rho-_ydown", theta_is_line_angle=True, rho_sign=-1.0, y_axis="down"),
]


@dataclass(frozen=True)
class Up2Info:
    version: int
    width: int
    height: int
    header_bytes: int
    count: int


def read_up2_info(path: Path) -> Up2Info:
    with path.open("rb") as file:
        version, width, height, header_bytes = struct.unpack("<4I", file.read(16))
    count = (path.stat().st_size - header_bytes) // (width * height * 2)
    return Up2Info(version=version, width=width, height=height, header_bytes=header_bytes, count=count)


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, Up2Info]:
    info = read_up2_info(path)
    if not (0 <= index < info.count):
        raise IndexError(f"UP2 index {index} outside 0..{info.count - 1}: {path}")
    offset = info.header_bytes + index * info.width * info.height * 2
    with path.open("rb") as file:
        file.seek(offset)
        pattern = np.fromfile(file, dtype="<u2", count=info.width * info.height).reshape(info.height, info.width)
    return pattern, info


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_segment_response(image: np.ndarray, segment, half_width: int = 3) -> float:
    height, width = image.shape
    row0, col0, row1, col1 = segment.row0, segment.col0, segment.row1, segment.col1
    samples = max(2, int(np.hypot(row1 - row0, col1 - col0)))
    rows = np.linspace(row0, row1, samples)
    cols = np.linspace(col0, col1, samples)
    dcol = col1 - col0
    drow = row1 - row0
    norm = np.hypot(drow, dcol) + 1e-9
    ncol = -drow / norm
    nrow = dcol / norm
    values: list[np.ndarray] = []
    for offset in range(-half_width, half_width + 1):
        cc = np.round(cols + ncol * offset).astype(int)
        rr = np.round(rows + nrow * offset).astype(int)
        ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        if np.any(ok):
            values.append(image[rr[ok], cc[ok]])
    if not values:
        return float("nan")
    return float(np.mean(np.concatenate(values)))


def band_response_image(pattern: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = normalize_for_display(pattern, mask)
    background = ndimage.gaussian_filter(image.astype(np.float32), sigma=max(min(image.shape) / 22.0, 8.0), mode="nearest")
    high_pass = image.astype(np.float32) - background
    response = ndimage.gaussian_filter(np.abs(high_pass), sigma=0.75, mode="nearest")
    response[~mask] = 0.0
    return exposure.rescale_intensity(response, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)


def score_variant(map_group: h5py.Group, index: int, band_image: np.ndarray, mask: np.ndarray, variant: LineVariant) -> dict[str, Any]:
    height, width = band_image.shape
    header = read_ohp_header(map_group)
    bands = read_bands(map_group, index)
    line_values: list[float] = []
    weights: list[float] = []
    for band_order, band in enumerate(bands):
        segment = line_segment_from_band(band, header, height, width, variant, band_order)
        if segment is None:
            continue
        value = sample_segment_response(band_image, segment)
        if np.isfinite(value):
            line_values.append(value)
            weights.append(max(float(band.intensity), 1e-6))
    background = float(np.mean(band_image[mask])) if np.any(mask) else float(np.mean(band_image))
    weighted = float(np.average(line_values, weights=weights)) if line_values else float("nan")
    unweighted = float(np.mean(line_values)) if line_values else float("nan")
    return {
        "variant": variant.name,
        "band_count": len(line_values),
        "line_response": weighted,
        "line_response_unweighted": unweighted,
        "background_response": background,
        "response_minus_background": weighted - background if np.isfinite(weighted) else float("nan"),
    }


def candidate_up2_files(root: Path, target_count: int, target_shape: tuple[int, int]) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*.up2")):
        try:
            info = read_up2_info(path)
        except Exception:
            continue
        if info.count == target_count and (info.height, info.width) == target_shape:
            out.append(path)
    return out


def score_up2_candidate(path: Path, map_group: h5py.Group, index: int) -> dict[str, Any]:
    pattern, info = read_up2_pattern(path, index)
    mask = alpha_circle(info.height, info.width, 0.49).astype(bool)
    band_image = band_response_image(pattern, mask)
    row = score_variant(map_group, index, band_image, mask, PUBLICATION_VARIANT)
    row.update(
        {
            "up2_path": str(path),
            "up2_name": path.name,
            "up2_width": info.width,
            "up2_height": info.height,
            "up2_count": info.count,
        }
    )
    return row


def resolve_up2(path: Path | None, root: Path | None, map_group: h5py.Group, index: int, out_dir: Path) -> Path:
    if path is not None:
        return path
    if root is None:
        raise ValueError("Provide either --up2 or --up2-root")
    target_count = int(map_group["EBSD/OHP/DATA/DATA"].shape[0])
    # Use the first valid UP2 shape in the root as a filter target, then still score all same-count files.
    candidates = []
    for candidate in sorted(root.rglob("*.up2")):
        try:
            info = read_up2_info(candidate)
        except Exception:
            continue
        if info.count == target_count:
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No UP2 candidates with count={target_count} under {root}")
    rows = [score_up2_candidate(candidate, map_group, index) for candidate in candidates]
    rows.sort(key=lambda item: float(item["response_minus_background"]), reverse=True)
    write_csv(out_dir / "up2_candidate_ohp_scores.csv", rows)
    return Path(rows[0]["up2_path"])


def render_outputs(h5_path: Path, h5_group: str, up2_path: Path, index: int, out_dir: Path, row: int | None, col: int | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern, info = read_up2_pattern(up2_path, index)
    height, width = pattern.shape
    alpha = alpha_circle(height, width, 0.49)
    mask = alpha.astype(bool)
    pattern_display = normalize_for_display(pattern, mask)
    pattern_rgba = rgba_gray(pattern_display, alpha)
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        header = read_ohp_header(group)
        bands = read_bands(group, index)
        band_image = band_response_image(pattern, mask)
        variant_rows = [score_variant(group, index, band_image, mask, variant) for variant in VARIANTS]
    variant_rows.sort(key=lambda item: float(item["response_minus_background"]), reverse=True)
    write_csv(out_dir / "ohp_variant_scores.csv", variant_rows)

    bundle = SimpleNamespace(pattern_u16=pattern, bands=bands, ohp_header=header)
    band_rgba, band_rows = render_band_rgba(bundle, alpha)
    overlay_rgba = composite_pattern_bands(pattern_rgba, band_rgba)
    stem = f"idx{index:06d}"
    paths = {
        "pattern_png": out_dir / f"{stem}_pattern_transparent.png",
        "bands_png": out_dir / f"{stem}_h5_ohp_bands_transparent.png",
        "overlay_png": out_dir / f"{stem}_pattern_h5_ohp_overlay_transparent.png",
        "band_csv": out_dir / f"{stem}_h5_ohp_bands.csv",
    }
    save_rgba(paths["pattern_png"], pattern_rgba)
    save_rgba(paths["bands_png"], band_rgba)
    save_rgba(paths["overlay_png"], overlay_rgba)
    write_csv(paths["band_csv"], band_rows)
    metadata = {
        "h5_path": str(h5_path),
        "h5_group": h5_group,
        "up2_path": str(up2_path),
        "up2_info": info.__dict__,
        "index": index,
        "row": row,
        "col": col,
        "fixed_variant": PUBLICATION_VARIANT.name,
        "best_scored_variant": variant_rows[0]["variant"] if variant_rows else None,
        "fixed_variant_score": next((item for item in variant_rows if item["variant"] == PUBLICATION_VARIANT.name), None),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    (out_dir / "verified_kikuchi_ohp_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export verified Kikuchi/OHP overlay with fixed EDAX OHP convention and UP2 sanity check.")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--h5-group", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--row", type=int)
    parser.add_argument("--col", type=int)
    parser.add_argument("--up2", type=Path)
    parser.add_argument("--up2-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "verified_kikuchi_ohp_overlay")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as h5:
        group = h5[args.h5_group]
        up2 = resolve_up2(args.up2, args.up2_root, group, args.index, args.out_dir)
    metadata = render_outputs(args.h5, args.h5_group, up2, args.index, args.out_dir, args.row, args.col)
    print(json.dumps({"status": "completed", "up2": metadata["up2_path"], "overlay": metadata["outputs"]["overlay_png"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
