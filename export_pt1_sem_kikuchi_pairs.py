"""Export one SEM image and one corresponding raw Kikuchi pattern per Pt-1 EBSD map."""

from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


DEFAULT_DATA_DIR = Path(r"D:\EBSD-data\Pt-1")
DEFAULT_H5_PATH = DEFAULT_DATA_DIR / "20251209Pt.edaxh5"
DEFAULT_CLASSIFICATION_CSV = Path("outputs") / "pt1_classification" / "pt1_match_classification.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt1_sem_kikuchi_pairs"


@dataclass(frozen=True)
class Up2Info:
    version: int
    width: int
    height: int
    header_bytes: int
    count: int

    @property
    def pattern_bytes(self) -> int:
        return self.width * self.height * np.dtype("<u2").itemsize


def read_up2_info(path: Path) -> Up2Info:
    with path.open("rb") as stream:
        version, width, height, header_bytes = struct.unpack("<4I", stream.read(16))
    payload_bytes = path.stat().st_size - header_bytes
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    if payload_bytes < 0 or payload_bytes % pattern_bytes:
        raise ValueError(f"Inconsistent UP2 header for {path}")
    return Up2Info(
        version=version,
        width=width,
        height=height,
        header_bytes=header_bytes,
        count=payload_bytes // pattern_bytes,
    )


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, Up2Info]:
    info = read_up2_info(path)
    if index < 0 or index >= info.count:
        raise IndexError(f"Pattern index {index} is out of range for {path.name}: count={info.count}")
    offset = info.header_bytes + index * info.pattern_bytes
    with path.open("rb") as stream:
        stream.seek(offset)
        pattern = np.fromfile(stream, dtype="<u2", count=info.width * info.height)
    return pattern.reshape(info.height, info.width), info


def normalize_gray(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def circular_rgba(gray: np.ndarray, radius_fraction: float = 0.49) -> np.ndarray:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * radius_fraction
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = mask.astype(np.float32)
    return rgba


def safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("__", "_")
    )


def load_matched_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("classification") == "matched_h5_up2"]
    if not rows:
        raise ValueError(f"No matched_h5_up2 rows found in {path}")
    return rows


def choose_representative_index(map_group: h5py.Group) -> tuple[int, dict[str, Any]]:
    data = map_group["EBSD/ANG/DATA/DATA"]
    iq = data["IQ"][:].astype(np.float64)
    valid = data["Valid"][:].astype(bool)
    valid_iq = np.where(valid & np.isfinite(iq), iq, -np.inf)
    if np.isfinite(valid_iq).any():
        index = int(np.argmax(valid_iq))
    else:
        index = int(np.nanargmax(iq))
    record = data[index]
    meta: dict[str, Any] = {}
    for key in ["IQ", "CI", "Phase", "Fit", "SEM Signal", "Valid"]:
        value = record[key]
        if np.issubdtype(np.asarray(value).dtype, np.integer):
            meta[key] = int(value)
        else:
            meta[key] = float(value)
    return index, meta


def scan_to_sem_xy(index: int, nrows: int, ncols: int, sem_shape: tuple[int, int]) -> tuple[float, float, int, int]:
    row = index // ncols
    col = index % ncols
    sem_h, sem_w = sem_shape
    x = (col + 0.5) / max(ncols, 1) * sem_w
    y = (row + 0.5) / max(nrows, 1) * sem_h
    return x, y, row, col


def save_sem_marked(sem_gray: np.ndarray, x: float, y: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=220)
    ax.imshow(sem_gray, cmap="gray", vmin=0, vmax=1)
    ax.scatter([x], [y], s=95, facecolors="none", edgecolors="#ff2020", linewidths=1.8)
    ax.scatter([x], [y], s=12, c="#ff2020")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def save_pair_figure(
    sem_gray: np.ndarray,
    pattern_gray: np.ndarray,
    x: float,
    y: float,
    title: str,
    details: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), dpi=220)
    axes[0].imshow(sem_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter([x], [y], s=105, facecolors="none", edgecolors="#ff2020", linewidths=1.8)
    axes[0].scatter([x], [y], s=12, c="#ff2020")
    axes[0].set_title("SEM with selected EBSD point", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(pattern_gray, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Raw UP2 Kikuchi at the selected index", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"{title}\n{details}", fontsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def export_pair(row: dict[str, str], h5: h5py.File, output_dir: Path) -> dict[str, Any]:
    h5_path = row["h5_path"]
    map_group = h5[h5_path]
    nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
    ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])

    index, meta = choose_representative_index(map_group)
    up2_path = Path(row["up2_path"])
    pattern, up2_info = read_up2_pattern(up2_path, index)
    sem = np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:])

    sem_gray = normalize_gray(sem)
    pattern_gray = normalize_gray(pattern)
    pattern_rgba = circular_rgba(pattern_gray)
    x, y, scan_row, scan_col = scan_to_sem_xy(index, nrows, ncols, sem_gray.shape)

    key = safe_name(f"{row['h5_area']}_{row['h5_map']}_{up2_path.stem}")
    out = output_dir / key
    out.mkdir(parents=True, exist_ok=True)

    sem_path = out / f"{key}_sem_marked.png"
    kikuchi_path = out / f"{key}_kikuchi_raw_display.png"
    kikuchi_circular_path = out / f"{key}_kikuchi_circular_transparent.png"
    pair_path = out / f"{key}_sem_kikuchi_pair.png"

    title = f"{row['h5_area']} / {row['h5_map']}"
    details = (
        f"idx={index}, row={scan_row}, col={scan_col}, "
        f"IQ={meta['IQ']:.1f}, CI={meta['CI']:.3f}, phase={meta['Phase']}, "
        f"UP2={up2_info.width}x{up2_info.height}"
    )

    save_sem_marked(sem_gray, x, y, title, sem_path)
    plt.imsave(kikuchi_path, pattern_gray, cmap="gray", vmin=0, vmax=1)
    plt.imsave(kikuchi_circular_path, pattern_rgba)
    save_pair_figure(sem_gray, pattern_gray, x, y, title, details, pair_path)

    return {
        "h5_area": row["h5_area"],
        "h5_map": row["h5_map"],
        "h5_path": h5_path,
        "up2_file": up2_path.name,
        "up2_path": str(up2_path),
        "selected_rule": "maximum valid IQ in ANG/DATA",
        "selected_index": index,
        "scan_row": scan_row,
        "scan_col": scan_col,
        "sem_x_px": x,
        "sem_y_px": y,
        "iq": meta["IQ"],
        "ci": meta["CI"],
        "phase": meta["Phase"],
        "fit": meta["Fit"],
        "valid": meta["Valid"],
        "up2_width": up2_info.width,
        "up2_height": up2_info.height,
        "sem_marked_png": str(sem_path),
        "kikuchi_raw_display_png": str(kikuchi_path),
        "kikuchi_circular_transparent_png": str(kikuchi_circular_path),
        "pair_png": str(pair_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    images = []
    for row in rows:
        image = plt.imread(row["pair_png"])
        images.append((row, image))

    fig, axes = plt.subplots(len(images), 1, figsize=(9.5, 4.2 * len(images)), dpi=180)
    if len(images) == 1:
        axes = [axes]
    for ax, (row, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(f"{row['h5_area']} / {row['h5_map']} | idx={row['selected_index']}", fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Pt-1 SEM images and one corresponding Kikuchi pattern.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    matched_rows = load_matched_rows(args.classification_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    with h5py.File(args.h5, "r") as h5:
        for row in matched_rows:
            exported.append(export_pair(row, h5, args.output_dir))

    summary_path = args.output_dir / "pt1_sem_kikuchi_pairs_summary.csv"
    write_csv(summary_path, exported)
    make_contact_sheet(exported, args.output_dir / "pt1_sem_kikuchi_pairs_contact_sheet.png")

    print(f"Exported {len(exported)} SEM/Kikuchi pairs to {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Contact sheet: {args.output_dir / 'pt1_sem_kikuchi_pairs_contact_sheet.png'}")


if __name__ == "__main__":
    main()
