"""Align four Pt-1 in-plane EBSD SEM images and find their largest common circular ROI."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage import exposure
from skimage.registration import phase_cross_correlation


DEFAULT_H5_PATH = Path(r"E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5")
DEFAULT_UP2_ROOT = Path(r"E:\ZHL\EBSD-RAW\20251209Pt")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt1_inplane_sem_common_circle"


@dataclass(frozen=True)
class InplaneMap:
    label: str
    area: str
    map_name: str
    inplane_angle_deg: float
    expected_up2_name: str

    @property
    def h5_path(self) -> str:
        return f"20251209/Pt-1/{self.area}/{self.map_name}"


MAP_SEQUENCE = [
    InplaneMap(
        label="Pt-1 90 deg",
        area="Area 90degree",
        map_name="OIM Map 1",
        inplane_angle_deg=90.0,
        expected_up2_name="20251209_Pt-1_Area 5_OIM Map 1.up2",
    ),
    InplaneMap(
        label="Pt-1 180 deg",
        area="Area 180degree",
        map_name="OIM Map 1",
        inplane_angle_deg=180.0,
        expected_up2_name="20251209_Pt-1_Area 6_OIM Map 1.up2",
    ),
    InplaneMap(
        label="Pt-1 270 deg",
        area="Area 270degree",
        map_name="OIM Map 1",
        inplane_angle_deg=270.0,
        expected_up2_name="20251209_Pt-1_Area 7_OIM Map 1.up2",
    ),
    InplaneMap(
        label="Pt-1 360 deg",
        area="Area 360degree",
        map_name="OIM Map 1",
        inplane_angle_deg=360.0,
        expected_up2_name="20251209_Pt-1_Area 8_OIM Map 1.up2",
    ),
]


def normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def edge_image(image: np.ndarray) -> np.ndarray:
    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    finite = mag[np.isfinite(mag) & (mag > 0)]
    if finite.size:
        mag = mag / max(np.percentile(finite, 99.0), 1e-6)
    return np.clip(mag, 0.0, 1.0).astype(np.float32)


def read_up2_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError(f"{path} is too small to contain a UP2 header")
    version, width, height, header_bytes = struct.unpack("<4I", header)
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    payload_bytes = path.stat().st_size - header_bytes
    if payload_bytes < 0 or pattern_bytes <= 0 or payload_bytes % pattern_bytes:
        raise ValueError(f"Inconsistent UP2 header in {path}")
    return {
        "path": str(path),
        "version": version,
        "width": width,
        "height": height,
        "header_bytes": header_bytes,
        "count": payload_bytes // pattern_bytes,
        "pattern_bytes": pattern_bytes,
        "file_bytes": path.stat().st_size,
    }


def read_up2_pattern(path: Path, index: int) -> tuple[np.ndarray, dict[str, Any]]:
    info = read_up2_info(path)
    if index < 0 or index >= int(info["count"]):
        raise IndexError(f"{path.name}: index={index} outside count={info['count']}")
    offset = int(info["header_bytes"]) + index * int(info["pattern_bytes"])
    with path.open("rb") as stream:
        stream.seek(offset)
        pattern = np.fromfile(stream, dtype="<u2", count=int(info["width"]) * int(info["height"]))
    return pattern.reshape(int(info["height"]), int(info["width"])), info


def choose_max_iq_index(map_group: h5py.Group) -> tuple[int, dict[str, Any]]:
    data = map_group["EBSD/ANG/DATA/DATA"]
    iq = data["IQ"][:].astype(np.float64)
    valid = data["Valid"][:].astype(bool)
    score = np.where(valid & np.isfinite(iq), iq, -np.inf)
    index = int(np.argmax(score)) if np.isfinite(score).any() else int(np.nanargmax(iq))
    record = data[index]
    return index, {
        "IQ": float(record["IQ"]),
        "CI": float(record["CI"]),
        "Phase": int(record["Phase"]),
        "Valid": int(record["Valid"]),
    }


def affine_to_canvas(image_shape: tuple[int, int], canvas_shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    h, w = image_shape
    canvas_h, canvas_w = canvas_shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0).astype(np.float32)
    matrix[0, 2] += canvas_w / 2.0 - w / 2.0
    matrix[1, 2] += canvas_h / 2.0 - h / 2.0
    return matrix


def as_3x3(matrix_2x3: np.ndarray) -> np.ndarray:
    return np.vstack([matrix_2x3, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def warp_to_canvas(image: np.ndarray, matrix: np.ndarray, canvas_shape: tuple[int, int], interpolation: int) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    return cv2.warpAffine(image, matrix, (canvas_w, canvas_h), flags=interpolation, borderValue=0)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    pixels = mask > 0
    if int(np.count_nonzero(pixels)) < 1000:
        return float("nan")
    av = a[pixels].astype(np.float64)
    bv = b[pixels].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = math.sqrt(float(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(av * bv) / denom)


def find_largest_circle(mask: np.ndarray) -> tuple[tuple[float, float], float, np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(distance)
    center_x, center_y = max_loc
    return (float(center_x), float(center_y)), float(max_val), distance


def draw_circle(ax: plt.Axes, center_xy: tuple[float, float], radius: float, color: str = "red", lw: float = 1.8) -> None:
    circle = plt.Circle(center_xy, radius, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter([center_xy[0]], [center_xy[1]], s=10, c=color)


def save_raw_sem_with_projected_circle(records: list[dict[str, Any]], center_xy: tuple[float, float], radius: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=190)
    for ax, record in zip(axes.ravel(), records):
        image = record["sem"]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        inv = np.linalg.inv(record["total_affine_3x3"])
        pt = inv @ np.array([center_xy[0], center_xy[1], 1.0], dtype=np.float32)
        draw_circle(ax, (float(pt[0]), float(pt[1])), radius, color="#ff2020", lw=1.6)
        ax.set_title(f"{record['label']} raw SEM\ncommon circle projected back", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_input_sem(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=190)
    for ax, record in zip(axes, records):
        ax.imshow(record["sem"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{record['label']}\n{record['up2_name']}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_aligned_sem(records: list[dict[str, Any]], center_xy: tuple[float, float], radius: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=190)
    for ax, record in zip(axes.ravel(), records):
        ax.imshow(record["aligned_sem"], cmap="gray", vmin=0, vmax=1)
        ax.contour(record["aligned_mask"], levels=[0.5], colors=["#00ccff"], linewidths=0.5)
        draw_circle(ax, center_xy, radius, color="#ff2020", lw=1.6)
        ax.set_title(
            f"{record['label']} aligned\nrot={record['applied_rotation_deg']:.0f} deg, "
            f"shift=({record['phase_shift_y']:.1f}, {record['phase_shift_x']:.1f})",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_overlay(records: list[dict[str, Any]], common_mask: np.ndarray, center_xy: tuple[float, float], radius: float, path: Path) -> None:
    stack = np.stack([record["aligned_sem"] for record in records], axis=0)
    average = np.mean(stack, axis=0)
    spread = np.std(stack, axis=0)
    rgb = np.zeros((*average.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(average + 0.8 * spread, 0, 1)
    rgb[..., 1] = average
    rgb[..., 2] = np.clip(average - 0.4 * spread, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    ax.imshow(rgb)
    ax.contour(common_mask, levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
    draw_circle(ax, center_xy, radius, color="#ff2020", lw=2.0)
    ax.set_title("Aligned SEM average/spread with common mask and largest circle", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_common_mask(common_mask: np.ndarray, distance: np.ndarray, center_xy: tuple[float, float], radius: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=190)
    axes[0].imshow(common_mask, cmap="gray")
    draw_circle(axes[0], center_xy, radius, color="#ff2020", lw=1.8)
    axes[0].set_title("Common valid mask")
    axes[0].axis("off")
    im = axes[1].imshow(distance, cmap="magma")
    draw_circle(axes[1], center_xy, radius, color="#00e5ff", lw=1.6)
    axes[1].set_title(f"Distance transform, radius={radius:.1f} px")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_kikuchi_examples(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), dpi=190)
    for ax, record in zip(axes, records):
        ax.imshow(record["kikuchi"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{record['label']}\nidx={record['selected_index']}, IQ={record['selected_iq']:.0f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_records(h5_path: Path, up2_root: Path, canvas_shape: tuple[int, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_angle = MAP_SEQUENCE[0].inplane_angle_deg
    with h5py.File(h5_path, "r") as h5:
        for spec in MAP_SEQUENCE:
            map_group = h5[spec.h5_path]
            sem = normalize_gray(np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
            h5_count = int(map_group["EBSD/ANG/DATA/DATA"].shape[0])
            selected_index, selected = choose_max_iq_index(map_group)
            up2_path = up2_root / spec.expected_up2_name
            pattern, up2_info = read_up2_pattern(up2_path, selected_index)
            kikuchi = normalize_gray(pattern)
            applied_rotation = -(spec.inplane_angle_deg - reference_angle)
            matrix = affine_to_canvas(sem.shape, canvas_shape, applied_rotation)
            warped_sem = warp_to_canvas(sem, matrix, canvas_shape, cv2.INTER_LINEAR)
            warped_mask = warp_to_canvas(np.ones_like(sem, dtype=np.uint8), matrix, canvas_shape, cv2.INTER_NEAREST) > 0
            records.append(
                {
                    "label": spec.label,
                    "area": spec.area,
                    "map_name": spec.map_name,
                    "h5_path": spec.h5_path,
                    "h5_count": h5_count,
                    "up2_name": spec.expected_up2_name,
                    "up2_path": str(up2_path),
                    "up2_count": int(up2_info["count"]),
                    "up2_resolution": f"{up2_info['width']}x{up2_info['height']}",
                    "selected_index": selected_index,
                    "selected_iq": float(selected["IQ"]),
                    "selected_ci": float(selected["CI"]),
                    "inplane_angle_deg": spec.inplane_angle_deg,
                    "applied_rotation_deg": applied_rotation,
                    "sem": sem,
                    "kikuchi": kikuchi,
                    "initial_sem": warped_sem,
                    "initial_mask": warped_mask,
                    "initial_affine_3x3": as_3x3(matrix),
                }
            )
    return records


def apply_phase_translation(records: list[dict[str, Any]], use_phase_shift: bool) -> None:
    reference = edge_image(records[0]["initial_sem"])
    reference_mask = records[0]["initial_mask"]
    for idx, record in enumerate(records):
        if idx == 0 or not use_phase_shift:
            shift_yx = np.array([0.0, 0.0], dtype=np.float32)
            shifted_sem = record["initial_sem"]
            shifted_mask = record["initial_mask"]
        else:
            moving_edge = edge_image(record["initial_sem"])
            shift_yx, _error, _phase = phase_cross_correlation(
                reference * reference_mask.astype(np.float32),
                moving_edge * record["initial_mask"].astype(np.float32),
                upsample_factor=10,
            )
            # SEM contrast changes strongly with in-plane rotation.  Keep the
            # fine translation conservative so the physical 90-degree geometry
            # remains the dominant alignment.
            shift_yx = np.clip(np.asarray(shift_yx, dtype=np.float32), -80.0, 80.0)
            shifted_sem = ndi_shift(record["initial_sem"], shift=shift_yx, order=1, mode="constant", cval=0.0)
            shifted_mask = (
                ndi_shift(record["initial_mask"].astype(np.float32), shift=shift_yx, order=0, mode="constant", cval=0.0)
                > 0.5
            )

        shift_matrix = np.array(
            [[1.0, 0.0, float(shift_yx[1])], [0.0, 1.0, float(shift_yx[0])], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        overlap = reference_mask & shifted_mask
        record["phase_shift_y"] = float(shift_yx[0])
        record["phase_shift_x"] = float(shift_yx[1])
        record["edge_ncc_to_reference"] = ncc(reference, edge_image(shifted_sem), overlap)
        record["aligned_sem"] = shifted_sem
        record["aligned_mask"] = shifted_mask
        record["total_affine_3x3"] = shift_matrix @ record["initial_affine_3x3"]


def export_alignment(h5_path: Path, up2_root: Path, output_dir: Path, canvas_shape: tuple[int, int], use_phase_shift: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(h5_path, up2_root, canvas_shape)
    apply_phase_translation(records, use_phase_shift)
    common_mask = np.logical_and.reduce([record["aligned_mask"] for record in records])
    circle_center, circle_radius, distance = find_largest_circle(common_mask)

    save_input_sem(records, output_dir / "01_input_pt1_sem_sequence.png")
    save_kikuchi_examples(records, output_dir / "02_corresponding_up2_kikuchi_examples.png")
    save_aligned_sem(records, circle_center, circle_radius, output_dir / "03_aligned_sem_with_common_circle.png")
    save_overlay(records, common_mask, circle_center, circle_radius, output_dir / "04_aligned_sem_overlay_common_circle.png")
    save_common_mask(common_mask, distance, circle_center, circle_radius, output_dir / "05_common_mask_largest_circle.png")
    save_raw_sem_with_projected_circle(records, circle_center, circle_radius, output_dir / "06_common_circle_projected_to_raw_sem.png")

    transform_rows: list[dict[str, Any]] = []
    for record in records:
        transform_rows.append(
            {
                "label": record["label"],
                "h5_path": record["h5_path"],
                "h5_count": record["h5_count"],
                "up2_name": record["up2_name"],
                "up2_count": record["up2_count"],
                "up2_resolution": record["up2_resolution"],
                "selected_index": record["selected_index"],
                "selected_iq": record["selected_iq"],
                "selected_ci": record["selected_ci"],
                "inplane_angle_deg": record["inplane_angle_deg"],
                "applied_rotation_deg": record["applied_rotation_deg"],
                "phase_shift_y_px": record["phase_shift_y"],
                "phase_shift_x_px": record["phase_shift_x"],
                "edge_ncc_to_reference": record["edge_ncc_to_reference"],
                "common_circle_center_x_canvas": circle_center[0],
                "common_circle_center_y_canvas": circle_center[1],
                "common_circle_radius_px": circle_radius,
                "affine_original_to_canvas": np.array2string(record["total_affine_3x3"], precision=6, separator=","),
            }
        )
    write_csv(output_dir / "pt1_inplane_sem_alignment_transforms.csv", transform_rows)
    (output_dir / "pt1_inplane_sem_alignment_summary.md").write_text(
        "\n".join(
            [
                "# Pt-1 In-Plane SEM Alignment",
                "",
                f"- H5: `{h5_path}`",
                f"- UP2 root: `{up2_root}`",
                "- EBSD maps: Area 90degree, 180degree, 270degree, 360degree.",
                "- Alignment: counter-rotate each SEM to the first map's in-plane frame, then apply conservative phase-correlation translation on SEM edge images.",
                f"- Largest common circular ROI center on alignment canvas: ({circle_center[0]:.2f}, {circle_center[1]:.2f}) px.",
                f"- Largest common circular ROI radius: {circle_radius:.2f} px.",
                "",
                "The circular ROI is computed from the intersection of the four transformed SEM valid masks.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Align four Pt-1 in-plane SEM images and export the common circle.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--up2-root", type=Path, default=DEFAULT_UP2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canvas-size", type=int, default=760)
    parser.add_argument("--no-phase-shift", action="store_true", help="Disable conservative translation refinement.")
    args = parser.parse_args()

    export_alignment(
        h5_path=args.h5,
        up2_root=args.up2_root,
        output_dir=args.output_dir,
        canvas_shape=(args.canvas_size, args.canvas_size),
        use_phase_shift=not args.no_phase_shift,
    )
    print(f"Saved Pt-1 in-plane SEM alignment to {args.output_dir}")
    print(f"Summary: {args.output_dir / 'pt1_inplane_sem_alignment_summary.md'}")
    print(f"Transforms: {args.output_dir / 'pt1_inplane_sem_alignment_transforms.csv'}")


if __name__ == "__main__":
    main()
