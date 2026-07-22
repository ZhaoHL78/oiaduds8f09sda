from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import exposure

from export_h5_ipf_bse_maps import cubic_ipf_z_colors
from export_publication_h5_kikuchi_bands import (
    PUBLICATION_VARIANT,
    circular_mask,
    line_segment_from_band,
    normalize_for_display,
    read_bands,
    read_ohp_header,
    read_up2_info,
    read_up2_pattern,
)
from pt_highres_30deg_lightglue_calibration import (
    DEFAULT_H5,
    DEFAULT_OUTPUT_DIR as DEFAULT_CALIBRATION_DIR,
    DEFAULT_UP2_ROOT,
    build_map_specs,
    normalize_gray,
)
from single_kikuchi_pc_finetune import build_preprocessed_images


DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_highres_data_overview"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ffloat(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fint(row: dict[str, str], key: str, default: int = -1) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def selected_sem_xy(row: dict[str, str], sem_shape: tuple[int, int], nrows: int, ncols: int) -> tuple[float, float]:
    x = ffloat(row, "selected_sem_x")
    y = ffloat(row, "selected_sem_y")
    if np.isfinite(x) and np.isfinite(y):
        return x, y
    index = fint(row, "selected_index")
    scan_row = index // ncols
    scan_col = index % ncols
    sem_h, sem_w = sem_shape
    return (scan_col + 0.5) / max(ncols, 1) * sem_w, (scan_row + 0.5) / max(nrows, 1) * sem_h


def image_for_panel(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return np.clip(image, 0.0, 1.0)
    out = np.ones((*image.shape, 3), dtype=np.float32)
    clipped = np.clip(image, 0.0, 1.0)
    out[mask] = clipped[mask, None]
    return out


def read_ipf_map(map_group: h5py.Group) -> np.ndarray:
    data = map_group["EBSD/ANG/DATA/DATA"][:]
    nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
    ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
    valid = data["Valid"].astype(bool) & (data["Phase"] == 1)
    return cubic_ipf_z_colors(data["Orientations"], valid, data["CI"]).reshape(nrows, ncols, 3)


def draw_selected_point(ax, x: float, y: float, color: str = "red", size: float = 30.0) -> None:
    ax.scatter([x], [y], s=size, c=color, marker="o", edgecolors="white", linewidths=0.8, zorder=5)


def draw_pc_markers(ax, row: dict[str, str], width: int, height: int) -> None:
    pc_x = ffloat(row, "pc_original_pixel_x")
    pc_y = ffloat(row, "pc_original_pixel_y")
    rpc_x = ffloat(row, "pc_refined_pixel_x")
    rpc_y = ffloat(row, "pc_refined_pixel_y")
    if np.isfinite(pc_x) and np.isfinite(pc_y):
        ax.scatter([pc_x], [pc_y], s=24, c="white", marker="+", linewidths=1.4, zorder=7)
    if np.isfinite(rpc_x) and np.isfinite(rpc_y):
        ax.scatter([rpc_x], [rpc_y], s=28, facecolors="none", edgecolors="cyan", linewidths=1.2, zorder=7)


def draw_ohp_bands(ax, map_group: h5py.Group, index: int, shape: tuple[int, int], color: str = "magenta") -> int:
    height, width = shape
    header = read_ohp_header(map_group)
    bands = read_bands(map_group, index)
    clip_circle = Circle(((width - 1) / 2.0, (height - 1) / 2.0), 0.498 * min(height, width), transform=ax.transData)
    max_intensity = max((band.intensity for band in bands), default=1.0)
    count = 0
    for band_order, band in enumerate(bands, start=1):
        segment = line_segment_from_band(
            band=band,
            header=header,
            height=height,
            width=width,
            variant=PUBLICATION_VARIANT,
            band_index=band_order - 1,
        )
        if segment is None:
            continue
        alpha = 0.35 + 0.55 * min(max(band.intensity / max_intensity, 0.0), 1.0)
        line = ax.plot(
            [segment.col0, segment.col1],
            [segment.row0, segment.row1],
            color=color,
            lw=1.25,
            alpha=alpha,
            zorder=6,
        )[0]
        line.set_clip_path(clip_circle)
        count += 1
    return count


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


def score_ohp_alignment(map_group: h5py.Group, index: int, band_image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    height, width = band_image.shape
    header = read_ohp_header(map_group)
    bands = read_bands(map_group, index)
    line_values: list[float] = []
    weights: list[float] = []
    for band_order, band in enumerate(bands):
        segment = line_segment_from_band(
            band=band,
            header=header,
            height=height,
            width=width,
            variant=PUBLICATION_VARIANT,
            band_index=band_order,
        )
        if segment is None:
            continue
        value = sample_segment_response(band_image, segment)
        if np.isfinite(value):
            line_values.append(value)
            weights.append(max(float(band.intensity), 1e-6))
    background = float(np.mean(band_image[mask])) if np.any(mask) else float(np.mean(band_image))
    if line_values:
        weighted = float(np.average(line_values, weights=weights))
        unweighted = float(np.mean(line_values))
    else:
        weighted = float("nan")
        unweighted = float("nan")
    return {
        "ohp_band_count": float(len(line_values)),
        "ohp_line_response": weighted,
        "ohp_line_response_unweighted": unweighted,
        "ohp_background_response": background,
        "ohp_response_minus_background": weighted - background if np.isfinite(weighted) else float("nan"),
    }


def save_individual_views(
    output_dir: Path,
    angle: int,
    sem: np.ndarray,
    ipf: np.ndarray,
    pattern_display: np.ndarray,
    enhanced: np.ndarray,
    band_display: np.ndarray,
    mask: np.ndarray,
    row: dict[str, str],
    selected_xy: tuple[float, float],
    selected_row_col: tuple[int, int],
    map_group: h5py.Group,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sem_x, sem_y = selected_xy
    scan_row, scan_col = selected_row_col
    index = fint(row, "selected_index")

    fig, ax = plt.subplots(figsize=(5, 4), dpi=220)
    ax.imshow(sem, cmap="gray", vmin=0, vmax=1)
    draw_selected_point(ax, sem_x, sem_y)
    ax.set_title(f"{angle:03d} deg SEM, idx={index}")
    ax.axis("off")
    fig.savefig(output_dir / f"angle_{angle:03d}_sem_selected.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=220)
    ax.imshow(ipf)
    draw_selected_point(ax, scan_col, scan_row, color="black", size=26.0)
    ax.set_title(f"{angle:03d} deg H5 IPF-Z, row={scan_row}, col={scan_col}")
    ax.axis("off")
    fig.savefig(output_dir / f"angle_{angle:03d}_ipf_selected.png", bbox_inches="tight")
    plt.close(fig)

    for name, image in (("raw", pattern_display), ("enhanced", enhanced), ("band_response", band_display)):
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=240)
        ax.imshow(image_for_panel(image, mask), vmin=0, vmax=1)
        draw_pc_markers(ax, row, image.shape[1], image.shape[0])
        ax.set_title(f"{angle:03d} deg Kikuchi {name}")
        ax.axis("off")
        fig.savefig(output_dir / f"angle_{angle:03d}_kikuchi_{name}.png", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=240)
    ax.set_facecolor("black")
    ax.imshow(pattern_display, cmap="gray", vmin=0, vmax=1, alpha=mask.astype(np.float32))
    band_count = draw_ohp_bands(ax, map_group, index, pattern_display.shape)
    draw_pc_markers(ax, row, pattern_display.shape[1], pattern_display.shape[0])
    ax.set_title(f"{angle:03d} deg OHP bands, n={band_count}")
    ax.axis("off")
    fig.savefig(output_dir / f"angle_{angle:03d}_kikuchi_ohp_bands.png", bbox_inches="tight")
    plt.close(fig)


def build_records(
    h5_path: Path,
    up2_root: Path,
    calibration_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    summary_path = calibration_dir / "pt_highres_30deg_spherical_calibration_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing high-res calibration summary: {summary_path}")
    summary_rows = sorted(read_csv_rows(summary_path), key=lambda item: fint(item, "angle_deg"))
    row_by_angle = {fint(row, "angle_deg"): row for row in summary_rows}
    specs = build_map_specs()
    spec_by_up2 = {spec.up2_name: spec for spec in specs}

    raw_inventory: list[dict[str, Any]] = []
    for up2_path in sorted(up2_root.glob("*.up2")):
        info = read_up2_info(up2_path)
        spec = spec_by_up2.get(up2_path.name)
        raw_inventory.append(
            {
                "up2_name": up2_path.name,
                "is_12_map_rotation_series": bool(spec),
                "angle_deg": spec.angle_deg if spec else "",
                "h5_group": spec.h5_group if spec else "",
                "width": info.width,
                "height": info.height,
                "pattern_count": info.count,
                "size_gib": f"{up2_path.stat().st_size / 1024**3:.3f}",
                "last_write_time": up2_path.stat().st_mtime,
            }
        )
    write_csv(output_dir / "pt_highres_raw_up2_inventory.csv", raw_inventory)

    records: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        acquisition_order_angles = list(range(30, 360, 30)) + [0]
        acquisition_order_by_angle = {angle: order for order, angle in enumerate(acquisition_order_angles, start=1)}
        for spec in specs:
            row = row_by_angle.get(spec.angle_deg)
            if row is None:
                raise KeyError(f"No calibration summary row for angle {spec.angle_deg}")
            map_group = h5[spec.h5_group]
            data = map_group["EBSD/ANG/DATA/DATA"]
            nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
            ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
            sem = normalize_gray(np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
            up2_path = up2_root / spec.up2_name
            up2_info = read_up2_info(up2_path)
            index = fint(row, "selected_index")
            rec = data[index]
            records.append(
                {
                    "angle_deg": spec.angle_deg,
                    "label": spec.label,
                    "h5_group": spec.h5_group,
                    "up2_path": str(up2_path),
                    "h5_grid_rows": nrows,
                    "h5_grid_cols": ncols,
                    "h5_ang_count": data.shape[0],
                    "h5_sem_height": sem.shape[0],
                    "h5_sem_width": sem.shape[1],
                    "h5_acquisition_order": acquisition_order_by_angle[spec.angle_deg],
                    "h5_timestamp_ticks": int(np.asarray(map_group.attrs.get("TimeStamp", [0])).reshape(-1)[0]),
                    "up2_name": up2_path.name,
                    "up2_width": up2_info.width,
                    "up2_height": up2_info.height,
                    "up2_count": up2_info.count,
                    "selected_index": index,
                    "selected_row": fint(row, "row", index // ncols),
                    "selected_col": fint(row, "col", index % ncols),
                    "IQ": float(rec["IQ"]),
                    "CI": float(rec["CI"]),
                    "phase": int(rec["Phase"]),
                    "pc_original_x": ffloat(row, "pc_original_x"),
                    "pc_original_y": ffloat(row, "pc_original_y"),
                    "pc_original_z": ffloat(row, "pc_original_z"),
                    "pc_refined_x": ffloat(row, "pc_refined_x"),
                    "pc_refined_y": ffloat(row, "pc_refined_y"),
                    "pc_refined_z": ffloat(row, "pc_refined_z"),
                    "delta_pcx": ffloat(row, "delta_pcx"),
                    "delta_pcy": ffloat(row, "delta_pcy"),
                    "delta_pcz": ffloat(row, "delta_pcz"),
                    "original_score": ffloat(row, "original_score"),
                    "refined_score": ffloat(row, "refined_score"),
                    "score_gain": ffloat(row, "score_gain"),
                }
            )
    write_csv(output_dir / "pt_highres_12map_visual_inventory.csv", records)
    return records


def save_sem_ipf_overview(h5_path: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    specs = build_map_specs()
    rows_by_angle = {fint(row, "angle_deg"): row for row in rows}
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), dpi=180)
    with h5py.File(h5_path, "r") as h5:
        for idx, spec in enumerate(specs):
            row = rows_by_angle[spec.angle_deg]
            map_group = h5[spec.h5_group]
            data = map_group["EBSD/ANG/DATA/DATA"]
            nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
            ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
            sem = normalize_gray(np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
            ipf = read_ipf_map(map_group)
            sem_x, sem_y = selected_sem_xy(row, sem.shape, nrows, ncols)
            scan_row = fint(row, "row")
            scan_col = fint(row, "col")
            r = idx // 3
            c = (idx % 3) * 2
            ax_sem = axes[r, c]
            ax_ipf = axes[r, c + 1]
            ax_sem.imshow(sem, cmap="gray", vmin=0, vmax=1)
            draw_selected_point(ax_sem, sem_x, sem_y)
            ax_sem.set_title(f"{spec.angle_deg:03d} SEM\nidx={fint(row, 'selected_index')}, IQ={ffloat(row, 'IQ'):.0f}")
            ax_sem.axis("off")
            ax_ipf.imshow(ipf)
            draw_selected_point(ax_ipf, scan_col, scan_row, color="black", size=22.0)
            ax_ipf.set_title(f"{spec.angle_deg:03d} IPF-Z\nr{scan_row} c{scan_col}, CI={ffloat(row, 'CI'):.3f}")
            ax_ipf.axis("off")
            _ = data
    fig.suptitle("Pt high-resolution 12-map data overview: H5 SEM and H5-orientation IPF-Z", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "pt_highres_sem_ipf_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_kikuchi_ohp_overview(h5_path: Path, up2_root: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    specs = build_map_specs()
    rows_by_angle = {fint(row, "angle_deg"): row for row in rows}
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), dpi=180)
    diagnostics: list[dict[str, Any]] = []
    per_angle_dir = output_dir / "per_angle"
    with h5py.File(h5_path, "r") as h5:
        for idx, spec in enumerate(specs):
            row = rows_by_angle[spec.angle_deg]
            map_group = h5[spec.h5_group]
            index = fint(row, "selected_index")
            pattern, _info = read_up2_pattern(up2_root / spec.up2_name, index)
            mask = circular_mask(*pattern.shape, radius_fraction=0.498)
            pattern_display = normalize_for_display(pattern, mask)
            images = build_preprocessed_images(pattern, mask=mask)
            enhanced = images["enhanced"]
            band_display = exposure.rescale_intensity(images["band"], in_range="image", out_range=(0.0, 1.0))
            ohp_diag = score_ohp_alignment(map_group, index, band_display, mask)
            diagnostics.append(
                {
                    "angle_deg": spec.angle_deg,
                    "h5_group": spec.h5_group,
                    "up2_name": spec.up2_name,
                    "selected_index": index,
                    "ohp_line_variant": PUBLICATION_VARIANT.name,
                    **ohp_diag,
                }
            )
            r = idx // 3
            c = (idx % 3) * 2
            ax_kikuchi = axes[r, c]
            ax_ohp = axes[r, c + 1]
            ax_kikuchi.imshow(image_for_panel(enhanced, mask), vmin=0, vmax=1)
            draw_pc_markers(ax_kikuchi, row, pattern.shape[1], pattern.shape[0])
            ax_kikuchi.set_title(
                f"{spec.angle_deg:03d} enhanced Kikuchi\nPC d=({ffloat(row, 'delta_pcx'):+.3f},"
                f" {ffloat(row, 'delta_pcy'):+.3f}, {ffloat(row, 'delta_pcz'):+.3f})"
            )
            ax_kikuchi.axis("off")

            ax_ohp.set_facecolor("black")
            ax_ohp.imshow(pattern_display, cmap="gray", vmin=0, vmax=1, alpha=mask.astype(np.float32))
            band_count = draw_ohp_bands(ax_ohp, map_group, index, pattern.shape)
            draw_pc_markers(ax_ohp, row, pattern.shape[1], pattern.shape[0])
            ax_ohp.set_title(
                f"{spec.angle_deg:03d} raw + OHP bands\n"
                f"bands={band_count}, OHP resp-bg={ohp_diag['ohp_response_minus_background']:+.3f}"
            )
            ax_ohp.axis("off")

            save_individual_views(
                output_dir=per_angle_dir,
                angle=spec.angle_deg,
                sem=normalize_gray(np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32)),
                ipf=read_ipf_map(map_group),
                pattern_display=pattern_display,
                enhanced=enhanced,
                band_display=band_display,
                mask=mask,
                row=row,
                selected_xy=selected_sem_xy(
                    row,
                    map_group["SEM-PRIAS Images/DATA/SEM"].shape,
                    int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0]),
                    int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0]),
                ),
                selected_row_col=(fint(row, "row"), fint(row, "col")),
                map_group=map_group,
            )
    fig.suptitle("Pt high-resolution selected Kikuchi patterns: enhanced disk, EDAX/OHP band reconstruction and PC markers", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "pt_highres_kikuchi_ohp_overview.png", bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir / "pt_highres_ohp_overlay_diagnostics.csv", diagnostics)


def save_scores_overview(calibration_dir: Path, output_dir: Path) -> None:
    summary = read_csv_rows(calibration_dir / "pt_highres_30deg_spherical_calibration_summary.csv")
    forward = read_csv_rows(calibration_dir / "pt_highres_forward_detector_validation_scores.csv")
    pairs_path = calibration_dir / "pt_highres_pair_alignments.csv"
    pairs = read_csv_rows(pairs_path) if pairs_path.exists() else []

    angles = np.array([fint(row, "angle_deg") for row in summary], dtype=np.float64)
    order = np.argsort(angles)
    angles = angles[order]
    original = np.array([ffloat(row, "original_score") for row in summary], dtype=np.float64)[order]
    refined = np.array([ffloat(row, "refined_score") for row in summary], dtype=np.float64)[order]
    gains = np.array([ffloat(row, "score_gain") for row in summary], dtype=np.float64)[order]
    dpcx = np.array([ffloat(row, "delta_pcx") for row in summary], dtype=np.float64)[order]
    dpcy = np.array([ffloat(row, "delta_pcy") for row in summary], dtype=np.float64)[order]
    dpcz = np.array([ffloat(row, "delta_pcz") for row in summary], dtype=np.float64)[order]
    iq = np.array([ffloat(row, "IQ") for row in summary], dtype=np.float64)[order]
    ci = np.array([ffloat(row, "CI") for row in summary], dtype=np.float64)[order]

    f_angles = np.array([ffloat(row, "angle_deg") for row in forward], dtype=np.float64)
    f_base = np.array([ffloat(row, "base_combined_score") for row in forward], dtype=np.float64)
    f_cubic = np.array([ffloat(row, "cubic_combined_score") for row in forward], dtype=np.float64)
    f_delta = np.array([ffloat(row, "cubic_minus_base") for row in forward], dtype=np.float64)
    f_order = np.argsort(f_angles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    ax = axes[0, 0]
    ax.plot(angles, original, "o-", label="original PC")
    ax.plot(angles, refined, "o-", label="refined PC")
    ax.bar(angles, gains, width=8, alpha=0.25, label="score gain")
    ax.set_title("Sphere matching score after PC finetune")
    ax.set_xlabel("in-plane angle (deg)")
    ax.set_ylabel("combined score")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(angles, dpcx, "o-", label="dPCx")
    ax.plot(angles, dpcy, "o-", label="dPCy")
    ax.plot(angles, dpcz, "o-", label="dPCz")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Local PC correction used by the existing calibration")
    ax.set_xlabel("in-plane angle (deg)")
    ax.set_ylabel("PC delta")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(f_angles[f_order], f_base[f_order], "o-", label="detector-validated base/refined")
    ax.plot(f_angles[f_order], f_cubic[f_order], "o-", label="cubic diagnostic branch")
    ax.bar(f_angles[f_order], f_delta[f_order], width=8, alpha=0.25, label="cubic - base")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Forward detector validation: cubic branch is diagnostic only")
    ax.set_xlabel("in-plane angle (deg)")
    ax.set_ylabel("combined score")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(angles, iq / np.nanmax(iq), "o-", label="IQ normalized")
    ax.plot(angles, ci, "o-", label="CI")
    if pairs:
        moving = np.array([ffloat(row, "moving_angle") for row in pairs], dtype=np.float64)
        rmse = np.array([ffloat(row, "residual_rmse") for row in pairs], dtype=np.float64)
        inliers = np.array([ffloat(row, "inliers") for row in pairs], dtype=np.float64)
        ax2 = ax.twinx()
        ax2.plot(moving, rmse, "s--", color="tab:red", label="LightGlue RMSE")
        ax2.set_ylabel("pair RMSE (px)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        for x, y, n in zip(moving, rmse, inliers):
            ax2.text(x, y + 0.03, f"{int(n)}", fontsize=7, ha="center", color="tab:red")
    ax.set_title("Selected pattern quality and adjacent SEM alignment")
    ax.set_xlabel("in-plane angle (deg)")
    ax.set_ylabel("IQ norm / CI")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle("Pt high-resolution 12-map quality, PC and validation diagnostics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "pt_highres_quality_pc_score_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_existing_result_index(calibration_dir: Path, output_dir: Path) -> None:
    panel_files = [
        ("SEM LightGlue chain", "pt_highres_sem_lightglue_alignment_overview.png"),
        ("Same point selection", "pt_highres_same_point_selection.png"),
        ("Kikuchi + PC markers", "pt_highres_selected_kikuchi_pc_patterns.png"),
        ("Detector-validated sphere front views", "pt_highres_sphere_matching_front_views.png"),
        ("One master sphere lon/colat", "pt_highres_same_sphere_lon_colat.png"),
        ("PC anchor lon/colat", "pt_highres_pc_anchor_lon_colat.png"),
        ("Forward detector validation", "pt_highres_forward_detector_validation.png"),
        ("Cubic diagnostic front views", "pt_highres_cubic_symmetry_front_views.png"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=150)
    for ax, (title, filename) in zip(axes.ravel(), panel_files):
        path = calibration_dir / filename
        if path.exists():
            image = plt.imread(path)
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Index of existing Pt high-resolution calibration visualizations", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "pt_highres_existing_calibration_result_index.png", bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, h5_path: Path, up2_root: Path, calibration_dir: Path, records: list[dict[str, Any]]) -> None:
    angles = [int(record["angle_deg"]) for record in records]
    best = max(records, key=lambda item: float(item["refined_score"]))
    worst = min(records, key=lambda item: float(item["refined_score"]))
    lines = [
        "# Pt High-Resolution Data Overview",
        "",
        f"- H5: `{h5_path}`",
        f"- UP2 root: `{up2_root}`",
        f"- Calibration source: `{calibration_dir}`",
        f"- Formal 30-degree series angles: {', '.join(map(str, angles))}",
        f"- Best refined sphere score: {best['angle_deg']} deg, {float(best['refined_score']):+.5f}",
        f"- Lowest refined sphere score: {worst['angle_deg']} deg, {float(worst['refined_score']):+.5f}",
        "",
        "## Generated Visualizations",
        "",
        "- `pt_highres_sem_ipf_overview.png`: each formal angle as H5 SEM + H5 orientation IPF-Z with the selected EBSD pixel.",
        "- `pt_highres_kikuchi_ohp_overview.png`: enhanced Kikuchi disk and raw Kikuchi with EDAX/OHP bands using the corrected `normal_theta_rho+_yup` convention.",
        "- `pt_highres_ohp_overlay_diagnostics.csv`: per-angle OHP line response on the band-enhanced Kikuchi image, used as an overlay sanity check.",
        "- `pt_highres_quality_pc_score_overview.png`: score, PC residual, forward detector validation and LightGlue alignment diagnostics.",
        "- `pt_highres_existing_calibration_result_index.png`: index sheet for the existing full calibration visualizations.",
        "- `per_angle/`: individual SEM, IPF, Kikuchi, enhanced Kikuchi, band response and OHP-band PNGs for every angle.",
        "",
        "## Notes",
        "",
        "- The 12 formal EBSD mappings live in the H5 as `Area 8-0` through `Area 8-330`; the matching UP2 files are `Area 3` through `Area 14` in acquisition order `30, 60, ..., 330, 0`.",
        "- The extra small UP2 files in the same directory are listed in `pt_highres_raw_up2_inventory.csv` but are not part of this 12-map rotation series.",
        "- The detector-validated sphere placement remains the primary result. The cubic-symmetry branch is kept as a diagnostic and should not overwrite local sphere matching.",
        "",
    ]
    (output_dir / "pt_highres_visual_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.calibration_dir / "pt_highres_30deg_spherical_calibration_summary.csv"
    summary_rows = sorted(read_csv_rows(summary_path), key=lambda item: fint(item, "angle_deg"))
    records = build_records(args.h5, args.up2_root, args.calibration_dir, args.output_dir)
    save_sem_ipf_overview(args.h5, summary_rows, args.output_dir)
    save_kikuchi_ohp_overview(args.h5, args.up2_root, summary_rows, args.output_dir)
    save_scores_overview(args.calibration_dir, args.output_dir)
    save_existing_result_index(args.calibration_dir, args.output_dir)
    write_report(args.output_dir, args.h5, args.up2_root, args.calibration_dir, records)
    print(f"Saved Pt high-resolution data overview to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Pt high-resolution H5/UP2 metadata and existing 30-degree spherical calibration outputs "
            "into SEM/IPF/Kikuchi/OHP/score overview figures."
        )
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--up2-root", type=Path, default=DEFAULT_UP2_ROOT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
