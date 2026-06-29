"""Build a UP2-centric table linking each raw pattern stack to EBSD metadata and a Kikuchi example."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from export_h5_mapping_sem_correspondence import (
    Up2Candidate,
    collect_h5_maps,
    collect_up2_candidates,
    match_h5_to_up2,
    normalize_gray,
    safe_name,
)
from export_pt1_sem_kikuchi_pairs import (
    choose_representative_index,
    circular_rgba,
    read_up2_pattern,
    scan_to_sem_xy,
)


DEFAULT_H5_PATH = Path(r"E:\ZHL\EBSD-RAW\20251209Pt\20251209Pt.edaxh5")
DEFAULT_UP2_ROOT = Path(r"E:\ZHL\EBSD-RAW\20251209Pt")
DEFAULT_OUTPUT_DIR = Path("outputs") / "up2_ebsd_kikuchi_correspondence_E_20251209Pt"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sem_kikuchi_pair(
    sem_gray: np.ndarray,
    pattern_gray: np.ndarray,
    sem_x: float,
    sem_y: float,
    title: str,
    details: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=200)
    axes[0].imshow(sem_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter([sem_x], [sem_y], s=95, facecolors="none", edgecolors="#ff2020", linewidths=1.8)
    axes[0].scatter([sem_x], [sem_y], s=10, c="#ff2020")
    axes[0].set_title("EBSD SEM with selected point", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(pattern_gray, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Raw UP2 Kikuchi at same index", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"{title}\n{details}", fontsize=9)
    fig.tight_layout(pad=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_kikuchi_only(pattern_gray: np.ndarray, title: str, details: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=200)
    ax.imshow(pattern_gray, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{title}\n{details}", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def candidate_key(candidate: Up2Candidate) -> str:
    return str(candidate.actual_path.resolve()).lower()


def make_contact_sheet(rows: list[dict[str, Any]], image_key: str, title: str, output_path: Path) -> None:
    image_rows = [row for row in rows if row.get(image_key)]
    if not image_rows:
        return
    cols = 3
    rows_n = math.ceil(len(image_rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.0, rows_n * 3.8), dpi=170)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, row in zip(axes_arr.ravel(), image_rows):
        image = plt.imread(row[image_key])
        ax.imshow(image)
        status = "matched" if row["match_status"] == "matched_to_h5_ebsd" else "unmatched"
        ax.set_title(f"{row['up2_file']}\n{status}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    matched = [row for row in rows if row["match_status"] == "matched_to_h5_ebsd"]
    unmatched = [row for row in rows if row["match_status"] != "matched_to_h5_ebsd"]
    lines = [
        "# UP2 EBSD Kikuchi Correspondence",
        "",
        f"- UP2 files scanned: {len(rows)}",
        f"- UP2 files matched to H5 EBSD mappings: {len(matched)}",
        f"- UP2 files without H5 EBSD mapping: {len(unmatched)}",
        "",
        "## Matched UP2 Files",
        "",
        "| UP2 file | EBSD mapping | Count | Resolution | Selected Kikuchi index |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in matched:
        lines.append(
            f"| `{row['up2_file']}` | {row['specimen']} / {row['h5_area']} / {row['h5_map']} | "
            f"{row['up2_count']} | {row['up2_resolution']} | {row['selected_index']} |"
        )

    lines.extend(
        [
            "",
            "## UP2 Files Without H5 EBSD Mapping",
            "",
            "| UP2 file | Count | Resolution | Note |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in unmatched:
        lines.append(
            f"| `{row['up2_file']}` | {row['up2_count']} | {row['up2_resolution']} | {row['match_note']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_correspondence(h5_path: Path, up2_roots: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    h5_rows = collect_h5_maps(h5_path)
    up2_candidates = collect_up2_candidates(up2_roots)
    h5_to_up2_rows, _unmatched_candidates = match_h5_to_up2(h5_rows, up2_candidates)
    matched_by_up2 = {
        str(Path(row["up2_actual_path"]).resolve()).lower(): row
        for row in h5_to_up2_rows
        if row.get("up2_actual_path")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = output_dir / "per_up2"
    rows: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5:
        for candidate in sorted(up2_candidates, key=lambda item: (item.modified_time, item.display_name)):
            key = candidate_key(candidate)
            h5_row = matched_by_up2.get(key)
            out_key = safe_name(candidate.display_name)
            out_dir = per_file_dir / out_key
            kikuchi_png = ""
            kikuchi_transparent_png = ""
            pair_png = ""
            selected_index: int | None = None
            selected_rule = ""
            selected_iq: float | str = ""
            selected_ci: float | str = ""
            scan_row: int | str = ""
            scan_col: int | str = ""
            match_note = ""

            if h5_row is not None:
                map_group = h5[h5_row["h5_path"]]
                selected_index, meta = choose_representative_index(map_group)
                selected_rule = "maximum valid IQ in matched H5 ANG/DATA"
                selected_iq = float(meta["IQ"])
                selected_ci = float(meta["CI"])
                nrows = int(np.asarray(map_group["Sample/Number Of Rows"][()]).reshape(-1)[0])
                ncols = int(np.asarray(map_group["Sample/Number Of Columns"][()]).reshape(-1)[0])
                sem = np.asarray(map_group["SEM-PRIAS Images/DATA/SEM"][:])
                sem_gray = normalize_gray(sem)
                sem_x, sem_y, scan_row_int, scan_col_int = scan_to_sem_xy(selected_index, nrows, ncols, sem_gray.shape)
                scan_row = scan_row_int
                scan_col = scan_col_int
                pattern, _info = read_up2_pattern(candidate.actual_path, selected_index)
                pattern_gray = normalize_gray(pattern)

                out_dir.mkdir(parents=True, exist_ok=True)
                kikuchi_png_path = out_dir / f"{out_key}_selected_kikuchi.png"
                kikuchi_transparent_path = out_dir / f"{out_key}_selected_kikuchi_circular_transparent.png"
                pair_png_path = out_dir / f"{out_key}_ebsd_sem_kikuchi_pair.png"
                plt.imsave(kikuchi_png_path, pattern_gray, cmap="gray", vmin=0, vmax=1)
                plt.imsave(kikuchi_transparent_path, circular_rgba(pattern_gray))
                title = f"{h5_row['specimen']} | {h5_row['area']} | {h5_row['map_name']}"
                details = (
                    f"{candidate.display_name} | idx={selected_index}, row={scan_row}, col={scan_col}, "
                    f"IQ={selected_iq:.1f}, CI={selected_ci:.3f}"
                )
                save_sem_kikuchi_pair(sem_gray, pattern_gray, sem_x, sem_y, title, details, pair_png_path)
                kikuchi_png = str(kikuchi_png_path)
                kikuchi_transparent_png = str(kikuchi_transparent_path)
                pair_png = str(pair_png_path)
                match_status = "matched_to_h5_ebsd"
                match_note = "UP2 is linked to H5 EBSD by specimen + exact pattern count + chronological order."
            else:
                match_status = "no_h5_ebsd_mapping"
                if candidate.count > 0:
                    selected_index = candidate.count // 2
                    selected_rule = "center index only; no matched H5 EBSD metadata"
                    pattern, _info = read_up2_pattern(candidate.actual_path, selected_index)
                    pattern_gray = normalize_gray(pattern)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    kikuchi_png_path = out_dir / f"{out_key}_center_kikuchi_unmatched.png"
                    kikuchi_transparent_path = out_dir / f"{out_key}_center_kikuchi_circular_transparent_unmatched.png"
                    plt.imsave(kikuchi_png_path, pattern_gray, cmap="gray", vmin=0, vmax=1)
                    plt.imsave(kikuchi_transparent_path, circular_rgba(pattern_gray))
                    save_kikuchi_only(
                        pattern_gray,
                        candidate.display_name,
                        f"No matched H5 EBSD mapping | center idx={selected_index}",
                        out_dir / f"{out_key}_unmatched_kikuchi_preview.png",
                    )
                    kikuchi_png = str(kikuchi_png_path)
                    kikuchi_transparent_png = str(kikuchi_transparent_path)
                    pair_png = str(out_dir / f"{out_key}_unmatched_kikuchi_preview.png")
                    match_note = "No H5 EBSD map with same specimen/count was assigned; Kikuchi preview uses center index."
                else:
                    selected_index = None
                    selected_rule = "not readable: UP2 has zero patterns"
                    match_note = "UP2 header exists but pattern count is zero."

            rows.append(
                {
                    "up2_file": candidate.display_name,
                    "up2_path": str(candidate.actual_path),
                    "up2_original_path": candidate.original_path,
                    "up2_count": candidate.count,
                    "up2_resolution": candidate.resolution,
                    "up2_modified_time": candidate.modified_time,
                    "specimen": h5_row["specimen"] if h5_row is not None else candidate.specimen,
                    "h5_area": h5_row["area"] if h5_row is not None else "",
                    "h5_map": h5_row["map_name"] if h5_row is not None else "",
                    "h5_path": h5_row["h5_path"] if h5_row is not None else "",
                    "h5_scan_time": h5_row["scan_time"] if h5_row is not None else "",
                    "match_status": match_status,
                    "match_note": match_note,
                    "selected_index": "" if selected_index is None else selected_index,
                    "selected_rule": selected_rule,
                    "scan_row": scan_row,
                    "scan_col": scan_col,
                    "selected_iq": selected_iq,
                    "selected_ci": selected_ci,
                    "kikuchi_png": kikuchi_png,
                    "kikuchi_circular_transparent_png": kikuchi_transparent_png,
                    "sem_kikuchi_pair_png": pair_png,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Link every UP2 file to H5 EBSD metadata and export Kikuchi previews.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--up2-root", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = build_correspondence(args.h5, args.up2_root or [DEFAULT_UP2_ROOT], args.output_dir)
    write_csv(args.output_dir / "up2_ebsd_kikuchi_correspondence.csv", rows)
    write_markdown(args.output_dir / "up2_ebsd_kikuchi_correspondence.md", rows)
    make_contact_sheet(
        [row for row in rows if row["match_status"] == "matched_to_h5_ebsd"],
        "sem_kikuchi_pair_png",
        "Matched UP2 / EBSD / Kikuchi examples",
        args.output_dir / "matched_up2_ebsd_kikuchi_contact_sheet.png",
    )
    make_contact_sheet(
        [row for row in rows if row["match_status"] != "matched_to_h5_ebsd"],
        "sem_kikuchi_pair_png",
        "Unmatched UP2 Kikuchi previews",
        args.output_dir / "unmatched_up2_kikuchi_contact_sheet.png",
    )

    matched = sum(row["match_status"] == "matched_to_h5_ebsd" for row in rows)
    print(f"Scanned UP2 files: {len(rows)}")
    print(f"Matched to H5 EBSD: {matched}")
    print(f"Unmatched UP2 files: {len(rows) - matched}")
    print(f"CSV: {args.output_dir / 'up2_ebsd_kikuchi_correspondence.csv'}")


if __name__ == "__main__":
    main()
