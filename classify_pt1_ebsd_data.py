"""Classify Pt-1 EDAX H5/UP2 metadata without loading full pattern stacks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATA_DIR = Path(r"D:\EBSD-data\Pt-1")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt1_classification"


@dataclass(frozen=True)
class Up2Info:
    path: Path
    version: int
    width: int
    height: int
    header_bytes: int
    count: int
    file_bytes: int
    modified_time: str

    @property
    def pattern_bytes(self) -> int:
        return self.width * self.height * np.dtype("<u2").itemsize

    @property
    def resolution_label(self) -> str:
        if self.width >= 400 and self.height >= 400:
            return "HighR/raw 470x470"
        if self.width <= 128 and self.height <= 128:
            return "LowR/binned 117x117"
        return f"{self.width}x{self.height}"


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_value(value[()])
        if value.size == 1:
            return decode_value(value.reshape(-1)[0])
        return [decode_value(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def h5_scalar(h5: h5py.File, path: str) -> Any:
    if path not in h5:
        return None
    return decode_value(h5[path][()])


def dotnet_ticks_to_iso(ticks: Any) -> str:
    if ticks is None:
        return ""
    try:
        value = int(decode_value(ticks))
        when = dt.datetime(1, 1, 1) + dt.timedelta(microseconds=value // 10)
        return when.isoformat(sep=" ")
    except Exception:
        return ""


def read_up2_info(path: Path) -> Up2Info:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError(f"{path} is too small to contain a UP2 header")
    version, width, height, header_bytes = struct.unpack("<4I", header)
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    file_bytes = path.stat().st_size
    payload_bytes = file_bytes - header_bytes
    if payload_bytes < 0 or pattern_bytes <= 0 or payload_bytes % pattern_bytes:
        raise ValueError(
            f"{path} has an inconsistent UP2 header: "
            f"width={width}, height={height}, header={header_bytes}, size={file_bytes}"
        )
    return Up2Info(
        path=path,
        version=version,
        width=width,
        height=height,
        header_bytes=header_bytes,
        count=payload_bytes // pattern_bytes,
        file_bytes=file_bytes,
        modified_time=dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" "),
    )


def find_specimen_root(h5: h5py.File, specimen: str) -> str:
    candidates: list[str] = []
    for top in h5.keys():
        candidate = f"{top}/{specimen}"
        if candidate in h5:
            candidates.append(candidate)
    if not candidates:
        raise KeyError(f"Could not find specimen {specimen!r} in {list(h5.keys())}")
    if len(candidates) > 1:
        raise KeyError(f"Found multiple specimen roots for {specimen!r}: {candidates}")
    return candidates[0]


def phase_header_summary(h5: h5py.File, map_path: str) -> str:
    root = f"{map_path}/EBSD/ANG/HEADER/Phase"
    if root not in h5:
        return ""

    parts: list[str] = []
    for phase_id in sorted(h5[root].keys(), key=lambda x: int(x) if x.isdigit() else x):
        phase_path = f"{root}/{phase_id}"
        material = h5_scalar(h5, f"{phase_path}/Material Name") or ""
        lattice_a = h5_scalar(h5, f"{phase_path}/Lattice Constant A")
        hkl_path = f"{phase_path}/HKL Families"
        hkl_text = ""
        if hkl_path in h5:
            rows = h5[hkl_path][()]
            families = []
            for row in rows:
                families.append(f"({int(row['H'])}{int(row['K'])}{int(row['L'])})")
            hkl_text = ",".join(families)
        if lattice_a is not None:
            parts.append(f"{phase_id}:{material},a={float(lattice_a):g},HKL={hkl_text}")
        else:
            parts.append(f"{phase_id}:{material},HKL={hkl_text}")
    return "; ".join(parts)


def collect_h5_maps(h5_path: Path, specimen: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        specimen_root = find_specimen_root(h5, specimen)
        map_paths: list[str] = []

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not name.startswith(specimen_root + "/"):
                return
            if not isinstance(obj, h5py.Group):
                return
            if not name.rsplit("/", 1)[-1].startswith("OIM Map"):
                return
            if f"{name}/EBSD/ANG/DATA/DATA" in h5:
                map_paths.append(name)

        h5.visititems(visitor)
        for map_path in sorted(map_paths):
            data = h5[f"{map_path}/EBSD/ANG/DATA/DATA"]
            count = int(data.shape[0])
            phase_values, phase_counts = np.unique(data["Phase"][:], return_counts=True)
            phase_counts_text = "; ".join(
                f"{int(phase)}:{int(cnt)}" for phase, cnt in zip(phase_values, phase_counts)
            )
            iq = data["IQ"][:].astype(np.float64)
            ci = data["CI"][:].astype(np.float64)
            fit = data["Fit"][:].astype(np.float64)
            valid = data["Valid"][:]
            timestamp = decode_value(h5[map_path].attrs.get("TimeStamp", None))

            rows.append(
                {
                    "h5_file": str(h5_path),
                    "h5_path": map_path,
                    "specimen": specimen,
                    "area": map_path.split("/")[-2],
                    "map_name": map_path.split("/")[-1],
                    "scan_time": dotnet_ticks_to_iso(timestamp),
                    "rows": h5_scalar(h5, f"{map_path}/Sample/Number Of Rows"),
                    "cols": h5_scalar(h5, f"{map_path}/Sample/Number Of Columns"),
                    "point_count": count,
                    "step_x_um": h5_scalar(h5, f"{map_path}/Sample/Step X"),
                    "step_y_um": h5_scalar(h5, f"{map_path}/Sample/Step Y"),
                    "sample_tilt_deg": h5_scalar(h5, f"{map_path}/Sample/Sample Tilt"),
                    "pre_tilt_deg": h5_scalar(h5, f"{map_path}/Sample/Pre Tilt"),
                    "pc_x": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/X-Star"),
                    "pc_y": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/Y-Star"),
                    "pc_z": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/Z-Star"),
                    "ang_fields": ",".join(data.dtype.names or []),
                    "ohp_shape": str(h5[f"{map_path}/EBSD/OHP/DATA/DATA"].shape)
                    if f"{map_path}/EBSD/OHP/DATA/DATA" in h5
                    else "",
                    "ohp_band_float_count": 32
                    if f"{map_path}/EBSD/OHP/DATA/DATA" in h5
                    else 0,
                    "phase_counts": phase_counts_text,
                    "phase_header": phase_header_summary(h5, map_path),
                    "iq_min": float(np.nanmin(iq)),
                    "iq_mean": float(np.nanmean(iq)),
                    "iq_max": float(np.nanmax(iq)),
                    "ci_min": float(np.nanmin(ci)),
                    "ci_mean": float(np.nanmean(ci)),
                    "ci_max": float(np.nanmax(ci)),
                    "fit_mean": float(np.nanmean(fit)),
                    "valid_count": int(np.count_nonzero(valid)),
                }
            )
    return rows


def collect_up2_files(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.up2")):
        info = read_up2_info(path)
        area_match = re.search(r"Area\s+([0-9]+)", path.name, flags=re.IGNORECASE)
        map_match = re.search(r"OIM Map\s+([0-9.]+)", path.name, flags=re.IGNORECASE)
        rows.append(
            {
                **asdict(info),
                "path": str(info.path),
                "file_name": info.path.name,
                "area_from_name": area_match.group(1) if area_match else "",
                "map_from_name": map_match.group(1).rstrip(".") if map_match else "",
                "shape": f"({info.count}, {info.height}, {info.width})",
                "pattern_bytes": info.pattern_bytes,
                "resolution_label": info.resolution_label,
            }
        )
    return rows


def classify_pairs(h5_rows: list[dict[str, Any]], up2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    used_h5: set[str] = set()
    used_up2: set[str] = set()

    h5_by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    up2_by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in h5_rows:
        h5_by_count[int(row["point_count"])].append(row)
    for row in up2_rows:
        up2_by_count[int(row["count"])].append(row)

    for count in sorted(set(h5_by_count) & set(up2_by_count)):
        h5_candidates = sorted(h5_by_count[count], key=lambda r: (r.get("scan_time") or "", r["h5_path"]))
        up2_candidates = sorted(up2_by_count[count], key=lambda r: (r.get("modified_time") or "", r["file_name"]))
        for h5_row, up2_row in zip(h5_candidates, up2_candidates):
            used_h5.add(h5_row["h5_path"])
            used_up2.add(up2_row["path"])
            paired.append(
                {
                    "classification": "matched_h5_up2",
                    "use_recommendation": "usable for raw Kikuchi + H5 PC/orientation/OHP band workflows",
                    "point_count": count,
                    "up2_file": up2_row["file_name"],
                    "up2_path": up2_row["path"],
                    "up2_resolution": up2_row["resolution_label"],
                    "up2_shape": up2_row["shape"],
                    "up2_modified_time": up2_row["modified_time"],
                    "h5_area": h5_row["area"],
                    "h5_map": h5_row["map_name"],
                    "h5_path": h5_row["h5_path"],
                    "h5_scan_time": h5_row["scan_time"],
                    "pc_x": h5_row["pc_x"],
                    "pc_y": h5_row["pc_y"],
                    "pc_z": h5_row["pc_z"],
                    "phase_counts": h5_row["phase_counts"],
                    "phase_header": h5_row["phase_header"],
                    "iq_mean": h5_row["iq_mean"],
                    "ci_mean": h5_row["ci_mean"],
                    "fit_mean": h5_row["fit_mean"],
                    "match_basis": "exact point_count, then chronological pairing within same count",
                }
            )

    for row in up2_rows:
        if row["path"] in used_up2:
            continue
        if int(row["count"]) in h5_by_count:
            detail = (
                "Exact Pt-1 H5 point_count exists, but there are more UP2 stacks than H5 maps "
                "for this count; left unpaired after chronological matching"
            )
        elif row["width"] >= 400:
            detail = "HighR/raw UP2 stack without a matching Pt-1 H5 OIM map"
        elif row["width"] <= 128:
            detail = "LowR/binned UP2 stack without a matching Pt-1 H5 OIM map"
        else:
            detail = "UP2 stack without a matching Pt-1 H5 OIM map"
        paired.append(
            {
                "classification": "up2_only_unmatched_to_pt1_h5",
                "use_recommendation": "readable as raw pattern stack only; do not attach Pt-1 H5 PC/orientation/OHP unless matching metadata is found",
                "point_count": row["count"],
                "up2_file": row["file_name"],
                "up2_path": row["path"],
                "up2_resolution": row["resolution_label"],
                "up2_shape": row["shape"],
                "up2_modified_time": row["modified_time"],
                "h5_area": "",
                "h5_map": "",
                "h5_path": "",
                "h5_scan_time": "",
                "pc_x": "",
                "pc_y": "",
                "pc_z": "",
                "phase_counts": "",
                "phase_header": "",
                "iq_mean": "",
                "ci_mean": "",
                "fit_mean": "",
                "match_basis": detail,
            }
        )

    for row in h5_rows:
        if row["h5_path"] in used_h5:
            continue
        paired.append(
            {
                "classification": "h5_map_without_up2_in_folder",
                "use_recommendation": "H5 metadata exists, but the corresponding UP2 stack is not present in this folder",
                "point_count": row["point_count"],
                "up2_file": "",
                "up2_path": "",
                "up2_resolution": "",
                "up2_shape": "",
                "up2_modified_time": "",
                "h5_area": row["area"],
                "h5_map": row["map_name"],
                "h5_path": row["h5_path"],
                "h5_scan_time": row["scan_time"],
                "pc_x": row["pc_x"],
                "pc_y": row["pc_y"],
                "pc_z": row["pc_z"],
                "phase_counts": row["phase_counts"],
                "phase_header": row["phase_header"],
                "iq_mean": row["iq_mean"],
                "ci_mean": row["ci_mean"],
                "fit_mean": row["fit_mean"],
                "match_basis": "H5 point_count not present in UP2 files",
            }
        )
    return sorted(paired, key=lambda r: (r["classification"], str(r["h5_scan_time"]), str(r["up2_modified_time"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    path: Path,
    data_dir: Path,
    h5_path: Path,
    h5_rows: list[dict[str, Any]],
    up2_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
) -> None:
    counts = defaultdict(int)
    for row in classification_rows:
        counts[row["classification"]] += 1

    lines = [
        "# Pt-1 EBSD Data Classification",
        "",
        f"- Data directory: `{data_dir}`",
        f"- H5 file: `{h5_path}`",
        f"- H5 Pt-1 OIM maps with ANG/OHP metadata: {len(h5_rows)}",
        f"- UP2 files in folder: {len(up2_rows)}",
        f"- Matched H5-UP2 pairs: {counts['matched_h5_up2']}",
        f"- UP2-only unmatched stacks: {counts['up2_only_unmatched_to_pt1_h5']}",
        f"- H5 maps without UP2 in this folder: {counts['h5_map_without_up2_in_folder']}",
        "",
        "## Interpretation",
        "",
        "The matched class is the reliable set for combined raw Kikuchi pattern, EDAX PC, OIM orientation, IQ/CI/Phase, and OHP Kikuchi-band analysis. "
        "The unmatched UP2 stacks can still be read as raw patterns, but should not be joined to the Pt-1 H5 metadata until the corresponding OIM map is identified.",
        "",
        "## Matched H5-UP2 Pairs",
        "",
        "| UP2 file | Resolution | Count | H5 area/map | PC | Phase counts |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in classification_rows:
        if row["classification"] != "matched_h5_up2":
            continue
        pc = f"({float(row['pc_x']):.6f}, {float(row['pc_y']):.6f}, {float(row['pc_z']):.6f})"
        lines.append(
            f"| `{row['up2_file']}` | {row['up2_resolution']} | {row['point_count']} | "
            f"{row['h5_area']} / {row['h5_map']} | {pc} | {row['phase_counts']} |"
        )

    lines.extend(
        [
            "",
            "## Unmatched UP2 Stacks",
            "",
            "| UP2 file | Resolution | Count | Current class |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in classification_rows:
        if row["classification"] != "up2_only_unmatched_to_pt1_h5":
            continue
        lines.append(
            f"| `{row['up2_file']}` | {row['up2_resolution']} | {row['point_count']} | {row['match_basis']} |"
        )

    lines.extend(
        [
            "",
            "## H5 Pt-1 Map Summary",
            "",
            "| H5 area/map | Grid | Count | Step (um) | PC | IQ mean | CI mean |",
            "| --- | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(h5_rows, key=lambda r: r["scan_time"]):
        pc = f"({float(row['pc_x']):.6f}, {float(row['pc_y']):.6f}, {float(row['pc_z']):.6f})"
        grid = f"{row['rows']} x {row['cols']}"
        step = f"{float(row['step_x_um']):.3g}, {float(row['step_y_um']):.3g}"
        lines.append(
            f"| {row['area']} / {row['map_name']} | {grid} | {row['point_count']} | {step} | "
            f"{pc} | {float(row['iq_mean']):.1f} | {float(row['ci_mean']):.3f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, classification_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels: list[str] = []
    counts: list[int] = []
    grouped = defaultdict(int)
    for row in classification_rows:
        label = row["classification"]
        if label == "matched_h5_up2":
            label = "matched H5+UP2"
        elif label == "up2_only_unmatched_to_pt1_h5":
            label = "UP2 only"
        elif label == "h5_map_without_up2_in_folder":
            label = "H5 only"
        grouped[label] += 1
    for label, value in sorted(grouped.items()):
        labels.append(label)
        counts.append(value)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(labels, counts, color=["#1f77b4", "#d62728", "#7f7f7f"][: len(labels)])
    ax.set_ylabel("file/map groups")
    ax.set_title("Pt-1 EBSD data classification")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(counts):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the Pt-1 EDAX H5/UP2 dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--specimen", default="Pt-1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir
    h5_path = args.h5
    if h5_path is None:
        h5_files = sorted(data_dir.glob("*.edaxh5"))
        if len(h5_files) != 1:
            raise FileNotFoundError(f"Expected exactly one .edaxh5 in {data_dir}, found {h5_files}")
        h5_path = h5_files[0]

    h5_rows = collect_h5_maps(h5_path, args.specimen)
    up2_rows = collect_up2_files(data_dir)
    classification_rows = classify_pairs(h5_rows, up2_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pt1_h5_maps.csv", h5_rows)
    write_csv(output_dir / "pt1_up2_files.csv", up2_rows)
    write_csv(output_dir / "pt1_match_classification.csv", classification_rows)
    write_markdown_summary(
        output_dir / "pt1_classification_summary.md",
        data_dir,
        h5_path,
        h5_rows,
        up2_rows,
        classification_rows,
    )
    write_plot(output_dir / "pt1_classification_overview.png", classification_rows)
    (output_dir / "pt1_summary.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "h5_path": str(h5_path),
                "specimen": args.specimen,
                "h5_map_count": len(h5_rows),
                "up2_file_count": len(up2_rows),
                "classification": classification_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote Pt-1 classification to {output_dir}")
    print(f"H5 maps: {len(h5_rows)}")
    print(f"UP2 files: {len(up2_rows)}")
    print(f"Matched pairs: {sum(r['classification'] == 'matched_h5_up2' for r in classification_rows)}")
    print(
        "UP2-only unmatched: "
        f"{sum(r['classification'] == 'up2_only_unmatched_to_pt1_h5' for r in classification_rows)}"
    )


if __name__ == "__main__":
    main()
