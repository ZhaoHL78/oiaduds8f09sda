"""Export SEM images for every EDAX H5 mapping and match them to local UP2 names."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


DEFAULT_H5_PATH = Path(r"D:\EBSD-data\Pt-1\20251209Pt.edaxh5")
DEFAULT_OUTPUT_DIR = Path("outputs") / "h5_mapping_sem_correspondence"
DEFAULT_UP2_ROOTS = [Path(r"D:\EBSD-data"), Path(r"D:\$RECYCLE.BIN")]


@dataclass(frozen=True)
class Up2Candidate:
    display_name: str
    original_path: str
    actual_path: Path
    location_status: str
    modified_time: str
    file_bytes: int
    version: int
    width: int
    height: int
    header_bytes: int
    count: int
    specimen: str

    @property
    def resolution(self) -> str:
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


def parse_iso_time(text: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(text)
    except Exception:
        return dt.datetime.max


def clean_recycle_original_path(text: str) -> str:
    match = re.search(r"[A-Za-z]:\\", text)
    if match:
        return text[match.start() :]
    return text.strip("\x00")


def parse_specimen_from_name(name: str) -> str:
    match = re.search(r"_(Pt-\d+)_", name, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def read_up2_header(path: Path) -> tuple[int, int, int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError(f"{path} is too small to contain a UP2 header")
    version, width, height, header_bytes = struct.unpack("<4I", header)
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    payload_bytes = path.stat().st_size - header_bytes
    if payload_bytes < 0 or pattern_bytes <= 0 or payload_bytes % pattern_bytes:
        raise ValueError(f"Inconsistent UP2 header in {path}")
    return version, width, height, header_bytes, payload_bytes // pattern_bytes


def recycle_filetime_to_iso(filetime: int) -> str:
    try:
        when = dt.datetime(1601, 1, 1) + dt.timedelta(microseconds=filetime // 10)
        return when.isoformat(sep=" ")
    except Exception:
        return ""


def read_recycle_metadata(i_path: Path) -> tuple[str, int, str] | None:
    data = i_path.read_bytes()
    if len(data) < 24:
        return None
    original_size = struct.unpack("<Q", data[8:16])[0]
    deleted_filetime = struct.unpack("<Q", data[16:24])[0]
    original_path = clean_recycle_original_path(data[24:].decode("utf-16-le", "ignore").rstrip("\x00"))
    return original_path, original_size, recycle_filetime_to_iso(deleted_filetime)


def collect_up2_candidates(roots: list[Path]) -> list[Up2Candidate]:
    candidates: list[Up2Candidate] = []
    seen_actual: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        if "$RECYCLE.BIN" in str(root).upper():
            for i_path in root.rglob("$I*.up2"):
                metadata = read_recycle_metadata(i_path)
                if metadata is None:
                    continue
                original_path, original_size, _deleted_time = metadata
                r_path = i_path.with_name("$R" + i_path.name[2:])
                if not r_path.exists():
                    continue
                try:
                    version, width, height, header_bytes, count = read_up2_header(r_path)
                except Exception:
                    continue
                key = str(r_path.resolve()).lower()
                seen_actual.add(key)
                modified = dt.datetime.fromtimestamp(r_path.stat().st_mtime).isoformat(sep=" ")
                display_name = Path(original_path).name
                candidates.append(
                    Up2Candidate(
                        display_name=display_name,
                        original_path=original_path,
                        actual_path=r_path,
                        location_status="found_in_recycle_bin",
                        modified_time=modified,
                        file_bytes=int(original_size) if original_size else r_path.stat().st_size,
                        version=version,
                        width=width,
                        height=height,
                        header_bytes=header_bytes,
                        count=count,
                        specimen=parse_specimen_from_name(display_name),
                    )
                )
            continue

        for path in root.rglob("*.up2"):
            if "$RECYCLE.BIN" in str(path).upper():
                continue
            if path.name.startswith("$I"):
                continue
            key = str(path.resolve()).lower()
            if key in seen_actual:
                continue
            try:
                version, width, height, header_bytes, count = read_up2_header(path)
            except Exception:
                continue
            modified = dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ")
            candidates.append(
                Up2Candidate(
                    display_name=path.name,
                    original_path=str(path),
                    actual_path=path,
                    location_status="found_at_path",
                    modified_time=modified,
                    file_bytes=path.stat().st_size,
                    version=version,
                    width=width,
                    height=height,
                    header_bytes=header_bytes,
                    count=count,
                    specimen=parse_specimen_from_name(path.name),
                )
            )

    return sorted(candidates, key=lambda item: (item.specimen, item.count, item.modified_time, item.display_name))


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


def collect_h5_maps(h5_path: Path, specimen_filter: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        map_paths: list[str] = []

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Group):
                return
            if not name.rsplit("/", 1)[-1].startswith("OIM Map"):
                return
            if f"{name}/EBSD/ANG/DATA/DATA" in h5 and f"{name}/SEM-PRIAS Images/DATA/SEM" in h5:
                map_paths.append(name)

        h5.visititems(visitor)
        for map_path in sorted(map_paths):
            parts = map_path.split("/")
            specimen = parts[1] if len(parts) > 1 else ""
            if specimen_filter and specimen != specimen_filter:
                continue
            area = parts[-2]
            map_name = parts[-1]
            data = h5[f"{map_path}/EBSD/ANG/DATA/DATA"]
            count = int(data.shape[0])
            scan_time = dotnet_ticks_to_iso(decode_value(h5[map_path].attrs.get("TimeStamp", None)))
            rows.append(
                {
                    "h5_path": map_path,
                    "specimen": specimen,
                    "area": area,
                    "map_name": map_name,
                    "scan_time": scan_time,
                    "rows": h5_scalar(h5, f"{map_path}/Sample/Number Of Rows"),
                    "cols": h5_scalar(h5, f"{map_path}/Sample/Number Of Columns"),
                    "point_count": count,
                    "pc_x": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/X-Star"),
                    "pc_y": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/Y-Star"),
                    "pc_z": h5_scalar(h5, f"{map_path}/EBSD/ANG/HEADER/Pattern Center Calibration/Z-Star"),
                    "iq_mean": float(np.nanmean(data["IQ"][:].astype(np.float64))),
                    "ci_mean": float(np.nanmean(data["CI"][:].astype(np.float64))),
                }
            )
    return sorted(rows, key=lambda row: (row["specimen"], row["scan_time"], row["h5_path"]))


def match_h5_to_up2(
    h5_rows: list[dict[str, Any]],
    candidates: list[Up2Candidate],
) -> tuple[list[dict[str, Any]], list[Up2Candidate]]:
    h5_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    up2_by_key: dict[tuple[str, int], list[Up2Candidate]] = defaultdict(list)
    for row in h5_rows:
        h5_by_key[(row["specimen"], int(row["point_count"]))].append(row)
    for candidate in candidates:
        up2_by_key[(candidate.specimen, int(candidate.count))].append(candidate)

    paired_by_h5_path: dict[str, Up2Candidate] = {}
    used_up2: set[str] = set()
    for key in sorted(set(h5_by_key) & set(up2_by_key)):
        h5_list = sorted(h5_by_key[key], key=lambda row: (parse_iso_time(row["scan_time"]), row["h5_path"]))
        up2_list = sorted(up2_by_key[key], key=lambda item: (parse_iso_time(item.modified_time), item.display_name))
        for h5_row, up2 in zip(h5_list, up2_list):
            paired_by_h5_path[h5_row["h5_path"]] = up2
            used_up2.add(str(up2.actual_path.resolve()).lower())

    rows: list[dict[str, Any]] = []
    for row in h5_rows:
        up2 = paired_by_h5_path.get(row["h5_path"])
        if up2 is None:
            rows.append(
                {
                    **row,
                    "match_status": "no_local_up2_candidate",
                    "match_basis": "No local/recycle UP2 with same specimen and pattern count was found.",
                    "up2_display_name": "",
                    "up2_original_path": "",
                    "up2_actual_path": "",
                    "up2_location_status": "",
                    "up2_modified_time": "",
                    "up2_resolution": "",
                    "up2_count": "",
                }
            )
        else:
            rows.append(
                {
                    **row,
                    "match_status": "matched_by_specimen_count_time_order",
                    "match_basis": "Matched by specimen + exact pattern count, then chronological order within that count.",
                    "up2_display_name": up2.display_name,
                    "up2_original_path": up2.original_path,
                    "up2_actual_path": str(up2.actual_path),
                    "up2_location_status": up2.location_status,
                    "up2_modified_time": up2.modified_time,
                    "up2_resolution": up2.resolution,
                    "up2_count": up2.count,
                }
            )

    unmatched = [item for item in candidates if str(item.actual_path.resolve()).lower() not in used_up2]
    return rows, unmatched


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


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


def export_sem_images(h5_path: Path, rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    sem_dir = output_dir / "sem_images"
    sem_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        for row in rows:
            sem = np.asarray(h5[f"{row['h5_path']}/SEM-PRIAS Images/DATA/SEM"][:])
            sem_gray = normalize_gray(sem)
            key = safe_name(f"{row['specimen']}_{row['area']}_{row['map_name']}")
            sem_png = sem_dir / f"{key}_sem.png"
            label = row["up2_display_name"] or "UP2 not found locally"
            fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=220)
            ax.imshow(sem_gray, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{row['specimen']} | {row['area']} | {row['map_name']}\n{label}", fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            fig.savefig(sem_png)
            plt.close(fig)
            exported.append({**row, "sem_png": str(sem_png)})
    return exported


def make_contact_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    n = len(rows)
    cols = 4
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.5, rows_n * 3.5), dpi=180)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, row in zip(axes_arr.ravel(), rows):
        image = plt.imread(row["sem_png"])
        ax.imshow(image)
        up2_name = row["up2_display_name"] if row["up2_display_name"] else "UP2 not found"
        ax.set_title(
            f"{row['specimen']} {row['area']} {row['map_name']}\n{up2_name}",
            fontsize=7,
        )
        ax.axis("off")
    fig.tight_layout(pad=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]], unmatched: list[Up2Candidate]) -> None:
    matched = sum(row["match_status"] == "matched_by_specimen_count_time_order" for row in rows)
    lines = [
        "# H5 Mapping SEM Correspondence",
        "",
        f"- H5 mappings exported: {len(rows)}",
        f"- Mappings matched to a local/recycle UP2 name: {matched}",
        f"- Mappings without local UP2 candidate: {len(rows) - matched}",
        f"- Local/recycle UP2 candidates not assigned to an H5 mapping: {len(unmatched)}",
        "",
        "## Mapping To UP2 Names",
        "",
        "| Specimen | H5 area/map | Count | Scan time | UP2 name | Status |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        up2_name = row["up2_display_name"] or "not found locally"
        lines.append(
            f"| {row['specimen']} | {row['area']} / {row['map_name']} | {row['point_count']} | "
            f"{row['scan_time']} | `{up2_name}` | {row['match_status']} |"
        )

    if unmatched:
        lines.extend(
            [
                "",
                "## Local UP2 Candidates Not Assigned",
                "",
                "| UP2 name | Count | Resolution | Location |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for item in unmatched:
            lines.append(
                f"| `{item.display_name}` | {item.count} | {item.resolution} | {item.location_status} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SEM images for each H5 EBSD mapping and map them to local UP2 file names."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--specimen", default=None, help="Optional specimen filter, e.g. Pt-1.")
    parser.add_argument(
        "--up2-root",
        action="append",
        type=Path,
        default=None,
        help="Directory to scan for UP2 files. Repeatable. Defaults to D:\\EBSD-data and D:\\$RECYCLE.BIN.",
    )
    args = parser.parse_args()

    up2_roots = args.up2_root or DEFAULT_UP2_ROOTS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_rows = collect_h5_maps(args.h5, args.specimen)
    up2_candidates = collect_up2_candidates(up2_roots)
    correspondence_rows, unmatched = match_h5_to_up2(h5_rows, up2_candidates)
    exported_rows = export_sem_images(args.h5, correspondence_rows, args.output_dir)

    write_csv(args.output_dir / "h5_mapping_sem_correspondence.csv", exported_rows)
    write_csv(
        args.output_dir / "local_up2_candidates.csv",
        [
            {
                "display_name": item.display_name,
                "original_path": item.original_path,
                "actual_path": str(item.actual_path),
                "location_status": item.location_status,
                "modified_time": item.modified_time,
                "file_bytes": item.file_bytes,
                "resolution": item.resolution,
                "count": item.count,
                "specimen": item.specimen,
            }
            for item in up2_candidates
        ],
    )
    write_csv(
        args.output_dir / "unmatched_local_up2_candidates.csv",
        [
            {
                "display_name": item.display_name,
                "original_path": item.original_path,
                "actual_path": str(item.actual_path),
                "location_status": item.location_status,
                "modified_time": item.modified_time,
                "resolution": item.resolution,
                "count": item.count,
                "specimen": item.specimen,
            }
            for item in unmatched
        ],
    )
    make_contact_sheet(exported_rows, args.output_dir / "h5_mapping_sem_contact_sheet.png")
    write_markdown(args.output_dir / "h5_mapping_sem_correspondence.md", exported_rows, unmatched)

    print(f"Exported {len(exported_rows)} H5 mapping SEM images to {args.output_dir}")
    print(f"Matched mappings: {sum(row['match_status'] == 'matched_by_specimen_count_time_order' for row in exported_rows)}")
    print(f"Contact sheet: {args.output_dir / 'h5_mapping_sem_contact_sheet.png'}")
    print(f"CSV: {args.output_dir / 'h5_mapping_sem_correspondence.csv'}")


if __name__ == "__main__":
    main()
