from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from skimage import measure

from .ipf import ipf_colors_from_sample_directions


@dataclass
class EbsdMap:
    nrows: int
    ncols: int
    sem_shape: tuple[int, int]
    orientations_sample_to_crystal: np.ndarray
    phase: np.ndarray
    iq: np.ndarray
    ci: np.ndarray
    fit: np.ndarray
    sem_signal: np.ndarray
    valid: np.ndarray
    derived_grain_id: np.ndarray
    ipf_x: np.ndarray
    ipf_y: np.ndarray
    ipf_z: np.ndarray
    phase_metadata: dict[int, dict[str, Any]]
    metadata: dict[str, Any]


def _scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size == 1:
        v = arr.reshape(-1)[0]
        if isinstance(v, (bytes, np.bytes_)):
            return bytes(v).decode("utf-8", "ignore").rstrip("\x00")
        if isinstance(v, np.generic):
            return v.item()
        return v
    out: list[Any] = []
    for v in arr.reshape(-1):
        if isinstance(v, (bytes, np.bytes_)):
            out.append(bytes(v).decode("utf-8", "ignore").rstrip("\x00"))
        elif isinstance(v, np.generic):
            out.append(v.item())
        else:
            out.append(v)
    return out


def _read_phase_metadata(group: h5py.Group) -> dict[int, dict[str, Any]]:
    phases: dict[int, dict[str, Any]] = {}
    phase_root = group.get("EBSD/ANG/HEADER/Phase")
    if phase_root is None:
        return phases
    for phase_key, phase_group in phase_root.items():
        if not isinstance(phase_group, h5py.Group):
            continue
        phase_id = int(phase_key)
        row: dict[str, Any] = {}
        for key, dataset in phase_group.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            if key == "HKL Families":
                families = []
                for item in dataset[()]:
                    families.append({name: _scalar(item[name]) for name in dataset.dtype.names or []})
                row[key] = families
            else:
                row[key] = _scalar(dataset[()])
        phases[phase_id] = row
    return phases


def derive_grain_ids(
    phase_grid: np.ndarray,
    ipf_z: np.ndarray,
    valid_grid: np.ndarray,
    *,
    quantization_levels: int,
    min_area_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive conservative pseudo-grains when EDAX Grain ID is not stored.

    The H5 used here stores phase and orientation matrices but no grain ID. This
    function avoids cross-boundary averaging by connected components in phase +
    quantized IPF-Z color space. It may split one true grain into pieces, which
    is safer than merging neighboring grains for surface-index calculations.
    """
    q = np.floor(np.clip(ipf_z, 0.0, 0.999999) * quantization_levels).astype(np.int32)
    key = phase_grid.astype(np.int64) * (quantization_levels**3)
    key += q[..., 0] * (quantization_levels**2) + q[..., 1] * quantization_levels + q[..., 2]
    labels = np.zeros(phase_grid.shape, dtype=np.int32)
    next_label = 1
    for code in np.unique(key[valid_grid]):
        mask = (key == code) & valid_grid
        cc = measure.label(mask, connectivity=1)
        for region_label in range(1, int(cc.max()) + 1):
            region = cc == region_label
            if np.count_nonzero(region) < min_area_px:
                continue
            labels[region] = next_label
            next_label += 1
    unlabeled = valid_grid & (labels == 0)
    if np.any(unlabeled):
        cc = measure.label(unlabeled, connectivity=1)
        labels[unlabeled] = cc[unlabeled] + next_label - 1
    meta = {
        "source": "derived_phase_plus_quantized_ipf_connected_components",
        "warning": "EDAX Grain ID dataset is absent in this H5 group; these IDs are derived, not vendor grain IDs.",
        "quantization_levels": quantization_levels,
        "min_area_px": min_area_px,
        "grain_count": int(labels.max()),
    }
    return labels, meta


def read_edax_h5_map(
    h5_path: Path,
    h5_group: str,
    *,
    grain_quantization_levels: int,
    min_grain_area_px: int,
) -> EbsdMap:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        group = h5[h5_group]
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        step_x = float(np.asarray(group["Sample/Step X"][()]).reshape(-1)[0])
        step_y = float(np.asarray(group["Sample/Step Y"][()]).reshape(-1)[0])
        grid_type = _scalar(group["Sample/Grid Type"][()])
        sample_tilt = float(np.asarray(group["Sample/Sample Tilt"][()]).reshape(-1)[0])
        pre_tilt = float(np.asarray(group["Sample/Pre Tilt"][()]).reshape(-1)[0])
        sem = np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:])

        data = group["EBSD/ANG/DATA/DATA"]
        orientations = np.asarray(data["Orientations"][:], dtype=np.float64).reshape(-1, 3, 3)
        phase = np.asarray(data["Phase"][:], dtype=np.int16).reshape(nrows, ncols)
        iq = np.asarray(data["IQ"][:], dtype=np.float32).reshape(nrows, ncols)
        ci = np.asarray(data["CI"][:], dtype=np.float32).reshape(nrows, ncols)
        fit = np.asarray(data["Fit"][:], dtype=np.float32).reshape(nrows, ncols)
        sem_signal = np.asarray(data["SEM Signal"][:], dtype=np.float32).reshape(nrows, ncols)
        valid = np.asarray(data["Valid"][:], dtype=bool).reshape(nrows, ncols)
        phase_meta = _read_phase_metadata(group)
        attrs = {key: _scalar(value) for key, value in group.attrs.items()}

    flat_valid = valid.reshape(-1)
    ipf_x = ipf_colors_from_sample_directions(orientations, np.array([1.0, 0.0, 0.0]), flat_valid).reshape(nrows, ncols, 3)
    ipf_y = ipf_colors_from_sample_directions(orientations, np.array([0.0, 1.0, 0.0]), flat_valid).reshape(nrows, ncols, 3)
    ipf_z = ipf_colors_from_sample_directions(orientations, np.array([0.0, 0.0, 1.0]), flat_valid).reshape(nrows, ncols, 3)
    grain_id, grain_meta = derive_grain_ids(
        phase,
        ipf_z,
        valid,
        quantization_levels=grain_quantization_levels,
        min_area_px=min_grain_area_px,
    )

    metadata = {
        "path": str(h5_path),
        "h5_group": h5_group,
        "format": "EDAX edaxh5",
        "vendor_inferred": "EDAX/OIM/APEX H5 layout",
        "map_shape_rows_cols": [nrows, ncols],
        "step_x_um": step_x,
        "step_y_um": step_y,
        "grid_type": grid_type,
        "sample_tilt_deg": sample_tilt,
        "pre_tilt_deg": pre_tilt,
        "sem_prias_shape_rows_cols": list(sem.shape),
        "ang_fields": [
            "Orientations",
            "IQ",
            "CI",
            "Phase",
            "SEM Signal",
            "Fit",
            "PRIAS Bottom Strip",
            "PRIAS Center Square",
            "PRIAS Top Strip",
            "Valid",
            "Custom",
        ],
        "original_grain_id_dataset": "absent",
        "grain_id": grain_meta,
        "orientation_matrix_definition": (
            "Configured/validated as EDAX IPF sample_to_crystal matrix: crystal_direction = G @ sample_direction. "
            "This is validated by regenerating software IPF-Z from H5 Orientations."
        ),
        "group_attrs": attrs,
    }
    return EbsdMap(
        nrows=nrows,
        ncols=ncols,
        sem_shape=tuple(int(x) for x in sem.shape),
        orientations_sample_to_crystal=orientations.astype(np.float64),
        phase=phase,
        iq=iq,
        ci=ci,
        fit=fit,
        sem_signal=sem_signal,
        valid=valid,
        derived_grain_id=grain_id,
        ipf_x=ipf_x,
        ipf_y=ipf_y,
        ipf_z=ipf_z,
        phase_metadata=phase_meta,
        metadata=metadata,
    )

