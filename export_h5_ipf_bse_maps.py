from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, transform


DEFAULT_H5_PATH = Path("ebsd.edaxh5")
DEFAULT_OUTPUT_DIR = Path("outputs") / "h5_ipf_bse_maps"


@dataclass(frozen=True)
class MapSpec:
    key: str
    label: str
    area_group: str
    map_group: str


MAPS = [
    MapSpec(
        key="area1_high",
        label="Area 1 HighR",
        area_group="/20260512/Cu/Area 1",
        map_group="/20260512/Cu/Area 1/OIM Map 1HighR",
    ),
    MapSpec(
        key="area2_high",
        label="Area 2 HighR",
        area_group="/20260512/Cu/Area 2",
        map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
    ),
]
MAPS_BY_KEY = {spec.key: spec for spec in MAPS}


def _scalar(value) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return [_scalar(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image[np.isfinite(image)], [0.5, 99.5])
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def cubic_ipf_z_colors(
    orientations: np.ndarray,
    valid: np.ndarray,
    ci: np.ndarray | None = None,
    *,
    ci_weight: bool = False,
) -> np.ndarray:
    """EDAX/OIM-style cubic IPF-Z colors from H5 orientation matrices.

    EDAX H5 orientation matrices are stored row-major for the OIM IPF color
    convention used here.  Direct multiplication, G @ ND, reproduces the EDAX
    IPF-Z export for Pt-3.  Cubic symmetry is approximated by folding the
    resulting crystal direction to the standard [001]-[101]-[111] sector before
    barycentric coloring.

    EDAX software IPF exports are orientation-color maps, not CI/IQ-weighted
    confidence maps.  Keep ``ci_weight=False`` for software-matched IPF images.
    Enable ``ci_weight`` only for separate quality diagnostics.
    """
    n = orientations.shape[0]
    g = orientations.reshape(n, 3, 3)
    nd_sample = np.array([0.0, 0.0, 1.0])
    directions = np.einsum("nij,j->ni", g, nd_sample)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    folded = np.sort(np.abs(directions), axis=1)
    y = folded[:, 0]
    x = folded[:, 1]
    z = folded[:, 2]
    sector_dirs = np.column_stack([x, y, z])
    sector_dirs /= np.linalg.norm(sector_dirs, axis=1, keepdims=True) + 1e-12

    # Stereographic coordinates for [001], [101], [111] sector vertices.
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        ],
        dtype=np.float64,
    )

    def stereo(v: np.ndarray) -> np.ndarray:
        return np.column_stack([v[:, 0] / (1.0 + v[:, 2]), v[:, 1] / (1.0 + v[:, 2])])

    tri = stereo(vertices)
    pts = stereo(sector_dirs)
    mat = np.array(
        [
            [tri[0, 0], tri[1, 0], tri[2, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1]],
            [1.0, 1.0, 1.0],
        ]
    )
    rhs = np.column_stack([pts[:, 0], pts[:, 1], np.ones(n)])
    bary = np.linalg.solve(mat, rhs.T).T
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum(axis=1, keepdims=True) + 1e-12

    # EDAX/OIM-style convention: [001] red, [101] green, [111] blue.
    rgb = bary[:, [0, 1, 2]]
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))
    rgb /= rgb.max(axis=1, keepdims=True) + 1e-12

    if ci_weight:
        if ci is None:
            raise ValueError("ci must be provided when ci_weight=True")
        ci_values = np.clip(ci.astype(np.float64), 0.0, 1.0)
        rgb = rgb * (0.35 + 0.65 * ci_values[:, None])
    rgb[~valid] = 0.0
    return rgb.astype(np.float32)


def read_map(h5: h5py.File, spec: MapSpec) -> dict[str, object]:
    group = h5[spec.map_group]
    ncols = int(group["Sample/Number Of Columns"][0])
    nrows = int(group["Sample/Number Of Rows"][0])
    step_x = float(group["Sample/Step X"][0])
    step_y = float(group["Sample/Step Y"][0])
    grid_type = str(_scalar(group["Sample/Grid Type"][0]))
    data = group["EBSD/ANG/DATA/DATA"]
    orientations = data["Orientations"][:].astype(np.float64)
    iq = data["IQ"][:].astype(np.float32).reshape(nrows, ncols)
    ci = data["CI"][:].astype(np.float32)
    valid = data["Valid"][:].astype(bool)
    phase = data["Phase"][:].astype(np.int32).reshape(nrows, ncols)
    sem_signal = data["SEM Signal"][:].astype(np.float32).reshape(nrows, ncols)
    prias_center = data["PRIAS Center Square"][:].astype(np.float32).reshape(nrows, ncols)

    ipf = cubic_ipf_z_colors(orientations, valid, ci).reshape(nrows, ncols, 3)
    ci_map = ci.reshape(nrows, ncols)
    valid_map = valid.reshape(nrows, ncols)

    sem_image = np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:])
    fov_image = np.asarray(h5[f"{spec.area_group}/FOVIMAGE"][:])
    return {
        "ncols": ncols,
        "nrows": nrows,
        "step_x": step_x,
        "step_y": step_y,
        "grid_type": grid_type,
        "ipf": ipf,
        "iq": iq,
        "ci": ci_map,
        "valid": valid_map,
        "phase": phase,
        "sem_signal": sem_signal,
        "prias_center": prias_center,
        "sem_image": sem_image,
        "fov_image": fov_image,
    }


def save_outputs(spec: MapSpec, maps: dict[str, object], output_dir: Path) -> dict[str, object]:
    out_dir = output_dir / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)

    ipf = maps["ipf"]
    valid = maps["valid"]
    sem_image = normalize_gray(maps["sem_image"])
    fov_image = normalize_gray(maps["fov_image"])
    sem_signal = normalize_gray(maps["sem_signal"])
    prias_center = normalize_gray(maps["prias_center"])
    iq = normalize_gray(maps["iq"])
    ci = np.clip(maps["ci"], 0.0, 1.0)

    ipf_path = out_dir / f"{spec.key}_ipf_z.png"
    sem_path = out_dir / f"{spec.key}_sem_bse_image.png"
    fov_path = out_dir / f"{spec.key}_fov_bse_image.png"
    sem_signal_path = out_dir / f"{spec.key}_sem_signal_map.png"
    prias_path = out_dir / f"{spec.key}_prias_center_map.png"
    iq_path = out_dir / f"{spec.key}_iq_map.png"
    ci_path = out_dir / f"{spec.key}_ci_map.png"
    montage_path = out_dir / f"{spec.key}_ipf_bse_montage.png"

    plt.imsave(ipf_path, ipf)
    plt.imsave(sem_path, sem_image, cmap="gray")
    plt.imsave(fov_path, fov_image, cmap="gray")
    plt.imsave(sem_signal_path, sem_signal, cmap="gray")
    plt.imsave(prias_path, prias_center, cmap="gray")
    plt.imsave(iq_path, iq, cmap="gray")
    plt.imsave(ci_path, ci, cmap="gray", vmin=0, vmax=1)

    ipf_resized = transform.resize(ipf, sem_image.shape[:2], preserve_range=True, anti_aliasing=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(ipf)
    axes[0, 0].set_title(f"{spec.label} IPF-Z")
    axes[0, 1].imshow(sem_image, cmap="gray")
    axes[0, 1].set_title("SEM/BSE image from SEM-PRIAS Images/DATA/SEM")
    axes[0, 2].imshow(fov_image, cmap="gray")
    axes[0, 2].set_title("Area FOVIMAGE")
    axes[1, 0].imshow(iq, cmap="gray")
    axes[1, 0].set_title("IQ map")
    axes[1, 1].imshow(sem_signal, cmap="gray")
    axes[1, 1].set_title("ANG/DATA SEM Signal")
    axes[1, 2].imshow(sem_image, cmap="gray")
    axes[1, 2].imshow(ipf_resized, alpha=0.45)
    axes[1, 2].set_title("IPF over SEM image (size matched)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(montage_path, dpi=220)
    plt.close(fig)

    return {
        "map": spec.key,
        "label": spec.label,
        "nrows": maps["nrows"],
        "ncols": maps["ncols"],
        "step_x_um": maps["step_x"],
        "step_y_um": maps["step_y"],
        "grid_type": maps["grid_type"],
        "valid_fraction": float(np.mean(valid)),
        "ipf_png": str(ipf_path),
        "sem_bse_png": str(sem_path),
        "fov_bse_png": str(fov_path),
        "sem_signal_png": str(sem_signal_path),
        "prias_center_png": str(prias_path),
        "montage_png": str(montage_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export EBSD IPF-Z, IQ/CI, SEM/BSE, FOV, and montage images from an EDAX H5 file."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="EDAX H5 file path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--map",
        dest="map_keys",
        action="append",
        choices=sorted(MAPS_BY_KEY),
        help="Map key to export. Repeat to export multiple maps. Defaults to all configured maps.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_maps = [MAPS_BY_KEY[key] for key in args.map_keys] if args.map_keys else MAPS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(args.h5, "r") as h5:
        for spec in selected_maps:
            rows.append(save_outputs(spec, read_map(h5, spec), args.out_dir))

    summary_path = args.out_dir / "h5_ipf_bse_maps_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved IPF/BSE maps to {args.out_dir}")
    print(f"Summary: {summary_path}")
    for row in rows:
        print(f"{row['label']}: {row['nrows']}x{row['ncols']} valid={row['valid_fraction']:.3f}")


if __name__ == "__main__":
    main()
