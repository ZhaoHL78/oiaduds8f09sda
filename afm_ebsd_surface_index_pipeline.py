from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from afm_ebsd_fusion.coordinates import homography_center_polar_rotation, map_afm_pixels_to_ebsd
from afm_ebsd_fusion.crystal import (
    crystal_normals_from_sample_normals,
    grain_average_orientations,
    nearest_hkl_family,
    reduced_direction,
    sample_normals_from_crystal_normals,
    surface_ipf_rgb,
)
from afm_ebsd_fusion.io_afm import read_height_file, robust_rescale
from afm_ebsd_fusion.io_ebsd import read_edax_h5_map
from afm_ebsd_fusion.normals import compute_afm_normals
from afm_ebsd_fusion.validation import angle_between, compare_ipf_reference, summarize_angles
from afm_ebsd_fusion.visualization import (
    ensure_dir,
    normal_rgb,
    save_boundary_overlay,
    save_ipf_triangle_scatter,
    save_label_map,
    save_mask,
    save_overview,
    save_rgb,
    save_scalar,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_grid_nearest(grid: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    rr = np.clip(np.rint(row).astype(np.int32), 0, grid.shape[0] - 1)
    cc = np.clip(np.rint(col).astype(np.int32), 0, grid.shape[1] - 1)
    return grid[rr, cc]


def sample_grid_linear(grid: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(grid.astype(np.float32), [row, col], order=1, mode="nearest").astype(np.float32)


def make_boundary(grain_id: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros(grain_id.shape, dtype=bool)
    boundary[:, 1:] |= grain_id[:, 1:] != grain_id[:, :-1]
    boundary[:, :-1] |= grain_id[:, 1:] != grain_id[:, :-1]
    boundary[1:, :] |= grain_id[1:, :] != grain_id[:-1, :]
    boundary[:-1, :] |= grain_id[1:, :] != grain_id[:-1, :]
    return boundary & valid


def compute_grain_stats(
    grain_id: np.ndarray,
    phase: np.ndarray,
    valid: np.ndarray,
    slope_deg: np.ndarray,
    hkl_label_index: np.ndarray,
    hkl_angle_deg: np.ndarray,
    hkl_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gid in sorted(int(x) for x in np.unique(grain_id[valid]) if x > 0):
        mask = (grain_id == gid) & valid
        if np.count_nonzero(mask) == 0:
            continue
        phase_vals, phase_counts = np.unique(phase[mask], return_counts=True)
        main_phase = int(phase_vals[np.argmax(phase_counts)])
        row: dict[str, Any] = {
            "grain_id": gid,
            "phase_id": main_phase,
            "valid_pixels": int(np.count_nonzero(mask)),
            "mean_slope_deg": float(np.mean(slope_deg[mask])),
            "median_hkl_angle_deg": float(np.median(hkl_angle_deg[mask])),
        }
        for idx, name in enumerate(hkl_names):
            row[f"fraction_{name}"] = float(np.mean(hkl_label_index[mask] == idx))
        row["fraction_unassigned"] = float(np.mean(hkl_label_index[mask] < 0))
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, report: dict[str, Any]) -> None:
    text = f"""# AFM-EBSD Surface-Normal Crystal Index Test

This run uses AFM as the final spatial reference. EBSD phase/orientation maps are
backward-sampled onto the AFM display-height grid; this does not increase EBSD
spatial resolution.

## Coordinate Chain

- AFM raw height -> `{report['afm']['display_orientation']}` -> AFM display grid.
- AFM display grid is scaled to SEM-display size before applying the registration matrix.
- Registration matrix direction: `{report['registration']['matrix_direction']}`.
- SEM display frame: `{report['registration']['sem_display_orientation']}`.
- SEM display coordinates are converted back to raw SEM row order, then scaled to the EBSD map grid.
- EDAX orientation matrix convention used here:
  `crystal_direction = G_sample_to_crystal @ sample_direction`.

## Validation Status

- IPF-Z reproduction: `{report['validation']['ipf_reference']['passed']}`.
- Round-trip median error: `{report['validation']['roundtrip']['median_deg']:.6g} deg`.
- Flat-area surface-normal vs conventional IPF-Z median angle:
  `{report['validation']['flat_area_surface_vs_ipfz']['median_deg']:.6g} deg`.

## Important Limits

- H5 does not contain a vendor Grain ID dataset. The pipeline generated
  conservative derived grain IDs from phase + quantized IPF connected components.
- Continuous `n_crystal` and surface-normal IPF are the main quantitative result.
  Nearest `{{hkl}}` is only an angle-thresholded auxiliary classification.
- Lateral AFM normal components depend on the configured AFM-display to EBSD-sample
  in-plane frame relation. The configuration records this explicitly; do not change
  it silently between runs.
"""
    path.write_text(text, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    rng = np.random.default_rng(int(config.get("random_seed", 20260723)))
    out_dir = Path(config["output_dir"])
    ensure_dir(out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    registration_report = read_json(Path(config["registration"]["report_path"]))
    matrix = np.asarray(registration_report["matrix_afm_to_sem"], dtype=np.float64)
    afm_raw_um, afm_display_um, afm_meta = read_height_file(
        Path(config["afm"]["path"]),
        height_channel=config["afm"].get("height_channel"),
        height_unit_scale_to_um=float(config["afm"]["height_unit_scale_to_um"]),
        display_orientation=config["afm"].get("display_orientation", "raw"),
    )
    scan_size_um = float(afm_meta.get("scan_size_um", config["afm"].get("scan_size_um", np.nan)))
    if not np.isfinite(scan_size_um):
        scan_size_um = float(config["afm"]["scan_size_um"])

    ebsd = read_edax_h5_map(
        Path(config["ebsd"]["h5_path"]),
        config["ebsd"]["h5_group"],
        grain_quantization_levels=int(config["grain_segmentation"]["ipf_quantization_levels"]),
        min_grain_area_px=int(config["grain_segmentation"]["min_area_px"]),
    )
    ipf_validation = compare_ipf_reference(
        ebsd.ipf_z,
        Path(config["ebsd"]["ipf_reference_path"]),
        config["ebsd"].get("ebsd_grid_to_ipf_reference_orientation", "raw"),
    )
    if not ipf_validation["passed"] and config["validation"].get("halt_on_ipf_validation_failure", True):
        report = {
            "status": "halted_ipf_validation_failed",
            "config_path": str(config_path),
            "afm": afm_meta,
            "ebsd": ebsd.metadata,
            "validation": {"ipf_reference": ipf_validation},
        }
        write_json(out_dir / "diagnostic_report.json", report)
        return report

    sem_shape = tuple(int(x) for x in ebsd.metadata["sem_prias_shape_rows_cols"])
    map_data = map_afm_pixels_to_ebsd(
        afm_display_um.shape,
        sem_shape,
        (ebsd.nrows, ebsd.ncols),
        matrix,
        sem_display_orientation=config["registration"]["sem_display_orientation"],
    )
    valid_spatial = map_data["inside"]
    valid_ebsd = sample_grid_nearest(ebsd.valid, map_data["ebsd_row"], map_data["ebsd_col"]).astype(bool)
    phase_afm = sample_grid_nearest(ebsd.phase, map_data["ebsd_row"], map_data["ebsd_col"]).astype(np.int16)
    grain_afm = sample_grid_nearest(ebsd.derived_grain_id, map_data["ebsd_row"], map_data["ebsd_col"]).astype(np.int32)
    ci_afm = sample_grid_linear(ebsd.ci, map_data["ebsd_row"], map_data["ebsd_col"])
    iq_afm = sample_grid_linear(ebsd.iq, map_data["ebsd_row"], map_data["ebsd_col"])
    fit_afm = sample_grid_linear(ebsd.fit, map_data["ebsd_row"], map_data["ebsd_col"])
    valid_initial = valid_spatial & valid_ebsd & (grain_afm > 0)

    center_xy = ((sem_shape[1] - 1) / 2, (sem_shape[0] - 1) / 2)
    rot2d = homography_center_polar_rotation(matrix, center_xy)
    normals = compute_afm_normals(
        afm_display_um,
        scan_size_um,
        scan_size_um,
        plane_level_enabled=bool(config["afm_preprocessing"]["plane_level"]),
        smooth_sigma_px=float(config["normal_estimation"]["smooth_sigma_px"]),
        local_plane_window_px=int(config["normal_estimation"]["local_plane_window_px"]),
        image_y_to_sample_y=int(config["coordinate_convention"]["image_y_to_sample_y"]),
        afm_to_sample_rotation_2d=rot2d,
    )

    averaged_orientations = grain_average_orientations(
        ebsd.orientations_sample_to_crystal,
        ebsd.derived_grain_id,
        ebsd.valid,
    )
    matrices_afm = averaged_orientations[map_data["nearest_index"].reshape(-1)].reshape(*afm_display_um.shape, 3, 3)
    normals_sample = normals["normals_sample"]
    normals_crystal = crystal_normals_from_sample_normals(matrices_afm, normals_sample)
    normals_sample_roundtrip = sample_normals_from_crystal_normals(matrices_afm, normals_crystal)
    roundtrip_angle = angle_between(normals_sample, normals_sample_roundtrip)

    conventional_z_sample = np.zeros_like(normals_sample, dtype=np.float32)
    conventional_z_sample[..., 2] = 1.0
    conventional_z_crystal = crystal_normals_from_sample_normals(matrices_afm, conventional_z_sample)
    surface_vs_ipfz_angle = angle_between(normals_crystal, conventional_z_crystal)
    flat_mask = valid_initial & (normals["slope_deg"] <= float(config["validation"]["flat_slope_max_deg"]))

    surface_rgb = surface_ipf_rgb(normals_crystal, valid_initial)
    ipf_x_afm = sample_grid_nearest(ebsd.ipf_x, map_data["ebsd_row"], map_data["ebsd_col"])
    ipf_y_afm = sample_grid_nearest(ebsd.ipf_y, map_data["ebsd_row"], map_data["ebsd_col"])
    ipf_z_afm = sample_grid_nearest(ebsd.ipf_z, map_data["ebsd_row"], map_data["ebsd_col"])
    reduced = reduced_direction(normals_crystal)

    hkl_candidates = [tuple(int(v) for v in item) for item in config["hkl_classification"]["candidates"]]
    hkl_names = [f"{{{h}{k}{l}}}" for h, k, l in hkl_candidates]
    best_hkl_idx, hkl_angle, hkl_margin = nearest_hkl_family(normals_crystal, hkl_candidates)
    threshold = float(config["hkl_classification"]["default_threshold_deg"])
    assigned = valid_initial & (hkl_angle <= threshold)
    hkl_label_map = best_hkl_idx.astype(np.int16)
    hkl_label_map[~assigned] = -1
    label_for_plot = hkl_label_map + 1
    label_names = ["unassigned"] + hkl_names

    boundary_ebsd = make_boundary(ebsd.derived_grain_id, ebsd.valid)
    boundary_afm = sample_grid_nearest(boundary_ebsd, map_data["ebsd_row"], map_data["ebsd_col"]).astype(bool)
    boundary_radius = int(config["quality_masks"]["boundary_dilation_px_on_afm"])
    boundary_dilated = ndimage.binary_dilation(boundary_afm, iterations=boundary_radius)
    grain_core = valid_initial & ~boundary_dilated
    afm_quality = np.isfinite(afm_display_um)
    final_valid = valid_initial & afm_quality
    uncertainty = normals["normal_method_disagreement_deg"].astype(np.float32)

    validation = {
        "ipf_reference": ipf_validation,
        "roundtrip": summarize_angles(roundtrip_angle, final_valid),
        "flat_area_surface_vs_ipfz": summarize_angles(surface_vs_ipfz_angle, flat_mask),
        "normal_z_positive_fraction": float(np.mean(normals_sample[..., 2][final_valid] > 0)) if np.any(final_valid) else float("nan"),
        "valid_fraction_of_afm": float(np.mean(final_valid)),
        "grain_core_fraction_of_valid": float(np.count_nonzero(grain_core) / max(np.count_nonzero(final_valid), 1)),
    }

    # Uncertainty and threshold stability: normal-method disagreement plus hkl threshold sweep.
    threshold_rows = []
    for t in config["hkl_classification"]["thresholds_deg"]:
        m = final_valid & (hkl_angle <= float(t))
        threshold_rows.append(
            {
                "threshold_deg": float(t),
                "assigned_fraction_of_valid": float(np.count_nonzero(m) / max(np.count_nonzero(final_valid), 1)),
            }
        )

    paths = {
        "raw_height": fig_dir / "01_afm_raw_height_um.png",
        "leveled_height": fig_dir / "02_afm_leveled_height_um.png",
        "smoothed_height": fig_dir / "03_afm_smoothed_height_um.png",
        "slope": fig_dir / "04_afm_slope_deg.png",
        "aspect": fig_dir / "05_afm_aspect_deg.png",
        "sample_normal": fig_dir / "06_afm_sample_normal_map.png",
        "grain": fig_dir / "07_ebsd_derived_grain_map_on_afm.png",
        "phase": fig_dir / "08_ebsd_phase_map_on_afm.png",
        "ci": fig_dir / "09_ebsd_ci_on_afm.png",
        "fit": fig_dir / "09b_ebsd_fit_on_afm.png",
        "ipf_x": fig_dir / "10a_ebsd_ipf_x_on_afm.png",
        "ipf_y": fig_dir / "10b_ebsd_ipf_y_on_afm.png",
        "ipf_z": fig_dir / "10c_ebsd_ipf_z_on_afm.png",
        "surface_ipf": fig_dir / "11_afm_surface_normal_ipf.png",
        "angle": fig_dir / "12_surface_normal_vs_ipf_z_angle_deg.png",
        "hkl": fig_dir / "13_nearest_hkl_classification.png",
        "hkl_angle": fig_dir / "14_nearest_hkl_angle_deg.png",
        "unassigned": fig_dir / "15_high_index_unassigned_mask.png",
        "boundary": fig_dir / "16_grain_boundary_mask.png",
        "core": fig_dir / "17_grain_core_mask.png",
        "valid": fig_dir / "18_final_valid_mask.png",
        "uncertainty": fig_dir / "19_normal_uncertainty_deg.png",
        "triangle": fig_dir / "21_ipf_triangle_surface_direction_scatter.png",
        "overlay": fig_dir / "23_afm_ebsd_boundary_surface_index_overlay.png",
        "overview": fig_dir / "00_afm_ebsd_surface_index_overview.png",
    }
    height_vmin, height_vmax = np.nanpercentile(afm_display_um, [1, 99])
    save_scalar(paths["raw_height"], afm_display_um, "Raw AFM height in display frame", "viridis", "height (um)", vmin=height_vmin, vmax=height_vmax)
    save_scalar(paths["leveled_height"], normals["height_leveled_um"], "AFM height after global plane leveling", "viridis", "height (um)")
    save_scalar(paths["smoothed_height"], normals["height_smoothed_um"], "AFM height used for normal estimation", "viridis", "height (um)")
    save_scalar(paths["slope"], normals["slope_deg"], "AFM local slope", "magma", "deg", vmin=0, vmax=float(np.nanpercentile(normals["slope_deg"], 99)))
    save_scalar(paths["aspect"], normals["aspect_deg"], "AFM local normal azimuth/aspect", "twilight", "deg", vmin=-180, vmax=180)
    normal_map_rgb = normal_rgb(normals_sample, float(config["visualization"]["normal_tilt_ref_deg"]))
    save_rgb(paths["sample_normal"], normal_map_rgb, "AFM sample-frame normal direction")
    save_scalar(paths["grain"], grain_afm, "Derived EBSD grain map on AFM grid", "tab20", "derived grain id")
    save_scalar(paths["phase"], phase_afm, "EBSD phase map on AFM grid", "tab10", "phase id")
    save_scalar(paths["ci"], ci_afm, "EBSD CI on AFM grid", "viridis", "CI")
    save_scalar(paths["fit"], fit_afm, "EBSD Fit on AFM grid", "magma", "Fit")
    save_rgb(paths["ipf_x"], ipf_x_afm, "Conventional EBSD IPF-X mapped to AFM", final_valid)
    save_rgb(paths["ipf_y"], ipf_y_afm, "Conventional EBSD IPF-Y mapped to AFM", final_valid)
    save_rgb(paths["ipf_z"], ipf_z_afm, "Conventional EBSD IPF-Z mapped to AFM", final_valid)
    save_rgb(paths["surface_ipf"], surface_rgb, "AFM surface-normal IPF on AFM grid", final_valid)
    save_scalar(paths["angle"], surface_vs_ipfz_angle, "Angle: AFM surface-normal direction vs conventional IPF-Z direction", "magma", "deg")
    save_label_map(paths["hkl"], label_for_plot, f"Nearest {{hkl}} classification, threshold <= {threshold:g} deg", label_names, final_valid, unassigned_value=0)
    save_scalar(paths["hkl_angle"], hkl_angle, "Angle to nearest low-index {hkl}", "magma", "deg", vmin=0, vmax=float(config["hkl_classification"]["angle_plot_max_deg"]))
    save_mask(paths["unassigned"], final_valid & ~assigned, "High-index / unassigned mask")
    save_mask(paths["boundary"], boundary_dilated, "Mapped and dilated EBSD grain-boundary mask")
    save_mask(paths["core"], grain_core, "Grain-core mask")
    save_mask(paths["valid"], final_valid, "Final valid AFM+EBSD mask")
    save_scalar(paths["uncertainty"], uncertainty, "Normal-direction uncertainty from method disagreement", "magma", "deg", vmin=0, vmax=float(np.nanpercentile(uncertainty, 99)))
    save_ipf_triangle_scatter(paths["triangle"], reduced, final_valid, surface_rgb, "Surface-normal directions in cubic IPF triangle")
    save_boundary_overlay(paths["overlay"], normals["height_leveled_um"], boundary_dilated, surface_rgb, "AFM height + EBSD boundary + surface-normal IPF")
    save_overview(
        paths["overview"],
        [
            ("AFM leveled height", robust_rescale(normals["height_leveled_um"]), "gray"),
            ("AFM normal", normal_map_rgb, "rgb"),
            ("EBSD IPF-Z on AFM", ipf_z_afm, "rgb"),
            ("Surface-normal IPF", surface_rgb, "rgb"),
            ("Nearest {hkl}", label_for_plot, "tab10"),
            ("Boundary/core", boundary_dilated.astype(float), "gray"),
        ],
    )

    h5_path = data_dir / "afm_ebsd_surface_index_pixel_data.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["description"] = "AFM-reference EBSD surface-normal crystal-index data"
        h5.attrs["orientation_definition"] = "crystal_direction = G_sample_to_crystal @ sample_direction"
        for key in ["height_raw_um", "height_leveled_um", "height_smoothed_um", "dz_dx", "dz_drow", "slope_deg", "aspect_deg"]:
            h5.create_dataset(key, data=normals[key], compression="gzip", compression_opts=4)
        h5.create_dataset("normal_sample", data=normals_sample, compression="gzip", compression_opts=4)
        h5.create_dataset("normal_crystal", data=normals_crystal, compression="gzip", compression_opts=4)
        h5.create_dataset("reduced_ipf_direction", data=reduced, compression="gzip", compression_opts=4)
        h5.create_dataset("nearest_hkl_index", data=hkl_label_map, compression="gzip", compression_opts=4)
        h5.create_dataset("nearest_hkl_angle_deg", data=hkl_angle, compression="gzip", compression_opts=4)
        h5.create_dataset("hkl_margin_deg", data=hkl_margin, compression="gzip", compression_opts=4)
        h5.create_dataset("surface_ipf_rgb", data=surface_rgb, compression="gzip", compression_opts=4)
        h5.create_dataset("phase_id", data=phase_afm, compression="gzip", compression_opts=4)
        h5.create_dataset("grain_id", data=grain_afm, compression="gzip", compression_opts=4)
        h5.create_dataset("CI", data=ci_afm, compression="gzip", compression_opts=4)
        h5.create_dataset("IQ", data=iq_afm, compression="gzip", compression_opts=4)
        h5.create_dataset("Fit", data=fit_afm, compression="gzip", compression_opts=4)
        h5.create_dataset("ebsd_source_row", data=map_data["ebsd_row"], compression="gzip", compression_opts=4)
        h5.create_dataset("ebsd_source_col", data=map_data["ebsd_col"], compression="gzip", compression_opts=4)
        h5.create_dataset("valid_mask", data=final_valid, compression="gzip", compression_opts=4)
        h5.create_dataset("grain_core_mask", data=grain_core, compression="gzip", compression_opts=4)
        h5.create_dataset("uncertainty_deg", data=uncertainty, compression="gzip", compression_opts=4)

    npz_path = data_dir / "afm_ebsd_surface_index_preview_arrays.npz"
    np.savez_compressed(
        npz_path,
        height_leveled_um=normals["height_leveled_um"],
        normal_sample=normals_sample,
        normal_crystal=normals_crystal,
        surface_ipf_rgb=surface_rgb,
        nearest_hkl_index=hkl_label_map,
        nearest_hkl_angle_deg=hkl_angle,
        valid_mask=final_valid,
    )

    grain_stats = compute_grain_stats(grain_afm, phase_afm, final_valid, normals["slope_deg"], hkl_label_map, hkl_angle, hkl_names)
    save_csv(data_dir / "grain_statistics.csv", grain_stats)
    save_csv(data_dir / "hkl_threshold_stability.csv", threshold_rows)

    report = {
        "status": "completed",
        "config_path": str(config_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "afm": afm_meta,
        "ebsd": ebsd.metadata,
        "phase_metadata": ebsd.phase_metadata,
        "registration": {
            "report_path": config["registration"]["report_path"],
            "matrix_direction": registration_report["matrix_direction"],
            "sem_display_orientation": config["registration"]["sem_display_orientation"],
            "matrix_afm_display_resized_to_sem_display": matrix.tolist(),
            "control_point_metrics": registration_report.get("control_point_metrics", {}),
            "afm_to_sample_inplane_rotation_2d_from_homography_polar_center": rot2d.tolist(),
        },
        "coordinate_convention": config["coordinate_convention"],
        "validation": validation,
        "hkl_candidates": hkl_names,
        "threshold_stability": threshold_rows,
        "outputs": {
            "figures": {key: str(value.resolve()) for key, value in paths.items()},
            "pixel_hdf5": str(h5_path.resolve()),
            "preview_npz": str(npz_path.resolve()),
            "grain_statistics_csv": str((data_dir / "grain_statistics.csv").resolve()),
            "hkl_threshold_stability_csv": str((data_dir / "hkl_threshold_stability.csv").resolve()),
        },
        "limits": [
            "AFM is the final spatial reference; EBSD is only resampled onto this grid.",
            "H5 lacks EDAX Grain ID; derived grain IDs are conservative connected components, not vendor grains.",
            "Nearest {hkl} is thresholded auxiliary classification; continuous n_crystal is the main result.",
        ],
    }
    write_json(out_dir / "diagnostic_report.json", report)
    write_readme(out_dir / "README.md", report)
    print(json.dumps({"status": report["status"], "output_dir": str(out_dir), "validation": validation}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFM-reference EBSD surface-normal crystal-index fusion pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args().config)


if __name__ == "__main__":
    main()

