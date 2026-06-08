from __future__ import annotations

import csv
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import h5py
import numpy as np
from skimage import exposure, filters

from visualize_scan_position_pc_correction import (
    H5_PATH,
    MapConfig,
    adjusted_pc_from_scan_position,
    circular_mask,
    detector_directions_with_pc,
    index_to_scan_offset_um,
    read_scan_geometry,
)
from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    preprocess_master_hemisphere,
    preprocess_pattern,
    read_edax_inputs,
    sample_master,
)


MASTER_PATH = Path(
    r"E:\EBSD-projiect\.venv\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
)
OUTPUT_DIR = Path(r"E:\EBSD-projiect\EBSD\outputs\scan_position_pc_correction")


@dataclass(frozen=True)
class PcVariant:
    name: str
    x_sign: float
    y_sign: float
    z_sign: float
    x_scale: float
    y_scale: float
    z_scale: float


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (values - values.mean()) / (values.std() + 1e-8)


def adjusted_pc_with_scales(index: int, geometry, variant: PcVariant) -> tuple[float, float, float]:
    x_um, y_um = index_to_scan_offset_um(index, geometry)
    pcx, pcy, pcz = geometry.pc_edax_map
    dx = variant.x_sign * variant.x_scale * (x_um / 1000.0) / geometry.detector_diameter_mm
    dy = variant.y_sign * variant.y_scale * (y_um / 1000.0) / geometry.detector_diameter_mm
    dz = variant.z_sign * variant.z_scale * (y_um / 1000.0) / geometry.detector_diameter_mm
    return pcx + dx, pcy + dy, pcz + dz


def build_master_samplers():
    upper, lower, _upper_sampler, _lower_sampler = load_master_samplers(MASTER_PATH)
    upper_corr = preprocess_master_hemisphere(upper, "corrected")
    lower_corr = preprocess_master_hemisphere(lower, "corrected")
    upper_band = exposure.rescale_intensity(
        filters.meijering(upper_corr, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    lower_band = exposure.rescale_intensity(
        filters.meijering(lower_corr, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    return {
        "upper_corr": make_master_sampler(upper_corr),
        "lower_corr": make_master_sampler(lower_corr),
        "upper_band": make_master_sampler(upper_band),
        "lower_band": make_master_sampler(lower_band),
    }


def score_pattern(projection, mask: np.ndarray, pc_edax: tuple[float, float, float], samplers, stride: int = 4):
    corr = preprocess_pattern(projection.pattern)
    exp_band = exposure.rescale_intensity(
        filters.meijering(corr, sigmas=range(1, 6), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    )
    stride_mask = np.zeros(mask.shape, dtype=bool)
    stride_mask[::stride, ::stride] = True
    valid = mask & stride_mask
    indices = np.flatnonzero(valid.ravel())

    detector_directions = detector_directions_with_pc(projection, pc_edax)
    g_h5 = projection.orientation_flat.reshape(3, 3)
    crystal_vectors = detector_directions[indices] @ g_h5.T
    crystal_vectors /= np.linalg.norm(crystal_vectors, axis=1, keepdims=True) + 1e-12

    master_corr = sample_master(crystal_vectors, samplers["upper_corr"], samplers["lower_corr"])
    master_band = sample_master(crystal_vectors, samplers["upper_band"], samplers["lower_band"])
    exp_corr = corr.ravel()[indices]
    exp_band_values = exp_band.ravel()[indices]

    intensity_score = float(np.mean(zscore(exp_corr) * zscore(master_corr)))
    band_score = float(np.mean(zscore(exp_band_values) * zscore(master_band)))
    combined = 0.35 * intensity_score + 0.65 * band_score
    return intensity_score, band_score, combined


def score_map(config: MapConfig, variants: list[PcVariant], samplers) -> list[dict[str, object]]:
    geometry = read_scan_geometry(H5_PATH, config.map_group)
    projections = [
        read_edax_inputs(
            EdaxMapInputs(
                h5_path=H5_PATH,
                up2_path=config.up2_path,
                map_group=config.map_group,
                pattern_index=index,
            )
        )
        for index in config.indices
    ]
    raw_circles = [estimate_circular_detector_mask(projection.pattern)[1] for projection in projections]
    fixed_circle = tuple(int(round(v)) for v in np.median(np.array(raw_circles), axis=0))

    rows: list[dict[str, object]] = []
    for index, projection in zip(config.indices, projections):
        mask = circular_mask(projection.pattern.shape, fixed_circle)
        x_um, y_um = index_to_scan_offset_um(index, geometry)
        for variant in variants:
            if variant.name == "fixed_map_pc":
                pc_edax = geometry.pc_edax_map
            else:
                pc_edax = adjusted_pc_with_scales(index, geometry, variant)
            intensity, band, combined = score_pattern(projection, mask, pc_edax, samplers)
            rows.append(
                {
                    "map": config.name,
                    "index": index,
                    "x_um": x_um,
                    "y_um": y_um,
                    "variant": variant.name,
                    "x_sign": variant.x_sign,
                    "y_sign": variant.y_sign,
                    "z_sign": variant.z_sign,
                    "x_scale": variant.x_scale,
                    "y_scale": variant.y_scale,
                    "z_scale": variant.z_scale,
                    "pcx": pc_edax[0],
                    "pcy_edax": pc_edax[1],
                    "pcz": pc_edax[2],
                    "intensity_score": intensity,
                    "band_score": band,
                    "combined_score": combined,
                }
            )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["map"]), str(row["variant"])), []).append(row)

    output = []
    for (map_name, variant), items in groups.items():
        output.append(
            {
                "map": map_name,
                "variant": variant,
                "n": len(items),
                "mean_intensity_score": float(np.mean([r["intensity_score"] for r in items])),
                "mean_band_score": float(np.mean([r["band_score"] for r in items])),
                "mean_combined_score": float(np.mean([r["combined_score"] for r in items])),
            }
        )
    return sorted(output, key=lambda r: (str(r["map"]), -float(r["mean_combined_score"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cos70 = cos(radians(70.0))
    cos62 = cos(radians(70.0 - 8.0))
    cos78 = cos(radians(70.0 + 8.0))
    cos28 = cos(radians(90.0 - 70.0 + 8.0))
    sin70 = sin(radians(70.0))
    sin62 = sin(radians(70.0 - 8.0))
    sin78 = sin(radians(70.0 + 8.0))
    sin28 = sin(radians(90.0 - 70.0 + 8.0))
    variants = [PcVariant("fixed_map_pc", 0, 0, 0, 0, 0, 0)]
    y_scales = [
        ("raw_1", 1.0),
        ("cos70", cos70),
        ("cos_tilt_minus_elev", cos62),
        ("cos_tilt_plus_elev", cos78),
        ("sin70", sin70),
        ("inv_cos70", 1.0 / cos70),
    ]
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for y_name, y_scale in y_scales:
                variants.append(
                    PcVariant(
                        name=f"x{x_sign:+.0f}_y{y_sign:+.0f}_{y_name}",
                        x_sign=x_sign,
                        y_sign=y_sign,
                        z_sign=0.0,
                        x_scale=1.0,
                        y_scale=y_scale,
                        z_scale=0.0,
                    )
                )
    yz_scale_pairs = [
        ("surface_tilt70", cos70, sin70),
        ("surface_tilt_minus_elev62", cos62, sin62),
        ("surface_tilt_plus_elev78", cos78, sin78),
        ("detector_alpha28", cos28, sin28),
    ]
    for y_sign in (-1.0, 1.0):
        for z_sign in (-1.0, 1.0):
            for pair_name, y_scale, z_scale in yz_scale_pairs:
                variants.append(
                    PcVariant(
                        name=f"x-1_y{y_sign:+.0f}_z{z_sign:+.0f}_{pair_name}",
                        x_sign=-1.0,
                        y_sign=y_sign,
                        z_sign=z_sign,
                        x_scale=1.0,
                        y_scale=y_scale,
                        z_scale=z_scale,
                    )
                )

    configs = [
        MapConfig(
            name="Area 1 HighR original selected",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 1_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 1/OIM Map 1HighR",
            indices=(34, 14441, 28676, 29111),
            output_name="unused.png",
        ),
        MapConfig(
            name="Area 2 HighR original selected",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
            indices=(19802, 21230, 22834, 19625),
            output_name="unused.png",
        ),
        MapConfig(
            name="Area 2 HighR representative",
            up2_path=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
            map_group="/20260512/Cu/Area 2/OIM Map 2HighR",
            indices=(0, 177, 14151, 28124, 28301),
            output_name="unused.png",
        ),
    ]

    samplers = build_master_samplers()
    rows: list[dict[str, object]] = []
    for config in configs:
        rows.extend(score_map(config, variants, samplers))

    detailed_path = OUTPUT_DIR / "scan_position_pc_score_details.csv"
    aggregate_path = OUTPUT_DIR / "scan_position_pc_score_summary.csv"
    write_csv(detailed_path, rows)
    write_csv(aggregate_path, aggregate(rows))

    print(f"Saved details: {detailed_path}")
    print(f"Saved summary: {aggregate_path}")
    for row in aggregate(rows)[:15]:
        print(
            f"{row['map']}: {row['variant']} "
            f"combined={row['mean_combined_score']:+.5f} "
            f"band={row['mean_band_score']:+.5f} intensity={row['mean_intensity_score']:+.5f}"
        )


if __name__ == "__main__":
    main()
