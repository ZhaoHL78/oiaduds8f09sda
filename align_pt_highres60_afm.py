from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from afm_ebsd_surface_index import (
    height_to_normals,
    normal_direction_rgb,
    save_normalmap_with_legend,
    save_scalar_image,
)
from align_pt_afm_sem_ipf import (
    Candidate,
    afm_feature_images,
    find_best_alignment,
    make_lightglue_models,
    read_afm_channels,
    robust_rescale,
    save_match_figure,
    sem_feature_images,
    warp_to_sem,
    write_candidates,
)
from export_pt_highres_data_overview import read_ipf_map, read_reference_ipf
from pt_highres_30deg_lightglue_calibration import DEFAULT_H5, build_map_specs, normalize_gray


DEFAULT_AFM = Path(r"D:\EBSD project\3d数据\pt-afm\Pt-2high resolution.ibw")
DEFAULT_EDAX_IPF = Path(r"E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution\60.bmp")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_highres60_afm_alignment"
DEFAULT_ANGLE = 60


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_ebsd60(args: argparse.Namespace) -> dict[str, Any]:
    specs = {spec.angle_deg: spec for spec in build_map_specs()}
    spec = specs[args.angle]
    with h5py.File(args.h5, "r") as h5:
        group = h5[spec.h5_group]
        sem = normalize_gray(np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
        nrows = int(np.asarray(group["Sample/Number Of Rows"][()]).reshape(-1)[0])
        ncols = int(np.asarray(group["Sample/Number Of Columns"][()]).reshape(-1)[0])
        step_x_um = float(np.asarray(group["Sample/Step X"][()]).reshape(-1)[0])
        step_y_um = float(np.asarray(group["Sample/Step Y"][()]).reshape(-1)[0])
        ipf_h5 = read_ipf_map(group, sem.shape)
    ipf_ref = read_reference_ipf(args.edax_ipf) if args.edax_ipf.exists() else ipf_h5
    ipf_sem = cv2.resize(ipf_ref, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return {
        "spec": spec,
        "sem": sem,
        "ipf_sem": ipf_sem,
        "ipf_h5": ipf_h5,
        "nrows": nrows,
        "ncols": ncols,
        "step_x_um": step_x_um,
        "step_y_um": step_y_um,
        "physical_width_um": ncols * step_x_um,
        "physical_height_um": nrows * step_y_um,
    }


def choose_match_scales(scan_size_um: float, ebsd_width_um: float, sem_width_px: int, afm_width_px: int) -> tuple[list[float], float]:
    expected_sem_width_px = sem_width_px * scan_size_um / max(ebsd_width_um, 1e-6)
    expected_scale = expected_sem_width_px / max(afm_width_px, 1)
    multipliers = [0.55, 0.75, 0.90, 1.0, 1.15, 1.35, 1.65, 2.10]
    scales = [expected_scale * value for value in multipliers]
    scales.extend([0.16, 0.22, 0.30, 0.40, 0.55])
    unique = sorted({round(float(np.clip(scale, 0.10, 1.0)), 4) for scale in scales})
    return unique, float(expected_scale)


def scaled_afm_feature_images(
    channels: dict[str, np.ndarray],
    scales: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    base = afm_feature_images(channels)
    keep = (
        "HeightRetrace_hp",
        "HeightRetrace_sobel",
        "AmplitudeRetrace_hp",
        "AmplitudeRetrace_sobel",
        "ZSensorRetrace_hp",
        "PhaseRetrace_hp",
    )
    output: dict[str, np.ndarray] = {}
    scale_by_name: dict[str, float] = {}
    for name in keep:
        if name not in base:
            continue
        image = base[name]
        height, width = image.shape
        for scale in scales:
            out_w = max(64, int(round(width * scale)))
            out_h = max(64, int(round(height * scale)))
            resized = cv2.resize(
                image,
                (out_w, out_h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            ).astype(np.float32)
            key = f"{name}_scale{scale:.4f}"
            output[key] = resized
            scale_by_name[key] = scale
    return output, scale_by_name


def convert_candidate_to_original_afm(candidate: Candidate, scale: float) -> Candidate:
    converted = Candidate(
        afm_name=candidate.afm_name,
        sem_name=candidate.sem_name,
        matches=candidate.matches,
        inliers=candidate.inliers,
        inlier_ratio=candidate.inlier_ratio,
        rmse=candidate.rmse,
        affine_afm_to_sem=candidate.affine_afm_to_sem.copy(),
        inlier_mask=candidate.inlier_mask.copy(),
        key_sem=candidate.key_sem.copy(),
        key_afm=candidate.key_afm.copy(),
    )
    converted.affine_afm_to_sem[:, :2] *= scale
    return converted


def rank_candidates_with_scale(candidates: list[Candidate], expected_scale: float) -> list[Candidate]:
    def score(candidate: Candidate) -> tuple[int, float, int, float, float]:
        geom_scale = math.sqrt(max(abs(candidate.det), 1e-12))
        scale_error = abs(math.log(max(geom_scale, 1e-9) / max(expected_scale, 1e-9)))
        plausible = int(0.35 * expected_scale <= geom_scale <= 2.80 * expected_scale and candidate.det > 0)
        return (plausible, candidate.inliers, candidate.inlier_ratio, -candidate.rmse, -scale_error)

    return sorted(candidates, key=score, reverse=True)


def save_rgb(path: Path, image: np.ndarray, title: str | None = None, mask: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220, constrained_layout=True)
    if mask is None:
        ax.imshow(np.clip(image, 0.0, 1.0))
    else:
        ax.imshow(np.clip(image, 0.0, 1.0), alpha=np.clip(mask, 0.0, 1.0))
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_gray(path: Path, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220, constrained_layout=True)
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def warp_rgb_to_sem(image: np.ndarray, affine_afm_to_sem: np.ndarray, sem_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = sem_shape
    warped = cv2.warpAffine(
        image.astype(np.float32),
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones(image.shape[:2], dtype=np.float32),
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(warped, 0.0, 1.0).astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32)


def overlay_gray_with_rgba(gray: np.ndarray, rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    base = np.dstack([gray] * 3)
    a = alpha * mask[..., None]
    return np.clip(base * (1.0 - a) + rgb * a, 0.0, 1.0).astype(np.float32)


def save_candidate_overlay_previews(
    path: Path,
    sem: np.ndarray,
    normal_rgb: np.ndarray,
    candidates: list[Candidate],
    max_candidates: int = 8,
) -> None:
    count = min(max_candidates, len(candidates))
    cols = min(4, count)
    rows_n = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.6 * rows_n), dpi=180, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, candidate in zip(axes_arr, candidates[:count]):
        warped, mask = warp_rgb_to_sem(normal_rgb, candidate.affine_afm_to_sem, sem.shape)
        image = overlay_gray_with_rgba(sem, warped, mask, 0.72)
        ax.imshow(image)
        ax.set_title(
            f"inliers={candidate.inliers}/{candidate.matches}, rmse={candidate.rmse:.2f}\n"
            f"{candidate.afm_name[:30]}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes_arr[count:]:
        ax.axis("off")
    fig.suptitle("Top LightGlue AFM -> 60 deg EBSD SEM candidate overlays", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_alignment_outputs(
    args: argparse.Namespace,
    ebsd: dict[str, Any],
    channels: dict[str, np.ndarray],
    afm_meta: dict[str, Any],
    sem_features: dict[str, np.ndarray],
    afm_features_scaled: dict[str, np.ndarray],
    candidates: list[Candidate],
    expected_scale: float,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = candidates[0]
    sem = ebsd["sem"]
    ipf_sem = ebsd["ipf_sem"]
    height_channel = args.height_channel
    if height_channel not in channels:
        raise KeyError(f"AFM channel {height_channel!r} not found. Available: {list(channels)}")
    height_um = channels[height_channel].astype(np.float32) * 1e6
    amp = channels.get("AmplitudeRetrace", next(iter(channels.values())))
    height_norm_warped, afm_mask = warp_to_sem(height_um, best.affine_afm_to_sem, sem.shape)
    amp_warped, _ = warp_to_sem(amp, best.affine_afm_to_sem, sem.shape)

    normal_data = height_to_normals(
        height_um=height_um,
        scan_size_um=float(afm_meta["scan_size_um"]),
        affine_afm_to_sem=best.affine_afm_to_sem,
        smooth_sigma_px=args.normal_smooth_sigma_px,
        level=not args.no_plane_level,
    )
    normal_rgb = normal_direction_rgb(normal_data["normals_sample"], args.tilt_color_ref_deg)
    normal_rgb_warped, normal_mask = warp_rgb_to_sem(normal_rgb, best.affine_afm_to_sem, sem.shape)

    paths = {
        "ebsd60_sem": args.output_dir / "ebsd60_sem_h5.png",
        "ebsd60_ipf": args.output_dir / "ebsd60_ipf_edax_style_sem_frame.png",
        "afm_height": args.output_dir / "afm_height_nm.png",
        "afm_amplitude": args.output_dir / "afm_amplitude.png",
        "afm_normalmap": args.output_dir / "afm_scharr_normalmap.png",
        "afm_normalmap_colorbar": args.output_dir / "afm_scharr_normalmap_with_colorbar.png",
        "afm_normal_tilt": args.output_dir / "afm_normal_tilt_deg.png",
        "afm_normal_azimuth": args.output_dir / "afm_normal_azimuth_deg.png",
        "afm_scharr_dx": args.output_dir / "afm_scharr_dz_dx_um_per_um.png",
        "afm_scharr_dy": args.output_dir / "afm_scharr_dz_dy_um_per_um.png",
        "lightglue_matches": args.output_dir / "lightglue_afm_ebsd60_matches.png",
        "candidate_table": args.output_dir / "lightglue_afm_ebsd60_candidates.csv",
        "afm_height_warped": args.output_dir / "afm_height_warped_to_ebsd60_sem.png",
        "afm_amplitude_warped": args.output_dir / "afm_amplitude_warped_to_ebsd60_sem.png",
        "afm_normalmap_warped": args.output_dir / "afm_normalmap_warped_to_ebsd60_sem.png",
        "afm_height_on_sem": args.output_dir / "afm_height_overlay_on_ebsd60_sem.png",
        "afm_normal_on_sem": args.output_dir / "afm_normalmap_overlay_on_ebsd60_sem.png",
        "afm_normal_on_ipf": args.output_dir / "afm_normalmap_overlay_on_ebsd60_ipf.png",
        "candidate_previews": args.output_dir / "candidate_normalmap_overlay_previews.png",
        "data_npz": args.output_dir / "pt_highres60_afm_alignment_data.npz",
        "metadata": args.output_dir / "pt_highres60_afm_alignment_metadata.json",
    }

    save_gray(paths["ebsd60_sem"], sem, "60 deg EBSD SEM from H5")
    save_rgb(paths["ebsd60_ipf"], ipf_sem, "60 deg EDAX-style IPF-Z in SEM frame")
    save_scalar_image(paths["afm_height"], normal_data["height_um"] * 1000.0, "AFM height", "viridis", "height (nm)")
    save_gray(paths["afm_amplitude"], robust_rescale(amp), "AFM AmplitudeRetrace", cmap="magma")
    plt.imsave(paths["afm_normalmap"], normal_rgb)
    save_normalmap_with_legend(
        paths["afm_normalmap_colorbar"],
        normal_rgb,
        "AFM Scharr normalmap in EBSD top-view frame",
        args.tilt_color_ref_deg,
    )
    save_scalar_image(
        paths["afm_normal_tilt"],
        normal_data["tilt_deg"],
        "AFM normal tilt from sample Z",
        "magma",
        "tilt (deg)",
        vmin=0.0,
        vmax=float(np.nanpercentile(normal_data["tilt_deg"], 99.0)),
    )
    save_scalar_image(
        paths["afm_normal_azimuth"],
        normal_data["azimuth_deg"],
        "AFM normal azimuth in EBSD top-view frame",
        "twilight",
        "azimuth (deg)",
        vmin=-180.0,
        vmax=180.0,
    )
    save_scalar_image(
        paths["afm_scharr_dx"],
        normal_data["scharr_dz_dcol"],
        "Scharr dz/dx from AFM depthmap",
        "coolwarm",
        "dz/dx (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_dcol"], 99.0)),
    )
    save_scalar_image(
        paths["afm_scharr_dy"],
        normal_data["scharr_dz_drow"],
        "Scharr dz/dy from AFM depthmap",
        "coolwarm",
        "dz/dy (um/um)",
        vmin=float(np.nanpercentile(normal_data["scharr_dz_drow"], 1.0)),
        vmax=float(np.nanpercentile(normal_data["scharr_dz_drow"], 99.0)),
    )
    save_match_figure(paths["lightglue_matches"], sem_features[best.sem_name], afm_features_scaled[best.afm_name], best)
    write_candidates(paths["candidate_table"], candidates)
    save_gray(paths["afm_height_warped"], height_norm_warped, "AFM height warped to 60 deg EBSD SEM", cmap="viridis")
    save_gray(paths["afm_amplitude_warped"], amp_warped, "AFM amplitude warped to 60 deg EBSD SEM", cmap="magma")
    save_rgb(paths["afm_normalmap_warped"], normal_rgb_warped, "AFM normalmap warped to 60 deg EBSD SEM", normal_mask)
    save_rgb(paths["afm_height_on_sem"], overlay_gray_with_rgba(sem, np.dstack([height_norm_warped] * 3), afm_mask, 0.55), "AFM height overlay on 60 deg EBSD SEM")
    save_rgb(paths["afm_normal_on_sem"], overlay_gray_with_rgba(sem, normal_rgb_warped, normal_mask, 0.70), "AFM normalmap overlay on 60 deg EBSD SEM")
    save_rgb(paths["afm_normal_on_ipf"], overlay_gray_with_rgba(ipf_sem.mean(axis=2), normal_rgb_warped, normal_mask, 0.70), "AFM normalmap overlay on 60 deg IPF")
    save_candidate_overlay_previews(paths["candidate_previews"], sem, normal_rgb, candidates)

    np.savez_compressed(
        paths["data_npz"],
        affine_afm_to_ebsd60_sem=best.affine_afm_to_sem,
        height_um=normal_data["height_um"],
        height_smooth_um=normal_data["height_smooth_um"],
        normals_afm=normal_data["normals_afm"],
        normals_sample=normal_data["normals_sample"],
        normal_rgb=normal_rgb,
        normal_rgb_warped=normal_rgb_warped,
        afm_mask_in_ebsd60_sem=afm_mask,
        scharr_dz_dcol=normal_data["scharr_dz_dcol"],
        scharr_dz_drow=normal_data["scharr_dz_drow"],
        ebsd60_sem=sem,
        ebsd60_ipf_sem=ipf_sem,
    )

    metadata = {
        "afm": str(args.afm),
        "afm_metadata": afm_meta,
        "h5": str(args.h5),
        "h5_group": ebsd["spec"].h5_group,
        "angle_deg": args.angle,
        "edax_ipf_reference": str(args.edax_ipf),
        "ebsd_grid": {"rows": ebsd["nrows"], "cols": ebsd["ncols"]},
        "ebsd_step_um": {"x": ebsd["step_x_um"], "y": ebsd["step_y_um"]},
        "ebsd_physical_size_um": {"width": ebsd["physical_width_um"], "height": ebsd["physical_height_um"]},
        "expected_afm_to_sem_scale": expected_scale,
        "best_alignment": {
            "afm_feature": best.afm_name,
            "sem_feature": best.sem_name,
            "matches": best.matches,
            "inliers": best.inliers,
            "inlier_ratio": best.inlier_ratio,
            "rmse_px": best.rmse,
            "det": best.det,
            "sx": best.sx,
            "sy": best.sy,
            "affine_afm_to_ebsd60_sem_2x3": best.affine_afm_to_sem.tolist(),
        },
        "normalmap": {
            "method": "Plane-level AFM height, Scharr dz/dx and dz/dy, normal=(-dz/dx,-dz/dy,1), then rotate XY by AFM->SEM polar rotation.",
            "height_channel": height_channel,
            "smooth_sigma_px": args.normal_smooth_sigma_px,
            "tilt_color_ref_deg": args.tilt_color_ref_deg,
        },
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    ebsd = read_ebsd60(args)
    channels, afm_meta = read_afm_channels(args.afm)
    scan_size_um = float(afm_meta["scan_size_um"])
    first_channel = next(iter(channels.values()))
    scales, expected_scale = choose_match_scales(
        scan_size_um=scan_size_um,
        ebsd_width_um=ebsd["physical_width_um"],
        sem_width_px=ebsd["sem"].shape[1],
        afm_width_px=first_channel.shape[1],
    )
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    extractor, matcher = make_lightglue_models(device, args.max_keypoints)
    sem_features = sem_feature_images(ebsd["sem"])
    sem_features = {key: sem_features[key] for key in ("sem_norm", "sem_hp", "sem_sobel", "sem_canny") if key in sem_features}
    afm_features_scaled, scale_by_name = scaled_afm_feature_images(channels, scales)

    raw_candidates = find_best_alignment(
        sem_features=sem_features,
        afm_features=afm_features_scaled,
        extractor=extractor,
        matcher=matcher,
        device=device,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
    )
    candidates = [
        convert_candidate_to_original_afm(candidate, scale_by_name[candidate.afm_name])
        for candidate in raw_candidates
    ]
    candidates = rank_candidates_with_scale(candidates, expected_scale)
    if not candidates:
        raise RuntimeError("LightGlue/SuperPoint did not find a usable AFM -> EBSD60 SEM affine alignment")
    metadata = save_alignment_outputs(
        args=args,
        ebsd=ebsd,
        channels=channels,
        afm_meta=afm_meta,
        sem_features=sem_features,
        afm_features_scaled=afm_features_scaled,
        candidates=candidates,
        expected_scale=expected_scale,
    )
    best = candidates[0]
    print(f"Saved Pt high-resolution 60 deg AFM alignment to {args.output_dir}")
    print(
        f"Best LightGlue/SuperPoint: {best.afm_name} -> {best.sem_name}, "
        f"{best.inliers}/{best.matches} inliers, RMSE={best.rmse:.2f}px, "
        f"sx={best.sx:.4f}, sy={best.sy:.4f}, expected={expected_scale:.4f}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align Pt-2 high-resolution AFM to the 60-degree Pt high-resolution EBSD SEM/IPF frame and export AFM Scharr normalmap."
    )
    parser.add_argument("--afm", type=Path, default=DEFAULT_AFM)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--edax-ipf", type=Path, default=DEFAULT_EDAX_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-channel", default="HeightRetrace")
    parser.add_argument("--normal-smooth-sigma-px", type=float, default=1.2)
    parser.add_argument("--tilt-color-ref-deg", type=float, default=12.0)
    parser.add_argument("--no-plane-level", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=7.0)
    parser.add_argument("--cpu", action="store_true", help="Force LightGlue/SuperPoint to run on CPU.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
