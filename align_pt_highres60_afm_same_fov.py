from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from afm_ebsd_surface_index import height_to_normals, normal_direction_rgb, save_normalmap_with_legend, save_scalar_image
from align_pt_afm_sem_ipf import read_afm_channels, robust_rescale
from align_pt_highres60_afm import overlay_gray_with_rgba, save_gray, save_rgb, warp_rgb_to_sem
from export_pt_highres_data_overview import read_reference_ipf
from pt_highres_30deg_lightglue_calibration import DEFAULT_H5, build_map_specs, normalize_gray


DEFAULT_AFM = Path(r"D:\EBSD project\3d数据\pt-afm\Pt-2high resolution.ibw")
DEFAULT_EDAX_IPF = Path(r"E:\ZHL\20251209Pt-EBSD MAP\pt-high resolution\60.bmp")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_highres60_afm_same_fov_alignment"
DEFAULT_ANGLE = 60


def orient_image(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "raw":
        return image
    if orientation == "rot90":
        return np.rot90(image, 1)
    if orientation == "rot180":
        return np.rot90(image, 2)
    if orientation == "rot270":
        return np.rot90(image, 3)
    if orientation == "flipud":
        return np.flipud(image)
    if orientation == "fliplr":
        return np.fliplr(image)
    if orientation == "transpose":
        return image.T
    raise ValueError(f"Unsupported AFM orientation: {orientation}")


def read_sem_and_ipf(h5_path: Path, angle: int, edax_ipf_path: Path) -> dict[str, Any]:
    spec = {item.angle_deg: item for item in build_map_specs()}[angle]
    with h5py.File(h5_path, "r") as h5:
        group = h5[spec.h5_group]
        sem_raw = normalize_gray(np.asarray(group["SEM-PRIAS Images/DATA/SEM"][:], dtype=np.float32))
    ipf = read_reference_ipf(edax_ipf_path)
    ipf_sem = cv2.resize(ipf, (sem_raw.shape[1], sem_raw.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return {
        "spec": spec,
        "sem_raw": sem_raw,
        "sem_ipf_frame": np.flipud(sem_raw),
        "ipf_sem": ipf_sem,
    }


def centered_affine_same_fov(
    shape: tuple[int, int],
    rotation_deg: float,
    stretch_x: float,
    stretch_y: float,
    translate_x_px: float,
    translate_y_px: float,
) -> np.ndarray:
    height, width = shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    theta = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    # The anisotropic stretch term absorbs the one-axis SEM tilt correction.
    linear = rotation @ np.diag([stretch_x, stretch_y])
    offset = center + np.array([translate_x_px, translate_y_px], dtype=np.float64) - linear @ center
    return np.column_stack([linear, offset]).astype(np.float64)


def oriented_pixel_to_sem_affine(
    oriented_shape: tuple[int, int],
    sem_shape: tuple[int, int],
    same_fov_affine: np.ndarray,
) -> np.ndarray:
    oriented_h, oriented_w = oriented_shape
    sem_h, sem_w = sem_shape
    scale = np.array([[sem_w / oriented_w, 0.0, 0.0], [0.0, sem_h / oriented_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    affine33 = np.eye(3, dtype=np.float64)
    affine33[:2, :] = same_fov_affine
    return (affine33 @ scale)[:2, :]


def warp_same_fov(image: np.ndarray, affine: np.ndarray, sem_shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray]:
    sem_h, sem_w = sem_shape
    resized = cv2.resize(image.astype(np.float32), (sem_w, sem_h), interpolation=cv2.INTER_AREA)
    warped = cv2.warpAffine(
        resized,
        affine,
        (sem_w, sem_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones(resized.shape[:2], dtype=np.float32),
        affine,
        (sem_w, sem_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32)


def save_orientation_check(path: Path, afm_oriented: np.ndarray, sem_raw: np.ndarray, sem_ipf_frame: np.ndarray, ipf_sem: np.ndarray) -> None:
    sem_h, sem_w = sem_raw.shape
    afm_same = cv2.resize(robust_rescale(afm_oriented), (sem_w, sem_h), interpolation=cv2.INTER_AREA)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), dpi=200, constrained_layout=True)
    axes[0, 0].imshow(afm_same, cmap="afmhot")
    axes[0, 0].set_title("AFM software/display orientation, resized")
    axes[0, 1].imshow(sem_raw, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("H5 SEM raw: AFM alignment frame")
    axes[0, 2].imshow(sem_ipf_frame, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("H5 SEM flipud: IPF frame")
    axes[1, 0].imshow(sem_raw, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].imshow(afm_same, cmap="magma", alpha=0.42)
    axes[1, 0].set_title("Same-FOV AFM over raw SEM before rotation")
    axes[1, 1].imshow(np.clip(0.44 * np.dstack([sem_ipf_frame] * 3) + 0.72 * ipf_sem, 0, 1))
    axes[1, 1].set_title("Flipped SEM + EDAX IPF-Z")
    axes[1, 2].imshow(ipf_sem)
    axes[1, 2].set_title("EDAX IPF-Z reference")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_alignment_overlays(
    path: Path,
    sem_raw: np.ndarray,
    sem_ipf_frame: np.ndarray,
    ipf_sem: np.ndarray,
    afm_warped_raw: np.ndarray,
    afm_mask_raw: np.ndarray,
    normal_warped_raw: np.ndarray,
    normal_mask_raw: np.ndarray,
    title: str,
) -> None:
    afm_warped_ipf = np.flipud(afm_warped_raw)
    afm_mask_ipf = np.flipud(afm_mask_raw)
    normal_warped_ipf = np.flipud(normal_warped_raw)
    normal_mask_ipf = np.flipud(normal_mask_raw)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=210, constrained_layout=True)
    axes[0, 0].imshow(sem_raw, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("SEM raw frame")
    axes[0, 1].imshow(sem_raw, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].imshow(afm_warped_raw, cmap="magma", alpha=0.48 * afm_mask_raw)
    axes[0, 1].set_title("AFM height aligned to raw SEM")
    axes[0, 2].imshow(overlay_gray_with_rgba(sem_raw, normal_warped_raw, normal_mask_raw, 0.72))
    axes[0, 2].set_title("AFM normalmap aligned to raw SEM")

    axes[1, 0].imshow(ipf_sem)
    axes[1, 0].set_title("EDAX IPF-Z frame")
    axes[1, 1].imshow(np.clip(0.42 * np.dstack([sem_ipf_frame] * 3) + 0.72 * ipf_sem, 0, 1))
    axes[1, 1].imshow(afm_warped_ipf, cmap="magma", alpha=0.42 * afm_mask_ipf)
    axes[1, 1].set_title("AFM height in IPF frame")
    axes[1, 2].imshow(overlay_gray_with_rgba(ipf_sem.mean(axis=2), normal_warped_ipf, normal_mask_ipf, 0.72))
    axes[1, 2].set_title("AFM normalmap in IPF frame")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ebsd = read_sem_and_ipf(args.h5, args.angle, args.edax_ipf)
    channels, afm_meta = read_afm_channels(args.afm)
    if args.height_channel not in channels:
        raise KeyError(f"AFM channel {args.height_channel!r} not found; available={list(channels)}")

    height_um = channels[args.height_channel].astype(np.float32) * 1e6
    height_oriented = orient_image(height_um, args.afm_orientation)
    height_display = robust_rescale(height_oriented, 1.0, 99.0)
    sem_shape = ebsd["sem_raw"].shape
    affine_same = centered_affine_same_fov(
        sem_shape,
        args.rotation_deg,
        args.stretch_x,
        args.stretch_y,
        args.translate_x_px,
        args.translate_y_px,
    )
    affine_oriented_to_sem = oriented_pixel_to_sem_affine(height_oriented.shape, sem_shape, affine_same)

    afm_warped_raw, afm_mask_raw = warp_same_fov(height_display, affine_same, sem_shape)
    normal_data = height_to_normals(
        height_um=height_oriented,
        scan_size_um=float(afm_meta["scan_size_um"]),
        affine_afm_to_sem=affine_oriented_to_sem,
        smooth_sigma_px=args.normal_smooth_sigma_px,
        level=not args.no_plane_level,
    )
    normal_rgb = normal_direction_rgb(normal_data["normals_sample"], args.tilt_color_ref_deg)
    normal_warped_raw, normal_mask_raw = warp_same_fov(normal_rgb, affine_same, sem_shape)

    paths = {
        "orientation_check": output_dir / "afm_sem_same_fov_orientation_check.png",
        "sem_raw": output_dir / "ebsd60_sem_h5_raw_afm_alignment_frame.png",
        "sem_ipf_frame": output_dir / "ebsd60_sem_h5_flipud_ipf_frame.png",
        "ipf": output_dir / "ebsd60_ipf_edax_reference.png",
        "afm_height_oriented": output_dir / "afm_height_rot90_software_orientation.png",
        "afm_height_warped_raw": output_dir / "afm_height_same_fov_warped_to_raw_sem.png",
        "afm_height_warped_ipf": output_dir / "afm_height_same_fov_warped_to_ipf_frame.png",
        "normalmap": output_dir / "afm_scharr_normalmap_rot90.png",
        "normalmap_colorbar": output_dir / "afm_scharr_normalmap_rot90_with_colorbar.png",
        "normal_warped_raw": output_dir / "afm_normalmap_same_fov_warped_to_raw_sem.png",
        "normal_warped_ipf": output_dir / "afm_normalmap_same_fov_warped_to_ipf_frame.png",
        "alignment_overview": output_dir / "afm_sem_ipf_same_fov_alignment_overview.png",
        "metadata": output_dir / "pt_highres60_afm_same_fov_alignment_metadata.json",
        "data_npz": output_dir / "pt_highres60_afm_same_fov_alignment_data.npz",
    }
    save_orientation_check(paths["orientation_check"], height_display, ebsd["sem_raw"], ebsd["sem_ipf_frame"], ebsd["ipf_sem"])
    save_gray(paths["sem_raw"], ebsd["sem_raw"], "60 deg H5 SEM raw: AFM alignment frame")
    save_gray(paths["sem_ipf_frame"], ebsd["sem_ipf_frame"], "60 deg H5 SEM flipud: IPF frame")
    save_rgb(paths["ipf"], ebsd["ipf_sem"], "60 deg EDAX IPF-Z reference")
    save_gray(paths["afm_height_oriented"], height_display, f"AFM height in {args.afm_orientation} software/display orientation", cmap="afmhot")
    save_gray(paths["afm_height_warped_raw"], afm_warped_raw, "AFM height warped to raw SEM frame", cmap="afmhot")
    save_gray(paths["afm_height_warped_ipf"], np.flipud(afm_warped_raw), "AFM height warped to IPF frame", cmap="afmhot")
    plt.imsave(paths["normalmap"], normal_rgb)
    save_normalmap_with_legend(paths["normalmap_colorbar"], normal_rgb, "AFM Scharr normalmap after AFM display orientation", args.tilt_color_ref_deg)
    save_rgb(paths["normal_warped_raw"], normal_warped_raw, "AFM normalmap warped to raw SEM frame", normal_mask_raw)
    save_rgb(paths["normal_warped_ipf"], np.flipud(normal_warped_raw), "AFM normalmap warped to IPF frame", np.flipud(normal_mask_raw))
    save_alignment_overlays(
        paths["alignment_overview"],
        ebsd["sem_raw"],
        ebsd["sem_ipf_frame"],
        ebsd["ipf_sem"],
        afm_warped_raw,
        afm_mask_raw,
        normal_warped_raw,
        normal_mask_raw,
        (
            "Same-FOV AFM -> raw SEM alignment, then flipud into EDAX IPF frame | "
            f"AFM orientation={args.afm_orientation}, rotation={args.rotation_deg:g} deg"
        ),
    )
    save_scalar_image(
        output_dir / "afm_normal_tilt_deg.png",
        normal_data["tilt_deg"],
        "AFM normal tilt",
        "magma",
        "tilt (deg)",
        vmin=0.0,
        vmax=float(np.nanpercentile(normal_data["tilt_deg"], 99.0)),
    )
    save_scalar_image(
        output_dir / "afm_normal_azimuth_deg.png",
        normal_data["azimuth_deg"],
        "AFM normal azimuth in raw SEM frame",
        "twilight",
        "azimuth (deg)",
        vmin=-180.0,
        vmax=180.0,
    )

    np.savez_compressed(
        paths["data_npz"],
        sem_raw=ebsd["sem_raw"],
        sem_ipf_frame=ebsd["sem_ipf_frame"],
        ipf_sem=ebsd["ipf_sem"],
        afm_height_oriented_um=height_oriented,
        afm_height_warped_raw=afm_warped_raw,
        afm_mask_raw=afm_mask_raw,
        normal_rgb=normal_rgb,
        normal_rgb_warped_raw=normal_warped_raw,
        normal_mask_raw=normal_mask_raw,
        affine_same_fov_resized_afm_to_raw_sem=affine_same,
        affine_oriented_afm_pixels_to_raw_sem=affine_oriented_to_sem,
        normals_sample=normal_data["normals_sample"],
        normals_afm=normal_data["normals_afm"],
    )
    metadata = {
        "method_note": (
            "Corrected same-FOV AFM/SEM registration. The SEM scale bar is not used. "
            "AFM is first rotated into the software/display orientation, resized to the H5 SEM field of view, "
            "then aligned to H5 SEM raw row order by a center-based affine with explicit rotation and optional "
            "anisotropic stretch for SEM tilt-correction distortion. H5 SEM is flipud only when moving into the EDAX IPF-Z frame."
        ),
        "afm": str(args.afm),
        "afm_metadata": afm_meta,
        "h5": str(args.h5),
        "h5_group": ebsd["spec"].h5_group,
        "edax_ipf": str(args.edax_ipf),
        "angle_deg": args.angle,
        "height_channel": args.height_channel,
        "afm_orientation": args.afm_orientation,
        "sem_alignment_frame": "H5 SEM raw row order",
        "ipf_frame_relation": "IPF frame = flipud(raw SEM frame)",
        "scale_bar_used": False,
        "same_fov_affine_parameters": {
            "rotation_deg": args.rotation_deg,
            "stretch_x": args.stretch_x,
            "stretch_y": args.stretch_y,
            "translate_x_px": args.translate_x_px,
            "translate_y_px": args.translate_y_px,
            "affine_resized_afm_to_raw_sem_2x3": affine_same.tolist(),
            "affine_oriented_afm_pixels_to_raw_sem_2x3": affine_oriented_to_sem.tolist(),
        },
        "normalmap": {
            "operator": "Scharr depthmap gradient",
            "smooth_sigma_px": args.normal_smooth_sigma_px,
            "tilt_color_ref_deg": args.tilt_color_ref_deg,
        },
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if value.exists()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved same-FOV AFM/SEM/IPF alignment to {output_dir}")
    print(
        f"AFM orientation={args.afm_orientation}, raw SEM rotation={args.rotation_deg:g} deg, "
        f"stretch=({args.stretch_x:g}, {args.stretch_y:g}), translation=({args.translate_x_px:g}, {args.translate_y_px:g}) px"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align Pt high-resolution 60 deg AFM to H5 SEM as a same-field-of-view, tilt-corrected affine problem."
    )
    parser.add_argument("--afm", type=Path, default=DEFAULT_AFM)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--edax-ipf", type=Path, default=DEFAULT_EDAX_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-channel", default="HeightRetrace")
    parser.add_argument("--afm-orientation", default="rot90", choices=["raw", "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose"])
    parser.add_argument("--rotation-deg", type=float, default=10.0)
    parser.add_argument("--stretch-x", type=float, default=1.0)
    parser.add_argument("--stretch-y", type=float, default=1.0)
    parser.add_argument("--translate-x-px", type=float, default=0.0)
    parser.add_argument("--translate-y-px", type=float, default=0.0)
    parser.add_argument("--normal-smooth-sigma-px", type=float, default=1.2)
    parser.add_argument("--tilt-color-ref-deg", type=float, default=12.0)
    parser.add_argument("--no-plane-level", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
