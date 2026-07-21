from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from igor2.binarywave import load as load_ibw
from PIL import Image
from skimage import exposure, filters

from export_pt3_external_ipf_sem_mapping import normalize_gray


DEFAULT_AFM = Path(r"D:\EBSD project\3d数据\pt-afm\Pt-1.ibw")
DEFAULT_SEM = Path(r"E:\ZHL\20251209Pt-EBSD\2-90bse.tif")
DEFAULT_IPF = Path("outputs") / "pt3_area90_finetuned_ipf_map" / "pt3_area90_finetuned_ipf_clean_714x550.png"
FALLBACK_EDAX_IPF = Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\90.bmp")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pt_afm_sem_ipf_alignment"


@dataclass(frozen=True)
class EbsdRelation:
    up2_area: str = "Area 4"
    up2_file: str = "20251209_Pt-3_Area 4_OIM Map 1.up2"
    h5_mapping: str = "20251209/Pt-3/Area 3-90/OIM Map 1"
    inplane_angle_deg: str = "90"
    sem_bse: str = r"E:\ZHL\20251209Pt-EBSD\2-90bse.tif"
    edax_ipf: str = r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\90.bmp"


@dataclass
class Candidate:
    afm_name: str
    sem_name: str
    matches: int
    inliers: int
    inlier_ratio: float
    rmse: float
    affine_afm_to_sem: np.ndarray
    inlier_mask: np.ndarray
    key_sem: np.ndarray
    key_afm: np.ndarray

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.affine_afm_to_sem[:, :2]))

    @property
    def sx(self) -> float:
        return float(np.linalg.norm(self.affine_afm_to_sem[:, 0]))

    @property
    def sy(self) -> float:
        return float(np.linalg.norm(self.affine_afm_to_sem[:, 1]))


def parse_note(note: bytes | str) -> dict[str, str]:
    text = note.decode("utf-8", "ignore") if isinstance(note, bytes) else str(note)
    output: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        output[key.strip()] = value.strip()
    return output


def read_afm_channels(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = load_ibw(str(path))["wave"]
    data = np.asarray(payload["wData"], dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]

    labels_raw = payload.get("labels", [[], [], [], []])
    channel_labels = labels_raw[2] if len(labels_raw) > 2 else []
    labels: list[str] = []
    for raw in channel_labels:
        label = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        label = label.strip("\x00").strip()
        if label:
            labels.append(label)
    if len(labels) < data.shape[2]:
        labels.extend(f"channel_{index}" for index in range(len(labels), data.shape[2]))

    channels = {label: data[:, :, index] for index, label in enumerate(labels[: data.shape[2]])}
    note = parse_note(payload.get("note", b""))
    scan_size_um = float(note.get("ScanSize", "nan")) * 1e6 if "ScanSize" in note else float("nan")
    metadata = {
        "shape": list(data.shape[:2]),
        "channel_labels": labels[: data.shape[2]],
        "scan_size_um": scan_size_um,
        "scan_angle_deg": float(note.get("ScanAngle", "nan")) if "ScanAngle" in note else float("nan"),
        "imaging_mode": note.get("ImagingMode", ""),
        "points_lines": note.get("PointsLines", ""),
    }
    return channels, metadata


def robust_rescale(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def high_pass(image: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    normalized = robust_rescale(image)
    background = filters.gaussian(normalized, sigma=sigma, preserve_range=True)
    return robust_rescale(normalized - background, 0.2, 99.8)


def feature_variants(prefix: str, image: np.ndarray) -> dict[str, np.ndarray]:
    normalized = robust_rescale(image)
    hp = high_pass(normalized)
    sobel = robust_rescale(filters.sobel(normalized), 0.2, 99.8)
    canny = cv2.Canny((normalized * 255).astype(np.uint8), 35, 120).astype(np.float32) / 255.0
    return {
        f"{prefix}_norm": normalized,
        f"{prefix}_inv": 1.0 - normalized,
        f"{prefix}_hp": hp,
        f"{prefix}_hp_inv": 1.0 - hp,
        f"{prefix}_sobel": sobel,
        f"{prefix}_canny": canny,
    }


def afm_feature_images(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    preferred = ["HeightRetrace", "AmplitudeRetrace", "ZSensorRetrace", "PhaseRetrace"]
    for name in preferred:
        if name not in channels:
            continue
        output.update(feature_variants(name, channels[name]))
    return output


def sem_feature_images(sem: np.ndarray) -> dict[str, np.ndarray]:
    normalized = robust_rescale(sem)
    hp = high_pass(normalized, sigma=10.0)
    canny = cv2.Canny((normalized * 255).astype(np.uint8), 25, 110).astype(np.float32) / 255.0
    return {
        "sem_norm": normalized,
        "sem_inv": 1.0 - normalized,
        "sem_hp": hp,
        "sem_hp_inv": 1.0 - hp,
        "sem_sobel": robust_rescale(filters.sobel(normalized), 0.2, 99.8),
        "sem_canny": canny,
    }


def crop_sem_content(image: np.ndarray) -> np.ndarray:
    """Crop microscope footer text from the provided SEM/BSE frame.

    The Pt-3 external BSE TIFFs contain a black annotation/footer band at the
    bottom.  The first footer row is a nearly black horizontal separator; using
    it is more stable than detecting the later text rows.
    """
    height = image.shape[0]
    row_mean = image.mean(axis=1)
    search_start = int(height * 0.72)
    dark_threshold = min(0.08, max(0.025, float(np.percentile(row_mean[:search_start], 2.0) * 0.35)))
    dark = row_mean < dark_threshold
    for row in range(search_start, height - 8):
        if bool(np.all(dark[row : row + 6])):
            return image[:row, :]
    return image


def make_lightglue_models(device: str, max_keypoints: int):
    from lightglue import LightGlue, SuperPoint

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def match_lightglue(
    sem_image: np.ndarray,
    afm_image: np.ndarray,
    extractor,
    matcher,
    device: str,
    ransac_reproj_threshold: float,
) -> Candidate | None:
    from lightglue.utils import rbd

    with torch.inference_mode():
        tensor_sem = torch.from_numpy(sem_image.astype(np.float32))[None, None].to(device)
        tensor_afm = torch.from_numpy(afm_image.astype(np.float32))[None, None].to(device)
        feats_sem = extractor.extract(tensor_sem)
        feats_afm = extractor.extract(tensor_afm)
        matches_out = matcher({"image0": feats_sem, "image1": feats_afm})
        feats_sem = rbd(feats_sem)
        feats_afm = rbd(feats_afm)
        matches_out = rbd(matches_out)
        matches = matches_out["matches"]
        if matches.shape[0] < 6:
            return None
        key_sem = feats_sem["keypoints"][matches[:, 0]].detach().cpu().numpy()
        key_afm = feats_afm["keypoints"][matches[:, 1]].detach().cpu().numpy()

    affine, inliers = cv2.estimateAffine2D(
        key_afm,
        key_sem,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if affine is None or inliers is None:
        return None

    inlier_mask = inliers.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < 6:
        return None

    predicted = (affine @ np.vstack([key_afm.T, np.ones(key_afm.shape[0])])).T
    err = np.linalg.norm(predicted - key_sem, axis=1)
    rmse = float(np.sqrt(np.mean(err[inlier_mask] ** 2)))
    return Candidate(
        afm_name="",
        sem_name="",
        matches=int(key_sem.shape[0]),
        inliers=int(inlier_mask.sum()),
        inlier_ratio=float(inlier_mask.mean()),
        rmse=rmse,
        affine_afm_to_sem=affine.astype(np.float64),
        inlier_mask=inlier_mask,
        key_sem=key_sem,
        key_afm=key_afm,
    )


def rank_candidate(candidate: Candidate) -> tuple[int, int, float, float, int]:
    plausible_scale = (
        0.80 <= candidate.det <= 3.20
        and 0.75 <= candidate.sx <= 1.85
        and 0.75 <= candidate.sy <= 1.85
    )
    return (int(plausible_scale), candidate.inliers, candidate.inlier_ratio, -candidate.rmse, candidate.matches)


def find_best_alignment(
    sem_features: dict[str, np.ndarray],
    afm_features: dict[str, np.ndarray],
    extractor,
    matcher,
    device: str,
    ransac_reproj_threshold: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for afm_name, afm_image in afm_features.items():
        for sem_name, sem_image in sem_features.items():
            result = match_lightglue(
                sem_image=sem_image,
                afm_image=afm_image,
                extractor=extractor,
                matcher=matcher,
                device=device,
                ransac_reproj_threshold=ransac_reproj_threshold,
            )
            if result is None:
                continue
            result.afm_name = afm_name
            result.sem_name = sem_name
            candidates.append(result)
    return sorted(candidates, key=rank_candidate, reverse=True)


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        affine = candidate.affine_afm_to_sem
        rows.append(
            {
                "rank": rank,
                "afm_name": candidate.afm_name,
                "sem_name": candidate.sem_name,
                "matches": candidate.matches,
                "inliers": candidate.inliers,
                "inlier_ratio": candidate.inlier_ratio,
                "rmse": candidate.rmse,
                "det": candidate.det,
                "sx": candidate.sx,
                "sy": candidate.sy,
                "M00": affine[0, 0],
                "M01": affine[0, 1],
                "M02": affine[0, 2],
                "M10": affine[1, 0],
                "M11": affine[1, 1],
                "M12": affine[1, 2],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def warp_to_sem(image: np.ndarray, affine_afm_to_sem: np.ndarray, sem_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = sem_shape
    normalized = robust_rescale(image)
    warped = cv2.warpAffine(
        normalized,
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones_like(normalized, dtype=np.float32),
        affine_afm_to_sem,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32)


def resolve_ipf(path: Path) -> Path:
    if path.exists():
        return path
    if FALLBACK_EDAX_IPF.exists():
        return FALLBACK_EDAX_IPF
    raise FileNotFoundError(f"Could not find IPF image: {path} or fallback {FALLBACK_EDAX_IPF}")


def read_ipf_to_sem_frame(ipf_path: Path, sem_shape: tuple[int, int]) -> np.ndarray:
    ipf = np.asarray(Image.open(ipf_path).convert("RGB"), dtype=np.float32) / 255.0
    sem_h, sem_w = sem_shape
    return cv2.resize(ipf, (sem_w, sem_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def save_afm_channel_preview(path: Path, channels: dict[str, np.ndarray]) -> None:
    labels = list(channels)
    cols = min(4, len(labels))
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.0), dpi=180, constrained_layout=True)
    if cols == 1:
        axes = [axes]
    for ax, label in zip(axes, labels[:cols]):
        ax.imshow(robust_rescale(channels[label]), cmap="gray")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_candidate_sem_preview(path: Path, sem_content: np.ndarray, ipf_sem: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=180, constrained_layout=True)
    axes[0].imshow(sem_content, cmap="gray")
    axes[0].set_title("Provided SEM/BSE 2-90 crop")
    axes[0].axis("off")
    axes[1].imshow(ipf_sem)
    axes[1].set_title("Pt-3 90 deg IPF resized to SEM frame")
    axes[1].axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_match_figure(
    path: Path,
    sem_feature: np.ndarray,
    afm_feature: np.ndarray,
    candidate: Candidate,
    max_lines: int = 80,
) -> None:
    sem_rgb = np.dstack([sem_feature] * 3)
    afm_rgb = np.dstack([afm_feature] * 3)
    h0, w0 = sem_feature.shape
    h1, w1 = afm_feature.shape
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas[:h0, :w0] = sem_rgb
    canvas[:h1, w0 : w0 + w1] = afm_rgb

    inlier_indices = np.flatnonzero(candidate.inlier_mask)
    if inlier_indices.size > max_lines:
        inlier_indices = inlier_indices[np.linspace(0, inlier_indices.size - 1, max_lines).astype(int)]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    ax.imshow(canvas)
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, max(1, len(inlier_indices))))
    for color, index in zip(colors, inlier_indices):
        x0, y0 = candidate.key_sem[index]
        x1, y1 = candidate.key_afm[index]
        ax.plot([x0, x1 + w0], [y0, y1], color=color, linewidth=0.7, alpha=0.82)
        ax.scatter([x0, x1 + w0], [y0, y1], s=8, color=color)
    ax.set_title(
        f"LightGlue/SuperPoint inlier matches: SEM preprocessed (left) -> AFM preprocessed (right)\n"
        f"{candidate.sem_name} vs {candidate.afm_name}, inliers={candidate.inliers}/{candidate.matches}, RMSE={candidate.rmse:.2f}px"
    )
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    path: Path,
    sem_content: np.ndarray,
    afm_amp_warped: np.ndarray,
    afm_height_warped: np.ndarray,
    afm_mask: np.ndarray,
    ipf_sem: np.ndarray,
    best: Candidate,
) -> None:
    sem_rgb = np.dstack([sem_content] * 3)
    ipf_overlay = np.clip(0.38 * sem_rgb + 0.78 * ipf_sem, 0.0, 1.0)

    amp_rgba = plt.get_cmap("magma")(afm_amp_warped)
    amp_rgba[..., 3] = 0.72 * afm_mask
    height_rgba = plt.get_cmap("gray")(afm_height_warped)
    height_rgba[..., 3] = 0.48 * afm_mask

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5), dpi=190, constrained_layout=True)
    axes[0, 0].imshow(sem_content, cmap="gray")
    axes[0, 0].set_title("SEM/BSE 2-90 content")
    axes[0, 1].imshow(sem_content, cmap="gray")
    axes[0, 1].imshow(amp_rgba)
    axes[0, 1].set_title("AFM Amplitude warped onto SEM")
    axes[1, 0].imshow(ipf_overlay)
    axes[1, 0].set_title("EBSD/IPF in SEM frame")
    axes[1, 1].imshow(ipf_overlay)
    axes[1, 1].imshow(height_rgba)
    axes[1, 1].set_title("AFM warped onto EBSD/IPF frame")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        "AFM -> SEM/IPF registration using LightGlue/SuperPoint\n"
        f"{best.afm_name} vs {best.sem_name}, inliers={best.inliers}/{best.matches}, RMSE={best.rmse:.2f}px"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    afm_path: Path,
    sem_path: Path,
    ipf_path: Path,
    channels: dict[str, np.ndarray],
    afm_meta: dict[str, Any],
    sem_content: np.ndarray,
    sem_features: dict[str, np.ndarray],
    afm_features: dict[str, np.ndarray],
    candidates: list[Candidate],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = candidates[0]
    relation = EbsdRelation()
    ipf_sem = read_ipf_to_sem_frame(ipf_path, sem_content.shape)
    amp_source = channels.get("AmplitudeRetrace", next(iter(channels.values())))
    height_source = channels.get("HeightRetrace", next(iter(channels.values())))
    afm_amp_warped, afm_mask = warp_to_sem(amp_source, best.affine_afm_to_sem, sem_content.shape)
    afm_height_warped, _ = warp_to_sem(height_source, best.affine_afm_to_sem, sem_content.shape)

    paths = {
        "overview": output_dir / "afm_sem_ipf_alignment_overview.png",
        "matches": output_dir / "lightglue_afm_sem_inlier_matches.png",
        "candidate_table": output_dir / "lightglue_afm_sem_candidates.csv",
        "afm_channels": output_dir / "afm_channels_preview.png",
        "candidate_sem": output_dir / "candidate_sem_preview.png",
        "sem_content": output_dir / "sem_2_90_content_norm.png",
        "ipf_sem": output_dir / "ipf_resized_to_sem_frame.png",
        "warped_afm_amp": output_dir / "afm_amplitude_warped_to_sem.png",
        "warped_afm_height": output_dir / "afm_height_warped_to_sem.png",
        "metadata": output_dir / "afm_sem_ipf_alignment_metadata.json",
    }

    plt.imsave(paths["sem_content"], sem_content, cmap="gray")
    plt.imsave(paths["ipf_sem"], ipf_sem)
    plt.imsave(paths["warped_afm_amp"], afm_amp_warped, cmap="magma")
    plt.imsave(paths["warped_afm_height"], afm_height_warped, cmap="gray")
    save_afm_channel_preview(paths["afm_channels"], channels)
    save_candidate_sem_preview(paths["candidate_sem"], sem_content, ipf_sem)
    write_candidates(paths["candidate_table"], candidates)
    save_match_figure(paths["matches"], sem_features[best.sem_name], afm_features[best.afm_name], best)
    save_overview(paths["overview"], sem_content, afm_amp_warped, afm_height_warped, afm_mask, ipf_sem, best)

    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :] = best.affine_afm_to_sem
    metadata = {
        "afm_file": str(afm_path),
        "sem_file": str(sem_path),
        "ipf_file": str(ipf_path),
        "afm_metadata": afm_meta,
        "sem_content_shape": list(sem_content.shape),
        "best": {
            "afm_name": best.afm_name,
            "sem_name": best.sem_name,
            "matches": best.matches,
            "inliers": best.inliers,
            "inlier_ratio": best.inlier_ratio,
            "rmse": best.rmse,
            "det": best.det,
            "sx": best.sx,
            "sy": best.sy,
        },
        "afm_to_sem_affine_2x3": best.affine_afm_to_sem.tolist(),
        "sem_to_afm_homogeneous_3x3": np.linalg.inv(homogeneous).tolist(),
        "ebsd_relation": {
            "up2_area": relation.up2_area,
            "up2_file": relation.up2_file,
            "h5_mapping": relation.h5_mapping,
            "inplane_angle_deg": relation.inplane_angle_deg,
            "provided_sem_bse": relation.sem_bse,
            "edax_ipf": relation.edax_ipf,
        },
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    afm_path = args.afm
    sem_path = args.sem
    ipf_path = resolve_ipf(args.ipf)
    channels, afm_meta = read_afm_channels(afm_path)
    sem_full = normalize_gray(np.asarray(Image.open(sem_path), dtype=np.float32))
    sem_content = crop_sem_content(sem_full)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    extractor, matcher = make_lightglue_models(device, args.max_keypoints)
    sem_features = sem_feature_images(sem_content)
    afm_features = afm_feature_images(channels)
    candidates = find_best_alignment(
        sem_features=sem_features,
        afm_features=afm_features,
        extractor=extractor,
        matcher=matcher,
        device=device,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
    )
    if not candidates:
        raise RuntimeError("LightGlue/SuperPoint did not find a usable AFM -> SEM affine alignment")

    metadata = save_outputs(
        output_dir=args.output_dir,
        afm_path=afm_path,
        sem_path=sem_path,
        ipf_path=ipf_path,
        channels=channels,
        afm_meta=afm_meta,
        sem_content=sem_content,
        sem_features=sem_features,
        afm_features=afm_features,
        candidates=candidates,
    )
    best = candidates[0]
    print(f"Saved AFM/SEM/IPF alignment to {args.output_dir}")
    print(
        f"Best LightGlue/SuperPoint: {best.afm_name} -> {best.sem_name}, "
        f"{best.inliers}/{best.matches} inliers, RMSE={best.rmse:.2f}px"
    )
    print("EBSD relation: Pt-3 Area 4 UP2 -> H5 Area 3-90/OIM Map 1 -> SEM 2-90bse -> IPF 90 deg")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register the Pt AFM IBW image to the corresponding provided SEM/BSE image with "
            "LightGlue/SuperPoint, then place the matched AFM into the established Pt-3 90-degree EBSD/IPF frame."
        )
    )
    parser.add_argument("--afm", type=Path, default=DEFAULT_AFM)
    parser.add_argument("--sem", type=Path, default=DEFAULT_SEM)
    parser.add_argument("--ipf", type=Path, default=DEFAULT_IPF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force LightGlue/SuperPoint to run on CPU.")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
