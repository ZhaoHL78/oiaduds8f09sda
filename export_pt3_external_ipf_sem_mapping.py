from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import exposure


DEFAULT_OUTPUT_DIR = Path("outputs") / "pt3_corrected_external_ipf_sem_mapping"


@dataclass(frozen=True)
class ExternalMapSpec:
    up2_area: str
    up2_file: str
    h5_mapping: str
    inplane_angle: str
    ipf_bmp: Path
    sem_bse_tif: Path
    pc_text: str


DEFAULT_MAPS = [
    ExternalMapSpec(
        up2_area="Area 4",
        up2_file="20251209_Pt-3_Area 4_OIM Map 1.up2",
        h5_mapping="20251209/Pt-3/Area 3-90/OIM Map 1",
        inplane_angle="90",
        ipf_bmp=Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\90.bmp"),
        sem_bse_tif=Path(r"E:\ZHL\20251209Pt-EBSD\2-90bse.tif"),
        pc_text="(0.528977, 0.587473, 0.613881)",
    ),
    ExternalMapSpec(
        up2_area="Area 5",
        up2_file="20251209_Pt-3_Area 5_OIM Map 1.up2",
        h5_mapping="20251209/Pt-3/Area 3-180/OIM Map 1",
        inplane_angle="180",
        ipf_bmp=Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\180.bmp"),
        sem_bse_tif=Path(r"E:\ZHL\20251209Pt-EBSD\2-180bse.tif"),
        pc_text="(0.526476, 0.624077, 0.622152)",
    ),
    ExternalMapSpec(
        up2_area="Area 7",
        up2_file="20251209_Pt-3_Area 7_OIM Map 1.up2",
        h5_mapping="20251209/Pt-3/Area 3-270/OIM Map 1",
        inplane_angle="270",
        ipf_bmp=Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\270.bmp"),
        sem_bse_tif=Path(r"E:\ZHL\20251209Pt-EBSD\2-270bse.tif"),
        pc_text="(0.525932, 0.632043, 0.623952)",
    ),
    ExternalMapSpec(
        up2_area="Area 9",
        up2_file="20251209_Pt-3_Area 9_OIM Map 1.up2",
        h5_mapping="20251209/Pt-3/Area 3-360/OIM Map 1",
        inplane_angle="360/0",
        ipf_bmp=Path(r"E:\ZHL\ZHL-EDAX\20251209Pt\Pt-3\0.bmp"),
        sem_bse_tif=Path(r"E:\ZHL\20251209Pt-EBSD\2-360bse.tif"),
        pc_text="(0.527940, 0.602645, 0.617310)",
    ),
]


def normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def crop_bse_footer(image: np.ndarray) -> np.ndarray:
    """Remove a dark microscope footer if it is present at the bottom."""
    row_mean = image.mean(axis=1)
    smooth = np.convolve(row_mean, np.ones(9) / 9.0, mode="same")
    threshold = max(0.03, float(np.percentile(smooth, 3) + 0.03))
    dark = smooth < threshold

    run = 0
    start = image.shape[0]
    for row in range(image.shape[0] - 1, -1, -1):
        if dark[row]:
            run += 1
            start = row if run >= 20 else start
        elif run >= 20:
            break
        else:
            run = 0

    if start < image.shape[0] and start > image.shape[0] * 0.84:
        return image[:start, :]
    return image


def read_spec_images(spec: ExternalMapSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not spec.ipf_bmp.exists():
        raise FileNotFoundError(spec.ipf_bmp)
    if not spec.sem_bse_tif.exists():
        raise FileNotFoundError(spec.sem_bse_tif)

    ipf = np.asarray(Image.open(spec.ipf_bmp).convert("RGB"))
    sem_full = normalize_gray(np.asarray(Image.open(spec.sem_bse_tif)))
    sem_cropped = crop_bse_footer(sem_full)
    return ipf, sem_full, sem_cropped


def export_maps(specs: list[ExternalMapSpec], output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    rendered: list[tuple[ExternalMapSpec, np.ndarray, np.ndarray]] = []

    for spec in specs:
        ipf, sem_full, sem_cropped = read_spec_images(spec)
        key = f"pt3_up2_{spec.up2_area.lower().replace(' ', '')}_{spec.inplane_angle.replace('/', '_')}"
        ipf_png = output_dir / f"{key}_edax_ipf.png"
        sem_full_png = output_dir / f"{key}_provided_bse_full.png"
        sem_cropped_png = output_dir / f"{key}_provided_bse_crop.png"

        plt.imsave(ipf_png, ipf)
        plt.imsave(sem_full_png, sem_full, cmap="gray")
        plt.imsave(sem_cropped_png, sem_cropped, cmap="gray")

        rows.append(
            {
                "up2_area": spec.up2_area,
                "up2_file": spec.up2_file,
                "h5_mapping": spec.h5_mapping,
                "inplane_angle": spec.inplane_angle,
                "pc": spec.pc_text,
                "edax_ipf_source": str(spec.ipf_bmp),
                "provided_sem_bse_source": str(spec.sem_bse_tif),
                "edax_ipf_png": str(ipf_png),
                "provided_bse_full_png": str(sem_full_png),
                "provided_bse_crop_png": str(sem_cropped_png),
                "ipf_size": f"{ipf.shape[1]}x{ipf.shape[0]}",
                "bse_size": f"{sem_full.shape[1]}x{sem_full.shape[0]}",
                "bse_crop_size": f"{sem_cropped.shape[1]}x{sem_cropped.shape[0]}",
            }
        )
        rendered.append((spec, ipf, sem_cropped))

    csv_path = output_dir / "pt3_area4_5_7_9_corrected_external_mapping.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(rendered), 2, figsize=(10.8, 3.8 * len(rendered)), constrained_layout=True)
    if len(rendered) == 1:
        axes = np.asarray([axes])
    for ax_row, (spec, ipf, sem) in zip(axes, rendered):
        ax_row[0].imshow(ipf)
        ax_row[0].set_title(
            f"{spec.up2_area} | {spec.inplane_angle} deg | EDAX IPF\n"
            f"{spec.ipf_bmp.name} | {spec.h5_mapping}",
            fontsize=9,
        )
        ax_row[0].axis("off")
        ax_row[1].imshow(sem, cmap="gray")
        ax_row[1].set_title(f"{spec.up2_area} | provided BSE SEM\n{spec.sem_bse_tif.name}", fontsize=9)
        ax_row[1].axis("off")
    fig.suptitle(
        "Corrected Pt-3 UP2 Area 4/5/7/9: EDAX IPF and provided SEM/BSE correspondence",
        fontsize=13,
    )
    fig.savefig(output_dir / "pt3_area4_5_7_9_corrected_ipf_bse_contact_sheet.png", dpi=240)
    plt.close(fig)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the corrected Pt-3 UP2 Area 4/5/7/9 EDAX IPF and provided SEM/BSE correspondence."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = export_maps(DEFAULT_MAPS, args.output_dir)
    print(f"Saved corrected Pt-3 external IPF/SEM mapping to {args.output_dir}")
    for row in rows:
        print(
            f"{row['up2_area']} -> {row['h5_mapping']} -> "
            f"IPF {Path(row['edax_ipf_source']).name} -> SEM {Path(row['provided_sem_bse_source']).name}"
        )


if __name__ == "__main__":
    main()
