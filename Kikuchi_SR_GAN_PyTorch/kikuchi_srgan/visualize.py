from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _image_np(t: torch.Tensor):
    return t.detach().cpu().float().squeeze().clamp(0, 1).numpy()


def save_raw_pair_preview(dataset, indices: list[int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("LowR raw", "lr", False),
        ("LowR masked", "lr", True),
        ("HighR raw", "hr", False),
        ("HighR masked", "hr", True),
    ]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(2.2 * len(indices), 8.4), squeeze=False)
    for col, idx in enumerate(indices):
        meta = dataset.metadata(idx)
        title = f"{meta['area']} #{meta['local_index']}"
        raw = dataset.get_pair(idx, apply_mask=False)
        masked = dataset.get_pair(idx, apply_mask=True)
        for row, (label, key, is_masked) in enumerate(rows):
            item = masked if is_masked else raw
            axes[row, col].imshow(_image_np(item[key]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[row, col].set_title(f"{label}\n{title}" if row == 0 else label, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_sr_preview(generator, dataset, device, indices: list[int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator.eval()

    lows = []
    highs = []
    for idx in indices:
        item = dataset[idx]
        lows.append(item["lr"])
        highs.append(item["hr"])
    lr = torch.stack(lows).to(device)
    hr = torch.stack(highs).to(device)
    hr_mask = torch.stack([dataset[idx]["hr_mask"] for idx in indices]).to(device)
    sr = (generator(lr).clamp(0, 1) * hr_mask).clamp(0, 1)
    err = (sr - hr).abs().clamp(0, 1)

    rows = [("LowR masked", lr), ("Generated SR masked", sr), ("HighR masked", hr), ("Abs error", err)]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(2.2 * len(indices), 8.4), squeeze=False)
    for col, idx in enumerate(indices):
        meta = dataset.metadata(idx)
        title = f"{meta['area']} #{meta['local_index']}"
        for row, (label, tensor) in enumerate(rows):
            axes[row, col].imshow(
                _image_np(tensor[col]),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axes[row, col].set_title(f"{label}\n{title}" if row == 0 else label, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
