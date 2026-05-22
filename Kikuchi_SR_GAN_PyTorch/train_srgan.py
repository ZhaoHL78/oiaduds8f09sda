from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from kikuchi_srgan import Discriminator, GeneratorSR, KikuchiPairDataset, save_mask
from kikuchi_srgan.visualize import save_raw_pair_preview, save_sr_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch GAN on paired Kikuchi UP2 data.")
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/kikuchi_srgan_full_200"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--lambda-pixel", type=float, default=100.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--g-channels", type=int, default=32)
    parser.add_argument("--d-channels", type=int, default=32)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--upsample-mode", type=str, default="nearest")
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--no-mask", action="store_true", help="Disable circular detector masks.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 means use the full dataset.")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def add_png(writer: SummaryWriter, tag: str, path: Path, step: int) -> None:
    if not path.exists():
        return
    import matplotlib.image as mpimg

    image = mpimg.imread(path)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    writer.add_image(tag, image, global_step=step, dataformats="HWC")


def plot_loss_curves(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty or "global_step" not in df.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, color in [
        ("loss_g", "tab:blue"),
        ("loss_l1", "tab:green"),
        ("loss_adv", "tab:orange"),
        ("loss_d", "tab:red"),
    ]:
        if name in df.columns:
            axes[0].plot(df["global_step"], df[name], label=name, linewidth=1.4, color=color)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    for name, color in [("d_real_logit", "tab:purple"), ("d_fake_logit", "tab:brown")]:
        if name in df.columns:
            axes[1].plot(df["global_step"], df[name], label=name, linewidth=1.2, color=color)
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("logit")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def jsonable_args(args: argparse.Namespace) -> dict:
    data = {}
    for key, value in vars(args).items():
        data[key] = str(value) if isinstance(value, Path) else value
    return data


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    generator.load_state_dict(state["generator"])
    discriminator.load_state_dict(state["discriminator"])
    opt_g.load_state_dict(state["opt_g"])
    opt_d.load_state_dict(state["opt_d"])
    return int(state["epoch"]) + 1, int(state["global_step"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = args.out_dir.resolve()
    preview_dir = out_dir / "previews"
    ckpt_dir = out_dir / "checkpoints"
    tb_dir = out_dir / "tensorboard"
    curves_path = out_dir / "loss_curves.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(tb_dir))

    device = torch.device(args.device)
    dataset = KikuchiPairDataset(
        args.data_dir,
        use_mask=not args.no_mask,
        mask_radius_fraction=args.mask_radius_frac,
    )
    indices = dataset.preview_indices(args.preview_count)

    manifest = dataset.manifest()
    manifest["training"] = jsonable_args(args) | {"out_dir": str(out_dir), "device": str(device)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2), flush=True)
    save_mask(dataset.low_mask_np, out_dir / "masks" / "low_mask.png")
    save_mask(dataset.high_mask_np, out_dir / "masks" / "high_mask.png")
    save_raw_pair_preview(dataset, indices, preview_dir / "raw_pairs.png")
    tb.add_text("run/manifest", "```json\n" + json.dumps(manifest, indent=2) + "\n```", 0)
    add_png(tb, "masks/low_mask", out_dir / "masks" / "low_mask.png", 0)
    add_png(tb, "masks/high_mask", out_dir / "masks" / "high_mask.png", 0)
    add_png(tb, "previews/raw_pairs", preview_dir / "raw_pairs.png", 0)

    generator = GeneratorSR(
        base_channels=args.g_channels,
        num_res_blocks=args.res_blocks,
        high_size=dataset.high_shape,
        upsample_mode=args.upsample_mode,
    ).to(device)
    discriminator = Discriminator(base_channels=args.d_channels).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    scaler_g = make_scaler(args.amp and device.type == "cuda")
    scaler_d = make_scaler(args.amp and device.type == "cuda")

    start_epoch = 1
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, generator, discriminator, opt_g, opt_d, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}", flush=True)

    save_sr_preview(generator, dataset, device, indices, preview_dir / "epoch_000.png")
    add_png(tb, "previews/fixed_samples", preview_dir / "epoch_000.png", global_step)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    print(
        f"Training {len(dataset)} paired samples for {args.epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, batch_size={args.batch_size}",
        flush=True,
    )

    csv_path = out_dir / "training_log.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        if write_header:
            csv_writer.writerow(
                [
                    "epoch",
                    "step",
                    "global_step",
                    "loss_g",
                    "loss_l1",
                    "loss_adv",
                    "loss_d",
                    "d_real_logit",
                    "d_fake_logit",
                    "seconds",
                ]
            )

        try:
            for epoch in range(start_epoch, args.epochs + 1):
                generator.train()
                discriminator.train()
                t0 = time.time()
                sums = {
                    "loss_g": 0.0,
                    "loss_l1": 0.0,
                    "loss_adv": 0.0,
                    "loss_d": 0.0,
                    "d_real": 0.0,
                    "d_fake": 0.0,
                }

                progress = tqdm(
                    enumerate(loader, start=1),
                    total=steps_per_epoch,
                    desc=f"epoch {epoch}/{args.epochs}",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
                for step, batch in progress:
                    if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                        break
                    lr = batch["lr"].to(device, non_blocking=True)
                    hr = batch["hr"].to(device, non_blocking=True)
                    hr_mask = batch["hr_mask"].to(device, non_blocking=True)
                    global_step += 1

                    opt_d.zero_grad(set_to_none=True)
                    with autocast_ctx(device, args.amp):
                        with torch.no_grad():
                            fake_detached = generator(lr) * hr_mask
                        real_logits = discriminator(hr * hr_mask)
                        fake_logits = discriminator(fake_detached.detach())
                        loss_d_real = bce(real_logits, torch.ones_like(real_logits))
                        loss_d_fake = bce(fake_logits, torch.zeros_like(fake_logits))
                        loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()

                    opt_g.zero_grad(set_to_none=True)
                    with autocast_ctx(device, args.amp):
                        fake = generator(lr) * hr_mask
                        fake_logits_for_g = discriminator(fake)
                        loss_adv = bce(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                        loss_l1 = masked_l1_loss(fake, hr, hr_mask)
                        loss_g = args.lambda_pixel * loss_l1 + args.lambda_adv * loss_adv
                    scaler_g.scale(loss_g).backward()
                    scaler_g.step(opt_g)
                    scaler_g.update()

                    values = {
                        "loss_g": float(loss_g.detach().cpu()),
                        "loss_l1": float(loss_l1.detach().cpu()),
                        "loss_adv": float(loss_adv.detach().cpu()),
                        "loss_d": float(loss_d.detach().cpu()),
                        "d_real": float(real_logits.detach().mean().cpu()),
                        "d_fake": float(fake_logits.detach().mean().cpu()),
                    }
                    for key, value in values.items():
                        sums[key] += value

                    if step % args.log_every == 0 or step == 1 or step == steps_per_epoch:
                        elapsed = time.time() - t0
                        avg = {k: v / step for k, v in sums.items()}
                        csv_writer.writerow(
                            [
                                epoch,
                                step,
                                global_step,
                                avg["loss_g"],
                                avg["loss_l1"],
                                avg["loss_adv"],
                                avg["loss_d"],
                                avg["d_real"],
                                avg["d_fake"],
                                elapsed,
                            ]
                        )
                        f.flush()
                        tb.add_scalar("loss/g_total", avg["loss_g"], global_step)
                        tb.add_scalar("loss/l1_masked", avg["loss_l1"], global_step)
                        tb.add_scalar("loss/adversarial", avg["loss_adv"], global_step)
                        tb.add_scalar("loss/discriminator", avg["loss_d"], global_step)
                        tb.add_scalar("discriminator/real_logit", avg["d_real"], global_step)
                        tb.add_scalar("discriminator/fake_logit", avg["d_fake"], global_step)
                        tb.add_scalar("train/epoch", epoch, global_step)
                        progress.set_postfix(
                            loss_g=f"{avg['loss_g']:.4f}",
                            l1=f"{avg['loss_l1']:.4f}",
                            adv=f"{avg['loss_adv']:.4f}",
                            loss_d=f"{avg['loss_d']:.4f}",
                        )

                steps_done = max(1, min(step, steps_per_epoch))
                epoch_avg = {k: v / steps_done for k, v in sums.items()}
                for key, value in epoch_avg.items():
                    tb.add_scalar(f"epoch/{key}", value, epoch)

                if epoch % args.preview_every == 0:
                    preview_path = preview_dir / f"epoch_{epoch:03d}.png"
                    save_sr_preview(generator, dataset, device, indices, preview_path)
                    add_png(tb, "previews/fixed_samples", preview_path, global_step)

                plot_loss_curves(csv_path, curves_path)
                add_png(tb, "curves/loss", curves_path, global_step)

                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    epoch,
                    global_step,
                    generator,
                    discriminator,
                    opt_g,
                    opt_d,
                    args,
                )
                if epoch % args.save_every == 0 or epoch == args.epochs:
                    save_checkpoint(
                        ckpt_dir / f"epoch_{epoch:03d}.pt",
                        epoch,
                        global_step,
                        generator,
                        discriminator,
                        opt_g,
                        opt_d,
                        args,
                    )

                elapsed = time.time() - t0
                if math.isfinite(elapsed):
                    print(f"epoch {epoch} finished in {elapsed / 60.0:.1f} min", flush=True)
        finally:
            tb.flush()
            tb.close()


if __name__ == "__main__":
    main()
