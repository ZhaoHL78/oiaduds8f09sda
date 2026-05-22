from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.net(x)


class GeneratorSR(nn.Module):
    """LowR -> HighR generator for paired Kikuchi super-resolution."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_res_blocks: int = 4,
        high_size: tuple[int, int] = (470, 470),
        upsample_mode: str = "nearest",
    ):
        super().__init__()
        self.high_size = high_size
        self.upsample_mode = upsample_mode
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])
        self.body_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _resize_to_high(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(
                x,
                size=self.high_size,
                mode=self.upsample_mode,
                align_corners=False,
            )
        return F.interpolate(x, size=self.high_size, mode=self.upsample_mode)

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        x = self._resize_to_high(low)
        x0 = self.head(x)
        x = self.body_conv(self.body(x0)) + x0
        return self.tail(x)


def snconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
) -> nn.Module:
    return nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    )


class Discriminator(nn.Module):
    """Patch-style discriminator returning one real/fake logit per image."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            snconv(in_channels, c),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(c, c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(c * 2, c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(c * 4, c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(c * 8, c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(c * 8, c * 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(c * 4, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
