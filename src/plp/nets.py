"""Small U-Net used by the L1 baseline and by the diffusion noise predictor."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    angles = timesteps.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 32, with_time: bool = False):
        super().__init__()
        self.with_time = with_time
        self.down1 = Block(in_channels, base)
        self.down2 = Block(base, base * 2)
        self.down3 = Block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = Block(base * 4, base * 4)
        self.up3 = Block(base * 8, base * 2)
        self.up2 = Block(base * 4, base)
        self.up1 = Block(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)
        if with_time:
            self.to_time = nn.Sequential(nn.Linear(32, base * 4), nn.SiLU(), nn.Linear(base * 4, base * 4))

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        hidden = self.mid(self.pool(d3))
        if self.with_time:
            if t is None:
                raise ValueError("A timestep is required.")
            hidden = hidden + self.to_time(timestep_embedding(t))[:, :, None, None]
        hidden = self.up3(_cat(hidden, d3))
        hidden = self.up2(_cat(hidden, d2))
        hidden = self.up1(_cat(hidden, d1))
        return self.head(hidden)


def _cat(low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    low = nn.functional.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([low, skip], dim=1)


class Baseline(nn.Module):
    """Supervised RGB to thermal map. Output is tanh-scaled to [-1, 1]."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.net = UNet(3, 1, base=base, with_time=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(rgb))
