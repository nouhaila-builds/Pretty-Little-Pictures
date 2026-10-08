"""CPU check: nested splits, one baseline step, one diffusion step, two sampling steps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.failures import rank_by_mae  # noqa: E402
from plp.generative import ConditionalDiffusion  # noqa: E402
from plp.metrics import psnr, ssim  # noqa: E402
from plp.nets import Baseline  # noqa: E402
from plp.splits import nested_split  # noqa: E402


def main() -> None:
    names = [f"{index:03d}.jpg" for index in range(40)]
    splits = nested_split(names, seed=42, val_ratio=0.1)
    assert set(splits["train_01"]) <= set(splits["train_05"]) <= set(splits["train_20"]) <= set(splits["train_100"])
    assert set(splits["val"]).isdisjoint(splits["train_100"])

    rgb = torch.rand(2, 3, 32, 32) * 2 - 1
    thermal = torch.rand(2, 1, 32, 32) * 2 - 1
    baseline = Baseline(base=16)
    loss = torch.mean(torch.abs(baseline(rgb) - thermal))
    loss.backward()

    diffusion = ConditionalDiffusion(steps=8, base=16)
    noise = diffusion.loss(rgb, thermal)
    noise.backward()
    sample = diffusion.sample(rgb[:1], ddim_steps=2)
    assert sample.shape == thermal[:1].shape
    assert psnr(sample, thermal[:1]) > 0
    assert -1 <= ssim(sample, thermal[:1]) <= 1
    ranked = rank_by_mae(["a", "b"], thermal, thermal)
    assert ranked[0]["observation"] is None
    print("smoke ok", len(splits["train_01"]), len(splits["train_05"]), len(splits["train_20"]), len(splits["train_100"]))


if __name__ == "__main__":
    main()
