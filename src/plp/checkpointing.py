"""Checkpoint payload. The evaluated weights are the lowest validation loss."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model.load_state_dict(blob["model"])
    return blob
