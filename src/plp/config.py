"""YAML configuration. Missing keys are errors, not silent defaults at runtime."""

from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED = (
    "experiment_id",
    "model",
    "dataset_name",
    "dataset_version",
    "seed",
    "split_seed",
    "val_ratio",
    "training_fraction",
    "image_size",
    "batch_size",
    "epochs",
    "learning_rate",
    "optimizer",
    "scheduler",
    "loss",
    "precision",
    "num_workers",
    "flip_probability",
    "smoke",
    "smoke_batches",
    "checkpoint_every_epoch",
    "diffusion_steps",
    "ddim_steps",
    "gallery_size",
)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    if cfg["model"] not in {"baseline", "diffusion"}:
        raise ValueError("model must be baseline or diffusion")
    if cfg["training_fraction"] not in {1.0, 0.2, 0.05, 0.01}:
        raise ValueError("training_fraction must be 1.0, 0.2, 0.05, or 0.01")
    if cfg["image_size"] % 8 != 0:
        raise ValueError("image_size must be a multiple of 8")
    cfg["config_path"] = str(path)
    return cfg
