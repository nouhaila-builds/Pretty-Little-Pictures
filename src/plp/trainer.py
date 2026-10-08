"""Shared training loop. Equal epochs. The saved weights are the best validation loss."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plp import __version__
from plp.checkpointing import save_checkpoint
from plp.data import PairedThermal
from plp.generative import ConditionalDiffusion
from plp.nets import Baseline
from plp.tracking import Tracker


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> torch.nn.Module:
    if cfg["model"] == "baseline":
        return Baseline()
    return ConditionalDiffusion(steps=cfg["diffusion_steps"])


def batch_loss(model: torch.nn.Module, batch: dict, cfg: dict) -> torch.Tensor:
    rgb = batch["rgb"]
    thermal = batch["thermal"]
    if cfg["model"] == "baseline":
        return torch.mean(torch.abs(model(rgb) - thermal))
    return model.loss(rgb, thermal)


def run_training(cfg: dict, train_names: list[str], val_names: list[str], data_root: Path, out_dir: Path) -> dict:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    train_data = PairedThermal(
        data_root, "train", train_names, cfg["image_size"], cfg["flip_probability"], train=True
    )
    val_data = PairedThermal(data_root, "train", val_names, cfg["image_size"], train=False)
    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["precision"] == "amp")
    tracker = Tracker(enabled=True, name=cfg["experiment_id"], config=cfg)
    best = float("inf")
    history = []
    started = time.time()
    epochs = 1 if cfg["smoke"] else cfg["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_steps = 0
        for batch in train_loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = batch_loss(model, batch, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            train_steps += 1
            if cfg["smoke"] and train_steps >= cfg["smoke_batches"]:
                break
        val_loss = _validate(model, val_loader, cfg, device, scaler.is_enabled(), cfg["smoke_batches"] if cfg["smoke"] else None)
        row = {"epoch": epoch + 1, "train_loss": train_total / max(train_steps, 1), "val_loss": val_loss}
        history.append(row)
        tracker.log(row, step=epoch + 1)
        print(f"epoch {epoch + 1}/{epochs} train {row['train_loss']:.4f} val {val_loss:.4f}")
        payload = _payload(cfg, model, device, history, time.time() - started)
        save_checkpoint(out_dir / "last.pt", model, payload)
        if val_loss < best:
            best = val_loss
            save_checkpoint(out_dir / "best.pt", model, payload)

    tracker.finish()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_val_loss": best, "epochs_ran": epochs, "seconds": time.time() - started}


def _validate(model, loader, cfg, device, amp: bool, limit: int | None) -> float:
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = batch_loss(model, batch, cfg)
            total += float(loss)
            steps += 1
            if limit is not None and steps >= limit:
                break
    model.train()
    return total / max(steps, 1)


def _payload(cfg: dict, model: torch.nn.Module, device: torch.device, history: list, seconds: float) -> dict:
    return {
        "experiment_id": cfg["experiment_id"],
        "model_name": cfg["model"],
        "config": cfg,
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "history": history,
        "seconds": seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
