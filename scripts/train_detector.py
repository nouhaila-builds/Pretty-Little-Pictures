"""Train the real-infrared pedestrian detector once, then score the test set.

Synthetic infrared is not used here. That evaluation is a later step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from plp.detector import (  # noqa: E402
    InfraredDetection,
    average_precision,
    build_detector,
    detection_collate,
)
from plp.splits import read_list  # noqa: E402
from plp.trainer import seed_everything  # noqa: E402


def move_batch(images, targets, device):
    images = [image.to(device) for image in images]
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return images, moved


def run_epoch(model, loader, optimizer, device, scaler, limit: int | None, train: bool):
    model.train()
    total = 0.0
    steps = 0
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = sum(model(images, targets).values())
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        steps += 1
        if limit is not None and steps >= limit:
            break
    return total / max(steps, 1)


@torch.no_grad()
def score_split(model, loader, device, limit: int | None) -> dict:
    model.eval()
    model.roi_heads.score_thresh = 0.05
    rows = []
    seen = 0
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        predictions = model(images)
        for prediction, target in zip(predictions, targets):
            rows.append({
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
                "gt": target["boxes"].detach().cpu(),
            })
        seen += 1
        if limit is not None and seen >= limit:
            break
    return average_precision(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "detector.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        cfg["epochs"] = 1
        cfg["image_size"] = 160
        cfg["batch_size"] = 2
        cfg["num_workers"] = 0
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    train_names = read_list(args.splits / "train_100.txt")
    val_names = read_list(args.splits / "val.txt")
    if args.smoke:
        train_names = train_names[:4]
        val_names = val_names[:2]
    train_data = InfraredDetection(args.data_root, "train", train_names, cfg["image_size"], train=True)
    val_data = InfraredDetection(args.data_root, "train", val_names, cfg["image_size"], train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=detection_collate,
    )
    model = build_detector(pretrained=not args.smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["precision"] == "amp")
    out = ROOT / "experiments" / cfg["experiment_id"]
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    started = time.time()
    limit = 2 if args.smoke else None
    for epoch in range(cfg["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, device, scaler, limit, train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, device, scaler, limit, train=False)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch + 1}/{cfg['epochs']} train {train_loss:.4f} val {val_loss:.4f}")
        payload = {"config": cfg, "history": history, "seconds": time.time() - started}
        save_checkpoint(out / "last.pt", model, payload)
        if val_loss < best:
            best = val_loss
            save_checkpoint(out / "best.pt", model, payload)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    load_checkpoint(out / "best.pt", model, device)
    if not args.smoke:
        test_names = read_list(args.splits / "test.txt")
        test_data = InfraredDetection(args.data_root, "test", test_names, cfg["image_size"], train=False)
        test_loader = DataLoader(
            test_data,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=detection_collate,
        )
        metrics = score_split(model, test_loader, device, None)
        metrics["split"] = "real_ir_test"
        (out / "detection_real_ir.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    print({"best_val_loss": best, "epochs_ran": cfg["epochs"], "seconds": time.time() - started})


if __name__ == "__main__":
    main()
