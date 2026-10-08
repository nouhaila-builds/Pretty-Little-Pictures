"""Train the baseline or a diffusion fraction from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.config import load_config  # noqa: E402
from plp.splits import read_list, train_list_for_fraction  # noqa: E402
from plp.trainer import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "llvip")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["smoke"] = True
        cfg["image_size"] = min(cfg["image_size"], 64)
        cfg["batch_size"] = min(cfg["batch_size"], 2)
        cfg["num_workers"] = 0
    train_names = read_list(train_list_for_fraction(args.splits, cfg["training_fraction"]))
    val_names = read_list(args.splits / "val.txt")
    if cfg["smoke"]:
        train_names = train_names[: max(cfg["batch_size"] * cfg["smoke_batches"], 1)]
        val_names = val_names[: cfg["batch_size"]]
    out = ROOT / "experiments" / cfg["experiment_id"]
    summary = run_training(cfg, train_names, val_names, args.data_root, out)
    print(summary)


if __name__ == "__main__":
    main()
