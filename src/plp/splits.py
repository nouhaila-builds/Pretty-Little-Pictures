"""Deterministic LLVIP splits.

Validation is cut from the official training list first and then never changes.
The scarcity lists are nested prefixes of the remaining pool:

    1% ⊂ 5% ⊂ 20% ⊂ 100%
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def nested_split(train_names: list[str], seed: int, val_ratio: float = 0.1) -> dict[str, list[str]]:
    if not train_names:
        raise ValueError("The training list is empty.")
    items = list(train_names)
    random.Random(seed).shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio)))
    if n_val >= len(items):
        raise ValueError("val_ratio leaves no training images.")
    validation = items[-n_val:]
    pool = items[:-n_val]
    n20 = max(1, int(round(len(pool) * 0.20)))
    n05 = max(1, int(round(len(pool) * 0.05)))
    n01 = max(1, int(round(len(pool) * 0.01)))
    n05 = max(n05, n01)
    n20 = max(n20, n05)
    return {
        "val": validation,
        "train_100": pool,
        "train_20": pool[:n20],
        "train_05": pool[:n05],
        "train_01": pool[:n01],
    }


def gallery_names(test_names: list[str], seed: int, size: int) -> list[str]:
    items = list(test_names)
    random.Random(seed).shuffle(items)
    return items[: min(size, len(items))]


def write_lists(out_dir: Path, splits: dict[str, list[str]], test_names: list[str], gallery: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    (out_dir / "gallery.txt").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    counts = {"test": len(test_names), "gallery": len(gallery)}
    for key, names in splits.items():
        filename = "val.txt" if key == "val" else f"{key}.txt"
        (out_dir / filename).write_text("\n".join(names) + "\n", encoding="utf-8")
        counts[key] = len(names)
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_list_for_fraction(split_dir: Path, fraction: float) -> Path:
    name = {1.0: "train_100.txt", 0.2: "train_20.txt", 0.05: "train_05.txt", 0.01: "train_01.txt"}[fraction]
    return split_dir / name
