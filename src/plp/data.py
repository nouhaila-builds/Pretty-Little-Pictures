"""Paired LLVIP reading. RGB and thermal receive the same flip."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def pair_names(root: Path, split: str) -> list[str]:
    visible = root / "visible" / split
    infrared = root / "infrared" / split
    if not visible.is_dir() or not infrared.is_dir():
        raise FileNotFoundError(
            f"Expected {visible} and {infrared}. See data/README.md."
        )
    names = []
    missing = 0
    for path in sorted(visible.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if (infrared / path.name).is_file():
            names.append(path.name)
        else:
            missing += 1
    if not names:
        raise FileNotFoundError(f"No aligned pairs in {visible}")
    return names


def dataset_report(root: Path) -> dict:
    report = {"root": str(root), "splits": {}}
    for split in ("train", "test"):
        visible = root / "visible" / split
        infrared = root / "infrared" / split
        if not visible.is_dir():
            report["splits"][split] = {"status": "missing"}
            continue
        names = pair_names(root, split)
        sample = Image.open(visible / names[0])
        thermal = Image.open(infrared / names[0])
        report["splits"][split] = {
            "status": "ok",
            "pairs": len(names),
            "rgb_mode": sample.mode,
            "rgb_size": list(sample.size),
            "thermal_mode": thermal.mode,
            "thermal_size": list(thermal.size),
        }
    return report


class PairedThermal(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        names: list[str],
        image_size: int,
        flip_probability: float = 0.0,
        train: bool = False,
    ):
        self.root = root
        self.split = split
        self.names = list(names)
        self.image_size = image_size
        self.flip_probability = flip_probability
        self.train = train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        rgb = Image.open(self.root / "visible" / self.split / name).convert("RGB")
        thermal = Image.open(self.root / "infrared" / self.split / name).convert("L")
        rgb = TF.resize(rgb, [self.image_size, self.image_size], antialias=True)
        thermal = TF.resize(thermal, [self.image_size, self.image_size], antialias=True)
        if self.train and random.random() < self.flip_probability:
            rgb = TF.hflip(rgb)
            thermal = TF.hflip(thermal)
        rgb_t = TF.to_tensor(rgb) * 2 - 1
        thermal_t = TF.to_tensor(thermal) * 2 - 1
        return {"rgb": rgb_t, "thermal": thermal_t, "name": name}
