"""Write the frozen split files from a local LLVIP archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.data import dataset_report, pair_names  # noqa: E402
from plp.splits import gallery_names, nested_split, write_lists  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "llvip")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--gallery-size", type=int, default=8)
    args = parser.parse_args()
    report = dataset_report(args.root)
    train = pair_names(args.root, "train")
    test = pair_names(args.root, "test")
    splits = nested_split(train, seed=args.seed, val_ratio=args.val_ratio)
    gallery = gallery_names(test, seed=args.seed, size=args.gallery_size)
    write_lists(args.out, splits, test, gallery)
    args.out.joinpath("dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out.joinpath("counts.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
