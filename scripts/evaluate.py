"""Evaluate a checkpoint. Writes metrics only for images that were actually scored."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.checkpointing import load_checkpoint  # noqa: E402
from plp.data import PairedThermal  # noqa: E402
from plp.failures import rank_by_mae  # noqa: E402
from plp.generative import ConditionalDiffusion  # noqa: E402
from plp.metrics import lpips_or_none, psnr, ssim, to_unit  # noqa: E402
from plp.nets import Baseline  # noqa: E402
from plp.splits import read_list  # noqa: E402


def build(blob: dict):
    cfg = blob["config"]
    if cfg["model"] == "baseline":
        return Baseline()
    return ConditionalDiffusion(steps=cfg["diffusion_steps"])


def save_panel(rgb, thermal, pred, path: Path) -> None:
    def tile(tensor):
        image = to_unit(tensor.detach().cpu())
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(array)

    tiles = [tile(rgb), tile(pred), tile(thermal)]
    width, height = tiles[0].size
    canvas = Image.new("RGB", (width * 3, height))
    for index, item in enumerate(tiles):
        canvas.paste(item, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "llvip")
    parser.add_argument("--test-file", type=Path, default=ROOT / "data" / "splits" / "test.txt")
    parser.add_argument("--gallery-file", type=Path, default=ROOT / "data" / "splits" / "gallery.txt")
    parser.add_argument("--which", choices=["best", "last"], default="best")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build(blob).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    names = read_list(args.test_file)
    if args.limit:
        names = names[: args.limit]
    gallery = set(read_list(args.gallery_file)) if args.gallery_file.is_file() else set()
    loader = DataLoader(
        PairedThermal(args.data_root, "test", names, blob["config"]["image_size"]),
        batch_size=1,
        shuffle=False,
    )
    totals = {"psnr": 0.0, "ssim": 0.0}
    lpips_total = 0.0
    lpips_count = 0
    count = 0
    predictions = []
    targets = []
    seen = []
    out_dir = args.checkpoint.parent
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            if blob["config"]["model"] == "diffusion":
                pred = model.sample(rgb, ddim_steps=blob["config"]["ddim_steps"])
            else:
                pred = model(rgb)
            totals["psnr"] += psnr(pred, thermal)
            totals["ssim"] += ssim(pred, thermal)
            perceptual = lpips_or_none(pred, thermal)
            if perceptual is not None:
                lpips_total += perceptual
                lpips_count += 1
            if batch["name"][0] in gallery:
                save_panel(rgb[0], thermal[0], pred[0], out_dir / "samples" / f"{Path(batch['name'][0]).stem}.png")
            predictions.append(pred.cpu())
            targets.append(thermal.cpu())
            seen.append(batch["name"][0])
            count += 1
    if count == 0:
        raise SystemExit("No test images.")
    metrics = {key: totals[key] / count for key in totals}
    metrics["lpips"] = None if lpips_count == 0 else lpips_total / lpips_count
    metrics["images"] = count
    ranked = rank_by_mae(seen, torch.cat(predictions), torch.cat(targets))
    result = {
        "status": "completed",
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "highest_error": ranked[:8],
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
