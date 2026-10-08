"""Score the frozen real-infrared detector on diffusion-generated test images.

The detector weights are not updated. Each diffusion checkpoint is sampled
on the same RGB test frames. Pedestrian boxes stay the LLVIP annotations.

Generated frames are 128x128 squares, because that is how the translator was
trained. They are then resized onto the same canvas the detector used for
real infrared: the long side becomes 320 pixels and the original aspect
ratio is restored. Boxes are scaled onto that canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plp.checkpointing import load_checkpoint  # noqa: E402
from plp.detector import average_precision, build_detector, read_person_boxes  # noqa: E402
from plp.generative import ConditionalDiffusion  # noqa: E402
from plp.splits import read_list  # noqa: E402

FRACTIONS = ("100", "20", "05", "01")


def canvas_size(width: int, height: int, long_side: int) -> tuple[int, int, float]:
    scale = long_side / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def load_diffusion(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = ConditionalDiffusion(steps=cfg["diffusion_steps"]).to(device)
    load_checkpoint(path, model, device)
    model.eval()
    return model, cfg


def load_detector(path: Path, device: torch.device) -> torch.nn.Module:
    model = build_detector(pretrained=False).to(device)
    load_checkpoint(path, model, device)
    model.eval()
    model.roi_heads.score_thresh = 0.05
    return model


def rgb_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original = image.size
    resized = image.resize((size, size), Image.BILINEAR)
    tensor = TF.to_tensor(resized) * 2 - 1
    return tensor, original


def synthetic_to_detector(thermal: torch.Tensor, width: int, height: int, long_side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a 128x128 generation onto the real-infrared detector canvas."""
    canvas_w, canvas_h, scale = canvas_size(width, height, long_side)
    unit = (thermal.clamp(-1, 1) + 1) * 0.5
    frame = unit.repeat(3, 1, 1).unsqueeze(0)
    frame = F.interpolate(frame, size=(canvas_h, canvas_w), mode="bilinear", align_corners=False)[0]
    return frame, torch.tensor(scale)


def score_checkpoint(diffusion, detector, names, data_root, gen_size, ddim_steps, long_side, device, limit) -> dict:
    rows = []
    chosen = names if limit <= 0 else names[:limit]
    for index, name in enumerate(chosen, start=1):
        rgb, (width, height) = rgb_tensor(data_root / "visible" / "test" / name, gen_size)
        rgb = rgb.unsqueeze(0).to(device)
        generated = diffusion.sample(rgb, ddim_steps=ddim_steps)[0].cpu()
        frame, scale = synthetic_to_detector(generated, width, height, long_side)
        boxes = read_person_boxes(data_root / "Annotations" / f"{Path(name).stem}.xml") * float(scale)
        prediction = detector([frame.to(device)])[0]
        rows.append({
            "boxes": prediction["boxes"].detach().cpu(),
            "scores": prediction["scores"].detach().cpu(),
            "gt": boxes,
        })
        if index % 100 == 0 or index == len(chosen):
            print(f"  {index}/{len(chosen)}", flush=True)
    metrics = average_precision(rows)
    metrics["images"] = len(chosen)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "plp-detector-real-ir")
    parser.add_argument("--long-side", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    if device.type != "cuda":
        raise SystemExit("This evaluation is too slow on CPU. Use a GPU session.")
    names = read_list(args.splits / "test.txt")
    detector = load_detector(args.detector, device)
    args.out.mkdir(parents=True, exist_ok=True)
    for fraction in FRACTIONS:
        destination = args.out / f"detection_synth_{fraction}.json"
        if destination.is_file() and args.limit <= 0:
            print("already scored", fraction)
            continue
        checkpoint = args.checkpoints / f"plp-diffusion-{fraction}" / "best.pt"
        print("scoring", checkpoint, flush=True)
        diffusion, cfg = load_diffusion(checkpoint, device)
        started = time.time()
        metrics = score_checkpoint(
            diffusion,
            detector,
            names,
            args.data_root,
            cfg["image_size"],
            cfg["ddim_steps"],
            args.long_side,
            device,
            args.limit,
        )
        metrics["fraction"] = fraction
        metrics["checkpoint"] = str(checkpoint)
        metrics["seconds"] = time.time() - started
        metrics["geometry"] = (
            "Generated at the translator resolution as a square, then resized "
            "to the 320-pixel long-side canvas used for real infrared."
        )
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        del diffusion
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
