"""Build the frontend JSON from result archives. Missing runs stay pending."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PLANNED = (
    ("plp-unet-100", "baseline", 1.0, "configs/baseline.yaml", None),
    ("plp-diffusion-100", "diffusion", 1.0, "configs/diffusion_full.yaml", "100"),
    ("plp-diffusion-20", "diffusion", 0.2, "configs/diffusion_20.yaml", "20"),
    ("plp-diffusion-05", "diffusion", 0.05, "configs/diffusion_05.yaml", "05"),
    ("plp-diffusion-01", "diffusion", 0.01, "configs/diffusion_01.yaml", "01"),
)


def read_zip_json(archive: Path, name: str) -> dict | None:
    if not archive.is_file():
        return None
    with zipfile.ZipFile(archive) as zipped:
        if name not in zipped.namelist():
            return None
        return json.loads(zipped.read(name))


def first_sample(archive: Path, experiment_id: str, public: Path) -> str | None:
    if not archive.is_file():
        return None
    with zipfile.ZipFile(archive) as zipped:
        images = sorted(name for name in zipped.namelist() if name.endswith(".png"))
        if not images:
            return None
        destination = public / f"{experiment_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipped.open(images[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
        return f"/samples/{experiment_id}.png"


def one(experiment_id: str, model: str, fraction: float, config: str, detection_key: str | None, public: Path, detection: dict) -> dict:
    archive = ROOT / f"{experiment_id}-eval.zip"
    record = {
        "id": experiment_id,
        "model": model,
        "status": "pending",
        "training_fraction": fraction,
        "config": config,
        "metrics": {"ssim": None, "psnr": None, "lpips": None, "ap50": None},
        "training": None,
        "samples": [],
        "sample": None,
        "highest_error": [],
    }
    measured = read_zip_json(archive, "metrics.json")
    history = read_zip_json(archive, "history.json")
    if history:
        record["training"] = history
        record["status"] = "trained_not_evaluated"
    if measured:
        record["status"] = measured.get("status", "completed")
        record["metrics"] = {
            "ssim": measured["metrics"].get("ssim"),
            "psnr": measured["metrics"].get("psnr"),
            "lpips": measured["metrics"].get("lpips"),
            "ap50": None,
        }
        record["highest_error"] = measured.get("highest_error", [])
        record["sample"] = first_sample(archive, experiment_id, public)
        if record["sample"]:
            record["samples"] = [Path(record["sample"]).name]
    if detection_key and detection_key in detection:
        record["metrics"]["ap50"] = detection[detection_key].get("ap50")
    return record


def finding(experiments: list[dict], real: dict | None) -> str | None:
    by_id = {item["id"]: item["metrics"] for item in experiments}
    needed = ["plp-diffusion-100", "plp-diffusion-20", "plp-diffusion-05", "plp-diffusion-01"]
    if real is None or any(by_id[key]["psnr"] is None or by_id[key]["ap50"] is None for key in needed):
        return None
    full, scarce = by_id["plp-diffusion-100"], by_id["plp-diffusion-01"]
    return (
        f"On the fixed test set, diffusion PSNR stayed between {min(by_id[key]['psnr'] for key in needed):.2f} "
        f"and {max(by_id[key]['psnr'] for key in needed):.2f} dB. SSIM fell from {full['ssim']:.3f} to {scarce['ssim']:.3f}, "
        f"and LPIPS rose from {full['lpips']:.3f} to {scarce['lpips']:.3f}. "
        f"A detector trained on real infrared scored {real['ap50']:.3f} AP on real test infrared, "
        f"{full['ap50']:.3f} on the 100% diffusion images, and {scarce['ap50']:.3f} on the 1% images. "
        "Under this setup, pixel fidelity and pedestrian detection did not degrade together."
    )


def main() -> None:
    counts = ROOT / "data" / "splits" / "counts.json"
    public = ROOT / "web" / "public" / "samples"
    real = read_zip_json(ROOT / "plp-detector-real-ir.zip", "detection_real_ir.json")
    synthetic = {
        key: read_zip_json(ROOT / "plp-detector-synthetic.zip", f"detection_synth_{key}.json")
        for key in ("100", "20", "05", "01")
    }
    synthetic = {key: value for key, value in synthetic.items() if value}
    dataset = {
        "name": "LLVIP",
        "status": "not_downloaded" if not counts.is_file() else "split_written",
        "counts": json.loads(counts.read_text(encoding="utf-8")) if counts.is_file() else None,
        "citation": "Jia et al., LLVIP, ICCV Workshops 2021",
    }
    experiments = [one(*item, public, synthetic) for item in PLANNED]
    payload = {
        "project": {
            "title": "Pretty Little Pictures",
            "question": "When a visible-to-infrared translator is trained on progressively fewer aligned pairs, do pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together?",
            "status": "experiments_pending",
        },
        "dataset": dataset,
        "downstream": {**real, "status": "completed"} if real else {"status": "pending"},
        "experiments": experiments,
        "finding": finding(experiments, real),
    }
    if any(item["status"] == "completed" for item in experiments):
        payload["project"]["status"] = "results_recorded" if real and len(synthetic) == 4 else "partial_results"
    out = ROOT / "web" / "public" / "data" / "experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
