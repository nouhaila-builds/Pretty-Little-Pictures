# Pretty Little Pictures

Looks can be deceiving — even to a machine.

Independent research case study. It asks whether pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together when a visible-to-infrared translator is trained on fewer aligned pairs.

The protocol has been run. Measured scores are in `web/public/data/experiments.json`. The page reads that file. Images, checkpoints, and result archives are not in git.

## Question

When a visible-to-infrared translator is trained on progressively fewer aligned pairs, do pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together?

## Motivation

Visible images are abundant. Thermal images are not. Generating the missing modality is only useful if the synthetic image keeps the scene structure a downstream system relies on. Similarity to the ground truth and usefulness for perception are measured separately.

## Dataset

LLVIP (Jia et al., ICCV Workshops 2021): aligned visible–infrared pairs, mostly at night, with pedestrian boxes. KAIST and FLIR ADAS were compared and not selected. The comparison is in `docs/research-design.md`. Images are not in git. See `data/README.md`.

## Method

- Baseline: L1 U-Net, RGB to one thermal channel.
- Main model: conditional diffusion. RGB is concatenated with the noisy thermal image. The network predicts the noise. Sampling uses 50 DDIM steps.
- Fractions: 1% ⊂ 5% ⊂ 20% ⊂ 100% of the training pool, after a frozen validation cut. The test set does not change.
- Protocol: the same number of epochs for every fraction. Smaller fractions take fewer optimizer steps. That choice is declared in the config notes.
- Metrics: PSNR, SSIM, LPIPS, and pedestrian AP from one detector trained on real infrared and then frozen.

Vision-language conditioning, flow matching, rectified flow, and video are not implemented. The exact protocol is in `docs/implementation_specification.md`. Hypotheses are in `hypotheses/preregistered_hypotheses.md`.

## Layout

```text
pretty-little-pictures/
├── configs/          experiment YAML
├── docs/             research design
├── notebooks/        Colab or Kaggle entry, then one notebook per stage
├── scripts/          prepare, train, evaluate, export, smoke test
├── src/plp/          data, models, training, metrics
├── experiments/      one folder per run, created on the GPU machine
├── web/              case study, fed only by exported JSON
└── data/README.md    how to place LLVIP
```

The Python package is `plp` under `src/`. A top-level package named `data` would collide with the data directory.

## Install

```text
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/export_web_results.py
```

The smoke test uses random tensors. Its numbers are not results.

## GPU

Use Kaggle or Colab. Open `notebooks/00_colab_or_kaggle.ipynb`.

1. Select a GPU runtime.
2. Point `DATA_ROOT` at LLVIP.
3. Run the smoke path inside the notebook before a full job.
4. Train, then evaluate, then export. Copy `experiments/` and `web/public/data/experiments.json` back.

Weights & Biases is optional. It starts only when `WANDB_API_KEY` is set and `WANDB_MODE=online`. The key is not stored in the repository.

```text
python scripts/prepare_dataset.py --root data/llvip
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/diffusion_full.yaml
python scripts/train.py --config configs/diffusion_20.yaml
python scripts/train.py --config configs/diffusion_05.yaml
python scripts/evaluate.py --checkpoint experiments/plp-unet-100/best.pt
python scripts/export_web_results.py
```

Add `--smoke` to `scripts/train.py` for a two-batch check on the real files.

## Status

| Experiment | Status |
| --- | --- |
| U-Net reference, 100% | Recorded |
| Diffusion, 100% / 20% / 5% / 1% | Recorded |
| Detector on real infrared, then on synthetic infrared | Recorded |

## Results

The numbers are in `web/public/data/experiments.json` and on the case-study page. On this run, PSNR stayed low and flat while SSIM, LPIPS, and pedestrian AP did not move together. Generated infrared is 128×128. The detection scores compare that output with original LLVIP frames after both are placed on a long-side-320 canvas.

## Failure analysis

High-error frames can be ranked from residuals after evaluation. Category labels and written observations stay empty until a person reviews them.

## Limits

Mostly night-time scenes. The official split is not scene-disjoint. LPIPS was trained on ordinary photographs. Equal epochs mean unequal step counts. One seed.

## Later

A detector on real thermal images versus synthetic ones. A frozen caption as an extra condition. Flow matching or rectified flow as a second generator, only after this comparison exists. Video, on a dataset that is actually a video.

## Case study

`web/` reads the exported JSON. It does not contain typed-in scores.

## Reproducibility

Seed 42. Fraction lists, validation, and the gallery are written once by `prepare_dataset.py`. Each checkpoint stores the config, the torch version, the device, and the loss history.
