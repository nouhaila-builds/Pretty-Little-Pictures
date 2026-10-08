# Implementation specification

First deliverable. No LLVIP training has been started. Do not treat the runtime ranges below as measurements.

## 1. Repository

The runnable package is `src/plp/`. A package named `data` would collide with the data directory.

```text
pretty-little-pictures/
├── hypotheses/preregistered_hypotheses.md
├── configs/          baseline, diffusion 100/20/05/01, detector
├── docs/             research design and this specification
├── notebooks/        GPU entry points; they call scripts, they do not redefine models
├── scripts/          prepare, train, evaluate, export, smoke test
├── src/plp/          splits, models, diffusion, metrics, checkpoints
├── experiments/      created on the GPU machine; not filled here
├── web/              reads exported JSON only
└── data/README.md    LLVIP placement; images are not committed
```

Literature matrix and bibliography used to choose this protocol live in `Application/thales-multimodal/pretty-little-pictures/`.

## 2. LLVIP acquisition

Official paper: Jia, Zhu, Li, Tang, Zhou, “LLVIP: A Visible-infrared Paired Dataset for Low-light Vision,” ICCV Workshops 2021.

Official page: https://bupt-ai-cz.github.io/LLVIP/

The public archive used by the prepare script is the Google Drive file `1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv`. The release is for non-commercial research. Confirm the licence text on the official page before download.

Expected layout after extraction, as documented in `data/README.md`:

```text
visible/train, visible/test
infrared/train, infrared/test
Annotations/
```

Same filename pairs a visible image with an infrared image. `scripts/prepare_dataset.py` counts pairs from the archive. Those counts are not typed in advance. Raw images stay out of git.

## 3. Dataset audit

`scripts/prepare_dataset.py` checks that every training and test filename in `visible/` has a match in `infrared/`, records image sizes, and writes the split lists. It does not invent missing pairs. Pedestrian boxes are read from the annotation files that ship with the archive. If a box file is missing, the detector experiment stays pending rather than using guessed boxes.

Known limitation, recorded before training: the official split is not scene-disjoint. Frames from one location can appear in both train and test. The audit records whether filenames contain a scene id. It does not claim that leakage is absent.

## 4. Split strategy

One shuffle of the official training filenames, seed 42. The last 10% is validation. That validation list is shared by every experiment. The official test list is the test set and does not change. Sample ids are the filenames written under `data/splits/`.

## 5. Nested subsets

From the training pool that remains after the validation cut:

| File | Rule |
| --- | --- |
| `train_100.txt` | the full pool |
| `train_20.txt` | the first 20% of that pool |
| `train_05.txt` | the first 5% of that pool |
| `train_01.txt` | the first 1% of that pool |

So `train_01 ⊂ train_05 ⊂ train_20 ⊂ train_100`. Lists are written once and reloaded. Training does not shuffle a new subset.

Nesting keeps the smaller sets inside the larger ones, so the comparison is about how much paired data the model saw. It does not remove the uncertainty of which particular frames were drawn. One seed is the core protocol. Extra seeds are optional and are not claimed.

## 6. U-Net baseline

`plp-unet-100`, config `configs/baseline.yaml`.

`Baseline` in `src/plp/nets.py`: a U-Net with base width 32, three down blocks (32, 64, 128), a 128-channel middle, skip concatenations, and a 1×1 head. Input is 3-channel RGB. Output is one thermal channel passed through tanh, so the target range is \([-1, 1]\). Loss is L1. No time embedding. Trained at 128×128 for 20 epochs, batch 8, Adam, learning rate \(10^{-4}\). A scarce U-Net run is optional and is not required for the primary curve.

## 7. Conditional diffusion

`plp-diffusion-100`, `plp-diffusion-20`, `plp-diffusion-05`, `plp-diffusion-01`. The network class does not change between them.

`ConditionalDiffusion` in `src/plp/generative.py`:

- Forward process: linear \(\beta\) from \(10^{-4}\) to \(0.02\) over 1000 steps. \(x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\).
- Timestep: uniform integer in \(\{0,\ldots,999\}\) per image.
- Conditioning: RGB concatenated with the noisy thermal image, 4 input channels. The model is \(p(\mathrm{IR}\mid\mathrm{RGB})\).
- Denoiser: the same U-Net width as the baseline, plus a sinusoidal timestep embedding added at the bottleneck.
- Loss: mean squared error between predicted noise and \(\epsilon\).
- Sampler: 50 DDIM steps, predicted clean image clamped to \([-1, 1]\).
- Resolution: 128×128. Smoke tests may use 32 or 64. 256 is not the default.

PID, F-ViTA, SC-Diff, DiffV2IR, flow matching, rectified flow, video, and VLM conditioning are not in this model.

## 8. Detector

`plp-detector-real-ir`, config `configs/detector.yaml`. Status: pending. No score exists.

Choice: torchvision Faster R-CNN with a MobileNetV3-Large FPN, COCO initialization, fine-tuned for the person class on real infrared only. Infrared is repeated to three channels. Detector input is 320 pixels on the long side, which is larger than the generator’s 128 so that small pedestrians are not evaluated only at the generator resolution. That resize is part of the protocol and will be reported with the result.

The detector is trained once on `train_100` real infrared and frozen. It is then run on real test infrared and on synthetic infrared from the same RGB test frames at 100%, 20%, 5%, and 1%.

This asks whether the generated image still contains what a real-infrared detector uses. It does not ask whether synthetic infrared helps train a detector.

## 9. Configurations

| ID | File | Fraction |
| --- | --- | --- |
| `plp-unet-100` | `configs/baseline.yaml` | 100% |
| `plp-diffusion-100` | `configs/diffusion_full.yaml` | 100% |
| `plp-diffusion-20` | `configs/diffusion_20.yaml` | 20% |
| `plp-diffusion-05` | `configs/diffusion_05.yaml` | 5% |
| `plp-diffusion-01` | `configs/diffusion_01.yaml` | 1% |
| `plp-detector-real-ir` | `configs/detector.yaml` | real IR, 100% of the training pool |

Shared diffusion settings: seed 42, 128×128, batch 8, 30 epochs, learning rate \(10^{-4}\), Adam, mixed precision, 1000 diffusion steps, 50 DDIM steps, horizontal flip applied to both modalities.

## 10. Metrics

Implemented in `src/plp/metrics.py` for images in \([-1, 1]\), scored after mapping to \([0, 1]\):

- PSNR: pixel error. A high value is not semantic correctness.
- SSIM: local luminance, contrast, and structure (Wang et al., IEEE TIP 2004). Sensitive to misalignment, which is why LLVIP’s aligned pairs are the primary set.
- LPIPS: distance in a network trained on human judgements of RGB images (Zhang et al., CVPR 2018). It is not a thermal-physics score and not a detection score. The dependency is optional until the evaluation job installs it; a missing LPIPS value stays null.

Pedestrian AP at IoU 0.5 is the detection metric. It is not computed until `plp-detector-real-ir` has been trained.

## 11. GPU memory

Estimate for a free T4 (16 GB), 128×128, batch 8, base width 32, mixed precision: the U-Net and the diffusion trainer should fit, because the activation maps are small at this resolution. The detector at 320 pixels and batch 4 should also fit. These are expectations from the resolution and batch size, not a profiled run. If allocation fails, the config change is to lower the batch size and keep the architecture fixed.

## 12. Runtime

Estimates for one T4, not measured durations:

| Job | Rough range |
| --- | --- |
| U-Net 100%, 20 epochs | under 2 hours |
| Diffusion 1% and 5%, 30 epochs | about 1–2 hours together |
| Diffusion 20% | about 1–2 hours |
| Diffusion 100% | about 2–5 hours |
| Diffusion test sampling, 50 steps, one checkpoint | about 1–3 hours |
| Detector, 12 epochs | about 1–3 hours |

The full essential set is several sessions. Save the 1% and 5% checkpoints before the 100% diffusion run, because free sessions stop.

## 13. Checkpoints

`checkpoint_every_epoch: true`. Each epoch writes a checkpoint that includes the weights, the optimizer, the epoch index, and the config. The lowest validation loss checkpoint is the one evaluated. The last checkpoint is kept as well. Resume loads that file and continues the epoch loop. A run without a checkpoint is not treated as a result.

## 14. Kaggle and Colab

Open `notebooks/00_colab_or_kaggle.ipynb`.

1. Use the GPU runtime’s existing CUDA PyTorch. Do not reinstall `torch` from `requirements.txt` over that build.
2. Print GPU name, CUDA, and `torch.__version__`.
3. Put LLVIP on the local disk of the session, not on a mounted Drive path, during training.
4. Copy checkpoints back to Drive or Kaggle output at the end of each epoch.
5. Run `python scripts/smoke_test.py` first. It uses random tensors. Its numbers are not results.
6. Only after that smoke test passes, run a config with `smoke: true` on a few real batches, then the full configs in the order 1%, 5%, 20%, 100%.

Training is not started from this specification. It waits for approval.

## 15. Result schema

`scripts/export_web_results.py` writes `web/public/data/experiments.json`. A missing `experiments/<id>/metrics.json` stays `status: pending` with null PSNR, SSIM, and LPIPS. The detector block stays pending. Nothing in the frontend is typed by hand.

Per test sample, once evaluation exists, the stored record is:

```json
{
  "sample_id": "filename",
  "rgb": "path",
  "real_ir": "path",
  "generated": {"100": null, "20": null, "5": null, "1": null},
  "metrics": {
    "100": {"psnr": null, "ssim": null, "lpips": null},
    "20": {},
    "5": {},
    "1": {}
  },
  "annotations": [],
  "detector_outputs": {}
}
```

Null means the job has not produced that field.

## 16. Website

The Next.js case study already reads the exported JSON and must keep showing “experiment pending” while status is pending. The narrative order is: RGB sensing, low light, a real LLVIP pair, paired-data scarcity, the literature boundary (generation is not the question), the nested fractions, the three scores (pixel, LPIPS, detection), the curves, failure cases only from ranked residuals, limitations, and future work. The title “Caught lying” is not used unless a completed run supports it.

## 17. Risks

- Official LLVIP split can leak a scene across train and test.
- Alignment errors, if present in a pair, inflate pixel metrics without a detection meaning.
- Equal epochs give the 1% run fewer optimizer steps. That is declared, not hidden.
- LPIPS was trained on RGB photographs.
- One seed. No confidence interval will be invented.
- A high PSNR with a missed pedestrian, or the reverse, are both allowed outcomes.
- Banday and Lall (Neurocomputing, 2025) was not read in full. It cannot be used to claim a gap.
- Free GPU sessions die. Checkpointing is part of the method.
- The non-commercial LLVIP licence limits how images can be republished. The site will show selected test examples only, with that licence stated.
