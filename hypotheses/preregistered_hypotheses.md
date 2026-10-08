# Preregistered hypotheses

Written before any LLVIP training result exists in this repository. Metrics in the website and in `experiments/` are pending. These statements are not updated to fit a later plot.

**Question.** When a visible-to-infrared translator is trained on progressively fewer aligned pairs, do pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together?

**Public wording.** When the data disappear, what fails first?

## H1 — Fidelity degradation

Reducing paired training data will degrade visible-to-infrared translation quality, measured by PSNR and SSIM on the fixed test set.

## H2 — Metric divergence

PSNR, SSIM, LPIPS, and pedestrian-detection AP will not necessarily fall at the same rate across the nested fractions 100%, 20%, 5%, and 1%.

## H3 — Semantic robustness

Some generated infrared images may remain perceptually plausible while losing information that a detector trained on real infrared uses to find pedestrians.

H3 may be false. If the four scores move together, that is the result. If detection stays higher than pixel scores, that is the result. If the opposite happens, that is the result.

## What would count against H2 and H3

H2 is not supported if PSNR, SSIM, LPIPS, and detection AP change in the same direction and by similar relative amounts at every fraction. Four points are not enough for a significance test. Any correlation is descriptive.

H3 is not supported if the frames that look close to the real infrared, by LPIPS, are also the frames where the frozen detector agrees with its output on real infrared.

## Protocol locked with these hypotheses

- Dataset: LLVIP aligned pairs. Validation and test lists do not change.
- Nested training lists: 1% ⊂ 5% ⊂ 20% ⊂ 100%, seed 42.
- Main scarcity curve: one conditional diffusion model, architecture fixed.
- Reference: L1 U-Net at 100% (`plp-unet-100`). A scarce U-Net run is optional and is not part of the primary curve.
- Detector: trained once on real infrared, then frozen. Evaluated on real test infrared and on the four synthetic sets. Status: pending.
