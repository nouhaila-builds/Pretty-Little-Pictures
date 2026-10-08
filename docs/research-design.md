# Pretty Little Pictures — research design

Independent case study. No experiment in this repository has been run on LLVIP. Metrics stay null until a training job writes them.

## 1. Dataset

**Selected dataset: LLVIP** (Jia, Zhu, Li, Tang, Zhou, ICCV Workshops 2021).

LLVIP is a public set of 15,488 aligned visible–infrared pairs (30,976 images) from 26 scenes, mostly at night, with pedestrian boxes. The paper’s comparison table lists the aligned resolution as 1080×720. The copy in this repository measures 1280×1024 for the sampled visible and infrared frames, and the infrared files are stored as three-channel images. Training converts infrared to one channel. The thermal band in the paper is 8–14 µm. The release is for non-commercial research. Download and the licence are documented in `data/README.md`. Raw images are not committed.

Why this set, among the three candidates:

| | LLVIP | KAIST multispectral | FLIR ADAS |
| --- | --- | --- | --- |
| Pairing | Strictly aligned visible–infrared pairs | Beam-splitter colour–thermal pairs, about 95,000 frames at 20 Hz | Original thermal and RGB frames are not spatially aligned |
| Resolution | 1080×720 aligned | 640×480 | Often 640×512 thermal |
| Labels | Pedestrian boxes | Pedestrian boxes with temporal identity | Detection boxes |
| Day / night | 24 dark scenes, 2 daytime | Day and night | Driving, day and night |
| Low-data use | A 5% slice of the official training split is still hundreds of pairs, and pairs are the right unit | Consecutive frames repeat the same street. A random 5% can be near-duplicates | Misalignment would dominate PSNR, SSIM, and LPIPS |
| Detection later | Possible on the provided boxes | Possible, and it is a real video set | Possible only after a separate alignment step |
| Licence / access | Non-commercial, public Google Drive archive | Academic benchmark, large download | Teledyne FLIR terms. The aligned copy used in papers is a third-party reprocessing (Zhang et al., ICIP 2020) |

KAIST is the better video corpus. This project does not generate video. FLIR’s original pairs are the wrong geometry for a pixel-faithful translation study. LLVIP is the set on which a gap between “looks thermal” and “still contains the pedestrian” can be measured.

The official train and test counts will be recorded by `scripts/prepare_dataset.py` from the downloaded archive. They are not typed in by hand.

## 2. Hypothesis

Frozen in `hypotheses/preregistered_hypotheses.md` before any LLVIP result exists.

**Question.** When a visible-to-infrared translator is trained on progressively fewer aligned pairs, do pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together?

**H1.** Reducing paired training data degrades translation quality on the fixed test set.

**H2.** PSNR, SSIM, LPIPS, and pedestrian-detection AP need not fall at the same rate.

**H3.** Some generated images may stay perceptually plausible while losing pedestrian information a real-infrared detector uses. H3 may be false.

The conceptual claim the study is built to examine, without assuming the direction of the result:

visual realism is not the same thing as preserved scene information.

## 3. Experimental protocol

1. Official LLVIP **test** split: frozen.
2. Official LLVIP **train** split, shuffled with seed 42.
3. The last 10% of that shuffle is the **validation** set. It is identical for every run.
4. The remaining pool is the 100% training set. The 20% subset is a prefix of that pool. The 5% subset is a prefix of the 20% subset. The 1% subset is a prefix of the 5% subset. So 1% ⊂ 5% ⊂ 20% ⊂ 100%.
5. A gallery of test filenames is drawn with seed 42 **before** any training and written to `data/splits/gallery.txt`. Every model is compared on those same frames.
6. **Equal epochs, not equal step counts, and no early stopping.** Each fraction is trained for the epoch count in its config. Smaller fractions therefore receive fewer optimizer steps and the same number of passes per image. Step counts are logged so the compute difference is visible. The checkpoint with the lowest validation loss is the one that is evaluated, and the last checkpoint is kept as well.
7. Resolution for the first real runs: 128×128. 256×256 is a later run only if 128 is stable.
8. Augmentation: the same horizontal flip applied to the RGB image and the thermal image. No augmentation that would invent thermal geometry.
9. What changes across `plp-diffusion-100`, `plp-diffusion-20`, `plp-diffusion-05`, and `plp-diffusion-01` is only the training list.

Scene identity is not part of the official split. Frames from one street may appear in both train and test. Phase 3 records whether filenames reveal a scene id. Until that is checked, scene leakage is an open limitation, not a solved one.

## 4. Models

**Experiment 1 — supervised baseline.** A small U-Net maps 3-channel RGB to 1-channel thermal and is trained with L1. This is the conventional paired translator. A conditional GAN is not the baseline: an adversarial loss would mix “supervised translation” with a second training objective.

**Experiment 2 — conditional diffusion.** A second U-Net of the same width predicts the noise added to the thermal image. The noisy thermal channel is concatenated with the RGB image (4 input channels). Training follows the DDPM noise-prediction objective (Ho, Jain, Abbeel, NeurIPS 2020). Conditioning by concatenation is the same idea as Palette (Saharia et al., SIGGRAPH 2022). Inference uses 50 DDIM steps. The schedule, the loss, and the sampler are implemented in this repository so the model is not a wrapped pretrained pipeline.

Stable Diffusion, ControlNet, and LoRA are out. A photographic RGB prior is a poor match for long-wave thermal imagery, and it would make the scarcity result hard to interpret.

Both models see thermal values scaled to \([-1, 1]\).

## 5. Evaluation

On the fixed test set, after mapping predictions back to \([0, 1]\):

| Metric | What it measures | What it does not measure |
| --- | --- | --- |
| PSNR | Pixel error against the aligned ground-truth thermal image | Whether a pedestrian is still detectable |
| SSIM (Wang et al., IEEE TIP 2004) | Local structure and contrast agreement | Semantic correctness of heat patterns |
| LPIPS (Zhang et al., CVPR 2018) | Distance in a network trained on human perceptual judgements of RGB images | Thermal physics, and detection utility. The network was not trained on infrared |

FID is not used. The usual Inception embedding was trained on RGB photographs, and copying a thermal image into three channels does not fix that.

The same gallery frames are exported for the baseline and for diffusion at 100%, 20%, and 5%: RGB, prediction, ground truth.

Failure review uses the residual between prediction and ground truth to **rank** test frames. A person then assigns a category and writes an observation. The export leaves `observation` null until that review exists. A null observation is not filled with a generated sentence.

Pedestrian detection is part of the primary question. `plp-detector-real-ir` trains once on real infrared and is then frozen. Its status stays `pending` until that job has run. No AP value is filled in advance.

## 6. Compute

Local CPU: configuration, unit checks, the smoke test, and the website.

Kaggle or Colab GPU, in this order:

1. Smoke test (`smoke: true`, 64×64, a handful of batches).
2. Baseline at 100%, then diffusion at 100%.
3. Diffusion at 20% and 5%.
4. Evaluation and export.
5. Only then, the optional detector.

Defaults aimed at a free T4: 128×128, batch size 8, 20 epochs for the baseline, 30 epochs for diffusion. These durations are not measurements. From the model size, a full pass is likely several GPU sessions, and diffusion **evaluation** is the slow part because each test image takes 50 sampling steps. Checkpoints are written every epoch to the Drive or Kaggle output disk.

256×256 is not the default.

## 7. Repository

See the tree in `README.md`. Raw LLVIP files, Weights & Biases keys, and checkpoints stay out of git.

## 8. Notebooks

| Notebook | Role |
| --- | --- |
| `notebooks/00_colab_or_kaggle.ipynb` | GPU check, paths, smoke test, then a real run |
| `notebooks/01_dataset_analysis.ipynb` | Counts, sizes, pair integrity. Writes nothing into the result tables by hand |
| `notebooks/02_baseline_experiment.ipynb` | Baseline training |
| `notebooks/03_diffusion_experiment.ipynb` | Diffusion at 100% |
| `notebooks/04_data_scarcity.ipynb` | Diffusion at 20% and 5% |
| `notebooks/05_evaluation.ipynb` | Metrics, gallery, web export |

## 9. Website

A separate Next.js case study under `web/`. It reads `web/public/data/experiments.json` produced by `scripts/export_web_results.py`.

The page order is the research story in the brief: RGB, the limit of RGB, a paired view, scarcity, the generator, baseline and diffusion panels, the 100 / 20 / 5 control, the failure gallery, the curves, and future work.

Until a job has run, every metric is `null`, every status is `pending`, and the interface says that the result is not available. Placeholder frames are labelled as placeholders. They are not presented as model outputs.

The rest of the personal portfolio is left unchanged.

## 10. Risks and limits

- LLVIP is mostly night-time surveillance. Daytime driving is outside the result.
- The official split can leak a scene across train and test.
- Equal epochs give the 5% run fewer optimizer steps. That choice is declared. It is not a hidden advantage.
- SSIM, PSNR, and LPIPS can all look acceptable on an image that dropped a pedestrian. That is why detection stays a separate question.
- LPIPS is an RGB perceptual metric applied to a thermal image.
- One seed is the core protocol. Variance across seeds is not estimated.
- Free GPU sessions stop. A run that does not checkpoint is not reproducible.
- ThermalGAN, InfraGAN, and PID already study RGB-to-infrared generation. This project does not claim that translation itself is new. The object is the scarcity comparison and the separation of similarity scores from perception.
