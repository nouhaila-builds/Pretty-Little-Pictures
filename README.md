# Pretty Little Pictures

Version française. English version below.

## Version française

L’apparence peut tromper — même une machine.

Étude de cas de recherche indépendante. Elle demande si la fidélité pixel, la similarité perceptuelle et l’utilité pour la détection de piétons se dégradent ensemble lorsqu’un traducteur du visible vers l’infrarouge est entraîné sur moins de paires alignées.

Le protocole a été exécuté. Les scores mesurés sont dans `web/public/data/experiments.json`. La page lit ce fichier. Les images, les points de contrôle et les archives de résultats ne sont pas dans git.

## Question

Lorsqu’un traducteur du visible vers l’infrarouge est entraîné sur de moins en moins de paires alignées, la fidélité pixel, la similarité perceptuelle et l’utilité pour la détection de piétons se dégradent-elles ensemble ?

## Motivation

Les images visibles sont abondantes. Les images thermiques ne le sont pas. Générer la modalité manquante n’est utile que si l’image synthétique conserve la structure de scène dont dépend un système en aval. La similarité à la vérité terrain et l’utilité pour la perception sont mesurées séparément.

## Jeu de données

LLVIP (Jia et al., ICCV Workshops 2021) : des paires alignées visible–infrarouge, surtout de nuit, avec des boîtes de piétons. KAIST et FLIR ADAS ont été comparés et n’ont pas été retenus. La comparaison est dans `docs/research-design.md`. Les images ne sont pas dans git. Voir `data/README.md`.

## Méthode

- Référence : U-Net L1, du RVB vers un canal thermique.
- Modèle principal : diffusion conditionnelle. Le RVB est concaténé à l’image thermique bruitée. Le réseau prédit le bruit. L’échantillonnage utilise 50 pas DDIM.
- Fractions : 1 % ⊂ 5 % ⊂ 20 % ⊂ 100 % du pool d’entraînement, après une coupe de validation gelée. Le jeu de test ne change pas.
- Protocole : le même nombre d’époques pour chaque fraction. Les fractions plus petites font moins de pas d’optimisation. Ce choix est déclaré dans les notes de configuration.
- Métriques : PSNR, SSIM, LPIPS, et AP piéton d’un détecteur entraîné sur l’infrarouge réel puis gelé.

Le conditionnement par modèle de vision-langage, le flow matching, le rectified flow et la vidéo ne sont pas implémentés. Le protocole exact est dans `docs/implementation_specification.md`. Les hypothèses sont dans `hypotheses/preregistered_hypotheses.md`.

## Arborescence

```text
pretty-little-pictures/
├── configs/          YAML des expériences
├── docs/             conception de la recherche
├── notebooks/        entrée Colab ou Kaggle, puis un notebook par étape
├── scripts/          préparation, entraînement, évaluation, export, test de fumée
├── src/plp/          données, modèles, entraînement, métriques
├── experiments/      un dossier par exécution, créé sur la machine GPU
├── web/              étude de cas, alimentée seulement par le JSON exporté
└── data/README.md    comment placer LLVIP
```

Le paquet Python est `plp`, sous `src/`. Un paquet de premier niveau nommé `data` entrerait en collision avec le répertoire de données.

## Installation

```text
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/export_web_results.py
```

Le test de fumée utilise des tenseurs aléatoires. Ses nombres ne sont pas des résultats.

## GPU

Utiliser Kaggle ou Colab. Ouvrir `notebooks/00_colab_or_kaggle.ipynb`.

1. Choisir un environnement GPU.
2. Pointer `DATA_ROOT` vers LLVIP.
3. Lancer le chemin de fumée dans le notebook avant une exécution complète.
4. Entraîner, puis évaluer, puis exporter. Recopier `experiments/` et `web/public/data/experiments.json`.

Weights & Biases est facultatif. Il ne démarre que si `WANDB_API_KEY` est défini et `WANDB_MODE=online`. La clé n’est pas stockée dans le dépôt.

```text
python scripts/prepare_dataset.py --root data/llvip
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/diffusion_full.yaml
python scripts/train.py --config configs/diffusion_20.yaml
python scripts/train.py --config configs/diffusion_05.yaml
python scripts/evaluate.py --checkpoint experiments/plp-unet-100/best.pt
python scripts/export_web_results.py
```

Ajouter `--smoke` à `scripts/train.py` pour un contrôle de deux lots sur les fichiers réels.

## État

| Expérience | État |
| --- | --- |
| Référence U-Net, 100 % | Enregistrée |
| Diffusion, 100 % / 20 % / 5 % / 1 % | Enregistrée |
| Détecteur sur infrarouge réel, puis sur infrarouge synthétique | Enregistré |

## Résultats

Les nombres sont dans `web/public/data/experiments.json` et sur la page de l’étude de cas. Sur cette exécution, le PSNR est resté bas et plat, tandis que le SSIM, le LPIPS et l’AP piéton n’ont pas évolué ensemble. L’infrarouge généré est en 128×128. Les scores de détection comparent cette sortie aux images LLVIP d’origine après que les deux ont été placées sur un canevas dont le grand côté fait 320 pixels.

## Analyse des échecs

Les images à forte erreur peuvent être classées à partir des résidus après l’évaluation. Les étiquettes de catégorie et les observations écrites restent vides tant qu’une personne ne les a pas relues.

## Limites

Scènes surtout de nuit. La coupe officielle ne sépare pas les scènes. Le LPIPS a été entraîné sur des photographies ordinaires. Un nombre égal d’époques signifie un nombre inégal de pas. Une seule graine.

## Ensuite

Un détecteur sur images thermiques réelles, comparé à des images synthétiques. Une légende gelée comme condition supplémentaire. Le flow matching ou le rectified flow comme second générateur, seulement une fois cette comparaison établie. La vidéo, sur un jeu de données qui est réellement une vidéo.

## Étude de cas

`web/` lit le JSON exporté. Il ne contient pas de scores saisis à la main.

## Reproductibilité

Graine 42. Les listes de fractions, la validation et la galerie sont écrites une fois par `prepare_dataset.py`. Chaque point de contrôle conserve la configuration, la version de torch, le dispositif et l’historique de perte.

---

## English version

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
