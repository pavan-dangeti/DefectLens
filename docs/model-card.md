---
license: cc-by-4.0
library_name: pytorch
tags:
  - anomaly-detection
  - industrial-inspection
  - patchcore
  - visa
pipeline_tag: image-classification
---

# DefectLens: PatchCore models for VisA product categories

Per-category PatchCore models that flag defects on manufactured parts. Each model is trained only on
images of **good** parts; no defective examples are used, in fitting or in calibration.

Source code: <https://github.com/pavan-dangeti/DefectLens> (release v1.0.0).

## Model details

- **Method.** PatchCore: frozen ImageNet-pretrained WideResNet-50 features from layers 2 and 3,
  average-pooled, concatenated (1536 dimensions per patch). The memory bank of nominal patch features
  is reduced by greedy k-centre coreset selection, and each patch is scored by its nearest neighbour in
  the bank.
- **Input.** An RGB image, resized to 256 by 256 and centre-cropped to 224 by 224, normalised with the
  ImageNet statistics. Grayscale, CMYK and palette images are converted.
- **Output.** An image anomaly score, a pixel-level score map, and a decision: `good` or `anomaly`, from
  a per-category threshold.
- **Threshold.** Calibrated on a held-out 10% of the good images to reject 1% of good parts (target
  false-reject rate 1%).
- **Coreset ratio.** 0.01 for every category in these weights (`configs/coreset.yaml`).
- **Artefact format.** `torch.save` dictionary with the coreset, the threshold and the configuration,
  loaded with `weights_only=True`. Each file has a SHA-256 sidecar that the service checks before loading.
- **Version.** 1.0.0.

## Training data

The VisA dataset from Amazon Science (`github.com/amazon-science/spot-diff`), 12 categories, released
under **CC BY 4.0** (verified on the dataset's official page, 2026-10-30). The weights use the good
images of the training split. They contain feature vectors computed from those images, not the images
themselves. The 80% of good images used for fitting and the 10% used for calibration are drawn with
seed 0.

These weights are derived from VisA and carry the same licence. Credit VisA when you reuse them, with
this citation, as given on the dataset's official page:

> Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar. *SPot-the-Difference
> Self-Supervised Pre-training for Anomaly Detection and Segmentation.* arXiv:2207.14315, 2022 (ECCV 2022).

**No weights fitted on MVTec AD are included.** MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial only,
verified on its official page). It is used here only as a research benchmark, and the benchmark numbers in
this card are not weights. This card and its weights are CC BY 4.0.

## Backbone

The feature extractor is timm's `wide_resnet50_2`, loaded with `pretrained=True`. timm's default weights
for this architecture are the `racm_in1k` tag (`timm/wide_resnet50_2.racm_in1k` on the Hugging Face Hub),
not the torchvision weights. Its licence is **Apache-2.0**. It was trained on ImageNet-1k with the
timm "RACM" recipe from *ResNet Strikes Back* (arXiv:2110.00476), on the Wide Residual Network architecture
(arXiv:1605.07146). This release does not redistribute the backbone weights: they are downloaded from the
Hub when the service starts.

ImageNet-1k is a dataset with its own terms, which as commonly stated restrict it to non-commercial research
use. The ImageNet terms were not re-fetched for this card. Whether those terms apply to weights trained on
it is not settled here. Confirm it before any commercial use of
the service.

## Evaluation

Three seeds (0, 1, 2) per category. Each seed re-draws the 80 / 10 / 10 split of the good training
images. The test set is the benchmark's own test split. Intervals are 95% Student-t intervals across seeds.

| Dataset | Categories | Image AUROC | Pixel AUROC | AUPRO | Escape rate at 1% FRR |
|---|---|---|---|---|---|
| VisA | 12 | 0.891 ± 0.017 | 0.954 ± 0.025 | 0.736 ± 0.078 | 0.446 ± 0.069 |

Per-category results are in the repository's `results/evaluation/tables.md`. The spread is wide: the
weakest VisA categories are `capsules` (image AUROC 0.676 ± 0.028) and `macaroni2` (0.679 ± 0.051).

These numbers are above the 0.5 chance level but below the published VisA PatchCore reference (Real-IAD
paper, Table 2: image AUROC 0.951, AUPRO 0.912, reproduced with Anomalib). The protocol differs in the
coreset ratio, the fit split and the AUPRO integration; see the repository README.

**The escape rate is high for several categories.** At a 1% false-reject target, `macaroni2` lets 0.91 of
its defects through and `macaroni1` 0.79. A decision from these models is not a substitute for human
inspection where a missed defect is costly.

## Intended use

- Screening: flagging parts for a human to inspect, to reduce the inspection load.
- Onboarding a new product line from good images only, as a first model before defect data exists.

## Out of scope

- Autonomous rejection or acceptance without human review.
- Categories other than the 12 VisA categories these weights were fitted on. A new category needs its own
  fit and calibration.
- Rotated, heavily blurred, or extremely lit images: these are flagged as low confidence by the service,
  but rotation is not detected.
- Defects the training data never showed on good parts: the models only know what nominal looks like.

## Limitations and known gaps

- PaDiM, the baseline, has no measured result in this release.
- The MMD distribution-shift check is not implemented; drift monitoring uses a KS test on scores and a
  decision-rate alert.
- The anomalib configuration has been compared only on MVTec AD, not on VisA.
- Latency and memory were measured on Apple Silicon. The 2-core ARM target host has not been measured.
- Thresholds are calibrated on 45 to 91 good images per VisA category. The achieved false-reject rate on
  held-out data departs from 1% for some categories.

## How to reproduce

```bash
uv sync
uv run defectlens data download visa
uv run defectlens fit --root data/visa --category <name> --out <registry-dir>
uv run python results/evaluation/run.py --dataset visa --seeds 0 1 2
```
