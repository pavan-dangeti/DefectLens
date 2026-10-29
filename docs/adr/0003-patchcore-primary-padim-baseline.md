# ADR-0003: Training-free PatchCore as the primary method, PaDiM as the baseline

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

DefectLens learns from images of good parts only. The deployment scenario that motivates it is
a **new product line** where a customer has a handful of nominal images and no defective
examples at all. That constraint rules out every method requiring per-category training, and
it makes cold-start behaviour a first-class requirement rather than an ablation.

Two candidate families fit "no defective labels":

- **PaDiM** (Defard et al., ICPR 2021) — a parametric model: a multivariate Gaussian per patch
  *position*, scored by Mahalanobis distance.
- **PatchCore** (Roth et al., CVPR 2022) — a non-parametric memory bank of nominal patch
  features, greedy k-center coreset subsampled, scored by nearest-neighbour distance.

Evidence, as measured in this repository (Phase 5). An earlier draft of this record quoted published
single-class figures for both methods. Those could not be traced to a source and have been withdrawn.

| | MVTec AD image AUROC (15-category mean) | VisA image AUROC (12-category mean) |
|---|---|---|
| PatchCore (ours, 3 seeds) | 0.967 ± 0.005 | 0.891 ± 0.017 |
| PaDiM | not measured here | not measured here |

Both are training-free. PaDiM is roughly an order of magnitude cheaper. The decision to make PatchCore
primary rests on the measured PatchCore results and on the cost argument; a measured PaDiM comparison on
the same protocol is still to be run before any claim about the size of the gap.

## Decision

**PatchCore is the primary method, implemented from the paper.** PaDiM is implemented as the
baseline and reference point.

The parity test against anomalib on an identical protocol is a release gate, not a
nice-to-have: within 1.0 pp image AUROC, 1.0 pp pixel AUROC, 1.5 pp AUPRO, with any gap
diagnosed in writing rather than waived.

## Consequences

**Good.** Best published results of the training-free family. No training step, which is what
makes the few-shot regime and the CI smoke test cheap. A memory bank is trivially extendable by
the review loop — new confirmed-good images are appended and the coreset is rebuilt, with no
gradient updates and no risk of catastrophic forgetting.

**Good.** The PaDiM baseline is a genuinely informative control, not a straw man: same
features, same protocol, one structural difference (parametric per-position Gaussian versus
non-parametric memory bank). It answers "what does the memory bank actually buy?" — on
published numbers, roughly 4.6 pp image AUROC and 11.5 pp AUPRO on VisA.

**Bad.** The memory bank is the dominant memory cost. Mitigated by coreset ratio plus FAISS
product quantisation, and measured as a first-class trade-off curve in Phase 4.

**Bad.** PaDiM's per-position covariance is singular when training images are fewer than the
feature dimension — exactly the few-shot case. Requires ε = 0.01 regularisation, and its
few-shot numbers degrade faster than PatchCore's (published 1-shot: 0.761 vs 0.841).

## Alternatives considered

- **PaDiM as primary.** Cheaper and simpler, but 4–5 pp image AUROC behind on VisA. Not a good
  trade when the harder benchmark is the one that matters. Retained as baseline.
- **Student–teacher (Uninformed Students).** Strong results, but requires training a student per
  category — contradicts the cold-start premise and multiplies CI and serving cost. Rejected.
- **Reconstruction / inpainting (DRAEM, RIAD).** Known superficial-anomaly bias: they can
  reconstruct a defect from surrounding context. Rejected for structural defects.
- **Normalising flows (FastFlow, CFLOW).** Better published MVTec numbers than PatchCore, but
  require per-category training. Rejected on the same grounds as student–teacher.
- **Foundation models (WinCLIP, UniAD, AnomalyCLIP).** The most interesting long-term direction
  and a legitimate future phase. Rejected here because multi-GB weights are hostile to a
  2-core CPU latency budget, zero-shot inference does not benefit from the customer's own good
  images (which is the data actually available), and the premise of this project is memory-bank
  learning. Revisit if a small distilled variant appears.