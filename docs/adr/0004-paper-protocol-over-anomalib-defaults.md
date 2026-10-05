# ADR-0004: Evaluation follows the papers' protocol, not anomalib's defaults

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

anomalib is the standard reference implementation, and its published PatchCore numbers
(0.980 image / 0.980 pixel AUROC on MVTec AD) are **lower than the PatchCore paper's**
(0.991 / 0.981). A project that claims to reimplement PatchCore has to explain that gap, and
the explanation determines whether the gap is a bug or a protocol difference.

Two independent, mechanical differences account for it:

1. **Centre crop.** Both the PatchCore paper (§4.1) and the PaDiM paper (§IV-B) specify
   `Resize(256×256)` then `CenterCrop(224×224)`. anomalib's default pre-processor is
   `Resize(256)` with **no** crop unless `center_crop_size` is explicitly passed. Its pixel
   metrics are then computed at 256×256, because the pre-processor applies the same transform
   to image and ground-truth mask alike.
2. **PaDiM configuration.** anomalib's default PaDiM is **ResNet-18 with 100 random
   dimensions** — the localisation-only ablation in the PaDiM paper, whose image-level AUROC
   the paper never reports. The 0.975 pixel / 0.953 image headline comes from WideResNet-50
   with 550 dimensions. Comparing anomalib's default PaDiM to the published PaDiM number is
   comparing two different models.

## Decision

**DefectLens's protocol follows the papers**: `Resize((256,256))` → `CenterCrop(224)` →
ImageNet normalisation, no augmentation, bilinear score-map upsampling with Gaussian σ = 4,
pixel metrics at 224×224, class-averaged rather than pixel-pooled.

The anomalib parity test then **explicitly configures anomalib into this same protocol**
rather than running it on its defaults. This isolates implementation differences from
preprocessing differences, which is the only way the comparison can mean anything.

Both numbers get reported: DefectLens's own results under the paper protocol, and the
delta against anomalib under the identical protocol.

## Consequences

**Good.** Direct comparability with every published PatchCore and PaDiM number. The parity
test becomes a genuine test of my implementation rather than a test of whether two
configurations agree.

**Good.** The protocol divergence is documented as a finding in its own right. Anyone
comparing anomalib's published numbers to the papers' numbers is currently comparing across
protocols, and saying so is a real contribution rather than a footnote.

**Bad.** Two protocols exist in the ecosystem and mine matches only one of them. Results will
not match anomalib's README tables unless a reader knows to apply the crop. Mitigated by
stating the protocol in the README next to every number and shipping the config that produces
it.

**Bad.** Running anomalib under a non-default protocol means maintaining two configuration
paths. Accepted: it is the difference between a real parity test and a vacuous one.

## Alternatives considered

- **Follow anomalib defaults.** Simpler, one configuration path. Rejected: makes every
  comparison to the papers meaningless and hides the gap the parity test exists to find.
- **Evaluate pixel metrics at original image resolution.** Conceptually cleaner. Rejected: it
  introduces an interpolation-order confound between the score map and the ground-truth mask,
  and diverges from both anomalib and the papers, so there is no published number to compare to.
- **Report only anomalib's numbers.** Would mean the project measures the library, not the
  method. Rejected.