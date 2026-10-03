# ADR-0006: Thresholds are calibrated per category at a target false-reject rate

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

AUROC is threshold-free, which makes it a good research metric and a useless production one. A
quality engineer needs a verdict, and the business constraint is asymmetric: rejecting a good
part wastes material and line time; shipping a defective part risks the customer. The brief
frames the target as "reject at most 1% of good parts".

Two failure modes to avoid:

- **Picking 0.5 on a normalised score.** Meaningless, because the score distribution is not
  calibrated and shifts with model, coreset ratio and category.
- **Calibrating on data the model was fitted on.** Nominal training images are memorised by the
  memory bank, so their scores are optimistically low. A threshold derived from them will be
  too low and will reject far more than the target fraction of genuinely unseen good parts.

## Decision

For each category *c* and target false-reject rate *f*:

1. Fit the model on the **`fit` split** only (80% of nominal training images).
2. Score every image in the **`calibration` split** (10%, disjoint from `fit`).
3. Set `τ_c` to the *f*-quantile of those scores — the score above which at most *f* of good
   parts are rejected.
4. Freeze `τ_c`. Evaluate the published test set **once**. Report the escape rate: the
   fraction of anomalous test images scoring below `τ_c`.
5. Report the achieved false-reject rate measured on a third disjoint **`holdout` split**
   (10%), which is never used to choose anything.

The split is created by the data loader from a seeded permutation, not by the caller, so
calibration data cannot leak into fitting by accident.

**When the calibration split is too small to support the requested target, DefectLens reports
the widest target it can support and says so explicitly**, rather than printing a confident
number. Concretely: a 10% calibration split of a ~240-image MVTec category is ~24 images, and
the 99th percentile of 24 samples is not meaningfully estimable. Categories with fewer nominal
images get a larger calibration fraction.

Few-shot models draw their *n* images from `fit` only, so a few-shot model and its calibration
set never overlap.

## Consequences

**Good.** The reported operating point is one a business can act on: "at a 1% false-reject
rate this model misses X% of defects on this category." AUROC hides that trade-off; this
exposes it.

**Good.** Reporting the widest supportable target instead of a nominal one is the difference
between an honest system and a demo that overstates itself. On small categories this will
sometimes mean reporting 5% where 1% was requested.

**Good.** Three-way disjoint splits make leakage structurally impossible rather than a review
checklist item.

**Bad.** Thresholds are per category, so onboarding a new product category requires a
calibration pass before it can serve decisions. That is inherent to the approach, not a defect
in it — a per-category threshold is the price of a per-category model.

**Bad.** Fewer images are available for fitting than a naive 80/20 split would use.

**Bad.** At small positive escape rates, estimates need Wilson intervals to be honest. Escape
rate on VisA is measured over 100 anomalous images per category, so a 3% escape rate is three
images; the interval is wide and must be shown.

## Alternatives considered

- **Per-category threshold from the fitting split.** Cheaper, no extra split needed. Rejected:
  optimistically biased by memory-bank memorisation, which is precisely the failure this
  decision exists to prevent.
- **A single global threshold across categories.** Simpler to operate. Rejected: score
  distributions differ per category, so a global threshold implies a wildly varying false-reject
  rate — unacceptable when the business constraint is expressed per product.
- **Quantile of the calibration scores with no minimum sample check.** Would always produce a
  number. Rejected: it produces a *confident* number, which is worse than a caveated one.
- **Report escape rate at a range of targets (0.5%, 1%, 5%) rather than one.** Complementary,
  not an alternative — this is the default, and the curve makes the single-point choice
  unnecessary for most decisions.