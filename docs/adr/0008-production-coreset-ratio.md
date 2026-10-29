# ADR-0008: Coreset ratio 0.01 by default; 0.1 per category where localisation matters more

- **Status:** Accepted (2026-10-30)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

PatchCore keeps a memory bank of nominal patch features, reduced by greedy k-centre coreset selection to
a fraction `r` of the full bank. `r` sets three things at once: the bank's size on disk and in memory,
the time to score an image, and how well the bank covers the nominal distribution, which drives
localisation. The design fixes three service-level objectives that bear on it:

- **SLO-1:** `/inspect` p95 at most 1.5 s on the 2-core ARM VM (design §3). The VM has not been measured.
- **SLO-3:** serving RSS at most 2 GB.
- **SLO-4:** memory bank on disk at most 512 MB per category, with 200 MB as the target.

The cost and accuracy measurements below were taken on Apple Silicon. Latency is predict-only (backbone
and nearest-neighbour search, no HTTP), on nominal test images, median and p95 of 30 single-image calls.
Memory is from fits and serving processes measured separately (`results/evaluation/coreset_cost.py`,
`results/evaluation/serving_rss.py`). Accuracy is from the committed three-seed runs and sweep.

### Measured cost (VisA, seed 0, four categories)

| Category | Ratio | Bank entries | Artefact (MB) | Predict median (ms) | Predict p95 (ms) | Fit time (s) |
|---|---|---|---|---|---|---|
| candle | 0.01 | 5,644 | 33.1 | 70.2 | 76.6 | 406.5 |
| candle | 0.1 | 56,448 | 330.8 | 220.4 | 243.6 | 2,574.6 |
| capsules | 0.01 | 3,394 | 19.9 | 40.1 | 42.5 | 128.4 |
| capsules | 0.1 | 33,947 | 198.9 | 133.6 | 147.0 | 862.7 |
| macaroni1 | 0.01 | 5,644 | 33.1 | 60.8 | 63.6 | 428.4 |
| macaroni1 | 0.1 | 56,448 | 330.8 | 216.3 | 238.4 | 2,477.2 |
| macaroni2 | 0.01 | 5,644 | 33.1 | 56.3 | 61.9 | 410.3 |
| macaroni2 | 0.1 | 56,448 | 330.8 | 207.6 | 220.1 | 2,441.4 |

Serving process (candle, 0.01): peak resident memory 996 MB after loading the registry and scoring 30
images (`results/evaluation/serving_rss.json`). The 0.1 figure is an estimate, not a measurement: the
bank grows by about 300 MB, which puts the peak near 1.3 GB, under the 2 GB limit.

### Measured accuracy

- **VisA, four categories, seed 0** (coreset sweep, `results/evaluation/sweep/coreset_0.1/` against the seed-0
  committed results): going from 0.01 to 0.1 raises AUPRO by +0.034 on average (capsules +0.067,
  macaroni1 +0.013, macaroni2 +0.054, candle +0.001). It raises image AUROC by +0.010 on average (capsules
  +0.035, macaroni1 +0.013, macaroni2 −0.009, candle +0.000).
- **MVTec AD, three categories, seed 0** (diagnostic, `results/evaluation/diagnostic/`): the mean image
  AUROC gain from 0.1 is +0.080 (screw +0.186, toothbrush +0.053, leather +0.000). Only the screw gain is
  clearly outside that category's seed-to-seed range; the diagnostic is single-seed for each variant.

## Decision

1. **Default ratio: 0.01** for every category (`configs/coreset.yaml`). This is the production
   configuration the service is measured against.
2. **0.1 is available per category** through `configs/coreset.yaml`. Choose it for a category when
   localisation matters more than latency and disk, and when a refit and recalibration are acceptable.
   Changing a ratio changes the decision threshold, so refit the category and recalibrate (ADR-0006).

What the trade is, in numbers:

- **0.01:** a bank about 10 times smaller (5.6k entries against 56k on the VisA categories tested), and
  predictions 3.1 to 3.7 times faster. Its cost is accuracy on localisation: AUPRO lower by about 0.034 on
  average on the four VisA categories tested.
- **0.1:** better localisation (AUPRO +0.034 on that sample; image AUROC +0.08 on the MVTec three-category
  diagnostic, single seed), at the price of a bank 10 times larger and predictions 3 to 4 times slower.

Latency against SLO-1: at 0.1 the measured p95 is 220 to 244 ms on Apple Silicon. That is within the
1.5 s limit unless the 2-core VM is about six times slower than this machine. At 0.01 the p95 is 62 to
77 ms, within the limit unless the VM is about twenty times slower. The VM has not been measured, so
these are the limits the measurements imply, not an SLO verdict. **Measuring p95 on the VM at both ratios
is a precondition for any move of the default to 0.1.**

Memory against SLO-4 and SLO-3: both ratios meet the 512 MB disk limit. The 200 MB stretch target is met
at 0.01 on every category tested, and at 0.1 only on capsules (199 MB). Serving RSS at 0.01 is 996 MB,
under 2 GB; at 0.1 it is estimated near 1.3 GB.

## Why not 0.1 as the default

The accuracy gain is real but modest on the sample: +0.034 AUPRO and +0.010 image AUROC on VisA, against a
10-fold bank and a 3-to-4-fold slower prediction. Making it the default would commit every category to
that cost. Categories where localisation drives the decision can opt in individually, and the default can
move once the VM measurement is in hand.

## Consequences

- **ADR-0007 is partly superseded.** Its premise was that a flat exact index cannot serve a bank of about
  235k vectors within the latency SLO. That holds for the full, uncoreset bank. At 0.01 the bank is about
  5.6k entries, and exact search meets the measured latency on Apple Silicon. FAISS IVF-PQ is therefore not
  needed for the production configuration. It stays required for any configuration whose bank exact search
  cannot serve within SLO-1, including a 0.1 default if the VM measurement rules out the margin above.
- **Thresholds depend on the ratio.** A category switched to 0.1 must be refitted and recalibrated before it
  serves decisions (ADR-0006).
- **The accuracy comparison is thin.** The sweep covers four VisA categories at seed 0, and the MVTec
  diagnostic covers three categories at seed 0. Neither supports a claim about the full benchmark. Running the
  sweep at 0.1 across all categories and seeds is the measurement that would firm up the trade-off.

## Alternatives considered

- **0.1 as the default.** Rejected for now: a 10-fold bank and 3-to-4-fold slower predictions for a gain of
  +0.034 AUPRO on the sample, with the VM latency unmeasured.
- **Intermediate ratios (for example 0.03).** Not measured. Left as a future option once the VM measurement
  is in hand.
- **FAISS IVF-PQ at 0.1.** Not needed to meet the latency limits implied by these measurements, and not
  built. Revisit with the VM numbers.
