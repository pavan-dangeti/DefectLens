# DefectLens

Find defects on manufactured parts from a single photo. Trained only on images of **good**
parts, because that is how real production lines work — defects are rare, varied, and unknown
in advance.

> **Status: Phase 5 (serving and evaluation) complete, pending review.** The results below come from
> committed scripts and committed result files. Phase 6 (demo, model cards, runbook, release) has not
> started and waits for approval. The design is in [`docs/design.md`](docs/design.md).

## Read this first

| Document | What it is |
|---|---|
| [`docs/design.md`](docs/design.md) | Requirements, SLOs, evaluation protocol, system design, alternatives, hosting plan, threat model, risks, budget |
| [`docs/adr/`](docs/adr/) | Short records of the decisions that shaped the system, and what each one costs |

## What gets built

Per product category, from images of good parts only:

- an **anomaly score** for the image,
- a **heatmap** localising where the part departs from normal,
- a **pass/fail decision** calibrated against a business target such as "reject at most 1% of
  good parts".

The primary method is PatchCore — a memory bank of nominal patch features, coreset-subsampled
and scored by nearest-neighbour search — implemented from the paper. PaDiM serves as the baseline.

## Approach in one paragraph

An ImageNet-pretrained backbone (WideResNet-50) is used frozen, with no training, and mid-level
patch features are extracted from layers 2 and 3. Those features from the good training images form
a memory bank, reduced by greedy k-center coreset subsampling. At inference, each patch in a new
image is scored by its distance to the nearest memory-bank entry, the map is smoothed, and the image
is flagged when its score exceeds a per-category threshold. The threshold is calibrated on held-out
good images to reject at most 1% of them. Because no defective examples are needed, a new product line
can be onboarded from a handful of good images, which is measured below.

## Results

Every category was run with three seeds (0, 1, 2). Each seed re-draws the 80 / 10 / 10 split of the
good training images into a memory-bank fit set, a threshold calibration set and a holdout set. The
test set is the benchmark's own test split. Intervals are 95% Student-t intervals across seeds. The
protocol is `evaluate_category` in `src/defectlens/eval/benchmark.py`, run by
`results/evaluation/run.py`; the raw runs are in `results/evaluation/raw/`.

| Dataset | Categories | Image AUROC | Pixel AUROC | AUPRO | Escape rate at 1% FRR |
|---|---|---|---|---|---|
| MVTec AD | 15 | 0.967 ± 0.005 | 0.962 ± 0.004 | 0.852 ± 0.014 | 0.116 ± 0.073 |
| VisA | 12 | 0.891 ± 0.017 | 0.954 ± 0.025 | 0.736 ± 0.078 | 0.446 ± 0.069 |

Means are over categories; the intervals are across seeds of that category mean. Per-category
tables, with their intervals, are in [`results/evaluation/tables.md`](results/evaluation/tables.md).
The per-category spread is wide: on VisA, `capsules` (0.676 ± 0.028) and `macaroni2` (0.679 ± 0.051)
are the weakest categories, and on MVTec AD `screw` is the weakest (0.816 ± 0.088).

Few-shot: mean image AUROC over categories, as the number of good images used to fit the memory bank
grows.

| Good images used to fit | 1 | 5 | 10 | 50 |
|---|---|---|---|---|
| MVTec AD | 0.744 | 0.830 | 0.865 | 0.934 |
| VisA | 0.580 | 0.731 | 0.735 | 0.826 |

## Comparison with a published VisA PatchCore result

| Result | Image AUROC | Pixel AUPRO | Source |
|---|---|---|---|
| DefectLens, 12-category mean, 3 seeds, coreset ratio 0.01 | 0.891 ± 0.017 | 0.736 ± 0.078 | `results/evaluation/summary.json` |
| PatchCore, 12-category mean, as published | 0.951 | 0.912 | Real-IAD paper, Table 2 and §5.1 ([arXiv:2403.12580](https://arxiv.org/abs/2403.12580)) |

**We do not match the published numbers.** The gap is −0.060 image AUROC and −0.176 AUPRO. We have
not claimed a match anywhere in this repository.

Protocol differences:

- **Implementation.** The paper says it reproduced PatchCore "with open-sourced Anomalib". It does not
  give the version. We checked anomalib 2.7.0 against its `main` branch: the PatchCore defaults are the
  same (WideResNet-50, layers 2 and 3, coreset ratio 0.1, 9 neighbours, 256 resize with no crop). The
  paper's version is unknown, so its defaults may differ.
- **Backbone.** Ours is WideResNet-50, layers 2 and 3. The paper does not state its backbone.
- **Coreset ratio.** Ours is 0.01. The paper does not state its ratio. The diagnostic below shows what
  the ratio does.
- **Resolution.** Both resize to 256 and centre-crop to 224.
- **Memory-bank fit set.** Ours fits on 80% of the good training images, keeping 10% for threshold
  calibration and 10% as a holdout. The paper does not state its split.
- **AUPRO.** Ours integrates the false-positive rate up to 0.3, using 50 thresholds. The paper does not
  state its integration limit or threshold count.
- **Seeds.** Ours is the mean of three seeds. The paper does not state how many seeds it used.

## What explains the gap: coreset ratio and fit set

**Coreset ratio, VisA, seed 0.** Raising the coreset ratio from 0.01 to 0.1 on four categories:

| Category | Image AUROC change | AUPRO change |
|---|---|---|
| capsules | +0.035 | +0.067 |
| macaroni1 | +0.013 | +0.013 |
| macaroni2 | −0.009 | +0.054 |
| candle | +0.000 | +0.001 |

The ratio raises AUPRO on all four categories, and image AUROC on two. It does not close the gap on
its own, and it is a single seed on four of twelve categories.

**Anomalib-default diagnostic, MVTec AD, seed 0.** Each row changes one setting from our configuration
(coreset 0.01, fit on 80% of the good images, 224 crop). The last row changes all three to match
anomalib 2.7.0's defaults. Changes are relative to seed 0 of our committed run. Source:
[`results/evaluation/diagnostic/`](results/evaluation/diagnostic/).

| Setting changed | leather: image AUROC / AUPRO | screw: image AUROC / AUPRO | toothbrush: image AUROC / AUPRO | Mean image AUROC change | Mean AUPRO change |
|---|---|---|---|---|---|
| ours (baseline, seed 0) | 1.0000 / 0.9557 | 0.7868 / 0.8726 | 0.9472 / 0.7736 | — | — |
| coreset 0.01 → 0.1 | 1.0000 (+0.0000) / 0.9519 (−0.0038) | 0.9725 (+0.1857) / 0.9128 (+0.0402) | 1.0000 (+0.0528) / 0.8015 (+0.0279) | **+0.080** | +0.021 |
| fit on all good images (100%, not 80%) | 1.0000 (+0.0000) / 0.9514 (−0.0043) | 0.7258 (−0.0611) / 0.8872 (+0.0146) | 0.9833 (+0.0361) / 0.7727 (−0.0009) | −0.008 | +0.003 |
| no crop (256, not 224) | 1.0000 (+0.0000) / 0.9620 (+0.0063) | 0.7692 (−0.0176) / 0.8839 (+0.0113) | 0.8806 (−0.0667) / 0.8222 (+0.0486) | −0.028 | +0.022 |
| anomalib default (all three) | 1.0000 (+0.0000) / 0.9621 (+0.0064) | 0.9684 (+0.1816) / 0.9171 (+0.0445) | 0.9222 (−0.0250) / 0.8404 (+0.0668) | +0.052 | +0.039 |

Seed-to-seed range of the baseline image AUROC: leather 1.000 to 1.000, screw 0.787 to 0.855,
toothbrush 0.867 to 0.986.

**Finding.** The coreset ratio explains most of the change toward anomalib's configuration on these
three MVTec categories. On its own it adds +0.080 mean image AUROC, more than the full anomalib
configuration (+0.052), because the other two changes subtract on the categories where they move
the score (fit on all good images on screw, no crop on toothbrush). The screw gain of about +0.19 is
well outside that category's seed range. The toothbrush changes sit inside it, so we do not attribute
them. These are single-seed runs for each variant, and the diagnostic covers MVTec AD only: we have
not run the anomalib configuration on VisA, so the VisA gap is not explained by any one setting.

## Escape rate at a 1% false-reject target

Escape rate is the share of defective test images that the threshold lets through. The threshold is
set on held-out good images so that at most 1% of good parts are rejected. For inspection, the escape
rate is the number that matters: each escape is a defective part that ships.

Averages over categories, 3 seeds, 95% t-interval:

- MVTec AD (15 categories): **0.116 ± 0.073**
- VisA (12 categories): **0.446 ± 0.069**

The averages hide a wide spread. In MVTec AD, 9 of 15 categories have escape rates of 0.044 or below,
and `screw` escapes 0.66 of its defects. In VisA, 8 of 12 categories escape more than 0.3, and
`macaroni2` escapes 0.91.

What this means for inspection:

- At the 1% setting, a category with an escape rate of 0.66 lets two-thirds of its defects through.
  The system is not a substitute for a human check on that category. A 1% reject target protects good
  parts, and it does so by accepting more defects.
- The threshold is calibrated on few images. MVTec AD calibration and holdout sets hold 6 to 40 good
  images per category; VisA holds 45 to 91. The false-reject rate measured on the holdout set departs
  from the 1% target: `toothbrush` averages 0.22 (maximum 0.50), and `bottle` is 0.00. One extra
  false reject moves the holdout rate by 1/n: by 0.17 on the 6-image MVTec AD holdout and by 0.02 on a
  40-image one.
- The defective test set is 30 to 141 images per MVTec AD category and 100 per VisA category, so the
  per-category intervals are wide. Compare categories only with that in mind.
- A deployment should choose its false-reject target from the cost of rejecting a good part against
  the cost of an escape, report the escape rate at that target, and calibrate on more good images than
  the benchmark uses.

## Capacity

Measured on a Mac with an Apple Silicon CPU, one bottle model, two inference slots, the server bound to
`127.0.0.1`. Each level is a constant-arrival-rate run for two minutes. Latency is for accepted requests
only. Source: [`results/loadtest/`](results/loadtest/), built by `results/loadtest/capacity.py`.

| Offered load | Images/s | Requests | Accepted p50 | p95 | p99 | Refused (503) |
|---|---|---|---|---|---|---|
| 50% of 23.4/s | 11.7 | 1,404 | 45.2 ms | 47.2 ms | 51.7 ms | 0.00% |
| 80% of 23.4/s | 18.7 | 2,245 | 44.9 ms | 47.6 ms | 50.3 ms | 0.00% |
| 100% of 23.4/s | 23.4 | 2,809 | 74.6 ms | 90.3 ms | 93.6 ms | 5.87% |

The 23.4 images/s figure is an estimate: two slots divided by the saturated mean of 85.5 ms per image.
That mean includes contention between the two slots. Uncontended inference is about 45 ms, so the
refusal-free level is between 80% and 100% of 23.4 images/s. Measured levels are the only capacity
figures to quote.

**Overload.** With four clients and no pacing, 85,966 requests went to `/inspect` in two minutes. 2,109
were accepted and 83,857 were refused with 503 from the inference slots. There were no 5xx responses
other than 503, no crashes, and no transport failures.

**Rate limit.** By default each client may send 60 images a minute. A single load-test client exceeds
that almost at once, so the capacity runs lift the limit on the server. The limiter's behaviour under
load is tested separately.

## Production configuration: coreset ratio

The coreset ratio sets the memory bank's size and the prediction cost. The default is **0.01** for every
category, set in [`configs/coreset.yaml`](configs/coreset.yaml). Choose **0.1** for a category when
localisation matters more than latency and disk. The decision and its numbers are in
[ADR-0008](docs/adr/0008-production-coreset-ratio.md).

Cost on Apple Silicon, VisA, seed 0, four categories. Predictions are single-image, predict-only (no HTTP),
median and p95 of 30 calls. Source: [`results/evaluation/coreset_cost/coreset_cost.md`](results/evaluation/coreset_cost/coreset_cost.md).

| Category | Coreset ratio | Bank entries | Artefact (MB) | Predict median (ms) | Predict p95 (ms) | Fit time (s) | Peak RSS while fitting (MB) |
|---|---|---|---|---|---|---|---|
| candle | 0.01 | 5644 | 33.1 | 70.2 | 76.6 | 406.5 | 8664 |
| candle | 0.1 | 56448 | 330.8 | 220.4 | 243.6 | 2574.6 | 8841 |
| capsules | 0.01 | 3394 | 19.9 | 40.1 | 42.5 | 128.4 | 6770 |
| capsules | 0.1 | 33947 | 198.9 | 133.6 | 147.0 | 862.7 | 6195 |
| macaroni1 | 0.01 | 5644 | 33.1 | 60.8 | 63.6 | 428.4 | 8355 |
| macaroni1 | 0.1 | 56448 | 330.8 | 216.3 | 238.4 | 2477.2 | 8552 |
| macaroni2 | 0.01 | 5644 | 33.1 | 56.3 | 61.9 | 410.3 | 8904 |
| macaroni2 | 0.1 | 56448 | 330.8 | 207.6 | 220.1 | 2441.4 | 9088 |

The fit peak is offline cost, paid once per fit, and it is about the same at both ratios because the full
bank is built before reduction.

Serving process, candle at 0.01: peak resident memory **996 MB** after loading the registry and scoring 30
images (`results/evaluation/serving_rss.json`). At 0.1 it is estimated near 1.3 GB: the bank grows by about
300 MB. That estimate is not measured.

What the trade is:

- **0.01:** a bank about 10 times smaller, and predictions about 3.1 to 3.7 times faster. The cost is
  localisation: AUPRO is about 0.034 lower on average on these four categories.
- **0.1:** better localisation: AUPRO +0.034 on average on these four categories (seed 0), and image
  AUROC +0.08 on the three-category MVTec diagnostic (seed 0, single seed). The cost is a bank 10 times
  larger and predictions 3 to 4 times slower.

Latency against the design's SLO-1 (p95 at most 1.5 s on a 2-core ARM VM): 0.1's p95 of 220 to 244 ms on this
machine is within that limit unless the VM is about six times slower. 0.01's p95 of 62 to 77 ms is within it
unless the VM is about twenty times slower. The VM has not been measured. Measure it before changing the
default to 0.1.

Changing a category's ratio changes its decision threshold: refit the category and recalibrate it before it
serves decisions.

## INT8 quantisation

Static post-training INT8 was tried as a way to cut backbone latency. It is a negative result, and the
service keeps the fp32 backbone.

The backbone (WideResNet-50, whole feature extractor) was quantised with FX graph mode on the qnnpack
engine (Apple Silicon). Activation ranges were calibrated on 32 nominal training images from the seed-0
fit split, never test images. Accuracy was measured with the same `evaluate_category` protocol at
coreset 0.01 and seed 0, and compared with the committed fp32 results. Latency is the median of 30
backbone passes on an otherwise idle CPU. Source: [`results/optimisation/int8_static/`](results/optimisation/int8_static/),
built by `results/optimisation/int8_table.py`.

| Category | Image AUROC fp32 → INT8 | AUPRO fp32 → INT8 | Pixel AUROC fp32 → INT8 | Escape at 1% FRR fp32 → INT8 | Backbone latency fp32 → INT8 (ms) | Feature cosine |
|---|---|---|---|---|---|---|
| MVTec AD / screw | 0.787 → 0.823 | 0.873 → 0.865 | 0.969 → 0.970 | 0.76 → 0.73 | 29.2 → 89.4 | 0.865 |
| MVTec AD / toothbrush | 0.947 → 0.911 | 0.774 → 0.746 | 0.976 → 0.974 | 0.20 → 0.23 | 28.9 → 89.5 | 0.901 |
| VisA / capsules | 0.671 → 0.697 | 0.546 → 0.530 | 0.952 → 0.954 | 0.66 → 0.65 | 28.9 → 88.7 | 0.984 |

**Verdict.**

- **Slower, not faster.** In all three categories the INT8 backbone pass takes about 89 ms against 29 ms
  for fp32, roughly three times the latency. This alone rules it out for serving.
- **Accuracy does not move in a consistent direction.** Image AUROC changes by +0.036, −0.036 and
  +0.026, and AUPRO by −0.008, −0.028 and −0.016. The screw and toothbrush swings are inside those
  categories' seed-to-seed ranges (screw 0.787 to 0.855; toothbrush 0.867 to 0.986), so they are not
  evidence of a loss or a gain. These are single-seed runs on three categories.
- **Features drift.** Cosine similarity between fp32 and INT8 embeddings is 0.87 to 0.98, lowest on
  screw.
- **The earlier dynamic INT8 path was also negative.** The Phase-4 ONNX dynamic-quantisation export
  (`results/optimisation/summary.json`) ran at 61 ms against 59 ms for fp32, with a feature cosine of
  0.92.

**Decision.** Serving keeps the fp32 backbone. Quantisation would need a different graph or a
per-channel scheme that quantises residual additions more gently, and that is not worth doing unless
backbone latency becomes the bottleneck, which it is not at the measured 45 ms per image.

## Drift monitoring

The serving layer compares live anomaly scores with the calibration scores stored in each artefact,
over the most recent 200 scored images of each category. Two alerts are exported as Prometheus gauges
(`defectlens_score_drift_ks`, `defectlens_flag_rate_window`, `defectlens_drift_alert`).

- **Score drift.** The two-sample Kolmogorov–Smirnov statistic between the window and the calibration
  scores. The alert is set at the 0.1% level, using the large-sample critical value.
- **Decision rate.** The share of the window flagged as anomalous. The expected share is the target
  false-reject rate, 1%, because the threshold was calibrated to reject that share of good parts. The
  alert is set when the observed share leaves a band of four standard errors around that expectation.

The decision-rate alert assumes the mix of good and defective parts in production is stable. A
sustained change in the flag rate is a reason to investigate, not proof that the model has failed.
Artefacts written before this change carry no calibration scores, and categories with fewer than 20
calibration scores (few-shot onboarding) are too small for a meaningful KS test. Both load and serve
without drift monitoring. The monitor is unit-tested on synthetic streams: it stays quiet on the reference
distribution, raises a score alert on a shifted one, and raises a decision alert when the flag rate
moves. It has not been measured on real production traffic.

## Serving

`defectlens serve` exposes `POST /inspect` (one image), `POST /inspect/batch` (up to 16 images, with a
status for each), `GET /ready`, `GET /health`, `GET /metrics` (Prometheus) and the review endpoints.

- Uploads are limited to 20 MB, and the body is refused before parsing when its declared size is too
  large. Images are checked from their header before decoding, so a decompression bomb is refused
  without being decoded. Corrupt and truncated files get 400.
- Lighting extremes and blur are flagged on the response (`low_confidence`, `reasons`). They are
  heuristics, not calibrated out-of-distribution tests. Rotation is not detected; a strict test marks
  it as a known gap.
- Each client may send 60 images a minute, counted per image, with IPv6 clients bucketed by /64. When
  the inference slots are full the service answers 503 immediately with `Retry-After`, rather than
  queueing.
- Model artefacts carry a SHA-256 sidecar. A missing or mismatched sidecar stops the artefact from
  loading.
- Access logs are JSON, one line per request, with the request ID. They carry no client addresses. For
  requests to unknown routes they carry the URL path, without the query string.
- The rate limiter is in-process. Several workers would need a shared store to enforce one budget.

## Run locally

```bash
uv sync
uv run defectlens init          # workspace dirs + local MLflow tracking
uv run defectlens versions
uv run defectlens data download visa        # fetch + verify + manifest
uv run defectlens data verify visa
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/defectlens
uv run pytest --cov --cov-fail-under=85 --cov-report=term-missing
```

Serve the API:

```bash
uv run defectlens serve --registry <dir-of-artefacts>   # binds 127.0.0.1:8000
```

`defectlens serve` listens on loopback only unless you ask otherwise, with `--host` or
`DEFECTLENS_HOST`. The container image binds `0.0.0.0` so the published port is reachable from the
host:

```bash
docker build -t defectlens .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/registry:/app/artifacts/registry" \
  defectlens serve --host 0.0.0.0 --registry /app/artifacts/registry
```

The image's default command is `defectlens serve --host 0.0.0.0` with no model registry, so `/ready`
returns 503 until you mount one and pass `--registry`, as above.

## Known limits

- Rotation out of distribution is not detected (strict xfail in `tests/unit/test_failure_modes.py`).
- The rate limiter and the drift windows are in-process. Several workers would need shared state.
- The diagnostic uses one seed per variant. The INT8 comparison is also single-seed.
- PaDiM is implemented as the baseline, but no PaDiM result has been measured in this repository.
- The published VisA figures quoted in earlier drafts of the design doc and ADR-0003 could not be
  traced to a source and have been withdrawn. The one published VisA figure used here is the Real-IAD
  PatchCore row above.
- Trivy runs in CI only; it has not been run locally.
- The `docker compose` workflow listed in `AGENTS.md` has no compose file yet.
- Phase 6 (hosted demo, model cards, runbook, release) has not started.

## Licences

The benchmarks carry different licences, and this affects what ships:

| Dataset | Licence | Use here |
|---|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (15 categories) | CC BY-NC-SA 4.0 — **non-commercial** | Research benchmark only. No derived artefact ships. |
| [VisA](https://github.com/amazon-science/spot-diff) (12 categories) | CC BY 4.0 | Licence-clean path for the served model and the public demo |

Rationale in [ADR-0001](docs/adr/0001-mvtec-benchmark-visa-shipping.md). Neither dataset is
committed to this repository; both are fetched by script with checksum verification.

## Cost

$0. Free-tier model APIs, free-tier GPUs for training, and a free-tier VM for hosting. Any
action that would cost money is escalated for approval first. See
[§13 of the design document](docs/design.md#13-budget-and-cost-policy).
