# Changelog

All notable changes to DefectLens. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-29

First release: PatchCore anomaly detection on good-only images, with a serving API, a reproducible
three-seed evaluation on MVTec AD and VisA, drift monitoring, and an operations runbook.

### Added

- **Detection.** PatchCore (WideResNet-50, layers 2 and 3, greedy k-centre coreset, nearest-neighbour
  scoring), with a per-category threshold calibrated to a 1% false-reject target. PaDiM is implemented
  as the baseline; no PaDiM result has been measured yet.
- **Evaluation.** A resumable runner over 27 categories (15 MVTec AD, 12 VisA) and seeds 0 to 2. Each
  seed re-draws the good-image split. Results are summarised with 95% t-intervals, rendered to the README,
  and reproducible from committed scripts.
- **Serving.** `POST /inspect` (one image), `POST /inspect/batch` (up to 16 images, with a status per
  image), `GET /ready`, `GET /health`, `GET /metrics` (Prometheus), and review endpoints for operator
  feedback.
- **Input handling.** Size and pixel limits checked from the header before decoding; corrupt, truncated
  and decompression-bomb inputs are refused with 400; grayscale, CMYK and palette images are accepted;
  extreme lighting and blur are flagged as low confidence.
- **Traffic limits.** Per-client rate limit counted per image (60 a minute; IPv6 bucketed by /64), with a
  bounded client map. Overloaded inference answers 503 with `Retry-After` instead of queueing.
- **Drift monitoring.** Two-sample KS test on recent scores against the calibration scores (0.1% level),
  and a decision-rate alert on the flag rate (four standard errors from the 1% target). Exported as
  Prometheus gauges.
- **Artefacts.** SHA-256 sidecars checked before loading; loading uses `weights_only`.
- **Observability.** JSON access logs with request IDs, no client addresses, and the URL path for
  unknown routes.
- **Load testing.** A k6 script for overload behaviour and for paced capacity runs. Measured capacity
  is in the README.
- **CI.** Lint, strict type checking, tests with an 85% coverage gate, a dependency audit (`pip-audit`),
  and an image scan (Trivy).
- **Coreset ratio per category.** `configs/coreset.yaml` sets the default to 0.01, with per-category
  overrides. The production default is 0.01 (ADR-0008): a bank about 10 times smaller and predictions
  about 3 times faster than 0.1, for an AUPRO cost of about 0.034 on the VisA sample.
- **Runbook and model card.** `docs/runbook.md` and `docs/model-card.md`.

### Changed

- `defectlens serve` binds `127.0.0.1` by default. The container image binds `0.0.0.0` explicitly.
- The VisA dataset config lists the category as `candle`, matching the folder name.
- `configs/default.yaml`, which nothing read, is removed.

### Negative results (kept out of the release)

- Static INT8 quantisation with calibration is about three times slower than fp32 and does not improve
  accuracy consistently. fp32 is kept.
- Dynamic INT8 through ONNX gave no speed-up.

### Known limitations

- Rotation outside the training distribution is not detected.
- Our VisA image AUROC (0.891) is below the published Real-IAD reference (0.951). The gap is not explained
  by one setting on the categories tested. See the README.
- Escape rates at the 1% false-reject target are high for several categories; see the README.
- The rate limiter and drift windows are in-process.
- PaDiM, MMD drift checks, and the anomalib configuration on VisA are not measured.
