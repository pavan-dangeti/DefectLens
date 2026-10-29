# Runbook

Operating DefectLens: starting it, checking it, reading its signals, responding to them, and rolling
back a model. The service is one process. It serves PatchCore models from a directory of artefacts, and
it holds its inference slots and rate-limit counters in memory.

## 1. Start it

Run the service on the host, bound to loopback unless it is behind a reverse proxy:

```bash
uv run defectlens serve --registry <dir-of-artefacts>          # 127.0.0.1:8000
```

In a container, publish the port and mount the registry:

```bash
docker run --rm -p 8000:8000 -v "$PWD/artifacts/registry:/app/artifacts/registry" \
  defectlens serve --host 0.0.0.0 --registry /app/artifacts/registry
```

Settings come from the environment (`DEFECTLENS_HOST`, `DEFECTLENS_PORT`, `DEFECTLENS_MAX_UPLOAD_MB`)
or from `.env`. The defaults are loopback, port 8000 and a 20 MB upload limit.

## 2. Confirm it is healthy

| Check | Healthy | Meaning when it is not |
|---|---|---|
| `GET /health` | 200 | The process is up but not necessarily able to serve. |
| `GET /ready` | 200 with `ok: true` and the loaded categories | 503 with `no models loaded`: the registry directory is empty or was not passed with `--registry`. |
| `GET /metrics` | Prometheus text, including `defectlens_models_loaded` | Scrape failure: check the process and the port. |

A registry that contains a file without a valid checksum sidecar, or one that fails to load, stops the
process at startup. It does not start serving with a partial set of models. The error names the file.

## 3. Signals

Metrics (all on `/metrics`, all in a private registry):

| Metric | Meaning | Act when |
|---|---|---|
| `defectlens_requests_total{route,status}` | Requests by route and status | 5xx other than 503 appear, or 503 rises sharply (see §4). |
| `defectlens_request_seconds`, `defectlens_inference_seconds` | Request and model latency | p95 approaches the 1.5 s SLO on the target host. |
| `defectlens_decisions_total{category,decision}` | Decisions by category | The anomaly share changes sharply without a known cause (see §5). |
| `defectlens_anomaly_score` | Score distribution | Shifts (see §5). |
| `defectlens_score_drift_ks{category}` | KS statistic of the last 200 scores against calibration | Alert, see §5. |
| `defectlens_flag_rate_window{category}` | Share of the last 200 images flagged | Alert, see §5. |
| `defectlens_drift_alert{category,kind}` | 1 while an alert is active: `score` or `decision` | Alert, see §5. |
| `defectlens_rejected_total{reason}` | Images refused as invalid | Rises after a client change. |
| `defectlens_low_confidence_total{reason}` | Images flagged for lighting or blur | Rises: check the capture setup. |
| `review_cases_pending` | Flagged cases waiting for a label | Keeps growing: the review queue is not being worked. |

Logs: one JSON line per request on the `defectlens.access` logger, with `request_id`, `method`,
`route`, `status` and `duration_ms`. No client addresses are logged. For unknown routes the URL path is
logged, without the query string. Use the `X-Request-ID` response header to join a client report to a
log line.

## 4. Responding to refusals

| Symptom | Likely cause | Action |
|---|---|---|
| 503 on `/inspect`, `Retry-After: 1` | Both inference slots are busy. This is the designed overload response. | Clients should retry after a second. If it persists, add a replica; do not raise the slot count beyond the CPU cores. |
| 429, `Retry-After: 60` | A client is over 60 images a minute. Counted per image, IPv6 bucketed by /64. | Confirm the client is not batching badly. Raise the limit only with a reason. |
| 413 | Upload over the limit, or a batch over 16 images | Reject at the client. |
| 404 `category … not loaded` | No artefact for that category in the registry | Check the registry directory and the category name. |
| 400 on one image | Corrupt, truncated, too small, or over the pixel cap | The client's file is bad; no action on the service. |

## 5. Drift alerts

Two alerts are computed per category over the last 200 scored images:

- **Score drift** (`kind="score"`): the KS statistic between the window and the calibration scores is
  above the 0.1% critical value.
- **Decision rate** (`kind="decision"`): the share of the window flagged differs from the 1% target by more
  than four standard errors.

When an alert is active:

1. Check whether the input has changed: a new camera, lighting, a new product batch, or a new client.
   Look at a sample of recent images that were flagged or scored high, through the review endpoints.
2. Check the category's false-reject rate on fresh good images, if any are available.
3. If the input changed legitimately, add confirmed-good images through the review loop and refit the
   category (§6). If it did not, treat the model as suspect: hold the category's decisions for review.

The decision-rate alert assumes the mix of good and defective parts is stable. A change in defect rate
alone can raise it, so a decision alert is a reason to look, not proof of a fault.

An artefact without calibration scores, or a category with fewer than 20, runs without drift monitoring.
Its gauges are absent. Check `/metrics` for `defectlens_score_drift_ks{category=…}` before relying on
drift for a category.

## 6. Refit a category

Refitting reads confirmed-good images from the review log and calibrates a new threshold:

```bash
uv run defectlens fit --root <dataset-root> --category <name> --out <new-registry-dir> \
  --review-log artifacts/reviews.jsonl
```

The coreset ratio comes from `configs/coreset.yaml`, which defaults to 0.01 for every category. Pass
`--coreset-ratio` to override it for one run. `fit` writes `<category>.pt` and its SHA-256 sidecar, and
prints the threshold and the number of fit images.

Check the printed threshold and fit-image count against what you expected before the new registry goes
live, then switch the registry (§7). The fit command does not report an escape rate; to measure one, run
the benchmark protocol on the refitted category.

## 7. Roll back a model

Artefacts are versioned by file: the registry directory holds one `<category>.pt` and one
`<category>.pt.sha256` per category. To roll back:

1. Stop the service.
2. Replace the `<category>.pt` and its sidecar with the previous pair from the release archive.
3. Start the service and check `/ready` lists the category.
4. Score a known image for that category and compare with the recorded decision.

Keep the previous registry directory until the new one has been live for a full review cycle.

## 8. Upgrade the service

1. Run the full test suite, the strict type check and the dependency audit on the release commit.
2. Build the image and run it with a registry, as in §1.
3. Check `/ready` and score one image from each category against the recorded decisions.
4. Replace the running process. There is no in-place migration: the registry format is the contract.

## 9. Known limits to know about

- Rotation outside the training distribution is not detected.
- The rate limiter and the drift windows are in process. Several workers would each keep their own counts,
  so a per-client limit would be multiplied by the worker count.
- The 2-core VM latency target (SLO-1) is measured on the target host before each release; the numbers in
  the README were measured on a laptop.
- Refitting needs the training images on disk. The service does not fetch datasets.
