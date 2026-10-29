# DefectLens

Find defects on manufactured parts from a single photo. Trained only on images of **good**
parts, because that is how real production lines work — defects are rare, varied, and unknown
in advance.

> **Status: pre-implementation.** This repository currently contains the Phase 0 design
> documentation. There are no results yet, and nothing here should be read as a performance
> claim. See [`docs/design.md`](docs/design.md).

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
and scored by nearest-neighbour search — implemented from the paper and verified against a
reference implementation. PaDiM serves as the baseline.

## Approach in one paragraph

An ImageNet-pretrained backbone is used frozen, with no training, and mid-level patch features
are extracted from two layers. Those features from all good training images form a memory bank,
reduced by greedy k-center coreset subsampling. At inference, each patch in a new image is
scored by its distance to the nearest memory bank entry, the map is smoothed, and the image is
flagged when its score exceeds a per-category threshold calibrated on held-out good images.
Because no defective examples are needed, a new product line can be onboarded from a handful of
good images — which is measured and reported, not assumed.

## Run locally

```bash
uv sync
uv run defectlens init          # workspace dirs + local MLflow tracking
uv run defectlens versions
uv run defectlens data download visa        # fetch + verify + manifest
uv run defectlens data verify visa
uv run ruff check . && uv run ruff format --check .
uv run mypy src/defectlens
uv run pytest --cov --cov-report=term-missing
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