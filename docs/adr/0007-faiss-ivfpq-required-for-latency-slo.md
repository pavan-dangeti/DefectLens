# ADR-0007: FAISS IVF-PQ is required to meet the latency SLO, not an optional optimisation

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

SLO-1 sets p95 latency for a single image on the deployment target at ≤ 1.5 s (stretch 800 ms).
That target has **2 ARM cores** (ADR-0002). Whether it is reachable depends entirely on the
nearest-neighbour search, not on the backbone.

Order-of-magnitude cost for one representative category — ~240 nominal training images,
~980 patches each at 224×224, patch dimension 1536:

- Memory bank: ~235,000 vectors.
- Query image: ~980 patch queries.
- Exhaustive L2 search: 980 × 235,000 × 1536 ≈ **3.5 × 10¹¹ multiply-accumulates per image**.

Two ARM cores sustain nowhere near that. A flat `IndexFlatL2` misses SLO-1 by a wide margin,
and no amount of backbone quantisation fixes it — the search dominates.

The published reference shows the accuracy cost is tolerable: PatchCore-100% with IVFPQ
reports 0.980 image / 0.979 pixel / 0.930 PRO, versus 0.991 / 0.981 / 0.934 for the full
exact index. That is roughly **1 pp of image AUROC for a large latency and memory win**.

## Decision

**FAISS with an IVF index and product quantisation is mandatory for the deployment
configuration**, entered from the start rather than added after the latency target is missed.
`nprobe` and PQ sub-quantiser count are configuration, not constants, because they are the
knobs that trade accuracy against latency.

Consequence for the design: **PQ quality is a first-class accuracy parameter**, not merely a
memory optimisation. Phase 4 measures accuracy against PQ configuration on the same axes as
every other trade-off, and the recommended production configuration names the chosen setting.

Exact flat search remains available and is used for the scientific evaluation runs, where
inference cost does not matter and exactness is the reference.

## Consequences

**Good.** SLO-1 becomes reachable. Quantised search is what makes a 235k-vector memory bank
tractable on two cores.

**Good.** Treating PQ as an accuracy parameter means its cost is measured and reported rather
than discovered in production.

**Bad.** About 1 pp of image AUROC against the exact index, on the published evidence. Real,
and it will be visible in the results.

**Bad.** Two code paths to test: exact for evaluation, quantised for serving. Mitigated by a
test asserting that both produce the same decisions within the measured tolerance at a given
operating point, so a serving-path change cannot silently alter decisions.

**Bad.** `nprobe` interacts with the IVF clustering quality, which depends on how many vectors
went into training the index. Few-shot models produce small banks, and a small bank trains a
weak IVF index — so the few-shot and serving configurations are coupled. Handled by measuring
few-shot performance at the serving configuration, not only at exact search.

## Alternatives considered

- **Flat exact index with a smaller coreset.** Rejected as a dead end: coreset reduces the
  bank, but the cost is linear in bank size and 1% coresets still leave ~2,350 vectors, which
  is ~2.8 × 10¹⁰ MACs per image — still too slow, and the published 1%-coreset numbers show
  accuracy does not improve to compensate.
- **Reduce input resolution to cut the query count.** Rejected: it degrades localisation, which
  is half the product, and the accuracy cost is certain while the latency benefit is only
  linear in patch count.
- **Move nearest-neighbour search to a GPU.** Out of scope: ADR-0002 commits to a CPU-only
  deployment with no GPU available.
- **Deploy a smaller model instead (ResNet-18).** Complementary, not an alternative — patch
  dimension drops 1536 → 384, a 4× saving on the same arithmetic. Included in the Phase 4 sweep
  alongside PQ rather than instead of it.