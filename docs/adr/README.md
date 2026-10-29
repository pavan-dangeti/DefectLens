# Architecture decision records

Significant decisions get a short ADR. An ADR records **one** decision, the context that
forced it, and what it costs — not a design document.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-mvtec-benchmark-visa-shipping.md) | MVTec AD is a research benchmark; shipped weights are VisA-derived (licence) | Accepted |
| [0002](0002-oracle-cloud-always-free-hosting.md) | Deploy to an Oracle Cloud Always Free VM (hosting) | Accepted |
| [0003](0003-patchcore-primary-padim-baseline.md) | Training-free PatchCore primary, PaDiM baseline (method) | Accepted |
| [0004](0004-paper-protocol-over-anomalib-defaults.md) | Evaluation follows the papers' protocol, not anomalib's defaults (evaluation) | Accepted |
| [0005](0005-backbone-by-measurement.md) | Backbone chosen by measured trade-off, not assumption (features) | Accepted |
| [0006](0006-threshold-calibration-at-target-frr.md) | Thresholds calibrated per category at a target false-reject rate (decisioning) | Accepted |
| [0007](0007-faiss-ivfpq-required-for-latency-slo.md) | FAISS IVF-PQ required to meet the latency SLO (optimisation) | Partly superseded by 0008 |
| [0008](0008-production-coreset-ratio.md) | Coreset ratio 0.01 by default; 0.1 per category where localisation matters more (optimisation) | Accepted |

## Format

```markdown
# ADR-NNNN: Title stating the decision

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Deciders:** who

## Context
What forced the decision, including the evidence and its source.

## Decision
What was decided, in the active voice.

## Consequences
Good, bad, and neutral — each honestly. "Bad" is not optional.

## Alternatives considered
Each with why it was rejected.
```

## When to write one

Write an ADR when a decision is expensive to reverse, constrains other work, or would be
non-obvious to a reader six months from now. Do not write one for a default, a library choice,
or anything a code comment can explain.

Supersede rather than edit: a changed decision gets a new ADR that references the old one, so
the history of what was believed and why stays readable.