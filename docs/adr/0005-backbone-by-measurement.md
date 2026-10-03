# ADR-0005: Backbone chosen by measured trade-off, not assumption

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

The brief offers WideResNet-50 or ResNet-18 and asks for the choice to be made on a measured
trade-off. The deployment target is a 2-core ARM VM, so backbone cost is a first-class
constraint, not a detail.

Available evidence, before measuring anything myself:

- **PatchCore's own ablation** found "results are mostly stable over the choice of different
  backbones" — ResNet-50 and WideResNet-50 land within 0.1–0.3 pp image AUROC of each other.
- **anomalib's published numbers** show a real gap: ResNet-18 at 0.973 image / 0.976 pixel
  AUROC versus WideResNet-50 at 0.980 / 0.980 on MVTec AD. Roughly **0.7 pp image AUROC for
  ~4× less compute**.

Feature dimensions differ by 4×, which propagates into memory bank size, coreset selection cost
and nearest-neighbour search cost — so the compute gap compounds rather than staying at 4×.

## Decision

**Implement both, measure both, and let the curve decide.** Phase 3 measures accuracy against
CPU latency and memory across backbones; Phase 4 names a recommended production configuration
with the measured numbers attached.

The default configuration is WideResNet-50 `layer2+layer3` (patch dimension 1536), because that
is what the published reference numbers use and the parity test must run against it. The
edge/low-latency configuration is expected to be ResNet-18 (patch dimension 384), but that
expectation is **not** baked in — it is a hypothesis the sweep either confirms or refutes.

## Consequences

**Good.** The backbone choice becomes a measured, defensible recommendation instead of a
default copied from a paper. If ResNet-18 lands within an acceptable tolerance, the edge
configuration gets ~4× the latency headroom, which may be the difference between meeting and
missing SLO-1 on two cores.

**Good.** The parity test against anomalib runs on the default WideResNet-50 configuration, so
it is not confounded by a backbone difference.

**Bad.** Two backbones to maintain, and the trade-off curves take longer to produce. The sweep
grid grows by one dimension.

**Bad, and the real risk:** a measurement can say "ResNet-18 is 4× faster and 0.7 pp worse",
which does not resolve the decision. SLO-1, memory budget and the business cost of an escape
have to arbitrate. That trade-off is written down in Phase 4 with numbers rather than settled
by preference, and the recommendation will name which constraint binds.

## Alternatives considered

- **WideResNet-50 only, chosen for benchmark comparability.** Simplest and safest for parity.
  Rejected as leaving most of the latency budget on the table on a 2-core host.
- **ResNet-18 only, chosen for latency.** Rejected because it forfeits parity with the
  published reference numbers, and because PaDiM's random dimension reduction is defined
  against ResNet-18 in the paper while the headline number uses WideResNet-50 — choosing one
  backbone for both models would compromise one of the two comparisons.
- **EfficientNet-B5.** Used for PaDiM's best reported image AUROC (0.979), but that figure
  comes from per-category backbone selection which the PatchCore authors state they could not
  reproduce, so it is not a like-for-like target. Highest compute cost. Deferred.
- **DINOv2 / self-supervised ViT features.** Reported strong on MVTec and a legitimate future
  phase. Deferred: a second large dependency and a new CPU cost profile, before the core
  trade-off is even measured. Noted, not dismissed.