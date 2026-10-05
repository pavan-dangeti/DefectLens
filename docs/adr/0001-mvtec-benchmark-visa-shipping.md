# ADR-0001: MVTec AD is a research benchmark; shipped weights are VisA-derived

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

DefectLens must be evaluated on MVTec AD, the field's standard benchmark, because it is what
reviewers expect and what published comparisons use. MVTec AD is released under
**CC BY-NC-SA 4.0**; its dataset page states the data "is not allowed to be used for
commercial purposes", and adds that anyone unsure whether their application violates the
non-commercial clause should contact MVTec.

ShareAlike raises a question the licence does not answer: does a trained model, or the patch
feature memory bank derived from these images, count as "Adapted Material" and therefore
inherit the ShareAlike obligation? MVTec publishes no position. Guessing at a legal
conclusion is not an acceptable engineering shortcut.

VisA (Amazon) is **CC BY 4.0** — verified by SHA-256 matching the tarball's `LICENSE-DATASET`
against the canonical CC BY 4.0 legal code with no added restrictions. Commercial use and
redistribution are both permitted with attribution.

Note that the widely-used anomalib library mislabels VisA as CC BY-NC-SA 4.0, which is both
wrong and more restrictive than Amazon's actual terms. Anomalib's root `LICENSE` is
Apache-2.0, but that governs code only.

## Decision

Run and report the full MVTec AD benchmark as a **research result**. Ship **no
MVTec-derived artefact**. The model served by the public demo is trained on VisA.

Concretely:
- MVTec AD results appear in the README as benchmark numbers, produced by committed scripts.
- No coreset memory bank, Gaussian parameter set, or exported weight file derived from MVTec
  AD is included in a release, a container image, or the model registry.
- The dataset manifest records the licence for each dataset, and CI fails if an artefact
  derived from a non-commercial dataset is ever staged for release.

## Consequences

**Good.** DefectLens keeps the standard 15-category benchmark, so results are comparable to
the literature, while the served product sits on a licence-clean path. No unresolved legal
question attaches to anything shipped.

**Bad.** Two results tables instead of one, and a reader has to look twice at which numbers
correspond to the deployed model. Mitigated by labelling every table with its dataset and
licence.

**Neutral.** If an MVTec-derived model ever needs to ship commercially, this decision must be
revisited — either a direct licence from MVTec, or written confirmation that the artefacts are
not Adapted Material. That is a business decision, not an engineering one.

## Alternatives considered

- **Skip MVTec AD entirely.** Cleanest legally, but discards the benchmark every reviewer
  expects and most published comparisons use. Rejected.
- **Use both and treat NC as not applying to a portfolio project.** Fastest, but the design
  doc would carry an unresolved licence risk on its headline method. Rejected as accepting a
  known problem rather than avoiding it.
- **Email MVTec before building.** Worth doing only if Q4 in the design doc is answered yes.
  Not on the critical path for the current plan.