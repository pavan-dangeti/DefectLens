# ADR-0002: Deploy to an Oracle Cloud Always Free VM

- **Status:** Accepted (2026-10-04)
- **Deciders:** Pavan Sai Kumar Dangeti

## Context

The demo must run on free hosting: a CPU-only inference service, a demo UI, and a periodic
drift job, at $0, and it has to stay awake enough to be shown to someone.

The originally proposed host, Hugging Face Spaces on free CPU, **no longer offers free compute
to new accounts**. Their documentation states:

> "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a
> paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free
> personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."

CPU Basic is still listed with a `FREE` price, but that price is unreachable for a new compute
Space. The free allowance is now 2 ZeroGPU Gradio Spaces, and this project needs CPU.

Other options measured against the requirement:

| Option | Blocking limitation |
|---|---|
| Render free | 0.1 CPU / 512 MB — cannot hold the model; ephemeral disk; no workers; 15-min spin-down |
| Vercel Hobby | 4.5 MB request payload cap blocks industrial images; 4 h active CPU/month |
| Netlify Free | Same payload cap; ~30 GB-hours total compute |
| Fly.io / Railway / Koyeb | Free allowances removed or too small for an always-on service |
| HF Spaces PRO | $9/month — violates the $0 budget |

Oracle Cloud Always Free is the only option that is genuinely always-on at no charge:
**2 ARM OCPU / 12 GB RAM / 200 GB block volume / 10 TB egress per month**, with unrestricted
background processes, persistent disk and no cold start.

## Decision

Deploy everything to **one Oracle Always Free Ampere A1 VM**: the inference API, the demo UI
and the drift job in a single container behind Caddy for TLS and reverse proxy. Scheduled work
that does not need to touch the VM lives in GitHub Actions (2,000 free minutes/month on a
private repo) instead.

Two operational consequences are accepted as part of this decision:

1. **The VM is aarch64.** Every dependency needs an ARM64 build. Multi-arch Docker images are
   built in CI from Phase 1, not retrofitted in Phase 5.
2. **A heartbeat keeps the box out of Oracle's idle-reclamation band** (CPU p95 < 20%, network
   < 20%, memory < 20% over 7 days). This is a real constraint: a genuinely idle demo box is
   precisely what that rule targets. The heartbeat doubles as an external availability probe.

## Consequences

**Good.** $0, always-on, 12 GB RAM is ample for a PatchCore memory bank plus an ONNX CPU
runtime in one container, real background workers, persistent disk, and 10 TB of egress — far
more than a demo needs.

**Bad.** A credit card is required for identity verification (no charge without an explicit
upgrade). A1 provisioning frequently fails with "out of host capacity", and the home region is
permanent, so region selection matters. Ops are manual: no zero-downtime deploys, no managed
log UI, no automatic TLS renewal. ARM64 dependency gaps are a real risk.

**Bad, and worth stating plainly:** the allowance was halved from 4 OCPU / 24 GB to 2 / 12 GB
in June 2026. This decision is built on an allowance that has already been reduced once and
could be reduced again. The runbook must document how to redeploy elsewhere if that happens.

## Alternatives considered

- **HF Spaces PRO at $9/month.** Simplest deployment by a wide margin, and it would avoid
  ARM64 entirely. Rejected because it breaks the $0 budget. Revisit if the owner approves a
  small recurring cost — this is the single change that would most reduce Phase 5 risk.
- **Render free with a heavily reduced model.** Would require cutting resolution and coreset
  ratio hard enough to fit 512 MB, with a 15-minute spin-down in front of an audience. Rejected
  as optimising the wrong variable.
- **Split hosting across two providers.** More failure modes and two things to keep alive, for
  no benefit given 12 GB is sufficient.