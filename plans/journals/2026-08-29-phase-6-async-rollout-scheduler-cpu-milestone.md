---
title: Phase 6 async rollout scheduler CPU milestone
date: 2026-08-29
summary: Implemented and CPU-verified the ready-queue rollout path; real TRL target-GPU A/B remains pending.
---

# Phase 6 async rollout scheduler CPU milestone

## What happened

Implemented the Phase 6 rollout package: factory adapter, ready-queue scheduler, TRL `rollout_func`, drift-aware mask/logprob assembly, vLLM generation boundary, profiler, metrics, CLI benchmark, docs, and report artifact.

Independent review found real-schema signature failures, a polling-cycle false abort, incomplete worker-crash blast-radius replacement, misleading serial-baseline labeling, destroy-time misattribution, and a late-create cleanup leak. Each defect was reproduced, fixed, regression-tested, and re-reviewed.

## Verified outcome

- 33/33 focused Phase 6 tests pass.
- 269/269 repository tests pass; one pre-existing verifier-fixture deprecation warning remains.
- Ruff, formatting, strict mypy, and `git diff --check` pass.
- Transformers schema generation succeeds for all 191 vendored environment classes and all 3,548 tool schemas.
- Reviewer confirms all findings resolved with no remaining code blocker.

## Evidence boundary

The serial factory adapter is a semantic oracle only, not TRL's batched turn-synchronous performance baseline. The colocated-vLLM A/B, weight-sync and prefix-cache measurements, profile sweep, and episodes/hour comparison remain pending because the local RTX 3050 has 4 GB VRAM and there is no Phase 3 SFT checkpoint. Those results require separately constructed trainers on L4/A100.

## Next steps

1. Complete the Phase 3 target-GPU SFT run and checkpoint.
2. Wire the separate Phase 7 factory-oracle and production trainers.
3. Run L4/A100 rollout A/B and sweep active-pool/concurrency/KV settings.
4. Fill `artifacts/rollout/ab_report.md` with measured rows and close Phase 6 only after its remaining criteria pass.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
