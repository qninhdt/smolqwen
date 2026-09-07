---
title: Complete in-training benchmark eval contracts
date: 2026-09-03
summary: "Added timeout, explicit weight-version metrics, and SFT checkpoint sidecars; CPU gates pass while L4 validation remains blocked."
---

# Complete in-training benchmark eval contracts

## What happened
The benchmark callback declared a wall timeout and weight version, but the timeout was not enforced and the weight version was not present in the metric sink payload. SFT also lacked a result file beside the checkpoint it evaluated. The local machine cannot settle the remaining vLLM/L4 measurements.

## Decision
Use the shared runner for a Python-level main-thread wall timeout with exception isolation and elapsed-time SIGALRM restoration. Emit the weight identity as `bench_weight_version`. Write SFT outcomes to `checkpoint-N/bench_eval.json`, using a durable output-directory fallback for the base step-zero anchor and missing checkpoints; label step zero as `base`. Keep release-hook failures visible because continuing with an awake or unsynchronized engine is unsafe. Keep sidecars as local evidence while W&B scalars remain the durable metric path.

## Evidence
Focused callback/SFT tests: 22 passed. `make test-ci`: 477 passed, 18 deselected, 2 known TRL warnings. `make check` and `make smoke` passed. Plan validation passed at 5/10 phases and 85/105 tasks (80%).

## Next steps
Run Phase 10 on one L4/A100: real-weight agreement, throughput and stability, GRPO/SFT cost, sleep/envelope, Colab non-TTY, and final BFCL. Resolve the open vLLM adapter fixture and parent-process VRAM guard findings before relying on those GPU criteria.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
