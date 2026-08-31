---
title: CPU code completion and GPU gate
date: 2026-08-30
summary: "Completed Phase 6 factory-oracle wiring and Phase 8 serving code, then stopped at target-GPU and live-serving evidence gates."
---

# CPU code completion and GPU gate

## What happened

Completed the latest plan's remaining CPU/code scope: CI and Colab scaffolding, real TRL factory-oracle wiring, authenticated vLLM Compose/Colab serving, benchmark/sweep/report tooling, and a custom agent-shaped BFCL workload. Review found six Important issues; fixes covered the pinned vLLM flag, exact quality/config pairing, benchmark artifact accumulation, clean missing-key CLI errors, Compose benchmark mounts, and failure-safe GRPO resource cleanup.

## Evidence

Final delegated gate: Ruff and formatting clean, mypy strict clean across 118 files, 302 CPU CI tests passed with 6 deselected, smoke passed, GRPO cleanup/factory tests passed, shell syntax passed, and base plus bench Compose rendering passed. Final re-review found no Critical or Important findings.

## Decision

Keep the plan in progress. Do not invent L4/A100 measurements, trained checkpoints, live Docker HTTP results, MTP acceptance, quantization quality, or BFCL scores. Serving profiles remain explicitly unchosen until paired target-GPU speed and quality evidence exists.

## Next steps

Run full target-GPU training/evaluation and serving measurements on L4/A100. Resolve the privacy-hook tooling block before adding `.env.example` and ignoring `.env`; remote clean-checkout CI also remains pending. The local full suite subsequently passed all 308 tests.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
