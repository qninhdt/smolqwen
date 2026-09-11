---
title: "Single-L4 vLLM serving benchmark and production profiles"
description: "Upgrade smolqwen to vLLM 0.29.0 and productionize its final Qwen3.5-2B checkpoint on one NVIDIA L4 with BF16/FP8 sweeps, capacity-under-latency analysis, bounded admission, and minimal observability."
status: in-progress
priority: P1
effort: "8-12 engineering days plus target-L4 sweep time"
branch: main
tags: [serving, vllm, l4, fp8, benchmark, prometheus, grafana]
blockedBy: []
blocks: []
created: 2026-09-11
---

# Single-L4 vLLM serving benchmark and production profiles

## Overview

Serve the final SFT+GRPO Qwen3.5-2B checkpoint through one vLLM V1 engine on one
NVIDIA L4, benchmark BF16 and FP8 configurations with official vLLM tooling, and
generate reproducible `latency`, `balanced`, and `throughput` operating profiles.
Capacity is maximized subject to TTFT/TPOT, reliability, and resource constraints;
`tok/s/user` is not a production-selection objective.

## Scope Contract

- Outcome: OpenAI-compatible streaming/tool-calling service behind nginx, measured
  BF16/FP8 capacity and latency curves, generated profiles, bounded overload, APC
  and cancellation evidence, Prometheus/Grafana, and measured final documentation.
- Constraints: exactly one L4 and one vLLM engine; benchmark client on the same node;
  vLLM 0.29.0 across project/runtime; no application-side batching or custom engine.
- Non-goals: distributed inference, replicas, autoscaling, application API wrapper,
  external KV storage, tool execution, tracing, logs stack, billing, or user database.
- Evidence boundary: code-complete phases may run locally; GPU acceptance and final
  profile selection require the final checkpoint on a real L4.

## Cross-Plan Relationship

This plan supersedes the serving-optimization work in
[`260828-1048-smolqwen-post-training-serving/phase-08-vllm-serving-optimization.md`](../260828-1048-smolqwen-post-training-serving/phase-08-vllm-serving-optimization.md).
Training and evaluation contracts remain owned by the parent plan.

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | [Upgrade the complete vLLM runtime](./phase-01-start.md) | Serving upgraded to vLLM 0.29.0 (Docker); training/colab intentionally kept on 0.26/torch 2.11. Runtime proof pending L4 |
| 2 | [Serving config and single-engine deployment](./phase-02-serving-config-and-deployment.md) | CPU-complete; runtime proof pending L4 |
| 3 | [Build and validate the FP8 artifact](./phase-03-fp8-artifact.md) | CPU-complete: source guards + deterministic manifest test-guarded; quantize + BF16/FP8 L4 probes pending L4 |
| 4 | [Add minimal Prometheus and Grafana observability](./phase-04-observability.md) | CPU-complete; runtime proof pending L4 |
| 5 | [Freeze the serving benchmark specification](./phase-05-benchmark-specification.md) | Spec + sweep/serve/bench params frozen and test-guarded (CPU); dataset rows/manifest and tool-schema preflight pending |
| 6 | [Execute resumable clean-start sweeps](./phase-06-sweep-execution.md) | CPU-complete: metric capture, upstream param files, and the sweep runner committed and test-guarded; L4 sweep execution pending |
| 7 | [Analyze capacity/latency and generate profiles](./phase-07-analysis-and-profile-generation.md) | CPU-complete: deterministic frontiers/selection/rendering test-guarded; measured profiles pending L4 sweep data (config profiles are labelled placeholders) |
| 8 | [Derive native vLLM admission limits](./phase-08-admission-control.md) | CPU-complete: admission derivation, config validation, and argv rendering test-guarded; overload 503 validation pending L4 |
| 9 | [Validate security and runtime behavior](./phase-09-integration-validation.md) | Pending |
| 10 | [Run the L4 study and publish measured artifacts](./phase-10-target-l4-results-and-documentation.md) | Pending |

## Global Success Criteria

- [ ] vLLM 0.29.0 is the serving-runtime pin (official Docker image), using its bundled Qwen3.5 GDN kernels rather than external wheels; training/colab intentionally remain on vLLM 0.26.0 / torch 2.11 until an upstream Torch 2.13 causal-conv1d/FLA wheel exists.
- [ ] One L4 runs one authenticated nginx→vLLM service with streaming and tool calls.
- [ ] BF16 and FP8 W8A8+FP8-KV families pass startup and runtime smoke tests.
- [ ] Fixed BFCL tool-calling traffic is used identically for every sweep point.
- [ ] Raw runs include capacity, p95/p99 TTFT/TPOT, goodput when SLOs exist, failures, preemptions, and KV diagnostics.
- [ ] Invalid points (`failed_requests > 0` or `preemptions > 0`) are excluded.
- [ ] Throughput-vs-p95-TTFT and throughput-vs-p95-TPOT frontiers are generated.
- [ ] Three operating profiles are selected and emitted without manual retuning.
- [ ] Native queue limits produce controlled 503 overload without OOM or crash.
- [ ] APC reuse, SSE cancellation, authenticated routes, and metrics scraping are proven.
- [ ] README and serving/performance docs contain measured L4 results only.

## Unresolved Questions

- Resolved for serving: vLLM 0.29.0 ships its own Qwen3.5 GDN causal convolution
  Triton kernel (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`), so the
  serving image needs no external `causal-conv1d`. The upgrade is therefore scoped
  to the serving Docker runtime only (base image `vllm/vllm-openai:v0.29.0`).
- Deferred for training/colab: no upstream Torch 2.13 `causal-conv1d`/FLA wheel
  exists yet, so the training ABI intentionally stays on torch 2.11 + vLLM 0.26.0
  per user decision ("training thì kệ"). Revisit whole-project unification when an
  upstream wheel ships; do not mix ABIs or run an implicit runtime CUDA build.

<!-- slug: single-l4-vllm-serving-benchmark -->
