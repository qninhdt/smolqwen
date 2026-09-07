---
phase: 10
title: "L4 validation"
status: in_progress
priority: P1
effort: "1.5d (gated on a Colab card)"
dependencies: [4, 5, 6, 7]
---

# Phase 10: L4 validation

## Overview

Settle on a real L4 every claim a CPU suite cannot: agreement with the baseline,
throughput with its cost split, batch-composition stability with confounds
separated, and the SFT envelope by live measurement.

**This phase is now the only open work in the plan.** Every other phase's code is
landed and its CPU criteria are green; the remaining card measurements and final
test run close here rather than in their own phase files:

The partial L4 evidence from 2026-09-06 is recorded in
[`reports/l4-nonreasoning-training-smoke-20260906.md`](./reports/l4-nonreasoning-training-smoke-20260906.md).
It closes the Phase 2 sleep/adapter checks, the SFT 32K shared-card smoke, and the
Colab non-TTY check. The full GPU suite passes 11 tests with one intentional
envelope skip. The remaining items below still require the longer
training/evaluation sequence.

| From | Criterion | Step below |
|---|---|---|
| 4 | Agreement with the re-captured baseline, per-task disagreement list | 2 |
| 4 | Throughput split and batch-composition stability | 3–4 |
| 5 | GRPO in-training eval cost within 10% of training wall time | 5 |
| 6 | SFT in-training eval cost within 10% | 6 |
| 10 | Live failure isolation under the trainer | 7 |
| — | Base BFCL reference before training; SFT/SFT+RL once after selection | 9 |

The local development card is a 3.7 GB RTX 3050 and vllm is absent from CI by
construction (`pyproject.toml:33-36`), so none of these can be approximated here.
`tests/test_vllm_adapter_capability.py` and
`tests/test_sft_bench_eval_memory_guard.py` are written and `gpu`-marked; they are
the assertions, and this phase supplies the hardware.

## Requirements

- Functional: `evaluate` agrees with the re-captured baseline within Phase 1's
  tolerance, with the per-task disagreement list published.
- Functional: throughput recorded alongside the generation / tokenization /
  environment split.
- Functional: stability measured at concurrency 1 and N, with environment-timeout
  counts reported per run.
- Functional: `grpo/bench_*` and `sft/bench_*` match `evaluate` at the **recorded
  weight version**, on the **dev** set.
- Functional: BFCL is never used for tuning or checkpoint selection. A Base
  non-reasoning reference may run before training; SFT and SFT+RL run once each
  after selection under the same manifest invariant.
- Functional: the SFT envelope with eval enabled equals plan `260831-0808`'s
  figure, from a live non-monotonic VRAM reading.
- Non-functional: one Colab session at a time; every step commits its artifacts
  before the next begins.

## Architecture

Everything before this phase is verified against fakes and scripted backends,
which covers contracts and wiring. Four claims need a card, and the first draft's
CPU-green gating could not substantiate any of them: vllm is absent from CI by
construction, `make test-ci` deselects `gpu`, and two of its named tests were
structurally incapable of failing — a determinism test against a CPU fake takes
the same code path twice, and a memory guard comparing recorded constants cannot
notice a shrinking envelope.

| Claim | Why CPU cannot answer | Falsifies |
|---|---|---|
| Baseline agreement | needs real generation from real weights | Phase 4 |
| Throughput and its split | needs real kernels and a real KV cache | the plan's premise |
| Batch-composition stability | reduction order depends on real batching | manifest comparability |
| SFT envelope preserved | sleep-mode release is a runtime property | Phase 6, plan `260831-0808` |

Run order is chosen so a failure invalidates as little as possible. Agreement
first — if scores disagree beyond tolerance, throughput is irrelevant. Stability
second, since it decides whether any score comparison in the project is sound.
Then GRPO, then SFT, which carries the memory risk and the cross-plan dependency.

**Two measurement corrections the first draft needed.** Stability divergence has
a second cause the plan previously attributed entirely to kernels: `pool.py:239`
enforces worker timeouts with `ITIMER_REAL`, wall clock, so CPU contention at
concurrency N can turn a passing task into `AdapterResult(0.0, False)`
(`envscaler_heldout.py:158`). Every rung of the old mitigation ladder was a
GPU-side knob and could not touch it. So per-task terminal reasons and timeout
counts are recorded with the scores, and timeout-implicated divergence is
addressed by pool concurrency or `verify_timeout_s` before anything GPU-side is
tried. And VRAM is read with `reset_peak_memory_stats()` plus the worker's
driver-level footprint; vLLM's CuMem allocator can leave
`torch.cuda.memory_allocated()` unchanged after unmapping, while
`max_memory_allocated()` is monotonic and `memory_reserved()` does not shrink
without `empty_cache()`.

## Related Code Files

- Modify: `docs/evaluation.md` — measured throughput, utilization, stability
  result, new metric columns and their denominators
- Modify: `docs/grpo.md` — in-training eval cost per boundary
- Modify: `docs/optimization-ledger.md` — eval throughput row
- Modify: `plans/260828-1048-…/phase-05-…` — record the harness is vLLM-backed
- Create: phase report with every raw number

## Implementation Steps

1. Provision one L4. Run `smolqwen probe` and commit the artifact so the
   measurement environment is recorded with the numbers.
2. **Agreement.** Run `evaluate` on the Phase 1 baseline checkpoint and subset at
   concurrency 1. Publish the per-task disagreement list. Beyond tolerance, or
   clustered in one category, stops the phase and reopens Phase 4.
3. **Throughput.** Time the same set on both paths. Sample GPU utilization, and
   record the generation / tokenization / environment split — BFCL steps are
   in-process Python and EnvScaler's go through the worker pool, so running both
   locates the bottleneck by construction rather than by assumption.
4. **Stability.** Same set at concurrency 1 and N. Record both score sets **plus**
   per-task terminal reasons and timeout counts. If divergence correlates with
   timeouts, reduce pool concurrency or raise `verify_timeout_s` and re-measure
   before touching `enforce_eager` — which addresses CUDA graph capture, not
   reduction order. Record which intervention was needed. A recorded divergence is
   an acceptable outcome; an unmeasured one is not.
5. **GRPO.** Bounded run with eval enabled. Confirm the step-0 baseline,
   `grpo/bench_*` populated, `bench_weight_version` present, and agreement with
   `evaluate` at that same version. Record eval wall-time per boundary and the
   step-time impact.
6. **SFT.** Bounded run with eval enabled. Read worker VRAM between boundaries with
   `reset_peak_memory_stats()` plus the driver-level footprint through
   `OfflineEngine.memory_allocated_bytes()` to confirm the engine actually sleeps.
   The 2026-09-06 run completed the 32K envelope and both dev boundaries after
   temporarily offloading trainer state; only the production-cadence cost remains.
7. **Failure isolation.** Inject an eval failure in each trainer. Confirm training
   continues **and** every worker's episode set is empty — the leak path Phase 3
   fixed only shows up under real memory pressure.
8. **Notebook rendering.** Run one CLI command in a real Colab cell and confirm
   the non-TTY fallback emits plain lines rather than redraw frames. A local TTY
   cannot reproduce this.
9. **BFCL.** Capture the Base non-reasoning reference before training if it is not
   already recorded. After steps 2-7 and checkpoint selection, run SFT and SFT+RL
   once each under one manifest invariant. The report must state that no BFCL result
   selected a checkpoint or tuned a config, and must carry the caveat that
   deterministic tool-calling benchmarks have brittle state comparison and possible
   ground-truth error — so an unchanged or regressed number is not automatically a
   model deficiency. Discuss every regressed metric explicitly; a mixed result must
   not be presented as uniform improvement, which `artifacts/evaluation/final_results.md:28-30`
   already requires.
10. Stop the session. Write the report; update the docs above.

## Success Criteria

- [x] Probe artifact committed in the L4 smoke report
- [ ] Agreement within tolerance at concurrency 1, disagreement list published
- [ ] Throughput recorded with the three-way cost split
- [ ] Stability recorded at 1 and N with timeout counts, and the intervention
      named
- [ ] `grpo/bench_*` agrees with `evaluate` at the recorded weight version
- [x] `sft/bench_*` present; both base and checkpoint-1 boundaries completed with
      the engine sleeping between them
- [x] SFT envelope equals plan `260831-0808`'s figure: 32K step completed with
      about 19.3 GiB of 22.0 GiB in live total usage
- [ ] Injected eval failure: training survives, no episodes leaked
- [x] Colab cell shows the non-TTY fallback
- [ ] **BFCL Base reference plus one SFT and one SFT+RL run under one manifest
      invariant** — with the report stating all three were non-selecting and carrying
      the benchmark-brittleness caveat
- [ ] Every regressed metric discussed explicitly; no mixed result presented as
      uniform improvement
- [x] Docs updated with measured numbers; session stopped

## Risk Assessment

The largest risk is that throughput is bounded by something other than
generation. Agentic evaluation interleaves generation with environment steps, and
EnvScaler's steps carry `step_timeout_s: 10.0` and `verify_timeout_s: 60.0` — a
wide latency tail.

- Signal it broke: throughput improves far less than expected while GPU
  utilization stays low and the run is not generation-bound.
- Response: the three-way split is measured in step 3 before any conclusion, so
  the diagnosis comes from data. If environments dominate, raise pool concurrency —
  the pool supports it — and re-measure. Report the honest split rather than
  restating a target.

Second risk: session loss mid-measurement, which this project has already
experienced.

- Signal it broke: VM reclaimed with steps outstanding.
- Response: each step commits before the next begins, so a lost session costs one
  step. Never batch all measurements into one unsaved session.
