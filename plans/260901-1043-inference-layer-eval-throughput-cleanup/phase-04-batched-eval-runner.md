---
phase: 4
title: "Batched evaluation runner"
status: in_progress
priority: P1
effort: "2.5d"
dependencies: [3]
---

# Phase 4: Batched evaluation runner

## Overview

Drive evaluation through the shared turn engine on an in-process vLLM engine, add
the diagnostic metrics the benchmarks can actually support, and measure agreement
against the baseline within the tolerance Phase 1 registered.

## Requirements

- Functional: tasks advance concurrently, bounded by `min(concurrency, pool_capacity)`.
- Functional: agreement with the re-captured baseline within registered tolerance,
  with a published per-task disagreement list.
- Functional: the manifest records the values generation actually used, including
  concurrency and `enforce_eager`, in `recorded_free`; `invariant` unchanged.
- Functional: `--endpoint` keeps working and keeps recording serving provenance.
- Functional: `TransformersPolicy` stays. It is the only path that evaluates an
  adapter without merging, its removal was never requested, and keeping it makes
  vLLM's LoRA support an optimization rather than a dependency.
- Functional: **every evaluation run writes a trajectory record per task**, so a
  score can be re-derived and a failure attributed after the fact.

## Architecture

`evaluate_adapter` becomes a thin caller of the turn engine. The metric work is
what decides whether in-training eval is usable at all, and two of the five
metrics the first draft specified were wrong.

**What the benchmarks actually do.** BFCL (`bfcl.py:192-219`) returns 1.0 only
when four conditions hold: (a) `state.completed`, (b) snapshot count equals
ground-truth turn count, (c) every snapshot matches, (d) cumulative results
cover expected. EnvScaler is **not** all-or-nothing — `verifier.py:246` returns
`reward = passed / total`, and `metrics.py:27` already averages it as `score`.

So the first draft's `per_check_pass_rate` was a rename of an existing metric,
and its justification ("continuous, moves before the all-or-nothing reward") was
built on a false premise. What the verifier genuinely discards is *which* checks
failed and `name_error_count` — both already crossing the process boundary in
`pool.py:213`. That is the metric worth adding.

Already computed in `aggregate()` (`metrics.py:20`), just never surfaced per run:
`invalid_call_rate`, `truncation_rate`, `average_steps`,
`average_generated_tokens`, `exact_success_rate`.

New, each mapped to a condition it can actually resolve:

| Metric | Condition | Denominator |
|---|---|---|
| `completion_rate` | (a) | all tasks |
| `snapshot_count_ratio` | (b) | all tasks |
| `state_match_rate` | (c) | completed **and** length-matched only |
| `result_match_rate` | (d) | completed **and** length-matched only |
| `failed_check_names`, `name_error_count` | EnvScaler detail | tasks with a verdict |

The restricted denominators matter: `bfcl.py:194` returns before
`expected_snapshots` is ever built, so (c) and (d) are undefined for tasks
failing (a) or (b), and `zip(..., strict=True)` would raise on the length
mismatch that guard protects. Filling `0.0` instead would make "never finished"
and "finished wrong" identical again — the exact conflation `completion_rate`
exists to remove. The denominator is printed in the report, not implied.

`AdapterResult` (`base.py:41`) gains `completed: bool` and
`diagnostics: Mapping[str, float]`, **both with defaults** — it is a frozen
positional dataclass with 9 construction sites across 4 files, and undefaulted
fields break all 9.

**Agreement, not identity.** Byte-identical metrics cannot survive the backend
change; Phase 1 registered a tolerance and a disagreement budget instead, and the
two questions are separated: agreement at concurrency 1 tests backend
equivalence; agreement at concurrency N tests batch invariance. Conflating them
is why the first draft's mitigation ladder started with `enforce_eager`, which
addresses CUDA graph capture and not reduction order.

**Trajectories are the evidence; metrics are a projection of it.** Today
`runner.py:70-95` builds `history` per task and then discards it — only the
aggregate reaches the report. The rollout path already keeps the full record
(`episode.to_row()`: messages, observations, terminal reason, per-check bools,
timings), so evaluation is the only side that throws it away.

That matters most for exactly the benchmark this project reports. BFCL is
all-or-nothing over four conditions, so a `0.0` with no trajectory cannot be
attributed to any of them, and re-grading is impossible. The reproducibility
literature measures how large that gap gets: re-grading fixed artifacts moved
scores by up to 20.9pp, and τ-bench shifted 16.9pp with 12 ordering changes among
frontier models. The diagnostics above make failure *legible*; the trajectory
makes it *checkable*.

So each run writes JSONL beside its report: task id, category, full message
history, observations, terminal reason, which scoring condition failed, per-check
bools, generated-token count, and timing. `artifacts/evaluation/` is gitignored,
so these stay local until Phase 9 uploads them.

**Serving provenance.** Phase 8 drops the nine manifest-only flags because the
in-process engine knows those facts. That is false for `--endpoint`, where the
server is a separate process: `manifest.py:71-74` normalizes unsupplied fields to
`None` and `assert_comparable` compares `invariant` only, so an fp8-served row
and a bf16-served row would become indistinguishable. Fix: the client reads
`/v1/models` and the server's reported config at connect time and records what it
observes. The flags then go without losing provenance.

## Related Code Files

- Modify: `eval/runner.py` — drive the engine; replace the hardcoded
  `backend = "http" if endpoint else "transformers"` (`:158`) with the actual
  backend; record concurrency and `enforce_eager`
- Modify: `eval/policies.py` — `HttpPolicy` becomes a thin adapter over
  `inference/client.py`; narrow the `_openai_messages` reconstruction (gate it on
  the runner having emitted a call for that turn). `TransformersPolicy` stays as
  the adapter-on-base path
- Create: `src/smolqwen/eval/trajectories.py` — JSONL writer for the per-task
  record
- Modify: `eval/adapters/base.py` — `completed` and `diagnostics`, defaulted
- Modify: `eval/adapters/bfcl.py` — diagnostics for (a)-(d) with restricted
  denominators
- Modify: `eval/adapters/envscaler_heldout.py` — surface failed check names and
  `name_error_count`
- Modify: `eval/metrics.py`, `eval/report.py` — aggregate and render, printing
  each restricted denominator
- Modify: `cli.py` — route `_cmd_evaluate` through `resolve_eval_checkpoint`
  (`artifacts.py:208`), which is currently unwired, not dead; carry adapter
  revision pinning into the inference layer's `load_lora`
- Create: `tests/test_eval_concurrency_agreement.py` — `@pytest.mark.gpu`
- Create: `tests/test_adapter_diagnostics.py` — each condition maps to its
  metric; restricted denominators asserted
- Create: `tests/test_eval_admission_capacity.py` — 80 tasks, 32-episode pool
- Create: `tests/test_eval_trajectory_record.py` — one record per task, carrying
  the failing condition and per-check bools
- Modify: `tests/test_eval_runner.py`, `tests/test_http_policy_metrics.py`,
  `tests/test_adapter_protocol.py` (9th `AdapterResult` site)

## Implementation Steps

1. Extend `AdapterResult` with defaulted fields; fill diagnostics in both
   adapters with restricted denominators. Land with the serial loop still in
   place so the metric change is independently verifiable.
2. Add the trajectory writer and wire it into the serial loop, still before any
   batching. That gives the Phase 1 baseline a trajectory record too, so the
   Phase 10 agreement check can compare *trajectories*, not just scores — which
   is what makes a disagreement attributable.
3. Rewire `runner.py` onto the turn engine with `max_in_flight` from
   `min(concurrency, pool_capacity)`. Fix the backend recording at `:158`.
4. Add the capacity test first — it fails before step 3 lands and passes after,
   which is what makes it a regression test rather than a description.
5. Reduce `HttpPolicy` to a client adapter; have the client record observed
   serving config for the manifest.
6. Narrow `_openai_messages`. The defect is real but far smaller than first
   reported: `policies.py:164-173` already requires a `str` `name` and a
   `Mapping` `arguments`, so prose ending in JSON does not trigger it. Gate on
   emitted-call state anyway and keep a regression test.
7. Route checkpoint resolution through `resolve_eval_checkpoint`; enforce adapter
   revision pinning wherever the adapter enters the inference layer, so the check
   no longer lives only in the `TransformersPolicy` constructor.
8. Record which generation path each checkpoint kind takes: base and merged
   through vLLM, adapter through vLLM `LoRARequest` if it loads, otherwise
   `TransformersPolicy`. This is a recorded fact in `recorded_free`, not a gate.
9. Measure agreement at concurrency 1 against the re-baselined numbers. Publish
   the per-task disagreement list.

## Success Criteria

- [ ] Agreement within Phase 1's tolerance at concurrency 1, disagreement list
      published per task
- [x] Capacity test: 80 tasks against a 32-episode pool completes
- [x] One trajectory record per task, carrying the failing condition — a score is
      re-derivable from the record without re-running generation
- [x] `invariant` unchanged; concurrency, `enforce_eager`, observed serving
      config, and the generation path in `recorded_free`
- [x] `--endpoint` rows still carry serving provenance with the flags gone
- [x] Restricted denominators printed for `state_match_rate` and
      `result_match_rate`
- [x] `failed_check_names` and `name_error_count` surfaced; no metric that merely
      renames `score`
- [x] `AdapterResult` extended with defaults; all 9 sites compile untouched
- [x] Adapter revision refused when unpinned, with a test
- [x] `TransformersPolicy` retained and still exercised for adapter-on-base
- [x] **`evaluate` generates through the engine, asserted against the command**
      (added after this phase was wrongly closed — see Outcome)
- [x] CPU suite green; `gpu` tests listed as pending

## Outcome

Code complete. **The agreement criterion is the one open item and it cannot close
here**: `evaluate` needs weights, and no checkpoint exists in this repo, so Phase 1
recorded the command and deferred the baseline. Phase 10 step 2 is the first
measurement, and it re-captures the baseline on the same card as the vLLM run so
the comparison is not across two machines.

`eval/batched.py` drives the adapters through the shared engine; `eval/driver.py`
is the benchmark side of the driver protocol. The window is
`min(generation_concurrency, pool_capacity)`, with a test that reads both numbers
from the shipped configs and fails if a future edit makes the window stop mattering.

### Correction: this phase was recorded complete while step 3 was never done

Found 2026-09-03, after Phases 5-9 had all closed on top of it. Step 3 above says
*"Rewire `runner.py` onto the turn engine"*. That did not happen. `evaluate_batched`
was written and tested against a scripted backend, and its only caller was
`training/bench_eval.py` — so `smolqwen evaluate`, the command every reported number
comes from, still went `run_evaluation → load_policy → TransformersPolicy →
model.generate()` at batch size 1. Goal 2 of the plan was false for two days while
this file said "code complete".

Two process failures, not one:

**The success criteria could not detect it.** Every box above is about a *unit* —
the window, the records, the diagnostics, the pinning — and each passed. None asked
which path the **command** takes. `tests/test_eval_generation_path.py` is that
assertion, and it fails on the pre-fix runner.

**Phase 8 deleted the fix.** Phase 2 wrote `offline_engine_for_eval` to be exactly
this wiring. The dead-symbol audit found it with zero consumers across all eight
surfaces and deleted it in `990670c`. Zero consumers was evidence that step 3 was
outstanding; it was read as evidence the helper was dead. A property-aware audit
still cannot tell "unused" from "not yet wired" — the only thing that can is a test
naming the consumer, which is what the audit's own inventory should have demanded
for a symbol introduced by a still-open phase.

What landed with the fix: `generation_for` selects the path and records it in
`recorded_free.generation_path`, with three logged fallbacks to `TransformersPolicy`
(endpoint, vllm absent, adapter refused) and everything else raising. And
`recorded_free.dtype` now comes from the engine rather than the constant
`"bfloat16"`, which was about to become actively wrong: a Turing card resolves
float16, and a report claiming bf16 would present two numeric regimes as one
experiment.

Three corrections to this phase's plan:

**`per_check_pass_rate` would have renamed `score`.** `verifier.py:246` already
returns `round(passed / total, 4)` and `metrics.py` already averages it, so
EnvScaler was never all-or-nothing. What the aggregate genuinely discarded is which
checks failed and the `NameError` count — the signal separating "the state is
wrong" from "the verifier could not run". Those are what got surfaced.

**Diagnostics needed a denominator in the report, not just a restricted
population.** `state_match_rate` of 1.00 over 12 tasks and over 80 are different
claims about the same benchmark, so `aggregate` emits `<metric>_denominator` and the
report renders the pair.

**Checkpoint pinning had two holes, not one.** `resolve_eval_checkpoint` was tested
but unreachable from `evaluate`; adapter pinning lived only in
`TransformersPolicy.__init__`, which the vLLM adapter path does not enter.
`eval/checkpoints.py` closes both at the boundary where weights enter and records
which of local / pinned-Hub-pull / endpoint applied.

The engine gained `generate_ids`: the turn engine renders and tokenizes itself, and
re-tokenizing a decoded string would let a BPE seam move the boundary the mask
builder depends on. A position whose logprob vLLM did not report stays NaN rather
than shortening the row.

## Risk Assessment

The load-bearing assumption is that a tolerance can be set tight enough to be
meaningful and loose enough to pass. If backend divergence is larger than
expected, this phase cannot close honestly either way.

- Signal it broke: disagreements exceed the budget, or cluster in one category.
- Response: a clustered pattern is diagnostic, not noise — investigate the
  category before adjusting anything. Widening the tolerance to absorb a cluster
  is the one move that is not available; if the divergence is real and large,
  the comparison across Base/SFT/RL is what is at stake and the user decides.

Second risk: environment latency dominates, so batching buys less than expected.

- Signal it broke: low GPU utilization while generation is not the bottleneck.
- Response: BFCL steps are in-process Python and EnvScaler's go through a worker
  pool, so running both locates the bottleneck by construction. Report the
  generation / tokenization / environment split rather than restating a target.

Third risk: trajectory records grow large enough to matter. 80 tasks × up to 20
turns × 2048 max new tokens is a real file, and Phase 9 uploads it.

- Signal it broke: a record file large enough to slow the run or the upload.
- Response: the record is per-run, not per-step-boundary, and
  `artifacts/evaluation/` is gitignored so nothing enters git. If size becomes a
  problem, drop token ids and keep decoded text — the text is what a re-grade
  needs. Do not drop the failing condition or the per-check bools; those are the
  reason the record exists.
