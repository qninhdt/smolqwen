---
phase: 5
title: "GRPO in-training benchmark eval"
status: done
priority: P1
effort: "1.5d"
dependencies: [4]
---

# Phase 5: GRPO in-training benchmark eval

## Overview

Log held-out benchmark scores during GRPO training through the same engine and
scoring path as `smolqwen evaluate`, with an explicit weight sync and the weight
version recorded alongside every number.

## Requirements

- Functional: at a configured interval, and once before the first optimizer step,
  score a fixed **dev** subset and log the metrics. Dev is EnvScaler held-out;
  BFCL is unreachable from this path.
- Functional: scores go through the Phase 3 engine and Phase 4 aggregation — no
  second scoring implementation.
- Functional: **an explicit `sync_weights()` before evaluating**, with the weight
  version logged next to every `bench_*` series.
- Functional: log held-out metrics beside the existing training reward so the gap
  is visible as its own series.
- Functional: an eval failure logs, releases its environments, and training
  continues.
- Non-functional: no change to reward semantics, GRPO loss, or the curriculum
  sampler.

## Architecture

The engine exists and the pattern has a precedent in the repository, but the
first draft of this phase read that precedent backwards.

`run_profile_difficulty` (`grpo.py:633-660`) does call `sync_weights()` and then
`trainer.rollout_func()`. What it does **not** do is train: `grep "\.train()"
src/smolqwen/training/grpo.py` returns nothing. It builds a trainer, syncs, and
generates. So it establishes that an explicit sync is **required**, not that a
callback boundary inherits a current engine.

At `on_step_end` the optimizer step has already been applied, so the colocated
engine holds pre-step weights. Worse, `steps_per_generation` is read from TRL's
args rather than pinned (`grpo.py:122`, `:523`), and `CurriculumCursor.at_step`
divides by it, so the repo already treats it as possibly greater than 1 —
generation happens once per accumulation window, leaving the engine up to
`grad_accum` (8 on L4) optimizer steps stale.

An 8-step drift is invisible in the number itself and lands directly on Phase 10's
acceptance test, where `grpo/bench_*` is compared against `evaluate` "at the same
revision". A stale engine would look exactly like a Phase 4 parity bug. So:
sync explicitly, then log the weight version — global step plus a sync counter —
as a field on every `bench_*` row. That makes the comparison falsifiable instead
of merely plausible.

**Metrics.** GRPO logs `rollout/invalid_calls`, `rollout/replacements`, queue
depth, straggler time, `group_reward_variance/*`, and verifier reward — all on
*training* scenarios under curriculum weighting, a biased sample by
construction. Adding `grpo/bench_*` from held-out tasks makes the gap observable,
and the gap gets its own series rather than leaving the reader to subtract.

**Dev, not test.** The callback scores the **EnvScaler held-out** set only.
`eval.yaml:5-7` lists both `bfcl_multi_turn` and `envscaler_heldout`, so a
callback that reads `config.adapters` would score BFCL at every checkpoint — and
a benchmark used to pick checkpoints is a dev set, which would void the final
`Base | SFT | SFT+RL` table. Phase 1 asserts BFCL is unreachable from here.

The dev-adapter selection lives on `GrpoConfig`, not as a new named field on
`EvalConfig`: `test_adapter_protocol.py:28-35` forbids `EvalConfig` from carrying
`heldout_env_count`, `heldout_scenarios_per_env`, or `env`, and `:22-27` forbids
`runner.py` from naming a benchmark. That contract stays intact.

**Subset stability.** A subset that varies between evals produces a curve mixing
policy change with sample change. The held-out EnvScaler adapter already owns
deterministic selection and records the exact IDs in its manifest contribution;
reuse that rather than sampling per call.

**Kill switch.** The eval body runs under a timeout inside a broad exception
handler, and on failure it must release environments — Phase 3's
`live_episode_ids` is what makes that possible, since the old cleanup path
derived liveness from the mask-builder dict and would have leaked every episode
with masks off, turning one recoverable failure into permanent `PoolError` on
every later boundary.

## Related Code Files

- Create: `src/smolqwen/training/bench_eval.py` — shared callback, parameterized
  by engine source, used by this phase and Phase 6
- Create: `tests/test_bench_eval_callback.py` — interval firing, step-0 baseline,
  failure isolation, environments released after failure, subset identical across
  calls, weight version recorded
- Modify: `src/smolqwen/training/grpo.py` — register the callback; supply the
  colocated engine; call `sync_weights()` in the callback before evaluating
- Modify: `src/smolqwen/config_models.py` — eval-during-training block on
  `GrpoConfig`: interval, task cap, timeout, enable flag. Fields with defaults,
  so `extra="forbid"` (which rejects unknown YAML keys, not new model fields) is
  satisfied
- Modify: `configs/base/grpo.yaml` — defaults sized so total in-training eval cost
  stays **at or under 10% of training wall time**, the budget the user set. Record
  the measured cost per boundary so the setting is checkable, not asserted
- Modify: `docs/grpo.md` — what appears on the chart, its cost, and that the
  number carries a weight version

## Implementation Steps

1. Write `bench_eval.py`: takes a generation source, adapter set, fixed subset,
   and metric sink, then calls the Phase 3 engine and Phase 4 aggregation. No
   benchmark logic of its own.
2. Wrap the body in a timeout and a broad handler. On failure log
   `bench_eval/failed` with the reason, release environments via the engine's
   cleanup, and return control.
3. Register in `grpo.py`. Call `sync_weights()` first — assert the engine exists
   rather than skipping silently, matching `_assert_prefix_caching`
   (`grpo.py:405`).
4. Add the step-0 baseline on `on_train_begin`.
5. Log `grpo/bench_<category>_<metric>`, plus
   `grpo/bench_weight_version` and `grpo/bench_heldout_minus_train_reward`.
6. Add config fields and defaults on `GrpoConfig`. Size interval × task cap so
   total eval cost lands at or under 10% of training wall time, and log the
   measured per-boundary cost so the budget is verified rather than assumed.
7. Test with a scripted backend: interval, baseline, injected failure leaves no
   live episodes, subset identical across two calls, weight version present, and
   the resolved adapter set contains no BFCL entry.

## Success Criteria

- [x] `grpo/bench_*` logged including a step-0 baseline
- [x] Dev set is EnvScaler held-out only; a test-benchmark name is refused,
      asserted at construction and per run
- [ ] Measured in-training eval cost at or under 10% of training wall time
      (`bench_wall_s` is recorded per boundary; the measurement needs a card)
- [x] `sync_weights()` called before every eval, with `bench_weight_version`
      logged alongside
- [x] Held-out metrics use the Phase 3 engine and Phase 4 aggregation; no second
      scoring path exists
- [x] Injected failure: training continues **and** the adapter's episode set is
      empty afterwards
- [x] Subset identical across evals within a run
- [x] Reward semantics, loss, and curriculum sampler unchanged
- [ ] `grpo/bench_heldout_minus_train_reward` — see Outcome
- [x] CPU suite green

## Outcome

`training/bench_eval.py` is shared by both training stages, parameterized by engine
source exactly as planned. `grpo.py` supplies the colocated engine and an explicit
`sync_weights()` that raises when absent rather than skipping — matching
`_assert_prefix_caching`, because a silently unsynced eval reports a number for the
wrong weights.

Two criteria stay open, for different reasons.

**The 10% cost bound needs a card.** The mechanism is in place: `bench_wall_s` is
logged per boundary, so the budget is checked against a measurement rather than a
config comment. The number itself comes from Phase 10 step 5.

**`bench_heldout_minus_train_reward` is not implemented, deliberately.** It would
subtract two numbers logged at different times by different code paths — the
training reward arrives from `rollout/metrics.py` on TRL's own logging cadence, the
dev score from this callback at a save boundary. Computing the difference inside the
callback means reading the most recent training reward from wherever it was last
written, which is a coupling that produces a plausible number whenever the two
cadences disagree. Both series are logged; the gap is one subtraction in W&B, over
values whose timestamps a reader can see. Adding the field would hide that.

One divergence from the plan's step 6: `every_steps` defaults to 0, meaning save
boundaries only. That is the cadence which guarantees a checkpoint exists to
attribute the score to, and it is the cheapest useful one. An interval remains
configurable.

## Risk Assessment

The assumption is that driving the colocated engine at `on_step_end` is safe.
`run_profile_difficulty` proves the mechanism works between phases, not at a
callback boundary where optimizer state, accumulation position, and the sampler
cursor are all live.

- Signal it broke: a loss or reward discontinuity at exactly the eval interval, a
  resume that replays or skips scenarios, or an OOM at the first boundary.
- Response: move the callback to `on_save`, an existing boundary where state is
  already being flushed, and reduce the task cap. If OOM persists, sleep the
  engine's KV cache around the eval as Phase 6 does.

Second risk: eval cost eats the training budget. Up to 20 turns per task at 2048
max new tokens is not free, and Colab sessions vanish without warning.

- Signal it broke: measured step-time regression beyond a small fraction, or
  fewer optimizer steps per session than before.
- Response: interval and cap are config, so widen the interval first. Record
  measured eval wall-time per boundary in the phase report so the trade is a
  number rather than an impression.
