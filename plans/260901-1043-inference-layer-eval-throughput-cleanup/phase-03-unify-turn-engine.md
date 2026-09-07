---
phase: 3
title: "Unify the turn engine"
status: done
priority: P1
effort: "4d"
dependencies: [2]
---

# Phase 3: Unify the turn engine

## Overview

Generalize `rollout/scheduler.py` into one turn engine both training rollout and
benchmark evaluation drive, reconciling all seven divergences between the two
paths rather than the four the first draft of this plan assumed.

## Requirements

- Functional: one implementation advances concurrent episodes without a turn
  barrier, for both consumers.
- Functional: rollout keeps logprobs, `env_mask`, the NaN contract, and row order.
- Functional: evaluation gets per-step observation role, per-turn tool sets, a
  generation-turn budget, and windowed episode admission.
- Functional: failure cleanup releases environments regardless of whether mask
  building is on.
- Non-functional: the seven rollout and environment tests pass with assertions
  byte-identical; only import paths may change, and that diff must be reviewable.

## Architecture

The loop is right; what is woven into it is not. Seven reconciliations, five
structural.

**1. Observation role and per-turn tools.** `base.py:34` states the contract:
*"The role matters to Qwen3.5's chat template. Tool output must remain a
`role: tool` message; benchmark user turns are ordinary user messages."* BFCL
uses it — `bfcl.py:405` returns `observation_role="user"`, and every `StepResult`
carries `tools=state.active_tools`. The engine hardcodes `Message(role="tool", …)`
(`scheduler.py:464`) and renders from a frozen `ScenarioBinding`, so a user turn
would be wrapped as `<tool_response>` and the turn-0 tool set would render
forever. `multi_turn_miss_func` exists precisely because tools appear
mid-episode. Fix: the driver returns role and tool set per step; the render seam
takes a per-episode mutable tool set instead of reading `slot.binding`.

**2. Two turn budgets.** `runner.py:84` increments per generation;
`scheduler.py:390` increments `step_count` only inside `_on_step`. Both adapters
return non-terminal zero-env-step results for unparseable output, so a
prose-emitting model advances the engine's counter never and runs to
`episode_timeout_s: 600.0`. Fix: a distinct generation-turn counter with its own
`TerminalReason`, mapped from `max_steps_per_task`, kept separable from
`step_cap` in `terminals`.

**3. Decode convention and message shape.** Rollout decodes
`skip_special_tokens=False`, splits the continuation, and stores
`Message(content=…, reasoning_content=…)`; eval decodes `True` and stores raw
text. `scheduler.py:356` documents why the rollout shape is load-bearing:
treating the continuation as plain content makes the template synthesize a second
reasoning block and *"silently demotes earlier model tokens to env-mask zero on
every turn."* **The rollout convention wins** — the mask depends on it. That
changes what eval's adapters see from turn 2 onward, so it lands as its own
commit with the serial loop still in place, and Phase 1's baseline is re-captured
after it. Sequencing this first is what keeps Phase 4's agreement check
meaningful.

**4. Failure cleanup independent of masks.** `_cleanup_after_failure` computes
liveness as `set(self._builders)` (`scheduler.py:548`), and `_builders` is
populated in `_on_create` — the mask path. With masks off, cleanup issues no
destroys, environments leak, and every later eval fails on `PoolError`. Fix: an
explicit `live_episode_ids` set maintained by `_on_create`/`_on_destroy`,
independent of mask building, and cleanup reads that.

**5. Windowed admission.** `run()` opens every slot up front
(`scheduler.py:181`), which is correct for rollout — TRL needs one row per
prompt, positionally — and fatal for eval: 80 held-out tasks against a
32-episode pool means task 33 raises `PoolError` and the run scores zero. Fix:
`max_in_flight`. Rollout passes the full binding count, preserving today's
"all positions live" invariant exactly. Eval passes
`min(concurrency, pool_capacity)` and the engine admits in waves as episodes
retire. `MAX_REPLACEMENTS_PER_POSITION` and `_cleanup_after_failure` both assume
the current invariant, so both are audited against the windowed path.

**6. Positional assertions.** The four `assemble_output` checks are existence
tests — `any(isnan(...))`, `not all(env_mask)` — gated on `episode.observations`.
A mask shifted one token across every observation passes, and if the fork stops
populating that list both checks become no-ops. Fix: derive expected NaN
positions from `builder.spans` (`mask.py:184`) and assert
`env_mask[i] == 0 ⟺ isnan(logprobs[i])` for all `i`. That is invariant to
whether `observations` is maintained.

**7. Lazy rendering.** `_generate_ready` renders every ready slot then slices to
`generation_concurrency` (`scheduler.py:296-331`). At rollout's 16 that is 2×
waste; at eval's 80 tasks against width 8 it is 10× redundant chat-template
application on one vCPU, which Phase 10 would misread as environment latency.
Fix: select candidates first, render only those.

BFCL has no create/score/destroy to map onto `EnvDispatcher`'s four methods — its
`step` is in-process Python. The driver protocol keeps four methods; the
synchronous adapter supplies completed futures for create/destroy and a real one
for step. `bfcl.py:360` mutates `sys.path`, so the completed-future wrapper runs
on the engine thread, never a pool thread.

## Related Code Files

- Create: `src/smolqwen/inference/turn_engine.py`
- Create: `src/smolqwen/inference/{episode,mask}.py` — moved
- Create: `tests/test_turn_engine_shared_loop.py` — same scripted episodes as
  both consumers produce identical transcripts; includes a completed-future
  driver, which `rollout_fixtures.py:190` does not provide
- Create: `tests/test_turn_engine_admission.py` — more tasks than pool capacity
  completes rather than raising
- Create: `tests/test_turn_engine_cleanup.py` — masks off, injected failure,
  `pool.episodes_of(i)` empty for every worker
- Modify: `rollout/rollout_func.py` — construct with all capabilities on; keep
  `assemble_output` row order; replace the two existence assertions with
  positional ones
- Modify: `rollout/scheduler.py` — reduce to dispatcher and binding types
- Modify: `rollout/generation.py` — rename token-shaped `GenerationResult` to
  `TurnTokens`; delete dead `SamplingParams` (`:70`)
- Modify: `rollout/metrics.py`, `rollout/profiler.py` — import paths for the
  moved `episode.py` (both were omitted from the first draft)
- Modify: `tests/test_env_mask_alignment.py`, `tests/test_rollout_metrics.py` —
  import paths only
- Modify: `rollout/bench.py` — retarget the equivalence harness
- Move: `scheduler_config_for` (`bench.py:192`) to `inference/profiles.py` —
  **not** `training/grpo.py`, which would cycle with `bench.py:272`/`:394`

## Implementation Steps

1. Land the decode-convention change alone, serial loop intact. Re-capture the
   Phase 1 baseline against it. Nothing else in this phase starts first.
2. Move `episode.py` and `mask.py`; update all eight importers plus the two
   tests. Run mask and equivalence tests.
3. Extract the loop into `turn_engine.py` with injected driver, logprob capture,
   mask building, `max_in_flight`, and the generation-turn counter.
4. Add `live_episode_ids`; point cleanup at it.
5. Make rendering lazy.
6. Replace the two existence assertions with positional ones.
7. Rename to `TurnTokens`; confirm `grep -rc "class GenerationResult" src/`
   returns 1.
8. Rewire `rollout_func.py` with `max_in_flight = len(bindings)`.
9. Run the seven tests: `test_rollout_equivalence`,
   `test_scheduler_no_turn_barrier`, `test_env_mask_alignment`,
   `test_batch_shape_contract`, `test_state_isolation`, `test_pool_timeout`,
   `test_worker_crash_blast_radius`. Then the three new ones.
10. Move `scheduler_config_for`; update `grpo.py:23`.

## Success Criteria

- [x] One turn-loop implementation
- [x] `grep -rc "class GenerationResult" src/` returns 1
- [x] Seven existing tests green; assertion diff empty, import diff only
- [x] Shared-loop test proves identical transcripts, including a
      completed-future driver
- [x] Admission test: tasks > pool capacity completes
- [x] Cleanup test: masks off, failure injected, every worker's episode set empty
- [x] Positional NaN/mask assertion replaces both existence checks
- [x] Rendering count per cycle equals generation width, not ready count
- [x] `scheduler_config_for` relocated with no import cycle

## Outcome

`src/smolqwen/inference/turn_engine.py` is the only turn loop;
`rollout/scheduler.py` keeps `PoolDispatcher` and re-exports the binding and
config names its callers use. `rollout/driver.py` holds the rollout-specific
driver: XML parsing, pool dispatch, verifier reward, crash replacement.

Three corrections to this phase's plan, each from reading source:

**The decode convention was decided by measurement, not by the plan's rule.** The
plan said "the rollout convention wins because the mask depends on it." Neither
half survives. Only `<|im_end|>` and `<|endoftext|>` are special tokens in
Qwen3.5, so `skip_special_tokens` decides only whether the turn-end marker
survives — and keeping it breaks eval's string-equality markers *and* doubles the
marker in rollout's next render. What moves the mask is the message shape, and only
for a truncated turn: 12 supervised tokens of 52 under the split shape against 8
of 52 with one fork under raw content. Recorded in `inference/decoding.py` and
pinned by `tests/test_decode_convention.py`.

**The positional assertion is an implication, not a biconditional.** The plan
specified `env_mask[i] == 0 ⟺ isnan(logprobs[i])`. The reverse direction is false
on correct data: `generation.py` leaves a *sampled* token NaN when TRL supplied no
candidate for its position, and that token is legitimately supervised. Only
`mask == 0 → NaN` is asserted, with the NaN-but-supervised row as the negative
control.

**`max_generation_turns` is derived, not configured.** The plan mapped it from
`max_steps_per_task`, which lives on `EvalConfig` and is unreachable from
`GrpoConfig`. `turn_engine_config` derives it as `max_env_steps + 4` — one
generation per step, plus a final answer, plus head-room for invalid calls that
consume a generation without executing anything.

One implementation detail the plan could not have anticipated: `Episode.state`
defaults to `"ready"`, which was unreachable before windowed admission and now
means an un-admitted position would be selected for generation with an empty
message list. `_generate_ready` and `_dispatch_scoring` both filter on
`slot.admitted` for that reason.

Tests: `test_turn_engine_shared_loop.py` (3) drives one scripted episode through an
asynchronous pool driver and a fully synchronous one and requires identical
transcripts; `test_turn_engine_admission.py` (5) covers the window with its
negative control, rollout's all-positions invariant, the cleanup leak with masks
off, and the render count. The five existing rollout tests changed by import path
and function name only.

CPU suite: 376 passed, 16 deselected. `make check` and `make smoke` green.

## Risk Assessment

Highest-risk phase in the plan: this is the loop that produces training
gradients, and its invariants are easy to preserve in appearance while breaking
in substance.

- Signal it broke: any of the seven tests fails, or GRPO reward distribution
  shifts against the recorded baseline in Phase 10.
- Response: the equivalence harness is the gate. If it fails and the cause is not
  understood within the phase, revert and give evaluation its own concurrent
  loop — two loops to maintain, but the training path provably intact. Never
  weaken the equivalence test to pass.

Second risk: the decode-convention change alters eval scores before any batching
lands, so a later disagreement is hard to attribute.

- Signal it broke: step 1's re-baselined scores differ materially from Phase 1's
  first capture.
- Response: that difference is the measurement, not a failure — record it as the
  cost of unifying the seam, with the per-task disagreement list. If it is large
  enough to change a headline conclusion, stop and bring it to the user before
  continuing; changing what evaluation measures is their call.
