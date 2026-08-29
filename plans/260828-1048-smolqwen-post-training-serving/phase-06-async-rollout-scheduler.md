---
phase: 6
title: "Async rollout scheduler"
status: pending
priority: P1
effort: "5d"
dependencies: [4]
---

# Phase 6: Async rollout scheduler

## Overview

Replace turn-synchronous batched rollout with a ready-queue scheduler behind TRL's
`rollout_func`, so the GPU is fed continuously while Python environments execute.
Deliver a measured A/B in episodes/hour against the turn-synchronous baseline.

This is the project's signature systems problem. Prior probe on this exact
workload: A100 mean GPU utilization 49.2%, L4 mean 64.3% — the faster GPU finishes
generating sooner and therefore waits longer on CPU.

The barrier is removed **inside one `rollout_func` call**, not across training
steps. That boundary is set by TRL's contract, not by preference — see
Architecture.

## Requirements

**Functional**
- `environment_factory`-based rollout built first as the correctness oracle and A/B
  baseline. It is not the production path, and it lives in a **separately
  constructed trainer**: the production trainer passes `rollout_func` with
  `tools=None, environment_factory=None`, because `env_mask` is only read when
  `self.tools` is empty.
- Async scheduler: episodes in `READY | GENERATING | TOOL | DONE` states; any
  episode ready to continue is eligible for the next generation sub-batch,
  regardless of its turn index.
- The scheduler operates over exactly the prompts TRL passed in.
  `generation_batch_size` is set to the intended active-pool size, and
  `generation_concurrency` is the vLLM sub-batch width within that pool. There is
  no cross-step buffer and no sampler admission — see Architecture for why.
- Return value is positionally aligned 1:1 with the input prompts, preserving the
  group-contiguous ordering TRL's `RepeatSampler` produced.
- `rollout_func` returning `prompt_ids`, `completion_ids`, `logprobs`, and
  `env_mask` marking model tokens 1 / environment tokens 0.
- `len(logprobs) == len(completion_ids) == len(env_mask)` per episode, with **NaN**
  at appended-observation positions. `vllm_importance_sampling_correction` defaults
  to `True`, so `logprobs` feeds the IS ratio; TRL maps NaN to ratio exactly 1,
  while a length mismatch gets right-padded with 0.0 and shifts every later model
  token's logprob against the wrong position.
- `sampling/sampling_logp_difference/max` tracked with a stated threshold above
  which the run stops — it is the only visible symptom of logprob misalignment.
- Mask construction via ported `_SampleBuilder` drift classification, not
  hand-rolled span arithmetic: every turn's re-rendered prefix is compared against
  the accumulated tokens and classified `CLEAN | REALIGN | FORK`, with
  `drift_tokens` logged.
- Infrastructure failures (`worker_crash` from the Phase 4 pool) are dropped from
  the returned batch rather than scored, and a replacement episode fills the slot
  so the row count still matches.
- Timeline profiler attributing wall time to generation, `env.step`, tool parse,
  verifier, tokenization, and scheduling overhead.
- Metrics: episodes/hour, tokens/s, mean and peak GPU utilization, ready-queue
  depth over time, straggler distribution, drift-token distribution.
- Correctness equivalence: for a fixed seed and a deterministic scripted policy,
  the async path produces the same rewards as the baseline path.

**Non-functional**
- vLLM colocate mode with an explicit KV budget — never inherit the model's 262k
  advertised context.
- Prefix caching enabled; the G rollouts of a scenario share an identical initial
  prefix.
- A single hung environment cannot stall the scheduler.
- Report episodes/hour. GPU utilization percent is a diagnostic, not a target.

## Architecture

The barrier being removed is concrete. TRL's `_tool_call_loop` runs
`while idxs_with_tool and iteration_num < max_tool_calling_iterations`: it collects
every episode with a pending tool call, executes them, regenerates for all of them,
and repeats. The batch advances turn by turn in lockstep, so one 150 ms tool call
holds the other fifteen episodes that took 3 ms.

```
Turn-synchronous (TRL built-in)
  gen turn1 [A B C D] → wait for slowest tool → gen turn2 [A B C D] → ...
                          ▲ GPU idle here

Ready-queue (this phase), same N episodes, one rollout_func call
  ready pool = the N prompts TRL handed in
  sub-batch = whatever is ready right now, up to generation_concurrency
  A_turn3, B_turn1, D_turn7, C_turn2 → one generation sub-batch
  a slow tool blocks only its own episode
```

**Why the scope is one call.** `rollout_func(prompts, trainer)` is invoked
synchronously at `grpo_trainer.py:2237` with exactly `generation_batch_size`
prompts, pre-arranged into group-contiguous blocks by `RepeatSampler`
(`:1236-1237`). The return must be positionally aligned: `_calculate_rewards`
sizes its tensor from `len(prompts)` (`:1633`) and zips prompts against
completions with `strict=True` (`:1661`), and advantages come from
`rewards.view(-1, num_generations)` (`:2787`) — a pure positional reshape with no
`group_id` anywhere. So there is no cross-step buffer TRL will accept, no way to
return "whichever groups finished", and no way to pull extra rows from the
sampler, which sits upstream of the call. The active pool is intra-call
concurrency over the prompts already handed in.

TRL does ship a genuine cross-step pipeline in `trl.experimental.async_grpo`
(`free_slots` scheduler at `async_rollout_worker.py:505`, group join at `:649`,
`max_staleness` discard). It is rejected here because it requires a separate
`vllm serve` process and contains no PEFT path at all —
`grep -riE "peft|lora|colocate"` over that package returns 0 hits — so server +
trainer + LoRA cannot share one 24 GB L4. Revisit only if the Phase 1 probe
returns A100 80 GB. What is lifted from it is the drift reconciler, below.

Within one call, two barriers still exist and both must go. The turn barrier is
above. The second is the GRPO-group barrier: G rollouts of one scenario do not
need to advance in lockstep — only the whole batch must be complete before it is
returned. Removing the turn barrier while keeping the group barrier recovers
little.

```
TRL step
  │  prompts: N = generation_batch_size, group-contiguous
  ▼
  rollout_func
    ready pool (all N) ──ready──► vLLM colocate, continuous batching
        ▲                            │
        │                    completions
        └──observations── Worker pool (Phase 4)
    all N DONE?
  │  return N rows, positionally aligned, env_mask per row
  ▼
  TRL: advantages = rewards.view(-1, num_generations) → LoRA update
```

`env_mask` is the mechanism that makes this legal. Environment observations get
appended into the completion span, so without a mask the policy would be trained to
predict tool output. TRL pops `env_mask` from the `rollout_func` result and uses it
as the internal `tool_mask`, where 1 marks model tokens — **but only on the `else`
branch of `if self.tools:`** (`grpo_trainer.py:2270,2289-2291`). Since
`environment_factory` populates `self.tools` from every environment method
(`:649,666`), a trainer holding both silently discards our mask and rebuilds
`tool_mask` as all-ones over completions that already contain observations. Loss
descends, the model learns to predict tool output, and nothing errors. Hence two
separate trainers, enforced by test.

Mask construction cannot be trusted to span bookkeeping. Each turn re-renders the
whole conversation through the Qwen3.5 template, and the re-tokenized prefix can
differ from the tokens already held — a dropped reasoning block is the common case.
A fixture built by the same concatenation rule as the code will always pass while
the real mask walks off the boundary. So the semantics are ported from
`_SampleBuilder.classify_token_drift` (`async_rollout_worker.py:100-146`):
compare, classify `CLEAN | REALIGN | FORK`, and log `drift_tokens` as a
first-class metric rather than a blind threshold.

`logprobs` is subject to the same alignment requirement, and getting it wrong is
quieter still. `vllm_importance_sampling_correction` defaults to `True`
(`grpo_config.py:918`), so the returned array feeds
`(old_per_token_logps - sampling_per_token_logps) * mask` at
`grpo_trainer.py:2680`. vLLM produces no logprob for an appended observation, so
those positions must be **NaN** — TRL converts NaN to a zero difference, i.e. ratio
exactly 1 (`:2681-2683`). Returning a shorter array instead gets right-padded with
0.0 (`:2489`), which is logprob 0 = probability 1, and shifts every model token
after the first observation against the wrong position. The IS ratio becomes
garbage, gets clamped, and training continues without error. The only visible
symptom is `sampling/sampling_logp_difference/max`.

The `environment_factory` oracle has a subtler problem than the async path it is
meant to check. TRL does not build one environment per rollout: `_environment_pool`
hands out `pool[index]` per batch position and appends only when a batch needs more
concurrent instances (`grpo_trainer.py:636-645`), then calls `reset(**kwargs)`.
Upstream's `init_env_instance` restores only `init_config` keys via `setattr`
(`env_util.py:43-46`) while `get_state_info` returns `vars(instance)` wholesale
(`:94-99`) — so an attribute an environment created in a previous episode survives
into the next `final_state()` and can satisfy a `check_func` it should not. So
`factory_env.reset` constructs a brand-new `EnvInstance` and rebinds it rather than
mutating in place, and the equivalence test runs in the direction that matters: if
oracle and async path disagree, the oracle is suspect first, because it is the
pooled one.

Correctness first. The `environment_factory` path is built to establish what the
right answer is, then the async path must reproduce it under a fixed seed with a
scripted deterministic policy. Only then is throughput worth measuring — a fast
rollout that computes different rewards is not an optimization.

## Related Code Files

- Create: `src/smolqwen/rollout/factory_env.py` — `environment_factory` adapter: `reset`, tool methods, `get_reward`; the correctness oracle and A/B baseline
- Create: `src/smolqwen/rollout/scheduler.py` — episode state machine, ready queue over the call's prompts, batch completion join
- Create: `src/smolqwen/rollout/rollout_func.py` — TRL entry point returning `prompt_ids` / `completion_ids` / `logprobs` / `env_mask`; asserts `trainer.tools` is empty
- Create: `src/smolqwen/rollout/episode.py` — per-episode record; required field set below
- Create: `src/smolqwen/rollout/mask.py` — ported `_SampleBuilder` drift classification: `CLEAN | REALIGN | FORK`, mask assembly, `drift_tokens` accounting
- Create: `src/smolqwen/rollout/generation.py` — vLLM colocate client, prefix caching, KV budget, weight sync hook
- Create: `src/smolqwen/rollout/profiler.py` — timeline attribution over `trl.extras.profiling.profiling_context`, queue-depth sampling, straggler histogram
- Create: `src/smolqwen/rollout/metrics.py` — episodes/hour, tokens/s, GPU util mean/peak, drift distribution, W&B emission
- Create: `tests/test_env_mask_alignment.py` — **critical**: mask is 1 exactly on model-generated tokens, 0 on every appended observation, verified differentially against a fresh re-render — not against a fixture built by the same concatenation rule. Also asserts the three arrays are equal length with NaN logprobs at observation positions
- Create: `tests/test_factory_oracle_reuse.py` — **critical**: the pooled factory instance carries no state across two consecutive batches of the same scenario
- Create: `tests/test_trainer_kwargs_exclusive.py` — **critical**: constructing a trainer with both `rollout_func` and `tools`/`environment_factory` fails loudly
- Create: `tests/test_batch_shape_contract.py` — **critical**: the returned row count and group ordering match the input prompts exactly, including when episodes fail
- Create: `tests/test_scheduler_no_turn_barrier.py` — a deliberately slow episode does not delay others; simulated clock
- Create: `tests/test_infra_failure_excluded.py` — a `worker_crash` episode is replaced, not scored; a `timeout` episode is scored
- Create: `tests/test_rollout_equivalence.py` — **critical**: fixed seed + scripted policy → identical rewards on both paths
- Create: `tests/test_terminal_conditions.py` — final answer, step cap, unrecoverable state, and timeout each terminate correctly
- Modify: `configs/base/grpo.yaml` — timeouts and semantic settings only; step cap seeded from `budgets.json`
- Modify: `configs/profiles/{l4,a100}.yaml` — `generation_concurrency`, `active_pool_multiplier`, `vllm_kv_fraction` (this phase's owned fields only)
- Modify: `src/smolqwen/cli.py` — add `rollout-bench` running both paths and emitting the A/B table

### `Episode` required fields

The record every other module reads. Writers: `scheduler.py` (state, step count,
terminal reason), `generation.py` (token spans, logprobs, timings),
`mask.py` (mask spans, drift tally). Readers: `rollout_func.py`, `metrics.py`,
`profiler.py`, Phase 7 `trajectory_table.py`, Phase 7 `reward.py`.

| Field | Written by | Read by |
|---|---|---|
| `episode_id`, `scenario_id`, `group_index` | scheduler | group join, Phase 7 variance logging |
| `messages` | scheduler | trajectory table |
| `prompt_ids`, `completion_ids`, `prompt_completion_boundary` | generation | rollout_func |
| `logprobs` | generation | rollout_func (mandatory TRL key) |
| `mask_spans`, `drift_tally` | mask | rollout_func, metrics |
| `observations` (text, separate from messages) | scheduler | trajectory table |
| `step_count`, `invalid_call_count` | scheduler | metrics, Phase 7 secondary metrics |
| `terminal_reason` (`final_answer` / `step_cap` / `unrecoverable` / `timeout` / `worker_crash`) | scheduler | rollout_func (exclusion), metrics |
| `reward`, `per_check_bools` | Phase 7 reward | trajectory table, Phase 7 variance |
| `stage_timings` (generation / env.step / parse / verifier / tokenization) | generation, scheduler | profiler |

## Implementation Steps

1. `episode.py`: the record every other module writes into, with the field set
   tabulated above. Get this shape right before writing the scheduler; everything
   downstream reads it.
2. `factory_env.py`: the `environment_factory` adapter over Phase 4 — `reset(**row)`
   loads the scenario and returns the task prompt, public methods proxy to
   `EnvInstance.step`, `get_reward()` runs the verifier. Set
   `TRL_EXPERIMENTAL_SILENCE=1`. This path is deliberately simple; its job is to be
   obviously correct. It is constructed in its **own** trainer with no
   `rollout_func`. Because TRL pools and reuses factory instances across batches,
   `reset` must construct a fresh `EnvInstance` and rebind it — never mutate the
   existing one — and a test must run the oracle over two consecutive batches of the
   same scenario asserting identical `final_state()` key sets.
3. Write `test_rollout_equivalence.py` now, against the factory path only, with a
   scripted deterministic policy. It becomes the async path's acceptance gate.
4. `generation.py`: vLLM in colocate mode with an explicit
   `vllm_max_model_length` — the model advertises 262k and inheriting that is an
   instant OOM. Enable prefix caching. Wire the weight-sync hook TRL calls at step
   boundaries (`grpo_trainer.py:2229-2232` syncs before `rollout_func` is invoked,
   so all generation within one call runs under one weight version — no
   mixed-policy episodes, which is a property of the intra-call design worth
   keeping). Budget for what the sync actually costs on a PEFT model: it calls
   `merge_adapter()`, streams **every** named parameter into vLLM, then
   `unmerge_adapter()` (`vllm_generation.py:443-469,509-511`) — a full ~4 GB bf16
   round trip per step, not a delta — and then `reset_prefix_cache()` (`:516-517`),
   which invalidates the prefix cache unconditionally once per step. Expose
   `enable_sleep_mode` in config; `sync_weights` already handles the wake path at
   `:486-492`.
5. `scheduler.py`: the state machine over the call's N prompts. An episode is
   `READY` when it has an observation and no pending generation. Each cycle: take up
   to `generation_concurrency` ready episodes, generate, dispatch resulting tool
   calls to the Phase 4 pool asynchronously, and return each episode to `READY` the
   moment *its own* tool completes. Never wait on a cohort.
6. Batch completion join: the call returns only when all N episodes are `DONE`.
   An episode terminated `worker_crash` is re-admitted from the same scenario so the
   row count and group ordering still match; an episode terminated `timeout` is
   scored normally. Write `test_batch_shape_contract.py` and
   `test_infra_failure_excluded.py` here.
7. `mask.py`: port `_SampleBuilder` / `_chain_to_sequences` semantics. Re-render the
   conversation each turn, compare against accumulated tokens, classify
   `CLEAN | REALIGN | FORK`, and accumulate `drift_tokens`. Do not write span
   arithmetic that assumes concatenation is stable.
8. `rollout_func.py`: assemble the TRL return — `prompt_ids`, `completion_ids`,
   `logprobs`, `env_mask` — positionally aligned to the input prompts, with
   `logprobs` the same length as `completion_ids` and NaN at observation positions.
   Assert `trainer.tools` is empty on entry, since a non-empty `tools` sends TRL down
   `_tool_call_loop` and silently discards the mask. Write
   `test_env_mask_alignment.py` differentially against a fresh re-render — asserting
   both the mask boundaries and the three lengths — plus
   `test_trainer_kwargs_exclusive.py`, before trusting either.
9. Terminal conditions: model emits a final answer, step cap reached,
   unrecoverable environment state, worker timeout, or worker crash. No unbounded
   loops — an overnight Colab run must terminate.
10. `profiler.py`: attribute wall time across generation, `env.step`, parse,
    verifier, tokenization, scheduling, **and weight sync**, built on
    `trl.extras.profiling.profiling_context` rather than a bespoke timer. Weight
    sync is a first-class row because it is a full-parameter round trip every step
    and it invalidates the prefix cache. This is what tells us whether the remaining
    idle time is CPU-bound environments or scheduler overhead, and it is the input to
    any further optimization.
11. Run `test_rollout_equivalence.py` against the async path. It must produce
    rewards identical to the factory path under a fixed seed. Do not measure
    throughput until this passes.
12. `rollout-bench`: run both paths for a fixed episode count on each profile.
    Report episodes/hour, tokens/s, mean and peak GPU utilization, drift-token
    distribution, and the timeline breakdown. Sweep `generation_batch_size` (the
    active pool) and `generation_concurrency`, writing the chosen values into this
    phase's owned profile fields only. Write `artifacts/rollout/ab_report.md`. This
    is the artifact Phase 7 cites when choosing the RL profile.

## Success Criteria

- [ ] `environment_factory` path runs complete episodes and produces verifier rewards, in its own trainer with no `rollout_func`.
- [ ] Trainer-kwargs exclusivity test passes: both `rollout_func` and `tools`/`environment_factory` on one trainer fails loudly.
- [ ] `env_mask` alignment test passes differentially: mask is 1 exactly on model-generated tokens, 0 on every appended observation, verified against a fresh re-render rather than a same-rule fixture. `len(logprobs) == len(completion_ids) == len(env_mask)` asserted, with NaN at observation positions.
- [ ] `sampling/sampling_logp_difference/max` logged, with a stated stop threshold.
- [ ] Factory oracle reuse test passes: running the oracle over two consecutive batches of one scenario yields identical `final_state()` key sets — no attribute survives from the prior episode.
- [ ] Drift-token distribution logged per turn; `FORK` and `REALIGN` counts visible in W&B.
- [ ] Batch shape contract test passes: returned row count and group ordering match the input prompts, including when episodes fail and are replaced.
- [ ] Infrastructure-failure test passes: `worker_crash` episodes are replaced rather than scored; `timeout` episodes are scored.
- [ ] Rollout equivalence test passes: fixed seed + scripted policy → identical rewards on the factory and async paths.
- [ ] No-turn-barrier test passes: one deliberately slow episode does not delay others under a simulated clock.
- [ ] Terminal-condition test covers final answer, step cap, unrecoverable state, timeout, and worker crash.
- [ ] `artifacts/rollout/ab_report.md` reports episodes/hour for both paths on both profiles, with the timeline breakdown including a weight-sync row. Prefix-cache hit rate is measured with per-step sync invalidation, not from a sync-free bench.
- [ ] Async path beats the turn-synchronous baseline in episodes/hour on at least one profile — or the timeline profile explains why not, and that explanation is written down.
- [ ] `generation_batch_size` (active pool) and `generation_concurrency` swept, with chosen values written into this phase's owned profile fields only.
- [ ] vLLM KV budget explicitly capped in config; the 262k default is never inherited.

## Risk Assessment

**`env_mask` silently discarded by a non-empty `tools`.** The mask is only read on
the `else` branch of `if self.tools:`; `environment_factory` fills `tools` from
every env method. Signal: the exclusivity test fails, or `completions/*_length`
equals full completion length despite tool calls. Response: two separately
constructed trainers, an entry assertion in `rollout_func.py`, and the CI test.
Nothing about this failure raises on its own — loss still descends while the model
learns to predict tool output.

**`env_mask` misalignment trains on observations.** Silent in the same way. Signal:
the differential alignment test fails, `drift_tokens` non-zero with `FORK` events,
or `completions/*_length` metrics look implausibly large. Response: the ported
drift classifier plus `sum(env_mask) < len(completion_ids)` for any episode with at
least one tool call. Do not accept a fixture-based test as evidence here — the
fixture and the code share the assumption under test.

**Batch shape mismatch at the TRL boundary.** Returning more or fewer rows than
prompts, or breaking group-contiguous order, fails at `zip(..., strict=True)` or —
worse — silently mis-groups advantages via `rewards.view(-1, num_generations)`.
Signal: a length error at `_calculate_rewards`, or advantages that do not match
hand-computed group means. Response: the batch shape contract test, and episode
replacement on infrastructure failure so N is invariant.

**Async path computes different rewards than the baseline.** A throughput win with
wrong rewards is worse than no win. Signal: the equivalence test fails. Response:
the test gates the phase — the async path does not ship until it matches. Most
likely causes are episode-state bleed through shared mutable objects and mask span
arithmetic; both are testable in isolation.

**Misaligned `logprobs` corrupt the IS ratio silently.** Importance-sampling
correction is on by default, so a `logprobs` array shorter than `completion_ids`
gets right-padded with 0.0 and every model token after the first observation is
compared against the wrong position. Ratios become `exp(±large)`, get clamped, and
training proceeds with no error. Signal:
`sampling/sampling_logp_difference/max` climbing. Response: equal-length arrays with
NaN at observation positions, asserted in the alignment test, and a stop threshold
on that metric.

**The oracle leaks state the async path does not.** TRL pools factory instances and
re-`reset`s them across batches, while upstream's `reset` restores only
`init_config` keys. Signal: the equivalence test fails with the oracle scoring
*higher*, or oracle rewards varying with batch position. Response: `factory_env.reset`
rebinds a fresh `EnvInstance`; the reuse test gates it. If oracle and async still
disagree, suspect the oracle first — never conform the isolated implementation to
the pooled one.

**Weight sync is a per-step full-parameter round trip.** On a PEFT model
`sync_weights` merges, streams every parameter, unmerges, and resets the prefix
cache. Signal: OOM inside `sync_weights` rather than in generation or backward — in
which case the stated fallback order (generations → context → step cap → KV budget)
addresses none of it. Response: weight sync is a budgeted line in the VRAM plan and
a profiler row; `enable_sleep_mode` is available as the lever. Prefix-cache hit rate
must be measured with the invalidation in place.

**Scheduler overhead eats the gain.** Signal: the timeline profile shows scheduling
overhead comparable to generation. Response: the profiler is built before the
optimization for exactly this reason. If overhead dominates, coalesce dispatches
with a small timeout window rather than adding threads.

**Intra-call scope caps the achievable gain.** Removing only the turn barrier
leaves the tail of the batch: near the end of a call, few episodes remain ready and
the GPU drains. Signal: queue depth collapsing in the last quarter of each call
while the timeline shows generation idle. Response: this is the known ceiling of
the design, not a bug — report it with the queue-depth curve. Raising
`generation_batch_size` lengthens the useful middle at the cost of VRAM; that
trade-off is the sweep in step 12. Cross-step pipelining would need
`AsyncGRPOTrainer`, which does not support LoRA or colocate (see Architecture).

**vLLM colocate memory contention.** Signal: OOM during generation on L4 while the
trainer holds its own allocation. Response: lower the vLLM KV fraction first, then
generation concurrency, then step cap. Enable sleep mode between phases if TRL
exposes it. Never resolve it by shortening reasoning.

**TRL experimental API changes under us.** Both `rollout_func` and
`environment_factory` are marked experimental and may change without notice, and
the `if self.tools:` branch that gates `env_mask` is an implementation detail we
depend on. Signal: a signature or return-key mismatch after a version bump, or the
mask silently stopping. Response: pin the TRL version in `pyproject.toml`; upgrade
deliberately, with the equivalence test and the exclusivity test as the gates.

**Prefix caching stops helping.** The G rollouts share a prefix only until they
diverge, which is after the first tool call. Signal: cache hit rate collapsing
after turn 1 in vLLM metrics. Response: expected, not a bug — record the hit-rate
curve and do not over-invest in it.
