---
phase: 7
title: "Agentic GRPO and final results"
status: pending
priority: P1
effort: "5d"
dependencies: [5, 6]
---

# Phase 7: Agentic GRPO and final results

## Overview

Run online GRPO from the SFT checkpoint against held-out executable environments
with verifier rewards, then complete the headline table's third column. Answers the
project's central question: does stateful agentic RL add anything on top of
reasoning SFT at 2B?

## Requirements

**Functional**
- TRL `GRPOTrainer` driven by the Phase 6 async `rollout_func`, colocated vLLM,
  LoRA on the merged SFT checkpoint.
- Reward exactly `R = (checkpoints passed) / K` from the Phase 4 verifier.
  Unmodified.
- Difficulty profiling before training: run the SFT checkpoint over a scenario
  sample, record per-scenario success rate, and prioritize scenarios where
  `0 < P(success) < 1`.
- Per-group reward variance logged every step — the isolation-bug tripwire.
- Adapter pushed to HF Hub throughout; `--resume` survives VM reclamation and
  restores optimizer state, the curriculum/sampler position, and the W&B run id —
  not just weights.
- Final evaluation of SFT+RL under the Phase 5 manifest, asserted identical on the
  invariant set to the Base and SFT runs, reading a pinned revision sha.
- Completed results table plus a written interpretation, including anything that
  did not improve.

**Non-functional**
- No external LLM anywhere in the rollout: Non-Conv only.
- No OOM on the chosen profile at the swept Phase 6 settings.
- Training-reward curve and held-out benchmark curve both logged against GRPO step.

## Architecture

GRPO needs G rollouts of one scenario to compute relative advantage. All G share
`task_id` and `init_config` but must hold genuinely independent environment
instances — the Phase 4 isolation guarantee is what makes the advantage meaningful.

```
scenario x
  ├─ rollout 1 → 0.25
  ├─ rollout 2 → 0.75
  ├─ rollout 3 → 0.50
  └─ rollout 4 → 1.00   → group-relative advantage → LoRA update
```

Reward variance within a group is the signal that matters. A group where all G
rollouts score identically contributes nothing to the gradient. Two causes, opposite
responses: the scenario is trivially easy or impossible (fix by curriculum), or
state is leaking between rollouts (fix by going back to Phase 4). Logging
per-group variance distinguishes them, and confusing the two would mean tuning a
curriculum to work around a correctness bug.

Hence difficulty profiling first. Scenarios where the SFT policy always scores 0 are
unreachable; always 1.0 are already solved. Both waste rollouts. The band in between
is where GRPO advantage carries information. This costs one inference pass over a
scenario sample and needs no LLM judge — just the verifier.

Weight handling: base + SFT LoRA merged into a standalone checkpoint, then a fresh
LoRA for RL. Three cleanly separable artifacts — `base`, `sft`, `sft+rl` — each
independently loadable and evaluable. Continuing the same adapter would blur what
each stage contributed.

Conservative start, then widen. Fewer generations per group, a lower step cap, and a
tight context budget first; loosen only after a stable run. Failing at step 3 of 200
because everything was set to maximum wastes a Colab session.

Resume must restore more than weights. A GRPO run's state is the adapter, the
optimizer state, the position in the curriculum sampler, and the W&B run id.
Restoring only the adapter means the sampler restarts from the top and the run
re-trains on scenarios it already saw, biasing the curriculum toward whatever comes
first in the ordering — invisible in the loss curve. So the resume payload includes
a sampler cursor and the run id, and an interrupted short run in step 7 verifies it
by comparing the scenario sequence across the break.

## Related Code Files

- Create: `src/smolqwen/training/grpo.py` — trainer assembly, LoRA on merged SFT, reward wiring, callbacks, resume
- Create: `src/smolqwen/training/difficulty.py` — scenario success-rate profiling, curriculum ordering by `0 < P < 1`
- Create: `src/smolqwen/training/reward.py` — thin adapter from the Phase 4 verifier to TRL's reward interface
- Create: `src/smolqwen/logging/trajectory_table.py` — sampled rollouts to W&B: reasoning, calls, observations, per-checkpoint verdicts
- Create: `tests/test_group_reward_variance.py` — variance computed per group; a zero-variance group is flagged, not silently dropped
- Create: `tests/test_difficulty_bands.py` — profiler assigns scenarios to easy / band / impossible correctly from fixtures
- Create: `tests/test_grpo_smoke.py` — 2 GRPO steps against fixture scenarios on CPU with a tiny random-weight model
- Modify: `configs/base/grpo.yaml` — learning rate, beta, loss type, curriculum settings (semantics only)
- Modify: `configs/profiles/{l4,a100}.yaml` — this phase's owned fields: `num_generations`, `max_env_steps`
- Create: `tests/test_resume_sampler_cursor.py` — resuming restores the curriculum position; the scenario sequence continues rather than restarting
- Create: `tests/test_infra_failure_not_scored.py` — a `worker_crash` episode reaching the reward function raises instead of scoring
- Modify: `src/smolqwen/cli.py` — wire `profile-difficulty`, `train-grpo`
- Create: `notebooks/03-grpo.ipynb` — thin wrapper
- Create: `artifacts/evaluation/final_results.md` — the completed table plus interpretation

## Implementation Steps

1. `reward.py`: adapt the Phase 4 verifier to TRL's reward interface. Return the
   scalar and attach per-checkpoint booleans as metadata for the trajectory table.
   Nothing else enters the reward — invalid-call rate and step count are logged
   metrics only. Episodes carrying a `worker_crash` terminal reason are excluded by
   Phase 6 before reaching here; if one arrives, raise rather than score it. Scoring
   an infrastructure failure as a low reward teaches the model to avoid a crash it
   did not cause.
2. `difficulty.py`: run the merged SFT checkpoint over a sample of RL scenarios,
   several rollouts each, record success rate per scenario. Classify
   `always_zero` / `band` / `always_one`. Write
   `artifacts/rl/difficulty_profile.json`. Use only the verifier.
3. `grpo.py`: assemble `GRPOTrainer` with the Phase 6 `rollout_func`, colocated
   vLLM, LoRA over the merged SFT checkpoint, the W&B callback, and the Hub push
   callback. Curriculum sampling weighted toward the band.
4. `trajectory_table.py`: log a handful of rollouts per N steps — reasoning, tool
   calls, observations, per-checkpoint verdicts, final reward. This is how a broken
   rollout gets noticed by reading rather than by guessing from a flat curve.
5. Per-group reward variance: log mean, distribution, and the fraction of
   zero-variance groups every step. Set an explicit expectation from the difficulty
   profile; if the zero-variance fraction materially exceeds it, stop and check
   isolation before continuing to train.
6. Smoke run: `--max-steps 5` on the target profile with the conservative settings
   from Phase 6. Confirm reward is computed, advantages are non-degenerate, no OOM,
   and checkpoints push.
7. Short run (~30-50 steps). Confirm the training reward curve moves and held-out
   EnvScaler reward on a small subset does not collapse. A rising training reward
   with falling held-out reward is reward hacking against the verifier — investigate
   before scaling up. Interrupt this run deliberately and resume it: the scenario
   sequence after the break must continue rather than restart, and the W&B run must
   continue rather than fork.
8. Widen settings one axis at a time — `num_generations`, then step cap, then
   context — measuring episodes/hour and peak VRAM at each step. Write the final
   values into this phase's owned profile fields (`num_generations`,
   `max_env_steps`); do not touch Phase 6's `generation_concurrency` or
   `active_pool_multiplier`, which were swept against measured throughput.
9. Full GRPO run on the profile chosen from Phase 6's `artifacts/rollout/ab_report.md`
   episodes/hour comparison. Log the training-reward curve and periodic held-out
   evaluation against GRPO step.
10. Merge the RL adapter. Evaluate SFT+RL through the Phase 5 harness with
    `assert_comparable` against the Base and SFT manifests, reading a pinned
    revision sha rather than `latest_revision`.
11. Write `artifacts/evaluation/final_results.md`: the three-column table with
    per-category BFCL-MT, held-out EnvScaler reward, invalid-call rate, average
    steps, average tokens. Then the interpretation — including whatever did not
    improve. The paper's own 1.7B regressed on τ-bench; an honest mixed result is a
    stronger artifact than a uniformly rosy one.

## Success Criteria

- [ ] `artifacts/rl/difficulty_profile.json` classifies sampled scenarios into always-zero / band / always-one.
- [ ] Group reward variance logged every step, with the zero-variance fraction tracked against the difficulty profile's prediction.
- [ ] Group variance test passes: variance computed correctly; zero-variance groups flagged rather than silently dropped.
- [ ] CPU GRPO smoke test runs 2 steps against fixture scenarios.
- [ ] Full GRPO run completes on the chosen profile without OOM; adapter pushed throughout; `--resume` verified by an interrupted run.
- [ ] Resume test passes: the curriculum position and W&B run id survive a restart; the scenario sequence continues rather than replaying from the top.
- [ ] Infrastructure-failure test passes: a `worker_crash` episode never reaches the reward function as a scoreable rollout.
- [ ] The RL profile choice cites `artifacts/rollout/ab_report.md` episodes/hour, not the Phase 1 capability probe.
- [ ] Training-reward curve and held-out EnvScaler curve both logged against GRPO step in W&B.
- [ ] SFT+RL evaluated with `assert_comparable` passing on the invariant set against the Base and SFT manifests.
- [ ] `artifacts/evaluation/final_results.md` contains the complete three-column table with all secondary metrics.
- [ ] Interpretation written, explicitly covering anything that did not improve.
- [ ] Sampled trajectory tables in W&B show genuine interleaving: reasoning → call → observation → reasoning.

## Risk Assessment

**RL does not beat SFT.** The plan's load-bearing assumption is that a 2B hybrid
model explores well enough for GRPO to extract signal; the paper's 1.7B gained
least from RL of the three sizes tried. Signal: held-out EnvScaler reward flat
across GRPO steps while training reward also stays flat. Response: check the
difficulty profile first — if most groups have zero variance, the curriculum is
wrong and is fixable. If variance is healthy and reward still will not move, report
it as a negative result with the variance evidence. Do not manufacture a gain by
changing the reward.

**Reward hacking against the verifier.** Final-state checks allow many action paths,
which is intended, but a degenerate path may satisfy checks without doing the task.
Signal: training reward rising while held-out reward falls, or trajectory tables
showing a repeated trivial action sequence. Response: read the sampled trajectory
tables — that is what they are for — and report any degenerate strategy found rather
than silently filtering it.

**Zero-variance groups from state leakage, not difficulty.** Signal: zero-variance
fraction well above the difficulty profile's prediction. Response: return to Phase
4's isolation test with the specific failing scenario. Never work around it with
curriculum tuning — that hides a correctness bug behind a hyperparameter.

**Colab session loss mid-run.** Signal: VM reclaimed. Response: adapter pushes on
every save plus W&B run-id resume make this a restart, not a loss. Verify by
deliberately interrupting the short run in step 7.

**Resume silently restarts the curriculum.** Restoring weights but not the sampler
cursor makes the run re-train on already-seen scenarios, over-weighting whatever
sorts first and invalidating the difficulty-band targeting the phase depends on.
Nothing in the loss curve shows it. Signal: the scenario id sequence after a resume
matching the sequence from step 0. Response: the sampler cursor is part of the
resume payload and the resume test compares sequences across the break.

**OOM at wider settings.** Signal: OOM after increasing generations per group or
step cap. Response: revert one axis at a time, in the Phase 6 fallback order —
generations, then context, then step cap, then vLLM KV budget. Never zero out
reasoning.

**Non-monotonic secondary metrics.** RL may raise reward while raising invalid-call
rate or step count. Signal: those metrics worsening as reward improves. Response:
report all of them in the table. A reward gain bought with more invalid calls is a
finding, not something to hide — and this is exactly why they are logged and not
folded into reward.
