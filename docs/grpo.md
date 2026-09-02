# Agentic GRPO

Phase 7 trains a fresh LoRA adapter on the merged SFT checkpoint. Generation
uses the Phase 6 asynchronous `rollout_func` with colocated vLLM; executable
environment code and verifier functions remain inside the isolated worker pool.
The scalar reward is passed through unchanged as `checkpoints passed / K`.

## Prerequisites

Before a real run, complete the target-profile colocated-vLLM comparison in
[`artifacts/rollout/ab_report.md`](../artifacts/rollout/ab_report.md) and use its
measured episodes/hour result to select `l4` or `a100`. The merged SFT checkpoint
configured by `model_id` must exist, and a Hub repository plus W&B credentials
should be configured for a reclaimable Colab VM.

The local RTX 3050 has 4 GB of VRAM and cannot run the target profile. CPU tests
prove trainer, reward, rollout, and resume wiring; they do not provide OOM,
throughput, learning-curve, or benchmark evidence.

## Profile difficulty

Run several verifier-scored rollouts per sampled scenario before training:

```sh
smolqwen profile-difficulty --profile l4
```

Use `--checkpoint PATH` and `--revision SHA` to override the configured merged
SFT checkpoint. The command writes
`artifacts/rl/difficulty_profile.json`, classifying a scenario as:

- `always_zero`: no rollout fully passed its checklist;
- `band`: some, but not all, rollouts fully passed;
- `always_one`: every rollout fully passed.

Partial verifier rewards are retained as `mean_reward`, but success probability
means full-checklist success. GRPO samples the profiled scenarios in a seeded,
weighted order that prioritizes the band. The held-out evaluation slice is
excluded from this curriculum.

## Train and resume

Start training with:

```sh
smolqwen train-grpo --profile l4
```

Resume after interruption with:

```sh
smolqwen train-grpo --profile l4 --resume
```

Every saved checkpoint pushes the adapter and Transformers trainer state along
with `resume_state.json`. That marker carries the optimizer step, curriculum
sampler cursor, and W&B run ID. The trainer disables Transformers' automatic
data skip because the cursor-owning sampler is authoritative; resuming therefore
continues the scenario order rather than replaying its prefix.

The production trainer is constructed with `rollout_func`, `tools=None`, and
`environment_factory=None`. This is required for TRL to consume the returned
environment mask. Colocated vLLM prefix caching is required and asserted from
the live engine after construction.

## Monitoring and stop conditions

Training logs verifier reward, held-out evaluation reward, sampled trajectories,
and per-group reward variance. Trajectory rows include reasoning, calls,
observations, checkpoint verdicts, invalid calls, and step count.

### In-training dev eval

The reward series above is computed over *training* scenarios under curriculum
weighting — a biased sample by construction. `bench_eval` adds a held-out score on
the same dev set `smolqwen evaluate` uses, through the same turn engine and the
same aggregation, so one number means one thing in both places.

```yaml
bench_eval:
  enabled: true          # off by default: it needs a card
  adapter: envscaler_heldout
  every_steps: 0         # 0 = save boundaries only
  task_limit: 16
```

Three things about the resulting series are worth knowing before reading it:

- **`grpo/bench_weight_version` rides alongside every score.** At a callback
  boundary the optimizer step has already been applied, and generation runs once per
  accumulation window, so the colocated engine can hold weights up to `grad_accum`
  steps old. The callback calls `sync_weights()` explicitly and records
  `step-N.sync-M`, which is what makes a later comparison against `evaluate` "at the
  same revision" falsifiable rather than merely plausible.
- **`grpo/bench_wall_s` is the cost, measured.** The budget is 10% of training wall
  time; widen `every_steps` or lower `task_limit` if the recorded total exceeds it.
  The setting is checked against this number, not asserted in a comment.
- **The adapter is named, never inherited.** `configs/base/eval.yaml` lists BFCL too,
  and BFCL is the test set. A benchmark used to select checkpoints is a dev set, so
  scoring it here would void the final `Base | SFT | SFT+RL` comparison;
  `bench_eval.assert_dev_adapter` refuses a test-set name.

An eval failure logs `bench_failed`, releases its environments, and training
continues. Releasing matters: a leaked episode set turns one recoverable failure
into a pool at capacity at every later boundary.

Two conditions stop training rather than silently accepting corrupt evidence:

- TRL's sampling-logprob difference exceeds the configured alignment threshold.
- Observed zero-variance groups materially exceed the probability predicted by
  the difficulty profile after the configured warm-up steps. Investigate worker
  isolation before changing curriculum weights.

A `worker_crash` is an infrastructure failure and raises at the reward boundary;
it is never converted into a low policy reward.

## Final evaluation

After the full target-GPU run, merge the RL adapter and evaluate the pinned
SFT+RL revision through the Phase 5 harness. `assert_comparable` must pass on the
invariant set against Base and SFT manifests. Only then replace the pending cells
in [`artifacts/evaluation/final_results.md`](../artifacts/evaluation/final_results.md)
and write the interpretation, including metrics that stayed flat or regressed.
