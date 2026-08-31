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
