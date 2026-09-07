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

## Train and resume

Start training with:

```sh
smolqwen train-grpo --profile l4
```

The command logs scenario/catalog loading, tokenizer and worker-pool startup,
trainer/vLLM construction, optimizer progress, and checkpoint saves. Long phases
emit heartbeat lines. On Colab GPU runtimes human output streams on stdout; set
`SMOLQWEN_LOG_STREAM=stderr` when a shell pipeline needs stdout to remain
machine-readable.

Resume after interruption with:

```sh
smolqwen train-grpo --profile l4 --resume
```

Every saved checkpoint pushes the adapter and Transformers trainer state along
with `resume_state.json`. That marker carries the optimizer step, scenario
sampler cursor, and W&B run ID. The trainer disables Transformers' automatic
data skip because the cursor-owning sampler is authoritative; resuming therefore
continues the scenario order rather than replaying its prefix.

The production trainer is constructed with `rollout_func`, `tools=None`, and
`environment_factory=None`. This is required for TRL to consume the returned
environment mask. Colocated vLLM prefix caching is required and asserted from
the live engine after construction.

The shipped `all-linear` LoRA target excludes Qwen3.5's unused visual tower because
this pipeline is text-only; otherwise a colocated text-only vLLM adapter rejects the
visual keys during weight sync.

The rollout's reasoning mode is `enable_thinking` in
[`configs/base/grpo.yaml`](../configs/base/grpo.yaml) (default `false`). `false`
renders every generation prompt with Qwen's closed empty think block and decodes
completions as content. The in-training dev eval copies this value into its eval
config, so the dev set always measures the distribution rollout trains on — it
should also match the `enable_thinking` the SFT shards were prepared with.

## Monitoring and stop conditions

Training logs verifier reward, the in-training dev score, sampled trajectories,
and per-group reward variance. Scenarios are sampled uniformly in a seeded order.
Trajectory rows include reasoning, calls,
observations, checkpoint verdicts, invalid calls, and step count.

### In-training dev eval

The reward series above is computed over uniformly sampled training scenarios.
`bench_eval` adds a fixed dev slice
of BFCL multi-turn base on the same benchmark `smolqwen evaluate` runs, through
the same turn engine and the same aggregation, so one number means one thing in
both places. Dev and test coincide in this experiment's design (the upstream
EnvScaler setup validates on the same benchmark); the final BFCL number therefore
selects nothing.

```yaml
bench_eval:
  enabled: true
  adapter: bfcl_multi_turn
  every_steps: 20       # equals save_steps, so every score attaches to a checkpoint
  task_limit: 128       # of the 200 multi-turn-base tasks
```

The GRPO trainer disables Transformers' native evaluation loop and runs no
`eval_dataset`. Dev benchmark evaluation is owned by `BenchEvalCallback` above, so
it uses the shared turn engine and aggregation instead of triggering a second
GRPO/Liger loss pass. With `bench_eval.enabled: false`, training performs no
in-training benchmark evaluation.

Three things about the resulting series are worth knowing before reading it:

- **`grpo/bench_weight_version` rides alongside every score.** At a callback
  boundary the optimizer step has already been applied, and generation runs once per
  accumulation window, so the colocated engine can hold weights up to `grad_accum`
  steps old. The callback calls `sync_weights()` explicitly and records
  `step-N.sync-M`, which is what makes a later comparison against `evaluate` "at the
  same revision" falsifiable rather than merely plausible.
- **`grpo/bench_wall_s` is the cost, measured.** Each boundary scores up to 128
  tasks; read the recorded per-boundary cost before widening `task_limit` or
  tightening `every_steps`. The setting is checked against this number, not
  asserted in a comment.
- **A step is scored once.** If an interval and save boundary coincide, a successful
  eval is not repeated. A failed interval attempt can still retry at the save hook,
  without losing the boundary.
- **The adapter is named, never inherited.** `bench_eval.adapter` names the
  benchmark directly; the callback never iterates `eval.yaml`'s adapter list.
- **The turn engine is sized by `vllm_max_model_len`, not by the eval profile.**
  `resolve("eval")` takes no `--profile`, so the callback would otherwise carry
  `ProfileConfig` defaults — 32K context and width 8 against a colocated engine built
  at 16K, on every shipped profile. The turn engine would admit a prefix the engine
  cannot accept and vLLM raises `The decoder prompt (length N) is longer than the
  maximum model length`, turning every boundary into `bench_failed`.
  `bench_eval_config` substitutes this run's profile with `max_seq_length` set to the
  engine's own bound; `tests/test_grpo_args.py` asserts the two agree per profile.

An eval failure logs `bench_failed`, releases its environments, and training
continues. Releasing matters: a leaked episode set turns one recoverable failure
into a pool at capacity at every later boundary.

One condition stops training rather than silently accepting corrupt evidence:

- TRL's sampling-logprob difference exceeds the configured alignment threshold.

A `worker_crash` is an infrastructure failure and raises at the reward boundary;
it is never converted into a low policy reward.

## Final evaluation

After the full target-GPU run, merge the RL adapter and evaluate the pinned
SFT+RL revision through the Phase 5 harness. `assert_comparable` must pass on the
invariant set against Base and SFT manifests. Only then replace the pending cells
in [`artifacts/evaluation/final_results.md`](../artifacts/evaluation/final_results.md)
and write the interpretation, including metrics that stayed flat or regressed.
