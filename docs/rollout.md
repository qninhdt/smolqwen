# Rollout scheduling

The production GRPO path uses an intra-step ready queue: episodes that are ready
to generate can advance without waiting for slower environments at the same turn
index. The scheduler still returns exactly the prompts supplied by one TRL
`rollout_func` call, in their original order; it is not a cross-training-step
pipeline. The executable contract lives in
[`src/smolqwen/rollout/`](../src/smolqwen/rollout/), with semantic settings in
[`configs/base/grpo.yaml`](../configs/base/grpo.yaml) and GPU sizing in
[`configs/profiles/`](../configs/profiles/).

## Correctness before throughput

Run the benchmark from the repository root:

```sh
smolqwen rollout-bench --profile l4 --episodes 64
```

Use `--profile a100` for the A100 sizing profile, `--budgets PATH` when the
budget artifact is not at `artifacts/data/budgets.json`, and `--paths` to select
the scripted diagnostic rows (the default is `serial_oracle,async`). The CLI options
are owned by [`src/smolqwen/cli.py`](../src/smolqwen/cli.py); benchmark execution
and report labeling are owned by
[`src/smolqwen/rollout/bench.py`](../src/smolqwen/rollout/bench.py).

Every invocation first compares a serial factory-adapter semantic oracle with
the async scheduler under the same deterministic scripted policy, real worker
pool, and scenario. Rewards, observations, row count, and positional alignment
must be equivalent. This oracle is not TRL's batched `_tool_call_loop`, so its
throughput is diagnostic only and is not the accepted baseline A/B measurement.

The two rollout paths must also remain separate at training time:

- `factory_oracle` is the eventual TRL correctness baseline. Its trainer is
  constructed with `environment_factory` and without `rollout_func`.
- `async` is the production path. Its trainer is constructed with
  `rollout_func`, `tools=None`, and `environment_factory=None` so TRL consumes the
  returned `env_mask` instead of silently rebuilding an all-ones tool mask.

## Reading the report

The required artifact is
[`artifacts/rollout/ab_report.md`](../artifacts/rollout/ab_report.md). The current
command records two different evidence levels:

- The **scripted diagnostic comparison** contrasts the serial semantic oracle
  with the ready queue on CPU. It is
  useful for validating scheduler overhead and environment concurrency, but its
  zero GPU-utilization fields are not GPU measurements.
- The **colocated-vLLM A/B** must come from separately constructed real trainers
  so the timeline includes LoRA weight synchronization, prefix-cache
  invalidation, and actual GPU generation. Those runs are the evidence used to
  choose measured `generation_concurrency`, active-pool sizing, and KV budget for
  the L4 and A100 profiles.

The local RTX 3050 has only 4 GB of VRAM, and no SFT checkpoint is currently
available for the real trainer comparison. It therefore cannot produce the
target L4/A100 measurements. Until those profile sweeps run on the target GPU
with the required checkpoint, the L4/A100 values remain unmeasured placeholders
and the report's colocated-vLLM section must remain `PENDING`.
