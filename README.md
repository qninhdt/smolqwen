# smolqwen

## EnvScaler submodule

Clone with the pinned dependency included:

```sh
git clone --recurse-submodules <repository-url>
```

For an existing clone, initialize or restore it at the recorded revision:

```sh
git submodule update --init --recursive
```

`third_party/EnvScaler` is pinned to commit `87e667397abacf274858c0964796beb8f984aafe`.

## Evaluation

See [docs/evaluation.md](docs/evaluation.md) for the pinned-checkpoint
evaluation command, BFCL/EnvScaler adapters, manifests, and comparison reports.

## Rollouts

See [docs/rollout.md](docs/rollout.md) for the async rollout contract, the
correctness-first `rollout-bench` workflow, and how to interpret its A/B report.

## Agentic GRPO

See [docs/grpo.md](docs/grpo.md) for difficulty profiling, online GRPO,
checkpoint resume semantics, safety stops, and the target-GPU evidence still
required before final results can be reported.

## Serving

See [docs/serving.md](docs/serving.md) for authenticated Compose and Colab
operation, benchmark and sweep entry points, and the target-GPU evidence still
required before selecting a serving profile.
