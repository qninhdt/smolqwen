# Evaluation

The evaluation harness evaluates a pinned checkpoint through the same adapter
and policy boundary for local Transformers, adapter-on-base, merged, and
OpenAI-compatible HTTP runs. A run must provide an explicit `--revision`; the
evaluation path never resolves a moving branch tip.

## Dev and test

The pipeline is `data → baseline → SFT + dev eval → RL + dev eval → test
benchmark → serving`, and the two evaluation sets have different jobs:

| Set | Adapter | Role |
|---|---|---|
| **dev** | `envscaler_heldout` | Selects checkpoints, drives the learning curve, may be scored as often as useful |
| **test** | `bfcl_multi_turn` | Runs **once**, after all training and checkpoint selection, and influences no decision |

The dev set is the held-out EnvScaler slice that GRPO excludes from training
(`grpo.py:435-441`). That exclusion and the eval-side slice are sized by
*independent* config keys — `curriculum.heldout_env_count` /
`heldout_scenarios_per_env` in `configs/base/grpo.yaml` and
`adapter_options.envscaler_heldout.env_count` / `scenarios_per_env` in
`configs/base/eval.yaml`. Both hold 10/8 today, which makes the separation a
coincidence rather than a property, so `tests/test_dev_test_integrity.py`
computes both id sets from the shipped configs and asserts the dev set is a
subset of what training excludes. Raise either key and that test fails rather
than the run silently scoring environments it trained on.

The split is by **task id**, not by environment: the 10 dev environments keep
their other 42 scenarios in training. So a dev score measures generalization to
unseen scenarios, not to unseen environments.

Because BFCL is the test set, no in-training callback may score it. A BFCL number
that moved during a run would have selected something, and the final
`Base | SFT | SFT+RL` table would no longer be a held-out comparison.

One caveat on reading BFCL at all: it scores by comparing final environment state
against a recorded ground truth, all-or-nothing per task, so a task fails on any
state divergence including one caused by an error in the recorded answer. An
unchanged or regressed BFCL number is not automatically a model deficiency, and a
report must say which it is rather than assuming.

```sh
smolqwen evaluate \
  --checkpoint artifacts/models/qwen3.5-2b-sft-merged \
  --revision <checkpoint-commit-sha> \
  --tag sft \
  --adapter bfcl_multi_turn \
  --profile l4
```

The default BFCL set is `multi_turn_base`, `multi_turn_miss_func`,
`multi_turn_miss_param`, and `multi_turn_long_context`. User turns come from
the pinned benchmark files; no user-simulator or judge API key is needed. The
held-out EnvScaler adapter uses the Phase 4 worker pool and reports verifier
reward plus exact-success rate. Both adapters present the byte-identical
non-conversational system prompt used by the released SFT trajectories; the
EnvScaler environment introduction remains in that system message.

Benchmark plugins own their task lifecycle, provenance, and metric aggregation;
the generic runner only coordinates policies and reports. Add a module under
`src/smolqwen/eval/adapters/` exposing `ADAPTER_NAME` and `create_adapter`, then
select it through `adapters` or `--adapter`. Adapter-specific settings belong in
the corresponding `adapter_options` entry.

Each run writes `<tag>.json` and `<tag>.md` under `artifacts/evaluation/`. The
manifest hashes decoding, system prompts, tool schemas, benchmark revision, and
step limits. It also records execution-only details such as backend, dtype,
quantization, speculative decoding, KV budget, batching, caching, and library
versions. Use `write_comparison_report` (or `compare_reports`) from
`smolqwen.eval.report` to join two or more report tags; invariant drift is
rejected before a comparison artifact is written.

For an HTTP serving re-evaluation, record the serving fields on the command so
the resulting row remains self-describing:

```sh
smolqwen evaluate \
  --endpoint http://127.0.0.1:8000/v1 \
  --serving-backend vllm \
  --revision <served-checkpoint-sha> \
  --tag fp8 \
  --served-dtype float8_e4m3fn \
  --quantization fp8 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --chunked-prefill \
  --prefix-caching
```

Do not publish `base_vs_sft` until both pinned checkpoint reports exist and
`write_comparison_report` accepts their invariant manifests. Run on an L4/A100
or point the harness at an already-running OpenAI-compatible endpoint.
