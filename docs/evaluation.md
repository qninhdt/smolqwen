# Evaluation

The evaluation harness evaluates a pinned checkpoint through the same adapter
and policy boundary for local Transformers, adapter-on-base, merged, and
OpenAI-compatible HTTP runs. A run must provide an explicit `--revision`; the
evaluation path never resolves a moving branch tip.

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

Evaluation prints stage-level logs and shows a task-counted progress bar for each
adapter, including rate/ETA and periodic score, step, and generated-token
updates. The executable owner is `_evaluation_progress` in
`src/smolqwen/eval/runner.py`.

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
