# Test Report — 2026-09-03 — inference/adapter CPU verification

## Summary

Focused and full CPU verification passed. No source, test, or configuration files
were changed; the unrelated `third_party/EnvScaler` dirty state was left alone.

## Test Results

| Check | Result |
|---|---|
| Focused inference/eval/SFT/fixture command | 49 passed, 3 deselected, 5.26s |
| `make test-ci` | 483 passed, 18 deselected, 85.44s |
| `make check` | Ruff, format check, and strict mypy passed (157 files) |
| Full-suite warnings | 2 existing TRL experimental-feature warnings |
| GPU deselections | 5: 2 memory-guard, 3 vLLM adapter-capability |
| Dataset deselections | 13: release/data-dependent tests |

Focused command:

```text
uv run pytest tests/test_inference_engine_contract.py tests/test_eval_generation_path.py \
  tests/test_sft_bench_eval_boundary.py tests/test_vllm_adapter_capability.py \
  tests/test_vllm_fixture.py -m "not gpu and not dataset"
```

## Contract Verification

- Deterministic non-zero adapter preflight: CPU fake-vLLM tests passed. A forced
  non-zero adapter whose adapted probes equal base probes raises
  `AdapterCapabilityError`; `offline_engine_for_eval` performs the extra adapted
  and base calls before returning.
- Fallback boundary: recognized typed/message LoRA refusals select
  `transformers`. An additional no-write exception matrix verified OOM,
  `FileNotFoundError`, corrupt-checkpoint, and unrelated errors propagate.
- Worker memory accounting: `collective_rpc` is used with the reset flag in the
  worker and tensor-parallel readings `[17, 29]` sum to `46` in the contract test.
- Qwen3.5 fixture: the released wrapper shape, text/vision config metadata, and
  processor files passed `test_vllm_fixture.py`.
- SFT boundary: checkpoint-specific adapter names/IDs, validation, missing-save
  failure, step-zero base anchor, wake/score/sleep cleanup, profile inheritance,
  weight version, and checkpoint sidecar all passed.

## Coverage

Full CPU run with temporary `/tmp` coverage storage and branch measurement:

| Module | Statements | Missed | Branches | Partial | Coverage |
|---|---:|---:|---:|---:|---:|
| `eval/batched.py` | 112 | 6 | 24 | 3 | 93% |
| `inference/engine.py` | 244 | 63 | 78 | 10 | 70% |
| `training/checkpoint_eval.py` | 65 | 3 | 14 | 4 | 91% |
| Total (three modules) | 421 | 72 | 116 | 17 | 79% |

The remaining engine misses are mainly real-vLLM-only success/error paths and
adapter weight-file inspection (`safetensors`/`.bin`), plus empty-RPC/no-reading,
context-manager, backend, and no-completion branches. The automated suite has
refusal and OOM coverage, but no persistent cases for each missing/corrupt/
unrelated exception category; the supplemental matrix covered those behaviors in
this run.

## GPU Evidence Boundary

vLLM is absent locally, so no GPU-marked tests were run as passes. The existing
T4 rehearsal supports real T4 base-engine build, batched generation, token-ID
alignment, and sleep/wake; it does not establish adapter acceptance or any L4
claim. Real L4 adapter behavior, worker-memory release at the L4 profile, and
throughput/envelope measurements remain unverified.

## Recommendations

1. On a pinned vLLM GPU host, run the five deselected tests and record the actual
   adapter accept/refuse result and worker sleep-memory readings.
2. Add persisted CPU regression cases for typed/cause-chained refusals and the
   missing, corrupt, and unrelated propagation categories if test-suite coverage
   is expected to encode the supplemental matrix.

## Unresolved Questions

- Does the released Qwen3.5 `all-linear` adapter load and measurably affect output
  under the pinned vLLM version on the target L4?
- Does level-1 sleep release enough memory on that L4 beside both trainers?

Status: DONE_WITH_CONCERNS
Summary: Focused and full CPU verification passed, including the inference, fallback, worker-accounting, Qwen3.5 fixture, and SFT-boundary contracts. Real vLLM adapter behavior and L4 memory/throughput evidence remain unverified; full CPU coverage is 79% overall, with the largest gap in real-vLLM-only engine branches.
Concerns/Blockers: vLLM is absent locally; 5 GPU tests were correctly deselected. The T4 report is valid only for its measured T4 claims and does not substitute for L4 or adapter-acceptance evidence.
