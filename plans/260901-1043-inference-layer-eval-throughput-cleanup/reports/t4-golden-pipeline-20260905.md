---
title: "T4 golden pipeline validation"
status: partial
date: 2026-09-05
---

# T4 golden pipeline — 2026-09-05

## Scope and outcome

This was a free Colab Tesla T4 rehearsal driven by the `colab` CLI, with one
session at a time. It validates the shared inference and training wiring before
an L4 allocation; no T4 number closes Phase 10 or transfers to L4.

`probe`, kernel imports, bounded SFT, and bounded vLLM evaluation completed. The
bounded GRPO path also completed after one reproducible software failure and a
cause-aligned retry in the same session. The first `t4-golden-5` attempt reached
2/2 optimizer steps but entered an unrelated native Trainer evaluation and hit a
TorchDynamo shape-guard assertion. Native GRPO evaluation is now disabled in
favor of the project's shared benchmark callback; the retry reached 2/2 and
persisted its checkpoint, adapter, completions, and a passing controller row.

## Measured environment

- Hardware: Tesla T4, compute capability 7.5, 14.563 GiB VRAM.
- Software: torch 2.11.0+cu130, vLLM 0.26.0, Transformers 5.16.1, TRL 1.12.0,
  Python 3.13.15; installed with `uv sync --locked --extra colab`.
- Model: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- T4 routing: FP16 and padded SDPA for SFT; the profile uses a 16K engine
  context. The colocated GRPO smoke overrides KV fraction to `0.48`, uses
  `max_steps=2`, two generations, two environment steps, 256 new tokens per
  step, and vLLM sleep mode. Solo evaluation overrides KV fraction to `0.80`.

## Phase results

| Phase | Result | Evidence |
|---|---|---|
| `probe` | pass | 17.17 s; GPU and runtime identity recorded |
| `kernels` | pass | 274.17 s; `flash_attn`, `causal_conv1d`, `fla`, and `liger_kernel` imported |
| `sft` | pass | 386.73 s total, 302.9 s training; validation loss 10.7215; adapter and checkpoint artifacts written |
| `eval` | pass as a wiring smoke | 766.54 s; `generation_path=vllm`; vLLM startup 508.85 s and task execution 89.4 s |
| `grpo` | pass as a wiring smoke | first attempt failed in native Trainer eval; retry passed in 1263.99 s with `grpo two steps ok` and return code 0 |

The evaluation smoke recorded average generated tokens `815.5`, average steps
`2.0`, score `0.36875`, check-pass count `5.5`, exact success rate `0.0`, invalid
call rate `0.5`, terminal turn-cap rate `1.0`, and truncation rate `1.0`. These
settings intentionally use a tiny bounded subset and terminal caps, so this is
not a capability baseline or agreement measurement.

## GRPO findings

The following attempts were made on the T4:

1. KV fractions `0.20` and `0.35` could not reserve a KV cache after colocated
   trainer memory was accounted for.
2. At `0.45`, the engine started but the project guard correctly rejected the
   missing observable prefix-cache setting. A compatibility bridge now forces
   `enable_prefix_caching=True` while constructing the TRL 1.12 vLLM engine,
   whose constructor does not expose that parameter directly.
3. With the original 4K GRPO context, oversized tool-schema prompts were
   terminalized with an empty completion; TRL then indexed `ids[-1]`. The smoke
   harness now uses 16K, where the model generates non-empty rollouts.
4. At 16K without sleep mode, KV fractions `0.65`, `0.50`, and `0.48` reached
   Liger backward but exhausted the 14.56 GiB card. The latest persisted failure
   tried to allocate 908 MiB with 379.81 MiB free in
   `liger_kernel/chunked_loss/fused_linear_ppo.py`.
5. The first persisted `0.48`/16K sleep-mode attempt reached both optimizer
   steps, then failed at the final native `Trainer.evaluate()` call with
   `AssertionError: sources must not be empty for symbol s37` from
   TorchDynamo's shape-guard generation inside Liger's fused GRPO loss. It was
   not an OOM: the log reached `100% 2/2`, and the failure stack was entirely in
   `evaluation_loop`/`prediction_step`.
6. The cause-aligned fix sets GRPO's Transformers `eval_strategy` to `no`.
   Native evaluation is not the project's benchmark path: `BenchEvalCallback`
   owns held-out dev scoring through the shared turn engine when `bench_eval` is
   enabled. The local guard suite passed 36 tests after this change.
7. The same `t4-golden-5` session was retried with the fix. It completed in
   `1263.99 s`, logged `100%|...| 2/2`, printed `grpo two steps ok`, returned 0,
   and wrote `checkpoint-2/`, `adapter/`, and `completions/`. No retry log line
   contains a traceback, OOM, or the previous TorchDynamo assertion.

The callback lifecycle shim added no-op handling for unowned Transformers
`on_*` events. This fixed the concrete `on_train_begin` failure without importing
Transformers at module scope. The 16K sizing and sleep-mode overrides are
bounded harness settings, not production training defaults.

## Operational conclusion

The current T4 evidence closes the rehearsal's probe, kernel, SFT, vLLM
evaluation wiring, and bounded GRPO trainer/rollout smoke. It does not close the
L4-only criteria: baseline agreement, throughput split, concurrency stability,
in-training eval cost, SFT memory envelope, failure isolation, non-TTY
rendering, or the final BFCL run. The T4 smoke deliberately left
`bench_eval.enabled=false`, so `grpo/bench_*` agreement and its 10% cost bound
remain unmeasured. `t4-golden-5` was stopped after artifact download; no local
keepalive, controller, or remote session remains.

## Unresolved Questions

- Does the shared `BenchEvalCallback` complete and stay within the 10% training
  wall-time budget on the target L4 with `bench_eval.enabled=true`?
- Do the L4 adapter, agreement, throughput/stability, SFT envelope, failure
  isolation, non-TTY, and final BFCL criteria pass under the required manifest?
