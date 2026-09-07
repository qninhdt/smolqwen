---
title: "Per-card SFT attention: FA2 on Ampere-plus, sdpa below"
description: "Delete the flash-attn-triton fallback; resolve attention and dtype from the card with only two implementations."
status: complete
priority: P1
created: 2026-09-04
revised: 2026-09-04
---

# Per-card SFT attention: FA2 on Ampere-plus, sdpa below

Supersedes `plans/260904-0056-t4-padded-sft-fallback/`, which pursued a third
attention implementation for Turing. That approach is abandoned; see
[Why the Triton path was dropped](#why-the-triton-path-was-dropped).

## Outcome

`train-sft`, `train-grpo` and `evaluate` run on any card from sm75 up, with
exactly two attention implementations in the codebase: `flash_attention_2` on
Ampere-plus with the padding-free boundary contract, and `sdpa` below it with
right-padded batches and FP16. No third kernel, no vendored kernel patching, no
extra dependency extra.

## What changed

| Area | Before | After |
|---|---|---|
| Attention on sm75 | registered `t4_padded_attention` adapter over `flash-attn-triton` | `sdpa` |
| Selection input | wheel presence | compute capability, then wheel presence |
| Dtype | `bf16` config flag, per-branch string | `resolve_precision()` — one Toggle carrying the reason |
| SFT padded collator | `padded_flash_attention_collator` (omits mask, adds `position_ids`) | the existing `collator` (keeps `attention_mask`) |
| GRPO dtype | always `bfloat16` when `bf16: true` | resolved per card; FP16 keeps adapters in FP32 |
| Dependency extras | `colab` + `t4` | `colab` only |
| `setup_colab.sh` | `[t4\|colab]` argument | no argument; branches on capability |
| Validation script | `scripts/colab-t4-validation.py` (kernel/adapter/sweep) | `scripts/colab-gpu-validation.py` (probe/kernels/tests/sft/eval/grpo) |

`resolve_attn_implementation` gained a `capability` argument and now downgrades to
`sdpa` below sm80 regardless of whether the `flash_attn` wheel imports, because
FA2's kernels are Ampere-and-newer. `resolve_precision` is new and is the single
place bf16-vs-fp16 is decided; both training stages read it.

Padding-free batch shape is no longer a separate switch: it follows the attention
decision, since only FA2 consumes `cu_seq_lens`. CPU assembly (`require_kernels=False`)
still reports `padding_free=True` so the boundary-metadata tests keep working on a
dev box with no kernel wheels.

`EvalProfile.dtype` is now typed `VllmDtype` (a `Literal`) rather than `str`, which
removed the one standing mypy error in `inference/engine.py`.

## Why the Triton path was dropped

Three measurements from the live T4 session, in the order they landed:

1. **Shared memory.** `flash-attn-triton==0.1.1`'s fixed backward tiles request
   147,456 bytes for Qwen3.5's head dim 256; a T4 exposes 65,536. Worked around by
   executing a source-equivalent in-memory kernel with `(16,16,16,16)` tiles.
2. **Compile time.** Its forward kernel is `@triton.autotune`d over 32 surviving
   configs keyed on `(N_CTX, HEAD_DIM)`. Measured per-config compile times from the
   Triton cache mtimes climbed with shared-memory size: 3s → 7s → 24s → 45s → 133s →
   195s. After 25 minutes of wall time the phase had compiled 30 of 32 configs and
   the GPU had still executed nothing (`utilization.gpu 0%`, 537 MiB resident, the
   process 100% CPU-bound in `ptxas`). That cost recurs per new `(N_CTX, HEAD_DIM)`
   key, so it is not a one-time warm-up for a training run with varying batch widths.
3. **Import shadowing.** The package ships a `flash_attn/__init__.py` shim that
   re-exports its own Triton functions under the official name, so installing it made
   `import flash_attn` resolve to the Triton package on every card, and
   `transformers.utils.is_flash_attn_2_available()` returned False on an Ampere host
   that had the real wheel. It also imports the removed
   `triton.tools.experimental_descriptor`, needing a stub module injected at import.

Against that, `sdpa`'s memory-efficient backend was measured on the same T4 at
head dim 256, FP16, forward+backward: 0.02s / 0.12 GiB at 2K, 0.19s / 0.39 GiB at
8K, 0.65s / 0.75 GiB at 16K, with no compilation step. The default SDPA dispatcher
selects that backend, so nothing has to request it explicitly. `sdpa` reads the
padding mask natively, which is why the padded collator could go back to supplying
one instead of working around a kernel that cannot take it.

## Verification

Run on this host (RTX 3050, sm86, no `causal_conv1d`):

- `uv run pytest -m "not gpu and not dataset"` — 508 passed, 18 deselected
- `uv run mypy` — clean over 158 files
- `uv run ruff check src tests` / `ruff format --check` — clean
- `make smoke` — every stage config resolves on every profile

Measured on the live T4 before it was released: the SDPA envelope above, the
per-config Triton compile times above, and `shared_memory_per_block_optin = 65536`.

## Not verified

No end-to-end GPU run of the new path. `scripts/colab-gpu-validation.py` is written
for it — six phases (`probe`, `kernels`, `tests`, `sft`, `eval`, `grpo`), each in a
fresh child process, results streamed to `artifacts/gpu-validation.json` — but has
not been executed. The T4 session was stopped after the SDPA measurement, and T4
assignment had been returning HTTP 503 earlier in the day.

Specifically open:

- one real `train-sft --profile t4` step on the padded FP16 sdpa path
- `evaluate` through the in-process vLLM engine on a T4, asserting
  `generation_path: vllm` and non-zero generated tokens from the written report
- two GRPO steps with colocated vLLM
- the `gpu`-marked tests (`test_vllm_adapter_capability`, `test_sft_bench_eval_memory_guard`)
- the SFT memory envelope on 16 GB; `configs/profiles/t4.yaml` still says its
  `max_tokens_per_microbatch: 16384` needs live measurement

## Files

- Modified: `src/smolqwen/training/optim.py`, `sft.py`, `grpo.py`, `collate.py`
- Modified: `src/smolqwen/inference/profiles.py`
- Modified: `pyproject.toml`, `configs/profiles/t4.yaml`, `scripts/setup_colab.sh`
- Modified: `docs/optimization-ledger.md`, `docs/rollout.md`
- Modified: `tests/test_token_batching.py`
- Added: `tests/test_sft_device_runtime.py`, `scripts/colab-gpu-validation.py`
- Deleted: `tests/test_t4_sft_runtime.py`, `scripts/colab-t4-validation.py`
