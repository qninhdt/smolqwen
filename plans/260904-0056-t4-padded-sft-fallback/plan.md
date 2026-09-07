---
title: "T4 padded FP16 SFT fallback"
description: "Allow Qwen3.5 LoRA SFT on T4 through flash-attn-triton's padded causal attention path."
status: abandoned
priority: P1
created: 2026-09-04
revised: 2026-09-04
---

# T4 padded FP16 SFT fallback

**Abandoned 2026-09-04, superseded by `plans/260904-0741-per-card-sft-attention/`.**
The outcome below was reached a different way: sm75 now runs `sdpa` with padded
FP16 batches, and no third attention implementation exists. `flash-attn-triton`
spent 25 minutes of live T4 wall time autotune-compiling its d256 kernels without
executing a single step, and its `flash_attn` shim shadowed the official wheel on
every card. The superseding plan records both measurements. Everything below is
kept as the record of what was tried.


## Outcome

`smolqwen train-sft --profile t4` can assemble and execute the existing
full-trajectory SFT contract on a T4 using FP16 and the T4-compatible
`flash-attn-triton` padded attention implementation. L4/A100 keep the current
padding-free BF16 path.

## Constraints

- Preserve the stored schema-v2 `input_ids`/`labels` semantics and LoRA targets.
- Keep the L4/A100 padding-free path and its `cu_seq_lens` boundary contract unchanged.
- Do not claim that the T4 profile fits the 32K envelope; use the checked-in T4
  sizing cap and require live GPU validation for memory/performance claims.
- Keep the hybrid Qwen3.5 GDN and causal-convolution kernels in the path.
- Preserve the dirty worktree and do not modify `third_party/EnvScaler`.
- Add no dependency source build; pin the T4 attention package in an optional
  dependency extra and lock it.

## Non-goals

- Do not implement varlen support in `flash-attn-triton`.
- Do not replace or alter FLA GDN or `causal-conv1d` kernels.
- Do not change the L4/A100 benchmark contract, token envelope, or BF16 config.
- Do not commit, push, or create a PR without an explicit user request.

## Acceptance criteria

- The T4 runtime selects the registered `t4_padded_attention` adapter backed by
  `flash-attn-triton`, FP16, and a padded collator
  only on compute capability 7.5 when the package is installed.
- The padded collator right-pads rows, preserves labels, and omits the attention
  mask so the custom kernel receives only its supported causal path.
- A regression test proves L4/A100/default CPU assembly remains padding-free and
  a T4 runtime decision is padded FP16.
- The custom attention interface registers with Transformers and converts its
  `[batch, heads, sequence, head_dim]` tensors to the package's
  `[batch, sequence, heads, head_dim]` API.
- `flash-attn-triton` is available through a locked `t4` dependency extra and
  setup instructions identify the extra.
- Focused tests, lint, type checks, and the full CPU suite pass; live T4
  execution remains explicitly recorded as pending if no T4 is available.

## Files

- Modify: `src/smolqwen/training/collate.py`
- Modify: `src/smolqwen/training/optim.py`
- Modify: `src/smolqwen/training/sft.py`
- Modify: `src/smolqwen/config_models.py` only if the runtime decision needs a
  typed effective dtype/path contract
- Modify: `configs/profiles/t4.yaml`, `pyproject.toml`, `scripts/setup_colab.sh`
- Modify: focused SFT/config/optimization tests and the smallest owning docs
- Create: no product module unless the attention interface has a clear boundary

## Validation and rollback

Run focused unit tests first, then lint/mypy and the full CPU suite. On a T4,
run `probe`, package smoke tests, one padded forward/backward step, and a memory
sweep before treating the profile as usable. Roll back by removing the T4 runtime
branch and optional extra while preserving the existing padding-free path.

## Unresolved questions

- Whether 16,384 tokens and the current all-linear LoRA envelope fit in the
  available 14.56 GiB on the actual target T4 requires live measurement.
