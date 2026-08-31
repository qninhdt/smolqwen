# Optimization ledger — full-trajectory SFT

This document is the authority for the current SFT objective. It supersedes the
2026-08-30/31 numbers derived from per-user-turn segmented shards and padded
fixed-row batches. Those measurements (including 16K samples, `micro_batch: 2`,
2,043 valid tokens/s, and 44.5 h/epoch) do not measure the current training
workload and must not be used to size it.

## Current contract

- One released teacher trajectory produces one schema-v2 record. Historical
  assistant reasoning, tool calls, and text all remain in the context and are
  supervised; system/user/tool-observation tokens are masked.
- The initial cap is 32,768 tokens per trajectory. A longer trajectory is skipped
  whole during conversion; it is never split to manufacture extra samples.
- A padding-free micro-batch is a single flattened token array with complete
  trajectory boundaries. Its maximum is 32,768 total tokens, so the number of
  trajectories in a batch is variable.
- The loss is normalized by the actual count of supervised assistant tokens, not
  by a fixed row count or the 32K token envelope.
- The trainer rejects the old segmented shard schema before loading a model.

## Toggle status

| Toggle | Setting | Status |
|---|---|---|
| `bf16` | on | Required on L4/A100 target hardware. |
| `gradient_checkpointing` | on, non-reentrant | Required for the initial 32K attempt; final memory headroom is unmeasured. |
| `liger_fused_linear_cross_entropy` | on | Required to avoid materializing the large vocabulary logits activation. |
| `adapter_dtype` | `bfloat16` | Avoids PEFT's fp32 adapter path for this bf16 LoRA run. |
| `attn_implementation` | `flash_attention_2` | Required for the target fused-kernel validation. |
| `regional_torch_compile` | on | Uses the known-safe regional exclusions; throughput benefit must be remeasured. |

## Required L4 evidence before an SFT run is accepted

The local development machine has no CUDA L4 nor the required fused attention,
linear-attention, and causal-convolution kernels. CPU tests validate record,
sampler, and loss-accounting invariants only; they cannot prove recurrent or
causal-convolution boundary isolation.

On an L4 with the `colab` dependencies installed, the validation sequence is:

1. Run the padding-free equivalence probe in
   [`scripts/colab-l4-batch-sweep.py`](../scripts/colab-l4-batch-sweep.py). It
   compares padded and flattened logits, supervised loss, and LoRA gradients,
   then mutates trajectory A and requires trajectory B to remain unchanged.
2. Run a 32K one-step probe with a worst-case 32K trajectory and mixed shorter
   trajectories. Record allocated/reserved/peak VRAM and retain operational
   headroom.
3. Run at least 30 real-shard steps. Record supervised tokens/s, total tokens/s,
   finite loss, compile behavior, and stable post-warm-up memory.

If any boundary check fails, padding-free must remain disabled for training. If
the 32K envelope lacks safe headroom, measure 24,576, record the evidence, and
regenerate the full-trajectory artifacts with that cap. Do not infer either
decision from the previous segmented benchmark.
