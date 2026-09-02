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

## In-training dev eval and the envelope

`bench_eval` scores a held-out dev subset during SFT, so a run produces a
capability curve rather than only a loss curve. It is off by default because it
needs a card, and because it shares that card with the trainer.

```yaml
bench_eval:
  enabled: true
  adapter: envscaler_heldout
  every_steps: 0         # save boundaries only — see below
  task_limit: 16
```

`every_steps: 0` is not merely the cheapest cadence here, it is the only correct
one. The scored weights are `checkpoint-N` on disk, which exists only at a save
boundary; an interval eval would have nothing to score. With `save_steps: 100`
that is every 100 optimizer steps.

There is no weight sync. GRPO syncs because generation is part of its algorithm
and must run under the current policy; SFT evaluation scores a saved checkpoint,
and base weights never change during SFT — only the adapter does. So the cycle is:

```
on_save (TRL wrote checkpoint-N; the store copied and pushed it)
   → wake_up()                     # base weights back in VRAM
   → register checkpoint-N as a LoRA adapter under a fresh id
   → score the dev subset ; log sft/bench_*
   → sleep()                       # weights offloaded, KV freed
   → training continues, optimizer state never touched
```

Two ordering facts this depends on:

- **The engine is built and slept before the trainer exists.**
  `gpu_memory_utilization` sizes vLLM's KV pool against *total* GPU memory, not
  against what is free. Built after a resident trainer it either OOMs or reserves
  against a figure it cannot honour on 24 GB. Built first, the trainer sizes
  itself against what remains.
- **Each boundary registers a fresh adapter name.** vLLM caches LoRA weights by
  integer id, and the id is derived from how many adapters have been registered.
  Reusing one name would hand vLLM the same id with a new path and serve step
  100's weights for step 200's score — a flat curve made of plausible numbers.

The step-0 anchor scores the **base model with no adapter**, because TRL has saved
nothing yet. That is also the `base` arm of the final `Base | SFT | SFT+RL` table,
so the anchor is a number that already means something.

`eval_loss` and `sft/bench_*` are not the same measurement and are not expected to
move together. `eval_loss` is teacher-forced likelihood on `val.jsonl`;
`sft/bench_*` is generation under the real tool harness, scored by the verifier.
A run can improve one and not the other.

Whether both fit on an L4 is the open measurement. `sleep(level=1)` offloads
weights to the host and discards the KV cache, but how much that returns is a
runtime property of the pin and the card.
`tests/test_sft_bench_eval_memory_guard.py` measures it with
`reset_peak_memory_stats()` plus `memory_allocated()` — `max_memory_allocated()`
is a monotonic high-water mark and `memory_reserved()` does not shrink without
`empty_cache()`, so neither can show a release. If the release is too small to
hold the 32K envelope beside the engine, the fallback is to eval in a subprocess
against the saved adapter; do not shrink the envelope, which this document owns.

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
4. With `bench_eval.enabled: true`, confirm `sleep()` returns enough that the 32K
   envelope survives, that the engine is asleep during training steps (a
   non-monotonic `memory_allocated()` reading), and that the summed
   `sft/bench_wall_s` stays at or under 10% of training wall time.

If any boundary check fails, padding-free must remain disabled for training. If
the 32K envelope lacks safe headroom, measure 24,576, record the evidence, and
regenerate the full-trajectory artifacts with that cap. Do not infer either
decision from the previous segmented benchmark.
