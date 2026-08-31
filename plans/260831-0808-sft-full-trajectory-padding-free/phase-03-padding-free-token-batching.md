---
phase: 3
title: "Padding-free correctness and token batching"
status: in_progress
dependencies: [1, 2]
---

# Phase 3: Padding-free correctness and token batching

## Goal

Flatten complete trajectories into variable-row micro-batches without padding or
cross-trajectory attention/recurrent state, bounded by total tokens rather than a
fixed row count.

## Files

- Modify or replace `src/smolqwen/training/collate.py`
- Add `src/smolqwen/training/token_batching.py`
- Modify `src/smolqwen/training/sft.py`
- Modify `src/smolqwen/config_models.py`
- Modify `configs/base/sft.yaml`
- Modify `configs/profiles/l4.yaml`
- Add focused tests for token batching, flattening, resume order, and boundary isolation
- Extend `scripts/colab-l4-batch-sweep.py` with a padding-free equivalence probe

## Steps

1. Add explicit SFT sizing fields: `max_seq_length`,
   `max_tokens_per_microbatch`, and the optimizer accumulation policy. Do not use
   `per_device_train_batch_size` as the semantic token budget.
2. Implement a deterministic seeded batch sampler that greedily fills at most
   32,768 total tokens, never splits a row, and allows row count to vary. Preserve
   exact epoch coverage and a resumable sampler cursor. Keep length bucketing only
   as an ordering optimization.
3. Flatten each sampled row into one `[1, T]` tensor for ids and labels. Emit
   document boundary metadata required by the actual installed kernels:
   reset position ids, cumulative sequence lengths and maximum document length,
   plus causal-convolution sequence indices if the live kernel signature requires
   them.
4. Establish the native-TRL decision empirically. If TRL's own padding-free path
   propagates every boundary through full attention, GDN recurrent state, and
   causal conv, reuse it. Otherwise keep a project-owned collator/model-input
   adapter and do not set a misleading native flag.
5. Add an adversarial boundary test with two distinguishable trajectories. Run
   each separately under padded execution, then together flattened. Compare each
   document's logits, summed supervised loss, and LoRA gradients. Mutate document
   A and assert document B is unchanged.
6. Run the same test on real L4 kernels with FlashAttention 2,
   flash-linear-attention, and causal-conv enabled. A CPU/eager pass does not
   satisfy this gate.
7. Ensure Liger fused CE accepts flattened labels and produces summed loss or a
   trustworthy item count. Add a test where micro-batches have unequal lengths
   and supervised-token fractions.
8. Keep validation deterministic. Evaluation may use padding-free batches, but
   metrics must aggregate by supervised token, not average batch means.

## Validation

- Every flattened batch has `sum(seq_lengths) == input_ids.shape[-1]` and no more
  than 32,768 tokens.
- Boundary metadata reconstructs every original document exactly.
- Every retained uid appears once per epoch; resumed order equals uninterrupted
  order from the saved cursor.
- Padded and padding-free logits/loss/LoRA gradients match within a documented
  bf16 tolerance on CPU/eager where supported and on real L4 fused kernels.
- No document can attend to or inherit GDN/conv state from another document.
- A batch of one works, but telemetry reports that it has no padding-free saving.

## Risks and rollback

- Resetting only `position_ids` is insufficient for Qwen3.5 GDN. Missing
  recurrent or convolution boundaries is a hard correctness failure.
- Passing a custom collator while setting TRL `padding_free=True` currently raises.
  Integrate through one owned path rather than bypassing the guard accidentally.
- Highly variable flattened shapes can trigger compile churn. Measure compile
  counts; introduce a small set of token-envelope buckets only if needed.
- If fused kernels cannot isolate documents, retain full-trajectory shards and
  padded length-grouped training for diagnosis, but do not mark this phase done.
