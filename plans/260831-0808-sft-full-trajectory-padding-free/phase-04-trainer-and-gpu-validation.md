---
phase: 4
title: "Trainer integration and L4 validation"
status: in_progress
dependencies: [3]
---

# Phase 4: Trainer integration and L4 validation

## Goal

Integrate token-budget padding-free batches into SFT, normalize optimizer updates
by actual supervised tokens, measure the 32K L4 envelope, and replace stale
maintainer authority with verified results.

## Files

- Modify `src/smolqwen/training/sft.py`
- Modify `src/smolqwen/tracking.py`
- Modify `src/smolqwen/config_models.py`
- Modify `configs/base/sft.yaml`
- Modify `configs/profiles/{l4,a100}.yaml`
- Modify `scripts/colab-l4-sft-speed.py`
- Modify `scripts/colab-l4-batch-sweep.py`
- Modify `tests/test_sft_assembly.py`
- Modify `tests/test_sft_smoke.py`
- Modify `tests/test_batch_shape_contract.py`
- Update the owning Phase 2 and Phase 3 plan documents
- Update `docs/optimization-ledger.md`

## Steps

1. Replace fixed-row trainer assembly with the token-budget sampler and
   padding-free collator. Feed actual token counts to throughput telemetry; remove
   `micro_batch * max_seq_length` estimates.
2. Verify how the pinned Transformers/TRL versions pass `num_items_in_batch`
   across gradient accumulation. Use that mechanism only if it represents the
   total non-ignored labels for the whole accumulation window. Otherwise add a
   narrowly scoped trainer override that accumulates summed loss and scales
   gradients once by the actual supervised-token total.
3. Log per optimizer update: total tokens, supervised tokens, trajectories,
   micro-batches, maximum trajectory length, padding saved, tokens/s, step time,
   allocated/reserved/peak VRAM, and compile count.
4. Preserve effective optimization semantics across resume. Save sampler cursor,
   accumulation progress where supported, optimizer/scheduler state, and W&B run
   id. Fail if a checkpoint was created under a different shard/template schema.
5. Run a real-L4 one-step sweep at 32K total-token envelope, including worst-case
   one 32K trajectory and mixed shorter trajectories. Require operational
   headroom; do not select a point that merely avoids immediate OOM.
6. Run at least 30 real-shard steps with checkpointing, FlashAttention 2, GDN
   fused kernels, causal conv, Liger, LoRA bf16, and regional compile together.
   Confirm finite loss, no cross-document drift, no compile storm, and stable
   memory after warm-up.
7. Compare padding-free against length-grouped padded execution on the same sample
   order and token envelope. Record valid and supervised tokens/s, peak VRAM, and
   wall time. Keep padding-free because it is requested only after correctness is
   established; the comparison quantifies its actual benefit.
8. If 32K exceeds the safe L4 envelope, measure 24,576, update the profile and
   regenerate shards/reports. Do not lower the cap from a synthetic estimate.
9. Run the full verification ladder. Update the old Phase 2/3 architecture notes,
   tests, optimization ledger, and artifact claims so no completed checkbox still
   describes segmented or padded training.

## Validation

- Trainer preflight accepts only the new full-trajectory schema and pinned lineage.
- Unequal accumulation micro-batches produce the same parameter update as one
  equivalent concatenated supervised-token loss, within numeric tolerance.
- L4 one-step and 30-step probes complete without OOM or NaN and report fresh
  peak memory/throughput evidence.
- Resume produces the same next batch uids and optimizer-step numbering as an
  uninterrupted run.
- Focused tests, full pytest, Ruff, mypy, and Makefile gates pass.
- Documentation numbers match generated reports and GPU result artifacts.

## Risks and rollback

- Dynamic token batches change optimizer-step statistics. Normalize by supervised
  tokens and compare an equivalent update before accepting learning-rate reuse.
- Regional compile may specialize on every flattened length. If compile churn is
  material, use bounded token-envelope buckets or disable only the measured
  compile region; do not reintroduce padding silently.
- 32K may fit one synthetic step but fail after allocator warm-up/checkpoint save.
  The 30-step gate and reserved-memory headroom exist to catch that distinction.
- Old Phase 3 speed numbers were measured on segmented shards. Mark them stale and
  never project epoch time from their 327M-token corpus after this migration.
