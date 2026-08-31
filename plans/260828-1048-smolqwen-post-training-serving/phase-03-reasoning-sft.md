---
phase: 3
title: "Reasoning SFT"
status: code-complete-gpu-pending
priority: P1
effort: "4d"
dependencies: [2]
---

# Phase 3: Reasoning SFT

## Overview

LoRA-finetune Qwen3.5-2B on the rendered trajectories so the model reasons before
each tool call, on both GPU profiles, with throughput and VRAM measured well
enough to fill in the profile sizing fields left as placeholders in Phase 1.

## Requirements

**Functional**
- TRL `SFTTrainer` with LoRA over the precomputed prompt/completion records,
  using the Phase 2 loss mask (`completion_only_loss` semantics).
- Both profiles runnable: `--profile l4` and `--profile a100`.
- Adapter pushed to private HF Hub on every `save_steps`; `--resume` continues
  from the newest revision after a VM reclaim.
- Adapter merge command producing a standalone checkpoint for evaluation and RL.
- Optimizations enabled from the start, each behind a config flag with a measured
  justification recorded next to it: bf16, gradient checkpointing, flash-attn on
  the full-attention layers, FLA/causal-conv1d for the GDN mixer, Liger fused
  linear cross-entropy, and `torch.compile` where the mixer permits.
- Periodic small-subset eval during training; full val loss once at the end.

**Non-functional**
- No OOM on L4 24 GB at the Phase 2 budget cap.
- Throughput reported as seconds per million tokens, comparable across profiles.
- Every enabled optimization has a recorded before/after measurement or is
  disabled.

## Architecture

Vocabulary is 248,320 with tied embeddings. At the sequence lengths in play, a
dense logits tensor over that vocab is the dominant activation — this is the first
thing that OOMs on a 24 GB card, not the weights. Liger's fused linear
cross-entropy avoids materializing full logits and is therefore load-bearing, not
a nice-to-have.

The mixer is the second constraint. GDN layers run Triton kernels from
`flash-linear-attention` that are `torch.compiler.disable`d upstream; compiling
through them raises inductor errors. So compile regionally: MLPs, RMSNorms,
`self_attn` on the 6 full-attention layers, the final norm, and the LoRA-adapted
linears — but leave the GDN mixer body eager. Fused bf16 rounding shifts per-batch
loss slightly versus eager; that is within run-to-run noise and is not a bug.

LoRA target is all-linear. Keep adapters in bf16 rather than PEFT's fp32 default:
with all-linear targets, fp32 adapters force an upcast/downcast plus an fp32 GEMM
at every linear. The fp32 default exists for low-bit QLoRA bases, which this is
not.

```
Qwen3.5-2B bf16
  + LoRA (all-linear, bf16 adapters)
  + gradient checkpointing
  + Liger fused linear CE          ← prevents the 248k-vocab logits OOM
  + flash-attn on 6 full-attn layers
  + FLA / causal-conv1d on 18 GDN layers
  + regional torch.compile (mixer body excluded)
```

Order of enablement matters for attribution. Turn each on separately, measure
s/Mtok and peak VRAM, record it, then move to the next. A config where every flag
is true and nobody knows which one helped is not an optimized pipeline.

## Related Code Files

- Create: `src/smolqwen/training/sft.py` — trainer assembly, LoRA config, callbacks, resume
- Create: `src/smolqwen/training/optim.py` — optimization toggles: Liger head, regional compile, bf16 adapter cast, checkpointing
- Create: `src/smolqwen/training/merge.py` — merge LoRA into base, save standalone checkpoint
- Create: `src/smolqwen/training/collate.py` — pad prompt/completion records, assemble the loss mask into labels
- Create: `tests/test_collate_mask.py` — padding preserves mask alignment; label `-100` exactly where masked
- Create: `tests/test_sft_smoke.py` — 2 steps on a tiny random-weight config, CPU, asserts loss is finite and decreasing-capable
- Create: `notebooks/01-sft.ipynb` — thin wrapper: setup, `smolqwen train-sft`, tail W&B
- Modify: `configs/base/sft.yaml` — model, LoRA, optimizer, schedule, eval cadence, optimization flags
- Modify: `configs/profiles/{l4,a100}.yaml` — this phase's owned fields: measured `micro_batch`, `grad_accum`, `max_seq_length`
- Modify: `src/smolqwen/cli.py` — wire `train-sft`, `merge-adapter`

## Implementation Steps

1. `collate.py`: pad the Phase 2 records to a batch, build `labels` from the loss
   mask with `-100` at masked positions. Test alignment under padding before
   anything trains — a mask off by one token trains the model on tool output and
   nothing will report it.
2. `sft.py`: assemble `SFTTrainer` with the LoRA config from YAML, the W&B
   callback from Phase 1, and the `CheckpointStore` push callback. Wire
   `--resume` to `latest_revision` + `pull`.
3. `optim.py`: one function per toggle, each returning a description string that
   gets logged to W&B config. Liger fused linear CE; bf16 LoRA adapter cast;
   `gradient_checkpointing`; regional `torch.compile` with an explicit exclude
   list for the mixer body; attention implementation selection.
4. Baseline measurement: all toggles off except bf16 and gradient checkpointing
   (without checkpointing, activations at the Phase 2 cap will not fit on L4 even
   at batch 1). Record s/Mtok and peak VRAM on L4.
5. Enable Liger fused CE. Measure. This is expected to be the difference between
   fitting and OOM at longer sequences — confirm that and record the numbers.
6. Enable bf16 LoRA adapters. Measure.
7. Enable regional `torch.compile` with the mixer excluded. Measure. If inductor
   errors, narrow the compiled region rather than abandoning compile; record what
   had to be excluded and why.
8. Sweep micro-batch and `max_seq_length` upward on each profile until just below
   OOM; write the resulting values into this phase's owned profile fields
   (`micro_batch`, `grad_accum`, `max_seq_length`) with the measured peak VRAM as a
   comment. `max_seq_length` is seeded from `budgets.json` and may not exceed that
   cap — the budget bounds the distribution, the sweep only finds what fits under it.
9. Short training run (a few hundred steps) on each profile to confirm loss
   descends and no NaN appears with compile + fused CE + bf16 adapters together.
10. Full SFT run on the profile chosen from this phase's own throughput/VRAM sweep
    (s/Mtok and peak VRAM per profile) — the Phase 1 probe reports capability only
    and has no throughput number. Push adapters throughout. Merge at the end into
    `artifacts/models/qwen3.5-2b-sft-merged` and record the resolved revision sha for
    Phase 5 to pin.
11. Record the optimization ledger in `docs/` — one row per toggle with s/Mtok,
    peak VRAM, and the decision. This is the artifact that shows the
    optimizations were measured, not copied.

## Success Criteria

Checked items were verified on CPU in this environment. The unchecked ones all
require a card that can hold the 2B model; this host has 3.7 GB of VRAM against
~3.8 GB of bf16 weights, so they are blocked on hardware, not on code.

- [ ] `smolqwen train-sft --profile l4` completes without OOM at the Phase 2 budget cap.
- [ ] `smolqwen train-sft --profile a100` completes; both profiles' sizing fields are filled with measured values, not guesses, and only this phase's owned fields are written.
- [x] `max_seq_length` in both profiles is at or below the `budgets.json` cap. Stronger than asked: the profiles no longer carry the field at all, so `budgets.json` (16,384) is what resolves, and `test_committed_profiles_do_not_override_budget_seeded_fields` fails if a placeholder is reintroduced.
- [ ] The merged checkpoint's revision sha is recorded for Phase 5 to pin. (`merge_report.json` records `base_revision` on every merge; there is no trained checkpoint yet.)
- [ ] Optimization ledger recorded: every enabled flag has a before/after s/Mtok and peak VRAM; anything without a measured win is off. (`docs/optimization-ledger.md` exists with the decisions and the fallback order; the measurement columns are empty and marked pending.)
- [x] Collate mask test passes: `labels == -100` at exactly the masked positions, under padding.
- [x] CPU smoke test runs 2 steps on a tiny config with finite loss — and additionally recomputes the loss by hand from the mask, so a path that trained on the full sequence would fail.
- [ ] Interrupting a run and re-launching with `--resume` continues from the pushed revision, and the W&B run continues rather than forking. (Assembly is verified: `test_resume_continues_the_same_wandb_run` asserts the persisted run id is what `Tracker` resumes with, and a resume with nothing pushed raises rather than silently starting fresh. The end-to-end interrupt needs a GPU run.)
- [ ] `artifacts/models/qwen3.5-2b-sft-merged` loads standalone in `transformers` and generates a syntactically valid tool call for a held-out prompt. (The merge path is verified standalone on a tiny checkpoint: no adapter files remain, weights actually changed, the tokenizer is saved alongside. Generation needs a trained model.)
- [ ] Val loss recorded at the end of training; training curve in W&B shows descent without NaN.

## Verified in this environment

| What | Evidence |
|---|---|
| Budget seeding reaches the run | `smolqwen train-sft --profile l4 --dry-run` reports `max_seq_length: 16384`, `max_new_tokens_per_step: 5694` — the measured values, not the old 8192/1024 placeholders. |
| TRL never re-derives the mask | `dataset_kwargs={"skip_prepare_dataset": True}`, `max_length=None`, dataset columns are exactly `prompt_ids/completion_ids/loss_mask`, and the collator is the Phase 3 one. |
| Only adapters train, in bf16 | Every `requires_grad` parameter has `lora_` in its name and dtype `bfloat16`. |
| The mixer is never compiled | `select_compile_targets` excludes `Qwen3_5GatedDeltaNet` by class name even with an empty pattern list, and never returns the root module. |
| Real shards satisfy the mask contract | `validate_shard` over the actual conversion output: train 39,957 samples / 327,136,547 tokens / 76,033,852 supervised; val 885 / 7,350,599 / 1,633,346. Zero rejections. |
| Shard loading does not scale in RAM | Validation streams line by line; the trainer reads through a memory-mapped Arrow cache, because `train.jsonl` is ~2 GB of JSON. |
| L4 batch ceiling at the budget cap | One-step probe on NVIDIA L4 (22.034 GiB): SFT micro-batch 1/2/4 passed at 16,384 tokens; batch 8 OOM. The measured ceiling is 4, while the committed profile remains at 1 for headroom. See [`qa-260830-l4-batch-limits.md`](../reports/qa-260830-l4-batch-limits.md). |


## Risk Assessment

**248k-vocab logits OOM.** Signal: OOM inside the loss computation at batch 1
with checkpointing on. Response: Liger fused linear CE is the fix, and it is
enabled by design — if it is somehow unavailable for this architecture, fall back
to shortening `max_seq_length` before touching batch size, since the logits
tensor scales with sequence length.

**`torch.compile` breaks on the GDN mixer.** Signal: `InductorError` mentioning a
`torch.compiler.disable`d region. Response: narrow the compiled region;
compiling the MLP alone still helps. Record the exclusion list. Never disable the
upstream `torch.compiler.disable` — it is there because those kernels do not
compile.

**Overfitting on ~9k trajectories.** A 2B model on a few tens of thousands of
samples can memorize. Signal: val loss turning up while train loss falls.
Response: 1-2 epochs maximum, val-loss early stop, and report the epoch count in
the results table. More epochs is not the fix.

**Adapter merge changes behavior.** Signal: the merged checkpoint scores
differently from the adapter-loaded model on the same eval subset. Response:
evaluate both once in Phase 5 and use whichever path the RL stage will actually
consume, so no silent mismatch enters the comparison.

**Silent train/inference format drift.** If the collator renders differently from
the serving path, SFT teaches a format the model never sees again. Signal: the
merged model emits malformed tool calls despite low training loss. Response: the
Phase 2 round-trip test plus a single generation check on a held-out prompt at the
end of this phase — catch it here, not in Phase 5.
