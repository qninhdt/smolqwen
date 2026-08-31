---
title: "Full-trajectory reasoning SFT with padding-free batches"
status: planned
date: 2026-08-31
owners: [phase-2-data, phase-3-sft]
supersedes:
  - "Phase 2 per-user-turn segmentation decision"
  - "Phase 3 padded fixed-row batching contract"
---

# Full-trajectory reasoning SFT with padding-free batches

## Outcome

Rebuild the SFT path so one released teacher trajectory is one training sample,
all historical assistant reasoning is preserved, and loss covers every assistant
reasoning/tool-call/text token while system, user, and tool-observation tokens are
masked. Train those samples in document-isolated padding-free micro-batches whose
row count varies under a fixed total-token envelope.

## Evidence and corrected decisions

- The raw release has 9,022 trajectory rows and 4,684 task ids. A task id is a
  split group, not a unique trajectory id: 4,338 task ids each have one
  `conversation` and one `non_conversation` trajectory.
- Every conversation row ends in the fixed user sentinel `###STOP###`; every
  non-conversation row ends in `Task finished`. Neither has a later assistant
  target. Training trims generically after the last assistant rather than
  hard-coding either literal.
- The Qwen3.5 inference template strips reasoning before its last real user query.
  That inference-context policy must not define the SFT sample boundary. Training
  uses a separate, pinned template mode that preserves `reasoning_content` for
  every assistant message.
- The current converter expands 7,554 accepted trajectory rows into 40,842
  segmented records. The corrected invariant is one accepted row to one record.
- A read-only full-corpus measurement with all reasoning retained and the
  sentinel trimmed produced: p50 16,039; p90 24,139; p95 27,130; p99 34,760;
  max 81,897 tokens. Retention is 52.68% at 16,384, 90.96% at 24,576, and
  98.46% at 32,768.
- Native TRL 1.12 padding-free flattens rows and resets `position_ids`, but the
  Qwen3.5 Gated DeltaNet path does not consume `position_ids`; recurrent and
  causal-convolution state boundaries require separate proof. Padding-free is a
  correctness-gated feature, not a boolean enabled on trust.

## Constraints

- Keep inference rendering and rollout message formatting unchanged.
- Preserve `role: "tool"` as the shared SFT/inference observation shape.
- Never truncate or split an accepted trajectory. Over-cap trajectories are
  skipped whole and counted.
- Split train/validation by task id so the conversation and non-conversation
  variants of one task cannot leak across partitions.
- Keep every raw row independently identifiable with a deterministic unique
  trajectory uid.
- Single-GPU L4 and A100 are in scope. Distributed/context-parallel training is
  not part of this correction.
- Padding-free must isolate full-attention, GDN recurrent state, and causal-conv
  state at every trajectory boundary. If equivalence cannot be proven on the
  target kernels, the feature fails closed.
- Preserve LoRA rank 32, bf16 adapters, gradient checkpointing, FlashAttention 2,
  Liger fused linear CE, checkpoint/resume, and regional compile unless a measured
  incompatibility forces a separately recorded decision.

## Non-goals

- Changing rollout, GRPO, serving, tool syntax, or environment behavior.
- Training on terminal sentinels or tool observations.
- Packing by cutting a trajectory across blocks.
- Detached/no-grad prefixes or any objective that avoids backpropagating through
  masked context tokens.
- Treating the model's 262,144-position limit as the optimizer token budget.

## Target operating envelope

The first target is:

```text
max trajectory length       32,768 tokens
max total tokens/microbatch 32,768 tokens
trajectories/microbatch     variable (normally 1-6)
loss normalization          actual supervised assistant-token count
```

This retains 8,883 of 9,022 rows before the seeded train/validation split. The
32K token envelope is a starting point backed by the corrected distribution, not
a claim that all 32K shapes fit L4. Phase 4 measures the one-step and short-run
boundary. If 32K fails, 24,576 is the recorded fallback cap; lowering it requires
the measured OOM evidence and reports the resulting retention loss.

## Phases

| Phase | Status | Dependency | Deliverable |
|---|---|---|---|
| [1. Full-trajectory render contract](./phase-01-full-trajectory-render-contract.md) | planned | none | One row/one sample, preserved reasoning, exact labels |
| [2. Correct profiling and conversion](./phase-02-profile-and-convert.md) | planned | 1 | Correct distributions, unique ids, regenerated shards |
| [3. Padding-free correctness and token batching](./phase-03-padding-free-token-batching.md) | planned | 1, 2 | Boundary-safe flattening and variable-row batches |
| [4. Trainer integration and L4 validation](./phase-04-trainer-and-gpu-validation.md) | planned | 3 | Token-normalized SFT, GPU evidence, updated authority docs |

## Acceptance criteria

- Each accepted raw trajectory row emits exactly one persisted sample; no
  `segment_index` or historical-prefix duplication remains.
- Every sample ends at its last assistant message. No `Task finished` or
  `###STOP###` token appears in persisted training ids.
- The count of preserved `<think>` blocks equals the count of assistant messages
  in the trimmed trajectory, including messages before later real user turns.
- Labels supervise every rendered assistant token, including reasoning, tool
  calls, and final text; all system/user/tool-observation tokens are `-100`.
- Persisted `trajectory_uid` values are unique across all 9,022 rows, while
  `task_id` remains the deterministic split key.
- The corrected profiler reproduces the measured full-trajectory percentile and
  retention distribution within deterministic tokenizer equality, not a loose
  estimate.
- At the 32K cap, conversion accounts for 9,022 rows and emits 8,883 samples with
  139 `too_long` skips, unless a tokenizer/template pin changes and is explicitly
  recorded.
- Every token-budget batch contains complete trajectories, contains at most
  32,768 total tokens, and visits every retained row exactly once per epoch.
- Padded and flattened execution match per-token logits, supervised loss, and
  LoRA gradients within the selected numeric tolerance for mixed-length rows.
  Changing one row must not change another row's outputs.
- The equivalence test runs on a real L4 with FlashAttention, flash-linear-
  attention, and causal-conv enabled; CPU-only success is insufficient.
- Optimizer loss is normalized by the actual supervised-token count across an
  accumulation window. Unequal micro-batches receive per-token, not per-batch,
  weight.
- A one-step and at least 30-step L4 run complete without OOM/NaN, report actual
  tokens/s and peak VRAM, and retain operational headroom rather than using the
  allocator ceiling.
- The trainer rejects old segmented shards by schema/semantic version instead of
  silently consuming the existing 40,842-record artifacts.

## Validation ladder

1. Pure rendering and schema unit tests with the real pinned Qwen3.5 template.
2. Full 9,022-row profile/conversion accounting and artifact checksums.
3. CPU collator/sampler invariants and deterministic resume-order tests.
4. Real-L4 padded-versus-padding-free boundary equivalence.
5. Real-L4 32K one-step memory probe, then 30-step real-shard run.
6. Full focused suite, full pytest, Ruff, mypy, and project Makefile gates.

## Rollback boundary

Full-trajectory rendering and corrected labels remain valid independently of
padding-free. If target-kernel boundary isolation fails, do not ship a flattened
trainer that can cross-contaminate trajectories. The safe diagnostic fallback is
length-grouped padded batching, but this plan remains incomplete until the
padding-free correctness gate is solved or the user explicitly changes scope.
