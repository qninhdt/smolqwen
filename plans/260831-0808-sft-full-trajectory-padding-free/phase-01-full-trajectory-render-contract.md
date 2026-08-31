---
phase: 1
title: "Full-trajectory render contract"
status: planned
dependencies: []
---

# Phase 1: Full-trajectory render contract

## Goal

Replace the inference-derived per-user segmentation contract with a dedicated
training renderer that preserves every assistant reasoning block and produces one
token/label sequence per raw trajectory row.

## Files

- Modify `src/smolqwen/data/loader.py`
- Modify `src/smolqwen/data/render.py`
- Modify `src/smolqwen/data/convert_sft.py`
- Modify `src/smolqwen/data/splits.py`
- Modify `tests/helpers.py`
- Replace segmentation expectations in `tests/test_conv_split.py`
- Modify `tests/test_convert_sft.py`
- Modify `tests/test_loss_mask.py`
- Modify `tests/test_template_reasoning_retention.py`
- Modify `tests/test_collate_mask.py`

## Steps

1. Add a deterministic `trajectory_uid` per raw row. Keep `task_id` separately
   as the train/validation grouping key. Validate that all emitted uids are unique
   and that duplicate task ids stay in one partition.
2. Add a generic trajectory bound helper: require at least one real user and one
   later assistant, then retain messages only through the last assistant. Record
   how many trailing messages were removed and why.
3. Separate inference and training rendering APIs. Leave `render_prefix` and the
   rollout call path untouched. Add a training-template adapter based on the
   pinned tokenizer template that changes only the historical-reasoning retention
   condition. Assert the expected source clause appears exactly once and record a
   template fingerprint; fail on upstream drift.
4. Render the bounded conversation with fixed tools and `add_generation_prompt`
   disabled. Build exact token spans incrementally or from verified offsets so
   the concatenated ids equal a one-shot full render. Fail if a prefix rewrite or
   token seam makes ownership ambiguous.
5. Persist `input_ids` and `labels` directly. Assign the full assistant block —
   `<think>`, reasoning content, tool-call syntax, content, and assistant end
   marker — to loss. Assign system, tools declaration, user, and tool-observation
   blocks to `-100`.
6. Remove `Segment`, `split_segments`, segment indexes, prompt/completion schema,
   and any converter loop that emits more than one record for a row. Preserve a
   compatibility error with a clear regeneration instruction rather than reading
   old shards.
7. Add regression fixtures covering a conversation with multiple real users,
   multiple tool calls per user, empty assistant reasoning, consecutive tool
   observations, and a trailing sentinel.

## Validation

- A multi-user fixture emits one sample and preserves every distinct reasoning
  marker, including markers before the second user.
- One-shot tokenization equals the concatenation used to construct labels.
- Every assistant span is supervised and every non-assistant span is masked.
- Removing/changing a terminal sentinel does not change the rendered sample.
- Inference `render_prefix(..., add_generation_prompt=True)` remains byte-identical
  on existing rollout fixtures.

## Risks and rollback

- Template drift can silently mislabel tokens. The adapter fails closed on the
  pinned clause/fingerprint and the tests use the real tokenizer template.
- Tokenizing message deltas independently can change BPE seams. Prefer full-render
  ids plus verified boundaries; never accept a merely equal decoded string.
- Do not restore per-user segmentation as a workaround. If exact span ownership
  cannot be established, stop at the renderer boundary and revise the span method.
