---
phase: 2
title: "Correct profiling and conversion"
status: in_progress
dependencies: [1]
---

# Phase 2: Correct profiling and conversion

## Goal

Make the profiler, cap recommendation, split, shard schema, and accounting use
the complete preserved-thinking trajectory as their only sample unit.

## Files

- Modify `src/smolqwen/data/profiler.py`
- Modify `src/smolqwen/data/convert_sft.py`
- Modify `src/smolqwen/data/cli_actions.py`
- Modify `src/smolqwen/data/splits.py`
- Modify `src/smolqwen/config_models.py`
- Modify `configs/base/data.yaml`
- Regenerate `artifacts/data/profile.json`
- Regenerate `artifacts/data/budgets.json`
- Regenerate `artifacts/data/conversion_report.json`
- Regenerate ignored `artifacts/data/sft/{train,val}.jsonl`
- Modify focused profiler/converter/split tests

## Steps

1. Replace per-segment `sample_tokens` and longest-segment retention with one
   complete training-render length per row. Keep raw reasoning, observation, user,
   and assistant-turn distributions as diagnostics.
2. Report row trajectories and unique task groups separately. Never label unique
   task ids as unique trajectories.
3. Add candidate caps 16,384, 24,576, 32,768, 49,152, and 65,536. Select 32,768
   as the initial SFT cap because it retains 98.46% under the pinned measurement;
   include counts and mode-specific retention next to every candidate.
4. Split by `task_id`, then route each unique `trajectory_uid` row. Assert paired
   conversation/non-conversation variants never straddle train/validation.
5. Version the persisted schema and semantic contract. Store `input_ids`,
   `labels`, `seq_length`, `supervised_tokens`, `trajectory_uid`, `task_id`, mode,
   env id, tokenizer revision, template fingerprint, and source checksums.
6. Skip over-cap rows whole. Do not truncate, wrap, or split. Account for every
   raw row as converted, too long, or malformed.
7. Regenerate profile, budgets, report, and train/validation shards from the raw
   pinned release. Archive no duplicate 2 GB shard in git; generated outputs must
   be reproducible from the report and checksums.
8. Add a trainer preflight that rejects the old prompt/completion/segment schema
   and reports the exact `prepare-sft` command required to regenerate.

## Validation

- Profile reads 9,022 rows, 0 malformed, and reports 4,684 unique task groups.
- Corrected full-render statistics reproduce p50 16,039, p90 24,139, p95 27,130,
  p99 34,760, max 81,897 with the pinned tokenizer/template.
- At 32,768, 8,883 rows convert and 139 are skipped as too long.
- `samples == converted_rows`; mode and split subtotals sum exactly.
- No sentinel text appears in decoded persisted rows.
- No train task id appears in validation.

## Risks and rollback

- A changed tokenizer/template revision legitimately changes every count. Treat
  that as a new data lineage, not a tolerance adjustment.
- The 32K cap may later fail the L4 memory gate. Only Phase 4 may lower it, with
  measured evidence and regenerated retention/accounting.
- Existing artifacts encode the wrong objective. The trainer must reject them;
  silently accepting both schemas is not a compatibility feature.
