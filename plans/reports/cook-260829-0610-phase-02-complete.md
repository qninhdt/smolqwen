---
title: "cook — Phase 2 completion run"
type: session-cook
date: 2026-08-29
scope: "Phase 2: data pipeline + trajectory profiler (complete) + real profile/conversion run"
branch: feat/smolqwen-pipeline
---

# Cook — Phase 2 completion

Cooking `plans/260828-1048-smolqwen-post-training-serving`, Phase 2, to
completion. Phase 1 is done and green (40 tests pass); Phase 2 is partially
built (`data/loader.py`, `data/profiler.py`, `data/render.py`, `configs/base/data.yaml`).
This session finishes Phase 2 and runs the real profile/conversion pass.

## Scope

**Implement**
- `src/smolqwen/data/convert_sft.py` — trajectory → sample, Non-Conv (1/sample)
  / Conv (n real-user-turn samples), budget-cap filtering with skip reasons,
  streaming so nothing materializes all 9k rendered trajectories.
- `src/smolqwen/data/splits.py` — seeded train/val split **by trajectory id**;
  env-split manifest (140 `_sft` / 51 `_rl` from `env_id` suffixes in
  `191_env_metadata.json`, asserting every RL scenario's env_id is RL).
- `src/smolqwen/data/cli_actions.py` — `run_profile_data` / `run_prepare_sft`
  wiring for the CLI (currently imported but missing).
- `tests/fixtures/trajectories.json` — real excerpts (1 Non-Conv, 1 Conv,
  1 malformed) from the cached release.
- `tests/test_template_reasoning_retention.py`, `tests/test_tool_call_roundtrip.py`,
  `tests/test_loss_mask.py`, `tests/test_conv_split.py`.

**Repair / gate hygiene**
- `src/smolqwen/cli.py`: `env-selftest` currently imports the not-yet-existing
  `smolqwen.env.selftest` (Phase 4). Convert to `_not_implemented(4, ...)`.
- `ruff format` the 3 existing data modules; clear the unused `type: ignore` in
  `tests/test_tokenizer_not_processor.py`.

**Run (real data)**
- `smolqwen profile-data` over the cached 701 MB SFT file → `profile.json` +
  `budgets.json` + `env_split.json`.
- `smolqwen prepare-sft` → train/val jsonl + `conversion_report.json`.
- Commit the resulting budget decisions / env split as artifacts.

## Out of scope (intentional)

- Phase 3–8 modules and training.
- Phase-1 leftover infra (`.github/workflows/ci.yml`, `scripts/setup_colab.sh`,
  `notebooks/00-probe-gpu.ipynb`, `.env.example`, `.gitignore`) — a separate
  infra session.
- A dedicated `schema.py` pydantic module: the data model is already expressed
  as frozen dataclasses with validation in `data/loader.py` (`Trajectory`,
  `Message`, `ToolCall`) and `data/render.py` (`RenderedSample`). A parallel
  pydantic schema would duplicate them (DRY). The plan's "reject malformed rows
  loudly and count them" requirement is satisfied by `parse_trajectory` +
  `LoadStats` + `DataError`.

## Acceptance

- [ ] `make check` (ruff + mypy --strict) and `make test` green, no regressions.
- [ ] `smolqwen profile-data` writes profile.json / budgets.json / env_split.json.
- [ ] `smolqwen prepare-sft` writes train/val jsonl + conversion_report.json
      accounting for every input row (converted / skipped-too-long / malformed).
- [ ] Reasoning-retention test passes against real excerpts via the cached
      Qwen3.5 template (template keeps reasoning across a tool chain, strips
      across a real user turn).
- [ ] Tool-call parse/serialize round-trips the `<tool_call><function=...>` XML.
- [ ] Loss-mask and conv-split tests pass.
- [ ] Train/val split reproducible from seed; env_split is 140/51 and every RL
      scenario env_id lands in the RL set.
