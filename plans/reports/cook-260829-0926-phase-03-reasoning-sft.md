# Cook report — Phase 3 Reasoning SFT

Date: 2026-08-29 · Branch: `feat/smolqwen-pipeline` · Plan: `plans/260828-1048-smolqwen-post-training-serving/phase-03-reasoning-sft.md`

## Outcome

Phase 3 is code-complete and CPU-verified. All gates green: `ruff check`,
`ruff format --check`, `mypy --strict` (37 files), `pytest` (108 tests),
`make smoke`. The GPU sweep and the training run are blocked on hardware, not on
code — see "Blocked" below.

## What landed

| File | Role |
|---|---|
| `src/smolqwen/training/collate.py` | Padding + `IGNORE_INDEX` labels from the Phase 2 `loss_mask` (already existed at session start) |
| `src/smolqwen/training/optim.py` | One function per toggle, each returning a `Toggle` with the reason; compile-target selection that excludes the GDN mixer |
| `src/smolqwen/training/sft.py` | Trainer assembly, streaming shard validation, Arrow-backed loading, throughput + checkpoint-push callbacks, resume |
| `src/smolqwen/training/merge.py` | LoRA merge into a standalone checkpoint, with `merge_report.json` recording the base revision |
| `src/smolqwen/cli.py` | `train-sft` and `merge-adapter` wired to real handlers |
| `configs/profiles/{l4,a100}.yaml` | Budget-seeded fields removed (see the defect below) |
| `docs/optimization-ledger.md` | Toggle decisions, empty measurement columns marked pending, OOM fallback order |
| `notebooks/01-sft.ipynb` | Thin Colab wrapper; tokens prompted, never committed |
| `.gitignore` | `artifacts/` ignored except the three measurement JSONs later stages read |
| `tests/test_optim_toggles.py` (12), `tests/test_sft_assembly.py` (9), `tests/test_sft_smoke.py` (7), `tests/test_merge_adapter.py` (3) | New coverage |

## The defect fixed first

`smolqwen train-sft --profile l4 --dry-run` reported `max_seq_length: 8192` while
`budgets.json` said 16384. Cause: the profile YAMLs carried placeholder values for
the three budget-seeded fields, and the profile layer merges *after* `budgets.json`,
so a guess silently won over a measurement. `_profile_cap_violations` only catches
a profile going *above* the cap, so this direction was invisible.

Fix: the profiles no longer carry `max_seq_length`, `max_new_tokens_per_step` or
`max_env_steps` at all, with a comment saying why, and
`test_committed_profiles_do_not_override_budget_seeded_fields` fails if one is
reintroduced. Verified: the dry-run now reports 16384 / 5694.

## Real data

`smolqwen prepare-sft` ran against the full release at the corrected 16,384 cap
(~28 min):

- 9,022 rows read, 0 malformed, `accounted: true`
- 7,554 trajectories converted → 40,842 samples; 1,468 skipped, all `too_long`
- by mode: conversation 4,301 trajectories / 37,589 samples; non-conversation 3,253 / 3,253
- shards: `train.jsonl` 39,957 samples (2.0 GB), `val.jsonl` 885 samples

Every record in both shards passes `record_to_sequence` — the same function the
collator calls. Train: 327,136,547 tokens, 76,033,852 supervised. Val: 7,350,599 /
1,633,346.

That 2 GB shard forced a change: the first draft of `load_shards` did
`list(iter_records(...))` and `Dataset.from_dict`, which would need the whole file
resident as Python lists on a machine with 15 GB. Validation now streams, and the
trainer reads a memory-mapped Arrow cache.

## Blocked, and why

The local GPU is an RTX 3050 Laptop: 3.7 GB VRAM, compute 8.6. The bf16 weights
alone are ~3.8 GB, so the model does not load. The `colab` extra
(`liger-kernel`, `flash-attn`, `flash-linear-attention`, `causal-conv1d`) is also
not installed here — which is why the toggle resolvers downgrade with a recorded
reason rather than raising, and why the tests assert the downgrade path as well as
the enabled one. On this host the ledger prints liger off ("not importable") and
attention downgraded to `sdpa` ("flash_attn wheel is not installed").

Steps 4–11 of the phase (baseline measurement, per-toggle measurement, the OOM
sweep, the short run, the full run, filling the ledger table) need an L4 or A100.
`docs/optimization-ledger.md` records the exact commands and which W&B keys to
read.

## Deliberate deviations from the phase file

- **`warmup_ratio` → `warmup_steps`.** transformers 5.x dropped `warmup_ratio`;
  `warmup_steps` accepts a float in [0, 1) as a ratio of total steps, which is the
  same quantity. `TrainingConfig.warmup_ratio` keeps its name in our schema.
- **W&B through the Phase 1 `Tracker`, not `report_to="wandb"`.** The trainer's own
  integration starts a fresh run, which forks a resumed run into a second curve.
- **`max_length=None` in `SFTConfig`.** Truncation is the collator's job against
  the profile cap, and TRL's truncation path is unreachable once dataset
  preparation is skipped. Leaving the 1024 default would be a false claim about
  sample length.
- **A `tests/helpers.py` tiny-checkpoint factory.** Three test files needed a local
  Qwen3.5 checkpoint plus tokenizer; `write_tiny_checkpoint` / `tiny_qwen35_model`
  is one definition instead of three. `layer_types` is spelled out rather than
  derived from `full_attention_interval`, which the config accepts only via
  `**kwargs` and therefore fails `mypy --strict` at the call site.

## Test additions worth naming

- `test_loss_is_computed_over_exactly_the_supervised_positions` recomputes the
  cross-entropy by hand from the mask and requires the model's own loss to match.
  A path that trained on the full sequence would pass every other test in the
  file, because the loss would still be finite and still descend.
- `test_the_forbidden_class_is_excluded_even_with_an_empty_pattern_list` — the
  exclusion pattern list is configurable, so an empty list must not become
  permission to compile the GDN mixer.
- `test_resume_without_anything_pushed_fails_loudly` — a resume that silently
  starts from scratch is worse than one that refuses.

## Open questions

1. `scripts/setup_colab.sh` is referenced by `notebooks/01-sft.ipynb` but is a
   Phase 1 deliverable and does not exist yet. The notebook's first cell will fail
   until it lands.
2. Phase 1 leftovers still outstanding: `.env.example`,
   `.github/workflows/ci.yml`, `notebooks/00-probe-gpu.ipynb`.
3. 1,468 trajectories (16.3%) are dropped as `too_long` at the 16,384 cap. That is
   consistent with the profiler's 83.7% trajectory retention and above the 0.70
   floor, so it is the measured decision rather than a surprise — but it is a
   stated slice of the distribution, worth repeating in the results table.
