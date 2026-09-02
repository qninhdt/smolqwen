# Deletion inventory

Produced by `scripts/audit-dead-symbols.py`, then re-checked by hand across the
eight consumer surfaces: `src`, `tests`, `configs`, `docs`, `notebooks`,
`scripts`, `Makefile` + CI, and `docker-compose.yml`. `plans/` is excluded from
the search on purpose — a plan naming a symbol it intends to delete would keep
that symbol alive forever.

Audit totals: **1,296 declarations — 1,235 live, 10 unreferenced, 51
framework-invoked.**

## Why kind-aware

A reference count answers the question for a function and gets it wrong for
everything else. Three kinds fail in different directions:

| Kind | Failure mode | Case |
|---|---|---|
| pydantic field | consumers are YAML keys and a `@property` body a definition-only walk never enters | `active_pool_multiplier` |
| framework hook | nothing in this repo calls it, because transformers does | `on_save` |
| type alias / sentinel | read as an annotation or compared against, not called | `Stage`, `MASKED` |

## The three config fields

| Field | Verdict | Consumers |
|---|---|---|
| `ProfileConfig.active_pool_multiplier` | **live** | `config_models.py:75` (`@property generation_batch_size`), `configs/profiles/l4.yaml:25`, `configs/profiles/a100.yaml:23`, `tests/test_rollout_metrics.py:22` |
| `TrackingConfig.local_artifact_dir` | **live** | `tests/test_sft_assembly.py:79` |
| `DataConfig.max_trajectories` | **dead** | none on any surface |

`active_pool_multiplier` is the field that motivated the rewrite. Its only `src`
consumer is the property two lines below its declaration; both profiles set it to
2, so deleting it drops the GRPO prompt pool from 16 to 8 **without raising
anything**, and then `extra="forbid"` rejects both profile files at load. The
audit now reports it live, from all four sites.

`generation_batch_size` — the property that reads it — is itself live from
`grpo.py:121,252,260,463,521` and `scripts/colab-l4-batch-sweep.py:476,562,623`.

## Framework-invoked, never proposed

16 distinct names across 51 declarations. `on_save` appears twice
(`GrpoCheckpointCallback` at `grpo.py:162`, `CheckpointPushCallback` at
`sft.py:486`) — the previous audit named four hooks and missed both, so it would
have proposed a live callback for deletion.

Allowlisted: `on_log`, `on_save`, `on_step_begin`, `on_step_end`,
`_get_train_sampler`, `get_eval_dataloader`, `training_step`, `compute_loss`,
`model_config`, `ADAPTER_NAME`, `create_adapter`, and the dunders.

The last two are this repo's own protocol: `eval/adapters/__init__.py` resolves
adapter modules through `pkgutil.iter_modules`, so neither name is ever
referenced by a caller.

## The seven symbol candidates, re-checked

All seven confirmed unreferenced. Each was additionally grepped without word
boundaries across all eight surfaces.

| Symbol | Kind | Site | Note |
|---|---|---|---|
| `_not_implemented` | function | `cli.py:238` | module-private; every stage now has a real handler |
| `Trajectory.is_conversation` | property | `data/loader.py:220` | callers read `traj_type` directly |
| `split_trajectories` | function | `data/splits.py:48` | a two-line pass-through to `split_trajectory_ids`, which is live |
| `ScenarioSet.sample_ids` | method | `env/scenarios.py:143` | `grpo.py:644` samples via `profile_scenario_sample` against the dataset, not through this |
| `probe_payload` | function | `probe.py:149` | `asdict(probe())` — callers use `probe()` |
| `SamplingParams` | class | `rollout/generation.py:70` | the real params come from TRL's own object; nothing constructs this |
| `safe_command_text` | function | `serving/server.py:94` | duplicates `shlex.join`; `run_server` calls that directly |

Two more the audit surfaced that the previous inventory never listed:

| Symbol | Kind | Site | Note |
|---|---|---|---|
| `Stage` | type alias | `config_models.py:21` | `STAGES` (the tuple) is live; this `Literal` alias is annotated nowhere |
| `MASKED` | constant | `data/render.py:19` | `SUPERVISED` and `IGNORE_INDEX` beside it are live; `MASKED = 0` is compared against nowhere |

Both are Group A candidates for Phase 8, with the same one-commit-per-group rule.

## Phase 8 targets and their live consumers

### Group C — scripts

| Script | Lines | Consumers | Disposition |
|---|---|---|---|
| `colab-probe-status.py` | 27 | none | delete |
| `colab-l4-batch-sweep.py` | 783 | `docs/optimization-ledger.md:44` | **gated** — the instrument for plan `260831-0808` phase 4's envelope measurement |
| `colab-l4-sft-speed.py` | 710 | none directly | **gated** — a Modify target of that plan's phase 4 |
| `build-sft-benchmark-shard.py` | 82 | produces the `benchmark_id` field `colab-l4-sft-speed.py:283` reads | **gated** — same instrument chain |
| `colab-l4-smoke.py` | — | `docs/serving.md:131` | **keep and edit** — `:290` invokes `smolqwen bench`, a subcommand Group D deletes |

The gate condition: plan `260831-0808` phase 4 is `status: in_progress`. It stays
open until that plan records the 32K L4 token envelope.

### Group D — serving wrappers

| Target | Consumers | Disposition |
|---|---|---|
| `serving/bench.py` (345) | `cli.py:378`, `docs/serving.md:58,67,90`, `docker-compose.yml:60`, `scripts/colab-l4-smoke.py:290` | delete after moving three things out |
| `serving/sweep.py` (209) | `cli.py:389`, `docs/serving.md:106,109` | delete |
| `wait_for_readiness` | `bench.py:167,299`, `tests/test_serving_commands.py:15,98` | **moves to `inference/client.py`** in Phase 2, with its test |
| `assert_quality_matches_serving` | `serving/report.py:43,91`, `bench.py:320,323`, `tests/test_bench_result_parsing.py:18,138,147` | **moves to `eval/`** — the only `recorded_free` pairing guard |
| `serving/workload.py` | `bench.py:267,270,277,280`, `tests/test_serving_commands.py:19,163,173`, `docs/serving.md:90,98` | **moves to `eval/workload.py`**, needs a CLI entry point or it survives with no caller |

`BenchResult` (`bench.py:27`) is referenced from `serving/report.py:13` under
`TYPE_CHECKING`, so the pairing function's signature changes when it moves.

### Compose, the surface the previous inventory omitted

`docker-compose.yml:57-71` defines a `bench` service whose `command` is
`["bench", "--profile", ..., "--dataset", "random"]` — the deleted subcommand.
`tests/test_auth_all_paths.py:39-41` reads that service's `volumes` and asserts
two mounts. `profiles: [bench]` at `:71` means no default `docker compose up`
exercises it, so the break stays latent until someone follows the documented
benchmark path.

Both files change in the same commit as the subcommand deletion.

### Group B — the nine manifest flags

Declared at `cli.py:167-198`. Seven consumer surfaces, not two:

`cli.py`, `eval/runner.py`, `scripts/colab-l4-smoke.py:336-347`,
`docs/evaluation.md:34,45-53`, `tests/test_cli_dry_run.py:96-113`,
`tests/test_eval_runner.py:108-125`, and prose in `docs/serving.md`.

The argparse entries and the `getattr` reads go in **one** commit. Split, every
serving field silently records `None` — a wrong measurement rather than a crash,
and `manifest.py:71-74` normalizes to `None` without objecting.
