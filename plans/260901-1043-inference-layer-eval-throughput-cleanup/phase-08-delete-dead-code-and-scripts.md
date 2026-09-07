---
phase: 8
title: "Delete dead code and one-shot scripts"
status: done
priority: P2
effort: "1d"
dependencies: [1]
---

# Phase 8: Delete dead code and one-shot scripts

## Overview

Delete from Phase 1's property-aware inventory, with every live consumer
enumerated first — including the two surfaces the previous inventory missed, and
excluding the two entries it wrongly called dead.

## Requirements

- Functional: every deletion traces to a Phase 1 entry with its consumer set.
- Functional: `active_pool_multiplier` and `local_artifact_dir` are **not**
  deleted; only `max_trajectories` is.
- Functional: the readiness probe and the missing-key assertion survive.
- Functional: the `recorded_free` serving-pairing guard survives in some form.
- Non-functional: one commit per group, so a bisect isolates a regression.

## Architecture

Four groups. Two entries from the previous version are struck, and two consumer
surfaces are added.

**Group A — unreferenced symbols.** From Phase 1's re-checked list. Struck:
`active_pool_multiplier` (read at `config_models.py:75`, reaching
`grpo.py:252/463/521/526/648`, set to 2 in both profiles, asserted by
`test_rollout_metrics.py:21-23` — deleting it drops GRPO pool width 16→8 without
crashing, then fails every profile load through `extra="forbid"`) and
`local_artifact_dir` (used at `test_sft_assembly.py:79`). Only `max_trajectories`
remains among config fields. The seven symbol candidates land only with Phase 1's
consumer sets attached.

**Group B — CLI boilerplate and the nine manifest flags.** `cli.py:275-296` holds
four character-identical functions; one generic `_as_config(config, kind)`
replaces them. The nine flags have **seven** consumers, not two: `cli.py`,
`eval/runner.py`, `scripts/colab-l4-smoke.py:336-347`,
`docs/evaluation.md:34,45-53`, `tests/test_cli_dry_run.py:96-113`,
`tests/test_eval_runner.py:108-125`, and prose in `docs/serving.md`. They come out
only after Phase 4 has the client record observed serving config, and the argparse
entries plus the `getattr` reads go in **one** commit — otherwise every serving
field silently records `None`, which is the wrong-measurement outcome rather than
a crash.

**Group C — one-shot scripts.** `colab-l4-sft-speed.py` (710) and
`build-sft-benchmark-shard.py` (82) are **gated** alongside
`colab-l4-batch-sweep.py` (783): plan `260831-0808` phase 4 lists the first two as
Modify targets, and `colab-l4-sft-speed.py:283` reads the `benchmark_id` field
`build-sft-benchmark-shard.py:54` produces. The previous version gated only one of
three instruments while stating the principle that would have caught it. That
leaves `colab-probe-status.py` (27) ungated. `colab-l4-smoke.py` stays — but for
the right reason: `docs/serving.md:131` references it, not any notebook. It also
invokes `smolqwen bench` (`:290`), so it is edited here, not merely kept.

**Group D — serving wrappers.** `bench.py` (345) and `sweep.py` (209) go: both
shell out to commands vLLM ships, `sweep.py`'s docstring says it delegates
execution and Pareto logic upstream, and `bench.py`'s bulk renames vLLM's result
fields into a 21-field dataclass. Three things do **not** go:

- `wait_for_readiness` — already moved to `inference/client.py` in Phase 2, with
  its test. It refuses an open port as readiness and proves the bearer key on a
  real model path; the surviving alternative (`colab-l4-smoke.py:183-187`) polls
  `/health` on the raw vLLM port and bypasses the nginx bearer check entirely.
- `serving/report.py` — it holds `assert_quality_matches_serving`
  (`:43-58`), which compares eight **`recorded_free`** fields. The previous
  version called it dead and proposed `assert_comparable` instead, which compares
  **`invariant`** only (`manifest.py:98-112`) and would let a row measured at
  `max_num_seqs: 128` pair with a score measured at 8. Under the reproducibility
  standard this is the declared reporting rule for a paired speed/quality row, so
  the function moves into `eval/` beside the manifest it reads and keeps its test.
- `serving/workload.py` — moves to `eval/workload.py`, keeping BFCL-shaped
  benchmark traffic instead of `--dataset-name random`. It needs a CLI entry point
  in this phase or it survives with no caller.

Two consumer surfaces the previous version omitted: `docker-compose.yml:57-71`
runs a `bench` service on the deleted subcommand, and
`tests/test_auth_all_paths.py:39` reads that service's volumes. `profiles: [bench]`
means no default `docker compose up` exercises it, so the break would be latent
until someone followed the documented benchmark path.

## Related Code Files

- Delete: `scripts/colab-probe-status.py`
- Delete (gated on plan `260831-0808` phase 4 complete):
  `scripts/colab-l4-batch-sweep.py`, `scripts/colab-l4-sft-speed.py`,
  `scripts/build-sft-benchmark-shard.py`
- Delete: `src/smolqwen/serving/{bench,sweep}.py`,
  `tests/test_bench_result_parsing.py`
- Move: `serving/report.py`'s pairing function → `eval/`; `serving/workload.py` →
  `eval/workload.py` with a CLI entry point
- Modify: `src/smolqwen/cli.py` — drop `bench`/`sweep` subcommands, the nine
  flags, `_not_implemented`, the four `_as_*_config`
- Modify: `docker-compose.yml` — delete the `bench` service or rewrite it to
  invoke `vllm bench serve`
- Modify: `tests/test_auth_all_paths.py` — follow that decision
- Modify: `scripts/colab-l4-smoke.py` — replace its `smolqwen bench` call
- Modify: `src/smolqwen/eval/runner.py` — `--require-serving-match` comparing
  `recorded_free` subsets
- Modify: `config_models.py` (only `max_trajectories`), `probe.py`,
  `data/loader.py`, `data/splits.py`, `env/scenarios.py`,
  `rollout/generation.py`, `serving/server.py` — Group A entries
- Modify: `tests/test_serving_commands.py` — keep serve argv, serving env, and
  workload tests; the readiness test moved in Phase 2
- Modify: `docs/serving.md`, `docs/evaluation.md`, `Makefile`,
  `.github/workflows/ci.yml`

## Implementation Steps

1. Re-run Phase 1's audit. Anything that gained a reference during Phases 2-7 is
   not deleted. Confirm `active_pool_multiplier` and `local_artifact_dir` report
   live.
2. Group A, one commit, each entry with its consumer set.
3. Group B, one commit, after Phase 4's observed-config recording exists. Delete
   argparse entries and `getattr` reads together.
4. Group C: `colab-probe-status.py` now; the other three recorded as gated, with
   the gate condition named.
5. Group D: move the pairing function and `workload.py`, add
   `--require-serving-match` and the workload CLI entry point, then delete
   `bench.py` and `sweep.py`. Decide the compose `bench` service explicitly and
   update `test_auth_all_paths.py` to match.
6. Rewrite the `docs/serving.md` benchmark section as the two `vllm bench`
   commands with their flags.
7. Grep every deleted name across `src`, `tests`, `scripts`, `notebooks`, `docs`,
   `Makefile`, CI, **and `docker-compose.yml`**.

## Success Criteria

- [x] `active_pool_multiplier`, `local_artifact_dir` still present; profiles load
      (every stage x profile resolved after the cut)
- [x] `max_trajectories` removed
- [x] Eight flags gone with serving provenance preserved on `--endpoint`
- [x] `--require-serving-match` compares `recorded_free`, with a test that an
      `invariant`-only comparison would have passed
- [x] Readiness probe and missing-key CLI assertion both still exercised
- [x] `eval/workload.py` reachable from the CLI via `smolqwen build-workload`
- [x] Compose `bench` service rewritten as `vllm bench serve`;
      `test_auth_all_paths.py` asserts it targets the proxy and reads the key
      variable that command honours
- [x] `colab-l4-smoke.py` no longer calls a deleted subcommand
- [x] Three scripts recorded as gated, not deleted
- [x] No dangling reference across all eight surfaces
- [x] `test_rollout_equivalence.py` green
- [x] CPU suite green

## Outcome

1,183 lines deleted, 485 added. The audit reports **zero** unreferenced declarations
out of 1,456, run immediately before each deletion rather than trusting the
inventory written several phases earlier.

Group A found two entries the Phase 1 inventory had not listed — `config_models.Stage`
and `render.MASKED` — and five symbols this plan's own earlier phases introduced and
never wired. Deleting those now matters: left in place, the next audit would read
them as pre-existing debt.

Group B: it is **eight** flags, not nine. `--serving-backend` survives, because the
served process is a separate one whose engine `evaluate` cannot inspect; the other
eight described facts the in-process engine now records itself.

Group C: `colab-probe-status.py` deleted. The other three stay gated on plan
`260831-0808` phase 4, which is `in_progress`.

Group D deleted `bench.py` and `sweep.py` and kept all three things they carried:
the readiness probe (moved in Phase 2), the `recorded_free` pairing guard (now
`eval/serving_pairing.py`, with the negative control showing an `invariant`-only
comparison accepting a pairing it refuses), and the workload builder (now
`eval/workload.py` with a CLI entry point, so it is reachable rather than
test-only).

One consumer surface resolved beyond the plan's list: `config_metadata` in
`serving/server.py` existed only to feed the deleted wrapper's fingerprint, and
`vllm bench serve --save-result` records its own config.

## Risk Assessment

The assumption is that Phase 1's re-checked inventory is now correct. Its previous
version was wrong on two of three config fields, so the prior is not favorable.

- Signal it broke: an `AttributeError` at runtime, or a config key that silently
  stops taking effect.
- Response: config fields fail closed — `StrictModel(extra="forbid")` turns an
  orphaned YAML key into a load error. Function deletions are the riskier half, so
  each lands with its consumer set recorded in the commit message.

Second risk: deleting the wrappers removes an operator workflow rather than a
wrapper. Group D keeps the readiness probe, the pairing guard, and the workload
builder precisely because each carries behavior the commands do not.

- Signal it broke: `docs/serving.md`'s rewritten section cannot produce a paired
  speed/quality row.
- Response: walk the rewritten commands end to end on the Phase 10 card before
  closing this phase. If the pairing cannot be reproduced, the guard did not
  survive the move.
