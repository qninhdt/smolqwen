---
phase: 1
title: "Inventory, baseline, and document reconciliation"
status: pending
priority: P1
effort: "1d"
dependencies: []
---

# Phase 1: Inventory, baseline, and document reconciliation

## Overview

Produce a deletion inventory that a property-aware audit can defend, register
the agreement tolerance later phases are measured against, capture a
re-runnable baseline, and reconcile the authority documents that still describe
superseded decisions.

## Requirements

- Functional: the audit resolves pydantic fields and `@property` bodies, not only
  call references. The previous pass missed `active_pool_multiplier` because a
  property read is neither a call nor an assignment it inspected.
- Functional: every candidate carries its live-consumer set — `src`, `tests`,
  `configs`, `docs`, `notebooks`, `scripts`, `Makefile`, CI, **and
  `docker-compose.yml`**.
- Functional: a registered tolerance and disagreement budget exist before any
  phase claims agreement with the baseline.
- Functional: **dev/test integrity is enforced by an assertion, not by two config
  files happening to hold the same numbers.** The EnvScaler dev set must be
  provably disjoint from GRPO's training scenarios, and BFCL must be unreachable
  from any in-training eval path.
- Non-functional: no source change beyond documentation, the audit script, and the
  dev/test integrity test.

## Architecture

Four deliverables. Three fix a specific way the previous version of this phase was
wrong; the fourth closes a leakage path nobody had checked.

**Dev/test integrity.** The pipeline is
`data → baseline → SFT + dev eval → RL + dev eval → test benchmark → serving`.
EnvScaler held-out is the dev set — it selects checkpoints. BFCL is the test set —
it runs once, at the end, and never influences a decision. Two things currently
make that a coincidence rather than a property:

- `grpo.py:434-441` excludes held-out task ids from training, sized by
  `curriculum.heldout_env_count` / `heldout_scenarios_per_env` (`grpo.yaml:60-61`,
  both 10/8). The eval adapter sizes its own slice from
  `adapter_options.envscaler_heldout.env_count` / `scenarios_per_env`
  (`eval.yaml:23-24`, also 10/8). The two configs are **independent** — grep finds
  no assertion tying them. Raising the eval side to 12 leaves two environments
  both trained on and scored, silently.
- `eval.yaml` lists `adapters: [bfcl_multi_turn, envscaler_heldout]`. Any
  in-training callback that reads that list scores BFCL every checkpoint, which
  turns the test set into a dev set and makes the final `Base | SFT | SFT+RL`
  table meaningless.

So this phase adds a test that computes both id sets and asserts they are
disjoint, plus an assertion that no in-training eval path can reach a BFCL
adapter. Note the split is by **task id**, not by environment: the 10 dev
environments keep their other 42 scenarios in training. That measures
generalization to unseen scenarios, not unseen environments — the existing
design's choice, recorded here rather than changed.

**Where the in-training adapter list lives.** `test_adapter_protocol.py:22-35` is
a deliberate contract: `EvalConfig` must not carry `heldout_env_count`,
`heldout_scenarios_per_env`, or `env`, and `runner.py` must not name `bfcl` or
`envscaler`. So the dev-adapter selection for Phases 5 and 6 goes on
`GrpoConfig` / `SftConfig`, or through the already-opaque
`EvalConfig.adapter_options` — never as a new named field on `EvalConfig`. This is
a constraint on Phases 5-6, resolved here so they do not rediscover it.

**The audit.** `active_pool_multiplier` reads as dead to an AST pass that walks
`FunctionDef`/`ClassDef` and counts name references, because its only `src`
consumer is a `@property` body two lines below its declaration
(`config_models.py:75`) and its other consumers are two YAML files and one test.
Deleting it would have dropped the GRPO prompt pool from 16 to 8 on L4 **without
crashing**, then failed every profile load through `extra="forbid"`. So the audit
classifies by symbol kind:

| Kind | Resolution |
|---|---|
| function / class | reference count outside its own definition |
| pydantic field (`AnnAssign` under a `StrictModel`) | `src` reads including property bodies, plus every `configs/**/*.yaml` key, plus test constructor kwargs |
| `TrainerCallback` hook | allowlisted, never proposed — HF invokes by protocol |
| trainer / sampler override | allowlisted — `get_eval_dataloader`, `_get_train_sampler` |

The allowlist needs `grpo.py:162 on_save` too; the previous version named four
hooks and missed it, so the script would have proposed a live callback.

**The tolerance.** "Byte-identical metrics" cannot survive replacing HF
`generate()` with vLLM. Three divergences exist in the current code before any
kernel difference: eval decodes with `skip_special_tokens=True` while the seam
Phase 3 unifies onto uses `False`; `finish_reason` is synthesized from a token
count (`policies.py:284`) rather than observed EOS, and feeds `truncation_rate`;
`generated_tokens` is a tensor width including EOS, and feeds
`average_generated_tokens`. Float summation order in `aggregate` also varies with
completion order. So this phase registers, in the report:

- a per-category score tolerance,
- a maximum count of per-task disagreements (task id level, not aggregate),
- the rule that a disagreement list is always published, never summarized.

**The baseline.** `.gitignore:12` ignores `artifacts/` and negates only
`artifacts/data/*`, so the previously chosen path was uncommittable without
either `git add -f` — against a rule whose stated purpose is keeping bearer
credentials out — or broadening the negation to a directory that carries
`recorded_free.endpoint`. The baseline goes under this plan's reports directory
instead, which is already tracked.

## Related Code Files

- Create: `scripts/audit-dead-symbols.py` — kind-aware audit with the allowlist
  and `--json`
- Create: `tests/test_dev_test_integrity.py` — GRPO training task ids and
  EnvScaler dev task ids are disjoint at the shipped config values; no
  in-training eval path resolves a BFCL adapter
- Create: `{plan-dir}/reports/tolerance.md` — registered tolerance and
  disagreement budget
- Create: `{plan-dir}/reports/baseline-preref.json` — baseline metrics, or the
  exact command when no card is available
- Modify: `plans/260828-1048-smolqwen-post-training-serving/plan.md`,
  `phase-02-…`, `phase-03-…`, `phase-05-…`, `phase-08-…` — reconcile superseded
  decisions (already partly done; verify and complete)
- Modify: `plans/260831-0808-sft-full-trajectory-padding-free/plan.md` — resolve
  the blocking direction to one edge, not two
- Modify: `docs/evaluation.md` — state the dev/test boundary explicitly: EnvScaler
  held-out selects checkpoints, BFCL runs once and selects nothing

## Implementation Steps

1. Write the audit with the four-kind classification above. For pydantic fields,
   resolve property bodies in the declaring class and grep `configs/**/*.yaml`
   for the bare key.
2. Run it. Confirm it now reports `active_pool_multiplier` and
   `local_artifact_dir` as **live**, and `max_trajectories` as the only dead
   field. If it still calls either live field dead, the classification is wrong
   and no deletion proceeds.
3. Re-check the seven remaining symbol candidates with the same pass:
   `cli.py:238 _not_implemented`, `data/loader.py:220 is_conversation`,
   `data/splits.py:48 split_trajectories`, `env/scenarios.py:143 sample_ids`,
   `probe.py:149 probe_payload`, `rollout/generation.py:70 SamplingParams`,
   `serving/server.py:94 safe_command_text`. Record each with its consumer set.
4. Enumerate live consumers for every Phase 8 deletion target, including
   `docker-compose.yml` and `tests/test_auth_all_paths.py`, which the previous
   inventory omitted and which together make the compose `bench` service a
   latent break.
5. **Write the dev/test integrity test.** Build GRPO's training candidate ids the
   way `grpo.py:434-441` does and EnvScaler's dev ids the way
   `envscaler_heldout.py:47-54` does, both from the shipped configs, then assert
   the two sets are disjoint. Assert separately that no in-training eval
   configuration resolves a BFCL adapter. The test reads both config files, so a
   later edit to `heldout_env_count` or `env_count` fails loudly instead of
   leaking environments in silence.
6. Register the tolerance. Derive the score tolerance from BFCL's scoring shape:
   one flipped task on an 80-task set moves a category score by 1/80, so the
   tolerance must be stated in tasks, not in score points, and converted.
7. Capture the baseline on a bounded subset with the current serial path. If no
   card is available, record the exact command and mark Phase 4's agreement
   criterion as settled in Phase 10.
8. Fix the blocking direction: this plan's `plan.md` keeps no `blocks` edge, and
   only Phase 6 carries `blockedBy`. Do not add a second field to the other plan.
9. State the dev/test boundary in `docs/evaluation.md`, with the caveat that
   deterministic tool-calling benchmarks carry brittle state comparison and
   possible ground-truth error, so a low BFCL score is not automatically a model
   deficiency.

## Success Criteria

- [ ] Audit reports `active_pool_multiplier` and `local_artifact_dir` live, and
      `max_trajectories` dead
- [ ] `grpo.py:162 on_save` allowlisted alongside the other framework hooks
- [ ] All seven symbol candidates re-checked, each with its consumer set recorded
- [ ] `docker-compose.yml` and `test_auth_all_paths.py` present in the consumer
      enumeration for the compose `bench` service
- [ ] **GRPO training ids and EnvScaler dev ids proven disjoint by test, from the
      shipped configs — not by two files holding matching numbers**
- [ ] **No in-training eval configuration can resolve a BFCL adapter, asserted**
- [ ] Dev-adapter selection placed on `GrpoConfig`/`SftConfig` or
      `adapter_options`, never as a named `EvalConfig` field
      (`test_adapter_protocol.py:28-35`)
- [ ] Tolerance and disagreement budget registered, expressed in tasks
- [ ] Baseline committed under the plan's reports directory, or its command
      recorded and Phase 4's criterion deferred to Phase 10
- [ ] Master plan phases 2, 3, 5, 8 and goal 6 reconciled
- [ ] Exactly one blocking edge between the two plans
- [ ] `docs/evaluation.md` states the dev/test boundary and the benchmark caveat

## Risk Assessment

The audit is still textual at bottom, so a symbol reached only through
`getattr`, a config string, or an entry-point table can read as dead. This
codebase uses that pattern in the trainer path (`grpo.py:406`, `:640`).

- Signal it broke: a Phase 8 deletion produces an `AttributeError`, or a config
  key silently stops taking effect.
- Response: for every candidate, grep the bare name without word boundaries
  across all eight consumer surfaces before deleting. Config fields are the safer
  case — `StrictModel(extra="forbid")` turns an orphaned YAML key into a load
  error rather than silence.

Second risk: the tolerance becomes a way to pass a phase that should fail. A
tolerance wide enough to absorb any divergence proves nothing.

- Signal it broke: Phase 10 reports agreement within tolerance while the
  per-task disagreement list is long, or while disagreements cluster in one
  category.
- Response: the disagreement list is published, not summarized, precisely so
  this is visible. Cluster analysis by category is part of Phase 10's read, and a
  clustered pattern reopens Phase 4 regardless of the aggregate.

Third risk: the integrity test encodes today's config values rather than the
property, so it passes while a future edit still leaks.

- Signal it broke: the test asserts literal `10`/`8` anywhere, or passes after
  `eval.yaml`'s `env_count` is raised to 12.
- Response: the test computes both id sets from the loaded configs and compares
  them. It must never hardcode a count. Verify by temporarily raising
  `env_count` and confirming the test fails.
