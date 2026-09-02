---
title: "Shared vLLM inference layer, batched evaluation, in-training benchmark eval"
description: "Collapse three divergent inference paths into one vLLM-backed layer, make evaluation batched and GPU-bound, log held-out benchmark scores during SFT and GRPO through that same path, put rich logging on every CLI, and delete dead and wrapper code."
status: pending
priority: P1
effort: "17.5 phase-days"
tags: [inference, vllm, evaluation, throughput, observability, cleanup]
created: 2026-09-01
revised: 2026-09-01
---

# Shared vLLM inference layer, batched evaluation, in-training benchmark eval

## Overview

Three code paths generate tokens today and none of them share an
implementation: `eval/policies.py` drives HuggingFace `generate()` one sequence
at a time, `rollout/generation.py` drives TRL's colocated vLLM engine, and
`serving/server.py` shells out to `vllm serve`. `grep "import vllm" src/`
returns zero hits — the evaluation path never touches vLLM at all.

The consequence is measurable. `eval/runner.py:69` iterates tasks serially and
`:76` iterates turns serially, so `policies.py:247` runs `model.generate()` at
batch size 1 for every turn of every task. A 2B model decoding one sequence
leaves the card almost idle.

Rollout and evaluation drive the same shape of loop — render, generate, parse,
step the environment, append, repeat to terminal, score. **They are not the same
loop.** A red-team pass (recorded below) found seven concrete divergences, five
of them structural: the observation role and per-turn tool set, two
incompatible turn-budget counters, the decode convention and assistant-message
shape, failure-cleanup liveness derived from the mask-builder dictionary, and an
episode-admission model that assumes every position stays live. Unification is
still the right direction — it is what makes one benchmark number mean the same
thing in training and in `evaluate` — but it is a seven-way reconciliation, not
a four-way one, and this plan budgets it as such.

Neither training stage reports a benchmark number. SFT computes teacher-forced
validation loss (`sft.py:299`); GRPO logs verifier reward over
curriculum-weighted *training* scenarios. Nothing measures held-out capability
until a run finishes and someone invokes `evaluate` by hand.

## Pipeline position

```
data → baseline → SFT + dev eval → RL + dev eval → test benchmark → serving
```

**Dev is EnvScaler held-out. Test is BFCL.** Dev selects checkpoints and drives
the learning curve; BFCL runs once at the end and influences no decision. That
boundary is currently a coincidence — `grpo.yaml:60-61` and `eval.yaml:23-24`
both happen to hold 10/8, with no assertion tying them, so raising either leaves
environments both trained on and scored. Phase 1 turns it into a tested property.

Training-side eval is **checkpoint-based, not live weight sync**: TRL's colocated
sync exists because GRPO's generation is part of the algorithm, while SFT scores a
checkpoint already written to disk (`sft.py:486-498`) and pushed to the Hub. That
is the pattern open post-training pipelines use, and it is why SFT needs no
weight-transfer protocol of its own.

## Goals

| # | Goal | Priority |
|---|------|----------|
| 1 | One inference layer owns every vLLM interaction, with per-task profiles derived from the existing `ProfileConfig` rather than parallel to it | P1 |
| 2 | `evaluate` runs tasks concurrently against an in-process `vllm.LLM`, within a measured agreement tolerance of the pre-refactor path and at materially higher GPU utilization | P1 |
| 3 | One turn engine drives both training rollout and benchmark evaluation, with all seven divergences reconciled explicitly | P1 |
| 4 | SFT and GRPO log **dev** benchmark scores during training through that same engine, against a recorded weight version, at or under 10% of training wall time | P1 |
| 5 | Dev/test separation is enforced by assertion: EnvScaler held-out selects checkpoints, BFCL runs once at the end and selects nothing | P1 |
| 6 | Every evaluation writes a per-task trajectory record, so a score is re-derivable and a failure attributable without re-running generation | P2 |
| 7 | Score stability across batch composition is measured and recorded, with environment-timeout confounds separated from kernel effects | P1 |
| 8 | Every CLI command reports progress and errors through rich on stderr, leaving every machine-readable stdout emitter byte-identical | P2 |
| 9 | Dead code, one-shot scripts, and thin wrappers are deleted from a property-aware inventory, with each deletion's live-consumer set enumerated first | P2 |
| 10 | Evaluation reports, budgets, difficulty profiles, trajectories, and the merged model survive VM loss, with ingress details redacted before upload | P2 |

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Phase 1: Inventory, baseline, and document reconciliation](./phase-01-start.md) | Pending |
| 2 | [Phase 2: vLLM runtime, profiles, and the shared client](./phase-02-vllm-runtime-and-inference-layer.md) | Pending |
| 3 | [Phase 3: Unify the turn engine](./phase-03-unify-turn-engine.md) | Pending |
| 4 | [Phase 4: Batched evaluation runner](./phase-04-batched-eval-runner.md) | Pending |
| 5 | [Phase 5: GRPO in-training benchmark eval](./phase-05-grpo-in-training-benchmark-eval.md) | Pending |
| 6 | [Phase 6: SFT in-training benchmark eval](./phase-06-sft-in-training-benchmark-eval.md) | Pending |
| 7 | [Phase 7: Rich logging across every CLI](./phase-07-rich-logging-across-cli.md) | Pending |
| 8 | [Phase 8: Delete dead code and one-shot scripts](./phase-08-delete-dead-code-and-scripts.md) | Pending |
| 9 | [Phase 9: Artifact persistence to HF and W&B](./phase-09-artifact-persistence-hf-wandb.md) | Pending |
| 10 | [Phase 10: L4 validation](./phase-10-gpu-validation-on-l4.md) | Pending |

Dependencies: 2 needs 1. 3 needs 2. 4 needs 3. 5 needs 4. 6 needs 5 **and**
plan `260831-0808` phase 4 marked complete. 8 needs 1. 7 needs 8 — both edit
`cli.py`, and deleting first means Phase 7 rewrites four error handlers instead
of six. 9 needs 7 for its logger. 10 needs 4, 5, 6, 7.

No phase blocks on vLLM's LoRA support. `TransformersPolicy` stays as the
adapter-on-base path, so whether vLLM accepts this project's `all-linear` adapter
decides how *fast* adapter evaluation is, not whether it works. Phase 2 records
which branch applies; Phase 6 reads that record when choosing how to load a
checkpoint.

## Constraints

- `torch==2.11.0` with `vllm==0.26.0`. The only pin for which prebuilt
  `flash-attn` and `causal-conv1d` wheels exist. vLLM lives in the `serve` and
  `colab` extras, so it is **absent from CI by construction**
  (`uv sync --locked --dev`), and `make test-ci` deselects `gpu`-marked tests.
- Single L4 24GB or A100 on Colab. One session at a time.
- `--dry-run` must resolve and validate config without importing torch or vllm
  (`cli.py:1-8`). The console and inference packages must not import either at
  module scope.
- 56 test files must stay green on CPU. A CPU-green result is **not** evidence
  for any claim that requires a card; those claims are marked `gpu` and settled
  in Phase 10.
- The evaluation manifest's `invariant` set must not change, and any field it
  records must be the value generation actually used.
- BFCL is the test set. No in-training path may score it, and it runs once, after
  checkpoint selection is complete.
- In-training eval must stay at or under 10% of training wall time.
- `TransformersPolicy` stays. It is the only path that evaluates an adapter
  without merging; its removal was never requested.
- Every machine-readable stdout emitter keeps byte-identical output. There are
  at least six, and two are consumed by committed notebooks.
- Verifier reward semantics, GRPO loss math, and LoRA configuration are
  untouched — including `target_modules: all-linear`.

## Non-goals

- No new benchmarks or benchmark adapters.
- No changes to `third_party/EnvScaler`.
- No notebook changes. This constrains Phase 7: an emitter a notebook parses
  cannot move to stderr.
- No quantization sweep. `configs/serving/{l4,a100}.yaml` keep
  `quantization: null` with the reason recorded in those files.

## Success Criteria

- [ ] `evaluate` agrees with the re-captured baseline within the tolerance
      registered in Phase 1, with per-task disagreements enumerated rather than
      summarized
- [ ] Measured throughput gain recorded with the generation / tokenization /
      environment split, so any shortfall is attributed from data
- [ ] One turn-loop implementation; `grep -rc "class GenerationResult" src/`
      returns 1
- [ ] All seven rollout and environment tests green at their post-move import
      paths, with assertions byte-identical as a reviewable diff
- [ ] Batch-composition stability measured at concurrency 1 and N, with
      environment-timeout counts reported alongside so confounds are separable
- [ ] GRPO training ids and EnvScaler dev ids proven disjoint by test, computed
      from the shipped configs rather than from matching literals
- [ ] No in-training eval path can resolve a BFCL adapter; BFCL runs once at the
      end and the report states it selected nothing
- [ ] Every evaluation writes a per-task trajectory record carrying the failing
      scoring condition
- [ ] `sft/bench_*` and `grpo/bench_*` carry the weight version they were
      measured at, and match `evaluate` at that same version within tolerance
- [ ] Measured in-training eval cost at or under 10% of training wall time
- [ ] An eval failure during training logs, releases its environments, and
      training continues
- [ ] Every machine-readable stdout emitter byte-identical; rich output on
      stderr only
- [ ] `smolqwen --dry-run` works for every subcommand without vllm installed
- [ ] Deletions traced to a property-aware inventory with live consumers
      enumerated per entry
- [ ] Eval reports, `budgets.json`, difficulty profiles, and the merged model
      recoverable after VM loss, with no artifact containing a routable ingress
      URL

## Red Team Review

### Session — 2026-09-01
**Findings:** 40 raw across 4 reviewers, 39 accepted, 1 recorded as not-a-finding.
**Severity:** 13 Critical, 17 High, 9 Medium. **Tier:** Full (10 phases).
Every finding carried `file:line` evidence; none rejected by the evidence filter.

The pre-revision plan was not implementable. Phases 2-6 were re-derived from
these findings rather than patched; Phases 1 and 7-10 were corrected in place.

#### A — the four-concern claim was false (7 divergences, 5 structural)

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 1 | `scheduler.py:464` hardcodes `Message(role="tool", ...)` and `ScenarioBinding` is frozen at bind; `bfcl.py:405` returns `observation_role="user"` and every `StepResult` carries `tools=state.active_tools`. `multi_turn_miss_func` exists because tools appear mid-episode | Critical | Accept | P3, P4 |
| 2 | Two turn budgets: `runner.py:84` counts every generation, `scheduler.py:390` counts only inside `_on_step`. Prose output freezes the engine counter until `episode_timeout_s: 600.0` | Critical | Accept | P3, P4 |
| 3 | Decode seam diverges: `rollout_func.py:138` `skip_special_tokens=False` + `Message(reasoning_content=...)` vs `policies.py:282` `True` + raw. `scheduler.py:356` documents the consequence | Critical | Accept | P2, P3, P4 |
| 4 | `_cleanup_after_failure` derives liveness from `set(self._builders)` (`scheduler.py:548`); mask-off eval makes cleanup a no-op and leaks environments into later `PoolError` | Critical | Accept | P3, P5, P6 |
| 5 | The four `assemble_output` assertions are existence checks gated on `episode.observations` — the list written by the method that must fork. A uniformly shifted mask passes | High | Accept | P3 |
| 6 | BFCL has no create/score/destroy to map onto `EnvDispatcher`'s four methods; completed-future adaptation covered `submit_step` only | High | Accept | P3 |

#### B — capacity, throughput, determinism

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 7 | 80 tasks (`eval.yaml` 10×8) vs 32 episodes (`l4.yaml` 4×8); `scheduler.py:181` opens all slots, task 33 raises `PoolError`, run scores zero | Critical | Accept | P2, P4 |
| 8 | `pool.py:239` uses `ITIMER_REAL`, so CPU contention at concurrency N turns a passing task into `AdapterResult(0.0, False)`. Every mitigation rung was GPU-side | High | Accept | P4, P10 |
| 9 | `_generate_ready` renders all ready slots then slices to `generation_concurrency` (`scheduler.py:296-331`) — 10× redundant tokenization at eval scale, misdiagnosable as env latency | High | Accept | P3, P4, P10 |
| 10 | `enforce_eager` was the wrong first rung: CUDA graph capture does not reorder reductions | Medium | Accept | P4 |

#### C — deleting `TransformersPolicy` had no proven replacement

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 11 | `target_modules: all-linear` (`sft.yaml:18`, `grpo.yaml:13`) emits GDN mixer weights; vLLM validates against a per-architecture allowlist. Loud mode raises with no fallback; **silent mode serves base output**, making `bench_*` flat and Phase 10's cross-check pass while both sides are wrong | Critical | Accept | P2, P4, P6 |
| 12 | Adapter-revision pinning exists only in `TransformersPolicy.__init__` (`policies.py:217-219`); `resolve_eval_checkpoint` takes no adapter argument | High | Accept | P2, P4 |

#### D — the headline criterion was unachievable

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 13 | "Byte-identical metrics" cannot hold across two stacks, yet Phase 10 halted on it. Beyond kernels: the decode-flag difference; `finish_reason` synthesized from a token count (`policies.py:284`) feeding `truncation_rate`; `generated_tokens` as tensor width feeding `average_generated_tokens`; float summation order in `aggregate` varying with completion order | Critical | Accept | P1, P4, P10 |

#### E — the deletion inventory was factually wrong

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 14 | `active_pool_multiplier` is read at `config_models.py:75` → `grpo.py:252/463/521/526/648`, set to 2 in both profiles, asserted by `test_rollout_metrics.py:21-23`. Deleting it drops pool width 16→8 with no crash, then `extra="forbid"` rejects both profiles | Critical | Accept | P1, P8 |
| 15 | `local_artifact_dir` is used at `test_sft_assembly.py:79`. Only `max_trajectories` is genuinely unreferenced | High | Accept | P1, P8 |
| 16 | The audit script could not see pydantic `AnnAssign` fields or `@property` bodies — exactly the shape it missed. The other seven "dead" symbols need re-checking | Critical | Accept | P1, P8 |

#### F — deletions broke live consumers absent from the file lists

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 17 | `docker-compose.yml:57-71` runs a `bench` service on the deleted subcommand; `test_auth_all_paths.py:39` reads that service. `profiles: [bench]` keeps the break latent | Critical | Accept | P8 |
| 18 | `episode.py` has 8 importers (listed 3), `mask.py` 2 (listed 1), including two tests the phase required to pass "unedited" while moving what they import | Critical | Accept | P3 |
| 19 | The nine flags have 7 consumers, not 2: adds `colab-l4-smoke.py:336-347`, `docs/evaluation.md:34,45-53`, `test_cli_dry_run.py:96-113`, `test_eval_runner.py:108-125` | High | Accept | P8 |
| 20 | `colab-l4-sft-speed.py` is a Modify target of plan `260831-0808` phase 4 — the plan gated the wrong script; `build-sft-benchmark-shard.py:54` produces the field `colab-l4-sft-speed.py:283` reads | High | Accept | P8 |
| 21 | `colab-l4-smoke.py:290` invokes `smolqwen bench`; the script was kept on the false grounds that notebooks invoke it (only `docs/serving.md:131` does) | High | Accept | P8 |
| 22 | `scheduler_config_for` has 2 files / 4 sites, and `training/grpo.py` as destination creates an import cycle with `rollout/bench.py` | Medium | Accept | P3, P8 |
| 23 | `AdapterResult` is frozen positional with 9 sites across 4 files (listed 3); missing `test_adapter_protocol.py:15`. New fields need defaults | Medium | Accept | P4 |

#### G — the quality-pairing replacement was strictly weaker

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 24 | `serving/report.py:43-58` compares eight **`recorded_free`** fields; the proposed `assert_comparable` compares **`invariant`** only (`manifest.py:98-112`). A row at `max_num_seqs: 128` would pair with a score at 8. The function was also cited in the wrong file | High | Accept | P8 |
| 25 | Dropping the nine flags nulls serving provenance on the `--endpoint` path Phase 4 keeps; `manifest.py:71-74` normalizes to `None` and `assert_comparable` never objects | High | Accept | P4, P8 |

#### H — security and hygiene

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 26 | vLLM usage-stats telemetry is default-on, disabled only by `VLLM_NO_USAGE_STATS`/`VLLM_DO_NOT_TRACK`/`DO_NOT_TRACK` — zero repo occurrences. Phase 6 put the engine inside the trainer holding the HF token, inside the boundary `test_worker_isolation_secrets.py` defends. CI sets offline flags for HF/transformers/W&B, none for vLLM | High | Accept | P2, P6 |
| 27 | `runner.py:167` writes `endpoint` into both artifacts (`report.py:64`), and Phase 9 uploaded both. Colab uses a public `trycloudflare.com` host; URL userinfo would be stored whole. The repo prints the key *file path*, never the key (`test_auth_all_paths.py:49-50`) | High | Accept | P9 |
| 28 | The unnamed scratch adapter can be picked by `merge.py:61` (`sorted(glob(...))[0]`) and pushed by Phase 9; `artifacts/` is gitignored so no diff shows it. Reusing `save_adapter` for the merged model `rmtree`s its target | High | Accept | P6, P9 |
| 29 | Phase 8 deleted `wait_for_readiness` — the probe refusing an open port as readiness — plus the only CLI test that a missing `VLLM_API_KEY` exits 2. The surviving probe (`colab-l4-smoke.py:183-187`) polls `/health` on the raw vLLM port, bypassing the nginx bearer check | Critical | Accept | P8 |
| 30 | `.gitignore:12` ignores `artifacts/` with negations only under `artifacts/data/*`, so Phase 1's baseline path was uncommittable | Medium | Accept | P1 |
| 31 | No test asserts `HttpPolicy` sends `Authorization: Bearer`, so the auth relocation had no guard | Medium | Accept | P2 |

#### I — specified metrics duplicated or uncomputable

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 32 | `per_check_pass_rate` renames the existing `score`: `verifier.py:246` returns `passed / total` and `metrics.py:27` already averages it. EnvScaler is **not** all-or-nothing. What is actually discarded is *which* checks failed and `name_error_count` | High | Accept | P4, P5 |
| 33 | `state_match_rate` / `result_match_rate` cannot be computed for the tasks they explain: `bfcl.py:194` returns before `expected_snapshots` exists, and `zip(..., strict=True)` is undefined on length mismatch. Filling `0.0` restores the conflation `completion_rate` removed | High | Accept | P4 |

#### J — false claims in the plan's own reasoning

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 34 | Phase 5 claimed the colocated engine is already current, citing `grpo.py:633-660`. `grep "\.train()" grpo.py` returns nothing — `run_profile_difficulty` never trains and calls `sync_weights()` *because* of that. With `grad_accum: 8` the engine can be 8 steps stale, indistinguishable from a parity bug | Critical | Accept | P5 |
| 35 | Phase 6's transfer cited `CheckpointPushCallback`, which returns early unless TRL wrote `checkpoint-N` (`sft.py:486-489`) — it cannot fire at an arbitrary boundary. "No new machinery" was false | High | Accept | P6 |
| 36 | Phase 6 specified no engine-construction ordering, and `gpu_memory_utilization` resolves against total memory: built after the trainer it OOMs, built before it shrinks the envelope the phase promises to preserve | High | Accept | P6 |
| 37 | `EvalProfile` duplicated `ProfileConfig` and sat outside the overlay chain (`config.py:245` resolves only into `profile`), so `--profile l4` would stop sizing eval while `runner.py:42` recorded the old value into `invariant` | Critical | Accept | P2 |

#### K — verification that could not verify

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 38 | Every phase gated on "full suite green on CPU", but vllm is absent from CI (`pyproject.toml:33-36`) and `make test-ci` deselects `gpu`. The determinism test against a CPU fake passes unconditionally; the memory guard compared constant to constant; the engine contract test proves the wrapper, not `LLM.generate` | High | Accept | P2, P4, P6, P10 |
| 39 | Phase 7 asserted three stdout emitters; there are at least six — `selftest.py:197`, `rollout/bench.py:290`, `grpo.py:668`, `merge.py:126`, plus `serve --print-command` (`server.py:87`). `notebooks/03-grpo.ipynb` and `01-sft.ipynb` consume two, and notebook changes are a non-goal | High | Accept | P7 |

#### L — plan mechanics

| # | Finding | Sev | Disp | Applied |
|---|---------|-----|------|---------|
| 40 | Mutual blocking cycle across the two plan files; Phase 1 step 5 would have deepened it | Medium | Accept | plan.md, P1, P6 |
| 41 | Plan `260831-0808` phase 4 modifies the three tests Phase 6 required to pass "unedited" — unmeetable in any ordering. Both plans also edit `config_models.py` and `tracking.py` | High | Accept | P6 |
| 42 | Phases 7 and 8 were parallel-authorized while both editing `cli.py`; Phase 9 declared `dependencies: []` but needed Phase 7's logger | Medium | Accept | plan.md, P7, P8, P9 |
| 43 | Phase 9 turned a user question into a work phase, including an unrequested `CheckpointStore.push` behavior change whose contract is pinned by `test_checkpoint_pinning.py:110-116` | Medium | Accept | P9 |
| 44 | Effort did not reconcile (files summed 11.0 vs header 12.5), and Phases 3/4 were budgeted under the master plan's 5d/4d precedent for the same subsystems | Medium | Accept | plan.md, P3, P4 |
| 45 | `client.py` was justified by three `urlopen` sites, but two are the same readiness probe Phase 8 deletes, and it is a `GET /v1/models`, not a chat client | Medium | Accept | P2, P8 |
| 46 | Counting errors: 56 test files not 59; `resolve_eval_checkpoint` is *unwired*, not dead (`test_checkpoint_pinning.py:98,101`); `grep -c` in three criteria was missing `-r` | Medium | Accept | plan.md, P3, P4 |

#### Corrections to the plan author's own earlier claims

- The `_openai_messages` defect was reported as Critical. `policies.py:164-173`
  gates rewriting on a `str` `name` plus a `Mapping` `arguments`, so prose ending
  in JSON does not trigger it. **Downgraded to Medium**; the narrow bug stands.
- "Three unread config fields" — wrong on two of three (14, 15).
- "Nine flags, two consumers" — seven (19).
- "`episode.py`, three importers" — eight (18).
- "`serving/report.py` is dead once `bench.py` goes" — it holds the only
  `recorded_free` pairing guard (24).
- "Three stdout emitters" — at least six (39).

#### Not a finding

The Failure Mode Analyst could not construct a deadlock or busy-spin under a
completed-future driver: `_default_wait` returns immediately for done futures and
`_reap` always transitions state. Recorded because it was the pre-revision
Phase 3's stated secondary risk and did not survive tracing. Note that
`rollout_fixtures.py:190-193` returns an unresolved future, so no existing
fixture covers the completed-future path — Phase 3 adds one.

### Whole-Plan Consistency Sweep
- Files reread: `plan.md`, `phase-01` … `phase-10` (11 files)
- Decision deltas checked: 46
- Reconciled stale references: 46
- Unresolved contradictions: 0

<!-- slug: inference-layer-eval-throughput-cleanup -->
