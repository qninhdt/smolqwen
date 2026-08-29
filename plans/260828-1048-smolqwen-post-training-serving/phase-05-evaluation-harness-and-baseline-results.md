---
phase: 5
title: "Evaluation harness and baseline results"
status: code-complete-gpu-results-pending
priority: P1
effort: "4d"
dependencies: [3, 4]
---

# Phase 5: Evaluation harness and baseline results

## Delivery status

The evaluation implementation is complete and CPU-verified. Base/SFT result
generation remains pending because the Phase 3 SFT checkpoint does not exist in
this workspace and the available RTX 3050 has 4 GB VRAM. No comparison artifact
or score is fabricated. Execution evidence and the remaining GPU commands are in
[the Phase 5 completion report](../reports/cook-260830-0255-phase-05-code-complete.md).

## Overview

Build the benchmark-adapter layer, then produce the first two columns of the
headline table: Base and SFT on BFCL-v3 `multi_turn_*` plus held-out EnvScaler
environment reward. Identical decoding, prompt, tool harness, and step limits for
every checkpoint — the comparison is the deliverable, not the individual numbers.

## Requirements

**Functional**
- Benchmark adapter interface: task loading, prompt construction, stepping,
  scoring, invalid-call accounting, manifest provenance, and metric summarization.
  ACEBench-Agent and τ-bench can be added later without changing the runner or
  core eval config model.
- BFCL adapter over the four scoring multi-turn categories: `multi_turn_base`,
  `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`.
- Held-out EnvScaler evaluation reusing the Phase 4 runtime over the RL env split's
  held-out slice, reporting mean verifier reward and exact-success rate.
- Policy loaders: base model, adapter-on-base, merged checkpoint, and an HTTP
  OpenAI-compatible endpoint — same eval code for all four. `generate` returns a
  structured result carrying completion, generated-token count, and finish reason,
  because truncation rate and token stats must come from `usage.completion_tokens`
  and `finish_reason` on the HTTP path.
- Checkpoint reads are pinned by explicit revision sha, never `latest_revision`.
  The resolved sha goes into the manifest.
- An eval manifest split into two sets:
  - **invariant**, asserted equal across every run in one table: decoding params,
    system prompt hash, tool schema hash, max steps, seed, benchmark commit,
    checkpoint revision sha.
  - **recorded-free**, captured and printed but exempt from the assertion:
    serving backend, served dtype, quantization scheme, speculative-decoding
    config, KV budget, `max_num_seqs`, `max_num_batched_tokens`, chunked prefill,
    prefix-caching state, library versions.
- Secondary metrics: invalid tool-call rate, average env steps, average generated
  tokens, truncation rate.

**Non-functional**
- A run is reproducible from its manifest alone.
- BFCL multi-turn drives user turns from the benchmark's static `questions` list —
  no LLM user simulator, so Base and SFT evaluation costs nothing but compute.
- Adding a benchmark means adding one adapter file, changing nothing else.

## Architecture

The comparison is worthless unless the harness makes divergence impossible rather
than unlikely. So the manifest is not a log — it is computed, hashed, and asserted
equal across the checkpoints being compared. If a system prompt is edited between
the Base run and the SFT run, the harness must refuse to put them in the same
table.

But a single strict-equality check over every captured field fails in both
directions, and Phase 8 is where both failures land. Too strict: the serving
endpoint runs under vLLM with a different library set, so the check raises on
`library_versions` — a field that is not the experiment — and the fastest fix under
time pressure is to loosen the check that protects the training table. Too loose in
the place that matters: the manifest as originally specified captures nothing about
quantization, speculative decoding, or KV budget, so the FP8 row, the MTP-1 row,
and the BF16 row produce *identical* hashes and are declared comparable while being
different numerics.

Hence two sets. The invariant set is what makes two scores mean the same thing;
the recorded-free set is what distinguishes one row from another. `assert_comparable`
enforces the first and prints the second. Both are defined here, in Phase 5, so
Phase 8 consumes a contract rather than negotiating one.

```
EvalRunner
  ├── Policy         base | adapter | merged | http-endpoint
  │                  generate() -> {completion, n_tokens, finish_reason}
  ├── Adapter        bfcl_multi_turn | envscaler_heldout | (later: acebench, tau)
  ├── Manifest       invariant: decoding, prompt hash, schema hash,
  │                             max_steps, seed, commits, revision sha
  │                  recorded:  backend, dtype, quantization, spec-decode,
  │                             kv budget, max_num_seqs, lib versions
  └── Report         per-category scores + secondary metrics → JSON + markdown
```

BFCL multi-turn is a stateful agentic evaluation with eight API classes
(file system, trading bot, travel booking, ticket API, message API, posting API,
vehicle control, math API) and a `questions` list that advances as the model
signals turn completion. `third_party/EnvScaler`'s `bfcl_env` adapter handles
**only `multi_turn_base`** — `env.py:319` asserts `mode in ["multi_turn_base"]` and
`bfcl_env/data/` contains only `data_multi_turn_base.json`. So `miss_func`,
`miss_param`, and `long_context` are ours to build from the pinned
`gorilla-llm/Berkeley-Function-Calling-Leaderboard` files; read `bfcl_env` for the
turn-advance and state-check semantics, not as a four-category implementation.

**Do not port upstream's ground-truth `eval()`.** `bfcl_env/env.py:203,207` builds
`modified_call = f"self.state[...].{func_call}"` from released answer strings and
`eval()`s it. That would put a third code-execution sink in the *evaluation*
process — which has no worker isolation, no credential scrub, and no per-call
timeout, all of which are Phase 4 worker properties. Ground-truth calls are parsed
with `ast.parse` + `ast.literal_eval` and dispatched via `getattr`, which is what
upstream's own `bfcl_reward.py:24-40` already does for the same call shape. The
model-output path upstream is already safe (`json.loads` + `hasattr`/`getattr`); it
is only the ground-truth branch that is dangerous.

`multi_turn_long_context` is the category most likely to expose a mismatch between
the training context cap chosen in Phase 2 and what evaluation demands. Report it
separately and note any truncation.

Policies share one interface so the Phase 8 HTTP endpoint is evaluated by the same
code that evaluated the local checkpoints. That is what makes the quantization
re-eval in Phase 8 comparable to the training table rather than a separate
experiment.

## Related Code Files

- Create: `src/smolqwen/eval/runner.py` — orchestration, manifest computation and equality assertion, report writing
- Create: `src/smolqwen/eval/adapters/base.py` — the adapter protocol
- Create: `src/smolqwen/eval/adapters/bfcl.py` — BFCL multi-turn categories, turn advancement, scoring
- Create: `src/smolqwen/eval/adapters/envscaler_heldout.py` — held-out env reward via the Phase 4 runtime
- Create: `src/smolqwen/eval/policies.py` — base / adapter / merged / HTTP policy loaders behind one interface returning `GenerationResult`
- Create: `src/smolqwen/eval/manifest.py` — invariant vs recorded-free capture, hashing, cross-run equality check over the invariant set only
- Create: `src/smolqwen/eval/metrics.py` — per-category aggregation, invalid-call rate, step and token stats
- Create: `src/smolqwen/eval/report.py` — JSON + markdown table emission, including the recorded-free fields per row
- Create: `tests/test_manifest_equality.py` — differing prompt or decoding is refused; a recorded-only difference (backend, quantization) is permitted and both values are reported
- Create: `tests/test_http_policy_metrics.py` — the HTTP policy populates token count and finish reason so truncation rate is computable
- Create: `tests/test_bfcl_turn_advance.py` — turn advances only on the completion signal; questions consumed in order
- Create: `tests/test_metrics_aggregation.py` — category scores and secondary metrics computed correctly from fixtures
- Create: `tests/fixtures/bfcl_tasks.json` — a few real BFCL multi-turn entries
- Modify: `configs/base/eval.yaml` — categories, decoding, max steps, seed, benchmark pin
- Modify: `src/smolqwen/cli.py` — wire `evaluate --checkpoint ... --tag ... --adapter ...`
- Create: `notebooks/02-eval.ipynb` — thin wrapper

## Implementation Steps

1. Pin the BFCL harness: record the commit of `ShishirPatil/gorilla` used, and
   place it under `third_party/`. BFCL v4 is current and layers agentic categories
   on top of v3 — `multi_turn_*` remain scoring categories and results still carry
   the `BFCL_v3_` prefix. Do not chase v4's `web_search_*` (needs SerpAPI) or
   `memory_*` (a capability EnvScaler does not train). While pinning, confirm the
   four scoring categories need no LLM user simulator — the "no API key" claim in
   this phase's non-functional requirements depends on it.
2. Read `third_party/EnvScaler/interact_with_env/bfcl_env/` — `env.py`,
   `bfcl_reward.py`, `tool_util.py`, `xml_parser.py`, and the eight
   `bfcl_envs/*.py` classes — before writing the adapter. Note that it implements
   `multi_turn_base` only (`env.py:319`); the other three categories are built from
   the pinned gorilla files using the same turn-advance and state-check semantics.
   Do not carry over the ground-truth `eval()` at `env.py:207`.
3. `adapters/base.py`: define the protocol. `load_tasks() -> list[Task]`,
   `build_prompt(task, history) -> messages`, `step(task, call) -> observation`,
   `score(task, trajectory) -> TaskResult`. Keep it narrow enough that an ACEBench
   or τ-bench adapter is a single file.
4. `manifest.py`: capture the invariant set (decoding params, system prompt hash,
   tool schema hash, max steps, seed, benchmark commit, checkpoint revision sha) and
   the recorded-free set (serving backend, served dtype, quantization, spec-decode
   config, KV budget, `max_num_seqs`, `max_num_batched_tokens`, chunked prefill,
   prefix-caching state, library versions). `assert_comparable(manifests)` raises
   with a diff on any **invariant** mismatch and ignores the recorded set. Write the
   test first — this is the mechanism that protects the whole table, and the test
   must cover both directions: an invariant mismatch is refused, a recorded-only
   difference is permitted and both values appear in the report.
5. `policies.py`: four loaders behind one
   `generate(messages, tools) -> GenerationResult` interface carrying completion,
   generated-token count, and finish reason. Base and merged load through
   `transformers`; adapter wraps base with PEFT; HTTP calls an OpenAI-compatible
   endpoint and maps `usage.completion_tokens` and `finish_reason` into the same
   shape. Same decoding config path for all four. Checkpoint loads take an explicit
   revision sha — never `latest_revision`, or a concurrent training push can swap
   the model under a running eval.
6. `adapters/bfcl.py`: instantiate the involved API classes per task from
   `initial_config`, feed `questions[0]`, advance on the completion signal, execute
   tool calls against the live API objects, score per category. Ground-truth calls
   are parsed with `ast.parse` + `ast.literal_eval` and dispatched by `getattr`,
   never `eval()`ed — no module under `src/smolqwen/eval/` may contain an execution
   sink.
7. `adapters/envscaler_heldout.py`: reuse the Phase 4 registry, pool, and verifier
   over a held-out slice of the RL env split. The release has only two sides —
   140 `_sft` / 51 `_rl` — so the held-out slice is carved from the 51 RL
   environments and its `env_id` list is recorded in the manifest. Report mean reward
   and exact-success rate. This is the in-domain signal that tells us in Phase 7
   whether RL learned anything at all.
8. `metrics.py` + `report.py`: per-category accuracy, multi-turn overall, plus
   invalid-call rate, average steps, average generated tokens, truncation rate.
   Emit both JSON (for later diffing) and a markdown table (for the README).
9. Evaluate Base. Record the manifest.
10. Evaluate SFT (merged) under an asserted-identical manifest. Also evaluate the
    adapter-on-base path once to confirm merge did not shift behavior — if it did,
    pick the path RL will consume and use it consistently.
11. Write `artifacts/evaluation/base_vs_sft.md`. If SFT does not beat Base on
    BFCL-MT overall, stop and diagnose before building Phase 6 — the paper shows
    +8.38 at 1.7B, so a flat result means a pipeline defect, not a null finding.

## Success Criteria

- [x] Adapter protocol defined; BFCL and EnvScaler-held-out both implement it; adding a third benchmark touches no core runner/config code.
- [ ] All four `multi_turn_*` categories run and score; per-category and overall numbers reported. `miss_func`, `miss_param`, and `long_context` are implemented from the pinned gorilla files, since upstream's `bfcl_env` covers only `multi_turn_base`.
- [x] No module under `src/smolqwen/eval/` contains `eval(` or `exec(` — ground-truth calls are parsed and dispatched by `getattr`.
- [x] Manifest equality test passes both ways: a changed system prompt or decoding param is refused with a readable diff; a recorded-only difference (backend, quantization, library versions) is permitted and both values reach the report.
- [x] HTTP policy test passes: token count and finish reason are populated so truncation rate is computable over an endpoint.
- [x] Every checkpoint load in the eval path takes an explicit revision sha; the sha appears in the manifest.
- [x] Turn-advance test passes: questions consumed in order, only on the completion signal.
- [ ] Base and SFT evaluated under asserted-identical manifests; both manifests committed.
- [ ] `artifacts/evaluation/base_vs_sft.md` shows BFCL-MT per category and overall, plus held-out EnvScaler reward, invalid-call rate, average steps, average tokens, truncation rate.
- [ ] SFT beats Base on BFCL-MT overall — or the phase ends with a written diagnosis instead of a number.
- [x] No LLM API key is required to run either evaluation.

## Risk Assessment

**SFT does not beat Base.** The paper's 1.7B gains +8.38 on BFCL-MT, so a flat or
negative result points at our pipeline, not at the method. Signal: BFCL-MT overall
within noise of Base. Response: check in this order — the tool-result message shape
matches between training and rollout (Phase 2 risk: `role: "tool"` vs
`<tool_response>`-wrapped user differ by a newline on every observation), tool-call
syntax is Qwen3.5 XML rather than Qwen3 JSON, tool schema text matches between
training and eval (Phase 4 risk), loss mask covered the right tokens (Phase 2 test),
the merged checkpoint is the one being evaluated at the pinned sha, and reasoning is
actually being emitted at inference. Do not proceed to RL on a broken SFT.

**Manifest drift makes the table dishonest.** Signal: two runs disagree on prompt
or schema hash. Response: the harness refuses to combine them — that is the design,
and the test enforces it.

**The manifest is too strict in one direction and too loose in the other.** Strict
equality over library versions would refuse the Phase 8 serving re-eval that the
plan requires; capturing no serving fields would let three differently-quantized
rows share one hash. Signal: `assert_comparable` raising on a field that is not the
experiment, or two serving rows with identical manifest hashes. Response: the
invariant/recorded split defined above, tested in both directions here — never
loosened later under Phase 8 time pressure.

**A concurrent push swaps the checkpoint mid-eval.** `latest_revision` resolves at
call time, so a training run still pushing adapters can change what "SFT" means
between two eval invocations. Signal: two runs of the same tag disagreeing beyond
decoding noise. Response: reads are pinned by sha and the sha is in the manifest, so
the comparison is reproducible and a swap is visible rather than silent.

**BFCL may require an external LLM for some categories.** The `multi_turn_*`
categories drive user turns from the static `questions` list, which is why no API key
is needed. Signal: a category attempting an outbound LLM call, or a missing-key
error at scoring. Response: verify at pin time (step 1) that the four scoring
categories are simulator-free; if any is not, drop that category and record why
rather than introducing an external dependency into the headline table.

**`multi_turn_long_context` exceeds the training context cap.** Signal: high
truncation rate on that category only. Response: report the truncation rate
alongside the score rather than quietly scoring truncated episodes, and note the
cap in the README.

**An `eval()` sink lands in the eval process.** Upstream's ground-truth path
`eval()`s strings from the benchmark data, in a process holding the merged
checkpoint and the Hub token, with none of Phase 4's worker isolation, scrub, or
timeout. Signal: `eval(` or `exec(` appearing under `src/smolqwen/eval/`. Response:
parse and dispatch by `getattr` — the criterion above is the gate. This sink is
outside what the plan's accepted `exec()` posture covers, which was scoped to
environment classes and verifiers.

**Three of four BFCL categories are ours to build.** Upstream `bfcl_env` implements
only `multi_turn_base`. Signal: discovering it mid-phase and either dropping
categories or under-estimating the work. Response: already reflected in the steps —
read upstream for semantics, build `miss_func` / `miss_param` / `long_context` from
the pinned gorilla files.

**BFCL harness version churn.** v4 added categories and could rename or reweight
things between now and the final run. Signal: category names or scoring change
against the pinned commit. Response: pin the commit, vendor it under
`third_party/`, and record the pin in the manifest — never evaluate two checkpoints
against two different benchmark commits.

**Adapter-vs-merged behavioral gap.** Signal: the two paths score differently on
the same subset. Response: measure it once here, pick the path RL will consume, and
use only that path in the headline table.
