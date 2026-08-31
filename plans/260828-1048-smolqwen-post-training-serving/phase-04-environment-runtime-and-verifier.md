---
phase: 4
title: "Environment runtime and verifier"
status: completed
priority: P1
effort: "4d"
dependencies: [1]
---

# Phase 4: Environment runtime and verifier

## Overview

Build the executable environment layer the RL loop stands on: instantiate any of
the 191 released environment classes from a scenario's `init_config`, expose its
tools, step it, and score the final state with the scenario's Python checklist —
all inside timeout-bounded process workers, with per-rollout state isolation
proven by test.

Independent of Phases 2 and 3, so it can be built while SFT trains.

## Requirements

**Functional**
- Environment registry: load `191_env_metadata.json`, compile each
  `env_class_code` once, cache the class object by `env_id`.
- Instantiate from `(env_id, env_class_name, init_config)` producing an isolated
  live instance.
- Tool surface: derive callable tool schemas from the environment's public
  methods, in OpenAI tool format matching what the chat template renders.
- `step(tool_call) -> observation` executing the real Python method against live
  state.
- Verifier runner: compile `check_func` sources once per scenario, execute each
  against the final state, return `R = passed / K` plus the per-checkpoint
  booleans. Signature is `score(checklist, initial_state, final_state)` —
  `initial_state` is injected as a global because 1,208 of the 40,231 released
  check functions reference it, spanning 779 of 2,550 RL scenarios (30.5%). `K` is
  the total check count, so a check that raises counts as a failed check in the
  denominator, and the result is rounded to 4 places — matching upstream's
  arithmetic exactly. `K` ranges 2 → 445 with median 14, so the timeout budget
  cannot assume a handful of checks.
- Process worker pool: N persistent workers, each owning multiple environments;
  per-call timeout; crash containment.
- **No `exec()` of dataset-supplied source runs in the trainer process.** The parent
  holds only raw JSON strings; each worker compiles its own registry and checklists
  after the credential scrub, under the per-call timeout. Compile-once is therefore
  once *per worker*, not once globally.
- Worker isolation by `spawn` (not `fork`) plus an explicitly constructed minimal
  environment: credential env vars absent, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and
  `HOME`/`HF_HOME` redirected so `~/.cache/huggingface/token` and `~/.netrc` are
  unreachable. Asserted by `huggingface_hub.get_token() is None` inside a worker, not
  merely by inspecting `os.environ`.
- Dataset revision pinning: the metadata files whose contents get `exec()`ed are
  identified by revision sha, and a sha256 of each is recorded in the conversion
  report and every manifest. A count check does not detect a modified
  `env_class_code` body.
- Episode-level state isolation — no shared mutable state between rollouts of the
  same scenario.
- Tool-call parsing from the model's `<tool_call><function=...>` output, and
  invalid-call classification (unknown tool, bad arguments, malformed syntax).

**Non-functional**
- Compile `env_class_code` and `check_func` **once**, never per step or per reward.
- No `deepcopy` of the full state per step; isolation is per episode.
- A hanging or crashing environment kills its worker's current call, not the run.
- Zero external LLM calls anywhere in this module.

## Architecture

Three layers, each with one job.

```
EnvRegistry            compile env_class_code once → {env_id: class}
                       compile check_func once     → {task_id: [callables]}
        │
        ▼
EnvInstance            reset(init_config) → live state
                       initial_state() captured at reset
                       tools() → OpenAI schemas
                       step(call) → observation
                       final_state() → dict
        │
        ▼
WorkerPool             N persistent processes
                       submit(episode_id, call) → observation | timeout | error
                       one worker owns many EnvInstances
```

`initial_state` is captured at reset and carried for the episode's lifetime, not
reconstructed at scoring time. Nearly a third of RL scenarios have at least one
check that reads it as a global; a verifier called with only `final_state` raises
`NameError` on those, which the design would swallow as `False` — systematically
depressing reward on 779 scenarios. Phase 7's difficulty profiler would then
classify many of them `always_zero` and the curriculum would drop them, shrinking
and skewing the RL training set with no error anywhere. Hence a success criterion
that zero checks fail with `NameError`.

Isolation is the correctness hinge. A GRPO group runs G rollouts of the *same*
scenario from the *same* `init_config`. If rollout 1's mutations bleed into
rollout 2, the group's rewards become correlated garbage — advantages go wrong,
training silently learns noise, and **no error is raised**. So isolation gets a
dedicated test, and the design puts each episode's instance behind a fresh
construction from `init_config` rather than a copy of a shared object.

Cost discipline: `exec()` of environment source and `exec()` of verifier source
happen at registry build time. Per-step cost must be a plain Python method call.
Same for verifiers — compile at scenario load, call at episode end. Re-`exec`ing
per reward would put a compile in the hot path for no reason.

Threat model for `exec()`: the code is MIT-licensed data from a university lab,
the worker holds no secrets, and process workers are required for async rollout
anyway. Process isolation plus per-call timeout is the accepted posture — the real
risk being managed is a buggy `check_func` hanging an overnight Colab run, not
malice.

"The worker holds no secrets" is not free, and env-var scrubbing does not deliver
it. `huggingface_hub.get_token()` reads three sources in order — the Colab secrets
vault (cached in a **module global**, so a `fork` child inherits the plaintext token
in memory no matter what `os.environ` says), then the env var, then
`~/.cache/huggingface/token` on disk (`_auth.py:30,49,71-72,121-125`). W&B
credentials additionally sit in `~/.netrc`. So the pool uses `spawn` with an
explicitly built minimal environment, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and a
redirected `HOME`/`HF_HOME`; the test asserts `get_token()` returns `None` inside a
worker rather than checking `os.environ`.

Where the `exec()` happens matters as much as what it can reach. Compiling in the
trainer process would run 516 module-level statements from dataset text — plus
40,231 `check_func` module bodies — inside the process holding the Hub token, the
W&B session, the CUDA context, and the model weights, with no timeout and no scrub,
bypassing every mitigation the accepted posture rests on. So compile-once means once
per worker, and the exec-count test counts per worker so it stops rewarding a parent
build.

Upstream's `run_check_function` passes full `__builtins__` (`env_util.py:106-108`) —
we keep that, because a restricted-builtins sandbox that silently breaks a
legitimate `check_func` would corrupt rewards, which is worse than the threat it
prevents. The `spawn`ed, credential-free worker plus the timeout is what makes full
builtins acceptable.

Crash blast radius is bounded by design, not by hope. A worker owning N live
episodes that dies takes all N with it. The pool therefore reports every affected
`episode_id` as failed and the scheduler re-admits them, rather than pretending a
single episode was lost. Episodes-per-worker is a config knob so the radius is a
chosen number.

Invalid tool calls are classified, counted, and returned as observations the model
can react to — not silently dropped. Their rate is a reported metric in Phase 7,
but never part of the reward.

## Related Code Files

- Create: `src/smolqwen/env/registry.py` — metadata load, one-time class/verifier compilation, caching
- Create: `src/smolqwen/env/instance.py` — `reset`/`tools`/`step`/`final_state`, per-episode construction
- Create: `src/smolqwen/env/tools.py` — public-method introspection → OpenAI tool schema; argument coercion
- Create: `src/smolqwen/env/parse.py` — parse `<tool_call><function=NAME><parameter=K>V` from model output; classify invalid calls
- Create: `src/smolqwen/env/verifier.py` — compiled checklist execution, `R = passed / K`, per-checkpoint detail
- Create: `src/smolqwen/env/pool.py` — persistent process workers, submit/timeout/crash containment
- Create: `src/smolqwen/env/scenarios.py` — RL scenario loading, filtering, sampling by env split
- Create: `tests/test_state_isolation.py` — **critical**: same `init_config`, two instances, mutate one, assert the other unchanged
- Create: `tests/test_verifier_rewards.py` — **critical**: known-correct → 1.0, known-partial → exact fraction, initial state → low
- Create: `tests/test_tool_schema.py` — public methods become schemas; reserved methods excluded
- Create: `tests/test_parse_invalid_calls.py` — unknown tool, bad args, malformed syntax each classified correctly
- Create: `tests/test_pool_timeout.py` — a deliberately hanging tool call times out without killing the pool
- Create: `tests/test_worker_crash_blast_radius.py` — a crashing worker reports every episode it held as failed, tagged `worker_crash`, not just the active one
- Create: `tests/test_worker_env_scrubbed.py` — no credential env var is visible inside a worker process
- Create: `tests/test_verifier_denominator.py` — a raising `check_func` counts as a failed check in `K`, not a skipped one; result matches upstream's `round(..., 4)`
- Create: `tests/test_verifier_initial_state.py` — **critical**: a `check_func` referencing `initial_state` resolves it; a scenario whose checks read it scores identically on two consecutive episodes; a check that mutates `initial_state` cannot affect the next call; zero checks across a scenario sample fail with `NameError`
- Create: `tests/test_worker_isolation_secrets.py` — `huggingface_hub.get_token()` returns `None` inside a worker, and no dataset `exec()` ran in the parent process
- Create: `tests/fixtures/scenarios.json` — a few real scenarios with known-good final states
- Modify: `configs/base/grpo.yaml` — env metadata paths, worker count, timeouts, step cap
- Modify: `src/smolqwen/cli.py` — add `env-selftest` running a scripted episode end to end

## Implementation Steps

1. Read `third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/` before
   writing anything — `base_env.py`, `utils/env_util.py` (`init_env_class`,
   `init_env_instance`, `get_state_diff`, `run_check_function`), and
   `utils/parse_util.py`. Reuse their semantics; do not reuse their ROLL/gem
   coupling.
2. `registry.py`: stream `191_env_metadata.json` **inside each worker**, `exec` each
   `env_class_code` into an isolated namespace, keep the class object. Same for
   scenario `checklist_with_func` sources at scenario-load time. The trainer process
   never execs dataset source. Assert the compiled count matches the manifest from
   Phase 2, and verify the file's sha256 against the pinned value — a count check
   passes trivially for a modified class body. State which artifact is authoritative:
   the vendored `third_party/EnvScaler` copy (`191_env_metadata.json`) or the Hub
   download (`191_env_metadata_processed.json`, a different file of a different
   size), and pin whichever is used.
3. `instance.py`: construct from `init_config`, capture `initial_state()` at reset,
   expose `final_state()`. Construction must be genuinely fresh per episode — write
   the isolation test now and let it drive the design.
4. `tools.py`: introspect public methods, build OpenAI-format schemas from
   signatures and docstrings, exclude reserved lifecycle methods. Coerce arguments
   to annotated types so a string `"3"` for an int parameter is either coerced or
   reported as an invalid call — decide once and test it.
5. `parse.py`: parse the `<tool_call><function=NAME><parameter=K>value</parameter>`
   syntax the chat template emits. Classify: `unknown_tool`, `bad_arguments`,
   `malformed_syntax`, `no_call`. Return structured results, never raise into the
   rollout loop.
6. `verifier.py`: `score(checklist, initial_state, final_state)`. Every released
   `check_func` has arity 1 with parameter `final_state`, so `initial_state` can only
   arrive through the callable's `__globals__`. Under compile-once that dict is fixed
   at scenario load, so: compile each `check_func` into **its own** dedicated globals
   dict, and before each episode's scoring write a fresh `deepcopy(initial_state)`
   into it. Never share one dict across a scenario's K checks — 11 scenarios also
   define module-level helpers that resolve through it, and a check that mutates a
   container reached via `initial_state` would corrupt every later check and episode.
   Catch per-check exceptions as `False` with a logged reason. Return
   `(reward, [per_check_bool], [reasons])`. Denominator is the full check count and
   the result is `round(..., 4)` — a raising check lowers the reward rather than
   being dropped from the average, matching `base_env.py:296-301`. Assert no check
   fails with `NameError`, which is the signal that `initial_state` was not supplied.
7. `pool.py`: N persistent processes; each holds a registry copy and a dict of live
   instances keyed by episode id. Protocol: `create(episode_id, env_id, init_config)`,
   `step(episode_id, call)`, `finalize(episode_id) -> final_state`,
   `destroy(episode_id)`. Per-call timeout; on timeout, mark that one episode
   failed and keep the worker. On worker crash, report **every** `episode_id` that
   worker held as failed — a worker owning many episodes cannot lose one in
   isolation — and replace the worker. Episodes-per-worker is a config knob that
   sets the blast radius explicitly. Distinguish `timeout` from `worker_crash` in
   the failure reason so Phase 6 can drop crashed episodes from the training
   buffer rather than scoring them.
8. `scenarios.py`: load the 2,550 RL scenarios, filter to the RL env split from
   the Phase 2 manifest, expose sampling. Include a difficulty-profiling hook that
   Phase 7 uses to find scenarios where `0 < P(success) < 1`.
9. `env-selftest`: scripted episode — reset a known scenario, issue a hardcoded
   correct tool sequence, assert reward 1.0; issue a partial sequence, assert the
   expected fraction. This is the end-to-end proof the layer works before any
   model touches it.
10. Write all five tests. The isolation and verifier tests are the two that, if
    absent, let the project produce plausible numbers that mean nothing.

## Success Criteria

- [x] All 191 environment classes compile from `env_class_code`; count asserted against the manifest.
- [x] State isolation test passes: two instances from one `init_config`, mutating one leaves the other untouched.
- [x] Verifier test passes: known-correct final state → 1.0, known-partial → the exact expected fraction, initial state → appropriately low.
- [x] Tool schema test passes: public methods become schemas, lifecycle methods excluded, argument coercion behaves as specified.
- [x] Invalid-call classification test passes for unknown tool, bad arguments, and malformed syntax.
- [x] Pool timeout test passes: a hanging tool call is bounded and the pool survives.
- [x] Worker-crash test passes: every episode the dead worker held is reported failed and tagged `worker_crash`, distinguishable from `timeout`.
- [x] Worker environment test passes: no credential env var is readable inside a worker.
- [x] Verifier denominator test passes: a raising `check_func` lowers the reward rather than being excluded from the average.
- [x] `initial_state` test passes: checks referencing `initial_state` resolve it; the same scenario scores identically on two consecutive episodes; a mutating check cannot affect the next call; zero `NameError` failures across a scenario sample.
- [x] Worker isolation test passes: `huggingface_hub.get_token()` is `None` inside a worker, and no dataset `exec()` executed in the trainer process.
- [x] `env_class_code` and `check_func` are compiled exactly once **per worker** — asserted by a test that counts `exec` calls across a multi-step, multi-episode run.
- [x] The metadata files' sha256 values are pinned and verified at registry build; the authoritative artifact (vendored vs Hub) is named in config.
- [x] `smolqwen env-selftest` runs a scripted episode end to end and reports the expected rewards.
- [x] No module in `src/smolqwen/env/` imports an LLM client, HTTP client, or API SDK.

## Risk Assessment

**State leaks between rollouts of a GRPO group.** The highest-impact silent
failure in the project: rewards still come out as plausible numbers, advantages
are wrong, and training learns noise. Signal: the isolation test fails, or in
Phase 7 the G rollouts of a group show near-identical rewards far more often than
scenario difficulty explains. Response: the isolation test gates this phase; in
Phase 7, log per-group reward variance and treat suspiciously low variance as an
isolation bug, not a training observation.

**A `check_func` hangs or crashes.** Signal: a worker stops responding, or an
overnight run dies mid-epoch. Response: per-call timeout plus per-check exception
capture, both already in the design. A failed check counts as `False` with a
logged reason rather than aborting the episode.

**A worker crash is mistaken for a bad episode.** If a crashed episode's partial
state is scored, GRPO learns that whatever the model did last was worth a low
reward — teaching it to avoid an infrastructure failure. Signal: `worker_crash`
count correlating with low-reward episodes. Response: infrastructure failures
carry a distinct terminal reason and are excluded from the training buffer
entirely, never scored as a legitimate low reward. This is the pool's contract
with Phase 6, and the crash test enforces the tagging half of it.

**Compile cost sneaks into the hot path.** Easy to introduce by calling
`exec(check_func_string)` at reward time. Signal: reward computation showing up in
the Phase 6 timeline profile. Response: the exec-count test catches it in CI
before it reaches a training run — counting per worker, not globally, so it does not
push compilation back into the trainer process.

**`initial_state` bound once instead of per episode.** Under compile-once the
callable's `__globals__` is fixed at scenario load, so a naive implementation binds
whatever `initial_state` existed then. Signal: the same scenario scoring differently
across two episodes of one GRPO group, or `NameError` in the logged reasons.
Response: a dedicated globals dict per check plus a fresh `deepcopy(initial_state)`
written before each episode's scoring. Note this defect biases rewards *downward*
without necessarily collapsing group variance, so Phase 7's zero-variance tripwire
would not catch it.

**Dataset source changes between runs.** The two metadata files whose contents get
`exec()`ed are the only inputs the plan pins by id alone. Signal: none, by default —
191 classes still compile, the suffix split is unchanged, `K` counts are unchanged.
Response: pin a revision sha, record a sha256 in the conversion report and every
manifest, and fail on mismatch. Without this, the artifact the `exec()` posture was
accepted for is not the artifact that runs.

**`initial_state` missing from the verifier signature.** 30.5% of RL scenarios
contain at least one check that reads `initial_state` as a global. Called with
only `final_state`, those raise `NameError`, which the design swallows as `False` —
so reward is silently depressed on 779 scenarios and nothing errors. Signal: a
`NameError` count above zero in the verifier's logged reasons; downstream, an
implausibly large `always_zero` band in Phase 7's difficulty profile. Response:
capture `initial_state` at reset and pass it; the dedicated test gates this phase.
Without it, Goal 3's "reward unmodified from the paper" is not met and the held-out
reward column is not comparable to any published number.

**Tool schema mismatch with the trained format.** If the schema this layer
produces differs from what SFT rendered, the model calls tools that do not exist.
Signal: high `unknown_tool` rate for an SFT checkpoint that trained cleanly.
Response: assert against Phase 2's rendered tool block for a shared fixture — same
environment, same schema text, both directions.

**Process-pool overhead exceeds the benefit.** Signal: single-episode wall time
worse than in-process execution by more than the timeout safety is worth.
Response: workers own many environments and persist across episodes; never spawn
per tool call. If overhead still dominates, measure it in Phase 6's timeline before
changing the design.
