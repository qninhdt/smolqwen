---
phase: 2
title: "vLLM runtime, profiles, and the shared client"
status: in_progress
priority: P1
effort: "2d"
dependencies: [1]
---

# Phase 2: vLLM runtime, profiles, and the shared client

## Overview

Create `src/smolqwen/inference/` as the single owner of every vLLM interaction,
with profiles derived from the existing `ProfileConfig` rather than parallel to
it, and settle by execution whether vLLM can load this project's adapters at all.

## Requirements

- Functional: an in-process `vllm.LLM` builds from a pinned checkpoint, batch
  generates, and exposes sleep/wake.
- Functional: profiles carry only genuinely new knobs; existing sizing is read
  from the resolved `ProfileConfig` so `--profile l4` keeps working and the
  manifest keeps recording what generation used.
- Functional: LoRA support is recorded as a **capability**, not a gate. If vLLM
  loads this project's `all-linear` adapter, adapter evaluation gets the fast path
  too; if not, `TransformersPolicy` keeps serving it. Nothing blocks on it.
- Functional: vLLM telemetry disabled explicitly at every construction site.
- Non-functional: importing `smolqwen.inference` must not import torch or vllm.

## Architecture

**Profiles derive, they do not duplicate.** The previous version declared
`EvalProfile` with `max_model_len`, `gpu_memory_utilization`, and `concurrency` —
all three already exist as `ProfileConfig.max_seq_length` (`config_models.py:54`),
`vllm_kv_fraction` (`:61`), and `generation_concurrency` (`:59`), and
`EvalConfig` already embeds a `ProfileConfig` (`:292`). Worse, the config overlay
resolves only into the `profile` subtree (`config.py:245`), so a parallel model
sits outside it: `--profile l4` would stop sizing eval, while `runner.py:42`
kept recording `max_context_tokens` from `max_seq_length` into the manifest
**invariant** — certifying comparability between runs that truncated differently.

So `EvalProfile.from_config(EvalConfig)` reads the resolved profile, and only
two genuinely new knobs are added to `ProfileConfig`: `enforce_eager` and the
LoRA slot count. Both are sizing, not semantics, which satisfies the closed-model
rule at `config_models.py:43-48`. If `max_model_len` ever needs to diverge from
`max_seq_length`, `runner.py:42` changes in the same commit.

**LoRA is a capability check, not a gate.** Both configs train
`target_modules: all-linear` (`sft.yaml:18`, `grpo.yaml:13`), which emits LoRA
weights for Qwen3.5's Gated DeltaNet mixer projections, and vLLM validates
adapter weights against a per-architecture allowlist. So vLLM may or may not
accept them.

That uncertainty does not need to block anything, because `TransformersPolicy`
(`policies.py:201`, 85 lines) stays. It is the only path that evaluates an
adapter without merging, and nothing in the user's request asked for its removal.
Keeping it means the whole question is an optimization, not a dependency:

| checkpoint kind | path | speed |
|---|---|---|
| base or merged | vLLM offline, batched | fast — the common case |
| adapter, if vLLM accepts it | vLLM offline + `LoRARequest` | fast |
| adapter, if vLLM refuses | `TransformersPolicy` | slow, unchanged |

When the adapter path is exercised on a card, record which branch was taken. One
caveat for whoever runs it: a freshly created adapter has `lora_B = 0` by
construction, so its delta is exactly zero and its output equals base **by
design** — comparing outputs only means something with an adapter that has
actually trained.

**Telemetry.** vLLM usage-stats collection is on by default and disabled only by
`VLLM_NO_USAGE_STATS`, `VLLM_DO_NOT_TRACK`, or `DO_NOT_TRACK`. Zero occurrences
exist repo-wide, while CI sets offline flags for HF, transformers, and W&B
(`ci.yml:14-18`). Phase 6 puts this engine inside the trainer process that holds
the HF token and W&B session — the boundary `test_worker_isolation_secrets.py`
exists to defend. One line at the single construction point closes it.

```
src/smolqwen/inference/
  __init__.py    # re-exports; no heavy imports
  profiles.py    # EvalProfile.from_config / RolloutProfile / ServeProfile
  engine.py      # vllm.LLM lifecycle: build, generate, sleep, wake, load_lora
  client.py      # one OpenAI-compatible chat client
```

`client.py` is justified by one real consumer, not three. Two of the three
`urlopen` sites are the same `wait_for_readiness` function and it is a
`GET /v1/models`, not a chat call — so the readiness probe is **absorbed here**
rather than deleted with `bench.py` in Phase 8. It is the only reusable HTTP
plumbing in the repo, and it refuses an open port as readiness.

## Related Code Files

- Create: `src/smolqwen/inference/{__init__,profiles,engine,client}.py`
- Create: `tests/test_inference_profiles.py` — `from_config` reads the resolved
  profile; serving argv matches today's `build_serve_command` exactly
- Create: `tests/test_inference_engine_contract.py` — surface against a fake;
  no heavy import at module scope; telemetry env set
- Create: `tests/test_vllm_adapter_capability.py` — `@pytest.mark.gpu`, adapter
  output differs from base on a trained adapter; also carries the sleep/wake VRAM
  measurement Phase 6 reads
- Create: `tests/test_http_client_auth.py` — asserts `Authorization: Bearer` is
  sent; no such assertion exists today
- Modify: `src/smolqwen/serving/server.py` — delegate argv to `ServeProfile`
- Modify: `src/smolqwen/config_models.py` — add `enforce_eager` and LoRA slots to
  `ProfileConfig`
- Move: `wait_for_readiness` from `serving/bench.py` into `inference/client.py`,
  with its test from `tests/test_serving_commands.py:82`

## Implementation Steps

1. `profiles.py`: `EvalProfile.from_config`, plus `ServeProfile.command()` moved
   verbatim from `serving/server.py:19` including both `--enable-*`/`--no-enable-*`
   pairs and the speculative-config JSON shape. Confirm
   `test_serving_commands.py` passes with argv unchanged — if it needs editing,
   the move was not mechanical.
2. `engine.py`: import vllm inside `build()`. Set `VLLM_NO_USAGE_STATS=1` there
   and in `serving_environment()`. Assert positional alignment between prompts
   and completions before returning.
3. Add `enforce_eager` and LoRA slots to `ProfileConfig`; leave every existing
   field where it is.
4. `client.py`: port base-URL normalization from `policies.py:76-100`, absorb
   `wait_for_readiness`, keep the injectable opener.
5. Verify `sleep()`/`wake_up()` on the offline `LLM` for this pin, and record the
   measured VRAM before and after with `torch.cuda.reset_peak_memory_stats()` plus
   `memory_allocated()` — not `max_memory_allocated()`, which is monotonic and
   cannot show a release. Phase 6's memory arithmetic reads this number.
   **Open — needs a card.** The test exists at
   `tests/test_vllm_adapter_capability.py::test_sleep_releases_memory_and_waking_restores_generation`
   and runs in Phase 10 step 1.
6. Record the adapter branch: attempt `LoRARequest` against a real trained
   `all-linear` adapter when one exists, and write down which path adapter
   evaluation takes. Nothing blocks on the outcome — `TransformersPolicy` covers
   the refusal case — but Phase 6 wants the answer before it picks its loading
   mechanism. **Open — needs a card.** Same file,
   `test_an_all_linear_adapter_either_loads_or_raises`; the adapter it probes with
   has non-zero `lora_B`, so an accepted-but-ignored adapter fails rather than
   reading as agreement.

## Success Criteria

- [x] `python -c "import smolqwen.inference"` leaves torch and vllm out of
      `sys.modules`
- [x] `test_serving_commands.py` passes with argv unchanged
- [x] `EvalProfile.from_config` reads the resolved `ProfileConfig`; no field is
      declared twice
- [x] `--profile l4 --dry-run` still shows eval sizing from the profile YAML
- [ ] Sleep/wake verified with a non-monotonic VRAM reading, and the released
      amount recorded for Phase 6
- [ ] Adapter branch recorded (vLLM `LoRARequest` or `TransformersPolicy`)
- [x] `VLLM_NO_USAGE_STATS` set at every construction site, asserted by test
- [x] Bearer header asserted by test
- [x] `wait_for_readiness` and its test live in the inference layer
- [x] CPU suite green; `gpu`-marked tests deselected and listed as pending

## Outcome

`src/smolqwen/inference/` created: `profiles.py`, `engine.py`, `client.py`,
`__init__.py`. Importing the package leaves torch and vllm out of `sys.modules`,
asserted by a subprocess test rather than by inspection of the current process.

`ServeProfile.command()` is `build_serve_command`'s body moved verbatim;
`build_serve_command` delegates and `test_serving_commands.py` passes with argv
unchanged. `EvalProfile.from_config` reads the resolved `ProfileConfig` and
`DecodingConfig`, with a test asserting no field is declared on both sides and a
negative control proving the two shipped profiles genuinely differ.

`ProfileConfig` gained exactly two fields: `enforce_eager` and `max_lora_slots`.

`wait_for_readiness` moved to `inference/client.py`; `serving/bench.py` delegates
and translates `ReadinessError` to `BenchError`. `HttpPolicy` now composes
`ChatClient`, so base-URL normalization and the bearer header have one owner and
a test — neither had one before.

Telemetry is set at both construction sites: `OfflineEngine.build()` before
`import vllm`, and `serving_environment()` for the subprocess.

**Two criteria remain open, and both need a card.** vllm is absent locally
(`serve`/`colab` extras only; torch 2.11.0+cu130 is installed) and the local GPU
is a 4 GB RTX 3050, below the plan's L4 floor.
`tests/test_vllm_adapter_capability.py` is written and `gpu`-marked: 3 tests
covering the adapter branch and the sleep/wake VRAM release, the second reading
`memory_allocated()` after `reset_peak_memory_stats()` because
`max_memory_allocated()` is monotonic and cannot show a release. Both numbers get
recorded in Phase 10 step 1, and Phase 6 reads them before choosing how to load a
checkpoint.

CPU suite: 368 passed, 10 deselected (7 `dataset`, 3 `gpu`). `make check` and
`make smoke` green.

## Risk Assessment

The load-bearing assumption is that the offline `LLM` on this pin exposes
sleep/wake and releases enough memory for Phase 6. vllm is not installed here
(`serve`/`colab` extras only) and the local card is a 4 GB RTX 3050, below the
plan's L4 floor, so step 5 is the first real measurement and it happens in
Phase 10.

- Signal it broke: `memory_allocated()` after `sleep()` stays near its pre-sleep
  value.
- Response: record the number and hand it to Phase 6, which then takes its
  subprocess fallback rather than discovering the problem against a live trainer.

Second risk: vLLM refuses this project's `all-linear` adapter.

- Signal it broke: `LoRARequest` raises at load, or the loaded adapter produces
  output identical to base on a trained adapter.
- Response: `TransformersPolicy` stays, so adapter evaluation keeps working —
  slower, but working. `merge-adapter` is the faster alternative when a merged
  copy is acceptable. Narrowing `target_modules` would also work but changes LoRA
  configuration, which the plan's constraints declare untouched; that is a
  decision for the user, not a workaround to apply silently.
