---
title: "Six pipeline defects, found and fixed"
status: complete
date: 2026-09-04
branch: feat/inference-layer-eval-throughput
---

# Six pipeline defects, found and fixed

Audit of `data → SFT (+dev eval) → GRPO (+dev eval) → evaluate → serve` on a fully
green worktree (CPU suite, lint, mypy all passing before any change). Every finding
verified against installed dependency source — `vllm==0.26.0` unpacked from its
wheel, `transformers 5.16.1`, `trl 1.12.0`, `peft 0.20.0` — or by running the command.

## What was wrong

| # | Sev | Defect | Verified by |
|---|-----|--------|-------------|
| 1 | Critical | `max_lora_rank` never passed to `vllm.LLM`; default 16 vs configs' `r: 32` | `vllm/config/lora.py`, `lora/peft_helper.py:118` |
| 2 | High | Capability markers matched no message vLLM 0.26 raises | grep over whole wheel; 4/4 real messages returned `False` |
| 3 | High | GRPO bench eval sized from `ProfileConfig` defaults: 32K ctx vs 16K engine | resolved all three profiles |
| 4 | Medium | `build-workload` loaded a tokenizer from `http_model: smolqwen` | ran the command; exit 1 |
| 5 | Medium | 8 serving fields always `None`, so `--require-serving-match` could never pass | constructed the pairing in-process |
| 6 | Low | Admission window never saw pool capacity (built lazily) | `pool_capacity_of` returned `None` before and after `load_tasks` |

### 1 and 2 compound into one dead branch

`peft_helper.validate_legal` raises `LoRA rank 32 is greater than max_lora_rank 16.`
on the first request carrying the `LoRARequest`. Because #2 left
`is_adapter_capability_error` matching nothing real, that refusal was fatal rather
than routed to the `TransformersPolicy` fallback:

- `evaluate --adapter-path` → run dies.
- SFT in-training dev eval → `bench_failed` at every boundary; training completes and
  reports success with **no dev curve at all**.

The fixture that should have caught it trained at `r: 4`, under vLLM's default.

## What changed

**`inference/engine.py`** — `OfflineEngine(max_lora_rank=...)`, passed only on the
LoRA path (vLLM builds no `LoRAConfig` without `enable_lora`). `lora_rank_slot`
rounds up to the `Literal` vLLM accepts; `adapter_rank` reads `r` from the adapter
directory; `widest_adapter_rank` covers a multi-adapter engine. A rank above what was
reserved is refused at registration with a message naming both sides — explicitly
*not* a capability refusal, because a sizing mistake with a one-line fix must not hide
behind a slower path. Markers replaced with the four vLLM 0.26 really raises, each
quoted with its raise site. New `serving_config()` reads dtype, quantization,
speculative decoding, KV budget, batching limits, chunked prefill and prefix caching
off the built engine's resolved `VllmConfig`.

**`training/checkpoint_eval.py`** — SFT's engine reserves `config.lora.r`; no
checkpoint exists at build time to read a rank from.

**`training/grpo.py`** — new `bench_eval_config`, replacing the bare
`resolve("eval")`. Substitutes this run's profile *and* sets `max_seq_length` to
`vllm_max_model_len`, which is the colocated engine's actual bound. Without the second
correction the turn engine admits prefixes vLLM rejects
(`input_processor.py:404`). SFT's `eval_config_for` needs only the first, since its
engine is sized from the same `ProfileConfig`.

**`eval/runner.py`** — serving provenance now comes from `_serving_config(generation)`
instead of seven `getattr(args, ...)` reads of flags deleted in Phase 8. A failed read
logs and records unknown rather than failing a scored run. Endpoint runs record
unknown, which is honest: that config belongs to a process this command cannot
inspect.

**`eval/batched.py` + `adapters/envscaler_heldout.py`** — adapter declares
`pool_capacity` from the profile it will use, so `min(generation_concurrency,
pool_capacity)` binds before the lazy pool exists.

**`cli.py`** — `build-workload` renders with the pinned base model's tokenizer at its
recorded revision. A merged checkpoint carries that tokenizer verbatim, so this is
what the server will see.

## Verification

- `uv run pytest`: **557 passed, 6 skipped** (was 538 collected).
- `make test-ci`: **544 passed, 19 deselected**.
- `uv run pytest -m gpu -rs`: 6 skipped, all on absent vllm.
- `make check`: ruff, ruff format, mypy --strict over 160 files — clean.
- `make smoke`: every stage × profile dry-run.
- `smolqwen build-workload --profile l4`: exit 0, 19.4 MB rendered (was exit 1).
- `git diff --check`: clean. Validation script byte-compiles.

New tests, each failing on the pre-fix code: reserved rank covers `lora.r` (engine
contract + SFT boundary); rank rounding and the not-offered case; over-wide adapter
refused at registration and *not* classified as a capability error; the four real
refusals recognized and four fixable/resource failures still fatal; `serving_config`
read from `vllm_config` and pairing against it; GRPO bench engine bound equals
`vllm_max_model_len` per profile; declared capacity binds the window; workload
tokenizer identity.

## Not changed, deliberately

- Verifier reward semantics, GRPO loss math, `target_modules: all-linear`.
- The nine deleted serving flags stay deleted — recording what ran replaces them.
- `TransformersPolicy` stays as the adapter-on-base path.

## Still open

Everything here is CPU-verified or read from dependency source. The GPU criteria in
`phase-10-gpu-validation-on-l4.md` are untouched: whether vLLM accepts this project's
`all-linear` adapter at rank 32 is now *reachable* rather than blocked, but it is
still unmeasured. `scripts/colab-gpu-validation.py` gained an assertion that the eval
phase's report carries a non-null serving config, so an all-`None` row fails loudly
instead of passing as "no serving config".
