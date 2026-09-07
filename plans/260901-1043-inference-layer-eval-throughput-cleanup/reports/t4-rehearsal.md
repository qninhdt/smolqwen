# T4 rehearsal — measured 2026-09-03

Free-tier Colab T4 (Tesla T4, 14.56 GB, sm75), one session, `colab` CLI. The point
was to find wiring bugs before spending L4 credits on them. It found two.

## What ran

| Step | Result |
|---|---|
| `smolqwen probe` | GPU visible, all four kernel wheels import: `flash_attn`, `causal_conv1d`, `fla`, `liger_kernel` |
| `resolve_dtype()` on sm75 | `float16`, downgrade logged |
| `OfflineEngine` on the released Qwen3.5-2B | built, batched-generated, `generate_ids` aligned, sleep/wake worked |
| `smolqwen evaluate` end to end | `generation_path: vllm` — the batched path, not `TransformersPolicy` |
| CPU suite on a GPU host | 4 real failures out of 16; the rest were missing-file artifacts of a partial upload |

Environment: torch 2.11.0+cu130, vllm 0.26.0, transformers 5.16.1, trl 1.12.0,
Python 3.13.15. `uv sync --locked --extra colab` resolved with no source build.

## Finding 1: a score computed from episodes that never generated

At `max_seq_length: 4096` the report read:

```
score: 0.25255
average_generated_tokens: 0.0
```

Every one of four held-out episodes terminated at admission — the rendered prompt
exceeded the window — and every one was still scored, because EnvScaler's verifier
grades the environment's **final state** and an untouched initial state is a valid
state. Per-task: `('step_cap', 0 turns, 0 env_steps, 0.3846)`.

The zero token average is the tell, but it only *implies* the failure. Nothing named
it and nothing refused.

Sizing, measured from the vendored metadata: tool schemas alone render to ~4,000
tokens median and ~6,800 at the widest environment (`env_61_sft`, 31 tools), before
the system prompt and the task. At 16,384 the same four tasks generated 505–838
tokens each and reached `turn_cap`.

Fixed: `terminal_<reason>_rate` with a denominator in every aggregate, plus a WARNING
when no episode generated at all. Not an exception — `evaluate` is also how a
genuinely mute model gets measured. `configs/profiles/t4.yaml` raised to 16,384 with
the measurement recorded beside it.

## Finding 2: `vllm_kv_fraction: 0.20` cannot load the model

0.20 of 14.56 GB is 2.9 GB; fp16 weights alone are 4.25 GB. The value is correct for
an engine *sharing* the card with a trainer (the in-training eval case) and unusable
for a solo `evaluate`. Annotated in the profile rather than changed, because both
cases are real and the asymmetry is the fact worth recording.

## Two test bugs the GPU host exposed

**`test_the_engine_sleeps_after_every_boundary_including_a_failed_one` passed for the
wrong reason.** It failed the boundary by leaving the checkpoint directory absent, but
`BenchEvalRunner.run` calls `create_adapter` before `engine_source()` — so the adapter
raised first and the engine never woke. Locally the adapter happened to construct
because `artifacts/data/env_split.json` exists; on a host without it the ordering
showed. The failure is now injected in `evaluate_batched`, which is the only point
that runs after the wake, and the missing-checkpoint shape has its own test.

**vLLM cannot load `write_tiny_checkpoint`'s output.** It saves
`Qwen3_5TextConfig` + `Qwen3_5ForCausalLM`, whose `model_type` is `qwen3_5_text`;
vLLM's config registry has only `qwen3_5` and `qwen3_5_moe`, so it routes to the
multimodal path and raises `Expected Qwen3_5Config, found Qwen3_5TextConfig`. The
released model is the wrapper shape (`Qwen3_5ForConditionalGeneration`, `text_config`
+ `vision_config`). A wrapper-shaped fixture then failed on a missing
`preprocessor_config.json`.

**Consequence for `tests/test_vllm_adapter_capability.py`: it could not pass on any
card, including the L4, with the original fixture.** It was a Phase 2 GPU criterion
built on `write_tiny_checkpoint`; the follow-up now gives it a tiny released-shaped
`Qwen3_5ForConditionalGeneration` wrapper and the processor metadata vLLM loads.
The fixture shape is covered by `tests/test_vllm_fixture.py`; the actual adapter
acceptance branch remains a Phase 10 measurement.

## Numbers for Phase 10

Sleep mode on the real model, from vLLM's own accounting:

```
CuMemAllocator: sleep freed 10.73 GiB total (4.33 GiB backed up to CPU, 6.40 GiB discarded)
Sleep mode freed 11.56 GiB, 0.85 GiB still in use
took 4.45 s to fall asleep
```

That is a ~93% release from vLLM's own allocator log, **on sm75 at 0.82 utilisation**.
It is rehearsal evidence only: it does not close the new Phase 2 worker-side
`memory_allocated()` criterion or measure an L4 at the profile's fraction. Note
`torch.cuda.memory_allocated()` in the parent process read 0.00 GB throughout: vLLM
V1 runs the engine in a spawned `EngineCore` process, so the parent's allocator sees
nothing. The memory guard now asks the engine workers for their allocator readings
through `OfflineEngine.memory_allocated_bytes()`; the live L4 measurement is still
pending.

Backends selected on sm75, for contrast with the L4 run: `TRITON_ATTN` (FA2 refused,
`compute capability >= 8` required), `TORCH_SDPA` for the vision encoder,
Triton/FLA for the GDN prefill.

## Follow-up contract verification — 2026-09-03

After the two fixture/runtime fixes above, the same locked T4 environment reran the
GPU-marked vLLM checks with result `s..`:

- A direct `OfflineEngine` construction resolved its dataclass `bfloat16` default to
  `float16` on sm75, then built and generated successfully. This closes the direct
  caller portability bug; it does not make T4 numerically equivalent to an L4.
- vLLM injected `MemoryWorkerExtension` and the worker-side RPC completed without
  insecure callable serialization. The tiny fixture's summed worker reading changed
  from `5,408,295,424` to `5,408,279,040` bytes after sleep (16 KiB, `0.0 MiB`
  rounded), while vLLM's own allocator log reported `3.38 GiB` freed. This validates
  the accounting wiring, not the L4 release/envelope criterion.
- The non-zero `all-linear` adapter registered, but deterministic adapted/base
  probes had no observable token or logprob delta. The central guard raised
  `AdapterCapabilityError`, the explicit signal used to select the Transformers
  fallback; the GPU test recorded that refusal. Unloaded-adapter generation and
  sleep/wake both passed. Adapter acceptance on the target L4 remains unmeasured.

## Operational notes

- `colab exec` gives up after ~120 s. Every real step must be launched detached
  (`setsid nohup … > log`) and polled; the kernel is never what holds the work.
- vLLM V1 spawns `EngineCore`, and `spawn` re-imports `__main__`. A script that builds
  an engine at module level dies with `An attempt has been made to start a new process
  before the current process has finished its bootstrapping phase`. Needs
  `if __name__ == "__main__":`. The installed console script is already guarded; this
  only bites ad-hoc scripts.
- Three sessions were lost mid-run (404/401). A lost session leaves a server-side
  assignment that `colab stop` cannot reach by name, and the next `colab new` fails
  `TooManyAssignmentsError` until it expires — roughly 5–10 minutes.
- Uploading only `src tests configs scripts` makes ~12 tests fail on absent files
  (`serving/proxy.conf`, `docker-compose.yml`, `.gitignore`, `.github/`,
  `third_party/gorilla`, `artifacts/data/*.json`). Ship the tracked artifacts too.

## Still needs the L4

Unchanged by this session: the 32K envelope beside a sleeping engine, the
non-monotonic asleep reading on an L4, the 10% cost bound for both stages, baseline
agreement, and the Colab non-TTY rendering check. The T4
cannot run `train-sft` at all — FA2 is Ampere and newer, and without a varlen kernel
nothing reads `cu_seq_lens`.
