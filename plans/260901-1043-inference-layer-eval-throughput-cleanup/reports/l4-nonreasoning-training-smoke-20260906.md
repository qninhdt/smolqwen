# L4 non-reasoning training smoke — 2026-09-06

## Scope

This run exercises the real SFT → merge → evaluation → difficulty profile → GRPO
path with `enable_thinking: false`. It uses the real Qwen3.5-2B revision and a
real non-reasoning SFT shard; only the number of steps/tasks is bounded for smoke.
No reasoning-mode result is included.

The run is evidence for “the configured path can start and complete a step”, not
evidence that the full 2-epoch SFT plus 200-step GRPO job has finished.

## Runtime identity

| Field | Value |
|---|---|
| GPU | NVIDIA L4, sm89 |
| VRAM | 22.034 GiB |
| Torch | 2.11.0+cu130 |
| Transformers / TRL / vLLM | 5.16.1 / 1.12.0 / 0.26.0 |
| Model | `Qwen/Qwen3.5-2B` |
| Model revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Reasoning | disabled |
| SFT smoke shard | 35 train rows / 4 validation rows; schema `full_trajectory_non_reasoning_v1` |

## Resolved configurations

| Stage | Relevant values |
|---|---|
| SFT | `max_seq_length=32768`, `max_tokens_per_microbatch=32768`, token-budget sampler, `grad_accum=8`, FA2, BF16, Liger FLCE, selective logits, compile off |
| GRPO | `vllm_max_model_len=16384`, `generation_batch_size=16`, `num_generations=4`, `generation_concurrency=8`, KV fraction `0.40`, `grad_accum=8`, `micro_batch=1` |
| Eval smoke | EnvScaler held-out, 1 environment × 1 scenario, 2 turns, `enable_thinking=false`, context 32768 |

The SFT `micro_batch` field does not size padding-free SFT rows; the 32K token
budget does. GRPO does use the profile micro-batch, and the L4 result below makes
the distinction load-bearing.

## Phase results

| Phase | Result | Evidence |
|---|---|---|
| Probe | PASS | L4, FA2, BF16, all mandatory kernel wheels import |
| Kernels | PASS | FA2, causal-conv1d, FLA GDN and Liger forward/backward |
| Data | PASS | Real shard validates against 32K sequence/token caps and non-reasoning semantics |
| SFT | PASS | One full-shape optimizer step with the engine resident and asleep; final bench-enabled run completed with `train_runtime=399.4s`, no training OOM; live `nvidia-smi` sample reached 19.3 GiB of 22.0 GiB |
| Merge | PASS | 2,213,241,664-parameter Qwen3.5 wrapper; processor, image, and video metadata present |
| Base eval | PASS | vLLM path, 196 average generated tokens, score `0.3846`, zero-generation `0` |
| SFT eval | PASS | vLLM path against merged checkpoint, 194 average generated tokens, score `0.3846`, zero-generation `0` |
| Stability | PASS | Concurrency 1 vs 8: same score `0.3846`, score delta `0.0`, no timeout/zero-generation discrepancy |
| Difficulty | PASS | 4 scenarios × 2 rollouts, 336.49s; counts: `always_zero=1`, `band=3`, `always_one=0` |
| GRPO, micro-batch 2 | OOM | After one 594.80s step, old-policy logits attempted another 10.21 GiB with 7.09 GiB free |
| GRPO, micro-batch 1 | PASS | One real optimizer step completed in 594.80s; checkpoint, adapter, optimizer/RNG/scheduler state and completions written; peak about 20.7 GiB, no OOM |
| SFT bench, first implementation | FAIL then fixed | Step 0 passed, but checkpoint-1 `wake_up()` OOMed while the trainer optimizer remained on GPU; the callback recorded the failure and training still finished |
| SFT bench, final implementation | PASS | Trainer model and optimizer state were moved to CPU around each vLLM boundary; step 0 and `checkpoint-1` both scored one dev task with `bench_failed=0` |
| Non-TTY | PASS | 21 plain stdout lines, no carriage-return redraw |
| Full GPU tests after fixes | PASS | `pytest -m gpu`: 11 passed, 1 skipped in 277.85s; the skip is the trainer-envelope test intentionally supplied by the Colab smoke phase |

## Fixes found by the card

1. `torch.cuda.memory_allocated()` is not a valid sleep-release metric for vLLM's
   `CuMemAllocator`: the allocator log freed 17.33 GiB while the PyTorch counter
   stayed at 18.85 GiB. `MemoryWorkerExtension` now reports the worker's
   driver-level footprint from `torch.cuda.mem_get_info()`. The L4 guard measured
   18.87 GiB awake → 1.53 GiB asleep, leaving 20.50 GiB.

2. PEFT text-only adapters used `base_model.model.model.layers.*`, while vLLM's
   released Qwen3.5 wrapper expects `language_model.model.layers.*`. vLLM loaded
   the old adapter and ran LoRA kernels but produced no output/logprob delta. The
   engine now creates a temporary prefixed safetensors view. The rank-32
   `all-linear` adapter probe then passed; the no-op guard remains enabled.

3. Two GPU unit tests were marked `gpu` but left the tiny model/batch on CPU. They
   now move tensors and model to CUDA. The isolated pair passes.

4. A direct ad-hoc pytest wrapper did not prepend `.venv/bin` to `PATH`, so
   FlashInfer reported `FileNotFoundError: ninja`. The official validation runner
   already exports the venv path; no package source build was needed.

5. The first clean GPU-suite run exposed a Python 3.13/Transformers fixture issue:
   `AutoModelForCausalLM` selected the multimodal Qwen3.5 wrapper and read
   `vocab_size` from the wrapper instead of its nested text config. The fixture now
   constructs `Qwen3_5ForCausalLM` from the serialized nested `text_config`; the
   full GPU suite passes with that change.

6. Production YAML defaults for data conversion, evaluation, and GRPO now set
   `enable_thinking: false`; the earlier L4 commands needed explicit overrides,
   which could otherwise make a normal run silently switch back to reasoning.

7. The first bench-enabled SFT run exposed the real shared-card failure: after the
   optimizer step, vLLM sleep had released its own memory but the live trainer still
   occupied the card, so the next wake failed with CUDA OOM. The callback now moves
   the trainer model and optimizer tensors to CPU, wakes and evaluates the saved
   checkpoint, sleeps vLLM again, then restores the trainer state. The final L4 run
   completed both the base anchor and `checkpoint-1` benchmark with no OOM.

8. The same run showed that PEFT `all-linear` also targets Qwen3.5's visual tower.
   The text-only SFT/GRPO configs still declare `all-linear`, but exclude the unused
   visual subtree so the adapter contains only language-model keys and is accepted by
   vLLM's text-only LoRA mapper. The downloaded rank-32 adapter had 372 language keys
   and zero visual keys.

## Training-time estimate

The full raw profile contains 9,022 rows. Using its conservative mean
`conversation_tokens≈7,257` gives about 65.5M tokens. At a 32K token envelope and
`grad_accum=8`, that is roughly 250 optimizer steps per SFT epoch.

Measured extrapolation on this L4:

| Work | Estimate |
|---|---:|
| SFT, 1 epoch | ~16.3 h |
| SFT, configured 2 epochs | ~32.6 h |
| GRPO, configured 200 steps at `micro_batch=1` | ~33.0 h |
| Pure SFT + GRPO | ~65.6 h |
| Practical allowance for JIT, checkpoints, eval and reconnects | ~67–70 h |

This is a step extrapolation, not a completion guarantee. A ±20–30% range is more
honest until a longer post-warm-up sample is collected. A free Colab runtime with
about one hour before reclaim would require roughly 70 resumable sessions, so a
full run there is operationally unattractive.

## Still unresolved

- The full 30-step SFT envelope and the full 200-step GRPO run were not executed.
- Production-cadence SFT in-training benchmark cost (task limit 16 over the full
  save cadence) was not measured. The one-task smoke boundary costs 75.71s for the
  base anchor and 86.19s for `checkpoint-1`; both pass after the offload fix.
- The vLLM process emits `CUDA Error: invalid argument` from `cumem_allocator.cpp`
  while destructing the sleeping engine after successful training/evaluation. It is
  post-artifact cleanup only: the process has no live requests, both sidecars are
  complete, and the VM was stopped. Treat it as a vLLM sleep-mode cleanup defect,
  not as evidence that the training step failed.
- The full three-way throughput split (generation/tokenization/environment), the
  old serial-baseline agreement with per-task disagreements, and injected
  live-pressure failure isolation remain open.
- BFCL was not run; it must remain a single final run after checkpoint selection.
