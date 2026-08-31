# L4 batch-limit probe — 2026-08-30

## Result

The real `Qwen/Qwen3.5-2B` checkpoint (revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`) was loaded on an NVIDIA L4 with
22.034 GiB visible VRAM. Each candidate ran one optimizer step in a fresh
process with bf16 weights, LoRA `all-linear` (rank 32), gradient checkpointing,
FlashAttention 2, and Liger fused loss. CUDA was `torch 2.11.0+cu130`, compute
capability 8.9.

| Probe | Envelope | Largest pass | First failure | Peak at largest pass |
|---|---|---:|---:|---:|
| SFT micro-batch | 16,384 tokens/sample | **4** | 8 — CUDA OOM | 21.133 GiB reserved (20.674 allocated) |
| GRPO generation batch | 2,118 prompt + 512 completion tokens; `num_generations=4`, train microbatch 1 | **64** | 128 — CUDA OOM | 19.463 GiB reserved (16.372 allocated) |
| GRPO train microbatch | 4,096 prompt + 4,096 completion tokens; generation batch 8 | **1** | 2 — CUDA OOM | 21.480 GiB reserved (17.483 allocated) |
| GRPO train microbatch | 2,048 prompt + 2,048 completion tokens; generation batch 8 | **2** | 4 — CUDA OOM | 18.330 GiB reserved (17.485 allocated) |

Additional boundary evidence:

- SFT batch 1/2/4 passed at the 16,384-token cap; batch 8 failed while trying
  to allocate 512 MiB with only 199 MiB free.
- GRPO trainer microbatch 1 at an 8,192+8,192-token envelope failed while
  trying to allocate 7.58 GiB; only 2.38 GiB was free.
- The GRPO generation probe passed batch 4/8/16/32/64 and failed at 128.

## Interpretation for the profiles

- The measured SFT ceiling at the current budget cap is `micro_batch=4`, but it
  leaves less than 1 GiB of reserved headroom. Keep the committed L4 value at
  `micro_batch=1` (or at most 2 for a deliberate throughput experiment) until
  a short run over real shards confirms allocator stability.
- The current GRPO settings (`num_generations=4`,
  `generation_concurrency=8`, `active_pool_multiplier=2`) imply a
  `generation_batch_size=16`. That is below the 64-row pass in the
  Transformers-generation envelope, but the probe did **not** initialize
  colocated vLLM; it is not evidence that vLLM KV cache can be raised to 64.
- For trainer-side GRPO backward, use microbatch 1 at an 8k-token envelope, or
  microbatch 2 only when the effective prompt+completion envelope is about 4k
  tokens. A full 16k GRPO completion envelope does not fit even at microbatch 1.

## Scope and limitations

This is a sizing probe, not a completed training run. Inputs use deterministic
valid vocabulary ids (SFT) or a deterministic rollout callback (GRPO
microbatch), so no EnvScaler environment, W&B/HF push, checkpoint resume,
real-shard streaming, or colocated-vLLM KV allocation was exercised. The
generation-batch row uses TRL's normal Transformers generation path to make the
batch dimension measurable; the production async/vLLM path still needs its
throughput/KV sweep. Therefore these pass/OOM boundaries are empirical limits
for the stated envelopes, not a guarantee that an overnight full SFT or GRPO
run cannot fail for another reason.

## Reproduction

The reusable runner is [scripts/colab-l4-batch-sweep.py](../../scripts/colab-l4-batch-sweep.py).
It archives the checkout, installs the locked `colab` extra once, runs each
candidate in a new child process, writes a machine-readable JSON result, and
does not continue after the first failed point in a monotonic sweep. Both
Colab sessions used for this report were terminated; `colab sessions` returned
no active sessions after collection.
