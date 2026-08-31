# Optimization ledger — Phase 3 SFT

Every toggle in `configs/base/sft.yaml` appears here with the reason it is set the
way it is and the state of its measurement. The point of the file is that a flag
is either **measured** or **off** — a config where everything is true and nobody
knows which one helped is not an optimized pipeline.

The ledger is generated, not transcribed: `smolqwen train-sft` prints
`format_ledger(...)` at startup and writes the same rows into the W&B run config
under `optimization/*`. A toggle that downgrades at runtime (no CUDA, missing
wheel) records *why* in its own detail string, so a run that quietly lost an
optimization is visible in the run config rather than only in the throughput
number.

## Toggles and why

| Toggle | Setting | Why this setting |
|---|---|---|
| `bf16` | on | Base is loaded bf16; the card targets are sm89/sm80, both native. |
| `gradient_checkpointing` | on, `use_reentrant=False` | Activations at the 16,384-token budget cap do not fit on 24 GB even at micro-batch 1, so this is a precondition for the baseline rather than something to sweep. |
| `liger_fused_linear_cross_entropy` | on | **Load-bearing.** Vocabulary is 248,320 with tied embeddings, so a dense `[batch, seq, 248320]` logits tensor is the dominant activation and the first thing to OOM — ahead of the weights. |
| `adapter_dtype` | `bfloat16` | PEFT upcasts adapters to fp32 by default. That default exists for low-bit QLoRA bases; with `all-linear` targets on a bf16 base it forces an upcast/downcast plus an fp32 GEMM at every linear. |
| `attn_implementation` | `flash_attention_2` | Applies to the 6 full-attention layers. The 18 GDN layers use their own `flash-linear-attention` kernels regardless. Downgrades to `sdpa` with a recorded reason when either the wheel or a CUDA device is absent. |
| `regional_torch_compile` | off until measured | GDN layers run Triton kernels that upstream marks `torch.compiler.disable`; compiling through them raises inductor errors. `select_compile_targets` therefore excludes `Qwen3_5GatedDeltaNet` by class name *and* honours `compile_exclude_patterns`, and a single `torch.compile(model)` is never used. |

## Measurements

`micro_batch`, `grad_accum` and `max_seq_length` are this phase's owned profile
fields (see the plan's cross-phase contract table), and `max_seq_length` is
seeded from `budgets.json` — a sweep may lower it, never raise it.

### Epoch-time sweep on one L4 (2026-08-30/31)

Measured by [`scripts/colab-l4-sft-speed.py`](../scripts/colab-l4-sft-speed.py)
on an NVIDIA L4 (22.034 GiB, sm89, torch 2.11+cu130) against 96 real prepared
records from the Phase 2 shard. Every candidate runs in a fresh child process at
the same effective batch of 16; the first optimizer step warms kernels and is
excluded. Throughput is **valid** (unpadded) tokens per second, so a candidate
cannot win by processing more padding, and the epoch estimate is that rate
against the shard's 327,136,547 valid tokens.

| Candidate | valid tok/s | epoch (h) | padding | peak reserved | vs. best |
|---|---:|---:|---:|---:|---:|
| **mb2 + group_by_length + regional compile** | **2043** | **44.5** | 3.6% | 12.66 GiB | — |
| mb2 + grouped + reentrant checkpoint | 2031 | 44.8 | 3.6% | 12.71 GiB | −0.6% |
| mb2 + grouped (no compile) | 2020 / 1991 | 45.0 / 45.6 | 2.8% / 3.6% | 12.40 / 12.67 GiB | −1.2% / −2.6% |
| mb2 + grouped + pad-to-64 | 2017 | 45.0 | 4.1% | 12.67 GiB | −1.3% |
| mb1 + grouped + compile | 1965 | 46.2 | 0% | 8.24 GiB alloc | −3.8% |
| mb1, sequential order | 1631 | 55.7 | 0% | 9.20 GiB | −20.2% |
| mb2, sequential order | 1394 | 65.2 | 10.7% | 16.29 GiB | −31.8% |
| mb4 + grouped + compile | 1425 | 63.8 | 10.7% | 20.66 GiB | −30.3% |
| mb4, sequential order | 1128 | 80.5 | 24.6% | 21.60 GiB | −44.8% |

The committed L4 profile is therefore `micro_batch: 2`, `grad_accum: 8`,
`group_by_length` sampling and regional compile: **44.5 h/epoch, down from 55.7 h
at the previous `micro_batch: 1` default — a 20% cut.**

**Length-grouped sampling is the single largest win, and it is what makes batching
work at all.** These trajectories run 4.4k–15.7k tokens, so a random batch of 2
pads to the longer sample and wastes 10.7% of every step; grouping drops that to
2.8% and turns micro-batch 2 from 31.8% *slower* than batch 1 into 20% faster.
Batch 4 is a hard ceiling for the wrong reason: even grouped it pads 10.7% and
lands at 20.66 GiB, so it pays more memory to go 30% slower.

### Knobs measured and rejected

| Knob | Result | Why |
|---|---|---|
| `gradient_checkpointing: false` | OOM at micro-batch 1 **and** 2 | 21.77 GiB reserved of 22.03 available. Checkpointing is a precondition on this card, not a tunable — this is the measurement the baseline row claimed without evidence. |
| `every_n_layers` 2/3/4 | OOM at both batches, even with `expandable_segments` | The coarsest setting still leaves 12 of 24 layers eager. At micro-batch 1 that peaks at 21.73 GiB against an 8.24 GiB fully-checkpointed baseline, so one eager layer costs >1.1 GiB and the ~9 GiB of headroom buys a handful of layers, never half of them. |
| `use_reentrant: true` | 2031 vs 2043 tok/s | Within noise and strictly worse; the non-reentrant default also composes with compile. Keeping the default. |
| `pad_to_multiple_of: 64` | 2017 vs 2043 tok/s | Fixed-width tiles do not pay for themselves once grouping has already cut padding to 3.6%; it *raises* padding to 4.1%. |
| `compile_mode: reduce-overhead` | Crash | `cudaErrorStreamCaptureInvalidated`: CUDA-graph capture is incompatible with the Liger SwiGLU kernel in the compiled MLP region. Default mode only. |
| dataloader workers / prefetch | Not swept | 6 CUDA-synchronized steps account for 559.3 s of 560.7 s wall — 0.24% of the step is outside the GPU, so there is nothing for the input pipeline to win. |

Regional compile is a real but small win (+2.6% over the same configuration
uncompiled, reproduced across two waves), and it costs ~250 s of one-time warmup
in step 1 — negligible against a 44-hour epoch.

### Batch ceiling probe (2026-08-30)

The pinned Qwen3.5-2B checkpoint was loaded on an NVIDIA L4 (22.034 GiB) with
bf16, LoRA `all-linear` rank 32, gradient checkpointing, FlashAttention 2, and
Liger. Each point took one optimizer step in a fresh process. The complete
evidence and caveats are in
[`plans/reports/qa-260830-l4-batch-limits.md`](../plans/reports/qa-260830-l4-batch-limits.md).

| Workload | Fixed token envelope | Largest pass | First OOM |
|---|---:|---:|---:|
| SFT micro-batch | 16,384/sample | 4 | 8 |
| GRPO generation batch (Transformers generation, not vLLM) | 2,118 prompt + 512 completion | 64 | 128 |
| GRPO trainer micro-batch | 4,096 + 4,096 | 1 | 2 |
| GRPO trainer micro-batch | 2,048 + 2,048 | 2 | 4 |

The SFT ceiling of 4 is a hard measured boundary for the stated cap, but its
peak reserved usage leaves under 1 GiB of headroom; the committed L4 profile
therefore remains conservative at `micro_batch: 1` until a real-shard run is
completed. The GRPO generation row is an upper bound for the tested
Transformers path only; colocated vLLM KV memory still needs a separate sweep.

### Why this host cannot fill them

The development machine has an RTX 3050 Laptop, 3.7 GB VRAM (compute 8.6). The
bf16 weights alone are ~3.8 GB, so the model does not load, let alone train. The
`colab` extra (`liger-kernel`, `flash-attn`, `flash-linear-attention`,
`causal-conv1d`) is likewise not installed here — which is why the toggle
resolvers downgrade with a recorded reason instead of raising, and why the CPU
tests assert the *downgrade path* as well as the enabled one.

To fill the table, on an L4 or A100:

```bash
uv sync --extra colab
smolqwen train-sft --profile l4 --override training.max_steps=30
```

Read `system/seconds_per_mtok` and `gpu/memory_max_allocated_gb` from the W&B
run, then repeat with one toggle changed via `--override`. The budget cap is the
ceiling for `max_seq_length`; `resolve()` raises `ConfigError` if a profile or an
override tries to exceed it.

## Fallback order on OOM

Stated up front so the response to an OOM is not improvised:

1. If it OOMs inside the loss computation at micro-batch 1 with checkpointing on,
   liger is the fix — confirm it is actually enabled (check the ledger row, not
   the YAML).
2. If liger is unavailable, lower `max_seq_length` before touching
   `micro_batch`: the logits tensor scales with sequence length, and batch is
   already 1.
3. If `torch.compile` raises an inductor error naming a disabled region, narrow
   the compiled set through `compile_exclude_patterns` and record what had to go.
   Never remove the upstream `torch.compiler.disable` — those kernels do not
   compile.
