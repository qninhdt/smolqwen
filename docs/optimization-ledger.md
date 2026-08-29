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

Empty by design rather than by omission: filling these needs a card that can hold
the 2B model, and the sweep writes them here as it runs. `micro_batch`,
`grad_accum` and `max_seq_length` are this phase's owned profile fields (see the
plan's cross-phase contract table), and `max_seq_length` is seeded from
`budgets.json` — a sweep may lower it, never raise it.

| Toggle | s/Mtok before | s/Mtok after | peak VRAM before | peak VRAM after | decision |
|---|---|---|---|---|---|
| baseline (bf16 + checkpointing) | — | — | — | — | pending L4 sweep |
| + liger fused CE | — | — | — | — | pending L4 sweep |
| + bf16 adapters | — | — | — | — | pending L4 sweep |
| + regional compile | — | — | — | — | pending L4 sweep |

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
