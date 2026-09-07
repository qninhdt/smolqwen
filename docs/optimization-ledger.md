# Optimization ledger — full-trajectory SFT

This document is the authority for the current SFT objective. It supersedes the
2026-08-30/31 numbers derived from per-user-turn segmented shards and padded
fixed-row batches. Those measurements (including 16K samples, `micro_batch: 2`,
2,043 valid tokens/s, and 44.5 h/epoch) do not measure the current training
workload and must not be used to size it.

## Current contract

- One released teacher trajectory produces one schema-v2 record. With the selected
  non-reasoning contract (`data.enable_thinking: false`), tool calls and answer text
  remain in the context while teacher reasoning is omitted; system/user/tool-
  observation tokens are masked.
- The initial cap is 32,768 tokens per trajectory. A longer trajectory is skipped
  whole during conversion; it is never split to manufacture extra samples.
- A padding-free micro-batch is a single flattened token array with complete
  trajectory boundaries. Its maximum is 32,768 total tokens, so the number of
  trajectories in a batch is variable.
- The loss is normalized by the actual count of supervised assistant tokens, not
  by a fixed row count or the 32K token envelope.
- The trainer rejects the old segmented shard schema before loading a model.

## Runtime paths

`train-sft` resolves attention and dtype from the card it is on. There are two
attention implementations and no third:

| Card | Attention | Batch shape | Dtype |
|---|---|---|---|
| Ampere-plus (sm80+) | `flash_attention_2` | padding-free, `cu_seq_lens` boundaries | BF16 |
| Turing (sm75) | `sdpa` | right-padded rows with `attention_mask` | FP16 |

FlashAttention-2's kernels are Ampere-and-newer, so on sm75 having the wheel
installed changes nothing — the capability number decides. `sdpa` is a correct
attention path rather than an approximation, and it reads the padding mask
natively, so the padded collator supplies the mask instead of working around a
kernel that cannot take one. Only FA2 consumes the `cu_seq_lens` boundary
metadata, so the padding-free sampler and collator are selected by the attention
decision rather than by a separate switch. On sm75 the token sampler budgets
`batch_size * max_row_length`, which is the dense tensor that actually executes.

BF16 tensor cores also arrive with Ampere. Torch emulates BF16 on Turing rather
than refusing it, which would be slower than FP16 *and* not the numerics an
Ampere run produced, so sm75 resolves to FP16 with the downgrade recorded in the
ledger. FP16 keeps LoRA parameters in FP32 because `GradScaler` rejects FP16
gradients. T4 measurements are a separate experiment and do not transfer to the
L4/A100 envelope.

One extra covers every GPU:

```sh
bash scripts/setup_colab.sh
```

`flash-attn` is installed by that extra and simply goes unused below sm80. The
GDN mixer, causal convolution and fused loss kernels have no usable fallback at
this model's sizes, so their absence is a setup failure on any card.

## Toggle status

| Toggle | Setting | Status |
|---|---|---|
| `bf16` | on | Honoured on Ampere-plus; sm75 resolves the effective run to FP16 because it has no BF16 tensor cores. |
| `gradient_checkpointing` | on, non-reentrant | Required for the initial 32K attempt; final memory headroom is unmeasured. |
| `liger_fused_linear_cross_entropy` | on | Required to avoid materializing the large vocabulary logits activation. |
| `selective_logit_loss` | on | Padding-free SFT gathers supervised next-token positions through Qwen3.5's `logits_to_keep` before Liger FLCE. Padded runs refuse the gather because one index cannot describe different rows safely. |
| `adapter_dtype` | `bfloat16` | Default config; BF16 adapters match the BF16 base, while an FP16 run keeps LoRA params in FP32 because GradScaler rejects FP16 gradients. |
| `attn_implementation` | `flash_attention_2` | Resolved per card: FA2 on sm80+, `sdpa` below it. An explicit `sdpa`/`eager` request is honoured as-is. |
| `regional_torch_compile` | off | Same-shape L4 measurement was faster with compile off and it avoids a long Inductor startup on reclaimable Colab VMs. |

## Training and evaluation boundary

SFT is train-only, matching upstream EnvScaler's SFT setup. `prepare-sft` emits
only `train.jsonl`; the trainer logs training loss and throughput, while BFCL
multi-turn evaluation belongs to the GRPO `bench_eval` callback and the final
explicit `evaluate` command.

The conversion keeps one complete released trajectory per record. Rows over the
profile cap are skipped as whole rows, and the trainer rejects older segmented
schemas before loading a model. No validation shard or SFT benchmark callback is
maintained.

Qwen3.5's released checkpoint wraps a visual tower beside the language model.
`all-linear` remains the configured LoRA target, but the text-only SFT and GRPO
builders exclude that unused `visual` subtree; otherwise a saved adapter contains
keys that vLLM's text-only LoRA mapper rejects.

## T4 fallback evidence

The T4 route is executable but is not a 32K replacement. Its checked-in profile
uses a 16K sequence cap and a 16K dense padded-token envelope; whether the
current all-linear LoRA setup fits in 14.56 GiB still requires a live T4
measurement. A valid T4 run must record FP16, the padded sampler budget, finite
loss, and a short unequal-row forward/backward smoke step. The original package
has passed a live T4 D64 forward/backward probe, but its unpatched D256 backward
path failed the shared-memory limit; the patched D256 and SFT claims remain
pending until Colab assigns another T4.

## Required L4 evidence before an SFT run is accepted

The local development machine has no CUDA L4 nor the required fused attention,
linear-attention, and causal-convolution kernels. CPU tests validate record,
sampler, and loss-accounting invariants only; they cannot prove recurrent or
causal-convolution boundary isolation.

On an L4 with the `colab` dependencies installed, the validation sequence is:

1. Run the padding-free equivalence probe in
   [`scripts/colab-l4-batch-sweep.py`](../scripts/colab-l4-batch-sweep.py). It
   compares padded and flattened logits, supervised loss, and LoRA gradients,
   then mutates trajectory A and requires trajectory B to remain unchanged.
2. Run a 32K one-step probe with a worst-case 32K trajectory and mixed shorter
   trajectories. Record allocated/reserved/peak VRAM and retain operational
   headroom.
3. Run at least 30 real-shard steps. Record supervised tokens/s, total tokens/s,
   finite loss, compile behavior, and stable post-warm-up memory.
4. With `bench_eval.enabled: true`, confirm `sleep()` returns enough that the 32K
   envelope survives, that the engine is asleep during training steps (a
   non-monotonic worker-side driver-footprint reading), and that the summed
   `sft/bench_wall_s` stays at or under 10% of training wall time.

If any L4/A100 boundary check fails, padding-free must remain disabled for that
hardware path. If the 32K envelope lacks safe headroom, measure 24,576, record
the evidence, and regenerate the full-trajectory artifacts with that cap. Do not
infer either decision from the previous segmented benchmark or from a T4 run.
