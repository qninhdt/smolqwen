---
title: smolqwen — brainstorm contract
date: 2026-08-28
status: accepted
type: brainstorm
---

# smolqwen — Brainstorm Contract

Small, cost-efficient Qwen-based tool-use assistant: post-train Qwen3.5-2B with
reasoning SFT + stateful agentic GRPO on executable environments with verifiable
rewards, then self-host it on an optimized vLLM stack. Two subsystems, one
lifecycle.

## Outcome

A repo that delivers two verifiable result tables.

**Post-training** — Qwen3.5-2B, LoRA, three checkpoints:

| | Base | SFT | SFT+RL |
|---|---|---|---|
| BFCL-v3 `multi_turn_base` | | | |
| BFCL-v3 `multi_turn_miss_func` | | | |
| BFCL-v3 `multi_turn_miss_param` | | | |
| BFCL-v3 `multi_turn_long_context` | | | |
| BFCL Multi-Turn Overall | | | |
| EnvScaler held-out reward | | | |
| Invalid tool-call rate | | | |
| Avg. env steps | | | |

Primary criterion: `BFCL-MT(SFT) > BFCL-MT(Base)` and
`BFCL-MT(SFT+RL) > BFCL-MT(SFT)`. Confirmation: held-out EnvScaler reward rises
with RL.

**Serving** — final checkpoint on an OpenAI-compatible vLLM endpoint:

| config | TTFT p50/p95 | TPOT p50/p95 | tok/s | VRAM | BFCL-MT |
|---|---|---|---|---|---|
| BF16 baseline | | | | | |
| + tuned sweep | | | | | |
| + MTP-1 spec decode | | | | | |
| + FP8 (L4) / int4 (A100) | | | | | |

Quantized rows carry a BFCL-MT re-eval so speed claims are paired with quality.

## Constraints

**Hardware.** Colab only. Single GPU. Two profiles: L4 24 GB and A100. Both
profiles use LoRA — the profiles differ in batch/seq/rollout budget and in
serving quantization strategy, not in finetuning method. L4 (sm89) has native
FP8; A100 (sm80) does not, so A100 serving uses AWQ/GPTQ int4 or BF16. Which GPU
to actually train on is a **project result**, not an input: run a bounded probe
on both (episodes/hour, $/1k episodes), then commit to one for the full run.

**No external LLM in RL.** RL rollout is `Qwen3.5-2B ↔ Python environment ↔
Python verifier`. No user simulator, no LLM judge, no teacher in the loop.
EnvScaler's Non-Conversation setting satisfies this and is what the paper's own
RL experiments use. External LLMs are permitted only in evaluation where the
benchmark requires them.

**Reward.** Unchanged from the paper: `R = (checkpoints passed) / K` over Python
boolean functions on the final environment state. Invalid tool-call rate and step
count are logged as metrics, not folded into reward.

**Model.** `Qwen/Qwen3.5-2B` only. Hybrid architecture: 24 layers, 18 linear
(Gated DeltaNet) + 6 full-attention on a `full_attention_interval: 4` pattern,
`head_dim` 256, vocab 248,320, `mtp_num_hidden_layers: 1`, native 262k context.
Advertised context must be capped explicitly for training — 262k is not a usable
KV budget under colocated vLLM.

**Stack.** HuggingFace TRL (`GRPOTrainer`, `SFTTrainer`) + transformers, kernel
libs via prebuilt wheels (flash-attn, flash-linear-attention, causal-conv1d,
liger-kernel), wandb, vLLM. Pin torch to the version the kernel wheels were
compiled against; a newer torch breaks their ABI.

**Artifacts.** LoRA adapters + eval results to a private HF Hub repo on every
save; W&B for metrics and resume state. Colab VMs are reclaimed without warning,
so no state may live only on the VM.

**Attribution.** README and methods state plainly that training uses executable
environments and public reasoning/RL assets released by EnvScaler. EnvScaler is
a training asset, not the project's identity.

## Non-goals

- **SkelBuilder / ScenGenerator.** No environment synthesis, no scenario
  generation, no verifier synthesis. All 191 environments, 4.7K SFT scenarios,
  2.5K RL scenarios, and 9,022 SFT trajectories are downloaded as released.
- **Conversation-mode RL.** Conv trajectories are used for SFT (static data, no
  API calls); RL stays Non-Conv so no user simulator ever enters a rollout.
- **A second model arm.** No Qwen3-1.7B reproduction control.
- **ACEBench-Agent and τ-bench in v1.** The evaluation layer is built with a
  benchmark-adapter boundary so both can be added later without touching the
  training code, but neither is implemented now.
- **BFCL v4 agentic categories.** `web_search_*` needs a SerpAPI key and
  `memory_*` tests a capability EnvScaler does not train. Multi-turn only.
- **A separate mock assistant backend.** The serving deliverable is a normal LLM
  endpoint, like any hosted model API. No CRM/ticket/calendar mock, no bespoke
  agent harness.
- **Custom GDN kernel work.** No Nsight-driven Triton tuning.
- **Kubernetes, frontend, auth, databases, vector stores.**

## Acceptance criteria

1. `smolqwen` CLI runs the full pipeline from config: data prep → SFT → GRPO →
   eval → serve. Colab notebooks stay thin wrappers around it.
2. Both GPU profiles exist as configs and both were probed with recorded
   episodes/hour before the full training run started.
3. RL rollouts are genuinely interactive: reasoning → tool call → real Python
   `env.step()` → observation → next reasoning. No flattened trajectory
   generation anywhere.
4. A test proves environment state isolation: two instances from the same
   `init_config`, mutate one, assert the other is unchanged. If this leaks
   between rollouts in a GRPO group, every number in the project is wrong and
   nothing reports an error.
5. A test proves the verifier runner: known-correct final state → 1.0, known
   partial → the right fraction, initial state → appropriately low.
6. A test proves chat-template rendering preserves reasoning at exactly the
   positions the training format assumes, and that tool-call
   parse/serialize round-trips.
7. Three checkpoints evaluated under identical decoding, system prompt, tool
   harness, and step limits.
8. Rollout throughput A/B recorded: baseline vs async scheduler, in
   episodes/hour — not GPU utilization percent.
9. `docker compose up` serves the final checkpoint on an OpenAI-compatible
   endpoint. Colab runs the same image definition via tunnel.
10. `ruff` + `mypy --strict` + `pytest` green in CI (CPU-only).

## Decisions taken

| Question | Decision | Why |
|---|---|---|
| Model arms | Qwen3.5-2B only | Hybrid linear-attention is the point of the project; a control arm doubles Colab cost |
| EnvScaler code | New repo, EnvScaler as read-only `third_party/` | Own code stays separable from upstream; ROLL/LlamaFactory/SkelBuilder are never used |
| SFT data | Conv + Non-Conv | Conv trajectories are static files — using them costs no API. Paper Table 6: Full (37.00) > Non-Conv (35.75) > Conv (35.50) on BFCL-MT overall |
| SFT split | At user-message boundary only | The Qwen3.5 template's `last_query_index` scan skips `<tool_response>` pseudo-user turns, so reasoning IS retained across a tool-calling chain. Splitting per tool step would multiply samples ~13× for supervision the first sample already carries |
| Finetuning | LoRA on both profiles | L4 must share VRAM with colocated vLLM during GRPO; identical method keeps the two profiles comparable |
| RL rollout | Go straight to async `rollout_func` | TRL's built-in `_tool_call_loop` iterates turn-synchronously over the whole batch — one slow episode holds the batch. `rollout_func` + `env_mask` is the documented escape hatch and stays inside TRL |
| Reward | Paper's checklist fraction, unmodified | Comparable to published numbers; penalties would change the objective |
| Ablations | Base / SFT / SFT+RL only | Direct-RL and env-scaling curves each cost another train+eval cycle |
| Eval v1 | BFCL-v3 `multi_turn_*` | Where the paper shows the largest small-model gain; adapter boundary keeps ACEBench/τ-bench addable |
| Verifier isolation | Process worker + per-call timeout | Needed anyway for async rollout; a hanging `check_func` must not kill an overnight run |
| Serving workload | Standard vLLM bench datasets | Treat it as an ordinary chatbot/agent endpoint: `sharegpt`/`random` for comparability, `prefix_repetition` for prefix-cache measurement, `bfcl` for tool-calling traffic shape |
| Serving deliverable | Docker Compose is the source of truth | One definition; Colab runs the same image via tunnel |
| Artifacts | Private HF Hub + W&B | Survives VM reclamation without Drive quota limits |
| Phasing | data → SFT → RL → serving | Serving needs a real checkpoint to measure |

## Evidence gathered

- Paper: arXiv 2601.05808, Findings of ACL 2026. Downloaded and read. RL uses
  Reinforce++ under Non-Conv; 140 envs for SFT / 51 for RL; official RL config
  is 8 GPUs, 32k sequence, 40 actions/traj, `train_group_size: 8`,
  `rollout_batch_size: 512`. Qwen3-1.7B BFCL-MT: 9.75 → 18.13 → 23.00.
  ACEBench-Agent: 31.95 → 43.61 → 50.00. τ-bench: 12.50 → 17.44 → **16.28**
  (RL regresses at 1.7B — the paper attributes it to weak exploration in small
  models).
- Repo: `RUC-NLPIR/EnvScaler` cloned. RL side is a ROLL graft (gem-registered
  `envscaler_non_conv_env` / `envscaler_conv_env`); SFT side is LlamaFactory
  with two preprocessing steps. Ships `191_env_metadata.json` (20.9 MB, contains
  `env_class_code`) and `envscaler_rl_scenario_metadata.json` (40.5 MB). Also
  ships a `bfcl_env` adapter with the eight BFCL API classes and reward — worth
  reading before writing our own.
- BFCL: v4 is current. It layers agentic categories (`web_search_*`,
  `memory_*`, `format_sensitivity`) on top of v3 rather than replacing it;
  `multi_turn_base/miss_func/miss_param/long_context` remain scoring categories
  and results still carry the `BFCL_v3_` prefix. BFCL multi-turn drives user
  turns from a static `questions` list, not a live simulator.
- TRL 1.12.0 source read. `environment_factory` (experimental,
  requires transformers ≥ 5.2.0) builds a stateful object per rollout, exposes
  public methods as tools, `reset` required, optional `get_reward` owns reward.
  `_tool_call_loop` batches turn-synchronously — the barrier we need to avoid.
  `rollout_func` must return `prompt_ids`/`completion_ids`/`logprobs`; extra
  keys forward to reward funcs, and `env_mask` is popped and used as the
  internal `tool_mask` (1 = model tokens, 0 = external).
- Qwen3.5-2B `config.json` and `chat_template.jinja` read directly. Template
  confirms per-turn reasoning retention past the last real user query, and
  renders tool responses as `<tool_response>`-wrapped user messages.
- Qwen3.5's hybrid linear attention limits speculative decoding to **MTP-1**:
  `conv_states`/`recurrent_states` have no sequence dimension, so a partially
  accepted draft cannot be rolled back. One-token drafts are all-or-nothing.
- Dataset sizes: `EnvScaler-SFT-Traj-9K` ships
  `envscaler_sft_traj_9k_metadata.json` (701 MB) and a pre-templated
  `mask_history_all_traj-9K_apply_qwen3_template.json` (4.59 GB). `tools`,
  `messages`, `user_messages` are JSON **strings** needing `json.loads`.
- Latest versions: trl 1.12.0, vllm 0.28.0, transformers 5.16.1,
  liger-kernel 0.8.2.
- `vllm bench serve` datasets include `sharegpt`, `random`,
  `prefix_repetition`, `bfcl`, `spec_bench`, `mt_bench`, `custom`, and more;
  `vllm bench sweep` provides parameter sweeps with Pareto plotting.

## Risks

**RL may not beat SFT.** Load-bearing assumption: a 2B hybrid model has enough
exploration capability for GRPO to extract signal. The paper's own 1.7B
regressed on τ-bench and gained least from RL of the three sizes it tried. This
fails first if held-out EnvScaler reward stays flat across GRPO steps. Mitigation
is diagnostic, not corrective: instrument reward variance per scenario early and
prioritize scenarios where `0 < P(success) < 1`, which is where GRPO advantage
carries information. Cheapest to abandon because it is the last training stage —
Base vs SFT still stands on its own.

**Single-L4 GRPO feasibility is unproven.** The official recipe is 8 GPUs at 32k
context. Nothing published shows EnvScaler GRPO on one 24 GB card. Fallback order
on OOM: fewer generations per group, then shorter context, then fewer env steps,
then a smaller vLLM KV budget — never zero reasoning.

**Trajectory length.** Non-Conv averages ~13 steps and the paper caps at 32k
tokens. Filtering to a short-trajectory subset means training on a
self-defined subset of the distribution, and that must be stated rather than
glossed. Profile the real length distribution before choosing the cap.

**`exec()` on dataset-supplied Python.** Environment classes and verifier
functions arrive as source strings from a third-party dataset. Process isolation
plus timeouts is the accepted posture — sufficient for the threat model (MIT
dataset from a university lab, no secrets in the worker), and process workers are
required for async rollout regardless.

**MTP-1 acceptance rate is unknown.** Agent traffic is heavy in structured
tool-call JSON, which *may* draft well, but a single-token draft caps the
achievable speedup. Measure before claiming anything.

## Open questions

- Paper says 140 SFT / 51 RL environments; the HF data release describes
  141 / 50. Treat the public release as source of truth and record exact
  `env_id` splits in an experiment manifest.
- Does the 4.59 GB pre-templated `mask_history` file already encode a split
  granularity that conflicts with the user-message-boundary decision? Read the
  metadata file instead and template it ourselves if so.
- Which A100 variant does Colab allocate (40 GB vs 80 GB)? Changes the RL
  batch ceiling.

## Handoff

Chosen direction: sequential build — data pipeline and profiler, then SFT with
Base/SFT eval, then async rollout plus GRPO with SFT+RL eval, then serving.
Next workflow: the plan skill, then `/ak:cook`.
