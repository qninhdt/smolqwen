---
title: "smolqwen — agentic post-training and optimized serving for Qwen3.5-2B"
description: "Post-train Qwen3.5-2B with reasoning SFT and stateful agentic GRPO over executable environments with verifiable rewards, then self-host it on an optimized vLLM stack."
status: in-progress
priority: P1
effort: "32 phase-days (~6.5w at 5d/w, before Colab session loss)"
tags: [post-training, agentic-rl, grpo, trl, vllm, serving, qwen]
created: 2026-08-28
---

# smolqwen — agentic post-training and optimized serving for Qwen3.5-2B

## Overview

Two subsystems, one model lifecycle. Post-train `Qwen/Qwen3.5-2B` into a
multi-step tool-use agent via reasoning SFT then online GRPO against executable
stateful Python environments scored by deterministic verifiers, and serve the
resulting checkpoint on a measured, tuned vLLM stack.

Training assets — 191 executable environments, 4.7K SFT scenarios, 2.5K RL
scenarios, 9,022 teacher trajectories — are consumed as released by
[EnvScaler](https://github.com/RUC-NLPIR/EnvScaler) (arXiv 2601.05808, Findings
of ACL 2026). No environment synthesis, no scenario generation, no verifier
synthesis. EnvScaler is a training asset in read-only `third_party/`; every line
of pipeline code is ours.

Accepted contract:
[brainstorm-260828-1029-smolqwen-contract.md](../reports/brainstorm-260828-1029-smolqwen-contract.md)

The single hardest engineering problem is Phase 6: agentic RL rollout is
interactive, so the GPU idles while Python environments execute. Measured on a
prior probe of this exact workload: A100 mean utilization 49.2%, L4 mean 64.3%
— the stronger GPU generates faster and therefore waits longer. TRL's built-in
`_tool_call_loop` batches turn-synchronously across the whole batch, so a single
slow episode holds every other episode. The fix is an async ready-queue rollout
behind TRL's `rollout_func` hook.

Single-turn RLVR (math, code) does not have this problem — one generation per
prompt, and vLLM's continuous batching keeps the GPU near-saturated with no
tuning. The difficulty here is specific to holding three constraints at once:
multi-turn agentic rollout, one Colab GPU, and LoRA. Any two are routine. All
three together have no settled recipe, which is why TRL marks both
`rollout_func` and `environment_factory` experimental and ships its async
multi-turn path as a separate `AsyncGRPOTrainer` rather than folding it into
`GRPOTrainer`.

## Goals

| # | Goal | Priority |
|---|------|----------|
| 1 | `Base < SFT < SFT+RL` on BFCL-v3 Multi-Turn, all three checkpoints evaluated under identical decoding, prompt, tool harness, and step limits | P1 |
| 2 | Genuinely interactive RL rollouts: reasoning → tool call → real `env.step()` → observation → next reasoning, with proven per-rollout state isolation | P1 |
| 3 | Reward from Python verifiers on final environment state, unmodified from the paper: `R = (checkpoints passed) / K` | P1 |
| 4 | Async rollout throughput A/B measured in episodes/hour against the TRL turn-synchronous baseline | P1 |
| 5 | `docker compose up` serves the final checkpoint on an OpenAI-compatible endpoint; Colab runs the same image | P1 |
| 6 | Serving sweep reports TTFT/TPOT/throughput/VRAM per config, with a BFCL re-eval paired to every quantized row | P1 |
| 7 | L4-vs-A100 decided by recorded measurement, not assumption: SFT profile from Phase 3's throughput/VRAM sweep, RL profile from Phase 6's episodes/hour A/B | P2 |
| 8 | Zero external LLM anywhere in an RL rollout — no user simulator, no judge, no teacher | P1 |

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Phase 1: Project scaffold, GPU profiles, CLI](./phase-01-start.md) | In progress — code, setup, and CI definitions CPU-verified; remote clean-checkout CI, GPU probe artifacts, hashed wheel URLs, and environment hygiene pending |
| 2 | [Phase 2: Data pipeline and trajectory profiler](./phase-02-data-pipeline-and-trajectory-profiler.md) | Complete — real profile and conversion run |
| 3 | [Phase 3: Reasoning SFT](./phase-03-reasoning-sft.md) | Code complete, CPU-verified; GPU sweep and training run pending a card |
| 4 | [Phase 4: Environment runtime and verifier](./phase-04-environment-runtime-and-verifier.md) | Complete — environment runtime and verifier CPU-verified |
| 5 | [Phase 5: Evaluation harness and baseline results](./phase-05-evaluation-harness-and-baseline-results.md) | Code complete, CPU-verified; Base/SFT GPU results pending Phase 3 checkpoint |
| 6 | [Phase 6: Async rollout scheduler](./phase-06-async-rollout-scheduler.md) | In progress — code/CPU verification and production trainer wiring complete; real-L4 rollout smoke is proven, while full profile measurements remain pending |
| 7 | [Phase 7: Agentic GRPO and final results](./phase-07-agentic-grpo-and-final-results.md) | In progress — trainer/reward/curriculum/resume code and CPU smoke complete; target-GPU profiling, training, and final evaluation pending |
| 8 | [Phase 8: vLLM serving optimization](./phase-08-vllm-serving-optimization.md) | In progress — code complete, CPU-verified, and bounded real-L4 serving/benchmark/eval smoke passed; sweeps and live Compose/tunnel evidence pending |

Dependency chain is linear except Phase 4, which only needs Phase 1: it can be
built while SFT trains. Phase 5 needs 3 (SFT checkpoint) and 4 (held-out
environment reward). Phase 6 needs 4. Phase 7 needs 5 and 6. Phase 8 needs 7.

## Architecture

```
third_party/EnvScaler (read-only)          HF datasets
  191_env_metadata.json                      EnvScaler-SFT-Traj-9K
  envscaler_rl_scenario_metadata.json        EnvScaler-RL-Scenario
        │                                         │
        └────────────────┬────────────────────────┘
                         ▼
              src/smolqwen/data/          ── Phase 2
              profiler, converter, splits
                         │
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
  training/sft.py  ── Phase 3      environments/  ── Phase 4
  TRL SFTTrainer                   env registry, worker pool,
  LoRA, per-user-turn split        verifier runner, state isolation
        │                                  │
        │                    ┌─────────────┴──────────────┐
        ▼                    ▼                            ▼
  qwen3.5-2b-sft      rollout/  ── Phase 6         eval/  ── Phase 5
        │             ready-queue scheduler        BFCL adapter,
        │             rollout_func + env_mask      held-out env reward
        │                    │                            │
        └────────┬───────────┘                            │
                 ▼                                        │
        training/grpo.py  ── Phase 7                       │
        TRL GRPOTrainer, colocate vLLM ────────────────────┤
                 │                                        │
                 ▼                                        ▼
        qwen3.5-2b-sft-grpo ──────────────────► Base/SFT/SFT+RL table
                 │
                 ▼
        serving/  ── Phase 8
        vLLM + sweep + MTP-1 + quantization
        docker compose (source of truth) │ Colab via tunnel
```

## Key decisions

Carried from the accepted contract. Do not re-litigate during implementation.

| Decision | Rationale |
|---|---|
| `Qwen/Qwen3.5-2B` only, no second model arm | Hybrid linear attention is the point; a Qwen3-1.7B control doubles Colab spend |
| New repo, EnvScaler as read-only `third_party/` | Own code stays separable; ROLL, LlamaFactory, SkelBuilder, ScenGenerator are never used |
| LoRA on both GPU profiles | L4 must share VRAM with colocated vLLM during GRPO; identical method keeps profiles comparable |
| SFT splits at real user-message boundaries only | The Qwen3.5 template's `last_query_index` scan skips `<tool_response>` pseudo-user turns, so reasoning survives a whole tool-calling chain. Splitting per tool step multiplies samples ~13x for supervision the first sample already carries |
| Conv + Non-Conv trajectories for SFT | Conv trajectories are static JSON — zero API cost. Paper Table 6 (Qwen3-8B, BFCL-MT overall): Full 37.00 > Non-Conv 35.75 > Conv 35.50 |
| RL is Non-Conv only | Conv mode requires an LLM user simulator in the rollout loop; the paper's own RL runs are Non-Conv |
| Straight to async `rollout_func`, no `environment_factory` production run | TRL's `_tool_call_loop` is turn-synchronous by construction. `environment_factory` is still built as the correctness oracle and A/B baseline, but the real trainer uses `rollout_func` + `env_mask` |
| The two rollout paths are two separately-constructed trainers | `env_mask` is read only on the `else` branch of `if self.tools:` (`grpo_trainer.py:2270,2289-2291`), and `environment_factory` populates `self.tools` from every env method (`:649,666`). A trainer holding both silently discards our mask and rebuilds `tool_mask` as all-ones, training on observations. Production: `rollout_func` with `tools=None, environment_factory=None`. Oracle: `environment_factory` with no `rollout_func` |
| Turn barrier removed *within* one `rollout_func` call, not across training steps | `rollout_func(prompts, trainer)` is synchronous, receives exactly `generation_batch_size` group-contiguous prompts, and must return rows positionally aligned 1:1 — `_calculate_rewards` sizes from `len(prompts)` (`:1633`), zips `strict=True` (`:1661`), and advantages come from `rewards.view(-1, num_generations)` (`:2787`) with no `group_id`. There is no cross-step buffer TRL will accept and no way to pull extra rows from the sampler |
| `AsyncGRPOTrainer` rejected, not overlooked | `trl.experimental.async_grpo` already ships the ready-queue (`async_rollout_worker.py:505`), group join at `num_generations` (`:649`), staleness discard, and rollout metrics. But it requires a separate `vllm serve` process and `grep -riE "peft\|lora\|colocate"` over that package returns **0 hits**. Server + trainer + LoRA on one 24 GB L4 does not fit. Revisit only if the Phase 1 probe returns A100 80 GB |
| Drift-reconciliation semantics lifted from `async_grpo`, not reinvented | `_SampleBuilder.classify_token_drift` → `CLEAN / REALIGN / FORK` (`async_rollout_worker.py:100-146`). Each turn re-renders the whole conversation through the template, so the re-tokenized prefix can differ from held tokens. Hand-rolled span arithmetic cannot detect that; upstream logs `drift_tokens` because the threshold is otherwise tuned blind |
| Manifest splits into invariant vs recorded-free | Strict equality over every field fails in both directions: it raises on serving-vs-training merely because `library_versions` differ, and it passes FP8/MTP-1/BF16 rows as identical because the manifest captures no quantization or KV-budget field. Invariant set is asserted; recorded set is captured, printed, and exempt |
| Reward unmodified from the paper | Comparable to published numbers. Invalid-call rate and step count are logged, never folded into reward |
| Ablations: Base / SFT / SFT+RL only | Direct-RL and env-count scaling each cost another train+eval cycle |
| Eval v1: BFCL-v3 `multi_turn_*` | Where the paper shows the largest small-model gain. Adapter boundary keeps ACEBench/τ-bench addable without touching training code |
| Verifier in a process worker with per-call timeout | Needed anyway for async rollout; a hanging `check_func` must not kill an overnight Colab run |
| Docker Compose is the serving source of truth | One image definition; Colab runs the same thing via tunnel |
| Artifacts to private HF Hub + W&B | Colab VMs are reclaimed without warning; Drive has quota and slow large-file I/O |

## Environment facts

Verified directly, not assumed. Anything here that changes invalidates
downstream sizing.

**Model** — `Qwen/Qwen3.5-2B`, `Qwen3_5ForConditionalGeneration`. **The checkpoint
is multimodal**: the config's top level holds `vision_config` (24 layers, hidden
1024, `out_hidden_size` 2048), `image_token_id` 248056, `video_token_id` 248057,
`vision_start_token_id` / `vision_end_token_id`, and every text field is nested
under `text_config` — including `max_position_embeddings`. The pipeline is
deliberately text-only and loads `AutoTokenizer` at every stage, never
`AutoProcessor`: a `ProcessorMixin` sets `trainer._is_vlm = True`
(`grpo_trainer.py:383-390`), which changes prompt rendering, the `max_model_len`
fallback, and `mm_token_type_ids` handling. Phase 1 asserts
`isinstance(processing_class, PreTrainedTokenizerBase)`.

Text config: 24 layers on a `full_attention_interval: 4` pattern — 18
`linear_attention` (Gated DeltaNet) + 6 `full_attention`. `hidden_size` 2048,
`head_dim` 256, 8 query / 2 KV heads, `linear_num_key_heads` 16,
`linear_conv_kernel_dim` 4, vocab 248,320, `tie_word_embeddings: true`,
`mtp_num_hidden_layers: 1`, `max_position_embeddings` 262,144.

The 262k advertised context is not a usable KV budget under colocated vLLM and
must be capped explicitly in every config.

**Chat template** — reasoning is retained for assistant turns after
`ns.last_query_index`, and the reverse scan that computes that index tests only
`message.role == "user"` (`chat_template.jinja:67-77`). Two consequences that are
easy to conflate: `<tool_response>`-wrapped **user** messages do not advance the
index, and `role: "tool"` messages are structurally invisible to the scan
entirely. The released trajectories carry tool results as `role: "tool"`, rendered
by a separate branch at `:131` that emits `<|im_start|>user` with **no trailing
newline** before `\n<tool_response>` — a different token stream from a
hand-wrapped user message at `:88`. Which shape the converter feeds the template
is a decision Phase 2 must state, and the same shape must be what Phase 6's
rollout appends.

Tool calls render as
`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`,
not JSON. `add_generation_prompt` emits `<think>\n` when `enable_thinking` is
true, `<think>\n\n</think>\n\n` when false.

**TRL 1.12.0** — `rollout_func(prompts, trainer)` must return `prompt_ids`,
`completion_ids`, `logprobs`; any extra key forwards to reward functions, and
`env_mask` is popped and used as the internal `tool_mask` (1 = model token,
0 = external token) — **only when `self.tools` is empty**
(`grpo_trainer.py:2270` takes `_tool_call_loop` otherwise and the pop at
`:2291` is never reached). `environment_factory` requires
`transformers>=5.2.0` (`:599`; plain `tools` needs only 5.0.0 at `:595`),
builds one stateful object per rollout, exposes public methods as tools, needs
`reset`, and optionally lets `get_reward` own the reward. Both are marked
experimental; `TRL_EXPERIMENTAL_SILENCE=1` suppresses the warning.

`trl.experimental.async_grpo` ships a full async multi-turn trainer —
`free_slots` ready scheduler, group join, `max_staleness` discard, drift
reconciliation, rollout metrics. It is server-mode only and has no PEFT path;
see the Key decisions table for why it is rejected here and what is lifted
from it.

**`logprobs` is load-bearing, not decorative.** With `use_vllm=True`,
`vllm_importance_sampling_correction` defaults to `True`
(`grpo_config.py:918`) and the returned `logprobs` feed
`(old_per_token_logps - sampling_per_token_logps) * mask` at
`grpo_trainer.py:2680`. So `logprobs` must be the same length as
`completion_ids`, with **NaN** — not 0.0 — at appended-observation positions;
TRL maps NaN to ratio exactly 1 (`:2681-2683`), while a right-pad with 0.0
(`:2489`) silently shifts every subsequent model token's logprob against the
wrong position. Track `sampling/sampling_logp_difference/max`.

**`environment_factory` pools and reuses instances across batches.** Not one
object per rollout: `self._environment_pool` hands out `pool[index]` per batch
position and only appends when a batch needs more concurrent instances
(`grpo_trainer.py:636-645`), then calls `environment.reset(**kwargs)`. Upstream's
`init_env_instance` restores only `init_config` keys via `setattr`
(`env_util.py:43-46`) while `get_state_info` returns `vars(instance)` wholesale
(`:94-99`), so any attribute an environment created during a previous episode
survives into the next `final_state()`. The factory oracle is therefore the
leakier path, and the equivalence test's direction matters: the oracle must be
made to match the isolated implementation, never the reverse.

**EnvScaler verifier arithmetic** — `calculate_reward` divides by
`len(checklist_with_func_result)` while summing only checks whose `result is
not None` (`base_env.py:296-301`), so a `check_func` that raises counts as a
failed check in the denominator, not a skipped one, and the result is
`round(..., 4)`. `run_check_function` execs with `'__builtins__': __builtins__`
and injects `initial_state` as a global (`env_util.py:106-109`) — **1,208 of the
40,231 released check functions reference `initial_state`, spanning 779 of 2,550
RL scenarios (30.5%)**, so the verifier signature must be
`score(checklist, initial_state, final_state)`. `K` ranges 2 → 445, median 14.

**Speculative decoding ceiling** — Qwen3.5's GDN `conv_states` and
`recurrent_states` carry no sequence dimension, so a partially accepted draft
cannot be rolled back to the accepted prefix. Only **MTP-1** is viable; a
1-token draft is all-or-nothing and needs no rollback. This is an architectural
limit, not a tunable.

**Quantization split** — L4 is sm89 with native FP8. A100 is sm80 without. The
two serving profiles therefore differ in strategy, not just batch size: FP8 on
L4, AWQ/GPTQ int4 or BF16 on A100.

**Datasets** — `XXHStudyHard/EnvScaler-SFT-Traj-9K` ships
`envscaler_sft_traj_9k_metadata.json` (701 MB) and a pre-templated
`mask_history_all_traj-9K_apply_qwen3_template.json` (4.59 GB). **The
pre-templated file is unusable** and this is settled, not open: EnvScaler's
`step1_process_messages_by_tool_template.py` builds it with
`tokenizer_path = "Qwen/Qwen3-4B"` (`:107`), inlines `reasoning_content` into
`content` as a `<think>` string (`:199-201`), regex-round-trips through
`<|im_start|>` markers keeping only `system|user|assistant` so `role: "tool"`
messages are destroyed as structured data (`:162-178`), and emits the Qwen3 JSON
tool-call format rather than Qwen3.5's `<function=...>` XML.
`envscaler_sft_traj_9k_metadata.json` is the only SFT input.

`tools`, `messages`, and `user_messages` are JSON **strings** requiring
`json.loads`. `191_env_metadata.json` is 20.9 MB and contains `env_class_code`;
`envscaler_rl_scenario_metadata.json` is 40.5 MB with `checklist_with_func`.

**Environment split — 140 SFT / 51 RL, settled.** Counted from `env_id` suffixes
in the vendored `191_env_metadata.json`: 191 total, 140 `_sft`, 51 `_rl`, 0 other.
All 2,550 RL scenarios draw from exactly those 51 `_rl` environments, none from an
`_sft` one. The dataset card's "141 / 50" is a description error, not a different
split. There is no third split in the release, so Phase 5's held-out slice must be
carved from the 51 RL environments.

**Reference numbers** (paper Table 4, Qwen3-1.7B Thinking — the closest
published scale, not a target):

| | BFCL-MT | Tau-Bench | ACEBench-Agent |
|---|---|---|---|
| Base | 9.75 | 12.50 | 31.95 |
| + SFT | 18.13 | 17.44 | 43.61 |
| + SFT & RL | 23.00 | **16.28** | 50.00 |

τ-bench regresses after RL at this scale. Official RL config is 8 GPUs, 32k
sequence, 40 actions/trajectory, `train_group_size: 8`, `rollout_batch_size:
512` — nothing published shows this workload on one 24 GB card.

**Versions** — trl 1.12.0, transformers 5.16.1, liger-kernel 0.8.2, vllm 0.28.0.
**torch is anchored by vllm, not by the kernel wheels.** vllm 0.28.0's PyPI
metadata declares `torch==2.13.0` — exact equality, not a floor — plus
`torchvision==0.28.0`, `torchaudio==2.11.0`, `transformers>=5.5.3`,
`requires_python >=3.10,<3.15`. Since Phase 6 needs vLLM colocated *in the
training process*, vllm and the kernel wheels share one torch, so the kernel
wheels must be selected for torch 2.13.0 rather than torch being pinned to
whatever wheels were on hand. Phase 1 resolves the full set with
`uv pip compile` and commits the lockfile — the pins are a resolution artifact,
not prose.

## Cross-phase contracts

Shared artifacts written by one phase and read by others. Ownership is stated
per field so a later phase cannot silently overwrite an earlier phase's measured
value.

**Config precedence** — `budgets.json` → `configs/base/*.yaml` → `configs/profiles/*.yaml` → `--override`. Deepest wins. `budgets.json` is in the chain, not a suggestion.

**Profile YAML field ownership**

| Field | Owner | Set by |
|---|---|---|
| `micro_batch`, `grad_accum` | Phase 3 | OOM sweep |
| `max_seq_length` | Phase 3, seeded from `budgets.json` | OOM sweep, may not exceed the budget cap |
| `generation_concurrency` (vLLM batch width in one rollout call) | Phase 6 | pool sweep |
| `active_pool_multiplier` | Phase 6 | pool sweep |
| `num_generations` (G per GRPO group) | Phase 7 | widening sweep |
| `max_env_steps` | Phase 7, seeded from `budgets.json` | widening sweep |
| `vllm_kv_fraction` | Phase 6 | OOM fallback order |
| `env_worker_count` | Phase 4 | timeline profile |

`generation_concurrency` and `num_generations` are different quantities. Phase 1
must name them separately in the schema.

**`artifacts/data/budgets.json`** (Phase 2) → read by Phase 3 (`max_seq_length`), Phase 6 (per-step generation cap), Phase 7 (`max_env_steps`). Every consumer reads the file; none hardcodes the number.

**`Episode` record** (Phase 6) → written by `scheduler.py`, `generation.py`, `rollout_func.py`; read by `rollout_func.py`, `metrics.py`, `profiler.py`, Phase 7 `trajectory_table.py`, Phase 7 `reward.py`. Required field set in Phase 6.

**Eval manifest** (Phase 5) → `assert_comparable` called by Phase 5, Phase 7, Phase 8. Invariant vs recorded-free split defined in Phase 5.

**`policies.py` generate signature** (Phase 5) → returns a structured result carrying completion, token count, and finish reason, because Phase 5 `metrics.py` needs truncation rate and generated-token counts and the HTTP policy can only source those from `usage.completion_tokens` and `finish_reason`.

## Success Criteria

- [x] `smolqwen` CLI drives the whole pipeline from config: data prep → SFT → GRPO → eval → serve. Colab notebooks are thin wrappers.
- [ ] Both GPU profiles exist as configs, and both were probed for VRAM/throughput before the full training run started; the RL profile choice cites Phase 6's episodes/hour A/B.
- [x] RL rollouts are interactive end to end; no flattened trajectory generation anywhere in the codebase.
- [x] Test proves environment state isolation: two instances from one `init_config`, mutate one, the other is unchanged.
- [x] Test proves the verifier runner: known-correct state → 1.0, known-partial → the exact expected fraction, initial state → appropriately low.
- [x] Test proves the chat template preserves reasoning at exactly the positions the training format assumes, and tool-call parse/serialize round-trips.
- [x] `env_mask` correctness is measured, not assumed: token drift between the accumulated spans and a fresh re-render is classified and logged per turn, not validated against a fixture built by the same concatenation rule.
- [x] No trainer is ever constructed with both `rollout_func` and `environment_factory`/`tools`; a test fails if both are configured.
- [ ] Three checkpoints evaluated under identical decoding, system prompt, tool harness, and step limits.
- [ ] `BFCL-MT(SFT) > BFCL-MT(Base)` and `BFCL-MT(SFT+RL) > BFCL-MT(SFT)`, with held-out EnvScaler reward rising under RL as confirmation.
- [ ] Rollout throughput A/B recorded in episodes/hour — baseline vs async scheduler — not GPU utilization percent.
- [ ] `docker compose up` serves the final checkpoint on an OpenAI-compatible endpoint; the same image runs on Colab via tunnel.
- [ ] Serving table complete with a BFCL-MT re-eval on every quantized row.
- [ ] `ruff` + `mypy --strict` + `pytest` green in CPU-only CI.

## Risks

**RL may not beat SFT.** Load-bearing assumption: a 2B hybrid model has enough
exploration capability for GRPO to extract signal. The paper's own 1.7B gained
least from RL of the three sizes tried and regressed on τ-bench. Signal it
broke: held-out EnvScaler reward flat across GRPO steps. Response: instrument
per-scenario reward variance early and prioritize scenarios where
`0 < P(success) < 1`, which is the only place GRPO advantage carries
information. Cheapest to abandon — it is the last training stage, and Base vs
SFT stands alone.

**Single-L4 GRPO feasibility is unproven.** Fallback order on OOM: fewer
generations per group → shorter context → fewer env steps → smaller vLLM KV
budget. Never zero reasoning.

**Trajectory length.** Non-Conv averages ~13 steps against a 32k training cap.
Filtering to a short subset means training on a self-defined slice of the
distribution — state that in the README rather than glossing it. Profile the real
distribution in Phase 2 before choosing any cap.

**`exec()` on dataset-supplied Python.** Environment classes and verifiers arrive
as source strings — 516 module-level statements across all 191 environments, plus
40,231 `check_func` module bodies. Process isolation plus timeouts is the accepted
posture: sufficient for the threat model (MIT-licensed dataset from a university
lab), and process workers are required for async rollout anyway. Three conditions
make that posture actually hold, and none is automatic: the `exec()` runs **only**
in the worker and never in the trainer process; the worker is `spawn`ed with a
minimal environment so no Hub or W&B credential is reachable in memory, in
`os.environ`, or on disk; and the dataset revision is pinned by sha so the artifact
that runs is the artifact the posture was accepted for. A third execution sink
exists in upstream's BFCL ground-truth path and is deliberately not ported.

**MTP-1 acceptance rate is unknown.** Agent traffic is heavy in structured
tool-call syntax, which may draft well, but a single-token draft caps the
achievable speedup. Measure before claiming anything.

## Red Team Review

### Session — 2026-08-28

Four hostile reviewers (failure-mode analyst, assumption destroyer, security
adversary, scope & complexity critic) read the plan against the pinned reference
sources: TRL 1.12.0 at `/tmp/trlsrc/trl/`, EnvScaler at `/tmp/EnvScaler/`, the
Qwen3.5-2B config and chat template, and the vLLM 0.28 CLI docs.

**Findings:** 30 accepted, 0 rejected. 10 Critical, 14 High, 6 Medium.

| # | Finding | Severity | Applied to |
|---|---------|----------|-----------|
| 1 | `env_mask` is read only when `self.tools` is empty; `environment_factory` fills `tools`, so a trainer holding both silently discards the mask and trains on observations | Critical | plan, Phase 6 |
| 2 | `rollout_func` is a synchronous fixed-size call — no cross-step buffer, no sampler admission; the scheduler's scope must be intra-call | Critical | plan, Phase 6 |
| 3 | `assert_comparable` over every field would refuse the Phase 8 serving re-eval, while capturing no serving field would give FP8/MTP-1/BF16 rows identical hashes | Critical | plan, Phase 5, Phase 8 |
| 4 | Infrastructure failures scored as legitimate low rewards teach the model to avoid crashes it did not cause | Critical | Phase 4, Phase 6, Phase 7 |
| 5 | 1,208 of 40,231 check functions read `initial_state` as a global — 779 of 2,550 RL scenarios (30.5%). A verifier taking only `final_state` raises `NameError`, swallowed as `False`, silently depressing reward on a third of the training set | Critical | plan, Phase 4 |
| 6 | `environment_factory` pools and reuses instances across batches; upstream's `reset` restores only `init_config` keys while `final_state` returns `vars()` wholesale — so the correctness oracle is the leakier path | Critical | plan, Phase 6 |
| 7 | `logprobs` feeds the IS ratio (`vllm_importance_sampling_correction` defaults True). Wrong length gets right-padded with 0.0 and shifts every later model token's logprob | Critical | plan, Phase 6 |
| 8 | The pre-templated 4.59 GB `mask_history` file is built with the Qwen3-4B tokenizer and emits Qwen3 JSON tool calls, not Qwen3.5 XML — unusable, and the "open question" would have led to using it | Critical | plan, Phase 2 |
| 9 | The template's reverse scan tests only `role == "user"`, so `role: "tool"` messages are invisible to it entirely. The two tool-result shapes render one newline apart on every observation, and the plan never fixed which one the converter feeds | Critical | plan, Phase 2, Phase 5 |
| 10 | The `env_mask` test as specified is a phantom test: fixture and code share the concatenation assumption under test | High | Phase 6 |
| 11 | TRL ships `experimental/async_grpo` with the ready-queue, group join, and drift reconciler — rejected for server-mode + zero PEFT support, but the rejection was unrecorded | High | plan, Phase 6 |
| 12 | "Phase 1 probe economics" is cited by Phases 3 and 7 but the probe produces no throughput or cost number | High | plan, Phase 1, Phase 3, Phase 7 |
| 13 | Phase 8 wraps `vllm bench sweep` then reimplements its grid, Pareto, and resume; `bfcl` is not a `--dataset-name` value and `--bfcl-categories` defaults to single-turn | High | Phase 8 |
| 14 | `--resume` restoring only weights silently replays the curriculum from the top | High | Phase 7 |
| 15 | vllm 0.28.0 pins `torch==2.13.0` exactly; the plan's "torch follows the kernel wheels, never downgrade" makes the set unresolvable once vLLM shares the training env | High | plan, Phase 1 |
| 16 | PEFT weight sync is a per-step full-parameter round trip (merge → stream all params → unmerge) plus unconditional `reset_prefix_cache()`; neither the VRAM budget nor the cache-hit-rate measurement accounts for it | High | plan, Phase 6 |
| 17 | The checkpoint is multimodal — `vision_config`, image/video token ids, text fields nested under `text_config`. An `AutoProcessor` sets `_is_vlm = True` and changes rendering, `max_model_len` fallback, and forward kwargs | High | plan, Phase 1 |
| 18 | Worker crash blast radius contradicted itself: one worker owns many episodes but a crash was said not to disturb others | Medium | Phase 4 |
| 19 | Forked workers inherit `HF_TOKEN` / `WANDB_API_KEY` into processes that `exec()` dataset code, contradicting the "no secrets in the worker" premise | Medium | Phase 1, Phase 4 |
| 20 | The env split is 140/51 in the shipped data, verified by `env_id` suffix; the plan's directive to prefer the dataset card's 141/50 would have written a wrong manifest and could leak RL envs into the held-out slice | Medium | plan, Phase 2, Phase 5 |
| 21 | Unpinned third-party wheel URLs; `latest_revision` reachable from an eval path; liveness-only health check | Medium | Phase 1, Phase 5, Phase 8 |
| 22 | Effort estimate was half the phase sum | Medium | plan |
| 23 | `--api-key` challenges only `GUARDED_PREFIX` paths, so a public tunnel serves `/tokenize`, `/metrics`, `/version`, `/load` unauthenticated — a free CPU-exhaustion vector plus model disclosure | Critical | Phase 8 |
| 24 | Env-var scrubbing cannot deliver "no secrets in the worker": the Colab vault token is cached in a module global a `fork` child inherits, and tokens also live in `~/.cache/huggingface/token` and `~/.netrc` | High | Phase 4 |
| 25 | Where the dataset `exec()` runs was unspecified; compile-once naturally puts 516 module-level statements in the trainer process, bypassing isolation, scrub, and timeout together | High | Phase 4 |
| 26 | The two files whose contents get `exec()`ed are pinned by dataset id only — everything else in the plan is pinned by sha or commit | High | Phase 2, Phase 4 |
| 27 | Upstream's BFCL ground-truth path `eval()`s benchmark strings; Phase 5 told the implementer to reuse those semantics, putting an execution sink in the unsandboxed eval process | High | Phase 5 |
| 28 | Under compile-once, `initial_state` arrives only via `__globals__`, which is fixed at scenario load — so per-episode rebinding and per-check dict isolation must be specified or 30.5% of scenarios score against a stale or shared state | High | Phase 4 |
| 29 | `bfcl_env` implements only `multi_turn_base`; three of the four categories are ours to build, not "reuse the semantics" | High | Phase 5 |
| 30 | The API key reaches argv and notebook output; the readiness probe hits a guarded path and would 401 forever | Medium | Phase 8 |

Five cross-phase contracts were verified INCOMPLETE and are now specified: the
`Episode` field set, `budgets.json` consumers and precedence, the `policies.py`
return shape, profile-YAML per-field ownership, and the manifest split. See
**Cross-phase contracts** above.

Independently confirmed while adjudicating, by counting the vendored release data
and the published metadata rather than trusting either the plan or the reviewers:

- `191_env_metadata.json`: 191 environments, 140 `_sft`, 51 `_rl`, 0 other. All
  2,550 RL scenarios draw from exactly those 51. The dataset card's 141/50 is wrong.
- `envscaler_rl_scenario_metadata.json`: 40,231 `check_func`s, 1,208 referencing
  `initial_state`, across 779 of 2,550 scenarios. `K` ranges 2 → 445, median 14.
- EnvScaler's `calculate_reward` (`base_env.py:296-301`) divides by the full check
  count while summing only non-`None` results, then rounds to 4 places — so a
  raising `check_func` is a failed check, not a skipped one.
- vllm 0.28.0 PyPI metadata: `torch==2.13.0`, `torchvision==0.28.0`,
  `torchaudio==2.11.0`, `transformers>=5.5.3`, `requires_python >=3.10,<3.15`.
- Qwen3.5-2B `config.json`: top level carries `vision_config`, `image_token_id`
  248056, `video_token_id` 248057, and nests all text fields under `text_config`.
- vLLM `server_utils.py:27,75`: `GUARDED_PREFIX = ("/v1", "/v2", "/inference",
  "/cohere")` — `--api-key` challenges nothing else.
- `bfcl_env/env.py:319` asserts `mode in ["multi_turn_base"]`, and
  `bfcl_env/data/` holds only `data_multi_turn_base.json`.
- vLLM's BFCL loader (`datasets.py:4687-4688`) uses `question[0]` with the comment
  "skip multi-turn categories in this loader", and selection additionally requires
  `--backend openai-chat` (`:2290-2302`).

Three fact-check failures were found in the plan's own citations and corrected: the
`environment_factory` pooling line reference (the behavior claim was right, the line
was the init-time probe rather than the per-batch handout at `:2370-2379`), the
claim that `bfcl_env` handles all four `multi_turn_*` categories, and the BFCL
`vllm bench` invocation.

Three claims remain **unverified** and are labeled as such in the plan rather than
asserted: paper Table 6's Non-Conv/Conv figures, the 9,022 trajectory count, and
the MTP-1 ceiling's premise about GDN `conv_states`/`recurrent_states` carrying no
sequence dimension — no Qwen3.5 GDN implementation was inspected, so Phase 8 treats
it as the hypothesis the acceptance-rate measurement tests.

Two of the plan's three open questions were closed by this counting rather than
deferred to a later phase.

### Whole-Plan Consistency Sweep

- Files reread: `plan.md`, `phase-01` … `phase-08`
- Decision deltas checked: 30 findings + 5 contracts
- Reconciled stale references: 21 — "GRPO buffer"/sampler-admission language removed
  from Phase 6; `test_group_completion.py` replaced by `test_batch_shape_contract.py`;
  "Phase 1 probe economics" removed from Phases 3 and 7 and Goal 7 restated;
  "generation count" split into `generation_concurrency` vs `num_generations`;
  `assert_comparable` phrasing aligned across Phases 5, 7, 8; profile-YAML modify lines
  scoped to owned fields in Phases 3, 6, 7; `test_sweep_grid.py` removed; effort
  corrected to the phase sum; torch pin inverted to vllm-anchored in plan Versions and
  Phase 1 step 1 and risks; the model description rewritten as
  multimodal-with-text-only-loading; the chat-template claim restated to distinguish
  `role: "tool"` from `<tool_response>` user; the `mask_history` open question closed as
  unusable in plan Datasets and Phase 2 step 1; env split fixed at 140/51 in plan
  Datasets, Phase 2 step 9 and criteria, and Phase 5's held-out slice; verifier
  signature extended to `initial_state` across plan Environment facts and Phase 4
  requirements, architecture, step 6, tests, criteria, risks; weight-sync cost added to
  Phase 6 step 4, profiler, criteria, risks; Phase 8's auth model rewritten across
  requirements, architecture, steps 1-2 and 11, criteria, risks, and file list;
  the BFCL benchmark-workload claim replaced in Phase 8 and its stale risk paragraph
  deleted; Phase 4 worker isolation moved from fork-and-scrub to
  `spawn`-with-minimal-env; dataset-`exec()` location fixed to worker-only with the
  duplicate exec-count criterion removed; dataset revision + sha256 pinning added
  across plan risks, Phase 2, and Phase 4.
- Unresolved contradictions: 0

## Open questions

- Which A100 variant does Colab allocate (40 GB vs 80 GB)? Changes the RL batch
  ceiling and is the one condition under which `AsyncGRPOTrainer` becomes worth
  revisiting. Resolved in Phase 1 by probing.
- Which message shape do the released trajectories use for a tool result in
  practice, and does any trajectory mix `role: "tool"` with `<tool_response>` user
  wrapping? Phase 2 step 1 reads real excerpts and fixes the converter's input
  contract accordingly; the SFT rendering and Phase 6's rollout appending must then
  be byte-identical for a shared fixture.

<!-- slug: smolqwen-post-training-serving -->
