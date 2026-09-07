# Evaluation

The evaluation harness evaluates pinned local base, merged, and adapter checkpoints
through vLLM, or a pinned remotely served model through the OpenAI-compatible HTTP
boundary. A run must provide an explicit `--revision`; the evaluation path never
resolves a moving branch tip.

## Dev and test

Dev and test **coincide in this experiment's design**: the benchmark is BFCL
multi-turn base (200 tasks, 1–7 turns each), and it is both the in-training dev
curve and the final test measurement. This mirrors upstream EnvScaler's own RL
setup, whose `val_env_manager` validates on BFCL multi-turn base at a fixed
cadence, temperature 0. The consequence is stated plainly: the final BFCL number
selects nothing, because the same benchmark already drove the in-training curve.
Upstream EnvScaler's SFT stage trains with no validation set at all — this
pipeline's SFT stage is train-only for the same reason.

GRPO scores the dev benchmark in-training through `bench_eval`
(`configs/base/grpo.yaml`): a prefix of the adapter's deterministic task order at
each boundary, logged to W&B as `grpo/bench_*` alongside the training reward. The
benchmark is named directly in `bench_eval.adapter`, never read from
`eval.yaml`'s adapter list.

One caveat on reading BFCL at all: it scores by comparing final environment state
against a recorded ground truth, all-or-nothing per task, so a task fails on any
state divergence including one caused by an error in the recorded answer. An
unchanged or regressed BFCL number is not automatically a model deficiency, and a
report must say which it is rather than assuming.

```sh
smolqwen evaluate \
  --checkpoint artifacts/models/qwen3.5-2b-sft-merged \
  --revision <checkpoint-commit-sha> \
  --tag sft \
  --adapter bfcl_multi_turn \
  --profile l4
```

Evaluation logs checkpoint resolution, tokenizer loading, and vLLM construction
before scoring starts. Blocking phases emit a heartbeat every 30 seconds; per-task
progress streams in the active human-output channel and final report paths are JSON.
On Colab GPU runtimes this is stdout so cell output is immediate; set
`SMOLQWEN_LOG_STREAM=stderr` when a shell pipeline needs stdout to remain JSON-only.

The shipped config scores `multi_turn_base` only (the other three v3 multi-turn
categories remain adapter-owned but are not part of this experiment). User turns
come from the pinned benchmark files; no user-simulator or judge API key is
needed. The adapter presents the byte-identical non-conversational system prompt
used by the released SFT trajectories.

Benchmark plugins own their benchmark semantics; the generic runner owns none. Add a
module under `src/smolqwen/eval/adapters/` exposing `ADAPTER_NAME` and
`create_adapter`, then select it through `adapters` or `--adapter`. Adapter-specific
settings belong in the corresponding `adapter_options` entry.

## Generation paths

`evaluate` has two generation paths. A local checkpoint always uses the in-process
`vllm.LLM`; `--endpoint` uses the OpenAI-compatible policy for a separately served
vLLM instance. `recorded_free.generation_path` records which one ran:

| Path | When | Shape |
|---|---|---|
| `vllm` | local or merged checkpoint | batched, `min(generation_concurrency, pool_capacity)` tasks in flight |
| `vllm+lora` | local adapter checkpoint | as above, adapter served via `LoRARequest` |
| `http` | `--endpoint` | one request per turn against the served vLLM process |

There is no local Transformers fallback. Install the GPU runtime with
`bash scripts/setup_colab.sh` (or the `colab` extra on an equivalent pinned Linux
CUDA environment) before local evaluation. A missing vLLM installation, an adapter
load failure, or a silent adapter no-op fails the run rather than switching
inference implementations. This keeps local evaluation on one generation engine
and prevents an adapter failure from being mislabeled as a valid base-model score.

vLLM 0.26 supports offline LoRA through `enable_lora=True` and `LoRARequest`, and
its Qwen3.5 implementation declares `SupportsLoRA` with packed mappings for the
attention, MLP, and Gated DeltaNet projections. The text-only SFT/GRPO builders
keep the configured `all-linear` target on the language branch and exclude the
unused visual tower, because vLLM's text-only mapper cannot consume visual LoRA
keys. The PEFT adapter namespace is then normalized when needed: vLLM resolves the
released Qwen3.5 checkpoint through its multimodal wrapper
(`language_model.model.layers...`), so `OfflineEngine` creates a temporary adapter
view with the wrapper prefix before registering `LoRARequest`. The rank-32 adapter
probe changes the deterministic output/logprob probe. Base/merged checkpoints and
HTTP vLLM remain usable.

The engine reserves `max_lora_rank` from the adapter artifact; this project's rank
32 would otherwise exceed vLLM's default rank 16. Before scoring any non-zero
adapter, deterministic adapted/base probes enforce the no-silent-no-op boundary.

`recorded_free.dtype` comes from the engine that ran. On a card without bf16 tensor
cores (Turing, sm75), the local engine resolves float16 and logs the downgrade;
fp16 has a narrower exponent range, so a T4 row and an L4 row are not one experiment.

## Reasoning mode

`enable_thinking` in [`configs/base/eval.yaml`](../configs/base/eval.yaml)
(default `false`) picks the prompt render mode: `true` ends generation prompts in
an open `<think>` block, `false` in Qwen's closed empty one
(`<think>\n\n</think>\n\n`), and completions are decoded as content. The mode is
recorded in the manifest's **invariant** set — runs at different render modes are
different experiments, and `assert_comparable` refuses to pair them. The
agent-shaped serving workload renders with the same mode.

Over HTTP, the request gains `"chat_template_kwargs": {"enable_thinking": false}`
only when the mode is off: strict OpenAI-compatible endpoints reject unknown body
fields, and the thinking default needs no override. Served checkpoints need no
server-side change.

A non-reasoning eval should score checkpoints trained in the same mode — SFT
shards prepared with `data.enable_thinking: false` self-describe that in their
`semantics` tag.

## Reading a score: check how the episodes ended

Every aggregate carries `terminal_<reason>_rate` with a denominator. Read it before
the score, because the verifier grades the environment's **final state** and an
untouched initial state is a valid state that scores whatever it scores:

| Reason | Meaning |
|---|---|
| `final_answer` | the episode reached its own conclusion — the only healthy majority |
| `turn_cap` | `max_steps_per_task` generations used without completing |
| `step_cap` | `max_env_steps` reached, **or the context window filled at admission** |
| `timeout`, `worker_crash` | infrastructure, not model behaviour |

A run measured on a T4 at `max_seq_length: 4096` reported `score: 0.25255` beside
`average_generated_tokens: 0.0`: every episode terminated at admission because the
rendered prompt exceeded the window, and the verifier scored four untouched
environments. `terminal_step_cap_rate: 1.0` is what names that; the zero token
average only implies it. `evaluate` also logs a WARNING when no episode generated at
all — it does not raise, because a genuinely mute model is a real thing to measure.

Each run writes `<tag>.json` and `<tag>.md` under `artifacts/evaluation/`, plus one
`<tag>-<adapter>.jsonl` trajectory file per adapter. The manifest hashes decoding,
system prompts, tool schemas, benchmark revision, and step limits. It also records
 execution-only details such as backend, dtype, quantization, speculative decoding,
 KV budget, batching, caching, and library versions. Use `write_comparison_report`
 from `smolqwen.eval.report` to join two or more report tags;
invariant drift is rejected before a comparison artifact is written.

For an HTTP serving re-evaluation, name the engine that served the request:

```sh
smolqwen evaluate \
  --endpoint http://127.0.0.1:8000/v1 \
  --serving-backend vllm \
  --revision <served-checkpoint-sha> \
  --tag fp8
```

The serving-detail flags this command used to carry are gone. The in-process engine
knows its own dtype, KV budget, batching and caching, so those eight fields are read
off the engine's resolved `VllmConfig` after it is built — asserting them on the
command line only created a way to record something other than what ran, and reading
a caller-supplied value would do the same. On the `--endpoint` path they are recorded
as unknown rather than as whatever was typed: the serving config belongs to a process
this command cannot inspect. `--serving-backend` stays for the same reason.
To refuse a paired speed/quality row measured under a different serving config, pass
`--require-serving-match <report.json>`; it compares only the fields the throughput
measurement recorded, so an unknown makes no claim rather than a false one.

## What survives the VM

Colab reclaims VMs without warning, so `artifacts/` is a cache and anything only
there does not exist. What is uploaded, and to where:

| Artifact | Destination | Trigger |
|---|---|---|
| Eval report JSON + Markdown + trajectories | W&B artifact `eval-<tag>` | every `evaluate`, when `WANDB_API_KEY` is set |
| LoRA adapter | HF `tracking.hub_repo_id` | every `save_steps` |
| Merged full weights | HF `tracking.merged_hub_repo_id` | `merge-adapter --push` |

Reports go to W&B; weights go to the Hub. Duplicating multi-gigabyte checkpoints
into W&B would be the actual quota problem, and `CheckpointStore` already owns
that path.

`merged_hub_repo_id` is a **separate** repo from `hub_repo_id`, not a default of
it. Both stores upload to their repo root, so one repo would interleave
adapter-only and merged-full revisions in a single history — after which a pinned
revision no longer tells a reader which kind it is. The merged push is opt-in
because a 2B bf16 checkpoint is several GB and a slow uplink can outlast the VM.

Endpoints are redacted before anything is written. `recorded_free.endpoint` keeps
its scheme, port and path but loses its hostname unless the host is loopback or
private — the Colab serving path is a public `trycloudflare.com` name, and a report
recording it verbatim publishes a routable ingress to a GPU box the moment it is
uploaded. Userinfo is stripped unconditionally.

Every path above no-ops without credentials. No `WANDB_API_KEY` means no artifact
and no crash; no `hub_repo_id` means a local-only run.

Do not publish `base_vs_sft` until both pinned checkpoint reports exist and
`write_comparison_report` accepts their invariant manifests. Run on an L4/A100
or point the harness at an already-running OpenAI-compatible endpoint.
