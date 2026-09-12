# Evaluation

`smolqwen evaluate` scores a pinned local or Hugging Face checkpoint on BFCL
`multi_turn_base` using an in-process vLLM engine. There is no HTTP evaluation
fallback: a missing engine or adapter fails the run instead of silently changing
the generation path.

## Measured result

| Checkpoint | Exact success | Terminal state |
|---|---:|---|
| Official `Qwen/Qwen3.5-2B` Base | 38/200 (19.0%) | 173 final answers, 27 step caps |
| SFT | 48/200 (24.0%) | 148 final answers, 52 step caps |
| SFT + GRPO | Not reported | No comparable completed run |

The committed Base and SFT trajectories are
`artifacts/evaluation/qwen-3.5-2b-non_reasoning-{base,sft}-bfcl_v4_multi_turn_base.jsonl`.
They use the same 200 task IDs and are one run each; they do not establish variance,
statistical significance, or a state-of-the-art claim.

## Run a checkpoint

```bash
uv run smolqwen evaluate \
  --checkpoint Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --tag base \
  --categories multi_turn_base \
  --profile l4
```

Hub reads require an immutable revision SHA. Local merged checkpoints may omit
`--revision` because their files are already fixed by the path.

The runner writes:

- one JSON and Markdown aggregate per tag;
- one append-only trajectory JSONL;
- a manifest separating benchmark invariants from execution details.

Completed trajectories are flushed immediately, so a reclaimed VM retains all
finished tasks.

## Comparability

A Base/SFT/GRPO comparison is valid only when the manifests agree on:

- BFCL dataset revision and category;
- tokenizer, system prompt, and tool schemas;
- thinking mode and decoding parameters;
- step and context limits.

Execution details such as dtype, batching, quantization, and vLLM version are
recorded but do not replace those semantic invariants. The report writer rejects
incompatible manifests rather than combining unlike experiments.

## Reading a score

BFCL verifies final executable state. Always inspect terminal reasons before the
headline score:

| Reason | Meaning |
|---|---|
| `final_answer` | Episode reached its own conclusion |
| `turn_cap` | Model exhausted generations for the current static turn |
| `step_cap` | Environment/context limit stopped the episode |
| `timeout`, `worker_crash` | Infrastructure failure, not model quality |

The recorded Base run contains 27 `step_cap` outcomes and SFT contains 52. SFT also
records 1,386 invalid calls versus 383 for Base. Those caveats belong next to the
scores whenever they are quoted.

## Artifact persistence

When configured, reports are uploaded to W&B and adapters/merged checkpoints to
separate Hugging Face repositories. Without credentials, those uploads no-op and
local evaluation still completes. Endpoints and userinfo are redacted before any
report is written.
