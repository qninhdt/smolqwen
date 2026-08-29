---
phase: 2
title: "Data pipeline and trajectory profiler"
status: pending
priority: P1
effort: "3d"
dependencies: [1]
---

# Phase 2: Data pipeline and trajectory profiler

## Overview

Turn 9,022 released teacher trajectories into Qwen3.5-rendered SFT samples whose
token layout matches inference exactly, and produce the length/step statistics
that set every context and step budget in Phases 3, 6, and 7. Measure first,
then convert.

## Requirements

**Functional**
- Load `envscaler_sft_traj_9k_metadata.json`; `json.loads` the string-encoded
  `tools`, `messages`, `user_messages` fields.
- Profiler emitting: trajectory count by mode (Conv / Non-Conv), real user-turn
  count, tool-step count, total tokens, reasoning tokens per assistant turn,
  tool-observation tokens, tool count per environment — with percentiles, not
  just means.
- Converter producing prompt/completion records split **only** at real
  user-message boundaries, rendered through the actual Qwen3.5-2B chat template.
- Deterministic train/val split by trajectory id, seeded.
- Length/step budget recommendations written as a machine-readable artifact the
  SFT and GRPO configs read.
- Environment split manifest: exact `env_id` lists for the SFT and RL sides,
  taken from the public release.

**Non-functional**
- Profiler runs on CPU over the 701 MB file in bounded memory (streaming or
  chunked; do not hold all 9k rendered trajectories at once).
- Rendered completion is byte-identical to what the model will see at inference
  for the same prefix.

## Architecture

The load-bearing subtlety is where to split, and it rests on a template detail
that is easy to state one step too strongly. The Qwen3.5 template scans messages
in reverse to find `ns.last_query_index`, and that scan tests **only**
`message.role == "user"` (`chat_template.jinja:67-77`). Two separate consequences
follow, and conflating them is how this decision goes wrong:

- A user message whose content is `<tool_response>`-wrapped does not advance the
  index (the scan checks the content prefix explicitly).
- A `role: "tool"` message is invisible to the scan altogether — it is not a user
  message, so it is never considered.

Assistant turns after that index render with `<think>\n{reasoning}\n</think>\n\n`;
turns before it render without reasoning. So within one tool-calling chain every
assistant turn keeps its reasoning — but *which* mechanism delivers that depends on
the message shape being fed in, and the two shapes produce **different token
streams**. The `role: "tool"` branch at `:131` emits `<|im_start|>user` with no
trailing newline before `\n<tool_response>`; a hand-wrapped user message at `:88`
emits `<|im_start|>user\n{content}<|im_end|>\n`. One newline apart, on every single
observation.

The released trajectories carry tool results as `role: "tool"`. EnvScaler's own
preprocessing rewrites them into `<tool_response>`-wrapped `role: "user"`. This
phase must state which shape the converter feeds the template — and Phase 6's
rollout must append the same shape, or SFT teaches a format inference never
produces.

So a Non-Conv trajectory is one sample with loss on every assistant turn. A Conv
trajectory splits at each real user message; within a segment, loss covers all
assistant turns of that segment. Splitting per tool step, as LlamaFactory
`mask_history` does, would produce ~13x the samples to teach reasoning the first
sample already contains.

```
Non-Conv trajectory (1 user msg, k tool steps)
  → 1 sample, loss on all k assistant turns

Conv trajectory (n real user msgs)
  → n samples; sample i = [msgs up to user_i] + assistant turns of segment i
    loss on segment i's assistant turns only
```

Verification is not optional here. A test renders a synthetic trajectory through
the real template and asserts reasoning appears at exactly the expected
positions — because if the template's behavior differs from this reading, the
entire SFT distribution is wrong and nothing reports an error.

Masking: loss on current-segment assistant tokens only. System, tools block, user
messages, historical assistant turns, and every tool observation are masked.
Never train the model to predict tool output.

```
system + tools block   MASK
user / task            MASK
historical assistant   MASK
tool observations      MASK
current reasoning      LOSS
current tool call      LOSS
current final answer   LOSS
```

## Related Code Files

- Create: `src/smolqwen/data/loader.py` — streaming reader for the metadata JSON, `json.loads` on string fields, schema validation
- Create: `src/smolqwen/data/profiler.py` — statistics pass, percentile report, budget recommendations
- Create: `src/smolqwen/data/render.py` — chat-template rendering, prompt/completion boundary, loss mask construction
- Create: `src/smolqwen/data/convert_sft.py` — trajectory → sample records, user-boundary split
- Create: `src/smolqwen/data/splits.py` — seeded train/val split by trajectory id; env-split manifest writer
- Create: `src/smolqwen/data/schema.py` — pydantic models for trajectory, message, tool call, sample record
- Create: `tests/test_template_reasoning_retention.py` — **critical**: reasoning position assertions against the real template
- Create: `tests/test_tool_call_roundtrip.py` — parse/serialize round-trip through the `<tool_call><function=...>` syntax
- Create: `tests/test_loss_mask.py` — mask covers exactly the current-segment assistant tokens
- Create: `tests/test_conv_split.py` — an n-user-message trajectory yields n samples with correct prefixes
- Create: `tests/fixtures/trajectories.json` — small real excerpts: one Non-Conv, one Conv, one malformed
- Modify: `configs/base/data.yaml` — dataset ids **with revision shas**, cache dir, split seed, filter thresholds
- Modify: `src/smolqwen/cli.py` — wire `profile-data` and `prepare-sft`

## Implementation Steps

1. Read real excerpts of `envscaler_sft_traj_9k_metadata.json` and fix the
   converter's input contract: which shape is a tool result — `role: "tool"`, or a
   `<tool_response>`-wrapped `role: "user"` — and does any trajectory mix both?
   Record the answer; it determines the rendered token stream and must match what
   Phase 6's rollout appends. Do **not** use
   `mask_history_all_traj-9K_apply_qwen3_template.json`: it was built with
   `tokenizer_path = "Qwen/Qwen3-4B"`, inlines `reasoning_content` into `content`,
   regex-round-trips through `<|im_start|>` keeping only `system|user|assistant`
   (destroying `role: "tool"` structure), and emits Qwen3 JSON tool calls instead of
   Qwen3.5 `<function=...>` XML. That question is closed; do not re-open it by
   inspecting the file.
2. `schema.py`: pydantic models for a trajectory, its messages
   (`role`, `content`, `reasoning_content`, `tool_calls`), and a rendered sample
   (`prompt_ids`, `completion_ids`, `loss_mask`, `trajectory_id`, `segment_index`,
   `mode`, `env_id`). Reject malformed rows loudly and count them.
3. `loader.py`: stream the metadata JSON; `json.loads` the three string fields;
   validate against the schema; yield trajectories. Classify Conv vs Non-Conv by
   presence of real (non-`tool_response`) user messages beyond the first.
4. `profiler.py`: single pass computing per-trajectory user-turn count, tool-step
   count, tokenized total length, per-turn reasoning length, observation length,
   tool count. Report p50/p90/p95/p99 and max for each, split by mode. Write
   `artifacts/data/profile.json` plus a readable table.
5. From the profile, derive and write `artifacts/data/budgets.json`: recommended
   `max_seq_length` for SFT, `max_env_steps` and per-step generation cap for RL,
   and how much of the dataset each cap retains. This file is the first layer of
   the config chain (`budgets.json → base → profile → --override`), so Phase 3
   seeds `max_seq_length` from it and may only lower it via the OOM sweep, Phase 6
   seeds the per-step generation cap, and Phase 7 seeds `max_env_steps`. No
   downstream phase hardcodes any of the three.
6. `render.py`: build the prompt/completion boundary using the tokenizer's real
   chat template with `add_generation_prompt` semantics matching inference.
   Construct the loss mask over completion positions. Return token ids, not
   strings, so nothing re-tokenizes and drifts.
7. `convert_sft.py`: for Non-Conv, emit one sample per trajectory. For Conv,
   split at each real user message and emit one sample per segment. Skip
   trajectories exceeding the budget cap and count the skips by reason.
8. Write the reasoning-retention test before trusting the converter, against
   **real excerpts of both message shapes** from the fixture file — not a synthetic
   trajectory built from the architecture note, which would only re-assert the note.
   Construct distinguishable reasoning strings per turn, render, and assert which
   survive at which positions. Assert additionally that a rendered tool observation
   is byte-identical between the two shapes or, if not, that the pipeline commits to
   exactly one. If reality contradicts the architecture note above, stop and revise
   the split decision rather than working around it.
9. `splits.py`: seeded train/val split by trajectory id (never by sample — a
   trajectory's segments must not straddle the split, or val leaks). Write the
   env-split manifest by deriving `_sft` / `_rl` from `env_id` suffixes in
   `191_env_metadata.json` (140 / 51) and assert every RL scenario's `env_id` is in
   the RL set. The dataset card's 141/50 is a description error; do not carry it into
   the manifest.
10. Run the full conversion; record final sample counts by mode, token totals,
    skip reasons, and the **sha256 of every input file** in
    `artifacts/data/conversion_report.json`. Those hashes are what let Phase 4 and
    Phase 7 assert they are executing the same dataset the pipeline was built
    against — the metadata files' contents get `exec()`ed, and a count check does not
    detect a modified class body.

## Success Criteria

- [ ] `smolqwen profile-data` writes `artifacts/data/profile.json` with per-mode percentiles for user turns, tool steps, total tokens, and per-turn reasoning tokens.
- [ ] `artifacts/data/budgets.json` exists and states, for each candidate cap, the fraction of the dataset retained. Its three keys have named consumers: `max_seq_length` (Phase 3), per-step generation cap (Phase 6), `max_env_steps` (Phase 7).
- [ ] Reasoning-retention test passes against real excerpts of both message shapes: reasoning survives across a tool-calling chain and is stripped across a real user turn, matching the split logic.
- [ ] The converter's tool-result message shape is recorded, and a shared fixture proves the SFT-rendered observation token stream matches what Phase 6's rollout appends.
- [ ] No converted sample contains a JSON-form `<tool_call>{` — all tool calls are Qwen3.5 `<function=...>` XML.
- [ ] Tool-call parse/serialize round-trip test passes against the `<tool_call><function=NAME><parameter=K>` syntax.
- [ ] Loss-mask test passes: exactly the current-segment assistant tokens are unmasked; observations and history are masked.
- [ ] Conv split test passes: an n-real-user-message trajectory yields n samples with correct cumulative prefixes.
- [ ] Train/val split is by trajectory id and reproducible from the seed.
- [ ] `artifacts/data/env_split.json` records 140 SFT / 51 RL derived from `env_id` suffixes, and asserts every RL scenario's `env_id` is in the RL set.
- [ ] `conversion_report.json` accounts for every input trajectory: converted, skipped-too-long, or malformed — and records the sha256 of every input file alongside its pinned revision.

## Risk Assessment

**The template may not behave as read.** This is the phase's central assumption
and everything downstream rests on it. Signal: the retention test shows reasoning
stripped inside a tool-calling chain, or retained across a real user turn.
Response: adopt whatever the template actually does — if reasoning is stripped
per tool step, fall back to per-tool-step splitting and accept the ~13x sample
cost, recording the reversal in the plan. Do not patch the template.

**The two tool-result message shapes render differently.** `role: "tool"` and a
`<tool_response>`-wrapped `role: "user"` differ by one newline on every
observation. If SFT trains on one and Phase 6's rollout appends the other, the
model sees a format at inference it never saw in training — and every test in this
phase still passes, because both shapes retain reasoning. Signal: the merged model
emits malformed tool calls or degrades specifically after the first observation,
despite low training loss. Response: commit to one shape, record it, and prove
byte-identity against Phase 6's appended observation for a shared fixture. This is
not on Phase 5's diagnosis list by default — add it.

**Reading the wrong SFT file.** `mask_history_all_traj-9K_apply_qwen3_template.json`
looks usable on inspection: `messages` with `role`/`content` and visible `<think>`
blocks. It is built with the Qwen3-4B tokenizer and emits Qwen3 JSON tool calls.
Signal: converted samples containing `<tool_call>{`; downstream, a ~100%
`malformed_syntax` invalid-call rate in Phase 5 against a cleanly-trained
checkpoint. Response: the file is excluded by decision, and a success criterion
asserts no JSON-form tool call reaches the training set.

**Budget cap silently reshapes the training distribution.** Filtering to short
trajectories means training on a self-defined slice. Signal: retained fraction
below ~70% at the chosen cap. Response: report the retained fraction and the
step-count distribution before and after in the README; if it drops below ~50%,
raise the cap and cut batch size in Phase 3 instead.

**Memory blowup on the 701 MB file.** Signal: OOM on a Colab CPU runtime.
Response: stream and write shards incrementally; never materialize all rendered
samples in one list.

**Val leakage through Conv segments.** Splitting by sample would put segments of
one conversation on both sides. Signal: suspiciously low val loss. Response:
split by trajectory id — enforced by a test, not by care.
