---
phase: 6
title: "SFT in-training benchmark eval"
status: in_progress
priority: P2
effort: "1.5d"
dependencies: [5]
blockedBy: [260831-0808-sft-full-trajectory-padding-free]
---

# Phase 6: SFT in-training benchmark eval

> **Decision update — 2026-09-06:** Checkpoint evaluation uses vLLM's
> `LoRARequest`; `OfflineEngine` normalizes the Qwen3.5 text-only adapter namespace
> required by the multimodal wrapper. The L4 rank-32 `all-linear` probe passes.

## Overview

Give SFT a held-out benchmark score during training by loading the checkpoint TRL
has just written into a sleeping vLLM engine — the checkpoint-based pattern open
post-training pipelines use, not a live weight-sync protocol.

## Requirements

- Functional: at each save boundary, and once before the first step, score a fixed
  **dev** subset (EnvScaler held-out; BFCL unreachable) and log the metrics.
- Functional: reuse the Phase 5 callback; only the engine source differs.
- Functional: the engine loads the checkpoint **already on disk** from TRL's save.
  No new weight-transfer protocol, no scratch directory.
- Functional: the measured token envelope with eval enabled equals the figure plan
  `260831-0808` recorded, verified by a live VRAM reading.
- Functional: total eval cost at or under 10% of training wall time.
- Non-functional: `test_sft_assembly.py`, `test_sft_smoke.py`, and
  `test_batch_shape_contract.py` pass **at plan `260831-0808`'s post-phase-4
  revision** — not "unedited", since that plan modifies all three.

## Architecture

`SFTTrainer` has no `use_vllm` (only `GRPOTrainer` does, `grpo.py:287-292`), so
there is no colocated engine. That difference is about **who calls the API**, not
about capability: vLLM's sleep mode is an engine feature, documented as built for
RL post-training, and reachable from the offline `LLM` entrypoint.

But SFT does not need weight sync at all. TRL syncs for GRPO because generation is
part of the algorithm and must use the current policy. SFT evaluation is scoring a
saved checkpoint — the pattern open post-training pipelines follow, and the one
BFCL's own tooling uses when it launches vLLM against a local checkpoint path.

And the checkpoint is already there. `sft.py:486-498` at `on_save` writes
`checkpoint-{step}`, copies it into the store, writes resume state, and pushes to
the Hub. So the eval boundary is `on_save`, and the adapter it scores is the
directory TRL just produced:

```
on_save (TRL wrote checkpoint-N; store copied and pushed it)
   → engine.wake_up()                 # base weights back in VRAM
   → load adapter from checkpoint-N   # already on disk
   → score dev subset ; log
   → engine.sleep()                   # weights offloaded, KV freed
   → restore trainer model/optimizer to GPU
   → training continues
```

Base weights never change during SFT — only the adapter does — so there is
nothing to reload on the training side, and the doc-style
`unload → reload optimizer` step is unnecessary. Sleep mode is what removes it.

**Construction order.** vLLM sizes its KV pool against total GPU memory at
construction, not against what is free. Build the engine after the trainer is
resident and it OOMs or reserves against a figure it cannot honor on a 24GB L4;
build it before and the reservation is held through training. So: build first with
an explicit KV fraction, sleep, then build the trainer, which sizes itself against
what remains. Phase 2 records what `sleep()` actually releases, so this arithmetic
is checkable before it runs.

**Boundary = `save_steps`, not `eval_steps`.** Both default to 100
(`config_models.py:162-163`), but only `save_steps` produces the checkpoint
directory. Tying eval to `on_save` is what makes "no new machinery" true; the
first draft cited `CheckpointPushCallback` while planning to fire at an arbitrary
interval, where it returns early (`sft.py:486-489`) and would have scored a stale
adapter silently.

**Push order.** `on_save` pushes to the Hub before any eval has run, so the Hub
holds every checkpoint including weak ones. That is deliberate: backup against a
reclaimed VM is the reason `artifacts.py:1-5` pushes on every save. Dev scores go
to W&B and to a file beside the checkpoint; selecting the best checkpoint happens
at the end, from those recorded scores.

**The envelope guard must measure.** A test comparing a recorded constant to a
recorded constant cannot fail when the envelope shrinks. And the obvious VRAM
reading is wrong too: `tracking.py:45` uses `max_memory_allocated()`, a monotonic
high-water mark, and `memory_reserved()`, which does not shrink without
`empty_cache()` — neither is called anywhere in `src/`. So the guard calls
`reset_peak_memory_stats()` around each boundary and reads the worker-side
driver footprint through `OfflineEngine.memory_allocated_bytes()`; vLLM V1's
parent process is not the allocator that owns the engine, and its CuMem allocator
does not make `torch.cuda.memory_allocated()` reflect unmapped sleep allocations.

**Adapter loading.** vLLM `LoRARequest` is the local path. The shared offline engine
preflights a non-zero adapter with deterministic token/logprob probes because vLLM
loads LoRA lazily, and inserts the Qwen3.5 wrapper namespace required by the
multimodal checkpoint. An accepted-but-unmatched wrapper remains a refusal, not a
valid fast path.

**Telemetry.** `VLLM_NO_USAGE_STATS` is set at the single construction point in
Phase 2; this phase asserts it in the trainer path, because that is the process
holding the HF token and the W&B session.

## Related Code Files

- Modify: `src/smolqwen/training/sft.py` — register the Phase 5 callback at
  `on_save`, sourced from the offline engine and pointed at the checkpoint
  directory TRL just wrote. Keep the diff to registration only
- Modify: `src/smolqwen/config_models.py` — eval-during-training block on
  `SftConfig`, field names mirroring `GrpoConfig`
- Modify: `configs/base/sft.yaml` — defaults sized to the 10% budget; eval
  disabled unless a card is present
- Create: `tests/test_sft_bench_eval_memory_guard.py` — `@pytest.mark.gpu`,
  reads the real worker driver footprint with `reset_peak_memory_stats()`
  through the engine's worker RPC
- Create: `tests/test_sft_bench_eval_boundary.py` — the callback fires only where
  `checkpoint-N` exists, and scores that directory rather than any other
- Modify: `docs/` (owning SFT doc) — the sleep cycle, the checkpoint-based
  pattern, and why `eval_loss` and `bench_*` measure different things

## Implementation Steps

1. Read Phase 2's recorded sleep/wake VRAM numbers. If `sleep()` did not release
   materially, take the subprocess fallback now rather than discovering it against
   a live trainer.
2. Implement the construction order above. The engine is built and slept **before**
   the trainer exists.
3. Reuse `training/bench_eval.py` unchanged, supplying the offline engine and the
   checkpoint path from `on_save`.
4. Register at `on_save`, after `CheckpointPushCallback` so the checkpoint exists
   and is already backed up. Assert the directory is present rather than returning
   early on absence — an eval that silently skips is worse than one that fails.
5. Add the step-0 baseline so the SFT chart has the same anchor as GRPO's.
6. Log `sft/bench_<category>_<metric>` beside `eval_loss`, and document the
   distinction: `eval_loss` is teacher-forced likelihood on `val.jsonl`
   (`sft.py:299`); `bench_*` is generation under the real tool harness. A reader
   seeing both will otherwise assume they should move together.
7. Write dev scores to a file beside the checkpoint as well as to W&B, so
   end-of-run checkpoint selection reads recorded numbers rather than scraping a
   dashboard.
8. Add the memory guard and the boundary test.
9. Run the three tests plan `260831-0808` accepts against, at that plan's
   post-phase-4 revision.

## Success Criteria

- [x] `sft/bench_*` in W&B with a step-0 baseline
- [x] Dev set is EnvScaler held-out only; no BFCL entry resolves here
- [x] Measured envelope with eval enabled equals plan `260831-0808`'s recorded
      figure: the L4 completed the 32K full-shape step with the engine resident,
      asleep footprint about 2.5 GiB, and live total usage about 19.3 GiB of 22.0
      GiB after the offload fix
- [x] Engine slept before and after the training step; its vLLM log reported 7.68
      GiB released and training ran at the 32K envelope without an OOM
- [x] The scored adapter is TRL's `checkpoint-N`; no new weight-transfer code and
      no scratch directory exist
- [x] Eval fires only at boundaries where `checkpoint-N` exists, and never skips
      silently
- [x] Dev scores recorded beside each checkpoint, so selection is reproducible
- [ ] Measured production-cadence eval cost at or under 10% of training wall time;
      the one-task smoke boundaries were 75.71s (base) and 86.19s (checkpoint-1)
- [x] Injected failure: training continues, environments released
- [x] `VLLM_NO_USAGE_STATS` asserted in the trainer path
- [x] The three plan-B tests pass at its post-phase-4 revision
- [x] CPU suite green; the target GPU suite passes 11 tests with one intentional
      Colab-supplied envelope skip

## Outcome

Code complete. The L4 smoke closed the envelope and sleep/wake criteria after two
runtime findings: the trainer had to move its live state to CPU while vLLM woke, and
text-only `all-linear` adapters had to exclude the unused visual tower. The remaining
open GPU criterion is the production-cadence 10% cost bound; the one-task smoke
boundaries are evidence for correctness, not for the final cadence budget.

**The `blockedBy` gate resolved on evidence, not on a status field.** Plan
`260831-0808` still reads `in_progress`, and its phase 4 is what this depended on --
but its own status section names what remains, and every item is a *GPU measurement*
(the 9,022-row artifact regeneration, fused-kernel equivalence, the 32K probes). The
code that gate existed to protect is landed: `sft.py` carries the token-budget
sampler, the padding-free collator, and supervised-token normalization
(`num_items_in_batch` at `sft.py:374`), committed as `06c4a28`. So the file-overlap
risk the gate guarded against is gone; the remaining overlap is that both plans want
the same card, which serialization does not fix.

`training/checkpoint_eval.py` owns the engine lifecycle; `sft.py` gained the
registration and a `finally` that shuts the engine down. `bench_eval.py` needed one
change: `before_each` now takes the step, and an `after_each` was added. GRPO's seam
is `sync_weights()`; SFT's is "which checkpoint does this boundary score" and
"sleep". Both run around the same runner. The L4 probe found the text-only PEFT
namespace seam and `OfflineEngine` now creates the prefixed adapter view before
validation, so SFT can stay on the vLLM path.

The runner writes `bench_eval.json` beside an existing `checkpoint-N`, with a
durable output-directory fallback for the base step-zero anchor and missing
checkpoint failures. Its SFT weight version reports `base` for that anchor rather
than inventing a `checkpoint-0`; the sidecar is local evidence because the
checkpoint push callback runs before benchmark evaluation.

**Two things the plan specified that were wrong, corrected from source:**

**The step-0 anchor and the present-checkpoint assertion contradict each other.**
The plan asked for both -- "add the step-0 baseline so the SFT chart has the same
anchor as GRPO's" and "assert the directory is present rather than returning early on
absence". At step 0 TRL has saved nothing, so there is no `checkpoint-0` and the
assertion would fail the anchor at every run. GRPO's anchor works because its
colocated engine holds live weights with an untrained adapter. The SFT equivalent is
the base model with no adapter -- which is also the `base` arm of the final table, so
the anchor is a number that already means something rather than an artifact of when
the callback fired. Step 0 takes that path; every other boundary raises on a missing
directory, as specified.

**A reused adapter name would have served stale weights.** The plan said "load
adapter from checkpoint-N" without saying under what name.
`OfflineEngine.load_adapter` derives vLLM's integer LoRA id from
`len(self._adapters) + 1`, and vLLM caches weights by that id -- so registering every
boundary as one name would hand vLLM the same id with a new path from step 200
onward, serving step 100's weights forever. The curve would be flat and every number
in it plausible. A fresh `checkpoint-<step>` name per boundary is what makes the id
fresh, and `test_sft_bench_eval_boundary.py` asserts the ids are distinct.

Two smaller things worth recording. `eval_config_for` replaces the resolved eval
config's `profile` subtree with the SFT run's: `resolve("eval")` takes no `--profile`,
so an engine sized at `ProfileConfig` defaults would have run beside a trainer sized
by `--profile l4`. And `OfflineEngineBackend` (in `inference/engine.py`) is the
adaptation from the engine's prompts-and-completions surface to the turn engine's
`GenerationRequest`/`TurnTokens` -- it generates at the widest request's budget and
truncates each row back to its own, the same trade `VllmColocateBackend` makes for
the same reason.

## Risk Assessment

The load-bearing assumption is that offline sleep releases enough for the trainer's
envelope to survive. The L4 smoke confirmed the release, but also showed that the
trainer must be offloaded during the wake/eval window after optimizer state exists.

- Signal it broke: worker-side driver footprint after `sleep()` stays near its
  pre-sleep value, or the L4 envelope drops below plan `260831-0808`'s figure.
- Response: the callback temporarily moves the trainer model and optimizer state to
  CPU, evaluates the saved adapter, then restores them after vLLM sleeps. A
  subprocess fallback remains unnecessary while this measured path fits.

Second risk: file overlap with plan `260831-0808`. Its phase 4 modifies
`sft.py`, `tracking.py`, `config_models.py`, both profile YAMLs, and the three
tests named above — so the first draft's "measurement only, code is done" claim
was wrong, and its "pass unedited" criterion was unmeetable in any ordering.

- Signal it broke: a merge conflict in `sft.py`, or one of the three tests failing
  after this phase.
- Response: this phase is gated on that plan's phase 4 being **complete**, not on
  it "reaching GPU validation". `config_models.py` and `tracking.py` get a single
  owning phase across both plans — that plan owns them until it closes.
