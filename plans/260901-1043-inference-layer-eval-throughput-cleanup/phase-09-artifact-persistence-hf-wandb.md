---
phase: 9
title: "Artifact persistence to HF and W&B"
status: done
priority: P3
effort: "1d"
dependencies: [7]
---

# Phase 9: Artifact persistence to HF and W&B

## Overview

Make evaluation reports, budgets, difficulty profiles, and the merged model
survive VM loss, with the serving ingress redacted before anything is uploaded.

## Provenance

This phase came from a question, not a request: *"hiện tại có cơ chế tự động
upload checkpoint lên HF và upload dữ liệu lên Wandb chưa"*. The answer is
**partly yes**, and it is recorded here so the phase is not mistaken for
something that was asked for:

| Artifact | Producer | Survives VM loss |
|---|---|---|
| LoRA adapter | `sft.py:474` + `:591`, `grpo.py:166`/`:174` | yes, pushed to HF |
| W&B scalar metrics | `tracking.py` `log` / `log_step` | yes |
| Merged model | `merge.py` | **no** — writes `merge_report.json` locally |
| Eval reports | `eval/report.py` | **no** |
| `budgets.json` | `profile-data` | **no** |
| Difficulty profile | `profile-difficulty` | **no** |

So the gap is file-artifact logging and the merged model. Everything below is
scoped to that gap. The previous version also proposed a startup warning when
`hub_repo_id` is unset — that changes a contract pinned by
`test_checkpoint_pinning.py:110-116` and nobody asked for it, so it is **dropped**;
the unconfigured state surfaces in the resolved-config summary instead, which is
already printed.

## Requirements

- Functional: `Tracker.log_artifact` exists; evaluation, `profile-data`, and
  `profile-difficulty` use it.
- Functional: `merge-adapter` pushes the merged model when a repo is configured,
  behind an explicit flag.
- Functional: no uploaded artifact contains a routable ingress URL.
- Non-functional: absent `WANDB_API_KEY` or `HF_TOKEN` keeps every path working,
  matching `tracking.py:1-8`.

## Architecture

`artifacts.py:1-5` states the governing fact — *"Colab VMs are reclaimed without
warning, so an adapter that only exists locally does not exist"* — and applies it
to exactly one artifact class. The merged model is the sharpest omission: serving
loads it (`ServeConfig.model_path`), it costs a GPU to produce, and it has no
backup.

**Redaction is the part that needs care.** `runner.py:167` writes
`"endpoint": args.endpoint` into `recorded_free`, and `report.py:64` prints it on
the `Recorded-free:` line of the markdown. The Colab serving path is a public
`trycloudflare.com` hostname (`run_colab_serve.sh:53`), so uploading both report
files publishes a routable ingress to a GPU box into an artifact store whose
visibility this code never asserts. A credentialed URL would be stored whole,
since nothing strips userinfo. The repo's existing posture is the opposite:
`run_colab_serve.sh` prints the key *file path*, never the key, and
`test_auth_all_paths.py:49-50` enforces it. So the endpoint is normalized to
scheme+host-shape before it reaches the report, and userinfo is stripped
unconditionally.

**`Tracker` needs a run to attach to.** It is constructed only at `grpo.py:499`
and `sft.py:557` — `eval/runner.py` and `data/cli_actions.py` have no tracker at
all, so this phase creates one for those commands rather than assuming it exists.
The previous version did not state that.

**The merged-model push is opt-in.** A 2B bf16 model is several GB, and a Colab
uplink can take long enough that a reclaimed VM interrupts the upload the phase
exists to guarantee. It also gets its **own** `CheckpointStore` with its own
`local_dir`: reusing the adapter store's `save_adapter` would `rmtree` the adapter
cache (`artifacts.py:127-133`) and copy gigabytes before uploading.

## Related Code Files

- Modify: `src/smolqwen/tracking.py` — `log_artifact(path, *, name, artifact_type)`,
  lazy wandb import, no-op when disabled
- Modify: `src/smolqwen/eval/runner.py` — normalize `endpoint` before it reaches
  the report; create a tracker; log report JSON and Markdown
- Modify: `src/smolqwen/eval/report.py` — ensure the normalized value is what is
  written
- Modify: `src/smolqwen/data/cli_actions.py` — create a tracker; log `budgets.json`
- Modify: `src/smolqwen/training/grpo.py` — log the difficulty profile
- Modify: `src/smolqwen/training/merge.py` — opt-in push through a dedicated
  `CheckpointStore`
- Modify: `src/smolqwen/config_models.py` — the merged-push flag
- Create: `tests/test_artifact_persistence.py` — artifacts logged with a fake
  tracker; no-credential path is a no-op; **no artifact contains a URL with
  userinfo or a full tunnel hostname**
- Modify: `docs/` — what is persisted where, and what an operator must configure

## Implementation Steps

1. Add `Tracker.log_artifact`, importing wandb lazily as `start()` already does.
   No-op when `self.enabled` is false.
2. Add endpoint normalization in `runner.py` before the manifest is built. Strip
   userinfo unconditionally; reduce the host to a stable shape.
3. Create trackers for `evaluate` and `profile-data`, reusing the existing
   disabled-without-credentials behavior.
4. Log after each report write: eval JSON and Markdown, `budgets.json`, difficulty
   profile, comparison table.
5. Give `merge.py` its own store and an opt-in flag; document expected transfer
   time beside it.
6. Test with a fake hub client and fake tracker, including the redaction
   assertion.

## Success Criteria

- [x] Eval reports, `budgets.json`, difficulty profiles, comparison tables land as
      W&B artifacts when a key is present
- [x] `merge-adapter` pushes the merged model when configured and the flag is set,
      through its own store
- [x] No artifact contains a routable ingress URL or any userinfo, asserted
- [x] `evaluate` and `profile-data` create their own tracker rather than assuming
      one
- [x] `CheckpointStore.push`'s no-op-when-unconfigured contract unchanged;
      `test_checkpoint_pinning.py:110-116` passes untouched
- [x] Every path works with no `WANDB_API_KEY` and no `HF_TOKEN`
- [x] CPU suite green

## Outcome

Done. `Tracker.log_artifact` uploads a file plus named siblings and never raises —
an upload failure must not fail the run that produced the file, which is still on
disk either way. `tracker_for` replaces the identical `Tracker(...)` block the two
training stages spelled out and gives `evaluate` and `profile-data` the run they
had none of.

**Redaction moved into `EvalManifest.__post_init__`, not the runner.** The plan put
it in `runner.py` before the manifest is built, which is one call site — and every
other path that constructs a manifest (a rehydrated report, a test, a future
caller) would have been unredacted. Doing it in the constructor makes it
unforgettable and idempotent, so `from_dict` on a stored report cannot reintroduce
a hostname.

The redaction rule is narrower than "strip the host": loopback, private and
link-local addresses survive intact, because they are not reachable off-host and
"this was measured against the local proxy on 8080" is provenance a reader needs.
Public hosts keep scheme, port and path and lose their name, which is what lets two
rows still be compared as "both went through a tunnel". Userinfo goes
unconditionally, including on the loopback branch — `hostname` is rebuilt rather
than passed through, so a `user:pw@` prefix cannot survive there. A value with no
scheme (`host:8000`) becomes `redacted` outright: nothing about it can be asserted
non-routable.

`tracking.merged_hub_repo_id` is a new field rather than a default of
`hub_repo_id`. Both stores upload to their repo root, so one repo would interleave
adapter-only and merged-full revisions in a single history — after which a pinned
revision no longer says which kind it is and `resolve_eval_checkpoint` would load
whichever was pushed last. `--push` stays opt-in for the reason the plan gave, and
logs the directory size before starting so a slow uplink is a visible choice.

Comparison tables are not logged. `write_comparison_report` has no CLI entry point
— nothing in `src/` calls it, only `docs/evaluation.md` and tests — so there was no
command at which to attach the upload. Adding one was not in scope. The per-run
reports it reads are uploaded, which is what makes the comparison reproducible.

`profile-difficulty` also gained the progress bar it was missing: a few hundred
scenarios times `profile_rollouts` generations, previously with one JSON line at
the end and nothing before it.

Two follow-ons from the tracker's widened protocol: `Run` now declares
`log_artifact`, so the fake in `test_resume_sampler_cursor.py` needed the method.
No production behavior changed.

## Risk Assessment

The assumption is that these uploads are cheap. True for reports and profiles
(kilobytes), false for the merged model.

- Signal it broke: `merge-adapter` wall time dominated by upload, or a partial
  upload leaving an unusable Hub revision.
- Response: the push is opt-in for exactly this reason. A partial upload is
  recoverable — `upload_folder` is re-runnable and the local copy survives until
  the VM dies.

Second risk: W&B storage accumulation across runs.

- Signal it broke: a quota warning, or uploads beginning to fail.
- Response: reports and profiles only, never checkpoints — the Hub is the
  checkpoint store, and duplicating multi-gigabyte weights into W&B would be the
  actual quota problem. Keep that boundary explicit in the docs so a later change
  does not blur it.
