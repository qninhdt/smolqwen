---
phase: 1
title: "Project scaffold, GPU profiles, CLI"
status: pending
priority: P1
effort: "2d"
dependencies: []
---

# Phase 1: Project scaffold, GPU profiles, CLI

## Overview

Stand up the `smolqwen` package, dependency pins, config-profile system, CLI
skeleton, artifact plumbing (HF Hub + W&B), and CPU-only CI — so every later
phase is a module drop-in rather than a fresh setup problem. Ends with a recorded
answer to which A100 variant Colab allocates.

## Requirements

**Functional**
- `src/smolqwen/` installable package, `smolqwen` console entry point.
- Config system: one YAML per stage (`data`, `sft`, `grpo`, `eval`, `serve`) with
  a GPU-profile overlay (`l4`, `a100`) that overrides only sizing fields.
- `--override SECTION.KEY=VALUE` on every subcommand, plus `--dry-run` that
  loads and validates config without touching the GPU.
- Checkpoint push to a private HF Hub repo on every save; `--resume` pulls the
  newest revision back.
- W&B run init with step time, tokens/s, and VRAM allocated/reserved/peak.
- `third_party/EnvScaler` present as a pinned read-only checkout.

**Non-functional**
- `ruff` + `mypy --strict` clean over `src/` and `tests/`.
- `pytest` runs on CPU with no GPU, no network, no HF token.
- Kernel wheels install only under a `colab` extra so local dev stays light.

## Architecture

Two-layer config. A base YAML holds everything semantic (dataset paths, LoRA
rank, reward settings, eval categories). A profile YAML holds only what the
GPU changes: micro-batch, gradient accumulation, sequence length,
`generation_concurrency`, `num_generations`, vLLM KV fraction, worker count.
Merge is `budgets.json <- base <- profile <- CLI --override`, deepest wins. That
keeps "which GPU" from leaking into semantics and makes the L4-vs-A100
comparison honest — the same experiment, different sizing.

`generation_concurrency` (vLLM batch width inside one rollout call) and
`num_generations` (G per GRPO group) are different quantities and must be
separate schema fields. See the plan's cross-phase contracts table for per-field
ownership; a profile field is written by exactly one phase.

```
configs/
  base/{data,sft,grpo,eval,serve}.yaml   # semantics — identical across GPUs
  profiles/{l4,a100}.yaml                # sizing only
```

Resolution is a pure function: `resolve(stage, profile, overrides) -> pydantic
model`. Pydantic validates, so a typo in an override fails at load, not thirty
minutes into a run.

Artifacts: HF Hub is the durable store. Local `artifacts/` is a cache that may
vanish with the VM. A `CheckpointStore` wraps save/push/resolve-latest/pull so
no training code touches `huggingface_hub` directly.

## Related Code Files

- Create: `pyproject.toml` — deps anchored on torch 2.13.0, `colab` extra with hash-pinned kernel wheel URLs, `serve` extra, ruff/mypy/pytest config
- Create: `uv.lock` — committed resolution artifact proving the pin set is satisfiable
- Create: `src/smolqwen/__init__.py`
- Create: `src/smolqwen/cli.py` — argparse subcommands, `--profile`, `--override`, `--dry-run`, `--resume`
- Create: `src/smolqwen/config.py` — pydantic models, deep-merge resolution
- Create: `src/smolqwen/artifacts.py` — `CheckpointStore`: save, push, resolve-latest, pull
- Create: `src/smolqwen/tracking.py` — W&B init + a `TrainerCallback` logging step time, tokens/s, VRAM
- Create: `configs/base/{data,sft,grpo,eval,serve}.yaml`
- Create: `configs/profiles/{l4,a100}.yaml`
- Create: `tests/test_config_resolution.py`
- Create: `tests/test_cli_dry_run.py`
- Create: `tests/test_checkpoint_pinning.py` — `pull` requires an explicit revision; `latest_revision` is not reachable from an eval path
- Create: `tests/test_tokenizer_not_processor.py` — the resolved `processing_class` is a tokenizer, so `trainer._is_vlm` stays False
- Create: `.github/workflows/ci.yml` — ruff, mypy, pytest on CPU
- Create: `Makefile` — `check`, `test`, `smoke`
- Create: `.gitignore`, `.env.example`
- Create: `scripts/setup_colab.sh` — install `colab` extra, pin torch, clone `third_party/EnvScaler`
- Create: `notebooks/00-probe-gpu.ipynb` — thin: detect GPU, print SM/VRAM/FP8 support, run `smolqwen probe`

## Implementation Steps

1. `pyproject.toml`. Base deps: `torch`, `transformers>=5.2`, `trl>=1.12`,
   `peft`, `datasets`, `accelerate`, `pydantic>=2`, `pyyaml`, `rich`,
   `huggingface-hub`, `wandb`. **torch is anchored by vllm 0.28.0's
   `torch==2.13.0`**, not by whatever the kernel wheels were built against —
   Phase 6 colocates vLLM in the training process, so they share one torch. Select
   flash-attn / causal-conv1d / flash-linear-attention / liger-kernel wheels built
   for torch 2.13.0, pin each URL with a hash, and resolve the whole set with
   `uv pip compile`. Commit the lockfile: the pins are a resolution artifact, not
   prose. A `serve` extra adds `vllm`. Dev group: `ruff`, `mypy`, `pytest`,
   `types-PyYAML`. Set `line-length = 100`, `select = ["E","F","I","UP","B"]`,
   mypy strict with per-module `ignore_missing_imports` for the ML stack.
2. `config.py`. One pydantic model per stage plus a `ProfileConfig` for sizing.
   Deep-merge helper; unknown keys rejected (`extra="forbid"`). An override
   parser that coerces via the target field's annotation so
   `--override sft.training.learning_rate=1e-4` lands as a float, not a string.
3. `cli.py`. Subcommands: `probe`, `profile-data`, `prepare-sft`, `train-sft`,
   `train-grpo`, `evaluate`, `serve`, `bench`. Every one takes `--config`,
   `--profile`, `--override`, `--dry-run`. Bodies are thin dispatchers; logic
   lives in the stage modules that later phases add.
4. `artifacts.py`. `CheckpointStore(repo_id, local_dir)` with `save_adapter`,
   `push`, `latest_revision`, `pull`. Push on every `save_steps` via a
   `TrainerCallback`. Adapters are small (tens to low hundreds of MB), so
   per-save push is affordable and makes VM loss a non-event. **Pin by revision
   sha for reads:** `pull` takes an explicit revision, and any evaluation records
   the resolved sha in its manifest. `latest_revision` is for `--resume` only —
   never for an eval, or a concurrent training push can silently swap the
   checkpoint under a running comparison.
5. `tracking.py`. W&B init reading project/run-name from config; a callback
   logging `system/step_time`, `system/tokens_per_second`,
   `gpu/memory_{allocated,reserved,max_allocated}_gb`. Resume by W&B run id
   stored alongside the checkpoint.
6. Write both profile YAMLs with placeholder sizing and a comment on every field
   saying what measurement will set it. No invented numbers — Phases 3 and 7
   fill them from real runs.
7. `scripts/setup_colab.sh`: install the `colab` extra, verify the torch pin
   survived, clone `third_party/EnvScaler` at a pinned commit, print
   `nvidia-smi` plus compute capability.
8. `smolqwen probe`: report GPU name, compute capability, total VRAM, whether
   FP8 is supported (sm89+), torch/transformers/trl/vllm versions, and whether
   each kernel lib imports. Write it to `artifacts/probe/{gpu}.json`. This is
   the machine-readable answer to the A100-variant open question. It reports
   **capability only** — no episodes/hour and no cost, which do not exist until
   Phase 6's `rollout-bench`. Phases 3 and 7 must not cite this probe for
   throughput economics.
9. CI: ruff, `mypy --strict`, pytest on ubuntu-latest, CPU only. No GPU job —
   nothing in CI may need a GPU or an HF token.
10. Run `smolqwen probe` on both a Colab L4 and a Colab A100 session; commit
    both JSON files. Record the A100 variant in the plan's open questions.

## Success Criteria

- [ ] `uv sync --dev && make check && make test` green locally; `uv.lock` committed and the resolved torch is 2.13.0.
- [ ] A test asserts the pipeline's `processing_class` is a `PreTrainedTokenizerBase`, not a `ProcessorMixin` — the checkpoint is multimodal and a processor flips TRL onto its VLM code paths.
- [ ] CI green on a clean checkout with no GPU, no network, no HF token.
- [ ] `smolqwen train-sft --config configs/base/sft.yaml --profile l4 --dry-run` validates config and exits 0 without importing torch CUDA paths.
- [ ] `--override sft.training.learning_rate=1e-4` type-coerces; an unknown key fails at load with a clear message.
- [ ] `artifacts/probe/l4.json` and `artifacts/probe/a100.json` committed; A100 VRAM recorded in the plan. Probe output contains no throughput or cost field.
- [ ] `third_party/EnvScaler` pinned and read-only; nothing in `src/` mutates it.
- [ ] A `CheckpointStore` round-trip test passes against a temp dir (Hub calls mocked).
- [ ] Checkpoint pinning test passes: reads require an explicit revision sha; no eval path can call `latest_revision`.
- [ ] Kernel-wheel URLs in the `colab` extra carry pinned hashes; the resolved torch version is asserted in `setup_colab.sh` before any other install.
- [ ] `.env.example` names required tokens without values; `.gitignore` covers `.env`, `artifacts/`, and the serving key file. CI asserts every committed `.ipynb` has empty output cells — a printed tunnel URL plus API key in a notebook cell is a published credential.

## Risk Assessment

**Kernel-wheel ABI drift.** The prebuilt flash-attn / causal-conv1d wheels are
compiled against one torch+CUDA combination, and vllm 0.28.0 pins
`torch==2.13.0` exactly. Signal: `ImportError` on a symbol, or `undefined symbol`
at import, typically right after installing the `serve` extra. Response: pick
wheels built for torch 2.13.0 and let the lockfile prove the set resolves. The
older instinct — pin torch to the wheels and refuse to move it — creates a hard
`ResolutionImpossible` the moment vLLM enters the same environment, which Phase 6
requires. Assert the resolved torch version in `setup_colab.sh` before installing
anything else.

**Kernel-wheel supply chain.** The prebuilt flash-attn / causal-conv1d wheels
come from third-party GitHub release URLs, not PyPI — one is a personal fork. An
unpinned URL can be re-tagged under the same name. Signal: a wheel's content
changes without a version bump. Response: pin each URL with a hash so the install
fails loudly rather than silently pulling different code.

**Multimodal checkpoint loaded as multimodal.** `Qwen3.5-2B` ships a
`vision_config` and image/video token ids, and `AutoProcessor` is the natural
choice for a `ForConditionalGeneration` architecture. Signal: `trainer._is_vlm`
True, or `max_position_embeddings` resolving through `config.text_config` on a path
that then inherits 262k. Response: the pipeline loads `AutoTokenizer` everywhere and
a test asserts it. If Phase 8's serving path needs a processor, that asymmetry is
stated rather than discovered.

**Credentials reachable from a worker.** `HF_TOKEN` and `WANDB_API_KEY` must be
exported for the training process, and env-var scrubbing alone does not keep them out
of a child: `huggingface_hub` caches the Colab vault token in a module global that a
`fork` child inherits in memory, and tokens also live in
`~/.cache/huggingface/token` and `~/.netrc`. The accepted `exec()` posture assumes
"no secrets in the worker", which is false by default. Signal:
`huggingface_hub.get_token()` returning a value inside a worker. Response: Phase 4
`spawn`s workers with an explicitly constructed minimal environment and a redirected
`HOME`/`HF_HOME`; this phase owns the `.env` handling and `.gitignore` coverage that
make that possible.

**Colab CUDA runtime mismatch for vLLM.** The PyPI vLLM wheel may want a
different CUDA runtime than Colab's torch ships on the loader path. Signal:
`ImportError: libcudart.so.N`. Response: export the pip-installed
`nvidia/cuXX/lib` directory onto `LD_LIBRARY_PATH` rather than downgrading torch.
Confirmed in Phase 8; noted here because `setup_colab.sh` is where the fix lives.

**Config system over-built.** Two-layer merge plus pydantic is already at the
edge of useful for a single-GPU project. Signal: adding a third layer, or a
profile that overrides a semantic field. Response: make `ProfileConfig` a closed
model with `extra="forbid"` — a semantic field in a profile then fails at load,
which is the actual guarantee wanted. Do not write a test asserting the profile
schema's field set is a subset of a hardcoded sizing list: that list would have
to be maintained in two places, it breaks on every legitimate field addition,
and it still cannot catch a profile *value* that changes semantics.
