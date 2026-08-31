"""Benchmark SFT epoch-time candidates on one real L4 and real prepared rows.

Each candidate runs in a fresh child process, while the parent keeps the VM and
Hugging Face cache alive.  All candidates process the same 48 records and the
same effective batch of 16.  The first optimizer step warms kernels and is
excluded from the steady-state throughput used to estimate a full epoch.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/content/smolqwen")
ARCHIVE = Path("/content/smolqwen-l4-sft-speed-src.tgz")
SHARD = Path("/content/sft-bench.jsonl")
RESULT = Path("/content/smolqwen-l4-sft-speed-results.json")
SCRIPT = ROOT / "scripts/colab-l4-sft-speed.py"
PYTHON = ROOT / ".venv/bin/python"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# The old constant was measured on segmented shards.  Supply this from the
# regenerated full-trajectory report when an epoch-time estimate is wanted.
EPOCH_VALID_TOKENS: int | None = None
EFFECTIVE_BATCH = 16
MARKER = "SFT_SPEED_RESULT="

WAVES = {
    "batch": {
        "steps": 3,
        "candidates": {
            "mb1-ga16": {"micro_batch": 1, "group_by_length": False},
            "mb2-ga8": {"micro_batch": 2, "group_by_length": False},
            "mb4-ga4": {"micro_batch": 4, "group_by_length": False},
            "mb2-ga8-grouped": {"micro_batch": 2, "group_by_length": True},
        },
    },
    "winner": {
        "steps": 6,
        "candidates": {
            "mb2-grouped-confirm": {
                "micro_batch": 2,
                "group_by_length": True,
            },
            "mb2-grouped-no-checkpoint": {
                "micro_batch": 2,
                "group_by_length": True,
                "gradient_checkpointing": False,
            },
            "mb2-grouped-regional-compile": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
            },
        },
    },
    "checkpoint-and-batch": {
        "steps": 6,
        "candidates": {
            "mb1-no-checkpoint": {
                "micro_batch": 1,
                "group_by_length": False,
                "gradient_checkpointing": False,
            },
            "mb1-no-checkpoint-regional-compile": {
                "micro_batch": 1,
                "group_by_length": False,
                "gradient_checkpointing": False,
                "regional_compile": True,
            },
            "mb4-grouped-regional-compile": {
                "micro_batch": 4,
                "group_by_length": True,
                "regional_compile": True,
            },
        },
    },
    "kernel-details": {
        "steps": 6,
        "candidates": {
            "winner-reentrant-checkpoint": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "use_reentrant": True,
            },
            "winner-pad64": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "pad_to_multiple_of": 64,
            },
            "winner-compile-reduce-overhead": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "compile_mode": "reduce-overhead",
            },
        },
    },
    # The winner peaks at 12.66 of 22.03 GiB, so full checkpointing is paying a
    # recompute on all 24 decoder layers to save memory that nothing uses. This
    # wave spends that headroom: `every_n_layers` checkpoints one layer in N and
    # lets the rest keep their activations. Ordered ascending so the first OOM
    # marks the boundary rather than wasting the whole wave.
    "selective-checkpoint": {
        "steps": 6,
        "monotonic_memory": True,
        "candidates": {
            "winner-gc-every2": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 2,
            },
            "winner-gc-every3": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 3,
            },
            "winner-gc-every4": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 4,
            },
        },
    },
    # `winner-gc-every2` OOMed by only 370 MiB at 21.22 GiB allocated / 21.52 GiB
    # reserved, so ~300 MiB of the shortfall is allocator fragmentation rather
    # than live tensors. `expandable_segments` is the direct fix for exactly that
    # gap, and the OOM message itself recommends it. Candidates are ordered by
    # increasing memory: `every_n_layers=2` checkpoints 12 of 24 layers, `=3`
    # checkpoints 8, `=4` checkpoints 6.
    "expandable-selective-checkpoint": {
        "steps": 6,
        "monotonic_memory": True,
        "alloc_conf": "expandable_segments:True",
        "candidates": {
            "mb2-gc-every2-expandable": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 2,
            },
            "mb2-gc-every3-expandable": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 3,
            },
        },
    },
    # Fallback axis for the same question at a batch that can afford uncheckpointed
    # activations. Micro-batch 1 pads nothing by construction, but wave 1 measured
    # it ~20% slower than the micro-batch 2 winner, so a selective-checkpointing
    # win here has to clear that handicap before it changes the profile. The
    # control row isolates the checkpointing axis from the batch change.
    "mb1-selective-checkpoint": {
        "steps": 6,
        "alloc_conf": "expandable_segments:True",
        "candidates": {
            "mb1-compile-control": {
                "micro_batch": 1,
                "group_by_length": True,
                "regional_compile": True,
            },
            "mb1-gc-every2": {
                "micro_batch": 1,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 2,
            },
            "mb1-gc-every4": {
                "micro_batch": 1,
                "group_by_length": True,
                "regional_compile": True,
                "gc_every_n_layers": 4,
            },
        },
    },
    # `every_n_layers` is the wrong-shaped knob for this card: its coarsest setting
    # (`2`) already leaves 12 of 24 layers eager, and both batches OOMed there --
    # mb1 at 21.73 GiB against an 8.24 GiB checkpointed baseline, so an eager layer
    # costs over 1.1 GiB. The headroom only ever bought a handful of layers, never
    # half of them. So this wave inverts the knob and asks how many layers the
    # headroom actually buys, keeping the LAST K eager: their activations are
    # consumed first in the backward pass, so they are held for the shortest time
    # and cost the least peak. Ascending K, aborting on the first OOM.
    "keep-eager-tail": {
        "steps": 6,
        "monotonic_memory": True,
        "alloc_conf": "expandable_segments:True",
        "candidates": {
            "mb2-keep2-eager": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_keep_eager_layers": 2,
            },
            "mb2-keep4-eager": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_keep_eager_layers": 4,
            },
            "mb2-keep6-eager": {
                "micro_batch": 2,
                "group_by_length": True,
                "regional_compile": True,
                "gc_keep_eager_layers": 6,
            },
        },
    },
}
DEFAULT_WAVE = "keep-eager-tail"
CANDIDATES = {
    name: {
        **spec,
        "wave": wave_name,
        "steps": wave["steps"],
        "alloc_conf": wave.get("alloc_conf"),
    }
    for wave_name, wave in WAVES.items()
    for name, spec in wave["candidates"].items()
}


class StepTimer:
    """CUDA-synchronized optimizer-step durations."""

    def __init__(self) -> None:
        self.started = 0.0
        self.durations: list[float] = []

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch

        torch.cuda.synchronize()
        self.started = time.monotonic()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch

        torch.cuda.synchronize()
        self.durations.append(time.monotonic() - self.started)


class RecordingCollator:
    """Production collator plus exact device-token accounting per microbatch."""

    def __init__(
        self,
        base: Any,
        *,
        pad_token_id: int,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.base = base
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.batches: list[dict[str, Any]] = []

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [len(row["input_ids"]) for row in features]
        width = max(lengths)
        if self.pad_to_multiple_of:
            width = math.ceil(width / self.pad_to_multiple_of) * self.pad_to_multiple_of
        self.batches.append(
            {
                "ids": [int(row["benchmark_id"]) for row in features],
                "valid_tokens": sum(lengths),
                "supervised_tokens": sum(
                    sum(label != -100 for label in row["labels"]) for row in features
                ),
                "padded_tokens": len(features) * width,
            }
        )
        batch = self.base(features)
        extra = width - int(batch["input_ids"].shape[1])
        if extra:
            import torch.nn.functional as functional

            batch["input_ids"] = functional.pad(
                batch["input_ids"], (0, extra), value=self.pad_token_id
            )
            batch["labels"] = functional.pad(batch["labels"], (0, extra), value=-100)
            batch["attention_mask"] = functional.pad(batch["attention_mask"], (0, extra), value=0)
        return batch


def _cast_trainable_bf16(model: Any, torch: Any) -> int:
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
            count += 1
    return count


def _decoder_layers(model: Any) -> list[Any]:
    """The repeated TEXT decoder blocks, in depth order.

    Filtered by class name rather than by collecting every
    `GradientCheckpointingLayer`: this checkpoint is a multimodal wrapper whose
    vision tower contributes its own checkpointable blocks, and counting those
    would report 48 layers for a 24-layer text decoder.
    """
    return [
        module
        for module in model.modules()
        if type(module).__name__.endswith("DecoderLayer")
        and hasattr(module, "gradient_checkpointing")
    ]


def _keep_last_layers_eager(model: Any, keep: int) -> int:
    """Turn checkpointing off for the last `keep` text decoder layers.

    The last layers' activations are the first ones the backward pass consumes, so
    they are live for the shortest span and add the least to peak memory. Skipping
    their recompute is therefore the cheapest speed the spare VRAM can buy.

    Must run AFTER `Trainer.train()` has enabled checkpointing -- it calls
    `gradient_checkpointing_enable` at the top of `train()`, which rewrites every
    layer's flag and would silently undo a pre-train override. Returns how many
    layers remain checkpointed.
    """
    layers = _decoder_layers(model)
    if not layers:
        raise RuntimeError("found no text decoder layer to un-checkpoint")
    if keep >= len(layers):
        raise RuntimeError(f"keep_eager={keep} would uncheckpoint all {len(layers)} layers")
    if not any(layer.gradient_checkpointing for layer in layers):
        raise RuntimeError("no layer is checkpointed; called before the Trainer enabled it")
    for layer in layers[len(layers) - keep :]:
        layer.gradient_checkpointing = False
    return sum(1 for layer in layers if layer.gradient_checkpointing)


def _device_info(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "vram_gb": round(props.total_memory / 1024**3, 3),
        "torch": torch.__version__,
    }


def _aggregate_steps(
    batches: list[dict[str, Any]], *, gradient_accumulation: int, max_steps: int
) -> list[dict[str, int]]:
    needed = max_steps * gradient_accumulation
    if len(batches) < needed:
        raise RuntimeError(f"collator recorded {len(batches)} batches; expected at least {needed}")
    steps: list[dict[str, int]] = []
    for offset in range(0, needed, gradient_accumulation):
        group = batches[offset : offset + gradient_accumulation]
        steps.append(
            {
                "valid_tokens": sum(row["valid_tokens"] for row in group),
                "supervised_tokens": sum(row["supervised_tokens"] for row in group),
                "padded_tokens": sum(row["padded_tokens"] for row in group),
            }
        )
    return steps


def _run_child(candidate: str) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

    from smolqwen.training.collate import collator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    spec = CANDIDATES[candidate]
    max_steps = int(spec["steps"])
    micro_batch = int(spec["micro_batch"])
    gradient_accumulation = EFFECTIVE_BATCH // micro_batch
    dataset = load_dataset("json", data_files=str(SHARD), split="train")
    # Precomputed length avoids a tokenizer/model-column probe at dataloader creation.
    dataset = dataset.add_column(
        "length", [len(row["input_ids"]) for row in dataset]
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_multiple = spec.get("pad_to_multiple_of")
    recorder = RecordingCollator(
        collator(tokenizer.pad_token_id, max_length=16_384),
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=int(pad_multiple) if pad_multiple else None,
    )
    # `every_n_layers` is popped by the Trainer before the rest of the dict reaches
    # torch.utils.checkpoint, so it belongs in the same kwargs mapping.
    every_n_layers = int(spec.get("gc_every_n_layers", 1))
    checkpoint_kwargs: dict[str, Any] = {"use_reentrant": bool(spec.get("use_reentrant", False))}
    if every_n_layers != 1:
        checkpoint_kwargs["every_n_layers"] = every_n_layers
    args = SFTConfig(
        output_dir=f"/tmp/smolqwen-sft-speed-{candidate}",
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=gradient_accumulation,
        max_steps=max_steps,
        learning_rate=1e-4,
        bf16=True,
        use_cpu=False,
        gradient_checkpointing=bool(spec.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs=checkpoint_kwargs,
        use_liger_kernel=True,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
        assistant_only_loss=False,
        packing=False,
        max_length=None,
        remove_unused_columns=False,
        train_sampling_strategy=("group_by_length" if spec["group_by_length"] else "sequential"),
        length_column_name="length",
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=1234,
    )
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=args,
        data_collator=recorder,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    cast_count = _cast_trainable_bf16(trainer.model, torch)
    keep_eager = int(spec.get("gc_keep_eager_layers", 0))
    compile_detail = "disabled"
    if spec.get("regional_compile", False):
        compile_mode = spec.get("compile_mode")
        if compile_mode:
            from smolqwen.training.optim import select_compile_targets

            lookup = dict(trainer.model.named_modules())
            targets = select_compile_targets(
                [(name, type(module).__name__) for name, module in lookup.items()],
                ("linear_attn", "mixer", "conv1d"),
            )
            for name in targets:
                lookup[name].compile(mode=str(compile_mode))
            compile_detail = f"compiled {len(targets)} submodules with mode={compile_mode}"
        else:
            from smolqwen.training.optim import apply_regional_compile

            compile_toggle = apply_regional_compile(
                trainer.model,
                exclude_patterns=("linear_attn", "mixer", "conv1d"),
            )
            if not compile_toggle.enabled:
                raise RuntimeError(compile_toggle.detail)
            compile_detail = compile_toggle.detail

    class CudaStepTimer(StepTimer, TrainerCallback):
        pass

    class EagerTailCallback(TrainerCallback):
        """Apply the eager-tail override once training has enabled checkpointing.

        `Trainer.train()` calls `gradient_checkpointing_enable` before it enters the
        loop, which rewrites every layer's flag. Doing this from `on_train_begin` is
        the first point after that where the override survives.
        """

        def __init__(self) -> None:
            self.checkpointed = -1

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            layers = _decoder_layers(trainer.model)
            if keep_eager:
                self.checkpointed = _keep_last_layers_eager(trainer.model, keep_eager)
            else:
                self.checkpointed = sum(1 for layer in layers if layer.gradient_checkpointing)

    timer = CudaStepTimer()
    trainer.add_callback(timer)
    eager_tail = EagerTailCallback()
    trainer.add_callback(eager_tail)
    torch.cuda.reset_peak_memory_stats()
    train_started = time.monotonic()
    output = trainer.train()
    torch.cuda.synchronize()
    train_wall = time.monotonic() - train_started
    checkpointed_layers = eager_tail.checkpointed
    if checkpointed_layers < 0:
        raise RuntimeError("on_train_begin never ran; checkpoint layout was not observed")
    loss = float(output.training_loss)
    if not math.isfinite(loss):
        raise RuntimeError(f"non-finite training loss: {loss}")
    if len(timer.durations) != max_steps:
        raise RuntimeError(f"timed {len(timer.durations)} optimizer steps, expected {max_steps}")

    steps = _aggregate_steps(
        recorder.batches,
        gradient_accumulation=gradient_accumulation,
        max_steps=max_steps,
    )
    steady_duration = sum(timer.durations[1:])
    steady_valid = sum(row["valid_tokens"] for row in steps[1:])
    steady_padded = sum(row["padded_tokens"] for row in steps[1:])
    valid_tokens_per_s = steady_valid / steady_duration
    estimated_epoch_s = (
        EPOCH_VALID_TOKENS / valid_tokens_per_s if EPOCH_VALID_TOKENS is not None else None
    )
    return {
        "candidate": candidate,
        "status": "passed",
        **_device_info(torch),
        "micro_batch": micro_batch,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch": EFFECTIVE_BATCH,
        "group_by_length": bool(spec["group_by_length"]),
        "gradient_checkpointing": bool(spec.get("gradient_checkpointing", True)),
        "gc_every_n_layers": every_n_layers,
        "gc_keep_eager_layers": keep_eager,
        "decoder_layer_count": len(_decoder_layers(trainer.model)),
        "checkpointed_layer_count": checkpointed_layers,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "regional_compile": bool(spec.get("regional_compile", False)),
        "regional_compile_detail": compile_detail,
        "use_reentrant": bool(spec.get("use_reentrant", False)),
        "pad_to_multiple_of": int(pad_multiple) if pad_multiple else None,
        "compile_mode": spec.get("compile_mode", "default"),
        "optimizer_step_s": timer.durations,
        "steady_steps": max_steps - 1,
        "steady_valid_tokens": steady_valid,
        "steady_padded_tokens": steady_padded,
        "valid_tokens_per_s": valid_tokens_per_s,
        "padded_tokens_per_s": steady_padded / steady_duration,
        "padding_fraction": 1.0 - steady_valid / steady_padded,
        "estimated_epoch_s": estimated_epoch_s,
        "train_wall_s": train_wall,
        "training_loss": loss,
        "trainable_cast_tensors": cast_count,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "step_token_counts": steps,
    }


def _child(candidate: str) -> int:
    started = time.monotonic()
    try:
        result = _run_child(candidate)
    except Exception as exc:  # pragma: no cover - remote hardware path
        message = str(exc)
        status = "oom" if re.search(r"out of memory", message, re.I) else "error"
        result = {
            "candidate": candidate,
            "status": status,
            "error": f"{type(exc).__name__}: {exc}"[-3000:],
            "traceback": traceback.format_exc()[-10000:],
        }
        try:
            import torch

            result.update(_device_info(torch))
            result["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            result["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        except Exception:
            pass
    result["child_wall_s"] = time.monotonic() - started
    print(MARKER + json.dumps(result, sort_keys=True), flush=True)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0 if result["status"] in {"passed", "oom"} else 2


def _prepare() -> None:
    """Refresh the source tree in place, keeping the resolved virtualenv.

    Deleting `ROOT` outright would take `.venv` with it, and re-resolving the
    `colab` extra costs ~14 minutes of paid GPU time per wave on a VM that
    already has the right packages. `uv sync --locked` is near-instant when the
    venv is present and the lockfile has not moved, so the only thing that has to
    be replaced is the code.
    """
    venv = ROOT / ".venv"
    if ROOT.exists():
        for entry in ROOT.iterdir():
            if entry == venv:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")


def _parse(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    return None


def _parent(wave_name: str) -> int:
    _prepare()
    install = subprocess.run(
        ["uv", "sync", "--locked", "--no-dev", "--extra", "colab"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if install.returncode:
        print((install.stdout or "")[-20000:], flush=True)
        return install.returncode
    results: list[dict[str, Any]] = []
    RESULT.write_text("[]\n", encoding="utf-8")
    wave = WAVES[wave_name]
    candidates = wave["candidates"]
    abort_on_oom = bool(wave.get("monotonic_memory", False))
    # The allocator config has to be in the environment before the child imports
    # torch; setting it inside the child would be read after the caching allocator
    # is already configured.
    child_env = dict(os.environ)
    alloc_conf = wave.get("alloc_conf")
    if alloc_conf:
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    for candidate in candidates:
        print(f"\n=== {candidate} ===", flush=True)
        completed = subprocess.run(
            [str(PYTHON), str(SCRIPT), "--child", "--candidate", candidate],
            cwd=ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
            check=False,
        )
        payload = _parse(completed.stdout or "")
        if payload is None:
            payload = {
                "candidate": candidate,
                "status": "error",
                "returncode": completed.returncode,
                "output_tail": (completed.stdout or "")[-12000:],
            }
        results.append(payload)
        RESULT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if abort_on_oom and payload.get("status") == "oom":
            # Candidates are ordered by increasing memory, so every later one would
            # OOM as well. Stopping here is minutes of paid GPU time saved.
            print(f"\nSKIPPING remainder of wave: {candidate} OOMed", flush=True)
            break
    # A wave is a measurement, not a pass/fail gate: an OOM is a recorded boundary.
    # Only a candidate that produced no parseable payload is a runner failure.
    return 0 if all(row.get("status") in {"passed", "oom"} for row in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--wave", choices=tuple(WAVES), default=DEFAULT_WAVE)
    args, _unknown = parser.parse_known_args()
    if args.child:
        if args.candidate is None:
            parser.error("--child requires --candidate")
        return _child(args.candidate)
    return _parent(args.wave)


if __name__ == "__main__":
    raise SystemExit(main())
