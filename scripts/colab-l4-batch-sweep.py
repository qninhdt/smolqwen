"""Measure real Qwen3.5-2B training batch ceilings on one L4.

The controller runs this file through ``colab exec``.  The parent process only
orchestrates isolated child processes: every candidate loads the model, takes
one real optimizer step, records peak CUDA memory, and exits.  This is
intentional.  A caught CUDA OOM can leave allocator state behind, so a fresh
process is the only reliable boundary between candidates.

Two GRPO measurements are reported separately:

* ``grpo_generation_batch`` uses TRL's normal Transformers generation path and
  sweeps ``generation_batch_size``.  It measures the active rollout batch with
  a fixed prompt/completion envelope, not the colocated-vLLM KV cache.
* ``grpo_train_microbatch_*`` uses the same GRPO trainer and Liger loss but a
  deterministic rollout function.  It sweeps the actual optimizer microbatch
  at explicit 4k- and 8k-token sequence envelopes, isolating trainer backward
  memory from rollout-engine memory.

The production async rollout and colocated-vLLM path still need their own
throughput/KV sweep; this script does not claim to replace that measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/content/smolqwen")
ARCHIVE = Path("/content/smolqwen-l4-batch-src.tgz")
RESULT = Path("/content/smolqwen-l4-batch-results.json")
SCRIPT = ROOT / "scripts/colab-l4-batch-sweep.py"
PYTHON = ROOT / ".venv/bin/python"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# Maximum absolute error accepted for the fused-vs-padded boundary probe.  The
# tolerance is intentionally tight enough to catch document-state leakage while
# allowing ordinary bf16 accumulation differences.
PADDING_FREE_BOUNDARY_TOLERANCE = 5e-2

# The first padding-free operating point is a 32K token envelope.  The probe
# measures whether an L4 can actually sustain it; it is not an OOM estimate.
SFT_SEQUENCE_LENGTH = 32_768
# Generation needs a finite envelope so the result is reproducible and fast
# enough to run on a reclaimable VM.  The production vLLM KV sweep remains a
# separate measurement because its limit also depends on vllm_kv_fraction.
GRPO_PROMPT_LENGTH = 2_048
GRPO_COMPLETION_LENGTH = 512
# The trainer-side GRPO microbatch probes use two explicit context envelopes.
# The 16k profile cap is also tested by the SFT probe; GRPO's extra logprob and
# advantage tensors make a full 16k completion OOM even at microbatch=1 on L4,
# so the lower envelopes below are the useful operating points to size.
GRPO_TRAIN_ENVELOPES = {
    "grpo_train_microbatch_4k": (2_048, 2_048),
    "grpo_train_microbatch_8k": (4_096, 4_096),
}

PROBE_MARKER = "PROBE_RESULT="


def _device_info(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "vram_gb": round(props.total_memory / 1024**3, 3),
        "torch": torch.__version__,
    }


def _memory(torch: Any) -> dict[str, float]:
    torch.cuda.synchronize()
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
    }


def _cast_trainable_bf16(model: Any, torch: Any) -> int:
    """Match production's ``cast_adapters`` after TRL attaches PEFT."""
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
            count += 1
    return count


def _token_ids(tokenizer: Any, length: int, *, offset: int = 1000) -> list[int]:
    """Deterministic valid vocabulary ids, excluding special tokens when possible."""
    vocab = int(tokenizer.vocab_size)
    start = min(max(offset, 0), max(vocab - 2, 0))
    return [((start + index) % max(vocab - 1, 1)) for index in range(length)]


def _sft_rows(tokenizer: Any, *, sequence_lengths: list[int]) -> list[dict[str, Any]]:
    from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS

    rows: list[dict[str, Any]] = []
    for row_index, sequence_length in enumerate(sequence_lengths):
        prompt_length = sequence_length // 2
        completion_length = sequence_length - prompt_length
        input_ids = _token_ids(tokenizer, prompt_length, offset=1000 + row_index) + _token_ids(
            tokenizer, completion_length, offset=20_000 + row_index
        )
        rows.append(
            {
                "schema_version": SFT_SCHEMA_VERSION,
                "semantics": SFT_SEMANTICS,
                "trajectory_uid": f"benchmark-{row_index}:non_conversation",
                "task_id": f"benchmark-{row_index}",
                "input_ids": input_ids,
                "labels": [-100] * prompt_length + input_ids[prompt_length:],
                "seq_length": sequence_length,
                "supervised_tokens": completion_length,
            }
        )
    return rows


def _make_prompt_text(tokenizer: Any, *, length: int) -> tuple[str, int]:
    """Create a plain-text prompt whose re-tokenized length is close to ``length``."""
    # Decode a known sequence and round-trip it.  This avoids assumptions about
    # how many BPE pieces a human-readable filler word occupies.
    ids = _token_ids(tokenizer, length, offset=4000)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    actual = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, actual


def _child_base(phase: str, batch: int) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {
        "phase": phase,
        "batch": batch,
        "status": "error",
        "started_at_unix": time.time(),
    }
    result.update(_device_info(torch))
    try:
        if phase == "sft":
            details = _run_sft(torch, batch)
        elif phase == "padding_free_equivalence":
            details = _run_padding_free_equivalence(torch)
        elif phase == "grpo_generation_batch":
            details = _run_grpo_generation_batch(torch, batch)
        elif phase in GRPO_TRAIN_ENVELOPES:
            prompt_length, completion_length = GRPO_TRAIN_ENVELOPES[phase]
            details = _run_grpo_train_microbatch(
                torch, batch, prompt_length=prompt_length, completion_length=completion_length
            )
        else:
            raise ValueError(f"unknown probe phase: {phase}")
        result.update(details)
        torch.cuda.synchronize()
        result["status"] = "passed"
    except torch.cuda.OutOfMemoryError as exc:
        # Keep this distinct from a wiring/dependency failure.  The parent can
        # then identify the largest successful candidate without guessing from
        # a killed process.
        result.update({"status": "oom", "error": str(exc)[-2000:]})
    except RuntimeError as exc:
        message = str(exc)
        if re.search(r"out of memory|cuda error: out of memory", message, re.I):
            result.update({"status": "oom", "error": message[-2000:]})
        else:
            result.update(
                {
                    "status": "error",
                    "error": message[-2000:],
                    "traceback": traceback.format_exc()[-8000:],
                }
            )
    except Exception as exc:  # pragma: no cover - exercised on the remote VM
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[-2000:],
                "traceback": traceback.format_exc()[-8000:],
            }
        )
    finally:
        result["duration_s"] = round(time.monotonic() - started, 2)
        try:
            result.update(_memory(torch))
        except Exception as exc:
            result["memory_error"] = f"{type(exc).__name__}: {exc}"
        print(PROBE_MARKER + json.dumps(result, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _run_sft(torch: Any, batch: int) -> dict[str, Any]:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig  # type: ignore[attr-defined]

    from smolqwen.training.collate import padding_free_collator
    from smolqwen.training.sft import _token_budget_trainer_class

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sequence_lengths = [SFT_SEQUENCE_LENGTH // batch] * batch
    sequence_lengths[-1] += SFT_SEQUENCE_LENGTH - sum(sequence_lengths)
    if batch > 1:
        # Keep the total envelope fixed while exercising mixed document lengths.
        shift = max(1, sequence_lengths[0] // 4)
        sequence_lengths[0] -= shift
        sequence_lengths[-1] += shift
    rows = _sft_rows(tokenizer, sequence_lengths=sequence_lengths)
    dataset = Dataset.from_list(rows)
    args = SFTConfig(
        output_dir="/tmp/smolqwen-l4-sft-probe",
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_train_epochs=1.0,
        learning_rate=1e-4,
        bf16=True,
        use_cpu=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
        assistant_only_loss=False,
        packing=False,
        max_length=None,
        remove_unused_columns=False,
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer_class = _token_budget_trainer_class()
    trainer = trainer_class(
        model=MODEL_ID,
        args=args,
        data_collator=padding_free_collator(
            SFT_SEQUENCE_LENGTH, max_sequence_length=SFT_SEQUENCE_LENGTH
        ),
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
        max_tokens_per_microbatch=SFT_SEQUENCE_LENGTH,
    )
    cast_count = _cast_trainable_bf16(trainer.model, torch)
    output = trainer.train()
    loss = float(output.training_loss)
    if not torch.isfinite(torch.tensor(loss, device="cuda")):
        raise RuntimeError(f"non-finite SFT loss: {loss}")
    return {
        "sequence_lengths": sequence_lengths,
        "trajectories_per_microbatch": batch,
        "tokens_per_microbatch": sum(sequence_lengths),
        "trainable_cast_tensors": cast_count,
        "training_loss": loss,
    }


def _boundary_records(*, vocab_size: int) -> list[dict[str, Any]]:
    """Two unequal rows with supervised tails for the fused-kernel boundary probe."""
    from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS

    records: list[dict[str, Any]] = []
    for index, length in enumerate((127, 193)):
        input_ids = [
            ((1000 + index * 1000 + offset) % (vocab_size - 1)) + 1
            for offset in range(length)
        ]
        prompt_length = length // 2
        records.append(
            {
                "schema_version": SFT_SCHEMA_VERSION,
                "semantics": SFT_SEMANTICS,
                "trajectory_uid": f"boundary-{index}:non_conversation",
                "task_id": f"boundary-{index}",
                "input_ids": input_ids,
                "labels": [-100] * prompt_length + input_ids[prompt_length:],
                "seq_length": length,
                "supervised_tokens": length - prompt_length,
            }
        )
    return records


def _sum_shifted_loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    shifted_logits = logits[..., :-1, :].float().reshape(-1, logits.shape[-1])
    shifted_labels = labels[..., 1:].reshape(-1)
    return functional.cross_entropy(
        shifted_logits,
        shifted_labels,
        ignore_index=-100,
        reduction="sum",
    )


def _run_padding_free_equivalence(torch: Any) -> dict[str, Any]:
    """Compare padded and flattened Qwen3.5 on the actual L4 kernel path."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from smolqwen.training.collate import collator, padding_free_collator
    from smolqwen.training.sft import assert_padding_free_runtime

    assert_padding_free_runtime()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    records = _boundary_records(vocab_size=int(model.config.vocab_size))
    padded = collator(pad_token_id=0, max_length=max(row["seq_length"] for row in records))(records)
    flattened = padding_free_collator(
        sum(row["seq_length"] for row in records),
        max_sequence_length=max(row["seq_length"] for row in records),
    )(records)
    padded = {key: value.cuda() for key, value in padded.items()}
    flattened = {key: value.cuda() for key, value in flattened.items()}
    lengths = [int(row["seq_length"]) for row in records]

    model.eval()
    with torch.no_grad():
        padded_logits = model(**padded).logits
        flat_logits = model(**flattened).logits
    flat_slices = []
    start = 0
    for length in lengths:
        flat_slices.append(flat_logits[:, start : start + length])
        start += length
    logit_errors = [
        float((padded_logits[:, :length].float() - flat_slice.float()).abs().max())
        for length, flat_slice in zip(lengths, flat_slices, strict=True)
    ]

    padded_loss = _sum_shifted_loss(padded_logits, padded["labels"])
    flat_loss = _sum_shifted_loss(flat_logits, flattened["labels"])
    loss_error = float((padded_loss - flat_loss).abs())

    model.train()
    model.zero_grad(set_to_none=True)
    _sum_shifted_loss(model(**padded).logits, padded["labels"]).backward()
    padded_grads = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    _sum_shifted_loss(model(**flattened).logits, flattened["labels"]).backward()
    gradient_errors = [
        float((parameter.grad.float() - padded_grads[name]).abs().max())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    mutated = [dict(row) for row in records]
    mutated[0] = dict(mutated[0])
    mutated[0]["input_ids"] = list(mutated[0]["input_ids"])
    mutated[0]["input_ids"][0] = (
        (mutated[0]["input_ids"][0] + 17) % (int(model.config.vocab_size) - 1) + 1
    )
    mutated[0]["labels"] = list(mutated[0]["labels"])
    prompt_index = mutated[0]["seq_length"] // 2
    mutated[0]["labels"][prompt_index] = mutated[0]["input_ids"][prompt_index]
    mutated_batch = padding_free_collator(
        sum(row["seq_length"] for row in mutated),
        max_sequence_length=max(lengths),
    )(mutated)
    mutated_batch = {key: value.cuda() for key, value in mutated_batch.items()}
    model.eval()
    with torch.no_grad():
        mutated_logits = model(**mutated_batch).logits
    second_start = lengths[0]
    document_b_error = float(
        (
            flat_logits[:, second_start : second_start + lengths[1]].float()
            - mutated_logits[:, second_start : second_start + lengths[1]].float()
        )
        .abs()
        .max()
    )
    loss_tolerance = PADDING_FREE_BOUNDARY_TOLERANCE + 0.01 * abs(float(padded_loss))
    errors = {
        "logits": max(logit_errors),
        "loss": loss_error,
        "gradients": max(gradient_errors, default=0.0),
        "document_b": document_b_error,
    }
    failed = {
        name: error
        for name, error in errors.items()
        if error > PADDING_FREE_BOUNDARY_TOLERANCE
    }
    if loss_error > loss_tolerance:
        failed["loss"] = loss_error
    if failed:
        raise RuntimeError(
            "padding-free boundary equivalence failed: "
            + ", ".join(f"{name}={error:.6g}" for name, error in sorted(failed.items()))
        )
    return {
        "model": MODEL_ID,
        "lengths": lengths,
        "logit_max_abs_error": max(logit_errors),
        "loss_abs_error": loss_error,
        "gradient_max_abs_error": max(gradient_errors, default=0.0),
        "document_b_mutation_max_abs_error": document_b_error,
        "padded_supervised_tokens": int((padded["labels"][..., 1:] != -100).sum()),
        "flat_supervised_tokens": int((flattened["labels"][..., 1:] != -100).sum()),
        "tolerance": PADDING_FREE_BOUNDARY_TOLERANCE,
        "status": "passed",
    }


def _grpo_args(
    *,
    output_dir: str,
    per_device_batch: int,
    generation_batch: int,
    num_generations: int,
    max_completion_length: int,
) -> Any:
    from trl import GRPOConfig  # type: ignore[attr-defined]

    return GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        generation_batch_size=generation_batch,
        num_generations=num_generations,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_train_epochs=1.0,
        learning_rate=1e-6,
        bf16=True,
        use_cpu=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
        beta=0.0,
        loss_type="dapo",
        temperature=1.0,
        top_p=1.0,
        max_completion_length=max_completion_length,
        generation_kwargs={"eos_token_id": None},
        remove_unused_columns=False,
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        use_vllm=False,
        disable_tqdm=True,
    )


def _lora() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )


def _reward(completions: list[Any], **_: Any) -> list[float]:
    # Alternating rewards keep the GRPO advantage non-degenerate without any
    # external judge or environment process in a memory probe.
    return [float(index % 2) for index in range(len(completions))]


def _run_grpo_generation_batch(torch: Any, generation_batch: int) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOTrainer  # type: ignore[attr-defined]

    num_generations = 4
    if generation_batch % num_generations:
        raise ValueError(
            f"generation batch {generation_batch} must be divisible by "
            f"num_generations={num_generations}"
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_text, actual_prompt_length = _make_prompt_text(tokenizer, length=GRPO_PROMPT_LENGTH)
    rows = [{"prompt": prompt_text} for _ in range(generation_batch // num_generations)]
    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=_reward,
        args=_grpo_args(
            output_dir="/tmp/smolqwen-l4-grpo-generation-probe",
            per_device_batch=1,
            generation_batch=generation_batch,
            num_generations=num_generations,
            max_completion_length=GRPO_COMPLETION_LENGTH,
        ),
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=_lora(),
        tools=None,
        rollout_func=None,
        environment_factory=None,
    )
    cast_count = _cast_trainable_bf16(trainer.model, torch)
    output = trainer.train()
    return {
        "generation_batch_size": generation_batch,
        "num_generations": num_generations,
        "per_device_train_batch_size": 1,
        "prompt_length_requested": GRPO_PROMPT_LENGTH,
        "prompt_length_actual": actual_prompt_length,
        "completion_length_requested": GRPO_COMPLETION_LENGTH,
        "trainable_cast_tensors": cast_count,
        "training_loss": float(output.training_loss),
    }


def _run_grpo_train_microbatch(
    torch: Any, microbatch: int, *, prompt_length: int, completion_length: int
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOTrainer  # type: ignore[attr-defined]

    generation_batch = 8
    num_generations = 4
    if generation_batch % microbatch:
        raise ValueError(
            f"fixed generation batch {generation_batch} is not divisible by {microbatch}"
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids = _token_ids(tokenizer, prompt_length, offset=4000)
    completion_ids = _token_ids(tokenizer, completion_length, offset=50_000)
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    rows = [{"prompt": prompt_text} for _ in range(generation_batch // num_generations)]
    prompt_lengths: list[int] = []

    def rollout(prompts: list[Any], _: Any) -> dict[str, Any]:
        prompt_lengths.extend([len(prompt_ids)] * len(prompts))
        return {
            "prompt_ids": [prompt_ids[:] for _ in prompts],
            "completion_ids": [completion_ids[:] for _ in prompts],
            "logprobs": None,
        }

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=_reward,
        args=_grpo_args(
            output_dir="/tmp/smolqwen-l4-grpo-microbatch-probe",
            per_device_batch=microbatch,
            generation_batch=generation_batch,
            num_generations=num_generations,
            max_completion_length=completion_length,
        ),
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=_lora(),
        tools=None,
        rollout_func=rollout,
        environment_factory=None,
    )
    cast_count = _cast_trainable_bf16(trainer.model, torch)
    output = trainer.train()
    return {
        "generation_batch_size": generation_batch,
        "num_generations": num_generations,
        "per_device_train_batch_size": microbatch,
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "observed_rollout_rows": len(prompt_lengths),
        "trainable_cast_tensors": cast_count,
        "training_loss": float(output.training_loss),
    }


def _prepare() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")


def _record(results: list[dict[str, Any]], item: dict[str, Any]) -> None:
    results.append(item)
    RESULT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_child_output(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                payload = json.loads(line[len(PROBE_MARKER) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _run_candidate(phase: str, batch: int, *, timeout: int = 1800) -> dict[str, Any]:
    command = [
        str(PYTHON),
        str(SCRIPT),
        "--child",
        "--phase",
        phase,
        "--batch",
        str(batch),
    ]
    print(f"\n=== {phase} batch={batch} ===", flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "phase": phase,
            "batch": batch,
            "status": "timeout",
            "duration_s": round(time.monotonic() - started, 2),
            "output_tail": output[-12_000:],
        }
    payload = _parse_child_output(completed.stdout or "")
    if payload is None:
        output = completed.stdout or ""
        lower = output.lower()
        status = (
            "oom_or_killed"
            if completed.returncode in {-9, 137} or "out of memory" in lower
            else "error"
        )
        payload = {
            "phase": phase,
            "batch": batch,
            "status": status,
            "returncode": completed.returncode,
            "duration_s": round(time.monotonic() - started, 2),
            "output_tail": output[-12_000:],
        }
    payload.setdefault("returncode", completed.returncode)
    payload.setdefault("controller_duration_s", round(time.monotonic() - started, 2))
    return payload


def _sweep() -> int:
    RESULT.write_text("[]\n", encoding="utf-8")
    results: list[dict[str, Any]] = []
    device_probe = subprocess.run(
        [str(PYTHON), "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(device_probe.stdout or "", flush=True)

    # Exponential candidates make the first OOM visible while keeping the VM
    # cost bounded.  Once a candidate fails, larger points are not informative
    # for the requested ceiling and are skipped.
    for phase, candidates in (
        ("sft", (1, 2, 4, 8)),
        ("padding_free_equivalence", (1,)),
        ("grpo_generation_batch", (4, 8, 16, 32, 64, 128)),
        ("grpo_train_microbatch_8k", (1, 2, 4, 8)),
        ("grpo_train_microbatch_4k", (1, 2, 4, 8)),
    ):
        failed = False
        for batch in candidates:
            if failed:
                _record(
                    results,
                    {"phase": phase, "batch": batch, "status": "not_run_after_failure"},
                )
                continue
            payload = _run_candidate(phase, batch)
            _record(results, payload)
            # SFT candidates keep total tokens fixed and vary document lengths;
            # a long-document OOM does not imply shorter mixed documents fail.
            if payload.get("status") not in {"passed"} and phase != "sft":
                failed = True

    print(f"\nRESULT_FILE={RESULT}", flush=True)
    return 0 if any(item.get("status") == "passed" for item in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--phase",
        choices=(
            "sft",
            "padding_free_equivalence",
            "grpo_generation_batch",
            *GRPO_TRAIN_ENVELOPES,
        ),
    )
    parser.add_argument("--batch", type=int)
    # ``colab exec -f`` runs the file through IPython, which appends its own
    # ``-f <kernel.json>`` arguments to ``sys.argv``.  They are unrelated to
    # this runner; keep the explicit probe arguments strict while tolerating
    # that launcher suffix.
    args, _unknown = parser.parse_known_args()
    if args.child:
        if args.phase is None or args.batch is None:
            parser.error("--child requires --phase and --batch")
        return 0 if _child_base(args.phase, args.batch).get("status") in {"passed", "oom"} else 2
    _prepare()
    subprocess.run(["uv", "sync", "--locked", "--no-dev", "--extra", "colab"], cwd=ROOT, check=True)
    return _sweep()


if __name__ == "__main__":
    raise SystemExit(main())
