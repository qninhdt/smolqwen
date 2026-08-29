"""Merge a LoRA adapter into the base model and save a standalone checkpoint.

Phase 5 evaluates and Phase 7 trains from a *merged* checkpoint, not from
base + adapter, so the merge is part of the pipeline rather than a convenience.
Two things it has to get right:

- **Load the base at the pinned revision.** Merging an adapter trained against
  one base revision into another produces a checkpoint that loads fine and is
  quietly wrong, and no later stage can detect it.
- **Merge in the base dtype.** `merge_and_unload` folds `B @ A * scaling` into the
  frozen weight; doing that in fp32 and saving would double the checkpoint and
  change what serving loads, while doing it under bf16 keeps the artifact the
  same shape the trainer and vLLM both expect.

The tokenizer is saved alongside the weights because a merged checkpoint without
its chat template is not loadable by the eval harness or vLLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.config_models import SftConfig
from smolqwen.tokenizer import load_tokenizer


class MergeError(RuntimeError):
    """Raised when an adapter cannot be located or merged."""


@dataclass(frozen=True)
class MergeResult:
    adapter_dir: str
    output_dir: str
    base_model_id: str
    base_revision: str | None
    merged_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_dir": self.adapter_dir,
            "output_dir": self.output_dir,
            "base_model_id": self.base_model_id,
            # Recorded so Phase 5 can pin exactly what this merge consumed; a
            # checkpoint merged against an unrecorded base is not comparable.
            "base_revision": self.base_revision,
            "merged_parameters": self.merged_parameters,
        }


def find_adapter_dir(candidate: Path | str) -> Path:
    """Resolve a directory containing `adapter_config.json`, searching one level.

    `train-sft` writes both `output_dir/` and `output_dir/adapter/`, and a pulled
    revision lands in whichever local dir the store was given. Looking one level
    down beats making the caller know which layout it got.
    """
    path = Path(candidate)
    if (path / "adapter_config.json").is_file():
        return path
    matches = sorted(path.glob("*/adapter_config.json"))
    if matches:
        return matches[0].parent
    raise MergeError(f"no adapter_config.json found in {path} or its immediate subdirectories")


def merge_adapter(
    *,
    base_model_id: str,
    adapter_dir: Path | str,
    output_dir: Path | str,
    base_revision: str | None = None,
    dtype: str = "bfloat16",
) -> MergeResult:
    """Fold the adapter into the base weights and write a standalone checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter = find_adapter_dir(adapter_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=base_revision,
        dtype=getattr(torch, dtype),
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(target))

    tokenizer = load_tokenizer(base_model_id, revision=base_revision)
    tokenizer.save_pretrained(str(target))

    result = MergeResult(
        adapter_dir=str(adapter),
        output_dir=str(target),
        base_model_id=base_model_id,
        base_revision=base_revision,
        merged_parameters=sum(parameter.numel() for parameter in merged.parameters()),
    )
    (target / "merge_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_merge_adapter(
    config: SftConfig,
    *,
    adapter_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> int:
    """`smolqwen merge-adapter`: merge and report where the checkpoint landed."""
    result = merge_adapter(
        base_model_id=config.model_id,
        adapter_dir=adapter_dir or config.output_dir,
        output_dir=output_dir or config.merged_dir,
        base_revision=config.model_revision,
        dtype="bfloat16" if config.optimization.bf16 else "float32",
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0
