"""Merge: the adapter is located, the base revision is recorded, weights change.

A merge that loads and is quietly wrong is the failure mode: `merge_and_unload`
against a different base revision produces a checkpoint no later stage can
detect as mismatched. So the report records the revision it consumed, and these
tests assert it survives to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.training.merge import MergeError, find_adapter_dir, merge_adapter
from tests.helpers import write_tiny_checkpoint

pytestmark = pytest.mark.slow

VOCAB = 256


def _tiny_base(directory: Path) -> Path:
    # The merge path saves the tokenizer next to the weights, so the base has to
    # carry one -- a merged checkpoint without a tokenizer is not loadable.
    return write_tiny_checkpoint(directory, vocab_size=VOCAB)


def _tiny_adapter(base_dir: Path, adapter_dir: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(base_dir))
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    # A zero-initialised B matrix makes the merge a no-op, which would make the
    # "weights actually changed" assertion vacuous. So perturb it first.
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
    peft_model.save_pretrained(str(adapter_dir))
    return adapter_dir


def test_find_adapter_dir_searches_one_level_down(tmp_path: Path) -> None:
    nested = tmp_path / "adapter"
    nested.mkdir()
    (nested / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert find_adapter_dir(tmp_path) == nested
    assert find_adapter_dir(nested) == nested


def test_find_adapter_dir_names_what_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="adapter_config.json"):
        find_adapter_dir(tmp_path)


def _load(path: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path))


def test_merge_folds_the_adapter_and_records_the_base_revision(tmp_path: Path) -> None:
    import torch

    base_dir = _tiny_base(tmp_path / "base")
    adapter_dir = _tiny_adapter(base_dir, tmp_path / "adapter")
    merged_dir = tmp_path / "merged"

    result = merge_adapter(
        base_model_id=str(base_dir),
        adapter_dir=adapter_dir,
        output_dir=merged_dir,
        base_revision=None,
        dtype="float32",
    )

    # Standalone: loadable with no PEFT involvement and no adapter files present.
    merged = _load(merged_dir)
    assert not list(merged_dir.glob("adapter_model*"))
    assert result.merged_parameters == sum(p.numel() for p in merged.parameters())

    # The merge is not a no-op: at least one folded weight differs from the base.
    original = _load(base_dir)
    differences = [
        name
        for (name, before), (_, after) in zip(
            original.named_parameters(), merged.named_parameters(), strict=True
        )
        if not torch.equal(before, after)
    ]
    assert differences, "merge_and_unload changed nothing; the adapter was not applied"

    report = json.loads((merged_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["base_model_id"] == str(base_dir)
    # Present-but-null is the point: the field is always recorded, so Phase 5 can
    # tell "unpinned" apart from "the key was never written".
    assert "base_revision" in report
