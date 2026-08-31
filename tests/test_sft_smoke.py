"""SFT smoke: two steps on a tiny random-weight Qwen3.5, loss finite.

The point is not that a 4-layer random model learns anything. It is that the
whole assembly path executes -- shard load, mask validation, `datasets` wrapping,
the Phase 3 collator, LoRA attachment, forward, backward, optimizer step --
before a full GPU run costs an hour to discover a wiring mistake.

The tiny model alternates `Qwen3_5GatedDeltaNet` and `Qwen3_5Attention` layers
rather than being a generic causal LM, so the step path being exercised is the
same mixed-mixer one the real checkpoint has. That is also why the steps run on
whichever device `smoke_device` reports: with the mandatory `causal_conv1d`
kernel installed, the GDN mixer has no CPU path.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS
from smolqwen.training.collate import IGNORE_INDEX, collator
from smolqwen.training.sft import (
    SftError,
    iter_records,
    load_shards,
    validate_shard,
)
from tests.helpers import smoke_device, tiny_qwen35_model

pytestmark = pytest.mark.slow

VOCAB = 256


def _record(index: int) -> dict[str, Any]:
    prompt = [(index + position) % VOCAB for position in range(6)]
    completion = [(index + position + 30) % VOCAB for position in range(5)]
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": f"t{index}:non_conversation",
        "task_id": f"t{index}",
        "env_id": "env_1_sft",
        "mode": "non_conversation",
        "input_ids": prompt + completion,
        "labels": [-100] * len(prompt) + completion[:3] + [-100, -100],
        "seq_length": len(prompt) + len(completion),
        "supervised_tokens": 3,
    }


def _write_shards(directory: Path, *, train: int = 4, val: int = 2) -> Path:
    shard_dir = directory / "sft"
    shard_dir.mkdir(parents=True)
    for name, count, offset in (("train", train, 0), ("val", val, 100)):
        lines = [json.dumps(_record(offset + index)) for index in range(count)]
        (shard_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shard_dir


def _tiny_model() -> Any:
    return tiny_qwen35_model(vocab_size=VOCAB)


def test_shard_loading_validates_and_counts_tokens(tmp_path: Path) -> None:
    shards = load_shards(_write_shards(tmp_path))
    assert shards.train_stats.samples == 4
    assert shards.eval_stats is not None and shards.eval_stats.samples == 2
    assert shards.train_stats.total_tokens == 4 * 11
    # Supervised count comes from the mask, not from the record's own field.
    assert shards.train_stats.supervised_tokens == 4 * 3


def test_a_shard_with_broken_labels_fails_at_load(tmp_path: Path) -> None:
    shard_dir = tmp_path / "sft"
    shard_dir.mkdir(parents=True)
    broken = _record(0)
    broken["labels"] = [1, 1]
    (shard_dir / "train.jsonl").write_text(json.dumps(broken) + "\n", encoding="utf-8")

    with pytest.raises(SftError, match="labels length"):
        load_shards(shard_dir)


def test_validation_streams_rather_than_materialising(tmp_path: Path) -> None:
    """The real train shard is ~2 GB of JSON; parsed lists would not fit in RAM."""
    shard_dir = _write_shards(tmp_path, train=3)
    stats = validate_shard(shard_dir / "train.jsonl", label="train")
    assert stats.samples == 3
    assert isinstance(iter_records(shard_dir / "train.jsonl"), Iterator)


def test_missing_shard_names_the_command_that_writes_it(tmp_path: Path) -> None:
    with pytest.raises(SftError, match="prepare-sft"):
        load_shards(tmp_path / "absent")


def test_profile_cap_is_checked_before_training(tmp_path: Path) -> None:
    shard_dir = _write_shards(tmp_path, train=1, val=0)
    with pytest.raises(SftError, match="profile.max_seq_length"):
        load_shards(shard_dir, max_sequence_length=10)


def test_dataset_carries_only_the_columns_a_step_needs(tmp_path: Path) -> None:
    shards = load_shards(_write_shards(tmp_path))
    assert set(shards.train.column_names) == {
        "schema_version",
        "semantics",
        "trajectory_uid",
        "input_ids",
        "labels",
        "seq_length",
        "supervised_tokens",
        "length",
    }


def test_two_steps_on_a_tiny_model_produce_a_finite_loss(tmp_path: Path) -> None:
    """The full step path: collator -> LoRA forward -> backward -> step."""
    import torch
    from peft import LoraConfig, get_peft_model

    device = smoke_device()
    shard_dir = _write_shards(tmp_path, train=4)
    records = list(iter_records(shard_dir / "train.jsonl"))

    model = _tiny_model().to(device)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    assert trainable, "LoRA attached no trainable parameter"

    collate = collator(pad_token_id=0, max_length=32)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    losses: list[float] = []
    for step in range(2):
        batch = collate(records[step * 2 : step * 2 + 2])
        # The mask must survive the tensor boundary, not just the list one.
        assert (batch["labels"] == IGNORE_INDEX).any()
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = peft_model(**batch)
        loss = outputs.loss
        assert torch.isfinite(loss), f"step {step} loss is not finite: {loss}"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))

    assert all(value > 0 for value in losses)


def test_loss_is_computed_over_exactly_the_supervised_positions(tmp_path: Path) -> None:
    """Recompute the loss by hand from the mask and require the same number.

    This is the assertion that catches a mask that is present but ignored: a path
    that trained on the full sequence would pass every other test in this file,
    because the loss would still be finite and still descend.
    """
    import torch

    device = smoke_device()
    shard_dir = _write_shards(tmp_path, train=2)
    records = list(iter_records(shard_dir / "train.jsonl"))
    model = _tiny_model().to(device)
    batch = collator(pad_token_id=0, max_length=32)(records[:2])
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        # Causal shift: position i predicts token i+1, which is why labels are
        # stored unshifted and the model shifts internally.
        shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
        shifted_labels = batch["labels"][:, 1:].reshape(-1)
        kept = shifted_labels != IGNORE_INDEX
        manual = torch.nn.functional.cross_entropy(
            shifted_logits[kept], shifted_labels[kept], reduction="mean"
        )

    assert int(kept.sum()) == sum(record["supervised_tokens"] for record in records[:2])
    assert float(outputs.loss) == pytest.approx(float(manual), rel=1e-4)
