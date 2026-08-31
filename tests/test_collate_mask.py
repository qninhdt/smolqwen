"""Schema-v2 label validation and diagnostic padding."""

from __future__ import annotations

from typing import Any

import pytest

from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS, sample_to_record
from smolqwen.data.loader import Message
from smolqwen.data.render import render_training_sample
from smolqwen.training.collate import IGNORE_INDEX, Batch, CollateError, collate, record_to_sequence
from tests.helpers import OfflineTokenizer

PAD = 0


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": "task:conversation",
        "task_id": "task",
        "input_ids": [11, 12, 13, 14],
        "labels": [-100, 12, -100, 14],
        "seq_length": 4,
        "supervised_tokens": 2,
    }
    record.update(overrides)
    return record


def test_record_uses_persisted_exact_labels() -> None:
    assert record_to_sequence(_record()) == ([11, 12, 13, 14], [-100, 12, -100, 14])


def test_padding_preserves_label_alignment() -> None:
    short = _record(input_ids=[1, 2], labels=[-100, 2], seq_length=2, supervised_tokens=1)
    long = _record()
    batch = collate([short, long], pad_token_id=PAD)
    assert batch.input_ids[0] == (1, 2, PAD, PAD)
    assert batch.labels[0] == (IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX)
    assert batch.attention_mask[0] == (1, 1, 0, 0)
    assert batch.supervised_tokens() == 3


def test_old_segmented_schema_is_rejected_with_regeneration_command() -> None:
    old = {"prompt_ids": [1], "completion_ids": [2], "loss_mask": [1]}
    with pytest.raises(CollateError, match="prepare-sft"):
        record_to_sequence(old)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"labels": [-100]}, "labels length"),
        ({"seq_length": 99}, "seq_length"),
        ({"labels": [-100] * 4, "supervised_tokens": 0}, "no supervised token"),
        ({"labels": [-100, 99, -100, 14]}, "equal its input token"),
        ({"supervised_tokens": 1}, "supervised_tokens"),
    ],
)
def test_corrupt_record_is_rejected(override: dict[str, Any], message: str) -> None:
    with pytest.raises(CollateError, match=message):
        record_to_sequence(_record(**override))


def test_over_cap_record_is_rejected_not_truncated() -> None:
    with pytest.raises(CollateError, match="regenerate shards"):
        collate([_record()], pad_token_id=PAD, max_length=3)


def test_empty_batch_is_empty() -> None:
    assert collate([], pad_token_id=PAD) == Batch((), (), ())


def test_round_trip_from_full_trajectory_render() -> None:
    tokenizer = OfflineTokenizer()
    sample = render_training_sample(
        tokenizer,
        [
            Message("system", "sys"),
            Message("user", "task"),
            Message("assistant", content="done", reasoning_content="reason"),
            Message("user", "Task finished"),
        ],
        trajectory_uid="task:non_conversation",
        task_id="task",
        mode="non_conversation",
    )
    batch = collate([sample_to_record(sample)], pad_token_id=PAD)
    assert batch.supervised_tokens() == sample.supervised_tokens
    assert batch.seq_length == sample.total_tokens
