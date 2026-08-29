"""Collation: labels are `-100` at exactly the masked positions, under padding.

A mask off by one token trains the model to predict tool output and nothing
reports it, so these assertions run before anything trains:

- `labels[i]` is `IGNORE_INDEX` for the whole prompt and for every masked
  completion position, and equals `input_ids[i]` at supervised positions;
- padding adds `IGNORE_INDEX` labels and zeros in the attention mask, without
  shifting the real positions;
- a record whose mask length disagrees with its completion is rejected, not
  silently padded into alignment.
"""

from __future__ import annotations

from typing import Any

import pytest

from smolqwen.data.convert_sft import sample_to_record
from smolqwen.data.loader import Message, ToolCall
from smolqwen.data.render import render_segment, split_segments
from smolqwen.training.collate import (
    IGNORE_INDEX,
    Batch,
    CollateError,
    collate,
    record_to_sequence,
)
from tests.helpers import OfflineTokenizer

PAD = 0


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trajectory_id": "t",
        "segment_index": 0,
        "prompt_ids": [11, 12, 13],
        "completion_ids": [21, 22, 23, 24],
        "loss_mask": [1, 1, 0, 0],
    }
    record.update(overrides)
    return record


def test_prompt_is_fully_masked_and_supervised_positions_carry_the_token() -> None:
    input_ids, labels = record_to_sequence(_record())
    assert input_ids == [11, 12, 13, 21, 22, 23, 24]
    # Prompt masked; completion masked exactly where loss_mask is 0.
    assert labels == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        21,
        22,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_padding_preserves_mask_alignment() -> None:
    """A short and a long record in one batch: the short one's real positions do not move."""
    short = _record(prompt_ids=[1], completion_ids=[2, 3], loss_mask=[1, 0])
    long = _record(prompt_ids=[1, 2, 3, 4], completion_ids=[5, 6, 7], loss_mask=[1, 1, 1])
    batch = collate([short, long], pad_token_id=PAD)

    assert batch.seq_length == 7
    # Row 0: 3 real positions then 4 pads.
    assert batch.input_ids[0] == (1, 2, 3, PAD, PAD, PAD, PAD)
    assert batch.labels[0] == (IGNORE_INDEX, 2, IGNORE_INDEX) + (IGNORE_INDEX,) * 4
    assert batch.attention_mask[0] == (1, 1, 1, 0, 0, 0, 0)
    # Row 1 is full width, no padding.
    assert batch.attention_mask[1] == (1,) * 7
    assert batch.labels[1] == (IGNORE_INDEX,) * 4 + (5, 6, 7)


def test_padded_positions_never_contribute_loss() -> None:
    batch = collate(
        [_record(prompt_ids=[1], completion_ids=[2], loss_mask=[1]), _record()],
        pad_token_id=PAD,
    )
    # Supervised count equals the sum of the input masks, not the padded width.
    assert batch.supervised_tokens() == 1 + 2
    for labels, mask in zip(batch.labels, batch.attention_mask, strict=True):
        for label, attend in zip(labels, mask, strict=True):
            if not attend:
                assert label == IGNORE_INDEX


def test_mask_length_mismatch_is_rejected() -> None:
    with pytest.raises(CollateError, match="loss_mask length"):
        record_to_sequence(_record(loss_mask=[1, 1]))


def test_all_masked_record_is_rejected() -> None:
    """A record with no supervised token should never have been written."""
    with pytest.raises(CollateError, match="no supervised token"):
        record_to_sequence(_record(loss_mask=[0, 0, 0, 0]))


def test_empty_batch_is_empty() -> None:
    batch = collate([], pad_token_id=PAD)
    assert batch == Batch((), (), ())
    assert batch.batch_size == 0


def test_round_trip_from_a_real_rendered_sample() -> None:
    """A record straight out of Phase 2 collates with its mask intact.

    This is the end-to-end alignment check: render through the real template,
    persist with `sample_to_record`, collate, and assert the supervised label count
    equals the sample's own supervised token count.
    """
    tokenizer = OfflineTokenizer(token_size=1)
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message(
            "assistant",
            content="",
            reasoning_content="r",
            tool_calls=(ToolCall("f", {"x": 1}),),
        ),
        Message("tool", "obs"),
        Message("assistant", content="done", reasoning_content="r2"),
    ]
    sample = render_segment(tokenizer, messages, split_segments(messages)[0], trajectory_id="t")
    record = sample_to_record(sample)
    batch = collate([record], pad_token_id=PAD)

    assert batch.supervised_tokens() == sample.supervised_tokens
    assert batch.seq_length == sample.total_tokens
