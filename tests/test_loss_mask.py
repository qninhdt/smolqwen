"""Exact full-trajectory assistant labels from the training chat template."""

from __future__ import annotations

import pytest

from smolqwen.data.loader import Message, ToolCall
from smolqwen.data.render import (
    IGNORE_INDEX,
    RenderedSample,
    RenderError,
    render_training_sample,
)
from tests.helpers import OfflineTokenizer


def _sample(messages: list[Message]) -> tuple[OfflineTokenizer, RenderedSample]:
    tokenizer = OfflineTokenizer()
    sample = render_training_sample(
        tokenizer,
        messages,
        trajectory_uid="task:conversation",
        task_id="task",
        mode="conversation",
    )
    return tokenizer, sample


def _assert_span(sample: RenderedSample, rendered: str, marker: str, *, supervised: bool) -> None:
    start = rendered.index(marker)
    labels = sample.labels
    span = labels[start : start + len(marker)]
    if supervised:
        assert all(label != IGNORE_INDEX for label in span)
    else:
        assert all(label == IGNORE_INDEX for label in span)


def test_all_assistant_reasoning_tool_call_and_text_are_supervised() -> None:
    tokenizer, sample = _sample(
        [
            Message("system", "UNIQUE_SYSTEM"),
            Message("user", "UNIQUE_USER_A"),
            Message(
                "assistant",
                content="UNIQUE_ASSISTANT_TEXT_A",
                reasoning_content="UNIQUE_REASON_A",
                tool_calls=(ToolCall("unique_function", {"x": "unique_arg"}),),
            ),
            Message("tool", "UNIQUE_OBSERVATION"),
            Message("user", "UNIQUE_USER_B"),
            Message(
                "assistant",
                content="UNIQUE_ASSISTANT_TEXT_B",
                reasoning_content="UNIQUE_REASON_B",
            ),
            Message("user", "###STOP###"),
        ]
    )
    rendered = tokenizer.decode(list(sample.input_ids))
    for marker in (
        "UNIQUE_REASON_A",
        "UNIQUE_ASSISTANT_TEXT_A",
        "unique_function",
        "unique_arg",
        "UNIQUE_REASON_B",
        "UNIQUE_ASSISTANT_TEXT_B",
    ):
        _assert_span(sample, rendered, marker, supervised=True)
    for marker in ("UNIQUE_SYSTEM", "UNIQUE_USER_A", "UNIQUE_OBSERVATION", "UNIQUE_USER_B"):
        _assert_span(sample, rendered, marker, supervised=False)
    assert "###STOP###" not in rendered


def test_terminal_user_is_trimmed_without_changing_ids_or_labels() -> None:
    base = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", content="done", reasoning_content="reason"),
    ]
    _, without = _sample(base)
    _, with_terminal = _sample(base + [Message("user", "Task finished")])
    assert with_terminal.input_ids == without.input_ids
    assert with_terminal.labels == without.labels
    assert with_terminal.trailing_messages_removed == 1


def test_no_assistant_after_real_user_raises() -> None:
    with pytest.raises(RenderError, match="no assistant turn"):
        _sample([Message("system", "sys"), Message("user", "task")])
