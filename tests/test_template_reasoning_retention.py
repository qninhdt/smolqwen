"""Training preserves full reasoning history while inference remains unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.data.loader import DataError, Message, parse_trajectory
from smolqwen.data.render import (
    RenderError,
    render_prefix,
    render_training_sample,
    training_chat_template,
)
from tests.helpers import OfflineTokenizer, load_trajectory_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_training_keeps_reasoning_before_and_after_later_user() -> None:
    tokenizer = OfflineTokenizer()
    messages = [
        Message("system", "sys"),
        Message("user", "first"),
        Message("assistant", content="answer-a", reasoning_content="REASON_A"),
        Message("tool", "obs"),
        Message("assistant", content="interim", reasoning_content="REASON_B"),
        Message("user", "second"),
        Message("assistant", content="final", reasoning_content="REASON_C"),
        Message("user", "###STOP###"),
    ]
    sample = render_training_sample(
        tokenizer,
        messages,
        trajectory_uid="task:conversation",
        task_id="task",
        mode="conversation",
    )
    rendered = tokenizer.decode(list(sample.input_ids))
    assert all(reason in rendered for reason in ("REASON_A", "REASON_B", "REASON_C"))
    assert rendered.count("<think>\n") == 3
    assert sample.trailing_messages_removed == 1


def test_inference_template_still_strips_reasoning_before_last_user() -> None:
    tokenizer = OfflineTokenizer()
    messages = [
        Message("system", "sys"),
        Message("user", "first"),
        Message("assistant", content="a", reasoning_content="REASON_OLD"),
        Message("user", "second"),
        Message("assistant", content="b", reasoning_content="REASON_NEW"),
    ]
    rendered = render_prefix(tokenizer, messages)
    assert "REASON_OLD" not in rendered
    assert "REASON_NEW" in rendered


def test_training_template_fails_closed_on_upstream_drift() -> None:
    tokenizer = OfflineTokenizer()
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{%- if loop.index0 > ns.last_query_index %}",
        "{%- if changed_upstream %}",
    )
    with pytest.raises(RenderError, match="chat template drift"):
        training_chat_template(tokenizer)


def test_consecutive_tool_roles_still_merge_in_inference_shape() -> None:
    rendered = render_prefix(
        OfflineTokenizer(),
        [
            Message("system", "sys"),
            Message("user", "task"),
            Message("assistant", content="a", reasoning_content="r"),
            Message("tool", "obs1"),
            Message("tool", "obs2"),
        ],
    )
    assert rendered.count("<tool_response>") == 2
    assert rendered.count("<|im_start|>user\n<tool_response>") == 1


def test_malformed_fixture_row_rejected() -> None:
    with pytest.raises(DataError):
        parse_trajectory(load_trajectory_rows()[2])


def test_fixtures_have_pinned_ids() -> None:
    rows = json.loads((FIXTURES / "trajectories.json").read_text(encoding="utf-8"))
    ids = {
        row["task_info"]["task_id"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("task_info"), dict)
    }
    assert ids == {"env_3_sft-task_28", "env_82_sft-task_29", "env_bad_sft-task_x"}
