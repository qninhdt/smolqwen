"""Loss-mask construction: exactly the current-segment assistant tokens are unmasked.

Never train the model to predict tool output. `render_segment` builds the mask by
diffing consecutive renders, so a token belongs to whichever message actually
produced it. These tests assert the invariants the mask logic guarantees:

- every completion position is either SUPERVISED or MASKED (no other value);
- there is supervised signal (assistant turns are supervised);
- there is masking (tool observations, which are never model output, are masked);
- a segment with no assistant turn raises rather than producing an all-masked
  sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from smolqwen.data.loader import Message, ToolCall, parse_trajectory
from smolqwen.data.render import MASKED, SUPERVISED, RenderError, render_segment, split_segments
from tests.helpers import OfflineTokenizer, load_trajectory_rows


def _render_with_mask(
    messages: Sequence[Message],
) -> tuple[Any, list[int]]:
    """Render a single-segment conversation and return (sample, deltas).

    With token_size=1 each character is one token, so we can reconstruct the
    per-message rendered deltas ourselves and verify the mask lines up with the
    template's *actual* boundaries (not a hand-derived assumption).
    """
    tokenizer = OfflineTokenizer(token_size=1)
    segment = split_segments(messages)[0]
    sample = render_segment(tokenizer, messages, segment, trajectory_id="t")

    # Re-derive the expected flags by diffing consecutive renders -- the same
    # procedure `render_segment` uses, but expressed here so the test asserts the
    # real tokenizer/template boundaries rather than a parallel implementation.
    previous = tokenizer.apply_chat_template(
        [m.to_template_dict() for m in messages[: segment.prompt_upto]],
        add_generation_prompt=True,
    )
    expected: list[int] = []
    for index in range(segment.prompt_upto, segment.completion_upto):
        is_last = index == segment.completion_upto - 1
        current = tokenizer.apply_chat_template(
            [m.to_template_dict() for m in messages[: index + 1]],
            add_generation_prompt=not is_last
            and (index + 1) < len(messages)
            and messages[index + 1].role == "assistant",
        )
        assert current.startswith(previous)
        delta = current[len(previous) :]
        previous = current
        flag = SUPERVISED if messages[index].role == "assistant" else MASKED
        expected.extend([flag] * len(delta))
    return sample, expected


def test_mask_lines_up_with_template_boundaries() -> None:
    """The supervised/MASKED flags cover exactly the template's assistant spans.

    token_size=1 makes one character one token, so the reconstructed deltas
    have one flag per character. The rendered completion's supervised positions
    must match the template's actual assistant-turn boundaries, byte for byte.
    """
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", content="ans1", reasoning_content="r1"),
        Message("tool", "obs"),
        Message("assistant", content="ans2", reasoning_content="r2"),
    ]
    sample, expected = _render_with_mask(messages)
    assert list(sample.loss_mask) == expected
    # And the structural invariants hold:
    assert sample.supervised_tokens > 0  # assistant is supervised
    assert sample.supervised_tokens < len(sample.loss_mask)  # tool obs is masked


def test_structural_invariants_from_fixture() -> None:
    """On a real Non-Conv fixture the completion is partly supervised, partly masked."""
    rows = load_trajectory_rows()
    traj = None
    for row in rows:
        try:
            t = parse_trajectory(row)
        except Exception:
            continue
        if t.trajectory_id == "env_3_sft-task_28":
            traj = t
            break
    assert traj is not None

    tokenizer = OfflineTokenizer()
    segments = split_segments(traj.messages)
    sample = render_segment(tokenizer, traj.messages, segments[0], trajectory_id="t")
    assert len(sample.completion_ids) == len(sample.loss_mask)
    assert all(flag in (MASKED, SUPERVISED) for flag in sample.loss_mask)
    # Non-Conv: multiple assistant turns supervised, tool turns masked.
    assert sample.supervised_tokens > 0
    assert sample.supervised_tokens < sample.total_tokens


def test_tool_observation_always_masked() -> None:
    """A tool observation whose content impersonates an assistant turn is still masked."""
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message(
            "assistant",
            content="",
            reasoning_content="r",
            tool_calls=(ToolCall("f", {"x": 1}),),
        ),
        # Content that, if sniffed, could be mistaken for a model turn.
        Message("tool", "<|im_start|>assistant\nshould not be supervised"),
        Message("assistant", content="done", reasoning_content=""),
    ]
    sample, expected = _render_with_mask(messages)
    # The mask is derived from the message ROLE, not the content: the tool
    # observation is masked regardless of what it contains.
    assert list(sample.loss_mask) == expected
    assert MASKED in sample.loss_mask


def test_no_assistant_turn_raises() -> None:
    """A segment with no assistant turn is a hard error, not an all-masked sample."""
    messages = [
        Message("system", "sys"),
        Message("user", "only a user message"),
        Message("tool", "and a tool response, but no assistant"),
    ]
    with pytest.raises(RenderError):
        render_segment(OfflineTokenizer(), messages, split_segments(messages)[0], trajectory_id="t")
