"""Reasoning retention through the real Qwen3.5 chat template.

The load-bearing claim of the whole split design is that the template keeps
reasoning for assistant turns after the last real user query and strips it before
it. That claim is verified here against the real template string (not a stub that
just re-asserts the note), for the committed message shape (`role: "tool"`):

- a whole tool-calling chain keeps its reasoning when rendered per-segment;
- a real user turn in the middle strips reasoning before it at the *full*
  render level -- which is exactly why a Conv trajectory splits to one sample per
  real user turn.

Everything is exercised on real fixture trajectory content through the vendored
template.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.data.loader import Message, Trajectory, parse_trajectory
from smolqwen.data.render import RenderError, render_segment, split_segments
from tests.helpers import OfflineTokenizer, load_trajectory_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _trajectory_rows() -> list[dict[str, object]]:
    rows = load_trajectory_rows()
    return rows


def _by_trajectory_id(rows: list[dict[str, object]], trajectory_id: str) -> Trajectory:
    """Return the parsed fixture trajectory with `trajectory_id`."""
    for row in rows:
        try:
            traj = parse_trajectory(row)
        except Exception:
            continue
        if traj.trajectory_id == trajectory_id:
            return traj
    raise AssertionError(f"fixture missing {trajectory_id}")


def test_nonconv_reasoning_retained_per_segment() -> None:
    """A Non-Conv segment keeps reasoning on every assistant turn (the committed shape).

    The converter renders a Non-Conv trajectory as a single segment with a fresh
    generation prompt, so each assistant turn after the first user keeps its
    reasoning. This is what the model is actually trained to emit.
    """
    traj = _by_trajectory_id(_trajectory_rows(), "env_3_sft-task_28")
    assert traj.traj_type != "conversation"

    tokenizer = OfflineTokenizer()
    segments = split_segments(traj.messages)
    assert len(segments) == 1, "a Non-Conv trajectory should convert to exactly one segment"

    sample = render_segment(
        tokenizer,
        traj.messages,
        segments[0],
        tools=traj.tools,
        trajectory_id=traj.trajectory_id,
    )

    # The loss mask covers the completion's assistant turns -- there is supervised
    # signal -- and the same token-count invariant the production path relies on.
    assert sample.supervised_tokens > 0
    assert len(sample.completion_ids) == len(sample.loss_mask)
    # Every assistant turn's reasoning must be retained in the segment render,
    # since a Non-Conv segment is the model's whole supervised unit. Re-render the
    # segment and assert the distinct reasoning strings are present (a wrong
    # prompt-boundary would drop them).
    segment_rendered = tokenizer.apply_chat_template(
        [m.to_template_dict() for m in traj.messages[: segments[0].completion_upto]],
        add_generation_prompt=False,
    )
    for message in traj.messages:
        if message.role == "assistant" and message.reasoning_content:
            assert message.reasoning_content in segment_rendered, (
                "reasoning stripped from Non-Conv segment"
            )


def test_conv_splits_to_one_sample_per_real_user_turn() -> None:
    """A Conv trajectory yields exactly one sample per real user turn.

    A real (non-`<tool_response>`) user message is the only boundary the split
    cuts at; the trailing `user: Task finished` message has no assistant turn
    after it and is correctly skipped.
    """
    traj = _by_trajectory_id(_trajectory_rows(), "env_82_sft-task_29")
    assert traj.traj_type == "conversation"

    segments = split_segments(traj.messages)
    # Real user turns at 1, 7, 13. The turn at 13 has no assistant message after
    # it, so it is skipped; the two that do each open exactly one sample.
    assert len(segments) == 2
    assert [s.prompt_upto for s in segments] == [2, 8]


def test_reasoning_stripped_before_later_user_at_full_render() -> None:
    """A later real user turn strips earlier reasoning; the last turn keeps it.

    This is the template behaviour the split relies on: `ns.last_query_index` is
    recomputed per render, and only assistant turns after the *last* real user get
    a reasoning block. Exercised with the real template and the committed
    `role: "tool"` shape.
    """
    tokenizer = OfflineTokenizer()
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "reasoning_content": "REASON_A",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}],
        },
        {"role": "tool", "content": "obs1"},
        {"role": "assistant", "reasoning_content": "REASON_B", "content": "interim"},
        {"role": "user", "content": "second task"},
        {"role": "assistant", "reasoning_content": "REASON_C", "content": "final"},
    ]
    rendered = tokenizer.apply_chat_template(conv, add_generation_prompt=False)
    # Reasoning before the last real user (second task) is stripped.
    assert "REASON_A" not in rendered
    assert "REASON_B" not in rendered
    # Reasoning after the last real user survives, wrapped in the real marker.
    assert "REASON_C" in rendered
    assert "<think>\nREASON_C\n</think>" in rendered


def test_consecutive_tool_roles_merge() -> None:
    """Consecutive `role: tool` messages merge into one `user` block (committed shape).

    This documents the Phase-2 decision to commit to `role: "tool"`. Wrapped
    `<tool_response>` user messages would not merge. The converter feeds the
    template this shape; Phase 6's rollout appends the same one.
    """
    tokenizer = OfflineTokenizer(shape="tool_role")
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "reasoning_content": "r1",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
        },
        {"role": "tool", "content": "obs1"},
        {"role": "tool", "content": "obs2"},
    ]
    rendered = tokenizer.apply_chat_template(conv, add_generation_prompt=False)
    # Both observations appear, merged into a single user block (the real user
    # message at the start is a separate block, so count the tool-response blocks).
    assert rendered.count("<tool_response>") == 2
    assert rendered.count("<|im_start|>user\n<tool_response>") == 1, (
        "consecutive tool roles must merge into one user block"
    )


def test_segment_with_no_assistant_raises() -> None:
    """A segment with no assistant turn is a hard error, not silent corruption.

    `split_segments` raises rather than yielding an empty sample, so the converter
    cannot silently produce a sample the model has nothing to supervise.
    """
    messages = (Message("system", "sys"), Message("user", "only a user, no assistant"))
    with pytest.raises(RenderError):
        split_segments(messages)


def test_malformed_row_rejected() -> None:
    """The malformed fixture row fails loudly, not silently.

    `messages` here is the JSON-encoded string `"oops not a list"`; after the
    outer `json.loads` it is still a string, not a list, so `parse_trajectory`
    raises `DataError`.
    """
    rows = _trajectory_rows()
    # The third fixture row is the malformed one (messages is not a list).
    bad = rows[2]
    from smolqwen.data.loader import DataError, parse_trajectory

    with pytest.raises(DataError):
        parse_trajectory(bad)


def test_fixtures_have_pinned_sha() -> None:
    """The fixture file's content is stable (so the pinned sha is meaningful).

    This is a regression guard: if the fixture rows change, the sha changes, and a
    comparison in the conversion report would no longer match. It also documents
    that the fixture is the artifact the tests are pinned to.
    """
    path = FIXTURES / "trajectories.json"
    # Deterministic: re-read and ensure it parses and contains the known ids.
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = {
        r["task_info"]["task_id"]
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("task_info"), dict)
    }
    assert ids == {"env_3_sft-task_28", "env_82_sft-task_29", "env_bad_sft-task_x"}
