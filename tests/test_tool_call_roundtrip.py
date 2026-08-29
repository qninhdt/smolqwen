"""Tool-call parse/serialize round-trip through the Qwen3.5 XML syntax.

The chat template emits tool calls as `<tool_call><function=NAME>...</function>`,
not JSON. The SFT samples must serialize that shape (Phase 2 owns the syntax),
and the rollout/eval loop parses the model's output (Phases 5-6). This test
proves the two directions are byte-compatible: a `ToolCall` serializes to the
exact XML the template emits, and parsing that XML recovers the original call.
"""

from __future__ import annotations

from smolqwen.data.loader import ToolCall
from smolqwen.data.tool_call_xml import parse_tool_calls, serialize_tool_call


def test_round_trip_scalar_arguments() -> None:
    call = ToolCall(name="get_max", arguments={"x": 1, "y": 2})
    assert parse_tool_calls(serialize_tool_call(call)) == [call]


def test_round_trip_string_and_multiline() -> None:
    call = ToolCall(name="send", arguments={"to": "alice", "body": "hello\nworld"})
    assert parse_tool_calls(serialize_tool_call(call)) == [call]


def test_round_trip_nested_object_argument() -> None:
    call = ToolCall(name="create", arguments={"user": {"id": 1, "tags": ["a", "b"]}})
    assert parse_tool_calls(serialize_tool_call(call)) == [call]


def test_round_trip_multiple_tool_calls() -> None:
    calls = [
        ToolCall(name="a", arguments={"x": 1}),
        ToolCall(name="b", arguments={"y": "two"}),
    ]
    text = "".join(serialize_tool_call(c) for c in calls)
    assert parse_tool_calls(text) == calls


def test_parse_empty_text_returns_no_calls() -> None:
    assert parse_tool_calls("no tool calls here") == []


def test_serialize_matches_template_shape() -> None:
    """The serialized XML is byte-compatible with what the chat template emits.

    The template's `loop.first` branch emits `<tool_call>\n<function=NAME>\n` and
    each parameter as `<parameter=K>\nVALUE\n</parameter>`. The round-trip test
    above already proves parse(serialize(call)) == call; this one pins the exact
    text shape so a drift in either direction is caught.
    """
    call = ToolCall(name="f", arguments={"a": 1, "b": "two"})
    text = serialize_tool_call(call)
    assert text.startswith("<tool_call>\n<function=f>")
    assert "<parameter=a>\n1\n</parameter>" in text
    assert "<parameter=b>\ntwo\n</parameter>" in text
    assert text.endswith("</function>\n</tool_call>")
