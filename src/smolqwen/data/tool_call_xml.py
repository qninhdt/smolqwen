"""Serialize and parse the Qwen3.5 tool-call XML syntax.

The chat template emits tool calls as XML, not JSON:

    <tool_call>
    <function=NAME>
    <parameter=K>
    VALUE
    </parameter>
    </function>
    </tool_call>

This is the format the model is trained and prompted to emit, so the SFT samples
must serialize it identically (Phase 2 owns the syntax) and the rollout/eval loop
will parse the model's output (Phases 5-6). Both directions live here so the two
never drift apart; `serialize_tool_call` is byte-compatible with the template's
own serialisation (verified by the round-trip test).
"""

from __future__ import annotations

import re
from typing import Any

from smolqwen.data.loader import ToolCall

# <tool_call> ... </tool_call> blocks, each possibly containing several
# <function=NAME> ... </function> entries.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"<function=(?P<name>[^\s>]+)>(?P<body>.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=(?P<key>[^\s>]+)>(?P<value>.*?)</parameter>", re.DOTALL)


def serialize_tool_call(call: ToolCall) -> str:
    """Render a tool call in the exact shape the chat template emits."""
    parts = ["<tool_call>\n<function=", call.name, ">"]
    for key, value in call.arguments.items():
        rendered = _render_argument(value)
        parts.append(f"\n<parameter={key}>\n{rendered}\n</parameter>")
    parts.append("\n</function>\n</tool_call>")
    return "".join(parts)


def _render_argument(value: Any) -> str:
    if isinstance(value, (dict, list)):
        import json as _json

        return _json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse zero or more `<tool_call>...</tool_call>` blocks from `text`.

    The model may concatenate several tool calls in one turn; the tour loop emits
    a fresh `<tool_call><function=...>` per call (see the template's `loop.first`
    branch). Each `<function>` becomes one `ToolCall`.
    """
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_BLOCK.finditer(text):
        body = block.group("body")
        for function in _FUNCTION.finditer(body):
            name = function.group("name")
            arguments: dict[str, Any] = {}
            for parameter in _PARAMETER.finditer(function.group("body")):
                arguments[parameter.group("key")] = _parse_value(
                    parameter.group("value").strip("\n")
                )
            calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def _parse_value(raw: str) -> Any:
    """Best-effort coercion of a parameter value to a scalar, preserving strings.

    Multi-line values (a string containing a newline) are kept verbatim; a value
    that parses as JSON (int, bool, number, list of scalars) is decoded.
    """
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        import json

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    if stripped in ("true", "false"):
        return stripped == "true"
    try:
        if stripped == stripped.strip() and " " not in stripped and "\n" not in stripped:
            return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return raw
