"""Parse the model's turn into a call, and classify what is wrong when it is not.

The chat template emits tool calls as XML — `<tool_call><function=NAME>
<parameter=K>V</parameter></function></tool_call>` — and `data/tool_call_xml.py`
already owns that syntax in both directions, so this module reuses it rather than
writing a second parser that can drift from the one Phase 2 trained against.

What this adds is **classification**. A rollout has to react differently to four
outcomes, and collapsing them loses the information Phase 7 reports:

- `ok` — a call the environment can execute;
- `unknown_tool` — well-formed, but names something the environment does not have;
- `bad_arguments` — right tool, arguments the signature cannot take;
- `malformed_syntax` — an unclosed tag or an empty function name;
- `no_call` — plain text with no call in it, which for a Conv episode is a message
  to the user rather than an error.

Nothing here raises into the rollout loop. An invalid call becomes an observation
the model can read and retry from, which is also how the released trajectories
behave. Invalid-call rates are a reported metric in Phase 7 and never part of the
reward — a model that is penalised for malformed syntax learns to avoid emitting
tool calls at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from smolqwen.data.tool_call_xml import parse_tool_calls

Outcome = Literal["ok", "unknown_tool", "bad_arguments", "malformed_syntax", "no_call"]

# An opening tag with no matching close: the model was cut off mid-call, which is
# a different failure from a well-formed call to a tool that does not exist.
_OPEN_TOOL_CALL = re.compile(r"<tool_call>")
_CLOSE_TOOL_CALL = re.compile(r"</tool_call>")
_OPEN_FUNCTION = re.compile(r"<function=")
_CLOSE_FUNCTION = re.compile(r"</function>")
_EMPTY_FUNCTION = re.compile(r"<function=\s*>")

# `<think>` may open without closing when generation hits the token cap.
_THINK_BLOCK = re.compile(r"<think>(?P<body>.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class ParsedTurn:
    """One assistant turn, classified.

    `content` is the text outside the reasoning block and outside the call, which
    is what a Conv episode delivers to the user.
    """

    outcome: Outcome
    name: str | None = None
    arguments: dict[str, Any] | None = None
    content: str = ""
    reasoning: str = ""
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"

    @property
    def is_invalid_call(self) -> bool:
        """Whether this counts toward the invalid-call rate.

        `no_call` does not: a turn addressed to the user is legitimate in a Conv
        episode, and counting it as invalid would make the metric a measure of
        episode type rather than of model behaviour.
        """
        return self.outcome in ("unknown_tool", "bad_arguments", "malformed_syntax")

    def observation(self) -> str:
        """What to hand back to the model for an invalid call.

        Names the failure and what was expected, because an observation the model
        cannot act on is a wasted step out of a capped budget.
        """
        if self.ok:
            raise ValueError("observation() is for invalid calls; this turn parsed cleanly")
        return f"Error: {self.outcome}: {self.reason or 'no detail'}"


def split_reasoning(text: str) -> tuple[str, str]:
    """Return `(reasoning, remainder)`, tolerating an unclosed `<think>`.

    An unclosed block means generation was truncated; the reasoning is still
    usable and the remainder is empty, which is more useful than discarding both.
    """
    match = _THINK_BLOCK.search(text)
    if match:
        return match.group("body").strip(), (text[: match.start()] + text[match.end() :]).strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def _malformed_reason(text: str) -> str | None:
    """Detect a structurally broken call before trying to interpret it."""
    if _EMPTY_FUNCTION.search(text):
        return "empty function name"
    if len(_OPEN_TOOL_CALL.findall(text)) > len(_CLOSE_TOOL_CALL.findall(text)):
        return "unclosed <tool_call> tag"
    if len(_OPEN_FUNCTION.findall(text)) > len(_CLOSE_FUNCTION.findall(text)):
        return "unclosed <function=...> tag"
    if "<tool_call>" in text and "<function=" not in text:
        return "<tool_call> block contains no <function=NAME>"
    return None


def parse_turn(
    text: str,
    *,
    available_tools: frozenset[str] | None = None,
    signature_lookup: Any = None,
) -> ParsedTurn:
    """Parse and classify one assistant turn.

    `available_tools` enables `unknown_tool` classification; `signature_lookup` is
    a callable `name -> bound method` enabling `bad_arguments`. Both are optional
    so a parse-only caller (a metrics pass over logged turns) needs neither.
    """
    reasoning, remainder = split_reasoning(text)

    malformed = _malformed_reason(text)
    if malformed is not None:
        return ParsedTurn(
            "malformed_syntax", content=remainder, reasoning=reasoning, reason=malformed
        )

    calls = parse_tool_calls(text)
    if not calls:
        return ParsedTurn("no_call", content=remainder, reasoning=reasoning)

    # Upstream takes the first call when several are present, and the system prompt
    # tells the model not to make parallel calls. Matching that keeps the step
    # accounting identical to the trajectories the model was trained on.
    call = calls[0]
    content = _content_outside_calls(remainder)

    if not call.name:
        return ParsedTurn(
            "malformed_syntax",
            content=content,
            reasoning=reasoning,
            reason="empty function name",
        )

    if available_tools is not None and call.name not in available_tools:
        return ParsedTurn(
            "unknown_tool",
            name=call.name,
            arguments=dict(call.arguments),
            content=content,
            reasoning=reasoning,
            reason=f"{call.name!r} is not one of this environment's tools",
        )

    arguments = dict(call.arguments)
    if signature_lookup is not None:
        from smolqwen.env.tools import coerce_arguments

        method = signature_lookup(call.name)
        if method is None:
            return ParsedTurn(
                "unknown_tool",
                name=call.name,
                arguments=arguments,
                content=content,
                reasoning=reasoning,
                reason=f"{call.name!r} is not callable on this environment",
            )
        coerced = coerce_arguments(method, arguments)
        if not coerced.ok:
            return ParsedTurn(
                "bad_arguments",
                name=call.name,
                arguments=arguments,
                content=content,
                reasoning=reasoning,
                reason=coerced.reason,
            )
        arguments = coerced.arguments

    return ParsedTurn(
        "ok", name=call.name, arguments=arguments, content=content, reasoning=reasoning
    )


def _content_outside_calls(text: str) -> str:
    """Text with every `<tool_call>` block removed."""
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
