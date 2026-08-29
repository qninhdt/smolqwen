"""Backend-neutral parsing for HTTP-normalized tool calls and completion markers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

NormalizedCall = tuple[str, dict[str, Any]]


def parse_normalized_json_calls(text: str) -> list[NormalizedCall]:
    """Parse HttpPolicy's final-line JSON call representation."""

    candidate = text.strip().rsplit("\n", 1)[-1].strip()
    if not candidate.startswith(("[", "{")):
        return []
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    calls = payload if isinstance(payload, list) else [payload]
    parsed: list[NormalizedCall] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return []
        name = call.get("name")
        arguments = call.get("arguments", call.get("args", {}))
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            return []
        parsed.append((name, dict(arguments)))
    return parsed


def is_completion_signal(text: str) -> bool:
    """Accept only the explicit turn markers used by the pinned BFCL runner."""

    body = text.rsplit("</think>", 1)[-1].strip().casefold()
    body = body.strip("<> \t\r\n.!?")
    return body in {"task_finished", "task finished", "task completed", "finish"}


def is_error_signal(text: str) -> bool:
    body = text.rsplit("</think>", 1)[-1].strip().casefold()
    return body.strip("<> \t\r\n.!?") == "task_error"
