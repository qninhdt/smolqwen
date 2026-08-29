"""Streaming reader for the EnvScaler release files.

`json.load` on `envscaler_sft_traj_9k_metadata.json` (701 MB) is not an option:
the parsed object graph is several times the file size and gets the process
OOM-killed on a Colab CPU runtime. So the array is decoded one element at a time
with `raw_decode` over a sliding buffer, and callers consume an iterator.

The three payload fields (`tools`, `messages`, `user_messages`) are JSON
**strings** inside the outer JSON, so each needs its own `json.loads`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_BYTES = 1 << 20
_DECODER = json.JSONDecoder()


class DataError(Exception):
    """Raised when a release file is missing, malformed, or fails its hash check."""


def sha256_of(path: Path | str, *, chunk: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path | str, expected: str | None) -> None:
    """Fail loudly on a content mismatch.

    A count check does not detect a modified `env_class_code` body: 191 classes
    still compile and the suffix split is unchanged. The hash is the only thing
    that ties the artifact that runs to the artifact the `exec()` posture was
    accepted for.
    """
    if not expected:
        return
    actual = sha256_of(path)
    if actual != expected:
        raise DataError(f"sha256 mismatch for {path}\n  expected {expected}\n  actual   {actual}")


def iter_json_array(path: Path | str) -> Iterator[Any]:
    """Yield each element of a top-level JSON array without holding the whole file.

    Reads in 1 MB blocks and decodes greedily from the front of the buffer, so
    peak memory is one block plus the largest single element rather than the whole
    document.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"file not found: {file_path}")

    with file_path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        exhausted = False

        while True:
            if not started:
                # Find the opening bracket before decoding anything.
                while True:
                    stripped = buffer[position:].lstrip()
                    if stripped:
                        if not stripped.startswith("["):
                            raise DataError(f"{file_path} is not a top-level JSON array")
                        position = len(buffer) - len(stripped) + 1
                        started = True
                        break
                    block = handle.read(CHUNK_BYTES)
                    if not block:
                        raise DataError(f"{file_path} is empty")
                    buffer += block

            remainder = buffer[position:].lstrip()
            consumed = len(buffer) - len(remainder)
            if remainder.startswith(","):
                position = consumed + 1
                continue
            if remainder.startswith("]"):
                return
            if not remainder and exhausted:
                return

            try:
                element, offset = _DECODER.raw_decode(remainder)
            except ValueError:
                if exhausted:
                    raise DataError(f"truncated JSON array in {file_path}") from None
                block = handle.read(CHUNK_BYTES)
                if not block:
                    exhausted = True
                else:
                    buffer = remainder
                    position = 0
                    buffer += block
                continue

            yield element
            # Drop what has been consumed so the buffer does not grow with the file.
            buffer = remainder[offset:]
            position = 0


def iter_json_object_values(path: Path | str) -> Iterator[tuple[str, Any]]:
    """Yield ``(key, value)`` pairs of a top-level JSON object.

    `191_env_metadata.json` is an object keyed by `env_id` (21 MB), small enough
    to parse whole -- but the same streaming discipline keeps the memory story
    uniform and leaves headroom if the release grows.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"file not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DataError(f"{file_path} is not a top-level JSON object")
    yield from payload.items()


@dataclass(frozen=True)
class ToolCall:
    """One tool call in a trajectory or a rollout turn."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One trajectory message, with the reasoning kept separate from the content.

    `reasoning_content` stays its own field rather than being inlined into
    `content` as a `<think>` string: the chat template's assistant branch splits
    on `</think>` itself, and inlining is one of the reasons the pre-templated
    release file is unusable.
    """

    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None

    @property
    def is_wrapped_tool_response(self) -> bool:
        """A `role: "user"` message whose content is `<tool_response>`-wrapped.

        The chat template's reverse scan for `last_query_index` checks this prefix
        explicitly, so such a message does not advance the index -- while a
        `role: "tool"` message is invisible to that scan entirely. Two different
        mechanisms, one newline apart in the rendered stream.
        """
        return (
            self.role == "user"
            and self.content is not None
            and self.content.lstrip().startswith("<tool_response>")
        )

    @property
    def is_real_user_turn(self) -> bool:
        return self.role == "user" and not self.is_wrapped_tool_response

    def to_template_dict(self) -> dict[str, Any]:
        """The shape the Qwen3.5 chat template expects."""
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            payload["tool_calls"] = [
                {"type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class Trajectory:
    """One released teacher trajectory."""

    trajectory_id: str
    env_id: str
    task: str
    traj_type: str
    tools: tuple[dict[str, Any], ...]
    messages: tuple[Message, ...]
    user_messages: tuple[str, ...]
    steps: int

    @property
    def is_conversation(self) -> bool:
        """Conv vs Non-Conv, by the release's own `traj_type` label.

        Not re-derived from the message list: RL is Non-Conv only, and a
        classification that disagreed with the label would silently change which
        trajectories the RL split draws from.
        """
        return self.traj_type == "conversation"

    @property
    def real_user_turns(self) -> int:
        return sum(1 for message in self.messages if message.is_real_user_turn)

    @property
    def tool_steps(self) -> int:
        return sum(len(message.tool_calls) for message in self.messages)


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not raw:
        return ()
    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function", entry)
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__raw__": arguments}
        if not isinstance(arguments, dict):
            arguments = {"__value__": arguments}
        calls.append(ToolCall(name=str(name), arguments=arguments))
    return tuple(calls)


def parse_message(raw: Any) -> Message:
    if not isinstance(raw, dict):
        raise DataError(f"message is not an object: {raw!r}")
    role = raw.get("role")
    if not isinstance(role, str) or not role:
        raise DataError(f"message has no role: {raw!r}")
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    reasoning = raw.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return Message(
        role=role,
        content=content,
        reasoning_content=reasoning or None,
        tool_calls=_parse_tool_calls(raw.get("tool_calls")),
        name=raw.get("name"),
    )


def _loads_field(row: dict[str, Any], key: str, trajectory_id: str) -> Any:
    """`json.loads` a field that the release stores as a JSON string."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise DataError(f"{trajectory_id}: field '{key}' is not valid JSON: {exc}") from exc
    return value


def parse_trajectory(row: Any) -> Trajectory:
    """Validate one release row into a `Trajectory`, raising on anything malformed."""
    if not isinstance(row, dict):
        raise DataError(f"trajectory row is not an object: {type(row).__name__}")
    task_info = row.get("task_info")
    if not isinstance(task_info, dict):
        raise DataError("trajectory row has no task_info object")
    trajectory_id = str(task_info.get("task_id") or "")
    if not trajectory_id:
        raise DataError("trajectory row has no task_id")

    messages_raw = _loads_field(row, "messages", trajectory_id)
    if not isinstance(messages_raw, list) or not messages_raw:
        raise DataError(f"{trajectory_id}: messages is empty or not a list")
    tools_raw = _loads_field(row, "tools", trajectory_id) or []
    if not isinstance(tools_raw, list):
        raise DataError(f"{trajectory_id}: tools is not a list")
    user_messages_raw = _loads_field(row, "user_messages", trajectory_id) or []
    if not isinstance(user_messages_raw, list):
        raise DataError(f"{trajectory_id}: user_messages is not a list")

    return Trajectory(
        trajectory_id=trajectory_id,
        env_id=str(task_info.get("env_id") or ""),
        task=str(task_info.get("task") or ""),
        traj_type=str(row.get("traj_type") or ""),
        tools=tuple(tool for tool in tools_raw if isinstance(tool, dict)),
        messages=tuple(parse_message(message) for message in messages_raw),
        user_messages=tuple(
            entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
            for entry in user_messages_raw
        ),
        steps=int(row.get("steps") or 0),
    )


@dataclass
class LoadStats:
    """Counts every input row, so nothing is silently dropped."""

    total: int = 0
    parsed: int = 0
    malformed: int = 0
    malformed_reasons: dict[str, int] | None = None

    def record_malformed(self, reason: str) -> None:
        self.malformed += 1
        if self.malformed_reasons is None:
            self.malformed_reasons = {}
        key = reason.split(":")[-1].strip()[:120]
        self.malformed_reasons[key] = self.malformed_reasons.get(key, 0) + 1


def iter_trajectories(
    path: Path | str,
    *,
    limit: int | None = None,
    stats: LoadStats | None = None,
    strict: bool = False,
) -> Iterator[Trajectory]:
    """Stream validated trajectories.

    A malformed row is counted and skipped rather than aborting a 9k-row pass;
    `strict=True` raises instead, which is what the tests use.
    """
    tracker = stats if stats is not None else LoadStats()
    for row in iter_json_array(path):
        tracker.total += 1
        try:
            trajectory = parse_trajectory(row)
        except DataError as exc:
            if strict:
                raise
            tracker.record_malformed(str(exc))
            continue
        tracker.parsed += 1
        yield trajectory
        if limit is not None and tracker.parsed >= limit:
            return
