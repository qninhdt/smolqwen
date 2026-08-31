"""Separate inference rendering from full-trajectory SFT rendering.

Inference keeps the tokenizer's native history policy. SFT uses a fail-closed
template adapter that preserves every assistant reasoning block and asks the
tokenizer for exact assistant-token spans.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from smolqwen.data.loader import Message

ToolResultShape = Literal["tool_role", "tool_response_user"]

MASKED = 0
SUPERVISED = 1
IGNORE_INDEX = -100

_HISTORY_CONDITION = "{%- if loop.index0 > ns.last_query_index %}"
_ASSISTANT_BRANCH = '{%- elif message.role == "assistant" %}'
_ASSISTANT_END = "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"


class Tokenizer(Protocol):
    """The slice of a HF tokenizer this module uses."""

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


class RenderError(Exception):
    """Raised when a conversation cannot be rendered or masked coherently."""


@dataclass(frozen=True)
class RenderedSample:
    """One complete, bounded teacher trajectory and its token labels."""

    trajectory_uid: str
    task_id: str
    env_id: str
    mode: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    template_fingerprint: str
    trailing_messages_removed: int = 0

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise RenderError(
                f"{self.trajectory_uid}: labels length {len(self.labels)} "
                f"!= input length {len(self.input_ids)}"
            )
        if not any(label != IGNORE_INDEX for label in self.labels):
            raise RenderError(f"{self.trajectory_uid}: no supervised assistant token")

    @property
    def total_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)


def to_template_messages(
    messages: Sequence[Message], *, shape: ToolResultShape = "tool_role"
) -> list[dict[str, Any]]:
    """Convert parsed messages into template dicts under the committed shape.

    Converting to `tool_response_user` is supported so the equivalence between the
    two shapes stays testable, but it is not the pipeline's input contract: for
    consecutive observations the two render differently, and the release uses
    `role: "tool"`.
    """
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if shape == "tool_response_user" and message.role == "tool":
            content = message.content or ""
            rendered.append(
                {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}
            )
            continue
        rendered.append(message.to_template_dict())
    return rendered


def render_prefix(
    tokenizer: Tokenizer,
    messages: Sequence[Message],
    *,
    tools: Sequence[dict[str, Any]] = (),
    shape: ToolResultShape = "tool_role",
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
) -> str:
    """Render a conversation prefix exactly as inference will see it."""
    payload = to_template_messages(messages, shape=shape)
    result = tokenizer.apply_chat_template(
        payload,
        tools=list(tools) or None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if not isinstance(result, str):
        raise RenderError("apply_chat_template(tokenize=False) did not return a string")
    return result


def training_chat_template(tokenizer: Tokenizer) -> tuple[str, str]:
    """Return the pinned SFT template and its fingerprint, failing on drift."""
    source = getattr(tokenizer, "chat_template", None)
    if not isinstance(source, str) or not source:
        raise RenderError("tokenizer has no chat_template for SFT rendering")
    for needle in (_HISTORY_CONDITION, _ASSISTANT_BRANCH, _ASSISTANT_END):
        count = source.count(needle)
        if count != 1:
            raise RenderError(
                f"chat template drift: expected one occurrence of {needle!r}, got {count}"
            )

    adapted = source.replace(_HISTORY_CONDITION, "{%- if true %}", 1)
    adapted = adapted.replace(
        _ASSISTANT_BRANCH,
        _ASSISTANT_BRANCH + "\n        {%- generation %}",
        1,
    )
    adapted = adapted.replace(
        _ASSISTANT_END,
        "        {{- '<|im_end|>\\n' }}\n        {%- endgeneration %}\n"
        '    {%- elif message.role == "tool" %}',
        1,
    )
    fingerprint = hashlib.sha256(adapted.encode("utf-8")).hexdigest()
    return adapted, fingerprint


def trim_after_last_assistant(messages: Sequence[Message]) -> tuple[tuple[Message, ...], int]:
    """Bound a trajectory at its last assistant target, independent of sentinel text."""
    real_users = [index for index, message in enumerate(messages) if message.is_real_user_turn]
    assistants = [index for index, message in enumerate(messages) if message.role == "assistant"]
    if not real_users:
        raise RenderError("trajectory has no real user message")
    has_supervised_turn = any(user < assistant for user in real_users for assistant in assistants)
    if not assistants or not has_supervised_turn:
        raise RenderError("trajectory has no assistant turn after a real user message")
    last_assistant = assistants[-1]
    bounded = tuple(messages[: last_assistant + 1])
    return bounded, len(messages) - len(bounded)


def _flat_ints(value: Any, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RenderError(f"{field}: expected one rendered conversation")
        value = value[0]
    if not isinstance(value, list):
        raise RenderError(f"{field}: tokenizer returned {type(value).__name__}, expected list")
    return tuple(int(item) for item in value)


def render_training_sample(
    tokenizer: Tokenizer,
    messages: Sequence[Message],
    *,
    tools: Sequence[dict[str, Any]] = (),
    shape: ToolResultShape = "tool_role",
    trajectory_uid: str = "",
    task_id: str = "",
    env_id: str = "",
    mode: str = "",
    training_template: tuple[str, str] | None = None,
) -> RenderedSample:
    """Render one raw row as one sample with loss on every assistant block."""
    bounded, removed = trim_after_last_assistant(messages)
    template, fingerprint = training_template or training_chat_template(tokenizer)
    encoded = tokenizer.apply_chat_template(
        to_template_messages(bounded, shape=shape),
        tools=list(tools) or None,
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if not isinstance(encoded, Mapping):
        raise RenderError("training chat template did not return a mapping")
    input_ids = _flat_ints(encoded.get("input_ids"), field="input_ids")
    mask_value = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    assistant_mask = _flat_ints(mask_value, field="assistant_masks")
    if len(input_ids) != len(assistant_mask):
        raise RenderError(
            f"{trajectory_uid}: assistant mask length {len(assistant_mask)} "
            f"!= input length {len(input_ids)}"
        )
    labels = tuple(
        token if flag == SUPERVISED else IGNORE_INDEX
        for token, flag in zip(input_ids, assistant_mask, strict=True)
    )
    return RenderedSample(
        trajectory_uid=trajectory_uid,
        task_id=task_id,
        env_id=env_id,
        mode=mode,
        input_ids=input_ids,
        labels=labels,
        template_fingerprint=fingerprint,
        trailing_messages_removed=removed,
    )


def render_training_length(
    tokenizer: Tokenizer,
    messages: Sequence[Message],
    *,
    tools: Sequence[dict[str, Any]] = (),
    shape: ToolResultShape = "tool_role",
) -> int:
    """Tokenize the SFT trajectory once without constructing masks or labels."""
    bounded, _ = trim_after_last_assistant(messages)
    template, _ = training_chat_template(tokenizer)
    input_ids = tokenizer.apply_chat_template(
        to_template_messages(bounded, shape=shape),
        tools=list(tools) or None,
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if isinstance(input_ids, Mapping):
        input_ids = input_ids.get("input_ids")
    return len(_flat_ints(input_ids, field="input_ids"))
