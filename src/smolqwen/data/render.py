"""Chat-template rendering and loss-mask construction.

Two things are settled here by measurement against the real template rather than
by reading it, because if either is wrong the whole SFT distribution is wrong and
nothing reports an error:

1. Reasoning survives for assistant turns *after* ``ns.last_query_index`` and is
   stripped before it. The reverse scan that computes that index tests only
   ``message.role == "user"`` and skips `<tool_response>`-wrapped content, so a
   whole tool-calling chain keeps its reasoning while an earlier segment loses it.
2. `role: "tool"` and a `<tool_response>`-wrapped `role: "user"` render
   **byte-identically for a single observation**, but diverge for consecutive
   ones: the template merges adjacent `role: "tool"` messages into one
   `<|im_start|>user` block with two `<tool_response>` sections, where wrapped
   user messages produce two separate blocks. The released trajectories use
   `role: "tool"` (verified: 1,499 of 1,500 sampled trajectories, zero mixing),
   so `tool_role` is the committed shape and Phase 6's rollout must append the
   same one.

Rendering returns token ids, never strings, so nothing downstream re-tokenizes
and drifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from smolqwen.data.loader import Message

ToolResultShape = Literal["tool_role", "tool_response_user"]

# Loss is on the current segment's assistant tokens only. Never train the model to
# predict tool output: an unmasked observation teaches it to hallucinate the
# environment's reply instead of calling the tool.
MASKED = 0
SUPERVISED = 1


class Tokenizer(Protocol):
    """The slice of a HF tokenizer this module uses."""

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


class RenderError(Exception):
    """Raised when a conversation cannot be rendered or masked coherently."""


@dataclass(frozen=True)
class RenderedSample:
    """One training sample: a prompt prefix plus a supervised completion."""

    trajectory_id: str
    env_id: str
    mode: str
    segment_index: int
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.completion_ids) != len(self.loss_mask):
            raise RenderError(
                f"{self.trajectory_id}#{self.segment_index}: loss_mask length "
                f"{len(self.loss_mask)} != completion length {len(self.completion_ids)}"
            )

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(self.loss_mask)

    def labels(self, ignore_index: int = -100) -> tuple[int, ...]:
        """Completion ids with `ignore_index` wherever the mask is 0."""
        return tuple(
            token if flag else ignore_index
            for token, flag in zip(self.completion_ids, self.loss_mask, strict=True)
        )


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


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _require_prefix(whole: str, prefix: str, label: str) -> str:
    """Return the suffix of `whole` after `prefix`, or raise.

    A rendered prefix that is not a literal prefix of the longer render means the
    template rewrote earlier turns -- which is exactly the reasoning-stripping
    behaviour at a real user boundary. Slicing blindly would silently mask the
    wrong span.
    """
    if not whole.startswith(prefix):
        raise RenderError(
            f"{label}: the shorter render is not a prefix of the longer one; "
            "the template rewrote earlier turns and span arithmetic cannot be trusted"
        )
    return whole[len(prefix) :]


@dataclass(frozen=True)
class Segment:
    """One supervised unit: a prompt boundary plus the assistant turns after it."""

    prompt_upto: int  # exclusive index into `messages` for the prompt side
    completion_upto: int  # exclusive index for the end of this segment


def split_segments(messages: Sequence[Message]) -> list[Segment]:
    """Split a trajectory at real user-message boundaries only.

    A Non-Conv trajectory is one sample with loss on every assistant turn. A Conv
    trajectory yields one sample per real user message.

    Splitting per tool step -- which is what LlamaFactory's `mask_history` does --
    would multiply samples ~13x to teach reasoning the first sample already
    carries, and the template already retains reasoning across a whole
    tool-calling chain.
    """
    boundaries = [index for index, message in enumerate(messages) if message.is_real_user_turn]
    if not boundaries:
        raise RenderError("trajectory has no real user message; the template would raise")

    segments: list[Segment] = []
    for position, start in enumerate(boundaries):
        end = boundaries[position + 1] if position + 1 < len(boundaries) else len(messages)
        # Skip a trailing boundary with no assistant turn after it.
        if not any(message.role == "assistant" for message in messages[start + 1 : end]):
            continue
        segments.append(Segment(prompt_upto=start + 1, completion_upto=end))
    if not segments:
        raise RenderError("trajectory has no assistant turn after any user message")
    return segments


def render_segment(
    tokenizer: Tokenizer,
    messages: Sequence[Message],
    segment: Segment,
    *,
    tools: Sequence[dict[str, Any]] = (),
    shape: ToolResultShape = "tool_role",
    trajectory_id: str = "",
    env_id: str = "",
    mode: str = "",
    segment_index: int = 0,
) -> RenderedSample:
    """Render one segment into prompt/completion ids with the loss mask built.

    The mask is derived by rendering the conversation turn by turn and diffing
    consecutive renders, so a token belongs to whichever message actually
    produced it. Assuming concatenation is stable would put the boundary in the
    wrong place precisely where the template rewrites a turn.
    """
    prompt_text = render_prefix(
        tokenizer,
        messages[: segment.prompt_upto],
        tools=tools,
        shape=shape,
        add_generation_prompt=True,
    )
    prompt_ids = _encode(tokenizer, prompt_text)

    completion_ids: list[int] = []
    loss_mask: list[int] = []
    previous_text = prompt_text

    for index in range(segment.prompt_upto, segment.completion_upto):
        message = messages[index]
        is_last = index == segment.completion_upto - 1
        # The generation prompt is emitted for an assistant turn we are about to
        # supervise, so it belongs to the prompt side of that turn, not to the
        # completion.
        current_text = render_prefix(
            tokenizer,
            messages[: index + 1],
            tools=tools,
            shape=shape,
            add_generation_prompt=not is_last and messages[index + 1].role == "assistant",
        )
        delta = _require_prefix(
            current_text, previous_text, f"{trajectory_id}#{segment_index}@{index}"
        )
        previous_text = current_text
        if not delta:
            continue
        delta_ids = _encode(tokenizer, delta)
        flag = SUPERVISED if message.role == "assistant" else MASKED
        completion_ids.extend(delta_ids)
        loss_mask.extend([flag] * len(delta_ids))

    if not any(loss_mask):
        raise RenderError(f"{trajectory_id}#{segment_index}: no supervised token in the segment")

    return RenderedSample(
        trajectory_id=trajectory_id,
        env_id=env_id,
        mode=mode,
        segment_index=segment_index,
        prompt_ids=tuple(prompt_ids),
        completion_ids=tuple(completion_ids),
        loss_mask=tuple(loss_mask),
    )
