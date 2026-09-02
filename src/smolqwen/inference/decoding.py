"""The decode seam: sampled token ids to one assistant message.

Rollout and evaluation both cross this boundary and crossed it differently, so a
completion that advanced one path's episode could stall the other's. Two axes were
tangled together, and measurement separates them.

**The decode flag decides almost nothing.** In Qwen3.5 only `<|im_end|>` and
`<|endoftext|>` are special tokens; `</think>`, `<tool_call>` and
`<tool_response>` are ordinary vocabulary entries. So `skip_special_tokens`
controls exactly one thing: whether the turn-end marker survives into the text.
Keeping it is actively wrong on both sides. Rollout stored it inside
`Message.content`, and the next re-render then emitted `<|im_end|><|im_end|>`;
evaluation's markers are string equalities (`tool_calls.py:41,47`), so a trailing
marker makes `TASK_FINISHED` fail to match and a JSON call's last line fail to
parse. `True` is therefore right for both, and the structural tags every parser
needs are untouched by it.

**The message shape decides the mask.** The chat template's assistant branch
splits on `</think>` only when `reasoning_content` is absent
(`qwen35_chat_template.jinja:89-99`), and for a well-formed turn both shapes
render identically. They diverge when generation is truncated before `</think>`
closes: as raw `content` the template synthesizes an empty reasoning block and
re-renders the model's own tokens in a different position, which the mask builder
sees as drift. Measured on the real tokenizer over a two-turn episode whose first
turn is truncated: the split shape yields 12 supervised tokens of 52 with no
fork, the raw shape 8 of 52 with one fork and 5 drift tokens. Four model tokens
silently demoted to mask 0, and the demotion compounds per turn.

So: decode with `skip_special_tokens=True`, and store the split shape. One
convention, satisfying both consumers, rather than each path's local habit.
"""

from __future__ import annotations

from typing import Any

from smolqwen.data.loader import Message


def decode_completion(tokenizer: Any, token_ids: Any) -> str:
    """Sampled ids as text, with the turn-end marker dropped and nothing else.

    `token_ids` is passed through untouched: the HuggingFace path hands over a
    tensor row and the rollout path a tuple of ints, and a tokenizer handles both.
    """
    return str(tokenizer.decode(token_ids, skip_special_tokens=True))


def split_generation_continuation(text: str) -> tuple[str, str]:
    """Split a continuation generated after the template's opening `<think>`.

    Qwen3.5's generation prompt ends in ``<think>\\n``, so the sampled text holds
    the reasoning body and its closing tag but not the opening one. A backend
    returning a complete block is tolerated for deterministic tests and alternate
    engines. If the block never closes -- usually token truncation -- the whole
    continuation is kept as reasoning, which is what stops the template from
    re-rendering those tokens somewhere else on the next turn.
    """
    from smolqwen.env.parse import split_reasoning

    if "<think>" in text:
        return split_reasoning(text)
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.strip(), content.strip()
    return text.strip(), ""


def assistant_message(text: str) -> Message:
    """The assistant turn in the shape the chat template re-renders faithfully."""
    reasoning, content = split_generation_continuation(text)
    return Message(role="assistant", content=content, reasoning_content=reasoning)
