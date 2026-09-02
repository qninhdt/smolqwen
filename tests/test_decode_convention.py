"""One decode convention, chosen by measurement rather than by local habit.

Rollout decoded `skip_special_tokens=False` and evaluation `True`, and the plan
recorded this as a structural divergence where "the rollout convention wins
because the mask depends on it." Measurement separates two axes the claim
conflates, and the answer is neither path's habit:

- The **decode flag** is nearly inert in Qwen3.5, because only the turn-end
  markers are special tokens. Keeping them is wrong for both consumers.
- The **message shape** -- `reasoning_content` split out versus raw text in
  `content` -- is what moves the mask, and only for a truncated turn.

These tests pin both findings against the real tokenizer and the real template,
so a future edit that reverts either one fails here rather than in a training run.
"""

from __future__ import annotations

from typing import Any

import pytest

from smolqwen.data.loader import Message
from smolqwen.eval.tool_calls import is_completion_signal, is_error_signal
from smolqwen.inference.decoding import (
    assistant_message,
    decode_completion,
    split_generation_continuation,
)

pytestmark = pytest.mark.dataset

MARKERS = ("</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")


def encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def test_only_the_turn_end_markers_are_special_tokens(tokenizer: Any) -> None:
    """Why the decode flag decides so little: the structural tags are ordinary.

    Every parser downstream reads `</think>`, `<tool_call>` and `<tool_response>`,
    and `skip_special_tokens` never touches any of them. It removes `<|im_end|>`
    and nothing else that matters here.
    """
    special = set(tokenizer.all_special_ids)
    for marker in MARKERS:
        ids = encode(tokenizer, marker)
        assert len(ids) == 1, marker
        assert ids[0] not in special, f"{marker} became a special token; the seam changed"
        assert decode_completion(tokenizer, ids) == marker

    end_of_turn = encode(tokenizer, "<|im_end|>")
    assert end_of_turn[0] in special
    assert decode_completion(tokenizer, end_of_turn) == ""


def test_a_surviving_turn_end_marker_breaks_every_string_equality_signal(
    tokenizer: Any,
) -> None:
    """What keeping the marker costs evaluation.

    `tool_calls.py:41,47` compare a stripped body against fixed strings, and
    `bfcl.py:335` parses the final line as JSON. A trailing `<|im_end|>` defeats
    all three, so the benchmark would neither advance its turn nor see the call.
    """
    from smolqwen.eval.adapters.bfcl import BfclMultiTurnAdapter

    finished = encode(tokenizer, "done\n</think>\n\nTASK_FINISHED<|im_end|>")
    failed = encode(tokenizer, "oops\n</think>\n\nTASK_ERROR<|im_end|>")
    called = encode(tokenizer, 'r\n</think>\n\n{"name":"lookup","arguments":{"id":1}}<|im_end|>')

    assert is_completion_signal(decode_completion(tokenizer, finished))
    assert is_error_signal(decode_completion(tokenizer, failed))
    assert len(BfclMultiTurnAdapter._parse_model_calls(decode_completion(tokenizer, called))) == 1

    # The rejected alternative, spelled out so the comparison is visible.
    kept = tokenizer.decode(finished, skip_special_tokens=False)
    assert not is_completion_signal(kept)
    assert not is_error_signal(tokenizer.decode(failed, skip_special_tokens=False))
    assert not BfclMultiTurnAdapter._parse_model_calls(
        tokenizer.decode(called, skip_special_tokens=False)
    )


def test_a_surviving_turn_end_marker_is_re_emitted_by_the_next_render(
    tokenizer: Any,
) -> None:
    """What keeping the marker costs rollout: it lands inside `Message.content`,
    and the template emits it again after its own."""
    from smolqwen.data.render import render_prefix

    ids = encode(tokenizer, "r\n</think>\n\ncall<|im_end|>")
    kept = tokenizer.decode(ids, skip_special_tokens=False)
    doubled = render_prefix(
        tokenizer,
        [
            Message("system", "S"),
            Message("user", "Q"),
            assistant_message(kept),
            Message("tool", "OBS"),
        ],
        tools=[],
        add_generation_prompt=True,
    )
    assert "<|im_end|><|im_end|>" in doubled

    clean = render_prefix(
        tokenizer,
        [
            Message("system", "S"),
            Message("user", "Q"),
            assistant_message(decode_completion(tokenizer, ids)),
            Message("tool", "OBS"),
        ],
        tools=[],
        add_generation_prompt=True,
    )
    assert "<|im_end|><|im_end|>" not in clean


def mask_over(tokenizer: Any, completions: list[str], *, split_shape: bool) -> tuple[int, int, int]:
    """Drive an episode through the mask builder under one message shape.

    Returns `(supervised, total, forks)`. Every turn re-renders the whole
    conversation, which is the operation the mask builder classifies -- so this is
    the real drift path, not a simulation of it.
    """
    from smolqwen.data.render import render_prefix
    from smolqwen.rollout.mask import EpisodeMaskBuilder

    messages = [Message("system", "S"), Message("user", "Q")]

    def prefix() -> list[int]:
        text = render_prefix(tokenizer, messages, tools=[], add_generation_prompt=True)
        return encode(tokenizer, text)

    builder = EpisodeMaskBuilder(prefix())
    for index, raw in enumerate(completions):
        sampled = encode(tokenizer, raw)
        builder.append_response(sampled, [-0.5] * len(sampled))
        text = decode_completion(tokenizer, sampled)
        messages.append(
            assistant_message(text) if split_shape else Message("assistant", content=text)
        )
        messages.append(Message("tool", f"OBS{index}"))
        builder.open_turn(prefix())
    mask = builder.env_mask
    return sum(mask), len(mask), builder.tally.fork


NORMAL = ["r1\n</think>\n\nc1<|im_end|>", "r2\n</think>\n\nc2<|im_end|>"]
TRUNCATED = ["reasoning that never closes", "r2\n</think>\n\nc2<|im_end|>"]


def test_a_well_formed_turn_renders_identically_under_either_shape(
    tokenizer: Any,
) -> None:
    """The negative control. Without it the test below could be read as showing the
    shapes always differ, which would make the choice arbitrary."""
    assert mask_over(tokenizer, NORMAL, split_shape=True) == mask_over(
        tokenizer, NORMAL, split_shape=False
    )


def test_a_truncated_turn_stored_as_plain_content_loses_supervised_tokens(
    tokenizer: Any,
) -> None:
    """The reason the split shape is the one to keep.

    Generation cut off before `</think>` closes. Stored as raw `content`, the
    template's assistant branch synthesizes an empty reasoning block and re-renders
    the model's own tokens elsewhere; the builder classifies that as a fork and the
    drifted tail becomes mask 0. Nothing raises, and the loss compounds per turn.
    """
    split_supervised, split_total, split_forks = mask_over(tokenizer, TRUNCATED, split_shape=True)
    raw_supervised, raw_total, raw_forks = mask_over(tokenizer, TRUNCATED, split_shape=False)

    assert split_total == raw_total
    assert split_forks == 0
    assert raw_forks > 0, "the raw shape no longer drifts; re-derive which shape to keep"
    assert raw_supervised < split_supervised, (
        f"raw shape supervised {raw_supervised} of {raw_total}, split shape "
        f"{split_supervised}; the drifted tail is what gets demoted to mask 0"
    )


def test_an_unclosed_block_keeps_its_text_as_reasoning() -> None:
    """Why the split survives truncation: the whole continuation stays reasoning,
    so the template re-renders it in the position it was generated in."""
    reasoning, content = split_generation_continuation("cut off mid-thought")
    assert (reasoning, content) == ("cut off mid-thought", "")

    reasoning, content = split_generation_continuation("thought\n</think>\n\nanswer")
    assert (reasoning, content) == ("thought", "answer")

    # A backend that returns the complete block is tolerated for the scripted
    # policy and for alternate engines.
    reasoning, content = split_generation_continuation("<think>thought</think>\n\nanswer")
    assert (reasoning, content) == ("thought", "answer")
