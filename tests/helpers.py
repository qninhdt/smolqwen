"""Offline test tokenizer that drives the real Qwen3.5-2B chat template.

Loading `AutoTokenizer` needs the Hub and a token, which the data-pipeline tests
must never require (`make test` runs CPU-only, no network, no token). So these
tests render through the *vendored* template string -- the same jinja the
released model ships -- via transformers' `render_jinja_template`, which is a pure
function needing only the template string and the conversation dicts.

This is not a "fake" tokenizer that re-asserts the architecture note. It renders
the *actual* template, so if the template's reasoning-retention behavior differs
from the note's reading, these tests fail. The note's job is to predict; the
template's job is to render. The note is wrong if they disagree, and this is the
test that catches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from transformers.utils.chat_template_utils import render_jinja_template

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_TEMPLATE_PATH = _FIXTURES / "qwen35_chat_template.jinja"
_TEMPLATE: str | None = None


def load_template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _TEMPLATE


def _render(
    conversation: list[dict[str, Any]],
    *,
    tools: Any = None,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    """Render a conversation through the real template, tolerating the wrong stub.

    `render_jinja_template` is declared to return `str`, but actually returns a
    `(rendered_chats, generation_indices)` pair. We cast to the real shape and
    take the first rendered chat.
    """
    out = render_jinja_template(
        [conversation],
        chat_template=load_template(),
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    pair = cast(tuple[list[str], list[Any]], out)
    return pair[0][0]


class OfflineTokenizer:
    """Drives the real chat template, encodes strings to a reversible id list.

    Encoding is a pure function of the string (a stable hash per distinct token
    chunk), not a model vocabulary. The data pipeline asserts on token *counts*
    and loss-mask *positions*, not on vocabulary semantics, so a reversible chunk
    encoding is sufficient and keeps the tests offline.
    """

    def __init__(self, *, token_size: int = 64, shape: str = "tool_role") -> None:
        self._token_size = token_size
        self.shape = shape

    # `render.render_segment` calls `apply_chat_template` and `__call__` only; we
    # implement exactly the protocol surface it touches.
    def apply_chat_template(
        self,
        conversation: Any,
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **_: Any,
    ) -> str:
        if tokenize:
            raise NotImplementedError("this tokenizer only tokenizes via __call__")
        return _render(
            conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": self._encode(text)}

    def _encode(self, text: str) -> list[int]:
        # Split into fixed-size chunks; map each distinct chunk to a stable id so
        # identical strings tokenize identically (the invariant render.py relies on
        # -- identical text produces identical ids so the prompt is a literal
        # prefix of the longer render).
        tokens = [text[i : i + self._token_size] for i in range(0, len(text), self._token_size)]
        return [_stable_id(token) for token in tokens]


def _stable_id(token: str) -> int:
    # Deterministic, collision-free-for-small-inputs hash. Not cryptographic; the
    # only requirement is stability within a test run so the prefix-slice invariant
    # holds.
    return hash(token) & 0x7FFFFFFF


def load_trajectory_rows() -> list[dict[str, Any]]:
    """The raw release-shaped rows from `tests/fixtures/trajectories.json`.

    Three rows: one Non-Conv, one Conv, one deliberately malformed (messages is a
    JSON string that is not a list, so `parse_trajectory` raises `DataError`).
    """
    raw = json.loads(_FIXTURES.joinpath("trajectories.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("fixture trajectories.json must be a JSON array")
    return raw


def write_tiny_tokenizer(directory: Path, *, vocab_size: int = 256) -> Path:
    """Save a trivial word-level fast tokenizer into `directory`.

    Needed wherever a test writes a local checkpoint that real code then loads
    with `AutoTokenizer` (the merge path saves the tokenizer next to the merged
    weights, because a checkpoint without its chat template is not loadable by
    the eval harness or vLLM). Built from `tokenizers` directly so nothing
    downloads and no sentencepiece/tiktoken conversion is attempted.
    """
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"<{index}>": index for index in range(vocab_size)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<0>"))
    # `PreTrainedTokenizerFast` resolves to an untyped backend class, so bind it
    # through `Any` rather than sprinkling ignores at the call site.
    factory: Any = PreTrainedTokenizerFast
    tokenizer = factory(
        tokenizer_object=backend,
        unk_token="<0>",
        eos_token="<1>",
        pad_token="<2>",
    )
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(directory))
    return directory


# Layers alternate GDN mixer / full attention, which is what makes the compile
# exclusion list and the attention-implementation choice meaningful in a test.
# Spelled out rather than derived from `full_attention_interval`, which the config
# accepts only through **kwargs.
TINY_LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]


def tiny_qwen35_model(*, vocab_size: int = 256) -> Any:
    """A 4-layer random-weight Qwen3.5 text model. No download, CPU-sized."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(TINY_LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=TINY_LAYER_TYPES,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    factory: Any = Qwen3_5ForCausalLM
    return factory(config)


def write_tiny_checkpoint(directory: Path, *, vocab_size: int = 256) -> Path:
    """Save a tiny model plus its tokenizer, so real loading code can read it."""
    tiny_qwen35_model(vocab_size=vocab_size).save_pretrained(str(directory))
    write_tiny_tokenizer(directory, vocab_size=vocab_size)
    return directory
