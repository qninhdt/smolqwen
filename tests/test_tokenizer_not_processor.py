"""The resolved `processing_class` must be a tokenizer, never a processor.

`Qwen/Qwen3.5-2B` ships a `vision_config` and image/video token ids, so
`AutoProcessor` is the natural choice for a `ForConditionalGeneration`
architecture -- and the wrong one. A `ProcessorMixin` sets `trainer._is_vlm = True`,
which changes prompt rendering, the `max_model_len` fallback, and forward kwargs.
"""

from __future__ import annotations

from typing import Any

import pytest

from smolqwen.tokenizer import (
    ProcessingClassError,
    assert_text_only_processing_class,
    text_config,
)


def test_a_real_tokenizer_passes() -> None:
    from transformers.tokenization_utils import PreTrainedTokenizer

    class FakeTokenizer(PreTrainedTokenizer):  # type: ignore[misc]
        def __init__(self) -> None:
            self._vocab = {"a": 0}
            super().__init__()

        @property
        def vocab_size(self) -> int:
            return len(self._vocab)

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
            return list(text)

        def _convert_token_to_id(self, token: str) -> int:
            return self._vocab.get(token, 0)

        def _convert_id_to_token(self, index: int) -> str:
            return "a"

    tokenizer = FakeTokenizer()
    assert assert_text_only_processing_class(tokenizer) is tokenizer


def test_a_processor_is_refused() -> None:
    from transformers.processing_utils import ProcessorMixin

    class FakeProcessor(ProcessorMixin):
        attributes: list[str] = []

        def __init__(self) -> None:
            pass

    with pytest.raises(ProcessingClassError, match="_is_vlm"):
        assert_text_only_processing_class(FakeProcessor())


def test_an_unrelated_object_is_refused() -> None:
    with pytest.raises(ProcessingClassError, match="PreTrainedTokenizerBase"):
        assert_text_only_processing_class(object())


def test_text_config_reaches_the_nested_text_fields() -> None:
    class Nested:
        max_position_embeddings = 262144

    class Outer:
        text_config = Nested()
        vision_config = object()

    # Reading max_position_embeddings off the top level of a Qwen3.5 config yields
    # nothing; reading it off text_config yields 262,144. Both are traps.
    assert text_config(Outer()).max_position_embeddings == 262144


def test_text_config_passes_through_a_flat_config() -> None:
    class Flat:
        max_position_embeddings = 4096

    flat = Flat()
    assert text_config(flat) is flat
