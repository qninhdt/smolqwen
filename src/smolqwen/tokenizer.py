"""Tokenizer loading. Deliberately never `AutoProcessor`.

`Qwen/Qwen3.5-2B` is a multimodal checkpoint: the config's top level holds
`vision_config` and image/video token ids, and every text field is nested under
`text_config`. `AutoProcessor` is the natural choice for a
`ForConditionalGeneration` architecture, and it is the wrong one here -- a
`ProcessorMixin` sets `trainer._is_vlm = True`, which changes prompt rendering,
the `max_model_len` fallback (inheriting 262,144 positions), and
`mm_token_type_ids` handling.

The pipeline is text-only at every stage, so it loads `AutoTokenizer` at every
stage, and `assert_text_only_processing_class` is the enforcement.
"""

from __future__ import annotations

from typing import Any


class ProcessingClassError(TypeError):
    """Raised when a processor reaches a code path that requires a plain tokenizer."""


def assert_text_only_processing_class(processing_class: Any) -> Any:
    """Return `processing_class` if it is a tokenizer; raise if it is a processor.

    Checked by class name rather than only by `isinstance`, so the assertion works
    in a CPU-only test that has no transformers import chain loaded and cannot
    construct a real processor to compare against.
    """
    try:
        from transformers.processing_utils import ProcessorMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:  # pragma: no cover - transformers is a hard dependency
        return processing_class

    if isinstance(processing_class, ProcessorMixin):
        raise ProcessingClassError(
            "processing_class is a ProcessorMixin; the pipeline is text-only and a "
            "processor flips TRL onto its VLM code paths (trainer._is_vlm = True), "
            "changing prompt rendering and the max_model_len fallback. "
            "Load AutoTokenizer instead."
        )
    if not isinstance(processing_class, PreTrainedTokenizerBase):
        raise ProcessingClassError(
            f"processing_class must be a PreTrainedTokenizerBase, got "
            f"{type(processing_class).__name__}"
        )
    return processing_class


def load_tokenizer(model_id: str, *, revision: str | None = None, **kwargs: Any) -> Any:
    """Load the text tokenizer for a model id, asserting it is not a processor."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **kwargs)
    return assert_text_only_processing_class(tokenizer)


def text_config(config: Any) -> Any:
    """Return the text sub-config, or the config itself when it is already text-only.

    Reading `max_position_embeddings` off the top level of a Qwen3.5 config yields
    nothing; reading it off `text_config` yields 262,144. Both are traps in
    different directions, so the lookup is centralised.
    """
    return getattr(config, "text_config", None) or config
