"""The local Qwen3.5 checkpoint fixture has the shape vLLM resolves."""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import write_tiny_vllm_checkpoint


def test_tiny_vllm_checkpoint_uses_the_released_wrapper_shape(tmp_path: Path) -> None:
    checkpoint = write_tiny_vllm_checkpoint(tmp_path / "base")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    assert config["model_type"] == "qwen3_5"
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config["text_config"]["model_type"] == "qwen3_5_text"
    assert config["vision_config"]["model_type"] == "qwen3_5"
    assert (checkpoint / "preprocessor_config.json").is_file()
    assert (checkpoint / "video_preprocessor_config.json").is_file()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    assert tokenizer.convert_tokens_to_ids("<|image_pad|>") == config["image_token_id"]
    assert tokenizer.convert_tokens_to_ids("<|video_pad|>") == config["video_token_id"]
    assert tokenizer.convert_tokens_to_ids("<|vision_start|>") == config["vision_start_token_id"]
    assert tokenizer.convert_tokens_to_ids("<|vision_end|>") == config["vision_end_token_id"]
