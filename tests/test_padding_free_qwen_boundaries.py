"""Boundary metadata survives the real Qwen3.5 model-forward routing."""

from __future__ import annotations

from types import MethodType
from typing import Any

from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS
from smolqwen.training.collate import padding_free_collator
from tests.helpers import tiny_qwen35_model


def _record(uid: str, offset: int, length: int) -> dict[str, Any]:
    input_ids = [(offset + index) % 250 + 1 for index in range(length)]
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": uid,
        "input_ids": input_ids,
        "labels": [-100] + input_ids[1:],
        "seq_length": length,
        "supervised_tokens": length - 1,
    }


def _capture_forward(module: Any, captures: list[dict[str, Any]]) -> None:
    original = module.forward

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        captures.append(dict(kwargs))
        return original(*args, **kwargs)

    module.forward = MethodType(wrapped, module)


def test_qwen_routes_document_boundaries_to_both_mixer_types() -> None:
    batch = padding_free_collator(32)([_record("a", 1, 5), _record("b", 20, 7)])
    model = tiny_qwen35_model()
    linear_captures: list[dict[str, Any]] = []
    attention_captures: list[dict[str, Any]] = []
    _capture_forward(model.model.layers[0].linear_attn, linear_captures)
    _capture_forward(model.model.layers[1].self_attn, attention_captures)

    model(**batch)

    assert linear_captures
    linear = linear_captures[0]
    assert linear["cu_seq_lens_q"].tolist() == [0, 5, 12]
    assert linear["seq_idx"].tolist() == [[0] * 5 + [1] * 7]
    assert linear["max_length_q"] == 7

    assert attention_captures
    attention = attention_captures[0]
    assert attention["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]
    assert attention["cu_seq_lens_q"].tolist() == [0, 5, 12]
    assert attention["cu_seq_lens_k"].tolist() == [0, 5, 12]
