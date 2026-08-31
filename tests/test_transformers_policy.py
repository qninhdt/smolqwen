from __future__ import annotations

from typing import Any

import pytest
import torch
import transformers
from transformers import BatchEncoding

from smolqwen.eval.policies import TransformersPolicy


class _Tokenizer:
    def __init__(self, rendered: Any) -> None:
        self.rendered = rendered

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> Any:
        return self.rendered

    def decode(self, tokens: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert tokens.tolist() == [7, 8]
        return "generated"


class _Model:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def eval(self) -> _Model:
        return self

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        completion = torch.tensor([[7, 8]], dtype=input_ids.dtype)
        return torch.cat((input_ids, completion), dim=-1)


def test_local_policy_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        TransformersPolicy("checkpoint", revision="a" * 40)


def test_local_policy_loads_the_entire_model_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    load_kwargs: dict[str, Any] = {}
    model = _Model()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _Tokenizer:
            return _Tokenizer(None)

    class _AutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: Any) -> _Model:
            load_kwargs.update(kwargs)
            return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _AutoModel)

    policy = TransformersPolicy("checkpoint", revision="a" * 40)

    assert load_kwargs["device_map"] == {"": 0}
    assert policy._model is model


def _policy(rendered: Any) -> tuple[TransformersPolicy, _Model]:
    policy: Any = TransformersPolicy.__new__(TransformersPolicy)
    policy.max_new_tokens = 2
    policy.temperature = 0.0
    policy.top_p = 1.0
    policy.top_k = -1
    policy.seed = 1234
    policy._torch = torch
    policy._tokenizer = _Tokenizer(rendered)
    model = _Model()
    policy._model = model
    return policy, model


def test_generate_unpacks_batch_encoding_into_model_inputs() -> None:
    rendered = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    policy, model = _policy(rendered)

    result = policy.generate([{"role": "user", "content": "hello"}], [])

    assert model.args == ()
    assert model.kwargs["input_ids"].tolist() == [[1, 2, 3]]
    assert model.kwargs["attention_mask"].tolist() == [[1, 1, 1]]
    assert result.completion == "generated"
    assert result.generated_tokens == 2
    assert result.finish_reason == "length"
