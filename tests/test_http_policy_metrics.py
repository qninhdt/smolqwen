from __future__ import annotations

import json
from typing import Any

import pytest

from smolqwen.eval.adapters.bfcl import BfclMultiTurnAdapter
from smolqwen.eval.policies import HttpPolicy

REVISION = "d" * 40


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "answer"}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 17},
            }
        ).encode()


def test_http_policy_captures_usage_and_finish_reason() -> None:
    seen: list[Any] = []

    def opener(request: object, *, timeout: float) -> _Response:
        seen.append(request)
        assert timeout == 60.0
        return _Response()

    policy = HttpPolicy(
        "http://localhost:8000",
        revision=REVISION,
        opener=opener,
    )
    result = policy.generate([{"role": "user", "content": "hi"}], [])
    assert result.completion == "answer"
    assert result.generated_tokens == 17
    assert result.truncated
    assert seen[0].full_url.endswith("/v1/chat/completions")


def test_http_policy_preserves_tool_calls_and_decoding_request() -> None:
    class ToolResponse(_Response):
        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "<think>lookup</think>",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"id": 1}',
                                        }
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"completion_tokens": 9},
                }
            ).encode()

    seen: list[Any] = []
    policy = HttpPolicy(
        "http://localhost:8000",
        revision=REVISION,
        model="smolqwen",
        max_new_tokens=123,
        temperature=0.25,
        top_p=0.9,
        top_k=20,
        seed=7,
        opener=lambda request, **_: _record_request(seen, request, ToolResponse()),
    )
    result = policy.generate([{"role": "user", "content": "hi"}], [])
    assert json.loads(result.completion.split("\n", 1)[1]) == {
        "name": "lookup",
        "arguments": {"id": 1},
    }
    assert BfclMultiTurnAdapter._parse_model_calls(result.completion) == [("lookup", (), {"id": 1})]
    body = json.loads(seen[0].data)
    assert body["model"] == "smolqwen"
    assert body["max_tokens"] == 123
    assert body["temperature"] == 0.25
    assert body["top_p"] == 0.9
    assert body["top_k"] == 20
    assert body["seed"] == 7


def test_http_policy_accepts_a_versioned_api_base_without_duplicate_path() -> None:
    policy = HttpPolicy("http://localhost:8000/v1", revision=REVISION)
    assert policy.completion_url == "http://localhost:8000/v1/chat/completions"


def test_http_policy_rejects_a_moving_revision_name() -> None:
    with pytest.raises(ValueError, match="revision sha"):
        HttpPolicy("http://localhost:8000", revision="main")


def test_http_policy_rebuilds_openai_tool_call_history() -> None:
    messages = HttpPolicy._openai_messages(
        [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": '<think>lookup</think>\n{"name":"lookup","arguments":{"id":1}}',
            },
            {"role": "tool", "content": "found"},
        ]
    )
    call = messages[1]["tool_calls"][0]
    assert call["function"] == {"name": "lookup", "arguments": '{"id": 1}'}
    assert messages[2]["tool_call_id"] == call["id"]


def test_http_policy_pairs_each_parallel_call_with_its_observation() -> None:
    messages = HttpPolicy._openai_messages(
        [
            {"role": "user", "content": "look up both"},
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {"name": "lookup", "arguments": {"id": 1}},
                        {"name": "lookup", "arguments": {"id": 2}},
                    ]
                ),
            },
            {"role": "tool", "content": "first"},
            {"role": "tool", "content": "second"},
        ]
    )
    calls = messages[1]["tool_calls"]
    assert len(calls) == 2
    assert messages[2]["tool_call_id"] == calls[0]["id"]
    assert messages[3]["tool_call_id"] == calls[1]["id"]


def _record_request(seen: list[Any], request: Any, response: _Response) -> _Response:
    seen.append(request)
    return response
