from __future__ import annotations

from typing import Any

import pytest

from smolqwen.config_models import EvalConfig, ProfileConfig
from smolqwen.eval import bfcl_runner
from smolqwen.eval.bfcl_runner import (
    BfclCompletion,
    BfclTask,
    evaluate_bfcl,
    load_bfcl_tasks,
)
from smolqwen.eval.trajectories import TrajectoryRecord
from tests.helpers import OfflineTokenizer


def _doc(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


def _task(task_id: str = "case") -> BfclTask:
    return BfclTask(
        task_id,
        "multi_turn_base",
        {
            "id": task_id,
            "question": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            "function": [_doc("first_call"), _doc("second_call")],
            "initial_config": {},
            "involved_classes": [],
        },
        (("first_call()",), ("second_call()",)),
    )


@pytest.mark.dataset
def test_loads_multi_turn_base_through_bfcl_itself() -> None:
    tasks, revision = load_bfcl_tasks()

    assert len(tasks) == 200
    assert len(revision) == 40
    assert tasks[0].entry["function"]
    assert all(message["role"] != "system" for message in tasks[0].entry["question"][0])


def test_understands_qwen35_and_qwen3_tool_call_formats() -> None:
    qwen35 = """<tool_call>
<function=lookup>
<parameter=id>
7
</parameter>
</function>
</tool_call>"""
    qwen3 = '<tool_call>\n{"name": "lookup", "arguments": {"id": 7}}\n</tool_call>'

    assert [(call.name, call.arguments) for call in bfcl_runner._parse_calls(qwen35)] == [
        ("lookup", {"id": 7})
    ]
    assert [(call.name, call.arguments) for call in bfcl_runner._parse_calls(qwen3)] == [
        ("lookup", {"id": 7})
    ]


def test_qwen35_arguments_follow_bfcl_schema_types() -> None:
    text = """<tool_call>
<function=route>
<parameter=zipcode>
94102
</parameter>
<parameter=avoid_tolls>
False
</parameter>
</function>
</tool_call>"""
    tools = [
        {
            "name": "route",
            "parameters": {
                "type": "dict",
                "properties": {
                    "zipcode": {"type": "string"},
                    "avoid_tolls": {"type": "boolean"},
                },
            },
        }
    ]

    assert bfcl_runner._parse_calls(text, tools=tools)[0].arguments == {
        "zipcode": "94102",
        "avoid_tolls": False,
    }


def test_uses_structured_history_json_results_and_a_per_turn_cap(
    monkeypatch: Any,
) -> None:
    tokenizer = OfflineTokenizer(token_size=1)
    outputs = iter(
        [
            "<tool_call>\n<function=first_call>\n</function>\n</tool_call>",
            "done",
            "<tool_call>\n<function=second_call>\n</function>\n</tool_call>",
            "done",
        ]
    )
    checked: list[list[list[list[str]]]] = []

    def execute(calls: list[str], *_args: Any) -> list[str]:
        return ['{"success": true}'] * len(calls)

    def check(responses: Any, *_args: Any) -> dict[str, bool]:
        checked.append(responses)
        return {"valid": True}

    monkeypatch.setattr(bfcl_runner, "_execute", execute)
    monkeypatch.setattr(bfcl_runner, "_check", check)

    def generate(requests: Any) -> list[BfclCompletion]:
        assert len(requests) == 1
        text = next(outputs)
        return [BfclCompletion(requests[0].task_id, text, 1, "stop")]

    records: list[TrajectoryRecord] = []
    config = EvalConfig(
        max_steps_per_task=1,
        profile=ProfileConfig(generation_concurrency=1, max_seq_length=32768),
    )
    run = evaluate_bfcl(
        config,
        tokenizer=tokenizer,
        generate=generate,
        tasks=[_task()],
        benchmark_revision="a" * 40,
        record_sink=records.append,
    )

    assert run.metrics["multi_turn_base"]["score"] == 1.0
    assert checked == [[[["first_call()"]], [["second_call()"]]]]
    assert [message["role"] for message in records[0].messages].count("system") == 0
    assistant_calls = [message for message in records[0].messages if message["role"] == "assistant"]
    assert assistant_calls[0]["content"] == ""
    assert assistant_calls[0]["tool_calls"][0]["function"]["name"] == "first_call"
    assert records[0].observations == ['{"success": true}', '{"success": true}']


def test_batches_ready_bfcl_tasks(monkeypatch: Any) -> None:
    monkeypatch.setattr(bfcl_runner, "_check", lambda *_args: {"valid": True})
    widths: list[int] = []

    def generate(requests: Any) -> list[BfclCompletion]:
        widths.append(len(requests))
        return [BfclCompletion(request.task_id, "done", 1, "stop") for request in requests]

    one_turn = []
    for index in range(4):
        task = _task(f"case-{index}")
        one_turn.append(
            BfclTask(
                task.task_id,
                task.category,
                {**task.entry, "question": [task.entry["question"][0]]},
                (task.ground_truth[0],),
            )
        )
    config = EvalConfig(profile=ProfileConfig(generation_concurrency=2))
    evaluate_bfcl(
        config,
        tokenizer=OfflineTokenizer(token_size=1),
        generate=generate,
        tasks=one_turn,
        benchmark_revision="a" * 40,
    )

    assert widths == [2, 2]
