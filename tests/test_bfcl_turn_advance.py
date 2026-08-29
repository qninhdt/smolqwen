from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.eval.adapters.bfcl import BfclMultiTurnAdapter
from smolqwen.eval.manifest import hash_json
from smolqwen.prompts import NON_CONVERSATIONAL


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_bfcl_turn_advances_only_after_completion_signal(tmp_path: Path) -> None:
    category = "multi_turn_base"
    entry = {
        "id": "multi_turn_base_case",
        "question": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        "initial_config": {},
        "involved_classes": [],
    }
    _write_jsonl(tmp_path / f"BFCL_v4_{category}.json", [entry])
    (tmp_path / "multi_turn_func_doc").mkdir()
    _write_jsonl(
        tmp_path / "possible_answer" / f"BFCL_v4_{category}.json",
        [{"id": "multi_turn_base_case", "ground_truth": [[], []]}],
    )

    adapter = BfclMultiTurnAdapter(tmp_path, [category])
    task = adapter.load_tasks()[0]

    assert adapter.build_prompt(task, [])[0] == {
        "role": "system",
        "content": NON_CONVERSATIONAL,
    }
    assert adapter.step(task, "plain prose").observation.startswith("Error:")
    assert adapter._state(task).turn_index == 0
    assert adapter.step(task, "unfinished response").observation.startswith("Error:")
    assert adapter._state(task).turn_index == 0
    assert adapter.step(task, "TASK_FINISHED").observation == "second"
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "TASK_FINISHED"},
        {"role": "user", "content": "second"},
    ]
    assert adapter.build_prompt(task, history) == history
    assert adapter.step(task, "TASK_FINISHED").complete


def test_miss_function_tools_are_revealed_at_the_holdout_turn(
    tmp_path: Path, monkeypatch: Any
) -> None:
    category = "multi_turn_miss_func"
    entry = {
        "id": "multi_turn_miss_func_case",
        "question": [[{"role": "user", "content": "first"}], []],
        "initial_config": {},
        "involved_classes": ["MathAPI"],
        "missed_function": {"1": ["add"]},
    }
    _write_jsonl(tmp_path / f"BFCL_v4_{category}.json", [entry])
    _write_jsonl(
        tmp_path / "possible_answer" / f"BFCL_v4_{category}.json",
        [{"id": entry["id"], "ground_truth": [[], []]}],
    )
    _write_jsonl(
        tmp_path / "multi_turn_func_doc" / "math_api.json",
        [{"name": "add", "description": "add", "parameters": {"type": "object"}}],
    )

    monkeypatch.syspath_prepend(
        Path("third_party/gorilla/berkeley-function-call-leaderboard").resolve()
    )
    adapter = BfclMultiTurnAdapter(tmp_path, [category])
    task = adapter.load_tasks()[0]
    assert not any(tool["function"]["name"] == "add" for tool in task.tools)
    advanced = adapter.step(task, "TASK_FINISHED")
    assert advanced.observation == (
        "I have updated some more functions you can choose from. What about now?"
    )
    assert advanced.observation_role == "user"
    assert advanced.tools is not None
    assert any(tool["function"]["name"] == "add" for tool in advanced.tools)
    assert adapter.step(task, "Task Completed").complete

    monkeypatch.setattr(adapter, "_lineage", lambda: ("a" * 40, "data-hash"))
    manifest = adapter.manifest_invariants([task])
    assert manifest["tool_schema_hash"] == hash_json([adapter._state(task).all_tools])


@pytest.mark.parametrize("task_id", ["multi_turn_base_0", "multi_turn_base_52"])
def test_ground_truth_positional_arguments_match_named_model_calls(task_id: str) -> None:
    root = Path("third_party/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    adapter = BfclMultiTurnAdapter(root, ["multi_turn_base"])
    task = next(task for task in adapter.load_tasks() if task.task_id == task_id)
    state = adapter._state(task)
    for turn in state.ground_truth:
        for raw_call in turn:
            name, args, kwargs = adapter._parse_ground_truth_call(raw_call)
            if args:
                method = next(
                    getattr(instance, name)
                    for instance in state.instances.values()
                    if callable(getattr(instance, name, None))
                )
                parameter_names = list(inspect.signature(method).parameters)[: len(args)]
                kwargs = {
                    **dict(zip(parameter_names, args, strict=True)),
                    **kwargs,
                }
            adapter.step(task, json.dumps({"name": name, "arguments": kwargs}))
        adapter.step(task, "TASK_FINISHED")
    result = adapter.score(task)
    assert result.exact_success
