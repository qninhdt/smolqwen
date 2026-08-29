"""Replay every pinned BFCL ground-truth trajectory through our safe adapter."""

from __future__ import annotations

import inspect
import json
from typing import Any, cast

from smolqwen.config import resolve
from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters import create_adapter
from smolqwen.eval.adapters.bfcl import BfclMultiTurnAdapter


def _named_arguments(
    adapter: BfclMultiTurnAdapter, task: Any, raw_call: str
) -> tuple[str, dict[str, Any]]:
    name, args, kwargs = adapter._parse_ground_truth_call(raw_call)
    if not args:
        return name, kwargs
    state = adapter._state(task)
    method = next(
        getattr(instance, name)
        for instance in state.instances.values()
        if callable(getattr(instance, name, None))
    )
    parameter_names = list(inspect.signature(method).parameters)[: len(args)]
    return name, {**dict(zip(parameter_names, args, strict=True)), **kwargs}


def test_every_pinned_bfcl_ground_truth_trajectory_scores_successfully() -> None:
    config = resolve("eval")
    assert isinstance(config, EvalConfig)
    adapter = cast(BfclMultiTurnAdapter, create_adapter("bfcl_multi_turn", config))
    tasks = adapter.load_tasks()

    for task in tasks:
        state = adapter._state(task)
        for turn in state.ground_truth:
            for raw_call in turn:
                name, arguments = _named_arguments(adapter, task, raw_call)
                adapter.step(task, json.dumps({"name": name, "arguments": arguments}))
            adapter.step(task, "TASK_FINISHED")
        assert adapter.score(task).exact_success, task.task_id

    category_counts: dict[str, int] = {}
    for task in tasks:
        category_counts[task.category] = category_counts.get(task.category, 0) + 1
    assert category_counts == {
        "multi_turn_base": 200,
        "multi_turn_long_context": 200,
        "multi_turn_miss_func": 200,
        "multi_turn_miss_param": 200,
    }
    manifest = adapter.manifest_invariants(tasks)
    assert manifest["task_count"] == 800
    assert manifest["checkout_revision"] == manifest["benchmark_commit"]
