from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from smolqwen.env.scenarios import Scenario
from smolqwen.eval.adapters.base import EvalTask
from smolqwen.eval.adapters.envscaler_heldout import (
    EnvScalerHeldoutAdapter,
    _Episode,
    select_heldout_scenarios,
)
from smolqwen.prompts import NON_CONVERSATIONAL


def _scenario(env_id: str, task_id: str) -> Scenario:
    return Scenario(task_id, env_id, "Demo", "task", {}, ({"description": "check"},))


def test_heldout_selection_is_stable_and_bounded_per_environment() -> None:
    selected = select_heldout_scenarios(
        (_scenario("b", "b-2"), _scenario("a", "a-2"), _scenario("a", "a-1")),
        env_count=1,
        per_env=1,
    )
    assert [(scenario.env_id, scenario.task_id) for scenario in selected] == [("a", "a-1")]


def test_heldout_prompt_preserves_the_training_system_role(monkeypatch: Any) -> None:
    adapter = object.__new__(EnvScalerHeldoutAdapter)
    monkeypatch.setattr(adapter, "_ensure_created", lambda task: None)
    task = EvalTask(
        "case",
        "envscaler_heldout",
        "do the task",
        (),
        {"system_prompt": NON_CONVERSATIONAL, "env_id": "env_1_rl"},
    )
    assert adapter.build_prompt(task, []) == [
        {"role": "system", "content": NON_CONVERSATIONAL},
        {"role": "user", "content": "do the task"},
    ]


def test_http_normalized_json_calls_execute_instead_of_ending_the_episode(
    monkeypatch: Any,
) -> None:
    adapter = object.__new__(EnvScalerHeldoutAdapter)
    episode = _Episode(_scenario("env_1_rl", "case"), "case")
    adapter._episodes = {"case": episode}
    seen: list[tuple[str, str, dict[str, Any]]] = []

    class Pool:
        def step(self, episode_id: str, name: str, arguments: dict[str, Any]) -> Any:
            seen.append((episode_id, name, arguments))
            return SimpleNamespace(ok=True, value={"found": 1}, reason=None, detail=None)

    monkeypatch.setattr(adapter, "_pool_or_raise", lambda: Pool())
    task = EvalTask(
        "case",
        "envscaler_heldout",
        "do the task",
        ({"type": "function", "function": {"name": "lookup"}},),
    )
    result = adapter.step(task, json.dumps({"name": "lookup", "arguments": {"id": 1}}))
    assert seen == [("case", "lookup", {"id": 1})]
    assert result.env_steps == 1
    assert result.tool_observations == ("{'found': 1}",)
    assert not result.complete
