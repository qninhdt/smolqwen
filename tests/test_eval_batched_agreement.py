"""The legacy batched adapter path used by in-training benchmark callbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult
from smolqwen.eval.batched import evaluate_batched
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.trajectories import TrajectoryRecord
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids
from tests.helpers import OfflineTokenizer

# Two tool calls then a completion marker. The marker is what advances a static
# benchmark to its next question, which is the path a `user`-role observation takes.
TURNS = [
    '\n</think>\n\n{"name": "lookup", "arguments": {"id": 1}}',
    '\n</think>\n\n{"name": "lookup", "arguments": {"id": 2}}',
    "\n</think>\n\nTASK_FINISHED",
]


class ScriptedAdapter:
    """A two-question benchmark: tool observations, then a `user`-role next turn."""

    def __init__(self, task_count: int = 4) -> None:
        self.task_count = task_count
        self.turn: dict[str, int] = {}
        self.calls: dict[str, list[str]] = {}
        self.invalid: dict[str, int] = {}

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask(
                f"task-{index}",
                "fixture",
                "do the thing",
                ({"type": "function", "function": {"name": "lookup"}},),
            )
            for index in range(self.task_count)
        ]

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if history:
            return [dict(message) for message in history]
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": task.prompt},
        ]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        from smolqwen.eval.tool_calls import is_completion_signal, parse_normalized_json_calls

        seen = self.calls.setdefault(task.task_id, [])
        calls = parse_normalized_json_calls(completion)
        if calls:
            seen.extend(name for name, _ in calls)
            return StepResult(
                "observed",
                env_steps=len(calls),
                tool_observations=tuple(f"result:{name}" for name, _ in calls),
            )
        if is_completion_signal(completion):
            index = self.turn.get(task.task_id, 0)
            if index == 0:
                self.turn[task.task_id] = 1
                # The next static question: a real user turn, not a tool result.
                return StepResult("and now the second thing", observation_role="user")
            return StepResult("finished", complete=True)
        self.invalid[task.task_id] = self.invalid.get(task.task_id, 0) + 1
        return StepResult("Error: no call and no marker.")

    def score(self, task: EvalTask) -> AdapterResult:
        calls = len(self.calls.get(task.task_id, ()))
        return AdapterResult(
            1.0 if calls >= 2 else 0.0,
            calls >= 2,
            diagnostics={"calls_made": float(calls)},
            failure_reason=None if calls >= 2 else "too_few_calls",
        )

    def invalid_call_count(self, task: EvalTask) -> int:
        return self.invalid.get(task.task_id, 0)

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        return {"fixture": "1"}

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        return aggregate(tasks)


def scripted_backend(tokenizer: OfflineTokenizer) -> ScriptedPolicyBackend:
    """The engine's generation seam, replaying the same script by turn index."""

    def policy(episode_id: str, turn_index: int, messages: Sequence[Any]) -> str:
        return TURNS[min(turn_index, len(TURNS) - 1)]

    return ScriptedPolicyBackend(policy, lambda text: encode_ids(tokenizer, text))


def config(**overrides: Any) -> EvalConfig:
    payload: dict[str, Any] = {"adapters": ("fixture",), "max_steps_per_task": 8}
    payload.update(overrides)
    return EvalConfig(**payload)


def test_the_batched_path_issues_one_generation_call_per_cycle_not_per_task() -> None:
    """The measurable difference: the serial path called generate once per turn per
    task, so a 2B model decoded one sequence at a time."""
    tokenizer = OfflineTokenizer(token_size=1)
    backend = scripted_backend(tokenizer)
    widths: list[int] = []
    inner = backend.generate

    def counting_generate(requests: Sequence[Any]) -> list[Any]:
        widths.append(len(requests))
        return inner(requests)

    backend.generate = counting_generate  # type: ignore[method-assign]

    evaluate_batched(
        config(),
        ScriptedAdapter(task_count=4),
        backend=backend,
        tokenizer=tokenizer,
    )

    assert widths, "no generation happened"
    assert max(widths) > 1, f"every call was batch size 1: {widths}"


def test_a_user_role_next_question_is_preserved_through_the_batched_path() -> None:
    """`base.py:34` states the contract and BFCL relies on it: a static benchmark's
    next question is a user turn, and the template renders that differently from a
    tool result -- which decides whether earlier reasoning survives the render."""
    tokenizer = OfflineTokenizer(token_size=1)
    records: list[TrajectoryRecord] = []

    evaluate_batched(
        config(),
        ScriptedAdapter(task_count=1),
        backend=scripted_backend(tokenizer),
        tokenizer=tokenizer,
        records=records,
    )

    roles = [message["role"] for message in records[0].messages]
    assert "and now the second thing" in records[0].observations
    # The second question is a `user` message, and the tool results are not.
    assert roles.count("user") >= 2
    assert "tool" in roles
