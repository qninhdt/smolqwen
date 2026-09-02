"""One record per task, carrying what a re-grade needs.

Evaluation built a full history per task and discarded it, so an all-or-nothing
`0.0` was attributable to nothing and could not be re-graded. These tests pin the
record's shape against the two failure modes that make it useless: a missing
`failure_reason`, and a record set that does not cover every task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.policies import GenerationResult
from smolqwen.eval.runner import evaluate_adapter
from smolqwen.eval.trajectories import (
    TrajectoryRecord,
    read_trajectories,
    write_trajectories,
)


class Policy:
    revision = "a" * 40

    def __init__(self, completion: str = "TASK_FINISHED") -> None:
        self._completion = completion

    def generate(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> GenerationResult:
        return GenerationResult(self._completion, 7, "stop")


class Adapter:
    """Two tasks: one scores, one fails a named condition."""

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask("passes", "fixture", "do it", ()),
            EvalTask("fails", "fixture", "do it", ()),
        ]

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if history:
            return [dict(message) for message in history]
        return [{"role": "user", "content": task.prompt}]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        return StepResult("observed", complete=True, env_steps=1, tool_observations=("observed",))

    def score(self, task: EvalTask) -> AdapterResult:
        if task.task_id == "passes":
            return AdapterResult(
                1.0, True, diagnostics={"completion_rate": 1.0, "state_match_rate": 1.0}
            )
        return AdapterResult(
            0.0,
            False,
            completed=True,
            diagnostics={"completion_rate": 1.0, "state_match_rate": 0.0},
            failure_reason="state_mismatch_at_turn_2",
            failed_check_names=("check_a",),
        )

    def invalid_call_count(self, task: EvalTask) -> int:
        return 0

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        return {"fixture": "1"}

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        return aggregate(tasks)


def test_every_task_gets_a_record_carrying_its_failing_condition() -> None:
    config = EvalConfig(adapters=("fixture",))
    records: list[TrajectoryRecord] = []

    evaluate_adapter(config, Policy(), Adapter(), records=records)

    assert [record.task_id for record in records] == ["passes", "fails"]
    passing, failing = records
    assert passing.failure_reason is None
    assert failing.failure_reason == "state_mismatch_at_turn_2"
    assert failing.failed_check_names == ["check_a"]
    # The history is kept, which is what makes a re-grade possible at all.
    assert any(message["role"] == "assistant" for message in failing.messages)
    assert failing.observations == ["observed"]
    assert failing.generation_turns == 1
    assert failing.env_steps == 1
    assert failing.generated_tokens == 7
    assert failing.terminal_reason == "final_answer"


def test_a_turn_capped_task_records_that_reason_rather_than_a_bare_zero() -> None:
    """A task that never terminated is a different failure from one that finished
    wrong, and the record has to say which."""

    class NeverCompletes(Adapter):
        def step(self, task: EvalTask, completion: str) -> StepResult:
            return StepResult("observed", complete=False, env_steps=1)

    config = EvalConfig(adapters=("fixture",), max_steps_per_task=3)
    records: list[TrajectoryRecord] = []

    evaluate_adapter(config, Policy(), NeverCompletes(), records=records)

    assert all(record.terminal_reason == "turn_cap" for record in records)
    assert all(record.generation_turns == 3 for record in records)


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    config = EvalConfig(adapters=("fixture",))
    records: list[TrajectoryRecord] = []
    evaluate_adapter(config, Policy(), Adapter(), records=records)

    path = write_trajectories(tmp_path, tag="sft", adapter="fixture", records=records)
    assert path.name == "sft-fixture.jsonl"

    rows = read_trajectories(path)
    assert [row["task_id"] for row in rows] == ["passes", "fails"]
    assert rows[1]["failure_reason"] == "state_mismatch_at_turn_2"
    assert rows[1]["diagnostics"]["state_match_rate"] == 0.0


def test_writing_the_same_tag_twice_replaces_rather_than_interleaves(
    tmp_path: Path,
) -> None:
    """A run is a whole set, certified by one manifest. Appending would leave a
    file half-certified by each of two runs."""
    first = [TrajectoryRecord(task_id="a", category="fixture", score=1.0)]
    second = [TrajectoryRecord(task_id="b", category="fixture", score=0.0)]

    write_trajectories(tmp_path, tag="sft", adapter="fixture", records=first)
    path = write_trajectories(tmp_path, tag="sft", adapter="fixture", records=second)

    assert [row["task_id"] for row in read_trajectories(path)] == ["b"]
