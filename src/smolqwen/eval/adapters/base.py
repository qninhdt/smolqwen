"""Narrow, stateful benchmark adapter contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from smolqwen.eval.metrics import TaskMetrics


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    category: str
    prompt: str
    tools: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: str
    complete: bool = False
    # Number of real environment methods executed by this model turn. Completion
    # markers and clarification messages consume generation budget but are not
    # environment steps in the reported metric.
    env_steps: int = 0
    # One observation per structured call. This preserves OpenAI tool_call_id
    # pairing when an endpoint returns several calls in one assistant message.
    tool_observations: tuple[str, ...] | None = None
    # The role matters to Qwen3.5's chat template.  Tool output must remain a
    # `role: tool` message; benchmark user turns are ordinary user messages.
    observation_role: Literal["tool", "user"] = "tool"
    # Benchmarks may change their visible tool set between turns. A result
    # carries the next set without making the runner understand why it changed.
    tools: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class AdapterResult:
    score: float
    exact_success: bool


class BenchmarkAdapter(Protocol):
    """Complete benchmark plugin contract; the runner owns no benchmark semantics."""

    def load_tasks(self) -> list[EvalTask]: ...

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...

    def step(self, task: EvalTask, completion: str) -> StepResult: ...

    def score(self, task: EvalTask) -> AdapterResult: ...

    def invalid_call_count(self, task: EvalTask) -> int: ...

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]: ...

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]: ...
