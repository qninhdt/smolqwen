"""Secondary metrics that make an agentic score interpretable."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskMetrics:
    category: str
    score: float
    invalid_calls: int
    steps: int
    generated_tokens: int
    truncated: bool
    exact_success: bool | None = None


def aggregate(tasks: Iterable[TaskMetrics]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TaskMetrics]] = {}
    for task in tasks:
        grouped.setdefault(task.category, []).append(task)
    result: dict[str, dict[str, float]] = {}
    for category, values in grouped.items():
        count = len(values)
        result[category] = {
            "score": sum(v.score for v in values) / count,
            "invalid_call_rate": sum(v.invalid_calls for v in values)
            / max(1, sum(v.steps for v in values)),
            "average_steps": sum(v.steps for v in values) / count,
            "average_generated_tokens": sum(v.generated_tokens for v in values) / count,
            "truncation_rate": sum(v.truncated for v in values) / count,
        }
        if any(value.exact_success is not None for value in values):
            result[category]["exact_success_rate"] = sum(
                bool(value.exact_success) for value in values if value.exact_success is not None
            ) / sum(value.exact_success is not None for value in values)
    return result
