"""Secondary metrics that make an agentic score interpretable.

The headline score answers "how often did it work". These answer "and when it did
not, which part failed" -- which for an all-or-nothing benchmark is the whole
question, since a bare `0.0` is attributable to nothing.

Per-task diagnostics are *sparse* on purpose. A benchmark reports only the
conditions it can resolve for a given task, so an averaged diagnostic carries its
own denominator: `state_match_rate` over the tasks where state comparison was
defined, not over every task. The denominator travels with the value as
`<metric>_denominator`, because a rate over an unstated population invites exactly
the misreading the diagnostic exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskMetrics:
    category: str
    score: float
    invalid_calls: int
    steps: int
    generated_tokens: int
    truncated: bool
    exact_success: bool | None = None
    # Sparse per-task conditions. An absent key means "not applicable to this
    # task", never "zero" -- see `AdapterResult.diagnostics`.
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    # How the episode ended. Defaulted because this is a frozen positional dataclass
    # with several construction sites, and because the serial path had no concept of
    # a terminal reason until the turn engine introduced one.
    terminal_reason: str | None = None


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
        result[category].update(_terminal_rates(values))
        result[category].update(_aggregate_diagnostics(values))
    return result


def _terminal_rates(values: list[TaskMetrics]) -> dict[str, float]:
    """How episodes ended, as a rate per reason.

    Measured on a T4: with the context window set too small for EnvScaler's tool
    schemas, every episode terminated at admission with zero generations -- and the
    verifier still scored each one, because it grades the environment's final state
    and an untouched initial state is a valid state. The report showed `score: 0.25`
    beside `average_generated_tokens: 0.0`, which is the tell, but nothing named the
    cause and nothing refused.

    So the reason is a first-class number: `terminal_step_cap_rate: 1.0` says the run
    never generated, where a zero average only implies it. `final_answer` is the only
    reason that means the episode ran to its own conclusion; anything else being
    dominant is a finding about the configuration, not about the model.
    """
    reported = [value.terminal_reason for value in values if value.terminal_reason]
    if not reported:
        return {}
    rates: dict[str, float] = {}
    for reason in sorted(set(reported)):
        rates[f"terminal_{reason}_rate"] = reported.count(reason) / len(reported)
    rates["terminal_reason_denominator"] = float(len(reported))
    return rates


def _aggregate_diagnostics(values: list[TaskMetrics]) -> dict[str, float]:
    """Mean each diagnostic over the tasks that reported it, denominator included.

    `state_match_rate` averaged over all tasks would silently count every
    never-completed task as a state mismatch, restoring the conflation
    `completion_rate` removes. So the population is whichever tasks carried the key,
    and its size is reported alongside.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for value in values:
        for name, number in value.diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(number)
            counts[name] = counts.get(name, 0) + 1
    aggregated: dict[str, float] = {}
    for name, total in sorted(totals.items()):
        aggregated[name] = total / counts[name]
        aggregated[f"{name}_denominator"] = float(counts[name])
    return aggregated
