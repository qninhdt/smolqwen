"""Each scoring condition maps to a metric that can resolve it, or is absent.

BFCL returns 1.0 only when four conditions hold (`bfcl.py:192-265`): the episode
completed, it produced one snapshot per ground-truth turn, every snapshot matched,
and cumulative results covered what each turn expected. A bare `0.0` is
attributable to none of them, which is what these diagnostics fix.

The subtle part is the denominator. `state_match_rate` is undefined for a task that
never completed -- `bfcl.py` returns before the expected snapshots exist, and the
`zip(..., strict=True)` that follows would raise on the length mismatch the guard
protects. Filling `0.0` would make "never finished" and "finished wrong" the same
number again, which is the conflation `completion_rate` exists to remove. So the
key is *absent*, and the aggregate reports the population it averaged over.
"""

from __future__ import annotations

from smolqwen.eval.adapters.base import AdapterResult
from smolqwen.eval.metrics import TaskMetrics, aggregate


def metrics(category: str, result: AdapterResult, **overrides: object) -> TaskMetrics:
    payload: dict[str, object] = {
        "category": category,
        "score": result.score,
        "invalid_calls": 0,
        "steps": 1,
        "generated_tokens": 10,
        "truncated": False,
        "exact_success": result.exact_success,
        "diagnostics": dict(result.diagnostics),
    }
    payload.update(overrides)
    return TaskMetrics(**payload)  # type: ignore[arg-type]


def test_adapter_result_gained_its_new_fields_with_defaults() -> None:
    """Nine positional construction sites exist across four files; an undefaulted
    field would break every one of them."""
    minimal = AdapterResult(1.0, True)
    assert minimal.completed is True
    assert minimal.diagnostics == {}
    assert minimal.failure_reason is None
    assert minimal.failed_check_names == ()


def test_a_never_completed_task_reports_no_state_comparison() -> None:
    """Condition (c) is undefined here, so the key must be absent, not zero."""
    result = AdapterResult(
        0.0,
        False,
        completed=False,
        diagnostics={"completion_rate": 0.0, "snapshot_count_ratio": 0.0},
        failure_reason="never_completed",
    )
    assert "state_match_rate" not in result.diagnostics
    assert "result_match_rate" not in result.diagnostics
    assert result.failure_reason == "never_completed"


def test_state_match_rate_is_averaged_over_only_the_tasks_that_reported_it() -> None:
    """The restricted denominator, and it is printed rather than implied.

    Two tasks: one never completed, one completed and matched. Averaging
    `state_match_rate` over both would report 0.5 for a benchmark where every task
    that *could* be state-compared matched perfectly.
    """
    never_completed = AdapterResult(
        0.0,
        False,
        completed=False,
        diagnostics={"completion_rate": 0.0, "snapshot_count_ratio": 0.0},
    )
    matched = AdapterResult(
        1.0,
        True,
        diagnostics={
            "completion_rate": 1.0,
            "snapshot_count_ratio": 1.0,
            "state_match_rate": 1.0,
            "result_match_rate": 1.0,
        },
    )

    aggregated = aggregate(
        [metrics("multi_turn_base", never_completed), metrics("multi_turn_base", matched)]
    )["multi_turn_base"]

    assert aggregated["completion_rate"] == 0.5
    assert aggregated["completion_rate_denominator"] == 2.0
    # 1.0 over the one task where state comparison was defined -- not 0.5 over both.
    assert aggregated["state_match_rate"] == 1.0
    assert aggregated["state_match_rate_denominator"] == 1.0
    assert aggregated["result_match_rate_denominator"] == 1.0


def test_the_headline_score_still_separates_from_every_diagnostic() -> None:
    """`score` is not renamed or replaced. The first draft's `per_check_pass_rate`
    would have been exactly that: `verifier.py:246` already returns `passed / total`
    and `aggregate` already averages it."""
    aggregated = aggregate(
        [
            metrics(
                "envscaler_heldout",
                AdapterResult(
                    0.5,
                    False,
                    diagnostics={
                        "check_pass_count": 2.0,
                        "check_total": 4.0,
                        "name_error_count": 1.0,
                    },
                    failed_check_names=("check_a", "check_b"),
                ),
            )
        ]
    )["envscaler_heldout"]

    assert aggregated["score"] == 0.5
    # What the aggregate genuinely discarded: which checks failed, and how many
    # failed because the verifier could not run rather than because state was wrong.
    assert aggregated["name_error_count"] == 1.0
    assert aggregated["check_total"] == 4.0
    assert "per_check_pass_rate" not in aggregated


def test_an_absent_diagnostic_does_not_appear_in_the_aggregate_at_all() -> None:
    """Not as zero, not as NaN. A reader must be able to tell "not measured" from
    "measured as zero"."""
    aggregated = aggregate([metrics("fixture", AdapterResult(1.0, True))])["fixture"]
    assert not any(name.endswith("_denominator") for name in aggregated)
    assert set(aggregated) == {
        "score",
        "invalid_call_rate",
        "average_steps",
        "average_generated_tokens",
        "truncation_rate",
        "exact_success_rate",
    }
