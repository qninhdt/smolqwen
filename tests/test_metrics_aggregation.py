from __future__ import annotations

from smolqwen.eval.metrics import TaskMetrics, aggregate


def test_category_metrics_include_agentic_secondary_signals() -> None:
    result = aggregate(
        [
            TaskMetrics("bfcl", 1.0, 0, 2, 10, False),
            TaskMetrics("bfcl", 0.0, 1, 2, 30, True),
        ]
    )["bfcl"]
    assert result == {
        "score": 0.5,
        "invalid_call_rate": 0.25,
        "average_steps": 2.0,
        "average_generated_tokens": 20.0,
        "truncation_rate": 0.5,
    }
