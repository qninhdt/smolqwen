from __future__ import annotations

from smolqwen.logging.trajectory_table import trajectory_columns


def test_trajectory_table_preserves_reasoning_calls_observations_and_verdicts() -> None:
    columns = trajectory_columns(
        [
            {
                "scenario_id": "task-1",
                "terminal_reason": "final_answer",
                "messages": [
                    {"role": "assistant", "reasoning_content": "inspect", "content": ""},
                    {"role": "assistant", "content": "<tool_call>query()</tool_call>"},
                ],
                "observations": ["found"],
                "per_check_bools": [True, False],
                "invalid_call_count": 0,
                "step_count": 1,
            }
        ],
        sample_limit=1,
    )
    assert columns["reasoning"] == ["inspect"]
    assert "query" in columns["tool_calls"][0]
    assert columns["observations"] == ['["found"]']
    assert columns["checkpoint_verdicts"] == ["[true, false]"]
