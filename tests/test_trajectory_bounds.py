"""Full-trajectory bounds."""

from __future__ import annotations

from pathlib import Path

from smolqwen.data.loader import Trajectory, parse_trajectory
from smolqwen.data.render import trim_after_last_assistant
from tests.helpers import load_trajectory_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_trajectory(task_id: str) -> Trajectory:
    for row in load_trajectory_rows():
        try:
            trajectory = parse_trajectory(row)
        except Exception:
            continue
        if trajectory.task_id == task_id:
            return trajectory
    raise AssertionError(f"fixture missing {task_id}")


def test_conversation_remains_one_trajectory_and_trims_terminal_user() -> None:
    trajectory = _fixture_trajectory("env_82_sft-task_29")
    bounded, removed = trim_after_last_assistant(trajectory.messages)
    assert removed == 1
    assert bounded[-1].role == "assistant"
    assert sum(message.is_real_user_turn for message in bounded) == 2


def test_nonconversation_remains_one_trajectory() -> None:
    trajectory = _fixture_trajectory("env_3_sft-task_28")
    bounded, removed = trim_after_last_assistant(trajectory.messages)
    assert bounded[-1].role == "assistant"
    assert removed == 1


def test_task_id_and_trajectory_uid_have_distinct_contracts() -> None:
    trajectory = _fixture_trajectory("env_82_sft-task_29")
    assert trajectory.task_id == "env_82_sft-task_29"
    assert trajectory.trajectory_uid == "env_82_sft-task_29:conversation"
