"""Full-trajectory bounds and seeded split by task id."""

from __future__ import annotations

from pathlib import Path

from smolqwen.data.loader import Trajectory, parse_trajectory
from smolqwen.data.render import trim_after_last_assistant
from smolqwen.data.splits import build_env_split_manifest, split_trajectory_ids
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


def test_split_by_task_id_is_seeded_and_reproducible() -> None:
    ids = [f"task-{index}" for index in range(120)]
    first = split_trajectory_ids(ids, seed=7, val_fraction=0.1)
    second = split_trajectory_ids(ids, seed=7, val_fraction=0.1)
    assert first == second
    assert len(first.val_ids) == 12
    assert first.train_ids.isdisjoint(first.val_ids)


def test_paired_row_uids_never_straddle_split() -> None:
    task_ids = [f"task-{index}" for index in range(50)]
    split = split_trajectory_ids(task_ids, seed=11, val_fraction=0.1)
    for task_id in task_ids:
        row_uids = [f"{task_id}:conversation", f"{task_id}:non_conversation"]
        assert len({split.partition(task_id) for _ in row_uids}) == 1


def test_env_split_manifest_from_metadata() -> None:
    env_meta = FIXTURES / "env_metadata_small.json"
    env_meta.write_text(
        '{"env_1_sft": {}, "env_2_sft": {}, "env_10_rl": {}, "env_11_rl": {}, "env_9_other": {}}',
        encoding="utf-8",
    )
    try:
        manifest = build_env_split_manifest(
            env_meta,
            rl_scenario_env_ids=["env_10_rl", "env_11_rl", "env_1_sft"],
        )
        assert manifest.sft_env_ids == ("env_1_sft", "env_2_sft")
        assert manifest.rl_env_ids == ("env_10_rl", "env_11_rl")
        assert manifest.other_env_ids == ("env_9_other",)
        assert manifest.other_env_count == 1
    finally:
        env_meta.unlink(missing_ok=True)
