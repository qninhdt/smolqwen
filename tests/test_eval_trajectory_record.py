from __future__ import annotations

from pathlib import Path

from smolqwen.eval.trajectories import (
    TrajectoryRecord,
    append_trajectories,
    read_trajectories,
    write_trajectories,
)


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    records = [
        TrajectoryRecord(task_id="passes", category="multi_turn_base", score=1.0),
        TrajectoryRecord(
            task_id="fails",
            category="multi_turn_base",
            score=0.0,
            failure_reason="multi_turn:instance_state_mismatch",
        ),
    ]

    path = write_trajectories(tmp_path, tag="base", adapter="bfcl_multi_turn_base", records=records)

    assert path.name == "base-bfcl_multi_turn_base.jsonl"
    rows = read_trajectories(path)
    assert [row["task_id"] for row in rows] == ["passes", "fails"]
    assert rows[1]["failure_reason"] == "multi_turn:instance_state_mismatch"


def test_trajectory_append_is_visible_before_the_evaluation_finishes(tmp_path: Path) -> None:
    first = TrajectoryRecord(task_id="a", category="multi_turn_base", score=1.0)
    second = TrajectoryRecord(task_id="b", category="multi_turn_base", score=0.0)

    with append_trajectories(tmp_path, tag="base", adapter="bfcl_multi_turn_base") as (
        path,
        append,
    ):
        append(first)
        assert [row["task_id"] for row in read_trajectories(path)] == ["a"]
        append(second)
        assert [row["task_id"] for row in read_trajectories(path)] == ["a", "b"]


def test_writing_the_same_tag_twice_replaces_rather_than_interleaves(
    tmp_path: Path,
) -> None:
    first = [TrajectoryRecord(task_id="a", category="multi_turn_base", score=1.0)]
    second = [TrajectoryRecord(task_id="b", category="multi_turn_base", score=0.0)]

    write_trajectories(tmp_path, tag="base", adapter="bfcl_multi_turn_base", records=first)
    path = write_trajectories(tmp_path, tag="base", adapter="bfcl_multi_turn_base", records=second)

    assert [row["task_id"] for row in read_trajectories(path)] == ["b"]
