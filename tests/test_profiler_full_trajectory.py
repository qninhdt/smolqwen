"""Profiler measures complete preserved-reasoning trajectories as samples."""

from __future__ import annotations

import pytest

from smolqwen.data.loader import Trajectory, parse_trajectory
from smolqwen.data.profiler import SEQ_LENGTH_CANDIDATES, choose_budgets, profile_dataset
from smolqwen.data.render import RenderError
from tests.helpers import OfflineTokenizer, load_trajectory_rows


def _trajectories() -> list[Trajectory]:
    return [parse_trajectory(row) for row in load_trajectory_rows()[:2]]


def test_profile_counts_rows_samples_tasks_and_unique_uids_separately() -> None:
    result = profile_dataset(
        OfflineTokenizer(),
        "not-used.json",
        trajectories=iter(_trajectories()),
    )
    payload = result.to_dict()
    assert payload["counts"]["rows_read"] == 2
    assert payload["counts"]["parsed"] == 2
    assert payload["counts"]["samples"] == 2
    assert payload["counts"]["unique_task_groups"] == 2
    assert payload["counts"]["unique_trajectory_uids"] == 2
    assert payload["by_mode"]["conversation"]["samples"] == 1
    assert payload["by_mode"]["non_conversation"]["samples"] == 1


def test_profile_uses_same_one_sample_length_as_converter() -> None:
    result = profile_dataset(
        OfflineTokenizer(),
        "not-used.json",
        trajectories=iter(_trajectories()),
    )
    assert result.all_total_tokens() == result.all_sample_tokens()
    assert len(result.all_sample_tokens()) == 2
    for cap in SEQ_LENGTH_CANDIDATES:
        assert result.trajectory_retention(cap) == result.retention(result.all_sample_tokens(), cap)


def test_budget_recommends_accepted_32k_operating_point() -> None:
    result = profile_dataset(
        OfflineTokenizer(),
        "not-used.json",
        trajectories=iter(_trajectories()),
    )
    assert choose_budgets(result)["recommended"]["max_seq_length"] == 32768


def test_duplicate_row_uid_fails_loudly() -> None:
    trajectory = _trajectories()[0]
    with pytest.raises(RenderError, match="duplicate trajectory_uid"):
        profile_dataset(
            OfflineTokenizer(),
            "not-used.json",
            trajectories=iter([trajectory, trajectory]),
        )
