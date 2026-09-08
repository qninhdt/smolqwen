from __future__ import annotations

import pytest

from smolqwen.training.reward import group_reward_stats, verifier_reward


def test_variance_is_computed_per_group_and_zero_groups_are_counted() -> None:
    stats = group_reward_stats([0.0, 1.0, 0.5, 0.5], [0, 0, 1, 1])
    assert stats.group_count == 2
    assert stats.mean_variance == pytest.approx(0.125)
    assert stats.zero_variance_fraction == pytest.approx(0.5)
    assert stats.useful_group_rate == pytest.approx(0.5)


def test_verifier_reward_logs_useful_group_rate() -> None:
    logged: dict[str, float] = {}
    verifier_reward(
        rollout_reward=[0.0, 1.0, 0.5, 0.5],
        terminal_reason=["answer"] * 4,
        group_index=[0, 0, 1, 1],
        trajectory=[{}] * 4,
        log_metric=lambda name, value: logged.__setitem__(name, value),
    )
    assert logged["useful_group_rate"] == pytest.approx(0.5)
