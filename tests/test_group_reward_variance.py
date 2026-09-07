from __future__ import annotations

import pytest

from smolqwen.training.reward import group_reward_stats


def test_variance_is_computed_per_group_and_zero_groups_are_counted() -> None:
    stats = group_reward_stats([0.0, 1.0, 0.5, 0.5], [0, 0, 1, 1])
    assert stats.group_count == 2
    assert stats.mean_variance == pytest.approx(0.125)
    assert stats.zero_variance_fraction == pytest.approx(0.5)
