from __future__ import annotations

from types import SimpleNamespace

import pytest

from smolqwen.training.grpo import GroupVarianceStopCallback
from smolqwen.training.reward import group_reward_stats, predicted_zero_variance_fraction


def test_variance_is_computed_per_group_and_zero_groups_are_counted() -> None:
    stats = group_reward_stats([0.0, 1.0, 0.5, 0.5], [0, 0, 1, 1])
    assert stats.group_count == 2
    assert stats.mean_variance == pytest.approx(0.125)
    assert stats.zero_variance_fraction == pytest.approx(0.5)


def test_profile_predicts_the_probability_of_an_all_equal_group() -> None:
    predicted = predicted_zero_variance_fraction([0.5, 0.5, 1.0, 1.0], [0, 0, 1, 1], 2)
    assert predicted == pytest.approx(0.75)


def test_material_excess_zero_variance_stops_training() -> None:
    callback = GroupVarianceStopCallback(after_steps=10, multiplier=1.5, margin=0.1)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(
        SimpleNamespace(),
        SimpleNamespace(global_step=10),
        control,
        logs={
            "group_reward_variance/zero_fraction": 0.8,
            "group_reward_variance/predicted_zero_fraction": 0.2,
        },
    )
    assert control.should_training_stop is True
