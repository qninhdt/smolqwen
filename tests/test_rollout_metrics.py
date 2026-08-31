"""Rollout observability exposes the scheduler's correctness tripwires."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smolqwen.config_models import ProfileConfig
from smolqwen.rollout.episode import Episode
from smolqwen.rollout.metrics import (
    LOGP_DIFFERENCE_METRIC,
    LogpDifferenceStopCallback,
    drift_distribution,
    summarize_episodes,
    wandb_log_payload,
)
from smolqwen.rollout.profiler import profile_rollout


def test_active_pool_size_is_concurrency_times_multiplier() -> None:
    profile = ProfileConfig(generation_concurrency=6, active_pool_multiplier=3)
    assert profile.generation_batch_size == 18


def test_payload_includes_drift_queue_stragglers_and_timeline() -> None:
    episode = Episode("e", "s", 0, terminal_reason="final_answer", reward=0.5)
    episode.completion_ids = [1, 2]
    episode.drift_tally.observe("realign", 3)
    episode.drift_tally.observe("fork", 5)
    timeline = profile_rollout(
        episodes=[episode],
        wall_s=2.0,
        events=[(1.0, "e", "open"), (2.5, "e", "done")],
        queue_depth=[1, 2, 1],
        stage_intervals=[
            ("generation", 1.0, 1.5),
            ("env.step", 1.3, 2.0),
            ("env.destroy", 1.9, 2.0),
        ],
    )
    payload = wandb_log_payload(
        summarize_episodes(episodes=[episode], wall_s=2.0),
        drift_distribution([episode]),
        {"gpu_util_mean": 10.0, "gpu_util_peak": 20.0},
        timeline,
    )
    assert payload["rollout/drift_realign"] == 1
    assert payload["rollout/drift_fork"] == 1
    assert payload["rollout/drift_drift_tokens"] == 8
    assert payload["rollout/ready_queue_depth_peak"] == 2
    assert payload["rollout/straggler_max_s"] == 1.5
    assert payload["rollout/timeline_generation_s"] == 0.5
    assert payload["rollout/timeline_env.destroy_s"] == pytest.approx(0.1)
    # Overlapping generation/env work is merged before scheduling overhead.
    assert payload["rollout/timeline_scheduling_s"] == 1.0


def test_logp_difference_callback_stops_above_the_configured_threshold() -> None:
    callback = LogpDifferenceStopCallback(threshold=2.0)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(
        None,
        None,
        control,
        logs={LOGP_DIFFERENCE_METRIC: 2.1},
    )
    assert control.should_training_stop is True
