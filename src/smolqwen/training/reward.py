"""Unmodified EnvScaler verifier reward at the TRL boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smolqwen.logging.trajectory_table import log_trajectory_columns


class RewardContractError(RuntimeError):
    """Raised when an infrastructure failure or malformed reward reaches GRPO."""


@dataclass(frozen=True)
class GroupRewardStats:
    mean_variance: float
    zero_variance_fraction: float
    group_count: int


def group_reward_stats(rewards: Sequence[float], group_indices: Sequence[int]) -> GroupRewardStats:
    """Population variance per positional GRPO group."""
    if len(rewards) != len(group_indices):
        raise RewardContractError("reward and group-index rows differ in length")
    grouped: dict[int, list[float]] = {}
    for reward, group_index in zip(rewards, group_indices, strict=True):
        grouped.setdefault(int(group_index), []).append(float(reward))
    variances: list[float] = []
    for values in grouped.values():
        mean = math.fsum(values) / len(values)
        variances.append(math.fsum((value - mean) ** 2 for value in values) / len(values))
    if not variances:
        return GroupRewardStats(0.0, 0.0, 0)
    return GroupRewardStats(
        mean_variance=math.fsum(variances) / len(variances),
        zero_variance_fraction=sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in variances)
        / len(variances),
        group_count=len(variances),
    )


def predicted_zero_variance_fraction(
    success_rates: Sequence[float], group_indices: Sequence[int], group_size: int
) -> float:
    """Bernoulli all-equal probability implied by difficulty profiling."""
    if not success_rates:
        return 0.0
    grouped: dict[int, float] = {}
    for rate, group_index in zip(success_rates, group_indices, strict=True):
        grouped.setdefault(int(group_index), float(rate))
    probabilities = [rate**group_size + (1.0 - rate) ** group_size for rate in grouped.values()]
    return math.fsum(probabilities) / len(probabilities) if probabilities else 0.0


def verifier_reward(
    *,
    rollout_reward: Sequence[float],
    terminal_reason: Sequence[str | None],
    group_index: Sequence[int],
    trajectory: Sequence[Mapping[str, Any]],
    difficulty_success_rate: Sequence[float] | None = None,
    num_generations: int | None = None,
    log_metric: Callable[[str, float], Any] | None = None,
    log_extra: Callable[[str, list[Any]], Any] | None = None,
    trajectory_sample_limit: int = 8,
    **_: Any,
) -> list[float]:
    """Return Phase 4's scalar unchanged and log diagnostics separately."""
    row_count = len(rollout_reward)
    if not (len(terminal_reason) == len(group_index) == len(trajectory) == row_count):
        raise RewardContractError("rollout reward metadata is not positionally aligned")
    if any(reason == "worker_crash" for reason in terminal_reason):
        raise RewardContractError("worker_crash reached the reward function; it must be replaced")
    rewards = [float(value) for value in rollout_reward]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rewards):
        raise RewardContractError("verifier rewards must be finite values in [0, 1]")

    stats = group_reward_stats(rewards, group_index)
    if log_metric is not None:
        log_metric("group_reward_variance/mean", stats.mean_variance)
        log_metric("group_reward_variance/zero_fraction", stats.zero_variance_fraction)
        if difficulty_success_rate is not None and num_generations is not None:
            log_metric(
                "group_reward_variance/predicted_zero_fraction",
                predicted_zero_variance_fraction(
                    difficulty_success_rate, group_index, num_generations
                ),
            )
    log_trajectory_columns(log_extra, trajectory, sample_limit=trajectory_sample_limit)
    return rewards
