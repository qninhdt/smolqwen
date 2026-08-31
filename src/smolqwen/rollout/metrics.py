"""Rollout metrics: throughput, drift, and the one silent-corruption tripwire.

Episodes/hour is the phase's headline number; GPU utilization percent is a
diagnostic, never a target. The drift distribution is emitted because
`fork_threshold_tokens` is otherwise a threshold tuned blind — upstream logs
the same tally for the same reason.

`sampling/sampling_logp_difference/max` is computed by TRL itself whenever
`vllm_importance_sampling_correction` is on. It is the only visible symptom of
a misaligned `logprobs` array: ratios go to exp(±large), get clamped, and
training continues without error. `should_stop_on_logp_difference` turns the
configured threshold into a decision the bench (and, in Phase 7, a trainer
callback) applies after every logged step.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smolqwen.rollout.episode import DriftTally, Episode
from smolqwen.rollout.profiler import TimelineProfile

LOGP_DIFFERENCE_METRIC = "sampling/sampling_logp_difference/max"


class GpuUtilizationSampler:
    """Mean and peak GPU utilization over a rollout call, sampled on a thread.

    A diagnostic only: the number that matters is episodes/hour. Uses NVML
    directly when available (the probe module's dependency posture) and reports
    zeros rather than guessing when it is not — a fabricated utilization curve
    would be indistinguishable from a real one in the A/B report.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._available = False
        self._handle = None
        try:  # pragma: no cover - depends on the machine
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._available = True
        except Exception:
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._available

    def sample_once(self) -> float:  # pragma: no cover - hardware path
        if not self._available:
            return 0.0
        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(utilization.gpu)
        except Exception:
            return 0.0

    def start(self) -> None:  # pragma: no cover - hardware path
        if not self._available:
            return
        stop = threading.Event()
        self._stop = stop

        def loop() -> None:
            while not stop.wait(self.interval_s):
                self._samples.append(self.sample_once())

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        if self._thread is not None and self._stop is not None:  # pragma: no cover
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        return {
            "gpu_util_mean": _mean(self._samples),
            "gpu_util_peak": max(self._samples, default=0.0),
        }


def summarize_episodes(
    *,
    episodes: Sequence[Episode],
    wall_s: float,
) -> dict[str, object]:
    """Throughput and behavior counters for one rollout call."""
    rewards = [episode.reward for episode in episodes if episode.reward is not None]
    terminals: dict[str, int] = {}
    for episode in episodes:
        key = episode.terminal_reason or "none"
        terminals[key] = terminals.get(key, 0) + 1
    tokens = sum(len(episode.completion_ids) for episode in episodes)
    return {
        "episodes": len(episodes),
        "episodes_per_hour": len(episodes) / wall_s * 3600 if wall_s else 0.0,
        "tokens_per_s": tokens / wall_s if wall_s else 0.0,
        "mean_reward": _mean(rewards),
        "mean_steps": _mean([episode.step_count for episode in episodes]),
        "invalid_calls": sum(episode.invalid_call_count for episode in episodes),
        "replacements": sum(1 for e in episodes if e.replaced_episode_id),
        "terminals": terminals,
    }


def drift_distribution(episodes: Sequence[Episode]) -> dict[str, int]:
    """Aggregate every episode's drift tally — the threshold's only feedback."""
    combined = DriftTally()
    for episode in episodes:
        tally = episode.drift_tally
        combined.clean += tally.clean
        combined.realign += tally.realign
        combined.fork += tally.fork
        combined.transitions += tally.transitions
        combined.drift_tokens += tally.drift_tokens
        combined.drift_max = max(combined.drift_max, tally.drift_max)
    return combined.to_dict()


def should_stop_on_logp_difference(logs: Mapping[str, float], *, threshold: float) -> bool:
    """Whether a logged step's max sampling-logp difference crosses the stop line.

    The metric name is TRL's own; absent means the step did not log it (IS
    correction off, or no vLLM), which is not a reason to stop.
    """
    value = logs.get(LOGP_DIFFERENCE_METRIC)
    if value is None or math.isnan(float(value)):
        return False
    return float(value) > threshold


def wandb_log_payload(
    summary: Mapping[str, object],
    drift: Mapping[str, float | int],
    gpu: Mapping[str, float],
    timeline: TimelineProfile | None = None,
) -> dict[str, float | int]:
    """The flat dict handed to `trainer.log` / W&B, prefixed per convention."""
    payload: dict[str, float | int] = {}
    for key in ("episodes_per_hour", "tokens_per_s", "mean_reward", "mean_steps"):
        payload[f"rollout/{key}"] = _as_float(summary.get(key, 0.0))
    payload["rollout/invalid_calls"] = _as_int(summary.get("invalid_calls", 0))
    payload["rollout/replacements"] = _as_int(summary.get("replacements", 0))
    for key in ("clean", "realign", "fork", "drift_tokens", "drift_max"):
        payload[f"rollout/drift_{key}"] = int(drift.get(key, 0))
    for key in ("gpu_util_mean", "gpu_util_peak"):
        payload[f"rollout/{key}"] = float(gpu.get(key, 0.0))
    if timeline is not None:
        payload["rollout/ready_queue_depth_mean"] = _mean(timeline.queue_depth)
        payload["rollout/ready_queue_depth_peak"] = max(timeline.queue_depth, default=0)
        payload["rollout/straggler_p50_s"] = _percentile(timeline.episode_wall_s, 50)
        payload["rollout/straggler_p90_s"] = _percentile(timeline.episode_wall_s, 90)
        payload["rollout/straggler_max_s"] = max(timeline.episode_wall_s, default=0.0)
        for row in timeline.stages:
            payload[f"rollout/timeline_{row.stage}_s"] = row.total_s
    return payload


class LogpDifferenceStopCallback:
    """Trainer callback that stops after TRL reports a corrupt IS alignment."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        if logs is not None and should_stop_on_logp_difference(logs, threshold=self.threshold):
            control.should_training_stop = True
        return control


def _mean(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    return math.fsum(numeric) / len(numeric) if numeric else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, round(percentile / 100 * (len(ordered) - 1))),
    )
    return ordered[index]


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


@dataclass(frozen=True)
class ABReportRow:
    """One path's headline numbers in the A/B table."""

    path: str
    profile: str
    episodes_per_hour: float
    tokens_per_s: float
    gpu_util_mean: float
    gpu_util_peak: float
    mean_reward: float
    notes: str = ""

    def as_markdown(self) -> str:
        return (
            f"| {self.path} | {self.profile} | {self.episodes_per_hour:.1f} | "
            f"{self.tokens_per_s:.0f} | {self.gpu_util_mean:.1f}% | "
            f"{self.gpu_util_peak:.1f}% | {self.mean_reward:.4f} | {self.notes} |"
        )


AB_TABLE_HEADER = (
    "| path | profile | episodes/hour | tokens/s | gpu util mean | gpu util peak "
    "| mean reward | notes |\n|---|---|---|---|---|---|---|---|"
)
