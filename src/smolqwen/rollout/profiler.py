"""Timeline attribution: where the rollout call's wall time actually went.

Answers the question the phase exists for: after removing the turn barrier, is
the remaining idle time CPU-bound environments, scheduler overhead, or the
known intra-call tail? The rows are assembled from what the run already
recorded — episode `stage_timings`, scheduler `events`, queue-depth samples —
so profiling adds no instrumentation of its own to the hot path.

Stages, and why each is its own row. Stage busy times may overlap — that is the
point of the ready queue — while `scheduling` is the complement of the union of
all recorded intervals, so it never double-counts concurrent environment work:

- `generation` — vLLM sampling, the only GPU row.
- `tokenization` — prefix re-renders; grows with conversation length and
  template cost, and it is the scheduler's own tax, not the environment's.
- `env.step` — worker round trips, including queue wait behind other episodes
  on the same worker.
- `env.destroy` — worker cleanup awaited before a rollout row is returned.
- `parse` — tool-call extraction per turn.
- `verifier` — checklist scoring per episode (once, at terminal).
- `scheduling` — cycle overhead the scheduler adds between the others,
  computed as the call's wall time minus everything attributable above; a
  first-class row because "scheduler overhead eats the gain" is a stated risk,
  and without the row it is indistinguishable from environment slowness.

Weight sync is measured by the bench around the trainer step, not here: it
happens before `rollout_func` is entered, once per step, and is a
full-parameter round trip that also invalidates the prefix cache — its cost
belongs to the step's timeline, not to any episode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from smolqwen.rollout.episode import Episode

STAGES = (
    "generation",
    "tokenization",
    "env.create",
    "env.step",
    "env.destroy",
    "parse",
    "verifier",
    "scheduling",
)


@dataclass
class StageRow:
    """One stage's share of the call."""

    stage: str
    total_s: float
    calls: int
    mean_s: float = 0.0
    max_s: float = 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "stage": self.stage,
            "total_s": round(self.total_s, 4),
            "calls": self.calls,
            "mean_s": round(self.mean_s, 6),
            "max_s": round(self.max_s, 4),
        }


@dataclass
class TimelineProfile:
    """The attribution plus the observability series the risks reference."""

    wall_s: float
    stages: list[StageRow] = field(default_factory=list)
    queue_depth: tuple[int, ...] = ()
    # Per-episode wall time, the straggler distribution's input: with the turn
    # barrier removed, the spread tells whether a few slow environments still
    # dominate the batch tail.
    episode_wall_s: tuple[float, ...] = ()
    episodes: int = 0
    terminals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "wall_s": round(self.wall_s, 4),
            "episodes": self.episodes,
            "terminals": dict(self.terminals),
            "stages": [stage.as_dict() for stage in self.stages],
            "queue_depth_final": list(self.queue_depth[-16:]),
            "straggler_p50": _percentile(self.episode_wall_s, 50),
            "straggler_p90": _percentile(self.episode_wall_s, 90),
            "straggler_max": max(self.episode_wall_s, default=0.0),
        }


def profile_rollout(
    *,
    episodes: Sequence[Episode],
    wall_s: float,
    events: Sequence[tuple[float, str, str]],
    queue_depth: Sequence[int],
    stage_intervals: Sequence[tuple[str, float, float]] = (),
) -> TimelineProfile:
    """Attribute one rollout call's wall time across stages."""
    totals: dict[str, list[float]] = {}
    for episode in episodes:
        for stage, samples in episode.stage_timings.items():
            totals.setdefault(stage, []).extend(samples)

    stage_rows: list[StageRow] = []
    for stage in STAGES:
        if stage == "scheduling":
            continue
        samples = totals.get(stage, [])
        intervals = [
            (start, end)
            for interval_stage, start, end in stage_intervals
            if interval_stage == stage
        ]
        if not samples and not intervals:
            continue
        total = _union_duration(intervals) if intervals else math.fsum(samples)
        durations = samples or [max(0.0, end - start) for start, end in intervals]
        stage_rows.append(
            StageRow(
                stage=stage,
                total_s=total,
                calls=len(durations),
                mean_s=math.fsum(durations) / len(durations),
                max_s=max(durations),
            )
        )
    busy = _union_duration([(start, end) for _, start, end in stage_intervals])
    if not stage_intervals:
        busy = min(wall_s, math.fsum(row.total_s for row in stage_rows))
    overhead = max(0.0, wall_s - busy)
    stage_rows.append(StageRow(stage="scheduling", total_s=overhead, calls=len(queue_depth)))

    episode_wall = _episode_walls(events)
    terminals: dict[str, int] = {}
    for episode in episodes:
        key = episode.terminal_reason or "none"
        terminals[key] = terminals.get(key, 0) + 1

    return TimelineProfile(
        wall_s=wall_s,
        stages=stage_rows,
        queue_depth=tuple(queue_depth),
        episode_wall_s=episode_wall,
        episodes=len(episodes),
        terminals=terminals,
    )


def _episode_walls(events: Sequence[tuple[float, str, str]]) -> tuple[float, ...]:
    """Per-episode open-to-done spans from the scheduler event log."""
    opened: dict[str, float] = {}
    walls: list[float] = []
    for at, episode_id, event in events:
        if event in ("open", "replacement_open"):
            opened[episode_id] = at
        elif event == "done":
            start = opened.pop(episode_id, None)
            if start is not None:
                walls.append(max(0.0, at - start))
    return tuple(sorted(walls))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(percentile / 100 * (len(ordered) - 1))))
    return round(ordered[index], 4)


def _union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    """Wall time covered by at least one interval, merging overlaps."""
    ordered = sorted((start, max(start, end)) for start, end in intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def format_timeline(profile: TimelineProfile) -> str:
    """A compact table the bench report embeds verbatim."""
    lines = [
        f"wall {profile.wall_s:.2f}s over {profile.episodes} episodes "
        f"({profile.episodes / profile.wall_s * 3600:.1f} episodes/h if sustained)"
    ]
    for row in profile.stages:
        share = row.total_s / profile.wall_s * 100 if profile.wall_s else 0.0
        lines.append(
            f"  {row.stage:<12} {row.total_s:>9.2f}s  {share:>5.1f}%  "
            f"n={row.calls:<5} mean={row.mean_s * 1000:>7.2f}ms max={row.max_s * 1000:>8.2f}ms"
        )
    lines.append(f"  terminals: {profile.terminals}")
    return "\n".join(lines)
