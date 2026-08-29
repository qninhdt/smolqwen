"""Measure the trajectory distribution, then write the budgets every later stage reads.

Measure first, convert second. Every context and step cap in Phases 3, 6 and 7
comes from this pass rather than from a guess, and `budgets.json` records for each
candidate cap what fraction of the dataset it retains -- because filtering to
short trajectories means training on a self-defined slice, and that has to be a
stated decision rather than a silent one.

The pass is streaming: the input is 701 MB and the parsed graph is several times
that, so nothing holds all 9k trajectories at once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolqwen.data.loader import LoadStats, Trajectory, iter_trajectories, sha256_of
from smolqwen.data.render import (
    RenderError,
    Tokenizer,
    render_prefix,
    render_segment,
    split_segments,
)

# Candidate sequence caps reported with their retention. 32k is the paper's
# training cap; the smaller ones are what a 24 GB card is likely to afford.
SEQ_LENGTH_CANDIDATES: tuple[int, ...] = (4096, 8192, 12288, 16384, 24576, 32768)
STEP_CANDIDATES: tuple[int, ...] = (8, 12, 16, 24, 32, 48)
PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)

# Below this, the cap is reshaping the distribution rather than trimming its tail,
# and the plan's stated response is to raise the cap and cut batch size instead.
RETENTION_FLOOR = 0.70


def percentile(values: Sequence[float], q: int) -> float:
    """Nearest-rank percentile. No numpy dependency in the data path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0:
        return float(ordered[0])
    if q >= 100:
        return float(ordered[-1])
    rank = (q / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarise(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    summary: dict[str, float] = {
        "count": len(values),
        "mean": sum(values) / len(values),
        "max": float(max(values)),
    }
    for q in PERCENTILES:
        summary[f"p{q}"] = percentile(values, q)
    return summary


@dataclass
class ModeAccumulator:
    """Per-mode (Conv / Non-Conv) distributions. Percentiles, not just means."""

    trajectories: int = 0
    user_turns: list[float] = field(default_factory=list)
    tool_steps: list[float] = field(default_factory=list)
    # The rendered length of the whole conversation. Informative, but NOT what the
    # SFT cap has to accommodate -- see `sample_tokens`.
    total_tokens: list[float] = field(default_factory=list)
    # The rendered length of each **sample** the converter emits. This is the unit
    # the cap applies to, and it is not bounded by `total_tokens`: a per-segment
    # render retains that segment's reasoning, which the whole-conversation render
    # strips for every turn before the last real user query. Measuring only the
    # full render therefore reports a retention the conversion cannot deliver.
    sample_tokens: list[float] = field(default_factory=list)
    # The longest sample per trajectory, so trajectory-level retention can be
    # computed: the converter drops a trajectory whole when any sample is over cap.
    longest_sample_per_trajectory: list[float] = field(default_factory=list)
    samples: int = 0
    reasoning_tokens_per_turn: list[float] = field(default_factory=list)
    observation_tokens: list[float] = field(default_factory=list)
    tool_count: list[float] = field(default_factory=list)
    assistant_turns: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "samples": self.samples,
            "user_turns": summarise(self.user_turns),
            "tool_steps": summarise(self.tool_steps),
            "conversation_tokens": summarise(self.total_tokens),
            "sample_tokens": summarise(self.sample_tokens),
            "reasoning_tokens_per_assistant_turn": summarise(self.reasoning_tokens_per_turn),
            "observation_tokens": summarise(self.observation_tokens),
            "tools_per_environment": summarise(self.tool_count),
            "assistant_turns": summarise(self.assistant_turns),
        }


@dataclass
class ProfileResult:
    modes: dict[str, ModeAccumulator]
    load_stats: LoadStats
    input_path: str
    input_sha256: str
    input_revision: str | None = None

    def all_total_tokens(self) -> list[float]:
        return [value for mode in self.modes.values() for value in mode.total_tokens]

    def all_sample_tokens(self) -> list[float]:
        """The per-sample rendered lengths -- the unit `max_seq_length` caps."""
        return [value for mode in self.modes.values() for value in mode.sample_tokens]

    def all_tool_steps(self) -> list[float]:
        return [value for mode in self.modes.values() for value in mode.tool_steps]

    def retention(self, values: Sequence[float], cap: int) -> float:
        if not values:
            return 0.0
        return sum(1 for value in values if value <= cap) / len(values)

    def trajectory_retention(self, cap: int) -> float:
        """The fraction of *trajectories* the converter keeps at `cap`.

        A trajectory is dropped whole when any of its samples exceeds the cap, so
        this is the number that describes the training distribution -- not the
        per-sample fraction, which is higher.
        """
        kept = 0
        total = 0
        for mode in self.modes.values():
            for longest in mode.longest_sample_per_trajectory:
                total += 1
                if longest <= cap:
                    kept += 1
        return kept / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        totals = self.all_total_tokens()
        samples = self.all_sample_tokens()
        steps = self.all_tool_steps()
        return {
            "input": {
                "path": self.input_path,
                "sha256": self.input_sha256,
                "revision": self.input_revision,
            },
            "counts": {
                "rows_read": self.load_stats.total,
                "parsed": self.load_stats.parsed,
                "malformed": self.load_stats.malformed,
                "malformed_reasons": self.load_stats.malformed_reasons or {},
                "samples": sum(mode.samples for mode in self.modes.values()),
            },
            "by_mode": {name: acc.to_dict() for name, acc in sorted(self.modes.items())},
            "overall": {
                "conversation_tokens": summarise(totals),
                "sample_tokens": summarise(samples),
                "tool_steps": summarise(steps),
            },
            "retention": {
                # Per-sample and per-trajectory retention differ, and the second is
                # the one that describes the training distribution: the converter
                # drops a trajectory whole when any of its samples exceeds the cap.
                "max_seq_length_samples": {
                    str(cap): round(self.retention(samples, cap), 4)
                    for cap in SEQ_LENGTH_CANDIDATES
                },
                "max_seq_length_trajectories": {
                    str(cap): round(self.trajectory_retention(cap), 4)
                    for cap in SEQ_LENGTH_CANDIDATES
                },
                "max_env_steps": {
                    str(cap): round(self.retention(steps, cap), 4) for cap in STEP_CANDIDATES
                },
            },
        }


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def profile_trajectory(
    tokenizer: Tokenizer, trajectory: Trajectory, accumulator: ModeAccumulator
) -> None:
    """Accumulate one trajectory's statistics.

    Two lengths are measured, and the distinction is load-bearing:

    - the **conversation** length, i.e. the whole trajectory rendered once. The
      template strips reasoning for every assistant turn before the last real user
      query, so this is *shorter* than the sum of the samples.
    - the **sample** lengths, i.e. what `convert_sft` actually emits: one render
      per real user boundary, each retaining that segment's reasoning. This is the
      unit `max_seq_length` caps, so it is what the retention numbers must be
      computed from. Measuring only the conversation length reports a retention the
      conversion cannot deliver.
    """
    accumulator.trajectories += 1
    accumulator.user_turns.append(trajectory.real_user_turns)
    accumulator.tool_steps.append(trajectory.tool_steps)
    accumulator.tool_count.append(len(trajectory.tools))

    rendered = render_prefix(tokenizer, trajectory.messages, tools=trajectory.tools)
    accumulator.total_tokens.append(_count_tokens(tokenizer, rendered))

    # The per-sample lengths, from the same splitter the converter uses.
    longest = 0.0
    try:
        segments = split_segments(trajectory.messages)
    except RenderError:
        segments = []
    for index, segment in enumerate(segments):
        sample = render_segment(
            tokenizer,
            trajectory.messages,
            segment,
            tools=trajectory.tools,
            trajectory_id=trajectory.trajectory_id,
            segment_index=index,
        )
        accumulator.sample_tokens.append(sample.total_tokens)
        accumulator.samples += 1
        longest = max(longest, float(sample.total_tokens))
    if segments:
        accumulator.longest_sample_per_trajectory.append(longest)

    assistant_turns = 0
    for message in trajectory.messages:
        if message.role == "assistant":
            assistant_turns += 1
            accumulator.reasoning_tokens_per_turn.append(
                _count_tokens(tokenizer, message.reasoning_content or "")
            )
        elif message.role == "tool" or message.is_wrapped_tool_response:
            accumulator.observation_tokens.append(_count_tokens(tokenizer, message.content or ""))
    accumulator.assistant_turns.append(assistant_turns)


def profile_dataset(
    tokenizer: Tokenizer,
    path: Path | str,
    *,
    limit: int | None = None,
    revision: str | None = None,
    trajectories: Iterable[Trajectory] | None = None,
) -> ProfileResult:
    """Single streaming pass over the release file."""
    stats = LoadStats()
    modes: dict[str, ModeAccumulator] = {
        "conversation": ModeAccumulator(),
        "non_conversation": ModeAccumulator(),
    }
    source = (
        trajectories
        if trajectories is not None
        else iter_trajectories(path, limit=limit, stats=stats)
    )
    for trajectory in source:
        mode = trajectory.traj_type or "unknown"
        accumulator = modes.setdefault(mode, ModeAccumulator())
        profile_trajectory(tokenizer, trajectory, accumulator)

    return ProfileResult(
        modes=modes,
        load_stats=stats,
        input_path=str(path),
        input_sha256=sha256_of(path) if Path(path).is_file() else "",
        input_revision=revision,
    )


def choose_budgets(result: ProfileResult) -> dict[str, Any]:
    """Pick the smallest cap that still retains most of the distribution.

    "Most" is `RETENTION_FLOOR`, measured on **trajectory** retention: the
    converter drops a trajectory whole when any of its samples exceeds the cap, so
    the per-sample fraction would overstate what training actually keeps.

    Below the floor the cap is reshaping the training distribution rather than
    trimming its tail, and the retained fraction is reported next to the choice so
    the trade-off is visible rather than implied.
    """
    samples = result.all_sample_tokens()
    steps = result.all_tool_steps()

    def smallest_above_floor(candidates: Sequence[int], values: Sequence[float]) -> int:
        for candidate in candidates:
            if result.retention(values, candidate) >= RETENTION_FLOOR:
                return candidate
        return candidates[-1]

    max_seq_length = SEQ_LENGTH_CANDIDATES[-1]
    for candidate in SEQ_LENGTH_CANDIDATES:
        if result.trajectory_retention(candidate) >= RETENTION_FLOOR:
            max_seq_length = candidate
            break
    max_env_steps = smallest_above_floor(STEP_CANDIDATES, steps)

    # Per-step generation cap: enough headroom for one reasoning block plus one
    # tool call, taken from the p99 of observed per-turn reasoning rather than
    # from the whole-trajectory length.
    reasoning = [
        value
        for accumulator in result.modes.values()
        for value in accumulator.reasoning_tokens_per_turn
    ]
    per_step = max(256, int(percentile(reasoning, 99) * 2) if reasoning else 1024)

    return {
        "recommended": {
            "max_seq_length": max_seq_length,
            "max_env_steps": max_env_steps,
            "max_new_tokens_per_step": per_step,
        },
        "retention_at_recommended": {
            # Trajectories is the number that describes the training distribution;
            # samples is reported alongside it because it is the cap's literal unit.
            "max_seq_length_trajectories": round(result.trajectory_retention(max_seq_length), 4),
            "max_seq_length_samples": round(result.retention(samples, max_seq_length), 4),
            "max_env_steps": round(result.retention(steps, max_env_steps), 4),
        },
        "retention_floor": RETENTION_FLOOR,
        "consumers": {
            "max_seq_length": "Phase 3 SFT (may lower via the OOM sweep, never raise)",
            "max_new_tokens_per_step": "Phase 6 per-step generation cap",
            "max_env_steps": "Phase 7 step cap (widening sweep)",
        },
        "candidates": result.to_dict()["retention"],
    }


def write_profile(result: ProfileResult, output_dir: Path | str) -> tuple[Path, Path]:
    """Write `profile.json` and `budgets.json`, returning both paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    profile_path = directory / "profile.json"
    profile_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    budgets_path = directory / "budgets.json"
    budgets_path.write_text(
        json.dumps(choose_budgets(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return profile_path, budgets_path


def format_profile_table(result: ProfileResult) -> str:
    """A readable summary for the terminal; the JSON is the machine-readable one."""
    payload = result.to_dict()
    counts = payload["counts"]
    lines = [
        f"rows read {counts['rows_read']}  parsed {counts['parsed']}  "
        f"malformed {counts['malformed']}  samples {counts['samples']}",
        "",
        f"{'mode':<18}{'trajs':>8}{'samples':>9}{'p50 smp':>9}{'p95 smp':>9}"
        f"{'max smp':>9}{'p50 stp':>9}{'p95 stp':>9}",
    ]
    for name, mode in payload["by_mode"].items():
        if not mode["trajectories"]:
            continue
        tokens = mode["sample_tokens"]
        steps = mode["tool_steps"]
        lines.append(
            f"{name:<18}{mode['trajectories']:>8}{mode['samples']:>9}"
            f"{tokens.get('p50', 0):>9.0f}{tokens.get('p95', 0):>9.0f}"
            f"{tokens.get('max', 0):>9.0f}"
            f"{steps.get('p50', 0):>9.1f}{steps.get('p95', 0):>9.1f}"
        )
    # Trajectory retention is the training-distribution number: a trajectory is
    # dropped whole when any of its samples exceeds the cap.
    lines += ["", f"{'cap':>7}{'trajectories':>15}{'samples':>10}"]
    trajectories = payload["retention"]["max_seq_length_trajectories"]
    samples = payload["retention"]["max_seq_length_samples"]
    for cap, retained in trajectories.items():
        flag = "" if retained >= RETENTION_FLOOR else "  <- below floor"
        lines.append(f"{cap:>7}{retained:>14.1%}{samples[cap]:>10.1%}{flag}")
    return "\n".join(lines)
