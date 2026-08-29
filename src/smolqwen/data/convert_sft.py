"""Convert trajectories into Qwen3.5-rendered SFT samples, split at real user boundaries.

The converter emits:

- Non-Conv trajectory -> one sample, with loss on every assistant turn.
- Conv trajectory -> one sample per real user turn; within each segment loss
  covers that segment's assistant turns only.

The prompt/completion boundary and the loss mask are produced by
`render.render_segment`, which slices at real user-message boundaries (the exact
positions the template retains reasoning after) and masks by diffing consecutive
renders rather than assuming concatenation is stable.

Conversion is streaming: it never holds all 9k rendered trajectories in memory.
Samples are written incrementally to a jsonl shard per key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from smolqwen.data.loader import Trajectory
from smolqwen.data.render import RenderedSample, split_segments

SKIP_TOO_LONG = "too_long"


@dataclass(frozen=True)
class Converted:
    """A trajectory that produced at least one rendered sample within the cap."""

    trajectory_id: str
    env_id: str
    mode: str
    samples: tuple[RenderedSample, ...]


@dataclass(frozen=True)
class Skipped:
    """A trajectory the converter did not emit, with the reason."""

    trajectory_id: str
    reason: str


ConversionEvent = Converted | Skipped


def convert_trajectories(
    trajectories: Iterator[Trajectory],
    *,
    render: Callable[..., RenderedSample],
    max_seq_length: int,
    shape: str = "tool_role",
) -> Iterator[ConversionEvent]:
    """Yield one `Converted` (with its samples) or `Skipped` per trajectory.

    `render` is injected so tests can pass a real-tokenizer renderer or a stub
    without the converter knowing which. `max_seq_length` is the budget cap that
    Phase 2 measures; a trajectory whose full render exceeds it is skipped and
    counted, never silently truncated.
    """
    for trajectory in trajectories:
        try:
            segments = split_segments(trajectory.messages)
        except Exception as exc:
            yield Skipped(trajectory.trajectory_id, f"unsegmentable: {exc}")
            continue

        samples: list[RenderedSample] = []
        too_long = False
        for index, segment in enumerate(segments):
            sample = render(
                trajectory.messages,
                segment,
                tools=trajectory.tools,
                shape=shape,
                trajectory_id=trajectory.trajectory_id,
                env_id=trajectory.env_id,
                mode=trajectory.traj_type,
                segment_index=index,
            )
            if sample.total_tokens > max_seq_length:
                too_long = True
                break
            samples.append(sample)

        if too_long:
            yield Skipped(trajectory.trajectory_id, SKIP_TOO_LONG)
            continue
        if not samples:
            yield Skipped(trajectory.trajectory_id, "no_supervised_segment")
            continue
        yield Converted(
            trajectory_id=trajectory.trajectory_id,
            env_id=trajectory.env_id,
            mode=trajectory.traj_type,
            samples=tuple(samples),
        )


def sample_to_record(sample: RenderedSample) -> dict[str, Any]:
    """The persisted shape: token ids and the loss mask, keyed for training."""
    return {
        "trajectory_id": sample.trajectory_id,
        "env_id": sample.env_id,
        "mode": sample.mode,
        "segment_index": sample.segment_index,
        "prompt_ids": list(sample.prompt_ids),
        "completion_ids": list(sample.completion_ids),
        "loss_mask": list(sample.loss_mask),
        "total_tokens": sample.total_tokens,
        "supervised_tokens": sample.supervised_tokens,
    }


@dataclass
class ConversionReport:
    """Summary written to `conversion_report.json`. Bare counts, no best-estimate."""

    converted: int = 0
    skipped: int = 0
    samples: int = 0
    supervised_tokens: int = 0
    # mode -> (trajectories, samples). Modes are the release's own `traj_type`.
    by_mode: dict[str, tuple[int, int]] | None = None
    skip_reasons: dict[str, int] | None = None

    def note_converted(self, event: Converted) -> None:
        self.converted += 1
        self.samples += len(event.samples)
        if self.by_mode is None:
            self.by_mode = {}
        trajectories, samples = self.by_mode.get(event.mode, (0, 0))
        self.by_mode[event.mode] = (trajectories + 1, samples + len(event.samples))
        for sample in event.samples:
            self.supervised_tokens += sample.supervised_tokens

    def note_skipped(self, event: Skipped) -> None:
        self.skipped += 1
        if self.skip_reasons is None:
            self.skip_reasons = {}
        self.skip_reasons[event.reason] = self.skip_reasons.get(event.reason, 0) + 1

    def to_dict(
        self,
        *,
        input_shas: dict[str, str],
        input_revisions: dict[str, str] | None = None,
        load_stats: Any = None,
    ) -> dict[str, Any]:
        """The persisted report.

        `load_stats` closes the accounting: every input row is converted, skipped
        with a reason, or malformed. Without it a malformed row would vanish
        between the row count and the converted count.
        """
        payload: dict[str, Any] = {
            "converted": self.converted,
            "skipped": self.skipped,
            "samples": self.samples,
            "supervised_tokens": self.supervised_tokens,
            "by_mode": {
                mode: {"trajectories": t, "samples": s}
                for mode, (t, s) in (self.by_mode or {}).items()
            },
            "skip_reasons": self.skip_reasons or {},
            "input_shas": input_shas,
            "input_revisions": input_revisions or {},
        }
        if load_stats is not None:
            payload["rows"] = {
                "read": load_stats.total,
                "parsed": load_stats.parsed,
                "malformed": load_stats.malformed,
                "malformed_reasons": load_stats.malformed_reasons or {},
            }
            payload["accounted"] = (
                load_stats.total == self.converted + self.skipped + load_stats.malformed
            )
        return payload
