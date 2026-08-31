"""Stream one released teacher trajectory into one full-trajectory SFT record."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from smolqwen.data.loader import Trajectory
from smolqwen.data.render import RenderedSample

SKIP_TOO_LONG = "too_long"
SFT_SCHEMA_VERSION = 2
SFT_SEMANTICS = "full_trajectory_reasoning_v1"


@dataclass(frozen=True)
class Converted:
    """A trajectory that produced its single rendered sample within the cap."""

    trajectory_uid: str
    task_id: str
    env_id: str
    mode: str
    sample: RenderedSample


@dataclass(frozen=True)
class Skipped:
    """A trajectory the converter did not emit, with the reason."""

    trajectory_uid: str
    task_id: str
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
            sample = render(
                trajectory.messages,
                tools=trajectory.tools,
                shape=shape,
                trajectory_uid=trajectory.trajectory_uid,
                task_id=trajectory.task_id,
                env_id=trajectory.env_id,
                mode=trajectory.traj_type,
            )
        except Exception as exc:
            yield Skipped(
                trajectory_uid=trajectory.trajectory_uid,
                task_id=trajectory.task_id,
                reason=f"unrenderable: {exc}",
            )
            continue
        if sample.total_tokens > max_seq_length:
            yield Skipped(trajectory.trajectory_uid, trajectory.task_id, SKIP_TOO_LONG)
            continue
        yield Converted(
            trajectory_uid=trajectory.trajectory_uid,
            task_id=trajectory.task_id,
            env_id=trajectory.env_id,
            mode=trajectory.traj_type,
            sample=sample,
        )


def sample_to_record(sample: RenderedSample) -> dict[str, Any]:
    """Versioned persisted shape consumed by the full-trajectory trainer."""
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": sample.trajectory_uid,
        "task_id": sample.task_id,
        "env_id": sample.env_id,
        "mode": sample.mode,
        "input_ids": list(sample.input_ids),
        "labels": list(sample.labels),
        "seq_length": sample.total_tokens,
        "supervised_tokens": sample.supervised_tokens,
        "template_fingerprint": sample.template_fingerprint,
        "trailing_messages_removed": sample.trailing_messages_removed,
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
        self.samples += 1
        if self.by_mode is None:
            self.by_mode = {}
        trajectories, samples = self.by_mode.get(event.mode, (0, 0))
        self.by_mode[event.mode] = (trajectories + 1, samples + 1)
        self.supervised_tokens += event.sample.supervised_tokens

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
