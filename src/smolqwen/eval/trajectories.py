"""Per-task trajectory records: the evidence a score is a projection of.

Evaluation built a full message history per task and then discarded it -- only the
aggregate reached the report. The rollout path already keeps the whole record
(`Episode.to_row()`), so evaluation was the only side throwing it away.

That matters most for the benchmark this project reports. BFCL is all-or-nothing
over four conditions, so a `0.0` with no trajectory cannot be attributed to any of
them and cannot be re-graded. The reproducibility literature measures how large
that gap gets: re-grading fixed artifacts has moved scores by up to 20.9pp, and
τ-bench shifted 16.9pp with 12 ordering changes among frontier models. Diagnostics
make a failure legible; the trajectory makes it checkable.

One record per task, JSONL, written beside the report. `artifacts/evaluation/` is
gitignored, so records stay local until an upload phase moves them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryRecord:
    """Everything needed to re-derive one task's score without generating again."""

    task_id: str
    category: str
    messages: list[Mapping[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    score: float | None = None
    exact_success: bool | None = None
    completed: bool | None = None
    # Which scoring condition failed, in the benchmark's own words. The reason a
    # bare 0.0 is attributable at all.
    failure_reason: str | None = None
    failed_check_names: list[str] = field(default_factory=list)
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    terminal_reason: str | None = None
    generation_turns: int = 0
    env_steps: int = 0
    generated_tokens: int = 0
    truncated: bool = False
    invalid_calls: int = 0
    wall_s: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "messages": [dict(message) for message in self.messages],
            "observations": list(self.observations),
            "score": self.score,
            "exact_success": self.exact_success,
            "completed": self.completed,
            "failure_reason": self.failure_reason,
            "failed_check_names": list(self.failed_check_names),
            "diagnostics": dict(self.diagnostics),
            "terminal_reason": self.terminal_reason,
            "generation_turns": self.generation_turns,
            "env_steps": self.env_steps,
            "generated_tokens": self.generated_tokens,
            "truncated": self.truncated,
            "invalid_calls": self.invalid_calls,
            "wall_s": round(self.wall_s, 4),
        }


def trajectory_path(output_dir: Path | str, *, tag: str, adapter: str) -> Path:
    """Where one adapter's records for one run tag live."""
    return Path(output_dir) / "trajectories" / f"{tag}-{adapter}.jsonl"


def write_trajectories(
    output_dir: Path | str,
    *,
    tag: str,
    adapter: str,
    records: Sequence[TrajectoryRecord],
) -> Path:
    """Write one JSONL row per task. Overwrites, because a run is a whole set.

    Appending would silently interleave two runs of the same tag, and the manifest
    that certifies comparability is per run -- a half-old, half-new file would be
    certified by neither.
    """
    path = trajectory_path(output_dir, tag=tag, adapter=adapter)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_row(), sort_keys=True) + "\n")
    return path


def read_trajectories(path: Path | str) -> list[dict[str, Any]]:
    """Read a record file back, for a re-grade or a disagreement comparison."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
