"""Readable rollout columns for TRL's W&B/parquet completion table."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def trajectory_columns(
    trajectories: Sequence[Mapping[str, Any]], *, sample_limit: int
) -> dict[str, list[Any]]:
    """Flatten episode histories without changing their reward semantics.

    TRL requires one value per completion for every extra table column. Rows
    beyond `sample_limit` retain identifiers and a `sampled=False` marker but
    omit the bulky history, keeping overnight W&B tables bounded.
    """
    columns: dict[str, list[Any]] = {
        "trajectory_sampled": [],
        "scenario_id": [],
        "terminal_reason": [],
        "reasoning": [],
        "tool_calls": [],
        "observations": [],
        "checkpoint_verdicts": [],
        "invalid_call_count": [],
        "step_count": [],
    }
    for index, trajectory in enumerate(trajectories):
        sampled = index < sample_limit
        messages = trajectory.get("messages") or []
        reasoning: list[str] = []
        calls: list[str] = []
        if sampled:
            for message in messages:
                if not isinstance(message, Mapping) or message.get("role") != "assistant":
                    continue
                thought = message.get("reasoning_content")
                if thought:
                    reasoning.append(str(thought))
                content = str(message.get("content") or "")
                if "<tool_call>" in content:
                    calls.append(content)
        columns["trajectory_sampled"].append(sampled)
        columns["scenario_id"].append(str(trajectory.get("scenario_id", "")))
        columns["terminal_reason"].append(str(trajectory.get("terminal_reason") or ""))
        columns["reasoning"].append("\n\n".join(reasoning) if sampled else "")
        columns["tool_calls"].append("\n".join(calls) if sampled else "")
        columns["observations"].append(
            json.dumps(trajectory.get("observations") or [], ensure_ascii=False) if sampled else ""
        )
        columns["checkpoint_verdicts"].append(
            json.dumps(trajectory.get("per_check_bools") or []) if sampled else ""
        )
        columns["invalid_call_count"].append(int(trajectory.get("invalid_call_count", 0)))
        columns["step_count"].append(int(trajectory.get("step_count", 0)))
    return columns


def log_trajectory_columns(
    log_extra: Callable[[str, list[Any]], Any] | None,
    trajectories: Sequence[Mapping[str, Any]],
    *,
    sample_limit: int,
) -> None:
    """Emit columns through TRL's reward-function logging seam."""
    if log_extra is None:
        return
    for name, values in trajectory_columns(trajectories, sample_limit=sample_limit).items():
        log_extra(name, values)
