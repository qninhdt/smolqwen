"""Load the pinned RL scenarios without compiling executable checklist code."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.data.loader import verify_sha256


class ScenarioError(Exception):
    """Raised when scenario metadata cannot be loaded or is inconsistent."""


@dataclass(frozen=True)
class Scenario:
    """One RL scenario. `checklist` stays raw: it is `exec()` fuel for a worker."""

    task_id: str
    env_id: str
    env_class_name: str
    task: str
    init_config: Mapping[str, Any]
    checklist: tuple[Mapping[str, Any], ...]

    @property
    def check_count(self) -> int:
        """K, the reward denominator. Ranges 2 to 445 across the release."""
        return len(self.checklist)


def parse_scenario(payload: Any) -> Scenario:
    if not isinstance(payload, dict):
        raise ScenarioError("scenario entry is not an object")
    try:
        checklist = payload["checklist_with_func"]
        scenario = Scenario(
            task_id=str(payload["task_id"]),
            env_id=str(payload["env_id"]),
            env_class_name=str(payload["env_class_name"]),
            task=str(payload.get("task", "")),
            init_config=payload.get("init_config") or {},
            checklist=tuple(checklist),
        )
    except KeyError as exc:
        raise ScenarioError(f"scenario entry is missing {exc}") from exc
    if not scenario.checklist:
        raise ScenarioError(f"{scenario.task_id}: empty checklist; reward would be undefined")
    return scenario


def load_scenarios(path: Path | str, *, sha256: str | None = None) -> list[Scenario]:
    """Read every RL scenario, verifying the file's sha256 first."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ScenarioError(f"RL scenario metadata not found: {file_path}")
    verify_sha256(file_path, sha256)

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ScenarioError(f"{file_path}: expected a JSON array of scenarios")
    return [parse_scenario(entry) for entry in payload]
