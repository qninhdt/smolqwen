"""Load the RL scenarios, filter them to the RL environment split, sample them.

2,550 scenarios across the 51 `_rl` environments. Two things this module is
careful about:

**The env split is read from the Phase 2 manifest, not re-derived.** The manifest
already recorded 140 `_sft` / 51 `_rl` / 0 other from `env_id` suffixes, plus zero
`rl_scenario_env_id_violations`. Re-deriving it here would be a second source of
truth for which environments RL is allowed to touch, and the two could disagree
after a data refresh with nothing reporting it.

**Scenarios are held as raw dicts until a worker wants one.** The `checklist_with_func`
sources are dataset text destined for `exec()`, so the parent holds strings and the
worker compiles. Filtering and sampling therefore operate on metadata only.

Sampling is seeded and returns scenario ids rather than payloads, so a difficulty
profile taken in Phase 7 can be reproduced without carrying 40,231 check sources
through it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.data.loader import sha256_of, verify_sha256


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


def load_rl_env_ids(manifest_path: Path | str) -> frozenset[str]:
    """The RL environment ids from the Phase 2 `env_split.json`.

    Read rather than re-derived: the manifest is the recorded decision about which
    environments RL may touch, and it also carries the violation count that says
    the two files agreed when it was written.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise ScenarioError(f"{path} missing -- run `smolqwen profile-data` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "rl_env_ids" not in payload:
        raise ScenarioError(f"{path}: not an env-split manifest (no rl_env_ids)")
    return frozenset(str(env_id) for env_id in payload["rl_env_ids"])


def filter_to_env_ids(
    scenarios: Iterable[Scenario], env_ids: frozenset[str]
) -> tuple[list[Scenario], list[str]]:
    """Split scenarios into (kept, rejected task ids) against an allowed env set.

    Rejections are returned rather than logged and dropped: a scenario whose
    `env_id` is outside the RL split is a data-lineage problem, and the count
    belongs in the run's manifest next to the sha256s.
    """
    kept: list[Scenario] = []
    rejected: list[str] = []
    for scenario in scenarios:
        if scenario.env_id in env_ids:
            kept.append(scenario)
        else:
            rejected.append(scenario.task_id)
    return kept, rejected


@dataclass(frozen=True)
class ScenarioSet:
    """The scenarios one RL run may sample from, plus what was excluded and why."""

    scenarios: tuple[Scenario, ...]
    rejected_task_ids: tuple[str, ...]
    source_path: str
    source_sha256: str

    def __len__(self) -> int:
        return len(self.scenarios)

    def by_id(self) -> dict[str, Scenario]:
        return {scenario.task_id: scenario for scenario in self.scenarios}

    def env_ids(self) -> frozenset[str]:
        return frozenset(scenario.env_id for scenario in self.scenarios)

    def task_ids(self) -> tuple[str, ...]:
        return tuple(scenario.task_id for scenario in self.scenarios)

    def sample_ids(self, count: int, *, seed: int) -> tuple[str, ...]:
        """A seeded sample of task ids, stable for a given `(count, seed)`.

        Sorted before sampling so the result does not depend on file order, which
        is what makes a Phase 7 difficulty profile reproducible.
        """
        ordered = sorted(self.task_ids())
        if count >= len(ordered):
            return tuple(ordered)
        return tuple(random.Random(seed).sample(ordered, count))

    def manifest(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "scenarios": len(self.scenarios),
            "environments": len(self.env_ids()),
            "rejected_outside_rl_split": len(self.rejected_task_ids),
            "check_count": {
                "total": sum(scenario.check_count for scenario in self.scenarios),
                "min": min((s.check_count for s in self.scenarios), default=0),
                "max": max((s.check_count for s in self.scenarios), default=0),
            },
        }


def build_scenario_set(
    scenario_path: Path | str,
    *,
    env_split_manifest: Path | str,
    sha256: str | None = None,
) -> ScenarioSet:
    """Load the RL scenarios and keep only those inside the Phase 2 RL env split."""
    scenarios = load_scenarios(scenario_path, sha256=sha256)
    kept, rejected = filter_to_env_ids(scenarios, load_rl_env_ids(env_split_manifest))
    if not kept:
        raise ScenarioError(
            f"no scenario survived the RL env split; {len(scenarios)} loaded from {scenario_path}"
        )
    return ScenarioSet(
        scenarios=tuple(kept),
        rejected_task_ids=tuple(rejected),
        source_path=str(scenario_path),
        source_sha256=sha256 or sha256_of(scenario_path),
    )


def iter_by_env(scenarios: Sequence[Scenario]) -> Iterator[tuple[str, tuple[Scenario, ...]]]:
    """Group scenarios by `env_id`, sorted, so a per-environment pass is stable."""
    grouped: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.env_id, []).append(scenario)
    for env_id in sorted(grouped):
        yield env_id, tuple(grouped[env_id])
