"""Verifier-only difficulty profiling and curriculum weights."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DifficultyBand = Literal["always_zero", "band", "always_one"]
SCHEMA_VERSION = 1


class DifficultyError(ValueError):
    """Raised when a profile is incomplete, malformed, or from another model."""


@dataclass(frozen=True)
class ScenarioDifficulty:
    task_id: str
    rollouts: int
    successes: int
    success_rate: float
    mean_reward: float
    band: DifficultyBand

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollouts": self.rollouts,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "band": self.band,
        }


@dataclass(frozen=True)
class DifficultyProfile:
    model_id: str
    model_revision: str | None
    seed: int
    scenarios: tuple[ScenarioDifficulty, ...]

    @property
    def by_task(self) -> dict[str, ScenarioDifficulty]:
        return {entry.task_id: entry for entry in self.scenarios}

    def to_dict(self) -> dict[str, Any]:
        counts = {band: 0 for band in ("always_zero", "band", "always_one")}
        for entry in self.scenarios:
            counts[entry.band] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {"id": self.model_id, "revision": self.model_revision},
            "seed": self.seed,
            "counts": counts,
            "scenarios": [entry.to_dict() for entry in self.scenarios],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DifficultyProfile:
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise DifficultyError("unsupported difficulty profile schema")
        model = payload.get("model")
        rows = payload.get("scenarios")
        if not isinstance(model, Mapping) or not isinstance(rows, list):
            raise DifficultyError("malformed difficulty profile")
        scenarios = tuple(
            ScenarioDifficulty(
                task_id=str(row["task_id"]),
                rollouts=int(row["rollouts"]),
                successes=int(row["successes"]),
                success_rate=float(row["success_rate"]),
                mean_reward=float(row["mean_reward"]),
                band=str(row["band"]),  # type: ignore[arg-type]
            )
            for row in rows
        )
        if any(entry.band not in ("always_zero", "band", "always_one") for entry in scenarios):
            raise DifficultyError("difficulty profile contains an unknown band")
        return cls(
            model_id=str(model["id"]),
            model_revision=(str(model["revision"]) if model.get("revision") is not None else None),
            seed=int(payload.get("seed", 0)),
            scenarios=scenarios,
        )


def profile_rewards(
    rewards_by_task: Mapping[str, Sequence[float]],
    *,
    model_id: str,
    model_revision: str | None,
    seed: int,
) -> DifficultyProfile:
    """Classify from verifier rewards; success means the full checklist passed."""
    rows: list[ScenarioDifficulty] = []
    for task_id, raw_rewards in sorted(rewards_by_task.items()):
        rewards = [float(value) for value in raw_rewards]
        if not rewards:
            raise DifficultyError(f"{task_id}: no profiling rollouts")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rewards):
            raise DifficultyError(f"{task_id}: reward outside [0, 1]")
        successes = sum(math.isclose(value, 1.0) for value in rewards)
        rate = successes / len(rewards)
        band: DifficultyBand
        if successes == 0:
            band = "always_zero"
        elif successes == len(rewards):
            band = "always_one"
        else:
            band = "band"
        rows.append(
            ScenarioDifficulty(
                task_id=task_id,
                rollouts=len(rewards),
                successes=successes,
                success_rate=rate,
                mean_reward=math.fsum(rewards) / len(rewards),
                band=band,
            )
        )
    return DifficultyProfile(model_id, model_revision, seed, tuple(rows))


def write_profile(profile: DifficultyProfile, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def read_profile(
    path: Path | str, *, model_id: str | None = None, model_revision: str | None = None
) -> DifficultyProfile:
    target = Path(path)
    if not target.is_file():
        raise DifficultyError(f"{target} missing -- run `smolqwen profile-difficulty` first")
    profile = DifficultyProfile.from_dict(json.loads(target.read_text(encoding="utf-8")))
    if model_id is not None and profile.model_id != model_id:
        raise DifficultyError(
            f"difficulty profile model {profile.model_id!r} != configured model {model_id!r}"
        )
    if model_revision is not None and profile.model_revision != model_revision:
        raise DifficultyError(
            "difficulty profile revision does not match the configured checkpoint"
        )
    return profile


def weighted_scenario_order(
    task_ids: Sequence[str],
    profile: DifficultyProfile | None,
    *,
    seed: int,
    band_weight: float,
    always_zero_weight: float,
    always_one_weight: float,
) -> list[str]:
    """Deterministic weighted permutation, without duplicating dataset rows."""
    rng = random.Random(seed)
    by_task = profile.by_task if profile is not None else {}
    weights = {
        "band": band_weight,
        "always_zero": always_zero_weight,
        "always_one": always_one_weight,
    }
    ranked: list[tuple[float, str]] = []
    for task_id in task_ids:
        entry = by_task.get(task_id)
        weight = weights[entry.band] if entry is not None else 1.0
        if weight <= 0.0:
            continue
        ranked.append((-math.log(max(rng.random(), 1e-12)) / weight, task_id))
    return [task_id for _, task_id in sorted(ranked)]
