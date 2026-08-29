"""Seeded train/val split by trajectory id, and the environment-split manifest.

The split is **by trajectory id, never by sample**: a Conv trajectory's segments
must not straddle the split or the validation set leaks. `val_fraction` of
trajectories are held out deterministically from `split_seed`.

The environment manifest is derived from `191_env_metadata.json` -- 140 `_sft` /
51 `_rl` by `env_id` suffix, 0 other -- and asserts that every RL scenario's
`env_id` (from the RL scenario metadata) is in the RL set. The dataset card's
"141 / 50" is a description error and is deliberately not carried in.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.data.loader import iter_json_object_values, sha256_of


@dataclass(frozen=True)
class Split:
    train_ids: frozenset[str]
    val_ids: frozenset[str]

    def partition(self, trajectory_id: str) -> str:
        return "val" if trajectory_id in self.val_ids else "train"


def split_trajectory_ids(ids: Sequence[str], *, seed: int, val_fraction: float) -> Split:
    """A deterministic, seed-reproducible split by trajectory id."""
    rng = random.Random(seed)
    ordered = sorted(set(ids))
    n_val = max(1, round(len(ordered) * val_fraction)) if val_fraction > 0 else 0
    val_ids: set[str] = set()
    if n_val:
        # `sample` is stable for a given seed and population order.
        val_ids = set(rng.sample(ordered, n_val))
    return Split(
        train_ids=frozenset(ordered) - val_ids,
        val_ids=frozenset(val_ids),
    )


def split_trajectories(trajectories: Iterable[str], *, seed: int, val_fraction: float) -> Split:
    """Split trajectories by id, provided as an iterable of trajectory id strings."""
    return split_trajectory_ids(list(trajectories), seed=seed, val_fraction=val_fraction)


@dataclass(frozen=True)
class EnvSplitManifest:
    sft_env_ids: tuple[str, ...]
    rl_env_ids: tuple[str, ...]
    other_env_ids: tuple[str, ...]
    all_env_ids: tuple[str, ...]
    # How many RL scenarios (not envs) reference each category.
    other_env_count: int = 0
    # Env-id suffixes that are neither `_sft` nor `_rl`, if any.
    input_sha256: str | None = None
    input_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sft_env_ids": list(self.sft_env_ids),
            "rl_env_ids": list(self.rl_env_ids),
            "other_env_ids": list(self.other_env_ids),
            "counts": {
                "total": len(self.all_env_ids),
                "sft": len(self.sft_env_ids),
                "rl": len(self.rl_env_ids),
                "other": len(self.other_env_ids),
            },
            "rl_scenario_env_id_violations": self.other_env_count,
            "input_sha256": self.input_sha256,
            "input_revision": self.input_revision,
        }


def build_env_split_manifest(
    env_metadata_path: Path | str,
    *,
    rl_scenario_env_ids: Iterable[str] | None = None,
    input_sha256: str | None = None,
    input_revision: str | None = None,
) -> EnvSplitManifest:
    """Derive the SFT/RL env split from `191_env_metadata.json`.

    The split is read from the release data by `env_id` suffix, never from the
    dataset card. `rl_scenario_env_ids`, if supplied, is checked against the RL
    set: any scenario env id that is not `_rl` is a data-lineage violation the
    manifest records as `rl_scenario_env_id_violations`.
    """
    all_ids = [eid for eid, _ in iter_json_object_values(env_metadata_path)]
    sft = tuple(sorted(eid for eid in all_ids if eid.endswith("_sft")))
    rl = tuple(sorted(eid for eid in all_ids if eid.endswith("_rl")))
    other = tuple(
        sorted(eid for eid in all_ids if not (eid.endswith("_sft") or eid.endswith("_rl")))
    )
    rl_set = set(rl)

    violations = 0
    if rl_scenario_env_ids is not None:
        for scenario_env in rl_scenario_env_ids:
            if scenario_env not in rl_set:
                violations += 1

    return EnvSplitManifest(
        sft_env_ids=sft,
        rl_env_ids=rl,
        other_env_ids=other,
        all_env_ids=tuple(sorted(all_ids)),
        other_env_count=violations,
        input_sha256=sha256_of(env_metadata_path) if input_sha256 is None else input_sha256,
        input_revision=input_revision,
    )
