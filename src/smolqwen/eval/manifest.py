"""Comparable evaluation manifests: invariants protect scores, records describe runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestMismatchError(ValueError):
    """Raised when runs differ in an experiment-defining field."""


# These fields describe the execution that produced a row, rather than the
# experiment being compared.  Keeping them present (with ``None`` when a local
# runner cannot observe a serving setting) makes the JSON artifact self-
# describing and gives later serving re-evaluations a stable place to record
# their configuration.
RECORDED_FREE_FIELDS: tuple[str, ...] = (
    "backend",
    "checkpoint",
    "endpoint",
    "served_model",
    "dtype",
    "quantization",
    "speculative_decoding",
    "kv_budget",
    "max_num_seqs",
    "max_num_batched_tokens",
    "chunked_prefill",
    "prefix_caching",
    "library_versions",
    "checkpoint_revision",
    "adapter_revision",
)


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_json(value: object) -> str:
    """Stable hash helper shared by adapters without exposing their semantics."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash a potentially large benchmark input without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvalManifest:
    invariant: Mapping[str, Any]
    recorded_free: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Normalize at construction so callers inspecting the object and callers
        # serializing it see the same complete recorded-free contract.
        normalized = {field: None for field in RECORDED_FREE_FIELDS}
        normalized.update(dict(self.recorded_free))
        object.__setattr__(self, "invariant", dict(self.invariant))
        object.__setattr__(self, "recorded_free", normalized)

    @property
    def invariant_hash(self) -> str:
        return hashlib.sha256(_canonical(self.invariant).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant": dict(self.invariant),
            "recorded_free": dict(self.recorded_free),
            "invariant_hash": self.invariant_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvalManifest:
        """Rehydrate a manifest stored in a report artifact."""

        invariant = payload.get("invariant")
        recorded_free = payload.get("recorded_free")
        if not isinstance(invariant, Mapping) or not isinstance(recorded_free, Mapping):
            raise ValueError("evaluation report has a malformed manifest")
        return cls(dict(invariant), dict(recorded_free))


def assert_comparable(*manifests: EvalManifest) -> None:
    if len(manifests) < 2:
        return
    reference = manifests[0].invariant
    for index, manifest in enumerate(manifests[1:], start=1):
        keys = sorted(set(reference) | set(manifest.invariant))
        differences = [
            f"{key}: {reference.get(key)!r} != {manifest.invariant.get(key)!r}"
            for key in keys
            if reference.get(key) != manifest.invariant.get(key)
        ]
        if differences:
            raise ManifestMismatchError(
                f"manifest 0 differs from manifest {index}: " + "; ".join(differences)
            )
