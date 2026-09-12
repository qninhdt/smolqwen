"""Comparable evaluation manifests: invariants protect scores, records describe runs."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


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

REDACTED_HOST = "redacted"


def redact_endpoint(endpoint: object) -> object:
    """Reduce an endpoint to scheme, host *shape*, port and path. Never routable.

    A public hostname recorded verbatim can publish a routable ingress to a GPU
    box, and a credentialed URL can expose userinfo. Both are stripped here.

    Loopback and private addresses survive intact: they are not reachable from
    outside the host, and "this was measured against the local proxy on 8080" is
    real provenance a reader needs. Everything else keeps its shape and loses its
    name, which is what makes two rows comparable ("both went through a tunnel")
    without either being usable.

    Idempotent, so a manifest rehydrated from a report is unchanged.
    """
    if not isinstance(endpoint, str) or not endpoint:
        return endpoint
    parts = urlsplit(endpoint)
    if not parts.scheme or not parts.netloc:
        # Not a URL this function can reason about (no scheme, or a bare
        # `host:port`), so nothing here can be asserted to be non-routable.
        return REDACTED_HOST
    host = parts.hostname or ""
    if _is_local(host):
        # `hostname` is already lowercased and userinfo-free; rebuilt rather than
        # passed through so a `user:pw@` prefix cannot survive on this branch.
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}{parts.path}"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{REDACTED_HOST}{port}{parts.path}"


def _is_local(host: str) -> bool:
    if host in {"localhost", REDACTED_HOST}:
        return host == "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_loopback or address.is_private or address.is_link_local)


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_json(value: object) -> str:
    """Stable hash helper shared by adapters without exposing their semantics."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EvalManifest:
    invariant: Mapping[str, Any]
    recorded_free: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Normalize at construction so callers inspecting the object and callers
        # serializing it see the same complete recorded-free contract.
        normalized: dict[str, Any] = {field: None for field in RECORDED_FREE_FIELDS}
        normalized.update(dict(self.recorded_free))
        # Redaction happens here rather than at the call site because every path
        # that produces a report -- the runner, a rehydrated report, a test -- goes
        # through this constructor. A redaction one caller can forget is one an
        # upload will eventually publish.
        normalized["endpoint"] = redact_endpoint(normalized["endpoint"])
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
