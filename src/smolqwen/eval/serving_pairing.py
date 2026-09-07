"""Pair a quality score with the serving config it was measured under.

`assert_comparable` (`manifest.py`) compares the **invariant** set — decoding, step
limits, benchmark revision — which is exactly right for putting Base, SFT and
SFT+RL in one table. It is not sufficient for a speed/quality row: a score measured
at `max_num_seqs: 8` and a throughput row measured at 128 have identical invariants
and are still not the same experiment. `recorded_free` is where the serving config
lives, so that is what a paired row must compare.

This lives beside the manifest it reads rather than beside the benchmark wrapper it
was written for, because the wrapper's job now belongs to the upstream
`vllm bench serve` commands and the guard's does not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.eval.manifest import EvalManifest, assert_comparable

# Every serving field a quality score is only valid under. Comparing fewer would
# let one of them differ silently; the whole point is that a paired row is one
# experiment, not two.
PAIRED_SERVING_FIELDS = (
    "dtype",
    "quantization",
    "speculative_decoding",
    "kv_budget",
    "max_num_seqs",
    "max_num_batched_tokens",
    "chunked_prefill",
    "prefix_caching",
)


class ServingPairingError(ValueError):
    """Raised when a quality score was measured under a different serving config."""


@dataclass(frozen=True)
class QualityResult:
    """One evaluation report reduced to what a paired row needs."""

    score: float
    manifest: EvalManifest


def _normalized(value: object) -> object:
    """JSON-decode a stringified value so `"null"` and `None` compare equal.

    Serving config crosses a CLI boundary as text and a manifest as JSON, so the
    same fact arrives in two shapes. Comparing them raw reports a mismatch that is
    only a serialization difference.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def assert_quality_matches_serving(
    observed: Mapping[str, Any], quality: QualityResult, *, label: str = "row"
) -> None:
    """Refuse to attach a quality score measured under another serving config.

    `observed` is what the throughput measurement recorded. Only fields present
    there are compared: a benchmark that did not record `quantization` makes no
    claim about it, and inventing `None` on its behalf would fail every paired row
    against a quantized score.
    """
    recorded = quality.manifest.recorded_free
    differences = [
        f"{field}: measured={_normalized(observed[field])!r}, "
        f"quality={_normalized(recorded.get(field))!r}"
        for field in PAIRED_SERVING_FIELDS
        if field in observed and _normalized(observed[field]) != _normalized(recorded.get(field))
    ]
    if differences:
        raise ServingPairingError(
            f"quality report does not match {label}: " + "; ".join(differences)
        )


def load_quality_result(
    path: Path | str, *, references: tuple[Path, ...] | list[Path] = ()
) -> QualityResult:
    """Read one evaluation report, refusing an invariant mismatch against references."""
    score, manifest = _read_report(path)
    reference_manifests = [_read_report(reference)[1] for reference in references]
    assert_comparable(*reference_manifests, manifest)
    return QualityResult(score=score, manifest=manifest)


def _read_report(path: Path | str) -> tuple[float, EvalManifest]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"quality report must be an object: {path}")
    manifest_payload = payload.get("manifest")
    metrics = payload.get("metrics")
    if not isinstance(manifest_payload, dict) or not isinstance(metrics, dict):
        raise ValueError(f"quality report is malformed: {path}")
    overall = metrics.get("multi_turn_overall") or {}
    score = overall.get("score")
    if score is None:
        raise ValueError(
            f"quality report has no multi_turn_overall score: {path}; a paired row "
            "needs one headline number, not a per-category table"
        )
    return float(score), EvalManifest.from_dict(manifest_payload)
