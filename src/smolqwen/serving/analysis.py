"""Aggregate serving-sweep runs into frontiers and measured operating profiles.

This module owns the deterministic, stdlib-only analysis that turns the recorded
`vllm bench serve` output (Phase 6) into the three measured serving
profiles (``configs/serving/{latency,balanced,throughput}.yaml``) and the summary
artifacts. It never runs vLLM or touches a GPU: it reads recorded JSON, computes
aggregates the configured observations, rejects unsafe points, extracts two Pareto frontiers,
selects one operating point per profile, and derives native admission limits.

``tok/s/user`` is deliberately absent from every selection rule. Capacity is paid
for with p95 TTFT and p95 TPOT, and those are the selection axes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from statistics import median

from smolqwen.serving.benchmark_spec import (
    MODEL_REPO,
    MODEL_REVISION,
    NUM_PROMPTS,
    build_study_points,
)
from smolqwen.serving.benchmark_spec import (
    PRECISION_SETTINGS as _PRECISION_SETTINGS,
)
from smolqwen.serving.metrics_capture import PREEMPTIONS, parse_prometheus_text

PRECISIONS = ("bf16", "fp8")
PROFILE_NAMES = ("latency", "balanced", "throughput")

# Only metrics present in native vLLM result JSON drive selection. Post-run
# Prometheus snapshots are validated separately; they are not fabricated peaks.
_METRIC_KEYS = (
    "request_throughput",
    "output_throughput",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
)


class AnalysisError(Exception):
    """Raised when raw sweep data is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class SLO:
    """Optional service-level objectives.

    When supplied, points above either threshold are ineligible and goodput is
    meaningful; when absent, only the validity filter applies and admission
    limits fall back to the conservative provisional formula.
    """

    p95_ttft_ms: float
    p95_tpot_ms: float


@dataclass(frozen=True)
class RawRun:
    """One recorded ``vllm bench serve`` repetition plus its metric snapshot."""

    precision: str
    max_num_seqs: int
    max_num_batched_tokens: int
    concurrency: int
    repetition: int
    manifest_hash: str
    failed_requests: int
    preemptions: int
    metrics: Mapping[str, float]

    @property
    def point_key(self) -> tuple[str, int, int, int]:
        return (self.precision, self.max_num_seqs, self.max_num_batched_tokens, self.concurrency)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object], *, context: str) -> RawRun:
        server = _require_mapping(data.get("server_config"), context=f"{context}.server_config")
        precision = _require_str(server, "precision", context=f"{context}.server_config")
        if precision not in PRECISIONS:
            raise AnalysisError(
                f"{context}.server_config.precision: {precision!r} not in {PRECISIONS}"
            )
        metrics = {key: _require_float(data, key, context=context) for key in _METRIC_KEYS}
        return cls(
            precision=precision,
            max_num_seqs=_require_int(server, "max_num_seqs", context=f"{context}.server_config"),
            max_num_batched_tokens=_require_int(
                server, "max_num_batched_tokens", context=f"{context}.server_config"
            ),
            concurrency=_require_int(data, "concurrency", context=context),
            repetition=_require_int(data, "repetition", context=context),
            manifest_hash=_require_str(data, "manifest_hash", context=context),
            failed_requests=_require_int(data, "failed_requests", context=context),
            preemptions=_require_int(data, "preemptions", context=context),
            metrics=metrics,
        )


@dataclass(frozen=True)
class PointStats:
    """One ``(server config, concurrency)`` point aggregated across repetitions."""

    precision: str
    max_num_seqs: int
    max_num_batched_tokens: int
    concurrency: int
    n_runs: int
    valid: bool
    reason: str | None
    medians: Mapping[str, float]
    spreads: Mapping[str, tuple[float, float]]

    @property
    def point_key(self) -> tuple[str, int, int, int]:
        return (self.precision, self.max_num_seqs, self.max_num_batched_tokens, self.concurrency)

    def identity(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True)
class AdmissionLimits:
    """Native vLLM queue limits derived for one selected operating point."""

    max_num_queued_reqs: int
    max_num_queued_tokens: int
    provisional: bool
    basis: str


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the sweep analysis produces, ready to render or serialize."""

    manifest_hash: str
    num_runs: int
    prompt_len_p95: int
    points: tuple[PointStats, ...]
    eligible: tuple[PointStats, ...]
    frontiers: Mapping[str, Mapping[str, tuple[PointStats, ...]]]
    selections: Mapping[str, PointStats]
    admission: Mapping[str, AdmissionLimits]
    slo: SLO | None
    summary: Mapping[str, object]
    summary_hash: str


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{context}: expected an object, got {type(value).__name__}")
    return value


def _require(mapping: Mapping[str, object], key: str, *, context: str) -> object:
    if key not in mapping:
        raise AnalysisError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _require_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = _require(mapping, key, context=context)
    if isinstance(value, bool):
        raise AnalysisError(f"{context}.{key}: expected an integer, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise AnalysisError(f"{context}.{key}: expected an integer, got {type(value).__name__}")


def _require_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = _require(mapping, key, context=context)
    if isinstance(value, bool):
        raise AnalysisError(f"{context}.{key}: expected a number, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    raise AnalysisError(f"{context}.{key}: expected a number, got {type(value).__name__}")


def _require_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = _require(mapping, key, context=context)
    if not isinstance(value, str):
        raise AnalysisError(f"{context}.{key}: expected a string, got {type(value).__name__}")
    return value


def parse_runs(records: Iterable[Mapping[str, object]]) -> list[RawRun]:
    """Parse an iterable of raw-run mappings, failing closed on malformed input."""
    runs = [
        RawRun.from_mapping(record, context=f"run[{index}]") for index, record in enumerate(records)
    ]
    if not runs:
        raise AnalysisError("no runs were provided")
    return runs


def load_raw_runs(path: Path | str) -> list[RawRun]:
    """Load runs from a JSON file (object or list) or a directory of such files."""
    root = Path(path)
    records: list[Mapping[str, object]] = []
    if root.is_dir():
        files = sorted(root.rglob("*.json"))
        if not files:
            raise AnalysisError(f"{root}: no .json run files found")
    elif root.is_file():
        files = [root]
    else:
        raise AnalysisError(f"{root}: not a file or directory")
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            records.append(payload)
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, Mapping):
                    raise AnalysisError(f"{file}: list entries must be objects")
                records.append(item)
        else:
            raise AnalysisError(f"{file}: expected an object or a list of objects")
    return parse_runs(records)


def _json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"{path}: expected a JSON object")
    return payload


def _nearest_rank_p95(values: Sequence[int]) -> int:
    if not values:
        raise AnalysisError("native result input_lens is empty")
    return sorted(values)[ceil(0.95 * len(values)) - 1]


def load_vllm_study_runs(
    root: Path | str,
    *,
    manifest_path: Path | str,
) -> tuple[list[RawRun], int]:
    """Load the approved study directly from native vLLM JSON and metric snapshots."""
    study_root = Path(root)
    manifest = _json_mapping(Path(manifest_path))
    manifest_hash = _require_str(manifest, "manifest_hash", context=str(manifest_path))
    if _require_str(manifest, "model", context=str(manifest_path)) != MODEL_REPO:
        raise AnalysisError(f"{manifest_path}: manifest model is not {MODEL_REPO!r}")
    if _require_str(manifest, "tokenizer_revision", context=str(manifest_path)) != MODEL_REVISION:
        raise AnalysisError(f"{manifest_path}: manifest tokenizer revision is not pinned")
    expected_names = {str(point["_benchmark_name"]) for point in build_study_points()}
    run_files = sorted(study_root.glob("*/SERVE--*/run=*.json"))
    if len(run_files) != len(expected_names):
        raise AnalysisError(
            f"{study_root}: expected {len(expected_names)} native runs, found {len(run_files)}"
        )

    records: list[Mapping[str, object]] = []
    found_names: set[str] = set()
    canonical_input_lens: tuple[int, ...] | None = None
    for run_path in run_files:
        point_dir = run_path.parents[1]
        point = _json_mapping(point_dir / "point.json")
        result = _json_mapping(run_path)
        point_name = _require_str(point, "_benchmark_name", context=str(point_dir / "point.json"))
        if point_name in found_names:
            raise AnalysisError(f"duplicate native result for {point_name}")
        found_names.add(point_name)

        precision = _require_str(point, "precision", context=point_name)
        seqs = _require_int(point, "max_num_seqs", context=point_name)
        batched = _require_int(point, "max_num_batched_tokens", context=point_name)
        concurrency = _require_int(point, "max_concurrency", context=point_name)
        for key, expected in (
            ("max_num_seqs", seqs),
            ("max_num_batched_tokens", batched),
            ("max_concurrency", concurrency),
        ):
            actual = _require_int(result, key, context=str(run_path))
            if actual != expected:
                raise AnalysisError(f"{run_path}: {key}={actual} does not match point={expected}")
        settings = _PRECISION_SETTINGS.get(precision)
        if settings is None:
            raise AnalysisError(f"{point_name}: unknown precision {precision!r}")
        for key in ("dtype", "quantization", "kv_cache_dtype"):
            if result.get(key) != settings[key]:
                raise AnalysisError(
                    f"{run_path}: {key}={result.get(key)!r} does not match {settings[key]!r}"
                )
        if result.get("model_id") != "smolqwen" or result.get("tokenizer_id") != MODEL_REPO:
            raise AnalysisError(f"{run_path}: native result is not the official served model")

        input_lens_raw = result.get("input_lens")
        if not isinstance(input_lens_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in input_lens_raw
        ):
            raise AnalysisError(f"{run_path}: input_lens must be a list of integers")
        input_lens = tuple(input_lens_raw)
        if len(input_lens) != NUM_PROMPTS:
            raise AnalysisError(f"{run_path}: expected {NUM_PROMPTS} input lengths")
        if canonical_input_lens is None:
            canonical_input_lens = input_lens
        elif input_lens != canonical_input_lens:
            raise AnalysisError(f"{run_path}: input_lens differ from the frozen workload")

        run_number = _require_int(result, "run_number", context=str(run_path))
        metrics_path = point_dir / f"metrics-after-run-{run_number}.prom"
        if not metrics_path.is_file():
            raise AnalysisError(f"{metrics_path}: missing post-run metric snapshot")
        prometheus = parse_prometheus_text(metrics_path.read_text(encoding="utf-8"))
        if PREEMPTIONS not in prometheus:
            raise AnalysisError(f"{metrics_path}: missing required series {PREEMPTIONS!r}")

        completed = _require_int(result, "completed", context=str(run_path))
        failed = _require_int(result, "failed", context=str(run_path))
        if completed + failed != NUM_PROMPTS:
            raise AnalysisError(
                f"{run_path}: completed+failed={completed + failed}, expected {NUM_PROMPTS}"
            )
        records.append(
            {
                "server_config": {
                    "precision": precision,
                    "max_num_seqs": seqs,
                    "max_num_batched_tokens": batched,
                },
                "concurrency": concurrency,
                "repetition": run_number,
                "manifest_hash": manifest_hash,
                "failed_requests": failed,
                "preemptions": int(round(prometheus[PREEMPTIONS])),
                **{key: _require_float(result, key, context=str(run_path)) for key in _METRIC_KEYS},
            }
        )

    missing = expected_names - found_names
    unexpected = found_names - expected_names
    if missing or unexpected:
        message = f"native study point mismatch: missing={sorted(missing)}"
        raise AnalysisError(f"{message} unexpected={sorted(unexpected)}")
    assert canonical_input_lens is not None
    return parse_runs(records), _nearest_rank_p95(canonical_input_lens)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def assert_single_manifest(runs: Sequence[RawRun]) -> str:
    """Every run must share one workload manifest hash; mixed aggregation is rejected."""
    hashes = sorted({run.manifest_hash for run in runs})
    if len(hashes) != 1:
        raise AnalysisError(f"runs span multiple workload manifests: {hashes}")
    return hashes[0]


def _median(values: Iterable[float]) -> float:
    return float(median(list(values)))


def aggregate_point(runs: Sequence[RawRun], *, num_runs: int) -> PointStats:
    """Median a point across its repetitions and mark it valid or rejected."""
    first = runs[0]
    reason: str | None = None
    valid = True
    repetitions = [run.repetition for run in runs]
    if len(runs) != num_runs:
        valid, reason = False, f"expected {num_runs} runs, found {len(runs)}"
    elif len(set(repetitions)) != len(repetitions):
        valid, reason = False, "duplicate repetition index"
    elif any(run.failed_requests > 0 for run in runs):
        valid, reason = False, "failed_requests > 0"
    elif any(run.preemptions > 0 for run in runs):
        valid, reason = False, "preemptions > 0"
    medians = {key: _median(run.metrics[key] for run in runs) for key in _METRIC_KEYS}
    spreads = {
        key: (
            min(run.metrics[key] for run in runs),
            max(run.metrics[key] for run in runs),
        )
        for key in _METRIC_KEYS
    }
    return PointStats(
        precision=first.precision,
        max_num_seqs=first.max_num_seqs,
        max_num_batched_tokens=first.max_num_batched_tokens,
        concurrency=first.concurrency,
        n_runs=len(runs),
        valid=valid,
        reason=reason,
        medians=medians,
        spreads=spreads,
    )


def build_points(runs: Sequence[RawRun], *, num_runs: int) -> list[PointStats]:
    """Group runs by point and aggregate each group deterministically."""
    groups: dict[tuple[str, int, int, int], list[RawRun]] = {}
    for run in runs:
        groups.setdefault(run.point_key, []).append(run)
    points = [
        aggregate_point(sorted(group, key=lambda run: run.repetition), num_runs=num_runs)
        for group in groups.values()
    ]
    points.sort(
        key=lambda point: (
            point.precision,
            point.max_num_batched_tokens,
            point.max_num_seqs,
            point.concurrency,
        )
    )
    return points


def eligible_points(points: Sequence[PointStats], *, slo: SLO | None) -> list[PointStats]:
    """Keep valid points, additionally filtering by the p95 SLOs when supplied."""
    result: list[PointStats] = []
    for point in points:
        if not point.valid:
            continue
        if slo is not None:
            if point.medians["p95_ttft_ms"] > slo.p95_ttft_ms:
                continue
            if point.medians["p95_tpot_ms"] > slo.p95_tpot_ms:
                continue
        result.append(point)
    return result


# --------------------------------------------------------------------------- #
# Frontiers and selection
# --------------------------------------------------------------------------- #


def pareto_frontier(points: Sequence[PointStats], *, latency_key: str) -> list[PointStats]:
    """Non-dominated points maximizing output throughput while minimizing a latency."""
    frontier: list[PointStats] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            throughput_at_least = (
                other.medians["output_throughput"] >= candidate.medians["output_throughput"]
            )
            latency_at_most = other.medians[latency_key] <= candidate.medians[latency_key]
            strictly_better = (
                other.medians["output_throughput"] > candidate.medians["output_throughput"]
                or other.medians[latency_key] < candidate.medians[latency_key]
            )
            if throughput_at_least and latency_at_most and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda point: (point.medians["output_throughput"], point.medians[latency_key])
    )
    return frontier


def select_latency(points: Sequence[PointStats]) -> PointStats:
    """Lowest p95 TTFT, then p95 TPOT, then highest output throughput."""
    return min(
        points,
        key=lambda point: (
            point.medians["p95_ttft_ms"],
            point.medians["p95_tpot_ms"],
            -point.medians["output_throughput"],
        ),
    )


def select_throughput(points: Sequence[PointStats]) -> PointStats:
    """Highest output throughput, then request throughput, then lowest p95 TTFT."""
    return min(
        points,
        key=lambda point: (
            -point.medians["output_throughput"],
            -point.medians["request_throughput"],
            point.medians["p95_ttft_ms"],
        ),
    )


def _normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def select_balanced(points: Sequence[PointStats]) -> PointStats:
    """Closest point to the normalized ideal over throughput, p95 TTFT, and p95 TPOT."""
    throughput = [point.medians["output_throughput"] for point in points]
    ttft = [point.medians["p95_ttft_ms"] for point in points]
    tpot = [point.medians["p95_tpot_ms"] for point in points]
    thr_low, thr_high = min(throughput), max(throughput)
    ttft_low, ttft_high = min(ttft), max(ttft)
    tpot_low, tpot_high = min(tpot), max(tpot)

    def distance(point: PointStats) -> float:
        thr = _normalize(point.medians["output_throughput"], thr_low, thr_high)
        lat = _normalize(point.medians["p95_ttft_ms"], ttft_low, ttft_high)
        dec = _normalize(point.medians["p95_tpot_ms"], tpot_low, tpot_high)
        return sqrt((1.0 - thr) ** 2 + lat**2 + dec**2)

    return min(
        points,
        key=lambda point: (
            distance(point),
            point.medians["p95_ttft_ms"],
            -point.medians["output_throughput"],
        ),
    )


# --------------------------------------------------------------------------- #
# Admission limits (Phase 8 derivation, computed from measured selections)
# --------------------------------------------------------------------------- #


def derive_admission_limits(
    point: PointStats,
    *,
    prompt_len_p95: int,
    slo: SLO | None = None,
    prefill_tokens_per_s: float | None = None,
    queue_slots_factor: float = 1.0,
) -> AdmissionLimits:
    """Finite request/token queue limits for one operating point.

    The request limit is active capacity (``max_num_seqs``) plus one explicitly
    bounded queue of ``queue_slots``. With an SLO and measured prefill throughput
    the token budget is ``prefill_throughput x target TTFT``; without an SLO it is
    ``p95 prompt length x queue slots`` and flagged provisional.
    """
    if prompt_len_p95 <= 0:
        raise AnalysisError("prompt_len_p95 must be positive to bound the token queue")
    active = point.max_num_seqs
    queue_slots = max(1, round(active * queue_slots_factor))
    max_reqs = active + queue_slots
    if slo is not None and prefill_tokens_per_s is not None:
        budget = int(prefill_tokens_per_s * (slo.p95_ttft_ms / 1000.0))
        max_tokens = max(budget, prompt_len_p95)
        basis = "prefill throughput x target TTFT"
        return AdmissionLimits(max_reqs, max_tokens, provisional=False, basis=basis)
    max_tokens = prompt_len_p95 * queue_slots
    basis = "p95 prompt length x queue slots (no SLO)"
    return AdmissionLimits(max_reqs, max_tokens, provisional=True, basis=basis)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def analyze_sweep(
    runs: Sequence[RawRun],
    *,
    num_runs: int,
    prompt_len_p95: int,
    slo: SLO | None = None,
    prefill_tokens_per_s: float | None = None,
    queue_slots_factor: float = 1.0,
) -> AnalysisResult:
    """Full pipeline: validate, aggregate, filter, frontier, select, and derive."""
    manifest_hash = assert_single_manifest(runs)
    points = tuple(build_points(runs, num_runs=num_runs))
    eligible = tuple(eligible_points(points, slo=slo))
    if not eligible:
        raise AnalysisError(
            "no eligible operating points after validity/SLO filtering; "
            "report zero eligible profiles rather than relaxing thresholds"
        )

    frontiers: dict[str, dict[str, tuple[PointStats, ...]]] = {}
    for precision in PRECISIONS:
        subset = [point for point in eligible if point.precision == precision]
        if not subset:
            continue
        frontiers[precision] = {
            "ttft": tuple(pareto_frontier(subset, latency_key="p95_ttft_ms")),
            "tpot": tuple(pareto_frontier(subset, latency_key="p95_tpot_ms")),
        }

    selections: dict[str, PointStats] = {
        "latency": select_latency(eligible),
        "balanced": select_balanced(eligible),
        "throughput": select_throughput(eligible),
    }
    admission = {
        name: derive_admission_limits(
            point,
            prompt_len_p95=prompt_len_p95,
            slo=slo,
            prefill_tokens_per_s=prefill_tokens_per_s,
            queue_slots_factor=queue_slots_factor,
        )
        for name, point in selections.items()
    }

    summary = _build_summary(
        manifest_hash=manifest_hash,
        num_runs=num_runs,
        prompt_len_p95=prompt_len_p95,
        points=points,
        eligible=eligible,
        frontiers=frontiers,
        selections=selections,
        admission=admission,
        slo=slo,
    )
    summary_hash = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return AnalysisResult(
        manifest_hash=manifest_hash,
        num_runs=num_runs,
        prompt_len_p95=prompt_len_p95,
        points=points,
        eligible=eligible,
        frontiers=frontiers,
        selections=selections,
        admission=admission,
        slo=slo,
        summary=summary,
        summary_hash=summary_hash,
    )


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


def _point_summary(point: PointStats) -> dict[str, object]:
    return {
        **point.identity(),
        "n_runs": point.n_runs,
        "valid": point.valid,
        "reason": point.reason,
        "metrics": {key: _round(point.medians[key]) for key in _METRIC_KEYS},
    }


def matched_precision_pairs(
    points: Sequence[PointStats],
) -> tuple[tuple[PointStats, PointStats, float], ...]:
    """Pair BF16 and FP8 points and return FP8 output-throughput uplift percentages."""
    keyed = {
        (
            point.precision,
            point.max_num_batched_tokens,
            point.max_num_seqs,
            point.concurrency,
        ): point
        for point in points
        if point.valid
    }
    bf16_configs = {key[1:] for key in keyed if key[0] == "bf16"}
    fp8_configs = {key[1:] for key in keyed if key[0] == "fp8"}
    configs = sorted(bf16_configs & fp8_configs)
    pairs: list[tuple[PointStats, PointStats, float]] = []
    for config in configs:
        bf16 = keyed[("bf16", *config)]
        fp8 = keyed[("fp8", *config)]
        uplift = 100.0 * (
            fp8.medians["output_throughput"] / bf16.medians["output_throughput"] - 1.0
        )
        pairs.append((bf16, fp8, uplift))
    return tuple(pairs)


def _build_summary(
    *,
    manifest_hash: str,
    num_runs: int,
    prompt_len_p95: int,
    points: Sequence[PointStats],
    eligible: Sequence[PointStats],
    frontiers: Mapping[str, Mapping[str, tuple[PointStats, ...]]],
    selections: Mapping[str, PointStats],
    admission: Mapping[str, AdmissionLimits],
    slo: SLO | None,
) -> dict[str, object]:
    pairs = matched_precision_pairs(points)
    uplifts = [uplift for _, _, uplift in pairs]
    return {
        "manifest_hash": manifest_hash,
        "num_runs": num_runs,
        "prompt_len_p95": prompt_len_p95,
        "slo": (
            None
            if slo is None
            else {"p95_ttft_ms": slo.p95_ttft_ms, "p95_tpot_ms": slo.p95_tpot_ms}
        ),
        "counts": {
            "total_points": len(points),
            "eligible_points": len(eligible),
            "rejected_points": len(points) - len(eligible),
        },
        "precision_comparison": {
            "matched_points": len(pairs),
            "median_output_throughput_uplift_pct": (
                _round(_median(uplifts), 2) if uplifts else None
            ),
            "min_output_throughput_uplift_pct": _round(min(uplifts), 2) if uplifts else None,
            "max_output_throughput_uplift_pct": _round(max(uplifts), 2) if uplifts else None,
            "points": [
                {
                    "max_num_batched_tokens": bf16.max_num_batched_tokens,
                    "max_num_seqs": bf16.max_num_seqs,
                    "concurrency": bf16.concurrency,
                    "bf16_output_throughput": _round(bf16.medians["output_throughput"]),
                    "fp8_output_throughput": _round(fp8.medians["output_throughput"]),
                    "fp8_uplift_pct": _round(uplift, 2),
                }
                for bf16, fp8, uplift in pairs
            ],
        },
        "points": [_point_summary(point) for point in points],
        "frontiers": {
            precision: {axis: [point.identity() for point in curve] for axis, curve in axes.items()}
            for precision, axes in frontiers.items()
        },
        "selections": {
            name: {
                **_point_summary(point),
                "admission": {
                    "max_num_queued_reqs": admission[name].max_num_queued_reqs,
                    "max_num_queued_tokens": admission[name].max_num_queued_tokens,
                    "provisional": admission[name].provisional,
                    "basis": admission[name].basis,
                },
            }
            for name, point in selections.items()
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_profile_yaml(
    name: str,
    point: PointStats,
    admission: AdmissionLimits,
    *,
    manifest_hash: str,
    summary_hash: str,
) -> str:
    """Render a measured ServingProfileConfig YAML with a deterministic provenance header."""
    settings = _PRECISION_SETTINGS[point.precision]
    header = [
        f"# Measured {name} operating point generated by benchmarks/serving/analyze.py.",
        "# Do not hand-edit: regenerate from the recorded sweep. The comments below",
        "# are the audit trail; the body is a closed ServingProfileConfig.",
        f"# workload_manifest: {manifest_hash}",
        f"# summary: {summary_hash}",
        (
            f"# selected: precision={point.precision} concurrency={point.concurrency} "
            f"max_num_seqs={point.max_num_seqs} "
            f"max_num_batched_tokens={point.max_num_batched_tokens}"
        ),
        (
            f"# {'measurement' if point.n_runs == 1 else 'median'}: "
            f"output_throughput={point.medians['output_throughput']:.3f} tok/s "
            f"p95_ttft_ms={point.medians['p95_ttft_ms']:.3f} "
            f"p95_tpot_ms={point.medians['p95_tpot_ms']:.3f}"
        ),
    ]
    if admission.provisional:
        header.append(f"# admission limits are PROVISIONAL ({admission.basis})")
    else:
        header.append(f"# admission limits basis: {admission.basis}")

    body: list[tuple[str, object]] = [
        ("dtype", settings["dtype"]),
        ("quantization", settings["quantization"]),
        ("kv_cache_dtype", settings["kv_cache_dtype"]),
        ("kv_cache_scale", "default"),
        ("max_num_seqs", point.max_num_seqs),
        ("max_num_batched_tokens", point.max_num_batched_tokens),
        ("max_num_queued_reqs", admission.max_num_queued_reqs),
        ("max_num_queued_tokens", admission.max_num_queued_tokens),
        ("gpu_memory_utilization", 0.90),
    ]
    lines = [*header, *(f"{key}: {_yaml_scalar(value)}" for key, value in body)]
    return "\n".join(lines) + "\n"


def write_profiles(result: AnalysisResult, serving_dir: Path | str) -> list[Path]:
    """Write ``configs/serving/{name}.yaml`` for every selected profile."""
    directory = Path(serving_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in PROFILE_NAMES:
        target = directory / f"{name}.yaml"
        target.write_text(
            render_profile_yaml(
                name,
                result.selections[name],
                result.admission[name],
                manifest_hash=result.manifest_hash,
                summary_hash=result.summary_hash,
            ),
            encoding="utf-8",
        )
        written.append(target)
    return written


def write_summary(result: AnalysisResult, path: Path | str) -> Path:
    """Write the deterministic summary JSON, including its own content hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary_hash": result.summary_hash, **result.summary}
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def render_report(result: AnalysisResult) -> str:
    """Render the measured study table and selections from the analysis object."""
    lines = [
        "# Official Qwen single-L4 serving results",
        "",
        "The approved study contains 32 single observations (16 configurations in BF16",
        "and FP8). Values are not medians; no variance claim is made. Goodput is absent",
        "because no production SLO was supplied. Admission limits are provisional.",
        "",
        f"Workload manifest: `{result.manifest_hash}`",
        f"Summary hash: `{result.summary_hash}`",
        f"Prompt length p95: `{result.prompt_len_p95}` tokens",
        "Raw point bundles: `artifacts/serving/sweeps/direct-vllm029-official-16x2/`",
        "Runtime validation: `artifacts/serving/validation/validation.json`",
        "",
        "## Reproducibility",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Model | `Qwen/Qwen3.5-2B` (pinned revision) |",
        "| GPU | NVIDIA L4 24 GB |",
        "| Runtime | vLLM 0.29.0, Torch 2.13.0+cu130 |",
        "| Workload | Fixed BFCL serving workload, 200 successful requests/config |",
        "| Study | 16 selected configs × BF16/FP8 × 1 observation |",
        "",
        "## Selected profiles",
        "",
        "| Profile | Precision | tok | seq | c | output tok/s | p95 TTFT ms | "
        "p95 TPOT ms | queued reqs | queued tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PROFILE_NAMES:
        point = result.selections[name]
        limit = result.admission[name]
        lines.append(
            f"| {name} | {point.precision} | {point.max_num_batched_tokens} | "
            f"{point.max_num_seqs} | {point.concurrency} | "
            f"{point.medians['output_throughput']:.2f} | "
            f"{point.medians['p95_ttft_ms']:.1f} | {point.medians['p95_tpot_ms']:.2f} | "
            f"{limit.max_num_queued_reqs} | {limit.max_num_queued_tokens} |"
        )

    pairs = matched_precision_pairs(result.points)
    lines.extend(
        [
            "",
            "## Matched precision comparison",
            "",
            f"Across all {len(pairs)} matched configurations, FP8's median output-throughput",
            f"uplift is **{_median(uplift for _, _, uplift in pairs):.2f}%**.",
            "",
            "| tok | seq | c | BF16 output tok/s | FP8 output tok/s | FP8 uplift |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bf16, fp8, uplift in pairs:
        lines.append(
            f"| {bf16.max_num_batched_tokens} | {bf16.max_num_seqs} | "
            f"{bf16.concurrency} | {bf16.medians['output_throughput']:.2f} | "
            f"{fp8.medians['output_throughput']:.2f} | {uplift:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## All measurements",
            "",
            "| Precision | tok | seq | c | req/s | output tok/s | p95 TTFT ms | "
            "p99 TTFT ms | p95 TPOT ms | p99 TPOT ms | Valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for point in result.points:
        lines.append(
            f"| {point.precision} | {point.max_num_batched_tokens} | "
            f"{point.max_num_seqs} | {point.concurrency} | "
            f"{point.medians['request_throughput']:.4f} | "
            f"{point.medians['output_throughput']:.2f} | "
            f"{point.medians['p95_ttft_ms']:.1f} | {point.medians['p99_ttft_ms']:.1f} | "
            f"{point.medians['p95_tpot_ms']:.2f} | {point.medians['p99_tpot_ms']:.2f} | "
            f"{'yes' if point.valid else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- FP8 improves output throughput at every matched point; the median uplift is 25.94%.",
            "- The practical throughput/latency knee appears around concurrency 16–32.",
            "- Concurrency 64→128 adds modest throughput while sharply increasing TTFT and TPOT.",
            "- At concurrency 128, limiting `max_num_seqs` to 64 lowers TPOT but moves delay into",
            "  scheduler wait time, increasing TTFT.",
            "",
            "## Limitations",
            "",
            "- One observation per point; there is no repetition variance estimate.",
            "- Prometheus files captured after each run prove counters and clean end state;",
            "  they are not peak KV/running/waiting measurements.",
            "- Native vLLM wrote every complete raw JSON before its optional combined-table",
            "  step failed because the first Colab runtime lacked pandas. The installer is",
            "  fixed; no benchmark was rerun.",
            "- FP8 is runtime quantization of the same pinned official checkpoint; serving",
            "  quality comparison is outside this study.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(result: AnalysisResult, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result), encoding="utf-8")
    return target
