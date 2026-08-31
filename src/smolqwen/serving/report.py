"""Join measured serving latency/throughput with paired BFCL quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from smolqwen.eval.manifest import EvalManifest, assert_comparable

if TYPE_CHECKING:
    from smolqwen.serving.bench import BenchResult


@dataclass(frozen=True)
class QualityResult:
    score: float
    manifest: EvalManifest


_PAIRED_SERVING_FIELDS = (
    "dtype",
    "quantization",
    "speculative_decoding",
    "kv_budget",
    "max_num_seqs",
    "max_num_batched_tokens",
    "chunked_prefill",
    "prefix_caching",
)


def _normalized_recorded_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def assert_quality_matches_serving(rows: list[BenchResult], quality: QualityResult) -> None:
    """Refuse to attach a quality score measured under another serving config."""
    recorded = quality.manifest.recorded_free
    for index, row in enumerate(rows):
        differences = []
        for field in _PAIRED_SERVING_FIELDS:
            if field not in row.serving:
                continue
            expected = _normalized_recorded_value(row.serving[field])
            actual = _normalized_recorded_value(recorded.get(field))
            if expected != actual:
                differences.append(f"{field}: benchmark={expected!r}, quality={actual!r}")
        if differences:
            raise ValueError(
                f"quality report does not match benchmark row {index}: " + "; ".join(differences)
            )


def _load_report(path: Path | str) -> tuple[float, EvalManifest]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"quality report must be an object: {path}")
    manifest_payload = payload.get("manifest")
    metrics = payload.get("metrics")
    if not isinstance(manifest_payload, dict) or not isinstance(metrics, dict):
        raise ValueError(f"quality report is malformed: {path}")
    overall = metrics.get("multi_turn_overall")
    if not isinstance(overall, dict) or not isinstance(overall.get("score"), int | float):
        raise ValueError(f"quality report has no multi_turn_overall score: {path}")
    return float(overall["score"]), EvalManifest.from_dict(manifest_payload)


def load_quality_result(
    path: Path | str, *, references: list[Path] | tuple[Path, ...] = ()
) -> QualityResult:
    score, manifest = _load_report(path)
    reference_manifests = [_load_report(reference)[1] for reference in references]
    assert_comparable(*reference_manifests, manifest)
    return QualityResult(score=score, manifest=manifest)


def write_serving_report(
    output_dir: Path | str,
    *,
    rows: list[BenchResult],
    quality: QualityResult | None,
) -> Path:
    if quality is not None:
        assert_quality_matches_serving(rows, quality)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "report.md"
    header = (
        "| dataset | concurrency | successful | req/s | output tok/s | total tok/s | "
        "TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | ITL p50/p95/p99 ms | "
        "E2EL p50/p95/p99 ms | peak VRAM GB | "
        "MTP acceptance | BFCL-MT |"
    )
    divider = (
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        body.append(
            "| {dataset} | {concurrency} | {successful} | {requests:.2f} | {output:.2f} | "
            "{total:.2f} | "
            "{ttft} | {tpot} | {itl} | {e2el} | {vram} | {acceptance} | {quality} |".format(
                dataset=row.dataset,
                concurrency=row.concurrency,
                successful=row.successful_requests,
                requests=row.request_throughput,
                output=row.output_throughput,
                total=row.total_token_throughput,
                ttft=f"{row.ttft_p50_ms:.2f}/{row.ttft_p95_ms:.2f}/{row.ttft_p99_ms:.2f}",
                tpot=f"{row.tpot_p50_ms:.2f}/{row.tpot_p95_ms:.2f}/{row.tpot_p99_ms:.2f}",
                itl=f"{row.itl_p50_ms:.2f}/{row.itl_p95_ms:.2f}/{row.itl_p99_ms:.2f}",
                e2el=f"{row.e2el_p50_ms:.2f}/{row.e2el_p95_ms:.2f}/{row.e2el_p99_ms:.2f}",
                vram="pending" if row.peak_vram_gb is None else f"{row.peak_vram_gb:.2f}",
                acceptance=(
                    "pending"
                    if row.speculative_acceptance_rate is None
                    else f"{row.speculative_acceptance_rate:.4f}"
                ),
                quality=(
                    f"{quality.score:.4f}"
                    if quality is not None
                    else "pending"
                    if row.bfcl_mt_score is None
                    else f"{row.bfcl_mt_score:.4f}"
                ),
            )
        )
    serving_configs = [dict(row.serving) for row in rows]
    recorded = (
        json.dumps(dict(quality.manifest.recorded_free), sort_keys=True)
        if quality is not None
        else json.dumps(
            [
                dict(row.quality_recorded_free) if row.quality_recorded_free is not None else None
                for row in rows
            ],
            sort_keys=True,
        )
    )
    path.write_text(
        "\n".join(
            [
                "# Serving benchmark report",
                "",
                header,
                divider,
                *body,
                "",
                "Serving configurations: `" + json.dumps(serving_configs, sort_keys=True) + "`",
                "",
                "Paired BFCL recorded-free fields: `" + recorded + "`",
                "Paired BFCL invariant hash: `"
                + (
                    quality.manifest.invariant_hash
                    if quality is not None
                    else json.dumps([row.quality_invariant_hash for row in rows])
                )
                + "`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
