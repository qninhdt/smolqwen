from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.config_models import ServeConfig
from smolqwen.eval.manifest import EvalManifest, ManifestMismatchError
from smolqwen.serving.bench import (
    BenchError,
    benchmark_config_id,
    load_normalized_rows,
    parse_bench_payload,
)
from smolqwen.serving.report import (
    QualityResult,
    assert_quality_matches_serving,
    load_quality_result,
    write_serving_report,
)


def _payload() -> dict[str, float]:
    payload = {
        "completed": 10,
        "request_throughput": 1.5,
        "output_throughput": 300.0,
        "total_token_throughput": 450.0,
        "peak_gpu_memory_gb": 19.25,
        "spec_decode_acceptance_rate": 0.72,
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        payload[f"median_{metric}_ms"] = 10.0
        payload[f"p95_{metric}_ms"] = 20.0
        payload[f"p99_{metric}_ms"] = 30.0
    return payload


def test_vllm_result_normalizes_all_required_latency_percentiles() -> None:
    row = parse_bench_payload(
        _payload(),
        dataset="random",
        concurrency=4,
        serving={"quantization": "fp8"},
    )
    assert row.successful_requests == 10
    assert row.ttft_p50_ms == 10.0
    assert row.tpot_p95_ms == 20.0
    assert row.e2el_p99_ms == 30.0
    assert row.peak_vram_gb == 19.25
    assert row.speculative_acceptance_rate == 0.72


def test_missing_promised_metric_fails_instead_of_writing_partial_row() -> None:
    payload = _payload()
    del payload["p95_e2el_ms"]
    with pytest.raises(BenchError, match="p95_e2el_ms"):
        parse_bench_payload(payload, dataset="random", concurrency=1, serving={})


def test_report_pairs_speed_with_bfcl_quality_and_recorded_config(tmp_path: Path) -> None:
    row = parse_bench_payload(
        _payload(),
        dataset="random",
        concurrency=4,
        serving={"quantization": "fp8", "max_num_seqs": 64},
    )
    quality = QualityResult(
        score=0.42,
        manifest=EvalManifest(
            {"decoding": {"temperature": 0.0}},
            {"quantization": "fp8", "max_num_seqs": 64},
        ),
    )
    report = write_serving_report(tmp_path, rows=[row], quality=quality)
    text = report.read_text(encoding="utf-8")
    assert "300.00" in text
    assert "0.4200" in text
    assert '"quantization": "fp8"' in text


def test_serving_config_serializes_benchmark_contract() -> None:
    dumped = json.loads(ServeConfig().model_dump_json())
    assert dumped["benchmark_percentiles"] == [50, 95, 99]


def test_normalized_rows_accumulate_across_serving_configs(tmp_path: Path) -> None:
    first = parse_bench_payload(
        _payload(), dataset="random", concurrency=1, serving={"dtype": "bfloat16"}
    )
    second = parse_bench_payload(
        _payload(), dataset="random", concurrency=1, serving={"dtype": "float16"}
    )
    for row in (first, second):
        suffix = benchmark_config_id(row.serving)
        (tmp_path / f"random-c1-{suffix}.json").write_text(
            json.dumps(row.to_dict()), encoding="utf-8"
        )
    loaded = load_normalized_rows(tmp_path)
    assert [row.serving["dtype"] for row in loaded] == ["bfloat16", "float16"]


def test_quality_pairing_refuses_invariant_drift(tmp_path: Path) -> None:
    def write_eval(name: str, temperature: float) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "manifest": EvalManifest(
                        {"decoding": {"temperature": temperature}}, {}
                    ).to_dict(),
                    "metrics": {"multi_turn_overall": {"score": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        return path

    reference = write_eval("base.json", 0.0)
    serving = write_eval("served.json", 0.1)
    with pytest.raises(ManifestMismatchError, match="temperature"):
        load_quality_result(serving, references=[reference])


def test_quality_pairing_refuses_a_different_serving_config() -> None:
    row = parse_bench_payload(
        _payload(),
        dataset="random",
        concurrency=4,
        serving={"dtype": "bfloat16", "quantization": "fp8", "kv_budget": 0.9},
    )
    quality = QualityResult(
        score=0.5,
        manifest=EvalManifest({}, {"dtype": "bfloat16", "quantization": "awq", "kv_budget": "0.9"}),
    )
    with pytest.raises(ValueError, match="quantization"):
        assert_quality_matches_serving([row], quality)


def test_quality_pairing_refuses_dtype_drift() -> None:
    row = parse_bench_payload(
        _payload(), dataset="random", concurrency=1, serving={"dtype": "bfloat16"}
    )
    quality = QualityResult(score=0.5, manifest=EvalManifest({}, {"dtype": "float16"}))
    with pytest.raises(ValueError, match="dtype"):
        assert_quality_matches_serving([row], quality)
