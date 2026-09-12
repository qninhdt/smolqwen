"""Unit tests for the CPU-only serving-sweep analyzer.

Every rule the Phase 7 analyzer must obey is exercised here with synthetic runs:
median-of-three aggregation, validity/SLO filtering, Pareto dominance, the three
profile selection rules, admission-limit derivation, and deterministic profile
emission that loads back into the closed serving schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smolqwen.config_models import ServingProfileConfig
from smolqwen.serving.analysis import (
    SLO,
    AnalysisError,
    analyze_sweep,
    assert_single_manifest,
    build_points,
    derive_admission_limits,
    eligible_points,
    load_raw_runs,
    load_vllm_study_runs,
    pareto_frontier,
    parse_runs,
    render_profile_yaml,
    render_report,
    select_balanced,
    select_latency,
    select_throughput,
    write_profiles,
    write_summary,
)
from smolqwen.serving.benchmark_spec import PRECISION_SETTINGS, build_study_points
from smolqwen.serving.plots import write_figures


def make_run(
    *,
    precision: str,
    seqs: int,
    batched: int,
    concurrency: int,
    repetition: int,
    req: float,
    out: float,
    ttft: float,
    tpot: float,
    failed: int = 0,
    preemptions: int = 0,
    manifest: str = "manifest-0",
) -> dict[str, object]:
    return {
        "server_config": {
            "precision": precision,
            "max_num_seqs": seqs,
            "max_num_batched_tokens": batched,
        },
        "concurrency": concurrency,
        "repetition": repetition,
        "manifest_hash": manifest,
        "failed_requests": failed,
        "preemptions": preemptions,
        "request_throughput": req,
        "output_throughput": out,
        "p95_ttft_ms": ttft,
        "p99_ttft_ms": ttft * 1.2,
        "p95_tpot_ms": tpot,
        "p99_tpot_ms": tpot * 1.2,
        "peak_kv_usage": 0.5,
        "max_running": float(seqs),
        "max_waiting": 0.0,
        "mean_queue_time_ms": 1.0,
    }


def runs_for(
    precision: str,
    seqs: int,
    batched: int,
    concurrency: int,
    req: float,
    out: float,
    ttft: float,
    tpot: float,
    *,
    failed: int = 0,
    preemptions: int = 0,
    manifest: str = "manifest-0",
    n: int = 3,
) -> list[dict[str, object]]:
    return [
        make_run(
            precision=precision,
            seqs=seqs,
            batched=batched,
            concurrency=concurrency,
            repetition=rep,
            req=req,
            out=out,
            ttft=ttft,
            tpot=tpot,
            failed=failed,
            preemptions=preemptions,
            manifest=manifest,
        )
        for rep in range(1, n + 1)
    ]


# precision, seqs, batched, concurrency, req/s, out tok/s, p95 ttft ms, p95 tpot ms
A = ("bf16", 32, 4096, 8, 5.0, 500.0, 50.0, 10.0)
B = ("bf16", 64, 8192, 32, 15.0, 1500.0, 120.0, 20.0)
C = ("bf16", 128, 16384, 128, 25.0, 2500.0, 400.0, 45.0)
D = ("fp8", 64, 8192, 64, 26.0, 2600.0, 300.0, 40.0)


def grid_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for point in (A, B, C, D):
        runs.extend(runs_for(*point))
    return runs


def test_parse_rejects_unknown_precision_and_missing_fields() -> None:
    with pytest.raises(AnalysisError, match="precision"):
        parse_runs(runs_for("int4", 32, 4096, 8, 5.0, 500.0, 50.0, 10.0, n=1))
    broken = runs_for(*A, n=1)[0]
    del broken["p95_ttft_ms"]
    with pytest.raises(AnalysisError, match="p95_ttft_ms"):
        parse_runs([broken])


def test_assert_single_manifest_rejects_mixed_workloads() -> None:
    runs = parse_runs(runs_for(*A) + runs_for(*B, manifest="other"))
    with pytest.raises(AnalysisError, match="multiple workload manifests"):
        assert_single_manifest(runs)


def test_aggregate_medians_three_runs_and_marks_validity() -> None:
    runs = parse_runs(
        [
            make_run(
                precision="bf16",
                seqs=32,
                batched=4096,
                concurrency=8,
                repetition=1,
                req=5.0,
                out=480.0,
                ttft=50.0,
                tpot=10.0,
            ),
            make_run(
                precision="bf16",
                seqs=32,
                batched=4096,
                concurrency=8,
                repetition=2,
                req=5.0,
                out=500.0,
                ttft=50.0,
                tpot=10.0,
            ),
            make_run(
                precision="bf16",
                seqs=32,
                batched=4096,
                concurrency=8,
                repetition=3,
                req=5.0,
                out=700.0,
                ttft=50.0,
                tpot=10.0,
            ),
        ]
    )
    points = build_points(runs, num_runs=3)
    assert len(points) == 1
    assert points[0].valid is True
    # Median of {480, 500, 700} is the middle value, not the mean.
    assert points[0].medians["output_throughput"] == 500.0
    assert points[0].spreads["output_throughput"] == (480.0, 700.0)


def test_incomplete_failed_and_preempted_points_are_rejected() -> None:
    runs = parse_runs(
        runs_for(*A) + runs_for(*B, n=2) + runs_for(*C, failed=2) + runs_for(*D, preemptions=1)
    )
    points = build_points(runs, num_runs=3)
    by_key = {point.point_key: point for point in points}
    assert by_key[("bf16", 32, 4096, 8)].valid is True
    assert by_key[("bf16", 64, 8192, 32)].reason == "expected 3 runs, found 2"
    assert by_key[("bf16", 128, 16384, 128)].reason == "failed_requests > 0"
    assert by_key[("fp8", 64, 8192, 64)].reason == "preemptions > 0"
    assert eligible_points(points, slo=None) == [by_key[("bf16", 32, 4096, 8)]]


def test_slo_filter_excludes_points_over_threshold() -> None:
    points = build_points(parse_runs(grid_runs()), num_runs=3)
    slo = SLO(p95_ttft_ms=150.0, p95_tpot_ms=30.0)
    kept = {point.point_key for point in eligible_points(points, slo=slo)}
    # Only A (ttft 50 / tpot 10) and B (120 / 20) satisfy both thresholds.
    assert kept == {("bf16", 32, 4096, 8), ("bf16", 64, 8192, 32)}


def test_pareto_frontier_drops_dominated_points() -> None:
    # out 1000 / ttft 200 is dominated by B (out 1500 / ttft 120).
    dominated = ("bf16", 48, 8192, 16, 10.0, 1000.0, 200.0, 25.0)
    points = build_points(
        parse_runs(runs_for(*A) + runs_for(*B) + runs_for(*C) + runs_for(*dominated)),
        num_runs=3,
    )
    frontier = pareto_frontier(points, latency_key="p95_ttft_ms")
    keys = {point.point_key for point in frontier}
    assert ("bf16", 48, 8192, 16) not in keys
    assert keys == {("bf16", 32, 4096, 8), ("bf16", 64, 8192, 32), ("bf16", 128, 16384, 128)}


def test_profile_selection_rules_pick_distinct_points() -> None:
    eligible = eligible_points(build_points(parse_runs(grid_runs()), num_runs=3), slo=None)
    assert select_latency(eligible).point_key == ("bf16", 32, 4096, 8)
    assert select_throughput(eligible).point_key == ("fp8", 64, 8192, 64)
    assert select_balanced(eligible).point_key == ("bf16", 64, 8192, 32)


def test_admission_limits_provisional_without_slo_and_measured_with_slo() -> None:
    point = build_points(parse_runs(runs_for(*B)), num_runs=3)[0]
    provisional = derive_admission_limits(point, prompt_len_p95=1000)
    assert provisional.provisional is True
    assert provisional.max_num_queued_reqs == point.max_num_seqs * 2  # active + one queue
    assert provisional.max_num_queued_tokens == 1000 * point.max_num_seqs

    measured = derive_admission_limits(
        point,
        prompt_len_p95=1000,
        slo=SLO(p95_ttft_ms=200.0, p95_tpot_ms=30.0),
        prefill_tokens_per_s=10_000.0,
    )
    assert measured.provisional is False
    assert measured.max_num_queued_tokens == int(10_000.0 * (200.0 / 1000.0))


def test_admission_rejects_nonpositive_prompt_length() -> None:
    point = build_points(parse_runs(runs_for(*B)), num_runs=3)[0]
    with pytest.raises(AnalysisError, match="prompt_len_p95 must be positive"):
        derive_admission_limits(point, prompt_len_p95=0)


def test_analyze_sweep_raises_when_no_point_is_eligible() -> None:
    runs = parse_runs(runs_for(*A, failed=1))
    with pytest.raises(AnalysisError, match="no eligible operating points"):
        analyze_sweep(runs, num_runs=3, prompt_len_p95=1024)


def test_end_to_end_writes_loadable_deterministic_profiles(tmp_path: Path) -> None:
    runs = parse_runs(grid_runs())
    result = analyze_sweep(runs, num_runs=3, prompt_len_p95=1024)

    serving_dir = tmp_path / "serving"
    first = {
        path.name: path.read_text(encoding="utf-8") for path in write_profiles(result, serving_dir)
    }
    assert set(first) == {"latency.yaml", "balanced.yaml", "throughput.yaml"}

    for text in first.values():
        # The generated body must load into the closed serving schema unchanged.
        profile = ServingProfileConfig.model_validate(yaml.safe_load(text))
        assert profile.max_num_queued_reqs is not None
        assert profile.max_num_queued_tokens is not None

    # The throughput profile selects the FP8 point and carries fp8 quantization.
    throughput = ServingProfileConfig.model_validate(yaml.safe_load(first["throughput.yaml"]))
    assert throughput.quantization == "fp8"
    assert throughput.kv_cache_dtype == "fp8"

    # Regeneration from identical data produces byte-identical files (no diff).
    again = analyze_sweep(runs, num_runs=3, prompt_len_p95=1024)
    assert again.summary_hash == result.summary_hash
    second = {
        path.name: path.read_text(encoding="utf-8") for path in write_profiles(again, serving_dir)
    }
    assert first == second


def test_summary_json_records_counts_selections_and_hash(tmp_path: Path) -> None:
    result = analyze_sweep(parse_runs(grid_runs()), num_runs=3, prompt_len_p95=1024)
    summary_path = write_summary(result, tmp_path / "summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["summary_hash"] == result.summary_hash
    assert payload["counts"]["eligible_points"] == 4
    assert payload["counts"]["rejected_points"] == 0
    assert set(payload["selections"]) == {"latency", "balanced", "throughput"}
    assert payload["selections"]["throughput"]["precision"] == "fp8"


def test_load_raw_runs_reads_directory_of_json(tmp_path: Path) -> None:
    for index, run in enumerate(grid_runs()):
        (tmp_path / f"run-{index:02d}.json").write_text(json.dumps(run), encoding="utf-8")
    (tmp_path / "batch.json").write_text(json.dumps([]), encoding="utf-8")  # empty list is skipped
    runs = load_raw_runs(tmp_path)
    assert len(runs) == len(grid_runs())


def test_render_profile_header_flags_provisional_admission() -> None:
    point = build_points(parse_runs(runs_for(*B)), num_runs=3)[0]
    limits = derive_admission_limits(point, prompt_len_p95=1024)
    text = render_profile_yaml("balanced", point, limits, manifest_hash="abc", summary_hash="def")
    assert "PROVISIONAL" in text
    assert "# workload_manifest: abc" in text
    assert text.endswith("\n")


def _write_native_study(root: Path) -> Path:
    input_lens = [100] * 189 + [808] * 11
    for index, point in enumerate(build_study_points()):
        point_dir = root / str(point["_benchmark_name"])
        run_dir = point_dir / "SERVE--point-BENCH--load"
        run_dir.mkdir(parents=True)
        (point_dir / "point.json").write_text(json.dumps(point), encoding="utf-8")
        settings = PRECISION_SETTINGS[str(point["precision"])]
        result = {
            "dtype": settings["dtype"],
            "quantization": settings["quantization"],
            "kv_cache_dtype": settings["kv_cache_dtype"],
            "model_id": "smolqwen",
            "tokenizer_id": "Qwen/Qwen3.5-2B",
            "max_num_seqs": point["max_num_seqs"],
            "max_num_batched_tokens": point["max_num_batched_tokens"],
            "max_concurrency": point["max_concurrency"],
            "run_number": 0,
            "completed": 200,
            "failed": 0,
            "input_lens": input_lens,
            "request_throughput": float(index + 1),
            "output_throughput": float((index + 1) * 10),
            "p95_ttft_ms": float(index + 10),
            "p99_ttft_ms": float(index + 12),
            "p95_tpot_ms": float(index + 2),
            "p99_tpot_ms": float(index + 3),
        }
        (run_dir / "run=0.json").write_text(json.dumps(result), encoding="utf-8")
        (point_dir / "metrics-after-run-0.prom").write_text(
            'vllm:num_preemptions_total{engine="0"} 0\n', encoding="utf-8"
        )
    manifest = root.parent / "workload-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_hash": "manifest-0",
                "model": "Qwen/Qwen3.5-2B",
                "tokenizer_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_native_single_run_study_generates_profiles_figures_and_report(tmp_path: Path) -> None:
    study = tmp_path / "study"
    manifest = _write_native_study(study)
    runs, prompt_len_p95 = load_vllm_study_runs(study, manifest_path=manifest)
    assert len(runs) == 32
    assert prompt_len_p95 == 808

    result = analyze_sweep(runs, num_runs=1, prompt_len_p95=prompt_len_p95)
    assert result.summary["num_runs"] == 1
    assert result.summary["prompt_len_p95"] == 808
    assert "metrics" in result.summary["points"][0]  # type: ignore[index]
    comparison = result.summary["precision_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["matched_points"] == 16

    profiles = write_profiles(result, tmp_path / "profiles")
    figures = write_figures(result, tmp_path / "artifacts")
    report = render_report(result)
    assert len(profiles) == 3
    assert all("# measurement:" in path.read_text(encoding="utf-8") for path in profiles)
    assert {path.name for path in figures} == {
        "concurrency-scaling.png",
        "matched-fp8-throughput-uplift.png",
        "throughput-vs-p95-ttft.png",
        "throughput-vs-p95-tpot.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)
    assert "32 single observations" in report
    assert "## Selected profiles" in report
