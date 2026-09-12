"""Tests for the frozen serving-benchmark specification and workload manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.serving.benchmark_spec import (
    DATASET_REPO,
    PRECISIONS,
    STUDY_POINTS,
    SpecError,
    build_server_configs,
    build_sweep_spec,
    build_workload_manifest,
    render_study_points,
    render_sweep_json,
    sweep_point_count,
    validate_categories,
)

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "serving"
_SWEEP_JSON = _BENCH_DIR / "sweep.json"


def test_server_configs_are_the_unique_configs_required_by_the_study() -> None:
    configs = build_server_configs()
    assert len(configs) == 22
    assert len(set(configs)) == len(configs)
    assert all(config.max_num_batched_tokens >= config.max_num_seqs for config in configs)


def test_point_count_is_the_approved_16x2_map() -> None:
    assert sweep_point_count() == len(STUDY_POINTS) * len(PRECISIONS) == 32


def test_validate_categories_rejects_multi_turn_and_unknown() -> None:
    assert validate_categories(["parallel", "simple"]) == ("parallel", "simple")
    with pytest.raises(SpecError, match="multi-turn"):
        validate_categories(["multi_turn_base"])
    with pytest.raises(SpecError, match="unsupported"):
        validate_categories(["irrelevance"])
    with pytest.raises(SpecError, match="at least one"):
        validate_categories([])


def test_committed_sweep_json_matches_the_spec_module() -> None:
    assert _SWEEP_JSON.exists(), "run scripts/prepare-serving-benchmark.py --regenerate-spec-only"
    on_disk = _SWEEP_JSON.read_text(encoding="utf-8")
    assert on_disk == render_sweep_json()
    spec = build_sweep_spec()
    assert spec["point_count"] == 32
    assert spec["num_runs"] == 1
    assert len(spec["points"]) == 32  # type: ignore[arg-type]


def test_committed_study_points_match_the_spec_module() -> None:
    path = _BENCH_DIR / "study-points.json"
    assert path.exists(), "run scripts/prepare-serving-benchmark.py --regenerate-spec-only"
    assert path.read_text(encoding="utf-8") == render_study_points()


def test_workload_manifest_hash_is_deterministic_and_content_addressed() -> None:
    kwargs = dict(
        dataset_revision="c" * 40,
        categories=["simple", "parallel"],
        num_prompts=200,
        seed=1234,
        model="smolqwen",
        tokenizer_revision="r1",
        generation={"temperature": 0.0, "max_tokens": 256},
        dataset_file_sha256={"simple": "a" * 64, "parallel": "b" * 64},
    )
    first = build_workload_manifest(**kwargs)  # type: ignore[arg-type]
    second = build_workload_manifest(**kwargs)  # type: ignore[arg-type]
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["num_prompts"] == 200
    assert first["categories"] == ["parallel", "simple"]  # normalized + sorted

    changed = build_workload_manifest(**{**kwargs, "seed": 7})  # type: ignore[arg-type]
    assert changed["manifest_hash"] != first["manifest_hash"]


def test_workload_manifest_rejects_bad_num_prompts_and_sha_mismatch() -> None:
    base = dict(
        dataset_revision="c" * 40,
        categories=["simple"],
        seed=1,
        model="m",
        tokenizer_revision="r",
        generation={},
        dataset_file_sha256={"simple": "a" * 64},
    )
    with pytest.raises(SpecError, match="num_prompts"):
        build_workload_manifest(**{**base, "num_prompts": 0})  # type: ignore[arg-type]
    # A sha per requested category is required: extra or missing keys are rejected.
    with pytest.raises(SpecError, match="one sha per requested category"):
        build_workload_manifest(
            **{  # type: ignore[arg-type]
                **base,
                "num_prompts": 10,
                "dataset_file_sha256": {"simple": "a" * 64, "parallel": "b" * 64},
            }
        )
    with pytest.raises(SpecError, match="one sha per requested category"):
        build_workload_manifest(
            **{**base, "num_prompts": 10, "dataset_file_sha256": {}}  # type: ignore[arg-type]
        )


def test_study_points_are_the_requested_16x2_map() -> None:
    points = json.loads(render_study_points())
    assert len(points) == 32
    assert {point["precision"] for point in points} == {"bf16", "fp8"}
    assert all(point["max_concurrency"] in {1, 4, 16, 32, 64, 128} for point in points)
    assert len({point["_benchmark_name"] for point in points}) == 32
    expected = [
        (f"{precision}-{name}", batched, seqs, concurrency)
        for precision in PRECISIONS
        for name, batched, seqs, concurrency in STUDY_POINTS
    ]
    actual = [
        (
            point["_benchmark_name"],
            point["max_num_batched_tokens"],
            point["max_num_seqs"],
            point["max_concurrency"],
        )
        for point in points
    ]
    assert actual == expected


def test_sweep_json_records_native_bfcl_dataset_and_manifest_pointer() -> None:
    spec = json.loads(render_sweep_json())
    assert spec["dataset_name"] == "hf"
    assert spec["dataset_repo"] == DATASET_REPO
    assert spec["num_prompts"] >= 1
    assert spec["workload_manifest"].endswith(".json")
    assert "workload" not in spec  # no locally rendered file to point at
    assert spec["num_runs"] == 1
    assert spec["point_count"] == 32
    assert spec["server_config_count"] == 22
