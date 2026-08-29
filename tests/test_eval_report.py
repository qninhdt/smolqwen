from __future__ import annotations

from pathlib import Path

import pytest

from smolqwen.eval.manifest import EvalManifest, ManifestMismatchError
from smolqwen.eval.report import write_comparison_report, write_report


def test_report_writes_metrics_and_recorded_free_fields(tmp_path: Path) -> None:
    manifest = EvalManifest({"seed": 1234}, {"backend": "http", "quantization": "fp8"})
    json_path, markdown_path = write_report(
        tmp_path,
        tag="base",
        manifest=manifest,
        metrics={
            "multi_turn_base": {
                "score": 1.0,
                "invalid_call_rate": 0.0,
                "average_steps": 2.0,
                "average_generated_tokens": 4.0,
                "truncation_rate": 0.0,
            },
            "envscaler_heldout": {
                "score": 0.5,
                "invalid_call_rate": 0.0,
                "average_steps": 2.0,
                "average_generated_tokens": 4.0,
                "truncation_rate": 0.0,
                "exact_success_rate": 0.25,
            },
        },
    )
    assert '"quantization": "fp8"' in json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Recorded-free" in markdown
    assert "| truncation rate | exact-success rate |" in markdown


def test_comparison_report_asserts_manifests_and_joins_secondary_metrics(tmp_path: Path) -> None:
    invariant = {"seed": 1234, "system_prompt_hash": "same"}
    for tag, score in (("base", 0.25), ("sft", 0.5)):
        write_report(
            tmp_path,
            tag=tag,
            manifest=EvalManifest(
                invariant,
                {"backend": "transformers" if tag == "base" else "vllm"},
            ),
            metrics={
                "multi_turn_base": {
                    "score": score,
                    "invalid_call_rate": 0.1,
                    "average_steps": 2.0,
                    "average_generated_tokens": 4.0,
                    "truncation_rate": 0.0,
                },
                "multi_turn_overall": {
                    "score": score,
                    "invalid_call_rate": 0.1,
                    "average_steps": 2.0,
                    "average_generated_tokens": 4.0,
                    "truncation_rate": 0.0,
                },
            },
        )

    json_path, markdown_path = write_comparison_report(tmp_path, tags=("base", "sft"))
    payload = json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert '"multi_turn_overall"' in payload
    assert "base score" in markdown
    assert "sft score" in markdown
    assert "Recorded-free execution fields" in markdown


def test_comparison_report_refuses_invariant_drift_before_writing(tmp_path: Path) -> None:
    metrics = {
        "multi_turn_overall": {
            "score": 0.5,
            "invalid_call_rate": 0.0,
            "average_steps": 1.0,
            "average_generated_tokens": 2.0,
            "truncation_rate": 0.0,
        }
    }
    write_report(
        tmp_path,
        tag="base",
        manifest=EvalManifest({"seed": 1}, {"backend": "transformers"}),
        metrics=metrics,
    )
    write_report(
        tmp_path,
        tag="sft",
        manifest=EvalManifest({"seed": 2}, {"backend": "transformers"}),
        metrics=metrics,
    )
    with pytest.raises(ManifestMismatchError, match="seed"):
        write_comparison_report(tmp_path, tags=("base", "sft"))
    assert not (tmp_path / "base_vs_sft.json").exists()
