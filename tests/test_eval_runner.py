from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import EvalConfig
from smolqwen.eval import runner
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult
from smolqwen.eval.manifest import EvalManifest
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.policies import GenerationResult
from smolqwen.eval.runner import evaluate_adapter


class _Policy:
    revision = "a" * 40

    def generate(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> GenerationResult:
        return GenerationResult("done", 3, "stop")


class _Adapter:
    summarized = False

    def load_tasks(self) -> list[EvalTask]:
        return [EvalTask("case", "fixture", "prompt", ())]

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return [{"role": "user", "content": task.prompt}]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        return StepResult("finished", complete=True, env_steps=1)

    def score(self, task: EvalTask) -> AdapterResult:
        return AdapterResult(1.0, True)

    def invalid_call_count(self, task: EvalTask) -> int:
        return 1

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        return {"fixture_revision": "1", "task_ids": [task.task_id for task in tasks]}

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        self.summarized = True
        return aggregate(tasks)


def test_runner_collects_secondary_metrics_from_a_structured_generation() -> None:
    config = resolve("eval", profile="l4")
    assert isinstance(config, EvalConfig)
    adapter = _Adapter()
    progress: list[tuple[str, TaskMetrics | None]] = []
    metrics = evaluate_adapter(
        config,
        _Policy(),
        adapter,
        progress=lambda task, result: progress.append((task.task_id, result)),
    )
    assert adapter.summarized
    assert [task_id for task_id, _ in progress] == ["case", "case"]
    assert progress[0][1] is None
    assert progress[1][1] is not None
    assert metrics["fixture"] == {
        "score": 1.0,
        "invalid_call_rate": 1.0,
        "average_steps": 1.0,
        "average_generated_tokens": 3.0,
        "truncation_rate": 0.0,
        "exact_success_rate": 1.0,
    }


def test_named_adapter_logs_exact_progress_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = EvalConfig(adapters=("fixture",))
    adapter = _Adapter()
    monkeypatch.setattr(runner, "create_adapter", lambda *_: adapter)

    metrics, invariants = runner._evaluate_named_adapter(config, _Policy(), "fixture")

    assert metrics["fixture"]["score"] == 1.0
    assert invariants["fixture_revision"] == "1"
    output = capsys.readouterr().out
    assert "evaluation fixture: loaded 1 tasks" in output
    assert "evaluation fixture: 1/1 tasks" in output
    assert "score=1.0000 steps=1 tokens=3" in output
    assert "evaluation fixture complete: 1/1 tasks" in output
    assert "evaluation fixture: summarized 1 metric categories" in output


def test_run_evaluation_records_actual_serving_locator_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    policy = SimpleNamespace(revision="a" * 40, adapter_revision=None)
    captured: list[EvalManifest] = []
    monkeypatch.setattr(runner, "load_policy", lambda **_: policy)
    monkeypatch.setattr(
        runner,
        "_evaluate_named_adapter",
        lambda *_: ({"fixture": {"score": 1.0}}, {"dataset_hash": "hash"}),
    )

    def write_report(
        output_dir: str,
        *,
        tag: str,
        manifest: EvalManifest,
        metrics: dict[str, dict[str, float]],
    ) -> tuple[Path, Path]:
        captured.append(manifest)
        return tmp_path / "result.json", tmp_path / "result.md"

    monkeypatch.setattr(
        runner,
        "write_report",
        write_report,
    )
    args = SimpleNamespace(
        checkpoint=None,
        revision="a" * 40,
        endpoint="http://localhost:8000/v1",
        adapter_path=None,
        adapter_revision=None,
        adapter=None,
        tag="served",
        serving_backend="vllm",
        served_dtype="float8_e4m3fn",
        quantization="fp8",
        speculative_decoding="mtp-1",
        kv_budget="8GiB",
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        chunked_prefill=True,
        prefix_caching=True,
    )
    assert runner.run_evaluation(config, args) == 0
    recorded = captured[0].recorded_free
    assert recorded["backend"] == "vllm"
    assert recorded["endpoint"] == args.endpoint
    assert recorded["served_model"] == config.http_model
    assert recorded["checkpoint_revision"] == "a" * 40
    assert recorded["quantization"] == "fp8"


def test_run_evaluation_refuses_an_empty_adapter_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EvalConfig()
    monkeypatch.setattr(
        runner,
        "load_policy",
        lambda **_: SimpleNamespace(revision="a" * 40, adapter_revision=None),
    )
    args = SimpleNamespace(
        checkpoint="model",
        revision="a" * 40,
        endpoint=None,
        adapter_path=None,
        adapter_revision=None,
        adapter=None,
    )
    with pytest.raises(ValueError, match="at least one benchmark adapter"):
        runner.run_evaluation(config, args)
