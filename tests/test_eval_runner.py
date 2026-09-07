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
    metrics = evaluate_adapter(config, _Policy(), adapter)
    assert adapter.summarized
    assert metrics["fixture"] == {
        "score": 1.0,
        "invalid_call_rate": 1.0,
        "average_steps": 1.0,
        "average_generated_tokens": 3.0,
        "truncation_rate": 0.0,
        "exact_success_rate": 1.0,
        # How the episode ended, aggregated as a rate. A run where this reads
        # `terminal_step_cap_rate: 1.0` generated nothing and its score describes the
        # environment's initial state -- which is what a T4 run reported before this
        # existed.
        "terminal_final_answer_rate": 1.0,
        "terminal_reason_denominator": 1.0,
    }


def test_run_evaluation_records_actual_serving_locator_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    policy = SimpleNamespace(revision="a" * 40, adapter_revision=None)
    captured: list[EvalManifest] = []
    monkeypatch.setattr(runner, "load_http_policy", lambda **_: policy)
    monkeypatch.setattr(
        runner,
        "_evaluate_named_adapter",
        lambda *_args, **_kwargs: ({"fixture": {"score": 1.0}}, {"dataset_hash": "hash"}),
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
        require_serving_match=None,
        # Attributes no CLI flag produces any more. Present here to prove the runner
        # does not read them: the eight serving-detail flags were deleted in favour
        # of recording what the engine resolved, and reading a caller-supplied value
        # was the way to record something other than what ran.
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
    # An endpoint's serving config belongs to a process this command cannot inspect,
    # so it is recorded as unknown rather than as whatever the caller typed.
    assert recorded["quantization"] is None
    assert recorded["max_num_seqs"] is None
    # What generation used, recorded rather than asserted on the command line.
    assert recorded["generation_concurrency"] == config.profile.generation_concurrency
    assert recorded["enforce_eager"] == config.profile.enforce_eager
    assert recorded["trajectory_records"]["fixture"].endswith("served-fixture.jsonl")


def test_run_evaluation_refuses_an_empty_adapter_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EvalConfig()
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
