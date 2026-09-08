from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config_models import EvalConfig
from smolqwen.eval import runner
from smolqwen.eval.bfcl_runner import BfclRequest, BfclRun
from tests.helpers import OfflineTokenizer

SHA = "a" * 40
METRICS = {
    "multi_turn_base": {
        "score": 0.5,
        "invalid_call_rate": 0.0,
        "average_steps": 2.0,
        "average_generated_tokens": 3.0,
        "truncation_rate": 0.0,
    }
}


class _Tracker:
    def __init__(self) -> None:
        self.finished = False
        self.artifacts: list[tuple[Path, list[Path]]] = []

    def start(self) -> None:
        pass

    def log(self, _payload: Any) -> None:
        pass

    def log_artifact(
        self, path: Path, *, name: str, artifact_type: str, extra_paths: list[Path]
    ) -> None:
        self.artifacts.append((path, extra_paths))

    def finish(self) -> None:
        self.finished = True


class _Engine:
    profile = SimpleNamespace(dtype="bfloat16")

    def __init__(self) -> None:
        self.shutdowns = 0

    def serving_config(self) -> dict[str, Any]:
        return {"dtype": "bfloat16"}

    def shutdown(self) -> None:
        self.shutdowns += 1


def _args(checkpoint: Path, **overrides: Any) -> SimpleNamespace:
    payload = {
        "checkpoint": str(checkpoint),
        "revision": SHA,
        "adapter_path": None,
        "adapter_revision": None,
        "tag": "base",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[_Engine, _Tracker]:
    engine = _Engine()
    tracker = _Tracker()
    monkeypatch.setattr(runner, "load_bfcl_tasks", lambda *_: ([object()], "b" * 40))
    monkeypatch.setattr(runner, "_engine_for", lambda *_: (engine, None))
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_: object())
    monkeypatch.setattr(runner, "tracker_for", lambda *_a, **_k: tracker)
    return engine, tracker


def test_command_runs_only_bfcl_through_vllm_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, tracker = _wire(monkeypatch, tmp_path)
    seen: list[str] = []

    def evaluate(*_args: Any, **kwargs: Any) -> BfclRun:
        seen.append(kwargs["benchmark_revision"])
        return BfclRun(METRICS, {"benchmark": "BFCL"})

    monkeypatch.setattr(runner, "evaluate_bfcl", evaluate)
    config = EvalConfig(output_dir=str(tmp_path))

    assert runner.run_evaluation(config, _args(tmp_path)) == 0

    assert seen == ["b" * 40]
    assert engine.shutdowns == 1
    assert tracker.finished
    assert (tmp_path / "base.json").is_file()
    assert (tmp_path / "base.md").is_file()
    assert json.loads(capsys.readouterr().out) == {
        "json": str(tmp_path / "base.json"),
        "markdown": str(tmp_path / "base.md"),
    }
    assert tracker.artifacts[0][1][-1].name == "base-bfcl.jsonl"


def test_engine_is_released_when_bfcl_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, tracker = _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "evaluate_bfcl",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        runner.run_evaluation(EvalConfig(output_dir=str(tmp_path)), _args(tmp_path))

    assert engine.shutdowns == 1
    assert tracker.finished


def test_endpoint_mode_is_not_part_of_checkpoint_evaluation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoints only"):
        runner.run_evaluation(EvalConfig(), _args(tmp_path, endpoint="http://localhost:8000/v1"))


def test_vllm_generator_batches_ids_and_uses_resolved_sampling() -> None:
    tokenizer = OfflineTokenizer(token_size=1)
    output_ids = tokenizer("answer")["input_ids"]
    seen: list[dict[str, Any]] = []

    class Engine:
        def generate_ids(self, prompts: Any, **kwargs: Any) -> list[Any]:
            seen.append({"prompts": prompts, **kwargs})
            return [SimpleNamespace(token_ids=output_ids, finish_reason="stop") for _ in prompts]

    config = EvalConfig()
    generate = runner._vllm_generator(Engine(), tokenizer, config, None)
    results = generate([BfclRequest("a", (1, 2), 20), BfclRequest("b", (3,), 20)])

    assert [result.text for result in results] == ["answer", "answer"]
    assert seen[0]["prompts"] == [(1, 2), (3,)]
    assert seen[0]["temperature"] == config.decoding.temperature
    assert seen[0]["top_p"] == config.decoding.top_p
    assert seen[0]["top_k"] == config.decoding.top_k
    assert seen[0]["presence_penalty"] == config.decoding.presence_penalty
