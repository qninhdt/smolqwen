"""SFT's eval boundary: the right checkpoint, a fresh adapter id, engine asleep.

Three failures this file exists to catch, none of which is visible in the resulting
number:

- **Scoring the wrong directory.** `checkpoint-N` exists only at a save boundary. An
  eval that fell back to the previous checkpoint would attribute one step's weights
  to another step's score.
- **A reused LoRA id.** vLLM caches adapter weights by integer id, and
  `OfflineEngine.load_adapter` derives that id from how many adapters it has
  registered. Reusing one name across boundaries hands vLLM the same id with a new
  path, so every later boundary would serve the first checkpoint's weights -- a flat
  curve made of plausible numbers.
- **An engine left awake.** `sleep()` runs in the callback's `finally`, so a failed
  boundary does not leave VRAM held through the next training step.

Construction order is asserted separately: `gpu_memory_utilization` sizes the KV pool
against *total* GPU memory, so the engine must exist and be asleep before the trainer
does. That is checked here by call order against a fake engine; the VRAM arithmetic
it protects is a GPU measurement and lives in
`test_sft_bench_eval_memory_guard.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import BenchEvalConfig, SftConfig
from smolqwen.training.bench_eval import BenchEvalCallback, BenchEvalRunner
from smolqwen.training.checkpoint_eval import (
    CheckpointEngine,
    CheckpointEvalError,
    build_bench_eval_callback,
    eval_config_for,
)


class FakeEngine:
    """`OfflineEngine`'s surface, recording lifecycle order and adapter ids."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.adapters: list[tuple[str, str, int]] = []
        self.asleep = False

    def build(self) -> None:
        self.events.append("build")

    def sleep(self, level: int = 1) -> None:
        self.events.append("sleep")
        self.asleep = True

    def wake_up(self) -> None:
        self.events.append("wake")
        self.asleep = False

    def load_adapter(self, name: str, path: str) -> Any:
        # The id vLLM would assign: derived from how many are already registered.
        self.adapters.append((name, path, len(self.adapters) + 1))
        return SimpleNamespace(lora_name=name, lora_int_id=len(self.adapters), lora_path=path)

    def generate_ids(self, prompt_ids: Any, **_: Any) -> list[Any]:
        raise AssertionError("this test never generates")

    def shutdown(self) -> None:
        self.events.append("shutdown")


def _config(tmp_path: Path, **bench: Any) -> SftConfig:
    base = resolve("sft", profile="l4")
    assert isinstance(base, SftConfig)
    payload: dict[str, Any] = {"enabled": True, "task_limit": 2}
    payload.update(bench)
    return base.model_copy(
        update={"output_dir": str(tmp_path), "bench_eval": BenchEvalConfig(**payload)}
    )


def _checkpoint(tmp_path: Path, step: int) -> Path:
    directory = tmp_path / f"checkpoint-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    return directory


def test_the_engine_is_built_and_asleep_before_anything_else(tmp_path: Path) -> None:
    """Built after a resident trainer, it reserves against memory it cannot honour."""
    engine = FakeEngine()
    holder = CheckpointEngine.build(_config(tmp_path), engine=engine)  # type: ignore[arg-type]
    assert engine.events == ["build", "sleep"]
    assert engine.asleep
    assert holder.step == 0


def test_the_scored_directory_is_the_boundary_s_own_checkpoint(tmp_path: Path) -> None:
    engine = FakeEngine()
    holder = CheckpointEngine.build(_config(tmp_path), engine=engine)  # type: ignore[arg-type]
    _checkpoint(tmp_path, 100)
    _checkpoint(tmp_path, 200)

    holder.prepare(100)
    holder.backend()
    holder.release(100)
    holder.prepare(200)
    holder.backend()
    holder.release(200)

    assert [name for name, _, _ in engine.adapters] == ["checkpoint-100", "checkpoint-200"]
    assert [path for _, path, _ in engine.adapters] == [
        str(tmp_path / "checkpoint-100"),
        str(tmp_path / "checkpoint-200"),
    ]
    # Distinct ids. A reused id with a new path serves the older weights, and every
    # later score would silently be the first checkpoint's.
    assert [identifier for _, _, identifier in engine.adapters] == [1, 2]


def test_a_missing_checkpoint_raises_rather_than_scoring_the_previous_one(
    tmp_path: Path,
) -> None:
    engine = FakeEngine()
    holder = CheckpointEngine.build(_config(tmp_path), engine=engine)  # type: ignore[arg-type]
    _checkpoint(tmp_path, 100)
    holder.prepare(100)
    holder.backend()

    holder.prepare(200)
    with pytest.raises(CheckpointEvalError, match="checkpoint-200"):
        holder.backend()
    # The one that does exist was not silently substituted.
    assert [name for name, _, _ in engine.adapters] == ["checkpoint-100"]


def test_the_step_zero_anchor_scores_the_base_model_with_no_adapter(tmp_path: Path) -> None:
    """TRL has saved nothing at step 0, and the base model is the right anchor.

    It is also the `base` arm of the final Base | SFT | SFT+RL table, so the anchor is
    a number that already means something rather than an artifact of callback timing.
    """
    engine = FakeEngine()
    holder = CheckpointEngine.build(_config(tmp_path), engine=engine)  # type: ignore[arg-type]
    holder.prepare(0)
    backend = holder.backend()
    assert backend.adapter is None
    assert engine.adapters == []


def test_the_engine_sleeps_after_every_boundary_including_a_failed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine left awake holds VRAM the next training step needs."""
    engine = FakeEngine()
    config = _config(tmp_path, baseline_at_step_zero=False)
    holder = CheckpointEngine.build(config, engine=engine)  # type: ignore[arg-type]
    runner = BenchEvalRunner(
        eval_config=eval_config_for(config),
        bench_config=config.bench_eval,
        engine_source=holder.backend,
        tokenizer_source=lambda: None,
        metric_prefix="sft",
        weight_version=lambda: f"checkpoint-{holder.step}",
    )
    callback = BenchEvalCallback(runner, before_each=holder.prepare, after_each=holder.release)

    # No checkpoint on disk, so the boundary fails inside the runner.
    callback.on_save(None, SimpleNamespace(global_step=100), SimpleNamespace())

    assert engine.events[-2:] == ["wake", "sleep"]
    assert engine.asleep
    outcome = runner.outcomes[-1]
    assert outcome.failed_reason is not None
    assert "checkpoint-100" in outcome.failed_reason


def test_the_eval_config_is_resized_by_the_sft_run_s_own_profile(tmp_path: Path) -> None:
    """`resolve("eval")` takes no `--profile`, so it would carry defaults.

    An engine sized at the default KV fraction while the trainer was sized by
    `--profile l4` is two sets of numbers for one card.
    """
    config = _config(tmp_path)
    resolved = eval_config_for(config)
    assert resolved.profile == config.profile
    assert resolved.adapters, "the eval stage config should still name its adapters"


def test_the_callback_names_the_checkpoint_as_its_weight_version(tmp_path: Path) -> None:
    """`sft/bench_*` rows must say which checkpoint produced them.

    GRPO records `step-N.sync-M` because its engine can hold stale weights. SFT's
    weights are a directory, so the directory is the honest answer -- and it makes a
    later comparison against `evaluate --checkpoint checkpoint-N` falsifiable.
    """
    engine = FakeEngine()
    config = _config(tmp_path)
    holder = CheckpointEngine.build(config, engine=engine)  # type: ignore[arg-type]
    logged: list[dict[str, Any]] = []
    trainer = SimpleNamespace(log=lambda payload: logged.append(dict(payload)))
    callback = build_bench_eval_callback(config, trainer, None, engine=holder)

    assert callback.runner.metric_prefix == "sft"
    holder.prepare(300)
    assert callback.runner._weight_version() == "checkpoint-300"
