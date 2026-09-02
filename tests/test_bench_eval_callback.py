"""In-training dev eval: fires where expected, fails safely, scores the dev set.

Neither training stage reports a benchmark number, so the callback is the only path
from a training run to a held-out score. Four properties decide whether that number
is worth having, and none of them is visible in the number itself:

- it fires at the boundaries it claims to, including a step-0 anchor;
- it scores the **dev** adapter, never the test benchmark;
- the subset does not move between evals, or the curve mixes policy change with
  sample change;
- a failure logs and releases environments rather than ending the run -- and rather
  than leaking every episode into a `PoolError` at the next boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config_models import BenchEvalConfig, EvalConfig
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.training.bench_eval import (
    BenchEvalCallback,
    BenchEvalError,
    BenchEvalRunner,
    assert_dev_adapter,
    dev_subset,
    should_evaluate,
)


class Adapter:
    """A dev adapter with a deterministic task order and a closable pool."""

    instances: list[Adapter] = []

    def __init__(self, *, task_count: int = 40, fail: bool = False) -> None:
        self.task_count = task_count
        self.fail = fail
        self.closed = 0
        self.live: set[str] = set()
        Adapter.instances.append(self)

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask(f"task-{index:03d}", "envscaler_heldout", "do it", ())
            for index in range(self.task_count)
        ]

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        self.live.add(task.task_id)
        if self.fail:
            raise RuntimeError("environment create failed")
        if history:
            return [dict(message) for message in history]
        return [{"role": "system", "content": "s"}, {"role": "user", "content": task.prompt}]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        return StepResult("done", complete=True, env_steps=1)

    def score(self, task: EvalTask) -> AdapterResult:
        return AdapterResult(1.0, True, diagnostics={"check_total": 4.0})

    def invalid_call_count(self, task: EvalTask) -> int:
        return 0

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        return {"held_out": [task.task_id for task in tasks]}

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        return aggregate(tasks)

    def close(self) -> None:
        self.closed += 1
        self.live.clear()


@pytest.fixture(autouse=True)
def _reset_adapters() -> Any:
    Adapter.instances.clear()
    yield
    Adapter.instances.clear()


def make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bench: BenchEvalConfig | None = None,
    fail: bool = False,
    sink: list[Mapping[str, float]] | None = None,
    versions: list[str] | None = None,
    artifact_dir: str | None = None,
) -> BenchEvalRunner:
    """A runner whose adapter and generation are both fakes.

    The scored path is the real `evaluate_batched`, so this exercises the shared
    engine and the shared aggregation -- there is no second scoring implementation to
    test against.
    """
    from tests.helpers import OfflineTokenizer
    from tests.test_eval_batched_agreement import scripted_backend

    monkeypatch.setattr(
        "smolqwen.training.bench_eval.create_adapter",
        lambda name, config: Adapter(fail=fail),
    )
    tokenizer = OfflineTokenizer(token_size=1)
    version_log = versions if versions is not None else []
    return BenchEvalRunner(
        eval_config=EvalConfig(adapters=("envscaler_heldout",), max_steps_per_task=6),
        bench_config=bench or BenchEvalConfig(enabled=True, task_limit=4),
        engine_source=lambda: scripted_backend(tokenizer),
        tokenizer_source=lambda: tokenizer,
        metric_prefix="grpo",
        sink=(lambda payload: sink.append(dict(payload))) if sink is not None else None,
        weight_version=lambda: _next_version(version_log),
        artifact_dir=artifact_dir,
    )


def _next_version(log: list[str]) -> str:
    log.append(f"step-{len(log)}.sync-{len(log) + 1}")
    return log[-1]


def test_the_test_benchmark_is_refused_at_the_boundary() -> None:
    """A benchmark used to select checkpoints is a dev set. Scoring BFCL here would
    void the final Base | SFT | SFT+RL comparison, and nothing in the resulting
    number would show it."""
    for name in ("bfcl_multi_turn", "BFCL", "gorilla_v3", "berkeley_fc"):
        with pytest.raises(BenchEvalError, match="test benchmark"):
            assert_dev_adapter(name)

    assert assert_dev_adapter("envscaler_heldout") == "envscaler_heldout"


def test_the_runner_refuses_a_test_adapter_at_construction() -> None:
    with pytest.raises(BenchEvalError, match="test benchmark"):
        BenchEvalRunner(
            eval_config=EvalConfig(),
            bench_config=BenchEvalConfig(enabled=True, adapter="bfcl_multi_turn"),
            engine_source=lambda: None,
            tokenizer_source=lambda: None,
            metric_prefix="grpo",
        )


def test_the_subset_is_identical_across_evals() -> None:
    """A subset that moves between evals produces a curve mixing policy change with
    sample change, and the mixture is not separable after the fact."""
    first = dev_subset(Adapter(), task_limit=8)
    second = dev_subset(Adapter(), task_limit=8)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert len(first) == 8

    # A prefix of the adapter's own deterministic order, not a sample of it.
    assert [task.task_id for task in first] == [f"task-{index:03d}" for index in range(8)]


def test_a_successful_eval_logs_scores_with_its_weight_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The weight version is what makes the later comparison against `evaluate`
    falsifiable. At a callback boundary the engine can hold weights up to
    `grad_accum` optimizer steps old, and that drift is invisible in the score."""
    sink: list[Mapping[str, float]] = []
    runner = make_runner(monkeypatch, sink=sink)

    outcome = runner.run(step=40)

    assert outcome.failed_reason is None
    assert outcome.task_count == 4
    assert outcome.weight_version == "step-0.sync-1"
    assert outcome.metrics["envscaler_heldout_score"] == 1.0
    assert outcome.wall_s >= 0.0

    payload = sink[0]
    assert payload["grpo/bench_envscaler_heldout_score"] == 1.0
    assert payload["grpo/bench_failed"] == 0.0
    assert payload["grpo/bench_task_count"] == 4.0
    # The cost per boundary, so the 10% budget is checked against a measurement.
    assert "grpo/bench_wall_s" in payload


def test_a_failed_eval_logs_and_releases_environments_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eval failure must not end a training run, and must not leave episodes live.

    Leaking them turns one recoverable failure into a pool at capacity for every
    later boundary -- a symptom one boundary removed from its cause.
    """
    sink: list[Mapping[str, float]] = []
    runner = make_runner(monkeypatch, fail=True, sink=sink)

    outcome = runner.run(step=20)

    assert outcome.failed_reason is not None
    assert "environment create failed" in outcome.failed_reason
    assert outcome.metrics == {}
    assert sink[0]["grpo/bench_failed"] == 1.0

    adapter = Adapter.instances[-1]
    assert adapter.closed == 1
    assert not adapter.live, f"episodes leaked: {sorted(adapter.live)}"


def test_every_boundary_records_its_cost_so_the_budget_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)

    for step in (0, 20, 40):
        runner.run(step=step)

    assert [outcome.step for outcome in runner.outcomes] == [0, 20, 40]
    assert all(outcome.wall_s >= 0.0 for outcome in runner.outcomes)
    # Distinct versions: two evals at the same step must be distinguishable.
    assert len({outcome.weight_version for outcome in runner.outcomes}) == 3


def test_records_land_beside_the_checkpoint_they_were_scored_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """End-of-run checkpoint selection then reads recorded numbers rather than
    scraping a dashboard."""
    from smolqwen.eval.trajectories import read_trajectories, trajectory_path

    runner = make_runner(monkeypatch, artifact_dir=str(tmp_path))
    runner.run(step=60)

    path = trajectory_path(tmp_path, tag="step-60", adapter="envscaler_heldout")
    assert path.is_file()
    rows = read_trajectories(path)
    assert len(rows) == 4
    assert all(row["score"] == 1.0 for row in rows)


def test_boundaries_fire_where_the_config_says(monkeypatch: pytest.MonkeyPatch) -> None:
    disabled = BenchEvalConfig(enabled=False, every_steps=10)
    assert not should_evaluate(disabled, step=10, at_save=True)

    # `every_steps == 0` is save-boundaries-only, the cadence that guarantees a
    # checkpoint exists to attribute the score to.
    saves_only = BenchEvalConfig(enabled=True, every_steps=0)
    assert should_evaluate(saves_only, step=37, at_save=True)
    assert not should_evaluate(saves_only, step=37, at_save=False)

    interval = BenchEvalConfig(enabled=True, every_steps=10)
    assert should_evaluate(interval, step=20, at_save=False)
    assert not should_evaluate(interval, step=25, at_save=False)
    # Step 0 is the baseline's job, not the interval's; firing both would double.
    assert not should_evaluate(interval, step=0, at_save=False)


def test_the_callback_anchors_the_curve_at_step_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A curve whose first point is at step 100 cannot separate "learned nothing"
    from "started there"."""
    runner = make_runner(monkeypatch)
    synced: list[int] = []
    callback = BenchEvalCallback(runner, before_each=lambda: synced.append(1))
    state = SimpleNamespace(global_step=0)

    callback.on_train_begin(None, state, SimpleNamespace())

    assert [outcome.step for outcome in runner.outcomes] == [0]
    # The explicit sync happens before every eval, including the baseline.
    assert synced == [1]


def test_the_callback_skips_every_boundary_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch, bench=BenchEvalConfig(enabled=False))
    callback = BenchEvalCallback(runner)

    callback.on_train_begin(None, SimpleNamespace(global_step=0), SimpleNamespace())
    callback.on_step_end(None, SimpleNamespace(global_step=20), SimpleNamespace())
    callback.on_save(None, SimpleNamespace(global_step=20), SimpleNamespace())

    assert runner.outcomes == []


def test_a_save_boundary_evaluates_and_syncs_first(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = make_runner(
        monkeypatch, bench=BenchEvalConfig(enabled=True, task_limit=2, baseline_at_step_zero=False)
    )
    order: list[str] = []
    callback = BenchEvalCallback(runner, before_each=lambda: order.append("sync"))

    callback.on_train_begin(None, SimpleNamespace(global_step=0), SimpleNamespace())
    assert runner.outcomes == [], "baseline was disabled but still ran"

    callback.on_save(None, SimpleNamespace(global_step=20), SimpleNamespace())
    assert [outcome.step for outcome in runner.outcomes] == [20]
    assert order == ["sync"]
