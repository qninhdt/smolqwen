"""In-training held-out benchmark eval, shared by both training stages.

Neither stage reports a benchmark number today. SFT computes teacher-forced
validation loss; GRPO logs verifier reward over curriculum-weighted *training*
scenarios, a biased sample by construction. So nothing measures held-out capability
until a run finishes and someone invokes `evaluate` by hand.

The callback is deliberately thin. It owns no benchmark logic, no scoring, and no
turn loop -- it calls the same engine and the same aggregation `smolqwen evaluate`
calls, which is the whole point: a number that means something different in
training than in `evaluate` is worse than no number.

Four things it does own:

**The weight version.** At `on_step_end` the optimizer step has already been
applied, so a colocated engine holds pre-step weights -- and generation happens once
per accumulation window, so the engine can be up to `grad_accum` steps stale. An
8-step drift is invisible in the number and would look exactly like a parity bug
when `bench_*` is later compared against `evaluate` "at the same revision". So the
weight version is logged as a field on every row, and the sync is explicit.

**The dev set, named.** `eval.yaml` lists both `bfcl_multi_turn` and
`envscaler_heldout`. A callback iterating that list would score BFCL every
boundary, and a benchmark used to pick checkpoints is a dev set -- which would void
the final Base | SFT | SFT+RL table. The adapter is named in config and asserted
here.

**A stable subset.** A subset that varies between evals produces a curve mixing
policy change with sample change. The adapter's own selection is deterministic, so
taking a prefix of it is stable across calls without this file owning selection.

**Failure isolation.** The body runs under a broad handler that releases
environments. Without that, one recoverable failure leaves every episode live and
turns into a `PoolError` at every later boundary -- a symptom one full boundary
removed from its cause.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from smolqwen.config_models import BenchEvalConfig, EvalConfig
from smolqwen.eval.adapters import create_adapter
from smolqwen.eval.adapters.base import BenchmarkAdapter, EvalTask
from smolqwen.eval.batched import evaluate_batched
from smolqwen.eval.trajectories import TrajectoryRecord, write_trajectories

# Benchmark names that must never be reachable from an in-training path. Spelled
# the way a leak would spell them, matching `test_dev_test_integrity.py`.
TEST_BENCHMARK_TOKENS = ("bfcl", "gorilla", "berkeley")

MetricSink = Callable[[Mapping[str, float]], None]


class BenchEvalError(RuntimeError):
    """Raised when in-training eval is configured incoherently. Never at runtime."""


@dataclass
class BenchEvalOutcome:
    """One boundary's result, so the cost is a recorded number not an impression."""

    step: int
    weight_version: str
    metrics: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    failed_reason: str | None = None
    task_count: int = 0


def assert_dev_adapter(name: str) -> str:
    """Refuse a test-set adapter at the in-training boundary.

    The check is here rather than only in config validation because this is the one
    place a name becomes a scored benchmark. A run that scored BFCL every hundred
    steps would produce a plausible-looking curve and an invalid final comparison.
    """
    lowered = name.casefold()
    for token in TEST_BENCHMARK_TOKENS:
        if token in lowered:
            raise BenchEvalError(
                f"{name!r} is the test benchmark; in-training eval scores the dev set "
                "only. A benchmark used to select checkpoints is a dev set, and using "
                "the test set here voids the final Base | SFT | SFT+RL comparison."
            )
    return name


def dev_subset(adapter: BenchmarkAdapter, *, task_limit: int) -> list[EvalTask]:
    """A stable prefix of the adapter's own deterministic selection.

    Prefix rather than sample: the adapter already orders its held-out slice
    deterministically and records the exact ids in its manifest contribution, so a
    prefix is reproducible without this module owning selection. Sampling per call
    would mix policy change with sample change in the resulting curve.
    """
    return list(adapter.load_tasks())[:task_limit]


class BenchEvalRunner:
    """Score a fixed dev subset through the shared engine, at a training boundary.

    `engine_source()` returns the generation backend for this boundary. GRPO passes
    the colocated engine after syncing weights; a checkpoint-based caller passes an
    offline engine pointed at what was just written. The runner does not know which,
    which is what keeps one scoring path.
    """

    def __init__(
        self,
        *,
        eval_config: EvalConfig,
        bench_config: BenchEvalConfig,
        engine_source: Callable[[], Any],
        tokenizer_source: Callable[[], Any],
        metric_prefix: str,
        sink: MetricSink | None = None,
        weight_version: Callable[[], str] | None = None,
        artifact_dir: str | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.bench_config = bench_config
        self.adapter_name = assert_dev_adapter(bench_config.adapter)
        self._engine_source = engine_source
        self._tokenizer_source = tokenizer_source
        self.metric_prefix = metric_prefix
        self._sink = sink
        self._weight_version = weight_version or (lambda: "unknown")
        self._artifact_dir = artifact_dir
        self.outcomes: list[BenchEvalOutcome] = []

    def run(self, step: int) -> BenchEvalOutcome:
        """Score once. Never raises: a failed eval must not end a training run.

        Cost is recorded per boundary in `outcomes`, so the budget is verified
        against measurements rather than asserted in a config comment.
        """
        version = self._weight_version()
        started = time.monotonic()
        adapter: BenchmarkAdapter | None = None
        try:
            adapter = create_adapter(self.adapter_name, self.eval_config)
            tasks = dev_subset(adapter, task_limit=self.bench_config.task_limit)
            records: list[TrajectoryRecord] = []
            metrics = evaluate_batched(
                self.eval_config,
                adapter,
                backend=self._engine_source(),
                tokenizer=self._tokenizer_source(),
                tasks=tasks,
                records=records,
            )
            outcome = BenchEvalOutcome(
                step=step,
                weight_version=version,
                metrics=self._flatten(metrics),
                wall_s=time.monotonic() - started,
                task_count=len(tasks),
            )
            self._write_records(step, records)
        except Exception as exc:
            outcome = BenchEvalOutcome(
                step=step,
                weight_version=version,
                wall_s=time.monotonic() - started,
                failed_reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # Releasing the adapter's environments is the difference between one
            # recoverable failure and a pool at capacity for the rest of the run.
            if adapter is not None:
                close = getattr(adapter, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        self.outcomes.append(outcome)
        if self._sink is not None:
            self._sink(self.log_payload(outcome))
        return outcome

    def log_payload(self, outcome: BenchEvalOutcome) -> dict[str, float]:
        """The metric row for this boundary, prefixed for the training stage.

        `bench_weight_version` and `bench_wall_s` ride alongside every score: the
        first makes the later comparison against `evaluate` falsifiable, the second
        makes the 10% budget checkable.
        """
        payload: dict[str, float] = {
            f"{self.metric_prefix}/bench_wall_s": outcome.wall_s,
            f"{self.metric_prefix}/bench_task_count": float(outcome.task_count),
            f"{self.metric_prefix}/bench_failed": float(outcome.failed_reason is not None),
        }
        payload.update(
            {f"{self.metric_prefix}/bench_{name}": value for name, value in outcome.metrics.items()}
        )
        return payload

    def _flatten(self, metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        return {
            f"{category}_{name}": float(value)
            for category, values in sorted(metrics.items())
            for name, value in sorted(values.items())
        }

    def _write_records(self, step: int, records: Sequence[TrajectoryRecord]) -> None:
        """Keep the trajectories beside the checkpoint they were scored at.

        End-of-run checkpoint selection then reads recorded numbers rather than
        scraping a dashboard, and a surprising score is inspectable after the fact.
        """
        if self._artifact_dir is None or not records:
            return
        try:
            write_trajectories(
                self._artifact_dir,
                tag=f"step-{step}",
                adapter=self.adapter_name,
                records=records,
            )
        except OSError:
            # Losing a record file must not end training; the metrics already logged.
            return


def should_evaluate(config: BenchEvalConfig, *, step: int, at_save: bool) -> bool:
    """Whether this boundary is an eval boundary.

    `every_steps == 0` means save boundaries only, which is the cheapest useful
    cadence and the one that guarantees a checkpoint exists to attribute the score
    to.
    """
    if not config.enabled:
        return False
    if at_save:
        return True
    if config.every_steps == 0:
        return False
    return step > 0 and step % config.every_steps == 0


class BenchEvalCallback:
    """`TrainerCallback` shape: baseline before training, then at each boundary.

    Not subclassing `TrainerCallback` here, because that would import transformers
    at module scope and this module is imported by config-only paths. The trainer
    accepts any object with the hook methods.
    """

    def __init__(
        self,
        runner: BenchEvalRunner,
        *,
        before_each: Callable[[], None] | None = None,
    ) -> None:
        self.runner = runner
        # GRPO's explicit `sync_weights()` goes here. It is a seam rather than a
        # branch because a checkpoint-based caller has nothing to sync -- the weights
        # it scores are already on disk.
        self._before_each = before_each

    @property
    def config(self) -> BenchEvalConfig:
        return self.runner.bench_config

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """The step-0 anchor. A curve whose first point is at step 100 cannot show
        early movement, and cannot separate "learned nothing" from "started there"."""
        if self.config.enabled and self.config.baseline_at_step_zero:
            self._evaluate(0)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0))
        if should_evaluate(self.config, step=step, at_save=False):
            self._evaluate(step)
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0))
        if should_evaluate(self.config, step=step, at_save=True):
            self._evaluate(step)
        return control

    def _evaluate(self, step: int) -> None:
        if self._before_each is not None:
            self._before_each()
        self.runner.run(step)
