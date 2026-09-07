"""In-training benchmark eval, run by GRPO.

GRPO logs verifier reward over uniformly sampled *training* scenarios. The bench
callback adds a fixed dev slice -- BFCL
multi-turn base in the current config -- scored through the same engine and the
same aggregation `smolqwen evaluate` calls, which is the whole point: a number that
means something different in training than in `evaluate` is worse than no number.
Dev and test coincide in this lab's design; the benchmark is named in config
rather than derived from `eval.yaml`'s adapter list.

The callback is deliberately thin. It owns no benchmark logic, no scoring, and no
turn loop.

Four things it does own:

**The weight version.** At `on_step_end` the optimizer step has already been
applied, so a colocated engine holds pre-step weights -- and generation happens once
per accumulation window, so the engine can be up to `grad_accum` steps stale. An
8-step drift is invisible in the number and would look exactly like a parity bug
when `bench_*` is later compared against `evaluate` "at the same revision". So the
weight version is logged as a field on every row, and the sync is explicit.

**A named adapter.** `bench_config.adapter` names the benchmark directly; the
callback never iterates `eval.yaml`'s adapter list.

**A stable subset.** A subset that varies between evals produces a curve mixing
policy change with sample change. The adapter's own selection is deterministic, so
taking a prefix of it is stable across calls without this file owning selection.

**Failure isolation.** The body runs under a broad handler that releases
environments. Without that, one recoverable failure leaves every episode live and
turns into a `PoolError` at every later boundary -- a symptom one full boundary
removed from its cause.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolqwen.config_models import BenchEvalConfig, EvalConfig
from smolqwen.console import logger
from smolqwen.eval.adapters import create_adapter
from smolqwen.eval.adapters.base import BenchmarkAdapter, EvalTask
from smolqwen.eval.batched import evaluate_batched
from smolqwen.eval.trajectories import TrajectoryRecord, write_trajectories

LOG = logger(__name__)

MetricValue = float | str
MetricSink = Callable[[Mapping[str, MetricValue]], None]


class BenchEvalTimeout(TimeoutError):
    """Raised internally when a training-time benchmark exceeds its wall budget."""


@contextmanager
def _wall_clock_timeout(timeout_s: float) -> Iterator[None]:
    """Interrupt a synchronous eval so a hung boundary cannot hang training.

    Training callbacks run on the main thread in the supported runtime, which is the
    only place Python can install a process signal handler. The environment worker
    pool has its own per-call timeout in child processes; this outer timeout covers
    adapter setup, generation, and aggregation as one training boundary.
    """
    if threading.current_thread() is not threading.main_thread():
        raise BenchEvalTimeout("benchmark eval timeout requires the main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    timer_started = time.monotonic()

    def _restore_alarm() -> None:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0.0 or previous_interval > 0.0:
            # Preserve a caller's alarm relative to the time it had remaining when
            # this context started. A tiny positive delay preserves an alarm that
            # elapsed while this context owned SIGALRM instead of silently dropping
            # it; nested timeout contexts therefore unwind in the right order.
            remaining = max(previous_delay - (time.monotonic() - timer_started), 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)

    def _alarm(_signum: int, _frame: Any) -> None:
        raise BenchEvalTimeout(f"benchmark eval exceeded {timeout_s:g}s")

    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    except BaseException:
        # If installing the handler/timer itself fails, restore the caller's signal
        # state before allowing the setup error to be recorded by the runner.
        _restore_alarm()
        raise
    try:
        yield
    finally:
        _restore_alarm()


@dataclass
class BenchEvalOutcome:
    """One boundary's result, so the cost is a recorded number not an impression."""

    step: int
    weight_version: str
    metrics: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    failed_reason: str | None = None
    task_count: int = 0


def dev_subset(adapter: BenchmarkAdapter, *, task_limit: int) -> list[EvalTask]:
    """A stable prefix of the adapter's own deterministic selection.

    Prefix rather than sample: the adapter already orders its task list
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
        outcome_path: Callable[[int], Path | str] | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.bench_config = bench_config
        self.adapter_name = bench_config.adapter
        self._engine_source = engine_source
        self._tokenizer_source = tokenizer_source
        self.metric_prefix = metric_prefix
        self._sink = sink
        self._weight_version = weight_version or (lambda: "unknown")
        self._artifact_dir = artifact_dir
        self._outcome_path = outcome_path
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
            with _wall_clock_timeout(self.bench_config.timeout_s):
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
                    label=self.metric_prefix,
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

        self._write_outcome(outcome)
        self.outcomes.append(outcome)
        if self._sink is not None:
            try:
                self._sink(self.log_payload(outcome))
            except Exception as exc:
                # Metric transport is observability, not part of the training
                # invariant. A transient Trainer/W&B logging failure must not turn
                # an otherwise completed eval boundary into a failed training run.
                LOG.warning(
                    "%s bench eval metrics could not be logged at step %d: %s: %s",
                    self.metric_prefix,
                    step,
                    type(exc).__name__,
                    exc,
                )
        if outcome.failed_reason is not None:
            # Never raised, so it must be visible: a silently failing eval otherwise
            # shows up only as `bench_failed=1` in a dashboard nobody is watching.
            LOG.warning(
                "%s bench eval failed at step %d after %.1fs: %s",
                self.metric_prefix,
                step,
                outcome.wall_s,
                outcome.failed_reason,
            )
        else:
            LOG.info(
                "%s bench eval step %d (%s): %d tasks in %.1fs -- %s",
                self.metric_prefix,
                step,
                outcome.weight_version,
                outcome.task_count,
                outcome.wall_s,
                ", ".join(f"{name}={value:.4g}" for name, value in sorted(outcome.metrics.items()))
                or "no metrics",
            )
        return outcome

    def log_payload(self, outcome: BenchEvalOutcome) -> dict[str, MetricValue]:
        """The metric row for this boundary, prefixed for the training stage.

        `bench_weight_version` and `bench_wall_s` ride alongside every score: the
        first makes the later comparison against `evaluate` falsifiable, the second
        makes the 10% budget checkable.
        """
        payload: dict[str, MetricValue] = {
            f"{self.metric_prefix}/bench_wall_s": outcome.wall_s,
            f"{self.metric_prefix}/bench_task_count": float(outcome.task_count),
            f"{self.metric_prefix}/bench_failed": float(outcome.failed_reason is not None),
            f"{self.metric_prefix}/bench_weight_version": outcome.weight_version,
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

    def _write_outcome(self, outcome: BenchEvalOutcome) -> None:
        """Persist the boundary result when the caller owns a checkpoint sidecar."""
        if self._outcome_path is None:
            return
        try:
            path = Path(self._outcome_path(outcome.step))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "step": outcome.step,
                        "weight_version": outcome.weight_version,
                        "metrics": outcome.metrics,
                        "wall_s": outcome.wall_s,
                        "failed_reason": outcome.failed_reason,
                        "task_count": outcome.task_count,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            # A sidecar is evidence, not a reason to kill a training run whose metric
            # sink already received the boundary result.
            LOG.warning(
                "%s bench eval sidecar could not be written at step %d: %s",
                self.metric_prefix,
                outcome.step,
                exc,
            )


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
        before_each: Callable[[int], None] | None = None,
        after_each: Callable[[int], None] | None = None,
    ) -> None:
        self.runner = runner
        # GRPO's explicit `sync_weights()` goes here; SFT's is "which checkpoint
        # directory does this boundary score". A seam rather than a branch, because
        # the two stages differ only in where the weights come from.
        self._before_each = before_each
        # SFT's `sleep()`. It runs in a `finally`, because an engine left awake after
        # a failed eval holds VRAM the next training step needs -- which would turn
        # one recoverable failure into an OOM one step later.
        self._after_each = after_each
        # Trainer emits both hooks when an interval and save boundary coincide. Keep
        # one successful result for that step, but leave failed boundaries retryable:
        # SFT's interval hook can run before `checkpoint-N` is written.
        self._completed_steps: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        """Supply no-op lifecycle hooks expected by Transformers' handler.

        This class intentionally does not import ``transformers`` at module
        scope. The handler invokes every ``on_*`` event on every callback, so
        unowned events must still return the current control object.
        """
        if not name.startswith("on_"):
            raise AttributeError(name)

        def no_op(*args: Any, **kwargs: Any) -> Any:
            return kwargs.get("control", args[2] if len(args) > 2 else None)

        return no_op

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
        if step in self._completed_steps:
            return
        completed = False
        try:
            if self._before_each is not None:
                self._before_each(step)
            outcome = self.runner.run(step)
            completed = outcome.failed_reason is None
        finally:
            if self._after_each is not None:
                self._after_each(step)
        if completed:
            self._completed_steps.add(step)
