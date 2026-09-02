"""W&B run handling plus the throughput/VRAM callback every stage shares.

Nothing here may require a token. An absent `WANDB_API_KEY` degrades to a
disabled run so CPU-only CI and offline Colab sessions still execute the same
code path -- a tracking layer that crashes without credentials would push every
test onto a second, untested branch.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from smolqwen.console import logger

LOG = logger(__name__)


class Run(Protocol):
    """The slice of a W&B run this module uses."""

    def log(self, data: Mapping[str, Any], *, step: int | None = ...) -> Any: ...

    def log_artifact(self, artifact_or_path: Any, **kwargs: Any) -> Any: ...

    def finish(self) -> Any: ...


@dataclass
class GpuMemorySnapshot:
    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    max_allocated_gb: float = 0.0


def read_gpu_memory() -> GpuMemorySnapshot:
    """Current CUDA allocation in GB, or zeros on CPU. Never raises."""
    try:
        import torch
    except ImportError:
        return GpuMemorySnapshot()
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        return GpuMemorySnapshot()
    scale = 1024**3
    return GpuMemorySnapshot(
        allocated_gb=torch.cuda.memory_allocated() / scale,
        reserved_gb=torch.cuda.memory_reserved() / scale,
        max_allocated_gb=torch.cuda.max_memory_allocated() / scale,
    )


@dataclass
class ThroughputMeter:
    """Step time and tokens/s, computed from what the caller reports.

    Deliberately not a TRL subclass: Phase 6's rollout loop and Phase 8's bench
    are not `Trainer`s, and the same numbers have to be comparable across all
    three.
    """

    step_times: list[float] = field(default_factory=list)
    total_tokens: int = 0
    _step_start: float | None = None

    def start_step(self) -> None:
        self._step_start = time.perf_counter()

    def end_step(self, tokens: int = 0) -> dict[str, float]:
        if self._step_start is None:
            raise RuntimeError("end_step() called before start_step()")
        elapsed = time.perf_counter() - self._step_start
        self._step_start = None
        self.step_times.append(elapsed)
        self.total_tokens += tokens
        metrics = {"system/step_time": elapsed}
        if tokens and elapsed > 0:
            metrics["system/tokens_per_second"] = tokens / elapsed
            # Comparable across profiles in a way tokens/s is not, since the two
            # GPUs run different batch sizes.
            metrics["system/seconds_per_mtok"] = elapsed / (tokens / 1e6)
        return metrics


def gpu_metrics(prefix: str = "gpu") -> dict[str, float]:
    snapshot = read_gpu_memory()
    return {
        f"{prefix}/memory_allocated_gb": snapshot.allocated_gb,
        f"{prefix}/memory_reserved_gb": snapshot.reserved_gb,
        f"{prefix}/memory_max_allocated_gb": snapshot.max_allocated_gb,
    }


class Tracker:
    """A W&B run that is safe to construct without credentials."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
        enabled: bool | None = None,
        run: Run | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = dict(config or {})
        self.resume_run_id = resume_run_id
        self._run: Run | None = run
        if enabled is None:
            enabled = run is not None or bool(os.environ.get("WANDB_API_KEY"))
        self.enabled = enabled
        self.meter = ThroughputMeter()

    def __enter__(self) -> Tracker:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()

    def start(self) -> None:
        if not self.enabled or self._run is not None:
            return
        import wandb

        self._run = wandb.init(  # type: ignore[assignment]
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=self.config,
            id=self.resume_run_id,
            # "must" rather than "allow": a resume that silently forks the run
            # splits one training curve across two charts and the break is easy
            # to miss.
            resume="must" if self.resume_run_id else None,
        )

    @property
    def run_id(self) -> str | None:
        return getattr(self._run, "id", None) if self._run is not None else None

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(dict(metrics), step=step)

    def log_step(self, *, tokens: int = 0, step: int | None = None, **extra: Any) -> None:
        """Close the current step and emit throughput plus VRAM in one payload."""
        metrics: dict[str, Any] = self.meter.end_step(tokens)
        metrics.update(gpu_metrics())
        metrics.update(extra)
        self.log(metrics, step=step)

    def log_artifact(
        self,
        path: Path | str,
        *,
        name: str,
        artifact_type: str,
        extra_paths: Sequence[Path | str] = (),
    ) -> None:
        """Upload one file (plus optional siblings) as a W&B artifact.

        Reports and profiles only. Checkpoints go to the Hub -- duplicating
        multi-gigabyte weights into W&B would be the actual quota problem, and
        `artifacts.CheckpointStore` already owns that path.

        Never raises. An artifact upload failing must not fail the evaluation or
        training run that produced the file, which is still on disk either way.
        `wandb` is imported lazily for the same reason `start()` does it.
        """
        if self._run is None:
            return
        files = [Path(path), *(Path(extra) for extra in extra_paths)]
        missing = [str(candidate) for candidate in files if not candidate.is_file()]
        if missing:
            LOG.warning("not logging artifact %s: missing %s", name, ", ".join(missing))
            return
        try:
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            for candidate in files:
                artifact.add_file(str(candidate))
            self._run.log_artifact(artifact)
        except Exception as exc:  # pragma: no cover - depends on a live W&B backend
            LOG.warning("artifact %s was not uploaded: %s: %s", name, type(exc).__name__, exc)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def tracker_for(
    tracking: Any,
    *,
    config: Mapping[str, Any] | None = None,
    resume_run_id: str | None = None,
) -> Tracker:
    """A `Tracker` from any config carrying a `TrackingConfig`.

    The three training call sites spelled this out identically, and `evaluate` and
    `profile-data` had no tracker at all -- so a report or a budgets file had nowhere
    to be logged. Named rather than inlined because "which W&B run does this command
    attach to" should have one answer.
    """
    return Tracker(
        project=tracking.wandb_project,
        entity=tracking.wandb_entity,
        run_name=tracking.run_name,
        config=config,
        resume_run_id=resume_run_id,
    )
