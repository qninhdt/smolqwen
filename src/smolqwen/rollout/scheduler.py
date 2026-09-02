"""The worker-pool dispatcher: `EnvDispatcher` over the synchronous `WorkerPool`.

The turn loop this file used to own now lives in `inference/turn_engine.py`, driven
by both rollout and evaluation. What remains is the thread pool that makes the
synchronous pool usable from a single-threaded loop, plus the binding and driver
types re-exported for callers that still import them from here.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import Future
from typing import Any

from smolqwen.env.pool import Result, WorkerPool
from smolqwen.env.scenarios import Scenario
from smolqwen.inference.turn_engine import (
    LENGTH_MARGIN,
    MAX_REPLACEMENTS_PER_POSITION,
    POLL_INTERVAL_S,
    TurnEngineConfig,
    TurnEngineError,
)
from smolqwen.rollout.driver import EnvDispatcher, RolloutDriver, ScenarioBinding

__all__ = [
    "LENGTH_MARGIN",
    "MAX_REPLACEMENTS_PER_POSITION",
    "POLL_INTERVAL_S",
    "EnvDispatcher",
    "PoolDispatcher",
    "RolloutDriver",
    "ScenarioBinding",
    "SchedulerConfig",
    "SchedulerError",
    "TurnEngineConfig",
]

# The loop's config and error type are the engine's now. Kept as aliases because
# `training/grpo.py` and the bench build them by these names.
SchedulerConfig = TurnEngineConfig
SchedulerError = TurnEngineError


class PoolDispatcher:
    """`EnvDispatcher` over the synchronous Phase 4 `WorkerPool`.

    The pool's public API blocks per call, and one worker executes its request
    queue serially under a per-worker lock. Concurrency therefore comes from a
    small thread pool: calls aimed at different workers proceed in parallel, calls
    aimed at one worker serialize exactly as the pool requires. Creation is
    serialized here as well because `WorkerPool.create` mutates shared bookkeeping
    (`_owner`, per-worker episode sets) outside any lock.

    Sizing contract: every admitted position holds one live pool episode, so
    `env_worker_count * env_episodes_per_worker` must be at least the engine's
    in-flight window. The pool raises `PoolError` otherwise, deliberately loud --
    silent episode queueing would break the ready-queue model. Generation never
    runs on these threads; only the engine loop drives the backend.
    """

    def __init__(self, pool: WorkerPool, *, max_workers: int | None = None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._pool = pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers or max(2, pool.worker_count))
        self._create_lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """How many live episodes the pool can hold; the engine's window ceiling."""
        return self._pool.worker_count * self._pool.episodes_per_worker

    def submit_create(self, episode_id: str, binding: ScenarioBinding) -> Future[Result]:
        return self._executor.submit(self._create, episode_id, binding.scenario)

    def _create(self, episode_id: str, scenario: Scenario) -> Result:
        with self._create_lock:
            return self._pool.create(
                episode_id,
                env_id=scenario.env_id,
                env_class_name=scenario.env_class_name,
                init_config=scenario.init_config,
                checklist=scenario.checklist,
                checklist_id=scenario.task_id,
            )

    def submit_step(
        self, episode_id: str, name: str, arguments: Mapping[str, Any]
    ) -> Future[Result]:
        return self._executor.submit(self._pool.step, episode_id, name, dict(arguments))

    def submit_score(self, episode_id: str) -> Future[Result]:
        return self._executor.submit(self._pool.score, episode_id)

    def submit_destroy(self, episode_id: str) -> Future[Result]:
        return self._executor.submit(self._pool.destroy, episode_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
