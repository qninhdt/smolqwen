"""Evaluation's `TurnDriver`: a benchmark adapter behind the shared turn engine.

Every difference between a benchmark and training rollout is expressed as an
`Advance`, so the engine branches on nothing:

- An adapter's `step` is ordinary in-process Python. BFCL's mutates `sys.path`
  (`bfcl.py:358`), so it must not run on a pool thread; `completed(...)` keeps it on
  the engine thread while still satisfying the engine's await protocol.
- The next static benchmark question arrives as a `user` turn, not a tool
  observation. The template renders those differently and the difference decides
  whether earlier reasoning survives the next render.
- A benchmark may reveal a tool mid-episode -- `multi_turn_miss_func` exists for
  exactly that -- so each `StepResult` can carry the next tool set.

Environment lifecycle is the adapter's own. EnvScaler creates its pool episode
lazily in `build_prompt` and destroys it inside `score`; BFCL has no environment at
all. So `open` and `close` are no-ops here, and the engine's `live_episode_ids`
tracking has nothing to release -- which is correct, not a gap: releasing a pool
episode the adapter still owns would double-destroy it.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from smolqwen.eval.adapters.base import AdapterResult, BenchmarkAdapter, EvalTask, StepResult
from smolqwen.inference.episode import Episode
from smolqwen.inference.turn_engine import Advance, completed

__all__ = ["AdapterDriver", "TaskBinding"]


class TaskBinding:
    """One benchmark task as the engine's binding: an id, a prompt, a tool set."""

    def __init__(self, task: EvalTask) -> None:
        self.task = task
        self.task_id = task.task_id
        self.group_index = 0
        self.tool_schemas = tuple(task.tools)

    @property
    def scenario(self) -> Any:
        """The engine reads `binding.scenario.task_id` for episode identity."""
        return self.task


class AdapterDriver:
    """Drives a `BenchmarkAdapter` through the shared engine, synchronously."""

    def __init__(self, adapter: BenchmarkAdapter, *, max_generation_turns: int) -> None:
        self._adapter = adapter
        self._max_turns = max_generation_turns
        self._tasks: dict[str, EvalTask] = {}
        # Per-episode scoring outcome and turn accounting, keyed by episode id. The
        # engine owns neither: an adapter's verdict is richer than a float, and the
        # report needs the whole `AdapterResult`.
        self.results: dict[str, AdapterResult] = {}

    # --- lifecycle ---

    def open(self, episode: Episode, binding: Any) -> Future[Any] | None:
        self._tasks[episode.episode_id] = binding.task
        return None

    def opened(self, episode: Episode, payload: Any) -> Advance:
        """The engine's initial messages come from the adapter's own prompt builder."""
        return Advance()

    def interpret(self, episode: Episode, text: str) -> Advance:
        task = self._tasks[episode.episode_id]
        step = self._adapter.step(task, text)
        return completed_advance(step, terminal_on_complete="final_answer")

    def resolve(self, episode: Episode, payload: Any) -> Advance:
        # Nothing is dispatched asynchronously, so nothing resolves. Present because
        # the protocol has six methods and a partial implementation would fail at
        # the first unusual episode rather than at construction.
        raise AssertionError("the adapter driver never issues a pending future")

    def score(self, episode: Episode) -> Future[Any] | None:
        task = self._tasks[episode.episode_id]
        return completed(self._adapter.score(task))

    def scored(self, episode: Episode, payload: Any) -> None:
        result = payload if isinstance(payload, AdapterResult) else AdapterResult(0.0, False)
        self.results[episode.episode_id] = result
        episode.reward = result.score

    def close(self, episode: Episode) -> Future[Any] | None:
        return None


def completed_advance(step: StepResult, *, terminal_on_complete: str) -> Advance:
    """One `StepResult` as an `Advance`, preserving the role and the tool set.

    A benchmark's several tool observations for one turn are joined rather than
    appended as separate messages: the engine appends exactly one observation per
    advance, and the template merges consecutive `role: tool` messages into a single
    `<tool_response>` block anyway.
    """
    observation = step.observation or None
    if step.tool_observations is not None:
        observation = "\n".join(step.tool_observations) or None
    return Advance(
        observation=observation,
        observation_role=step.observation_role,
        observation_label="ok" if step.env_steps else step.observation_role,
        env_steps=step.env_steps,
        tools=step.tools,
        terminal=terminal_on_complete if step.complete else None,  # type: ignore[arg-type]
    )
