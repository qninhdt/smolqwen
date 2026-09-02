"""Batched evaluation: many tasks in flight, one generation call per cycle.

The serial path in `runner.evaluate_adapter` advances one task at a time, so
`policies.py` calls `model.generate()` at batch size 1 for every turn of every
task. A 2B model decoding one sequence leaves the card almost idle.

This drives the same adapters through the shared turn engine instead, which holds
`max_in_flight` tasks concurrently and issues one batched generation per cycle. Two
constraints shape the window:

- **Pool capacity.** EnvScaler creates one live pool episode per task in flight,
  and `WorkerPool.create` raises once every worker is full. The shipped configs
  hold 80 held-out tasks against a 32-episode pool, so an unbounded window makes
  the 33rd task raise and the run score zero.
- **Generation width.** Beyond `generation_concurrency` the extra episodes only
  queue, so a window wider than the pool buys nothing.

Hence `min(generation_concurrency, pool_capacity)`, with capacity read from the
adapter when it exposes one and the concurrency alone when it does not (BFCL runs
in-process and has no pool).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.base import AdapterResult, BenchmarkAdapter, EvalTask
from smolqwen.eval.driver import AdapterDriver, TaskBinding
from smolqwen.eval.metrics import TaskMetrics
from smolqwen.eval.trajectories import TrajectoryRecord
from smolqwen.inference.decoding import decode_completion
from smolqwen.inference.episode import Episode
from smolqwen.inference.profiles import EvalProfile
from smolqwen.inference.turn_engine import TurnEngine, TurnEngineConfig


def pool_capacity_of(adapter: BenchmarkAdapter) -> int | None:
    """How many live episodes the adapter's environment layer can hold, if any.

    `None` means "no environment layer", which is BFCL: its steps are in-process
    Python, so the only bound on concurrency is generation width.
    """
    pool = getattr(adapter, "_pool", None)
    if pool is None:
        return None
    worker_count = getattr(pool, "worker_count", None)
    per_worker = getattr(pool, "episodes_per_worker", None)
    if worker_count is None or per_worker is None:
        return None
    return int(worker_count) * int(per_worker)


def admission_window(config: EvalConfig, adapter: BenchmarkAdapter) -> int:
    """`min(generation_concurrency, pool_capacity)`, or the concurrency alone."""
    concurrency = config.profile.generation_concurrency
    capacity = pool_capacity_of(adapter)
    return concurrency if capacity is None else min(concurrency, capacity)


def engine_config(config: EvalConfig, *, max_in_flight: int) -> TurnEngineConfig:
    """The engine's knobs from the resolved eval config.

    `max_generation_turns` is `max_steps_per_task`: that is what the serial loop
    bounded (`runner.py:87` counts generations, not environment steps), so keeping
    the same field on the same meaning is what makes the two paths comparable.
    """
    profile = EvalProfile.from_config(config)
    return TurnEngineConfig(
        generation_concurrency=profile.concurrency,
        max_env_steps=config.profile.max_env_steps,
        max_generation_turns=config.max_steps_per_task,
        max_new_tokens_per_step=profile.max_new_tokens,
        max_model_len=profile.max_model_len,
        temperature=profile.temperature,
        top_p=profile.top_p,
        max_in_flight=max_in_flight,
        # Evaluation trains nothing, so there is no mask to build. Turning it off is
        # what makes the engine's liveness tracking load-bearing rather than
        # incidental -- see `test_turn_engine_admission.py`.
        build_masks=False,
    )


def evaluate_batched(
    config: EvalConfig,
    adapter: BenchmarkAdapter,
    *,
    backend: Any,
    tokenizer: Any,
    tasks: Sequence[EvalTask] | None = None,
    records: list[TrajectoryRecord] | None = None,
) -> dict[str, dict[str, float]]:
    """Advance every task concurrently through the shared engine, then summarize.

    `backend` is anything satisfying the engine's generation protocol -- the offline
    vLLM engine in production, a scripted backend in tests. Keeping it injected is
    what lets this path be tested without a card.
    """
    task_list = list(tasks if tasks is not None else adapter.load_tasks())
    if not task_list:
        return adapter.summarize([])

    window = admission_window(config, adapter)
    driver = AdapterDriver(adapter, max_generation_turns=config.max_steps_per_task)
    engine = TurnEngine(
        backend=backend,
        driver=driver,
        initial_messages=lambda binding: _initial_messages(adapter, binding),
        render_prefix_ids=_renderer(tokenizer),
        decode=lambda ids: decode_completion(tokenizer, list(ids)),
        config=engine_config(config, max_in_flight=window),
    )

    started = time.monotonic()
    episodes = engine.run([TaskBinding(task) for task in task_list])
    wall_s = time.monotonic() - started

    metrics: list[TaskMetrics] = []
    for task, episode in zip(task_list, episodes, strict=True):
        result = driver.results.get(episode.episode_id) or AdapterResult(0.0, False)
        invalid_calls = adapter.invalid_call_count(task)
        metrics.append(
            TaskMetrics(
                category=task.category,
                score=result.score,
                invalid_calls=invalid_calls,
                steps=episode.step_count,
                generated_tokens=episode.generated_tokens,
                truncated=episode.truncated,
                exact_success=result.exact_success,
                diagnostics=dict(result.diagnostics),
            )
        )
        if records is not None:
            records.append(_record(task, episode, result, invalid_calls, wall_s / len(task_list)))
    return adapter.summarize(metrics)


def _initial_messages(adapter: BenchmarkAdapter, binding: Any) -> list[Any]:
    """The adapter's own opening prompt, in the engine's `Message` shape.

    Adapter-owned prompt construction is a contract (`base.py`): the runner never
    knows how a benchmark opens an episode. So the conversion happens here rather
    than the adapter learning about `Message`.
    """
    from smolqwen.data.loader import parse_message

    return [parse_message(dict(message)) for message in adapter.build_prompt(binding.task, [])]


def _renderer(tokenizer: Any) -> Any:
    from smolqwen.data.render import render_prefix
    from smolqwen.rollout.rollout_func import encode_ids

    def render_prefix_ids(messages: Sequence[Any], tools: Sequence[Mapping[str, Any]]) -> list[int]:
        text = render_prefix(
            tokenizer, messages, tools=[dict(tool) for tool in tools], add_generation_prompt=True
        )
        return encode_ids(tokenizer, text)

    return render_prefix_ids


def _record(
    task: EvalTask,
    episode: Episode,
    result: AdapterResult,
    invalid_calls: int,
    wall_s: float,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        task_id=task.task_id,
        category=task.category,
        messages=[message.to_template_dict() for message in episode.messages],
        observations=list(episode.observations),
        score=result.score,
        exact_success=result.exact_success,
        completed=result.completed,
        failure_reason=result.failure_reason,
        failed_check_names=list(result.failed_check_names),
        diagnostics=dict(result.diagnostics),
        terminal_reason=episode.terminal_reason,
        generation_turns=episode.generation_count,
        env_steps=episode.step_count,
        generated_tokens=episode.generated_tokens,
        truncated=episode.truncated,
        invalid_calls=invalid_calls,
        wall_s=wall_s,
    )
