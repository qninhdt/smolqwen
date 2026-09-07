"""Turn-engine evaluation: local vLLM batches tasks in flight.

The local vLLM path advances many tasks concurrently and issues one generation call per
cycle. The HTTP compatibility path remains text-native.

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
from dataclasses import dataclass
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.console import logger, progress_task
from smolqwen.eval.adapters.base import AdapterResult, BenchmarkAdapter, EvalTask
from smolqwen.eval.checkpoints import ResolvedCheckpoint
from smolqwen.eval.driver import AdapterDriver, TaskBinding
from smolqwen.eval.metrics import TaskMetrics
from smolqwen.eval.trajectories import TrajectoryRecord
from smolqwen.inference.decoding import decode_completion
from smolqwen.inference.episode import Episode
from smolqwen.inference.profiles import EvalProfile
from smolqwen.inference.turn_engine import TurnEngine, TurnEngineConfig

LOG = logger(__name__)

# The adapter name the engine registers a PEFT directory under. One slot, because
# `evaluate` scores one checkpoint per invocation.
ADAPTER_SLOT = "eval-adapter"


@dataclass(frozen=True)
class Generation:
    """The generation backend selected for this evaluation run."""

    backend: Any | None
    path: str
    engine: Any | None = None

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()


def generation_for(config: EvalConfig, resolved: ResolvedCheckpoint) -> Generation:
    """Use HTTP for a served endpoint and require in-process vLLM otherwise."""
    if resolved.source == "endpoint" or resolved.path is None:
        return Generation(backend=None, path="http")

    from smolqwen.inference.engine import OfflineEngineBackend, offline_engine_for_eval

    adapters = {ADAPTER_SLOT: resolved.adapter_path} if resolved.adapter_path else None
    engine = offline_engine_for_eval(
        resolved.path,
        EvalProfile.from_config(config),
        revision=resolved.revision,
        adapter=adapters,
    )
    path = "vllm+lora" if adapters else "vllm"
    LOG.info("generating through in-process vLLM (%s), dtype %s", path, engine.profile.dtype)
    return Generation(
        backend=OfflineEngineBackend(engine, adapter=ADAPTER_SLOT if adapters else None),
        path=path,
        engine=engine,
    )


def pool_capacity_of(adapter: BenchmarkAdapter) -> int | None:
    """How many live episodes the adapter's environment layer can hold, if any.

    `None` means "no environment layer", which is BFCL: its steps are in-process
    Python, so the only bound on concurrency is generation width.

    `pool_capacity` is asked first because a pool is built lazily -- EnvScaler's
    appears on the first `build_prompt`, which is *after* the window is computed --
    so reading `_pool` alone reported `None` for an adapter that does have a
    capacity, and `min()` silently degraded to the generation width. The live-pool
    read stays as the second source: it is the authority once a pool exists.
    """
    declared = getattr(adapter, "pool_capacity", None)
    if declared is not None:
        return int(declared)
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
        enable_thinking=config.enable_thinking,
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
    label: str = "evaluation",
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
    started = time.monotonic()
    scores: list[float] = []
    category_by_task_id = {task.task_id: task.category for task in task_list}

    with progress_task(label, total=len(task_list), unit="tasks", every=1) as advance:

        def on_episode_done(episode: Episode) -> None:
            result = driver.results.get(episode.episode_id) or AdapterResult(0.0, False)
            scores.append(result.score)
            running = sum(scores) / len(scores)
            category = category_by_task_id.get(episode.scenario_id, episode.scenario_id)
            advance(f"{category} mean {running:.3f}")

        engine = TurnEngine(
            backend=backend,
            driver=driver,
            initial_messages=lambda binding: _initial_messages(adapter, binding),
            render_prefix_ids=_renderer(tokenizer, config.enable_thinking),
            decode=lambda ids: decode_completion(tokenizer, list(ids)),
            config=engine_config(config, max_in_flight=window),
            on_episode_done=on_episode_done,
        )
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
                terminal_reason=episode.terminal_reason,
            )
        )
        if records is not None:
            records.append(_record(task, episode, result, invalid_calls, wall_s / len(task_list)))
    _warn_if_nothing_generated(episodes)
    return adapter.summarize(metrics)


def _warn_if_nothing_generated(episodes: Sequence[Episode]) -> None:
    """Say so when no episode generated anything. Measured, not hypothetical.

    On a T4 with the context window below EnvScaler's tool-schema size, every episode
    hit the window check at admission and terminated with zero generations -- and the
    run still reported `score: 0.25`, because the verifier grades final environment
    state and an untouched initial state scores whatever it scores. The only signal in
    the report was `average_generated_tokens: 0.0`, which requires a reader to already
    suspect the failure.

    Not an exception: `evaluate` is also how a genuinely mute model is measured, and
    refusing to report that would be its own kind of wrong. A log line at WARNING that
    names the likely cause is the honest middle.
    """
    if not episodes or any(episode.generation_count for episode in episodes):
        return
    reasons = sorted({episode.terminal_reason or "unknown" for episode in episodes})
    LOG.warning(
        "no episode generated a single turn (%d episodes, terminal reasons: %s). "
        "A score computed from this measures the environment's initial state, not the "
        "model. The usual cause is max_seq_length below the rendered prompt: EnvScaler "
        "tool schemas alone run to ~4k tokens, ~6.8k at the widest env.",
        len(episodes),
        ", ".join(reasons),
    )


def _initial_messages(adapter: BenchmarkAdapter, binding: Any) -> list[Any]:
    """The adapter's own opening prompt, in the engine's `Message` shape.

    Adapter-owned prompt construction is a contract (`base.py`): the runner never
    knows how a benchmark opens an episode. So the conversion happens here rather
    than the adapter learning about `Message`.
    """
    from smolqwen.data.loader import parse_message

    return [parse_message(dict(message)) for message in adapter.build_prompt(binding.task, [])]


def _renderer(tokenizer: Any, enable_thinking: bool = True) -> Any:
    from smolqwen.data.render import render_prefix
    from smolqwen.rollout.rollout_func import encode_ids

    def render_prefix_ids(messages: Sequence[Any], tools: Sequence[Mapping[str, Any]]) -> list[int]:
        text = render_prefix(
            tokenizer,
            messages,
            tools=[dict(tool) for tool in tools],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
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
