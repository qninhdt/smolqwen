"""The ready-queue scheduler: intra-call concurrency over TRL's prompts.

TRL invokes `rollout_func(prompts, trainer)` synchronously with exactly
`generation_batch_size` prompts in group-contiguous order, and requires rows
back, positionally aligned, in the same order. Within that one call, TRL's own
`_tool_call_loop` advances the batch in turn lockstep: every episode with a
pending tool call waits for the slowest one before the next generation. This
scheduler removes both locksteps *inside* the call — the turn barrier and the
GRPO-group barrier — by cycling:

    take up to `generation_concurrency` READY episodes -> one generation
    sub-batch -> parse -> dispatch each tool call independently -> an episode
    returns to READY the moment *its own* call completes.

A slow environment blocks only its own episode. The call returns only when
every position is DONE, with crashed episodes replaced so the row count and
group ordering still match what TRL handed in.

Concurrency comes from a thread pool over the synchronous Phase 4 `WorkerPool`:
each worker serializes its own requests (that is the pool's contract, held by a
per-worker lock), different workers proceed in parallel, and generation never
runs inside those threads — only this loop drives the backend. The pool reports
a lost worker through `Result.lost_episode_ids`; episodes of that worker that
were idle at the time discover the loss on their next call, and are replaced
then, lazily but with the same guarantees.

Episode state meanings: `ready` (may generate), `generating` (inside the
backend call), `tool` (a create/step/score request is in flight), `done`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from smolqwen.data.loader import Message
from smolqwen.env.parse import parse_turn
from smolqwen.env.pool import Result, WorkerPool
from smolqwen.env.scenarios import Scenario
from smolqwen.rollout.episode import Episode, TerminalReason
from smolqwen.rollout.generation import (
    GenerationBackend,
    GenerationRequest,
    GenerationResult,
)
from smolqwen.rollout.mask import EpisodeMaskBuilder

# How long a cycle may spend reaping before it dispatches again. Not a timeout
# for any episode — only the granularity at which completions are noticed.
POLL_INTERVAL_S = 0.02

# Ceiling on replacement attempts per position. A crash is random; a failure
# that reproduces deterministically on a fresh episode of the same scenario is
# not a crash but a bug, and looping forever would silently burn the run.
MAX_REPLACEMENTS_PER_POSITION = 3

# Head-room subtracted from `max_model_len` before comparing against a prefix
# or budgeting new tokens: the template's generation prompt and stop handling
# need a few positions beyond what the render already shows.
LENGTH_MARGIN = 8


class SchedulerError(RuntimeError):
    """Raised when the scheduler cannot make progress or is misconfigured."""


@dataclass(frozen=True)
class ScenarioBinding:
    """Everything one rollout position needs beside its prompt.

    `tool_schemas` are the parent-side JSON schemas from `EnvSpec.tools` — no
    env code is compiled in this process to obtain them.
    """

    scenario: Scenario
    group_index: int
    tool_schemas: tuple[dict[str, Any], ...] = ()
    env_introduction: str = ""
    initial_messages: tuple[Message, ...] = ()

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(
            str(tool.get("function", {}).get("name", "")) for tool in self.tool_schemas
        ) - {""}


class EnvDispatcher(Protocol):
    """Asynchronous environment access. Futures resolve to `Result`, never raise."""

    def submit_create(self, episode_id: str, binding: ScenarioBinding) -> Future[Result]: ...
    def submit_step(
        self, episode_id: str, name: str, arguments: Mapping[str, Any]
    ) -> Future[Result]: ...
    def submit_score(self, episode_id: str) -> Future[Result]: ...
    def submit_destroy(self, episode_id: str) -> Future[Result]: ...


@dataclass(frozen=True)
class SchedulerConfig:
    """The phase's semantic knobs. Sizing lives in the profiles."""

    generation_concurrency: int = 8
    max_env_steps: int = 16
    episode_timeout_s: float = 600.0
    max_new_tokens_per_step: int = 1024
    max_model_len: int = 16384
    temperature: float = 1.0
    top_p: float = 1.0
    fork_threshold_tokens: int = 1024


@dataclass
class _Slot:
    """One return position: the current episode plus its replacement history."""

    position: int
    binding: ScenarioBinding
    episode: Episode
    started_at: float
    replacements: int = 0

    def fresh_episode(self) -> Episode:
        suffix = "" if self.replacements == 0 else f"~{self.replacements}"
        return Episode(
            episode_id=f"{self.binding.scenario.task_id}@{self.position}{suffix}",
            scenario_id=self.binding.scenario.task_id,
            group_index=self.binding.group_index,
            replaced_episode_id=(
                self.episode.episode_id if self.replacements > 0 and self.episode else None
            ),
        )


class RolloutScheduler:
    """Drives every position to DONE and returns episodes in input order."""

    def __init__(
        self,
        *,
        backend: GenerationBackend,
        dispatcher: EnvDispatcher,
        initial_messages: Callable[[ScenarioBinding], list[Message]],
        render_prefix_ids: Callable[[Sequence[Message], ScenarioBinding], list[int]],
        decode: Callable[[Sequence[int]], str],
        config: SchedulerConfig,
        clock: Callable[[], float] = time.monotonic,
        wait_for: Callable[..., set[Future[Any]]] | None = None,
    ) -> None:
        self._backend = backend
        self._dispatcher = dispatcher
        self._initial_messages = initial_messages
        self._render = render_prefix_ids
        self._decode = decode
        self._config = config
        self._clock = clock
        self._wait = wait_for or _default_wait
        # (clock time, episode id, event): the timeline the profiler and the
        # no-turn-barrier test read — wall-clock values under the real clock,
        # virtual ones under the simulated clock.
        self.events: list[tuple[float, str, str]] = []
        self.stage_intervals: list[tuple[str, float, float]] = []
        self.queue_depth_samples: list[int] = []
        self._futures: dict[Future[Any], tuple[_Slot, Episode, str]] = {}
        self._submitted_at: dict[Future[Any], float] = {}
        self._builders: dict[str, EpisodeMaskBuilder] = {}
        self._prefix_lengths: dict[str, int] = {}
        self._slots_by_episode_id: dict[str, _Slot] = {}

    # --- public ---

    def run(self, bindings: Sequence[ScenarioBinding]) -> list[Episode]:
        """Run every binding to completion; the result aligns 1:1 with input."""
        if not bindings:
            return []
        slots = [self._open_slot(position, binding) for position, binding in enumerate(bindings)]
        try:
            while any(slot.episode.state != "done" for slot in slots):
                self._reap()
                self._mark_timeouts(slots)
                self._generate_ready(slots)
                self._dispatch_scoring(slots)
                self._wait_for_any()
                if (
                    not self._futures
                    and any(slot.episode.state != "done" for slot in slots)
                    and not any(
                        slot.episode.state == "ready" and slot.episode.terminal_reason is None
                        for slot in slots
                    )
                ):
                    raise SchedulerError(
                        "scheduler has unfinished episodes but no ready work or in-flight "
                        "environment calls"
                    )
        except BaseException:
            self._cleanup_after_failure(slots)
            raise
        return [slot.episode for slot in slots]

    def episode_builder(self, episode_id: str) -> EpisodeMaskBuilder:
        """The mask builder rollout_func reads the returned arrays from."""
        return self._builders[episode_id]

    # --- slot lifecycle ---

    def _open_slot(self, position: int, binding: ScenarioBinding) -> _Slot:
        episode = Episode(
            episode_id=f"{binding.scenario.task_id}@{position}",
            scenario_id=binding.scenario.task_id,
            group_index=binding.group_index,
        )
        slot = _Slot(
            position=position,
            binding=binding,
            episode=episode,
            started_at=self._clock(),
        )
        slot.episode.messages = list(self._initial_messages(binding))
        self._slots_by_episode_id[episode.episode_id] = slot
        self._log(slot.episode.episode_id, "open")
        self._begin_create(slot)
        return slot

    def _begin_create(self, slot: _Slot) -> None:
        episode = slot.episode
        episode.state = "tool"
        binding = slot.binding
        episode_id = episode.episode_id
        self._submit(
            slot,
            episode,
            "create",
            lambda: self._dispatcher.submit_create(episode_id, binding),
        )

    # --- cycle steps ---

    def _reap(self) -> None:
        for future in [f for f in self._futures if f.done()]:
            slot, episode, action = self._futures.pop(future)
            started = self._submitted_at.pop(future)
            if slot.episode is not episode:
                # Another result from the same crashed worker may already have
                # replaced this slot. Consume the stale future without letting
                # its result (or exception) affect the fresh replacement.
                try:
                    future.result()
                except Exception:
                    pass
                continue
            try:
                result = future.result()
            except Exception as exc:
                raise SchedulerError(
                    f"{episode.episode_id}: {action} future raised instead of returning Result"
                ) from exc
            finished = self._clock()
            if action == "step":
                episode.record_timing("env.step", finished - started)
                self.stage_intervals.append(("env.step", started, finished))
            elif action == "score":
                episode.record_timing("verifier", finished - started)
                self.stage_intervals.append(("verifier", started, finished))
            elif action == "create":
                episode.record_timing("env.create", finished - started)
                self.stage_intervals.append(("env.create", started, finished))
            elif action == "destroy":
                episode.record_timing("env.destroy", finished - started)
                self.stage_intervals.append(("env.destroy", started, finished))
            if result.is_infrastructure_failure:
                self._replace_lost_episodes(episode, result)
                continue
            {
                "create": self._on_create,
                "step": self._on_step,
                "score": self._on_score,
                "destroy": self._on_destroy,
            }[action](slot, episode, result)

    def _generate_ready(self, slots: Sequence[_Slot]) -> None:
        ready = [
            slot
            for slot in slots
            if slot.episode.state == "ready" and slot.episode.terminal_reason is None
        ]
        self.queue_depth_samples.append(len(ready))
        if not ready:
            return

        candidates: list[tuple[_Slot, list[int], int]] = []
        for slot in ready:
            episode = slot.episode
            render_started = self._clock()
            prefix = self._render(episode.messages, slot.binding)
            render_finished = self._clock()
            episode.record_timing("tokenization", render_finished - render_started)
            self.stage_intervals.append(("tokenization", render_started, render_finished))
            if len(prefix) >= self._config.max_model_len - LENGTH_MARGIN:
                # The conversation fills the window: budget exhausted, the
                # step-cap family of terminals. Scored, never replaced — the
                # same scenario would overflow again, deterministically.
                self._terminal(slot, "step_cap")
                self._log(episode.episode_id, "budget_exhausted")
                continue
            if episode.step_count >= self._config.max_env_steps:
                self._terminal(slot, "step_cap")
                continue
            candidates.append((slot, prefix, self._token_budget(prefix)))

        for slot, prefix, _budget in candidates[: self._config.generation_concurrency]:
            episode = slot.episode
            episode.state = "generating"
            self._builders[episode.episode_id].open_turn(prefix)
            self._prefix_lengths[episode.episode_id] = len(prefix)
            self._log(episode.episode_id, "generate")

        requests = [
            GenerationRequest(
                episode_id=slot.episode.episode_id,
                prompt_ids=tuple(prefix),
                max_new_tokens=budget,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
            )
            for slot, prefix, budget in candidates[: self._config.generation_concurrency]
        ]
        if not requests:
            return
        started = self._clock()
        results = self._backend.generate(requests)
        finished = self._clock()
        elapsed = finished - started
        self.stage_intervals.append(("generation", started, finished))
        dispatched = candidates[: self._config.generation_concurrency]
        for (slot, _, _), result in zip(dispatched, results, strict=True):
            self._on_generation(slot, result, elapsed)

    def _on_generation(self, slot: _Slot, result: GenerationResult, elapsed: float) -> None:
        episode = slot.episode
        episode.record_timing("generation", elapsed)
        parse_started = self._clock()
        text = self._decode(result.token_ids)
        turn = parse_turn(text, available_tools=slot.binding.tool_names)
        parse_finished = self._clock()
        episode.record_timing("parse", parse_finished - parse_started)
        self.stage_intervals.append(("parse", parse_started, parse_finished))

        reasoning, content = split_generation_continuation(text)
        # `add_generation_prompt=True` already emitted the opening `<think>\n`.
        # vLLM therefore returns only the continuation, normally
        # `reasoning\n</think>\n\ncontent`. Store the semantic fields the chat
        # template expects so the next full re-render reproduces the held model
        # tokens. Treating the continuation as plain `content` makes the template
        # synthesize a second reasoning block and silently demotes earlier model
        # tokens to env-mask zero on every turn.
        message = Message(role="assistant", content=content, reasoning_content=reasoning)
        episode.messages.append(message)
        self._mirror_scripted(episode, message)

        builder = self._builders[episode.episode_id]
        builder.append_response(result.token_ids, list(result.logprobs))
        episode.prompt_completion_boundary = builder.boundary
        episode.prompt_ids = list(builder.prompt_ids)

        if turn.outcome == "no_call":
            self._terminal(slot, "final_answer")
            return
        if turn.is_invalid_call:
            episode.invalid_call_count += 1
            self._append_observation(slot, turn.observation(), kind="invalid")
            return
        episode.state = "tool"
        assert turn.name is not None and turn.arguments is not None
        name, arguments = turn.name, dict(turn.arguments)
        episode_id = episode.episode_id
        self._submit(
            slot,
            episode,
            "step",
            lambda: self._dispatcher.submit_step(episode_id, name, arguments),
        )

    def _on_step(self, slot: _Slot, episode: Episode, result: Result) -> None:
        episode.step_count += 1
        if result.reason == "timeout":
            self._terminal(slot, "timeout")
            episode.state = "ready"
            return
        if result.reason == "error" and (result.detail or "").startswith("unrecoverable:"):
            self._terminal(slot, "unrecoverable")
            episode.state = "ready"
            return
        if not result.ok:
            # An env exception becomes an observation the model can act on; the
            # step cap and the episode timeout bound the loop it could open.
            episode.invalid_call_count += 1
            self._append_observation(slot, f"Error: env: {result.detail}", kind="error")
            return
        self._append_observation(slot, str(result.value), kind="ok")

    def _on_create(self, slot: _Slot, episode: Episode, result: Result) -> None:
        if not result.ok:
            raise SchedulerError(
                f"{episode.episode_id}: environment creation failed "
                f"({result.reason}): {result.detail}"
            )
        prefix_ids = self._render(episode.messages, slot.binding)
        self._builders[episode.episode_id] = EpisodeMaskBuilder(
            prefix_ids,
            fork_threshold_tokens=self._config.fork_threshold_tokens,
        )
        bind = getattr(self._backend, "bind", None)
        if bind is not None:
            bind(episode.episode_id, episode.messages)
        self._log(episode.episode_id, "ready")
        episode.state = "ready"

    def _on_score(self, slot: _Slot, episode: Episode, result: Result) -> None:
        if not result.ok:
            raise SchedulerError(
                f"{episode.episode_id}: scoring failed ({result.reason}): {result.detail}"
            )
        payload = result.value
        episode.reward = float(payload["reward"])
        checks = payload.get("checks") or []
        episode.per_check_bools = tuple(bool(check.get("passed")) for check in checks)
        builder = self._builders.get(episode.episode_id)
        if builder is not None:
            episode.drift_tally = builder.tally
            episode.prompt_ids = list(builder.prompt_ids)
            episode.completion_ids = list(builder.completion_ids)
            episode.logprobs = list(builder.logprobs)
            episode.prompt_completion_boundary = builder.boundary
            episode.mask_spans = list(builder.spans)
        # Destruction is part of completion: returning while the environment is
        # still live can exhaust the pool on the next rollout call. Keep the
        # builder until `rollout_func` has assembled the return arrays.
        episode.state = "tool"
        episode_id = episode.episode_id
        self._submit(
            slot,
            episode,
            "destroy",
            lambda: self._dispatcher.submit_destroy(episode_id),
        )

    def _on_destroy(self, slot: _Slot, episode: Episode, result: Result) -> None:
        self._prefix_lengths.pop(episode.episode_id, None)
        episode.state = "done"
        self._log(episode.episode_id, "done")
        self._log(episode.episode_id, "destroyed")

    # --- helpers ---

    def _append_observation(self, slot: _Slot, observation: str, *, kind: str) -> None:
        episode = slot.episode
        episode.observations.append(observation)
        message = Message(role="tool", content=observation)
        episode.messages.append(message)
        self._mirror_scripted(episode, message)
        self._log(episode.episode_id, f"observation:{kind}")
        episode.state = "ready"

    def _mirror_scripted(self, episode: Episode, message: Message) -> None:
        observe = getattr(self._backend, "observe", None)
        if observe is not None:
            observe(episode.episode_id, message.to_template_dict())

    def _terminal(self, slot: _Slot, reason: TerminalReason) -> None:
        slot.episode.terminal_reason = reason
        self._log(slot.episode.episode_id, f"terminal:{reason}")

    def _mark_timeouts(self, slots: Sequence[_Slot]) -> None:
        now = self._clock()
        for slot in slots:
            episode = slot.episode
            if episode.terminal_reason is not None or episode.state == "done":
                continue
            if now - slot.started_at > self._config.episode_timeout_s:
                self._terminal(slot, "timeout")

    def _dispatch_scoring(self, slots: Sequence[_Slot]) -> None:
        for slot in slots:
            episode = slot.episode
            if episode.terminal_reason is None or episode.state == "done":
                continue
            if episode.state == "tool":
                # A step or create is still in flight. The pool's own timeout
                # layers resolve it; scoring waits rather than queueing behind
                # a call that may already be dead.
                continue
            episode.state = "tool"
            episode_id = episode.episode_id
            self._submit(
                slot,
                episode,
                "score",
                partial(self._dispatcher.submit_score, episode_id),
            )

    def _replace(self, slot: _Slot, reason: TerminalReason) -> None:
        """Retire the current episode and re-admit the scenario fresh."""
        old = slot.episode
        old.terminal_reason = reason
        old.state = "done"
        self._slots_by_episode_id.pop(old.episode_id, None)
        self._builders.pop(old.episode_id, None)
        self._log(old.episode_id, f"replaced:{reason}")
        if slot.replacements >= MAX_REPLACEMENTS_PER_POSITION:
            raise SchedulerError(
                f"position {slot.position} exceeded {MAX_REPLACEMENTS_PER_POSITION} "
                f"replacements ({reason}); a deterministic failure is a bug, not a crash"
            )
        slot.replacements += 1
        slot.started_at = self._clock()
        slot.episode = slot.fresh_episode()
        slot.episode.messages = list(self._initial_messages(slot.binding))
        self._slots_by_episode_id[slot.episode.episode_id] = slot
        self._log(slot.episode.episode_id, "replacement_open")
        self._begin_create(slot)

    def _replace_lost_episodes(self, episode: Episode, result: Result) -> None:
        """Replace every live slot lost with a crashed Phase 4 worker."""
        lost_ids = set(result.lost_episode_ids)
        lost_ids.add(episode.episode_id)
        for episode_id in sorted(lost_ids):
            slot = self._slots_by_episode_id.get(episode_id)
            if slot is None or slot.episode.episode_id != episode_id:
                continue
            self._log(episode_id, "worker_crash")
            self._replace(slot, "worker_crash")

    def _cleanup_after_failure(self, slots: Sequence[_Slot]) -> None:
        """Drain bounded pool calls and await cleanup before preserving the error.

        A create may still be running when generation or scheduling raises. It
        has no mask builder yet, so looking only at `_builders` misses an
        environment that becomes live after failure handling starts. Phase 4
        bounds every pool call; drain those futures, discover successful late
        creates, then submit and await destruction for every possibly-live ID.
        """
        possibly_live = set(self._builders)
        pending = dict(self._futures)
        for future in pending:
            future.cancel()

        while pending:
            done = {future for future in pending if future.done()}
            if not done:
                done = self._wait(pending.keys(), timeout=POLL_INTERVAL_S)
            for future in done:
                _slot, episode, action = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    continue
                if result.is_infrastructure_failure:
                    possibly_live.difference_update(result.lost_episode_ids)
                    possibly_live.discard(episode.episode_id)
                elif action == "create" and result.ok:
                    possibly_live.add(episode.episode_id)
                elif action == "destroy" and result.ok:
                    possibly_live.discard(episode.episode_id)

        self._futures.clear()
        self._submitted_at.clear()
        cleanup: set[Future[Result]] = set()
        for episode_id in sorted(possibly_live):
            try:
                cleanup.add(self._dispatcher.submit_destroy(episode_id))
            except Exception:
                # A worker crash may already have removed the owner. Preserve
                # the original scheduler error; pool shutdown is the fallback.
                continue
        while cleanup:
            done = {future for future in cleanup if future.done()}
            if not done:
                done = self._wait(cleanup, timeout=POLL_INTERVAL_S)
            for future in done:
                cleanup.remove(future)
                try:
                    future.result()
                except Exception:
                    pass

    def _token_budget(self, prefix: list[int]) -> int:
        room = self._config.max_model_len - len(prefix) - LENGTH_MARGIN
        return max(1, min(self._config.max_new_tokens_per_step, room))

    def _submit(
        self, slot: _Slot, episode: Episode, action: str, factory: Callable[[], Future[Result]]
    ) -> None:
        future = factory()
        self._futures[future] = (slot, episode, action)
        self._submitted_at[future] = self._clock()

    def _wait_for_any(self) -> None:
        if self._futures:
            self._wait(self._futures.keys(), timeout=POLL_INTERVAL_S)

    def _log(self, episode_id: str, event: str) -> None:
        self.events.append((self._clock(), episode_id, event))


def _default_wait(futures: Any, timeout: float | None = None) -> set[Future[Any]]:
    done, _ = wait(list(futures), timeout=timeout)
    return done


def split_generation_continuation(text: str) -> tuple[str, str]:
    """Split a continuation generated after the template's opening `<think>`.

    Qwen3.5's generation prompt ends in ``<think>\n``. The sampled text thus
    contains the reasoning body and closing tag, but not the opening tag. A
    backend returning a complete block is tolerated for deterministic tests and
    alternate engines. If the block never closes (usually token truncation), the
    whole continuation is retained as reasoning and the episode terminates as a
    no-call turn.
    """
    from smolqwen.env.parse import split_reasoning

    if "<think>" in text:
        return split_reasoning(text)
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.strip(), content.strip()
    return text.strip(), ""


class PoolDispatcher:
    """`EnvDispatcher` over the synchronous Phase 4 `WorkerPool`.

    The pool's public API blocks per call, and one worker executes its request
    queue serially under a per-worker lock. Concurrency therefore comes from a
    small thread pool: calls aimed at different workers proceed in parallel,
    calls aimed at one worker serialize exactly as the pool requires. Creation
    is serialized here as well because `WorkerPool.create` mutates shared
    bookkeeping (`_owner`, per-worker episode sets) outside any lock.

    Sizing contract: every position holds one live pool episode for the whole
    call, so `env_worker_count * env_episodes_per_worker` must be at least the
    active pool size. The pool raises `PoolError` otherwise, deliberately loud:
    silent episode queueing would break the ready-queue model this phase is
    built on. Generation never runs on these threads — only the scheduler loop
    drives the backend.
    """

    def __init__(self, pool: WorkerPool, *, max_workers: int | None = None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._pool = pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers or max(2, pool.worker_count))
        self._create_lock = threading.Lock()

    def submit_create(self, episode_id: str, binding: ScenarioBinding) -> Future[Result]:
        scenario = binding.scenario
        return self._executor.submit(self._create, episode_id, scenario)

    def _create(self, episode_id: str, scenario: Scenario) -> Result:
        with self._create_lock:
            result = self._pool.create(
                episode_id,
                env_id=scenario.env_id,
                env_class_name=scenario.env_class_name,
                init_config=scenario.init_config,
                checklist=scenario.checklist,
                checklist_id=scenario.task_id,
            )
            return result

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
