"""One turn loop, driven by both training rollout and benchmark evaluation.

Both consumers advance concurrent episodes through the same cycle -- render,
generate, interpret, step the environment, append, repeat to terminal -- and both
previously owned a separate implementation of it. They are not, however, the same
loop by default. Five differences are structural, and each is a seam here rather
than a branch:

1. **Observation role and tool set are per turn, not per episode.** The template
   renders a `role: tool` message as `<tool_response>`-wrapped and a `role: user`
   message plainly, and the wrapping decides whether the reverse scan for
   `last_query_index` advances -- which decides whether earlier reasoning survives
   the next render. A benchmark that asks its next question as a user turn must
   therefore say so, and a benchmark that reveals a tool mid-episode must be able
   to change the rendered tool set. So the engine holds a mutable per-episode tool
   set and takes the role from each advance.

2. **Two budgets, not one.** `max_env_steps` counts executed environment methods.
   A model emitting prose executes none, so that counter never moves and the
   episode runs to `episode_timeout_s`. `max_generation_turns` bounds the
   generations themselves and terminates as `turn_cap`, kept separate so the two
   causes stay distinguishable in the terminal-reason tally.

3. **Liveness is not the mask-builder dictionary.** Failure cleanup previously
   derived the set of environments to destroy from `_builders`, which only the
   mask path populates. With mask building off, cleanup issued no destroys and
   every later run failed on a pool at capacity. `_live_episode_ids` is maintained
   by create and destroy directly.

4. **Admission is windowed.** Opening every position up front is correct for
   rollout -- TRL requires one row per prompt, positionally -- and fatal for
   evaluation, where a held-out set larger than the worker pool means the first
   task past capacity raises and the run scores zero. `max_in_flight` defaults to
   "all", so the rollout invariant is preserved exactly.

5. **Rendering is lazy.** Rendering every ready episode and then slicing to the
   generation width applied the chat template N times per cycle to use `width` of
   them. At evaluation scale that is an order of magnitude of redundant
   tokenization on one vCPU, which reads as environment latency in a profile.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from smolqwen.data.loader import Message
from smolqwen.inference.decoding import assistant_message
from smolqwen.inference.episode import Episode, TerminalReason
from smolqwen.inference.mask import EpisodeMaskBuilder
from smolqwen.rollout.generation import GenerationRequest, TurnTokens

__all__ = [
    "LENGTH_MARGIN",
    "MAX_REPLACEMENTS_PER_POSITION",
    "POLL_INTERVAL_S",
    "Advance",
    "TurnDriver",
    "TurnEngine",
    "TurnEngineConfig",
    "TurnEngineError",
    "completed",
]

# How long a cycle may spend reaping before it dispatches again. Not a timeout for
# any episode -- only the granularity at which completions are noticed.
POLL_INTERVAL_S = 0.02

# Ceiling on replacement attempts per position. A crash is random; a failure that
# reproduces deterministically on a fresh episode of the same scenario is not a
# crash but a bug, and looping forever would silently burn the run.
MAX_REPLACEMENTS_PER_POSITION = 3

# Head-room subtracted from `max_model_len` before comparing against a prefix or
# budgeting new tokens: the template's generation prompt and stop handling need a
# few positions beyond what the render already shows.
LENGTH_MARGIN = 8


class TurnEngineError(RuntimeError):
    """Raised when the engine cannot make progress or is misconfigured."""


ObservationRole = Literal["tool", "user"]


@dataclass(frozen=True)
class Advance:
    """What one interpreted model turn does to its episode.

    The driver returns this instead of the engine parsing the completion itself.
    Rollout's driver parses tool-call XML and dispatches to the worker pool; a
    benchmark's driver runs its own in-process step. Neither shape is privileged.
    """

    # Text to append as the next observation, or None to append nothing.
    observation: str | None = None
    # `tool` renders `<tool_response>`-wrapped and leaves `last_query_index` where
    # it is; `user` renders plainly and moves it, which drops earlier reasoning
    # from the next render. A benchmark's next question is a `user` turn.
    observation_role: ObservationRole = "tool"
    # What to call this observation in the event timeline. The role says how it
    # renders; this says what happened -- an environment result, an invalid call, an
    # environment error -- which is the distinction the profiler reads. The engine
    # cannot derive it, because "what happened" is exactly what the driver owns.
    observation_label: str | None = None
    # Executed environment methods, which is what `max_env_steps` bounds. A
    # completion marker or a clarification consumes a generation, not a step.
    env_steps: int = 0
    # The tool set the next render should carry, when the driver changes it.
    tools: tuple[Mapping[str, Any], ...] | None = None
    # Whether the episode is finished, and why.
    terminal: TerminalReason | None = None
    # Counts toward the invalid-call rate; never toward the reward.
    invalid_call: bool = False
    # A future the engine must await before this episode is ready again. The
    # synchronous case leaves it None and the episode returns to READY at once.
    pending: Future[Any] | None = None


class TurnDriver(Protocol):
    """Everything the engine cannot know: how an episode opens, advances, scores.

    Six methods, one per genuine difference between the two consumers. The engine
    never sees a `Result`, a parsed tool call, or a verifier payload -- rollout
    dispatches to an isolated worker pool and a benchmark runs its step in this
    process, and both are expressible here without the engine branching.
    """

    def open(self, episode: Episode, binding: Any) -> Future[Any] | None:
        """Begin the episode's environment. None means nothing to wait for."""
        ...

    def opened(self, episode: Episode, payload: Any) -> Advance:
        """Interpret what `open` produced. Terminal means the episode cannot run."""
        ...

    def interpret(self, episode: Episode, text: str) -> Advance:
        """One decoded model turn. May carry `pending` for asynchronous execution."""
        ...

    def resolve(self, episode: Episode, payload: Any) -> Advance:
        """Interpret a resolved `Advance.pending`."""
        ...

    def score(self, episode: Episode) -> Future[Any] | None:
        """Begin scoring a terminal episode."""
        ...

    def scored(self, episode: Episode, payload: Any) -> None:
        """Write the reward onto the episode."""
        ...

    def close(self, episode: Episode) -> Future[Any] | None:
        """Release the episode's environment."""
        ...


@dataclass(frozen=True)
class TurnEngineConfig:
    """Semantic knobs. Sizing lives in the profiles.

    `max_in_flight = None` means "every position at once", which is rollout's
    requirement: TRL wants one row per prompt and the engine must hold them all.
    Evaluation passes `min(generation_concurrency, pool_capacity)`.
    """

    generation_concurrency: int = 8
    max_env_steps: int = 16
    max_generation_turns: int = 20
    episode_timeout_s: float = 600.0
    max_new_tokens_per_step: int = 1024
    max_model_len: int = 16384
    temperature: float = 1.0
    top_p: float = 1.0
    fork_threshold_tokens: int = 1024
    max_in_flight: int | None = None
    build_masks: bool = True


@dataclass
class _Slot:
    """One return position: the current episode, its tools, its replacements."""

    position: int
    binding: Any
    episode: Episode
    started_at: float
    # The tool set the *next* render carries. Mutable per episode: a benchmark may
    # reveal a tool mid-episode, and rendering the turn-0 set forever would make
    # that reveal invisible to the model.
    tools: tuple[Mapping[str, Any], ...] = ()
    replacements: int = 0
    admitted: bool = False

    def fresh_episode(self) -> Episode:
        suffix = "" if self.replacements == 0 else f"~{self.replacements}"
        scenario_id = _scenario_id(self.binding)
        return Episode(
            episode_id=f"{scenario_id}@{self.position}{suffix}",
            scenario_id=scenario_id,
            group_index=getattr(self.binding, "group_index", 0),
            replaced_episode_id=(
                self.episode.episode_id if self.replacements > 0 and self.episode else None
            ),
        )


def _scenario_id(binding: Any) -> str:
    scenario = getattr(binding, "scenario", None)
    return str(getattr(scenario, "task_id", None) or getattr(binding, "task_id", binding))


class TurnEngine:
    """Drives every position to DONE and returns episodes in input order."""

    def __init__(
        self,
        *,
        backend: Any,
        driver: TurnDriver,
        initial_messages: Callable[[Any], list[Message]],
        render_prefix_ids: Callable[[Sequence[Message], Sequence[Mapping[str, Any]]], list[int]],
        decode: Callable[[Sequence[int]], str],
        config: TurnEngineConfig,
        clock: Callable[[], float] = time.monotonic,
        wait_for: Callable[..., set[Future[Any]]] | None = None,
    ) -> None:
        self._backend = backend
        self._driver = driver
        self._initial_messages = initial_messages
        self._render = render_prefix_ids
        self._decode = decode
        self._config = config
        self._clock = clock
        self._wait = wait_for or _default_wait
        # (clock time, episode id, event): the timeline the profiler and the
        # no-turn-barrier test read -- wall-clock under the real clock, virtual
        # under the simulated one.
        self.events: list[tuple[float, str, str]] = []
        self.stage_intervals: list[tuple[str, float, float]] = []
        self.queue_depth_samples: list[int] = []
        self.render_calls: int = 0
        self._futures: dict[Future[Any], tuple[_Slot, Episode, str]] = {}
        self._submitted_at: dict[Future[Any], float] = {}
        self._builders: dict[str, EpisodeMaskBuilder] = {}
        self._slots_by_episode_id: dict[str, _Slot] = {}
        # Liveness, maintained by open and close directly rather than derived from
        # `_builders`. With `build_masks=False` that dictionary is empty, and
        # cleanup reading it issued no destroys at all.
        self._live_episode_ids: set[str] = set()

    # --- public ---

    def run(self, bindings: Sequence[Any]) -> list[Episode]:
        """Run every binding to completion; the result aligns 1:1 with input."""
        if not bindings:
            return []
        slots = [
            _Slot(
                position=position,
                binding=binding,
                episode=Episode(
                    episode_id=f"{_scenario_id(binding)}@{position}",
                    scenario_id=_scenario_id(binding),
                    group_index=getattr(binding, "group_index", 0),
                ),
                started_at=self._clock(),
                tools=tuple(getattr(binding, "tool_schemas", ()) or ()),
            )
            for position, binding in enumerate(bindings)
        ]
        try:
            while any(slot.episode.state != "done" for slot in slots):
                self._admit(slots)
                self._reap()
                self._mark_timeouts(slots)
                self._generate_ready(slots)
                self._dispatch_scoring(slots)
                self._wait_for_any()
                self._assert_progress(slots)
        except BaseException:
            self._cleanup_after_failure()
            raise
        return [slot.episode for slot in slots]

    def episode_builder(self, episode_id: str) -> EpisodeMaskBuilder:
        """The mask builder `assemble_output` reads the returned arrays from."""
        return self._builders[episode_id]

    @property
    def live_episode_ids(self) -> frozenset[str]:
        """Environments this engine believes are still open. Cleanup reads this."""
        return frozenset(self._live_episode_ids)

    # --- admission ---

    def _admit(self, slots: Sequence[_Slot]) -> None:
        """Open as many un-admitted positions as the in-flight window allows.

        With `max_in_flight=None` every position is admitted on the first cycle,
        which is the invariant rollout depends on: TRL requires one returned row
        per prompt, so no position may wait. A finite window is what lets a
        held-out set larger than the worker pool run at all.
        """
        limit = self._config.max_in_flight
        for slot in slots:
            if slot.admitted:
                continue
            if limit is not None and self._in_flight(slots) >= limit:
                return
            self._open_slot(slot)

    def _in_flight(self, slots: Sequence[_Slot]) -> int:
        return sum(1 for slot in slots if slot.admitted and slot.episode.state != "done")

    def _open_slot(self, slot: _Slot) -> None:
        slot.admitted = True
        slot.started_at = self._clock()
        slot.episode.messages = list(self._initial_messages(slot.binding))
        self._slots_by_episode_id[slot.episode.episode_id] = slot
        self._log(slot.episode.episode_id, "open")
        self._begin_open(slot)

    def _begin_open(self, slot: _Slot) -> None:
        episode = slot.episode
        episode.state = "tool"
        future = self._driver.open(episode, slot.binding)
        if future is None:
            # A synchronous driver (a benchmark whose step runs in this process)
            # has nothing to await; complete the transition immediately.
            self._apply(slot, episode, self._driver.opened(episode, None), opening=True)
            return
        self._track(slot, episode, "create", future)

    # --- cycle steps ---

    def _reap(self) -> None:
        for future in [f for f in self._futures if f.done()]:
            slot, episode, action = self._futures.pop(future)
            started = self._submitted_at.pop(future)
            if slot.episode is not episode:
                # Another result from the same crashed worker may already have
                # replaced this slot. Consume the stale future without letting its
                # result (or exception) affect the fresh replacement.
                try:
                    future.result()
                except Exception:
                    pass
                continue
            try:
                payload = future.result()
            except Exception as exc:
                raise TurnEngineError(
                    f"{episode.episode_id}: {action} future raised instead of returning a payload"
                ) from exc
            finished = self._clock()
            self._record_stage(episode, action, started, finished)
            if action == "create":
                self._apply(slot, episode, self._driver.opened(episode, payload), opening=True)
            elif action == "step":
                self._apply(slot, episode, self._driver.resolve(episode, payload))
            elif action == "score":
                self._driver.scored(episode, payload)
                self._capture_mask(episode)
                self._begin_close(slot, episode)
            elif action == "destroy":
                self._live_episode_ids.discard(episode.episode_id)
                episode.state = "done"
                self._log(episode.episode_id, "done")
                self._log(episode.episode_id, "destroyed")

    _STAGE_NAMES = {
        "create": "env.create",
        "step": "env.step",
        "score": "verifier",
        "destroy": "env.destroy",
    }

    def _record_stage(self, episode: Episode, action: str, started: float, finished: float) -> None:
        stage = self._STAGE_NAMES.get(action)
        if stage is None:
            return
        episode.record_timing(stage, finished - started)
        self.stage_intervals.append((stage, started, finished))

    def _generate_ready(self, slots: Sequence[_Slot]) -> None:
        """Select up to the generation width, render only those, generate once.

        Rendering every ready episode and then slicing wasted one full chat-template
        application per non-selected episode, every cycle. At 80 tasks against a
        width of 8 that is ten renders to use one, on the single vCPU the
        environment workers also share -- so it reads as environment latency in a
        profile rather than as tokenization.
        """
        ready = [
            slot
            for slot in slots
            # `slot.admitted` is load-bearing, not defensive: `Episode.state`
            # defaults to "ready", so an un-admitted position would otherwise be
            # selected for generation with an empty message list. Under the old
            # open-everything-up-front model that state was unreachable.
            if slot.admitted
            and slot.episode.state == "ready"
            and slot.episode.terminal_reason is None
        ]
        self.queue_depth_samples.append(len(ready))
        if not ready:
            return

        candidates: list[tuple[_Slot, list[int], int]] = []
        for slot in ready:
            if len(candidates) >= self._config.generation_concurrency:
                break
            episode = slot.episode
            if episode.generation_count >= self._config.max_generation_turns:
                # Bounds a model that never calls a tool. `step_count` cannot: it
                # counts executed environment methods, of which prose executes none.
                self._terminal(slot, "turn_cap")
                continue
            if episode.step_count >= self._config.max_env_steps:
                self._terminal(slot, "step_cap")
                continue
            prefix = self._render_for(slot)
            if len(prefix) >= self._config.max_model_len - LENGTH_MARGIN:
                # The conversation fills the window: budget exhausted, the step-cap
                # family of terminals. Scored, never replaced -- the same scenario
                # would overflow again, deterministically.
                self._terminal(slot, "step_cap")
                self._log(episode.episode_id, "budget_exhausted")
                continue
            candidates.append((slot, prefix, self._token_budget(prefix)))

        if not candidates:
            return

        requests = []
        for slot, prefix, budget in candidates:
            episode = slot.episode
            episode.state = "generating"
            episode.generation_count += 1
            if self._config.build_masks:
                self._builders[episode.episode_id].open_turn(prefix)
            self._log(episode.episode_id, "generate")
            requests.append(
                GenerationRequest(
                    episode_id=episode.episode_id,
                    prompt_ids=tuple(prefix),
                    max_new_tokens=budget,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                )
            )

        started = self._clock()
        results = self._backend.generate(requests)
        finished = self._clock()
        self.stage_intervals.append(("generation", started, finished))
        for (slot, _, _), result in zip(candidates, results, strict=True):
            self._on_generation(slot, result, finished - started)

    def _render_for(self, slot: _Slot) -> list[int]:
        episode = slot.episode
        started = self._clock()
        prefix = self._render(episode.messages, slot.tools)
        finished = self._clock()
        self.render_calls += 1
        episode.record_timing("tokenization", finished - started)
        self.stage_intervals.append(("tokenization", started, finished))
        return prefix

    def _on_generation(self, slot: _Slot, result: TurnTokens, elapsed: float) -> None:
        episode = slot.episode
        episode.record_timing("generation", elapsed)
        parse_started = self._clock()
        text = self._decode(result.token_ids)
        advance = self._driver.interpret(episode, text)
        parse_finished = self._clock()
        episode.record_timing("parse", parse_finished - parse_started)
        self.stage_intervals.append(("parse", parse_started, parse_finished))

        message = assistant_message(text)
        episode.messages.append(message)
        self._mirror_scripted(episode, message)

        if self._config.build_masks:
            builder = self._builders[episode.episode_id]
            builder.append_response(result.token_ids, list(result.logprobs))
            episode.prompt_completion_boundary = builder.boundary
            episode.prompt_ids = list(builder.prompt_ids)

        self._apply(slot, episode, advance)

    # --- applying a driver's decision ---

    def _apply(
        self, slot: _Slot, episode: Episode, advance: Advance, *, opening: bool = False
    ) -> None:
        """The single place a driver's `Advance` changes episode state."""
        if opening and advance.terminal is None:
            self._live_episode_ids.add(episode.episode_id)
            if self._config.build_masks:
                self._builders[episode.episode_id] = EpisodeMaskBuilder(
                    self._render_for(slot),
                    fork_threshold_tokens=self._config.fork_threshold_tokens,
                )
            bind = getattr(self._backend, "bind", None)
            if bind is not None:
                bind(episode.episode_id, episode.messages)
            self._log(episode.episode_id, "ready")

        if advance.tools is not None:
            slot.tools = tuple(advance.tools)
        episode.step_count += advance.env_steps
        if advance.invalid_call:
            episode.invalid_call_count += 1

        if advance.observation is not None:
            self._append_observation(
                slot,
                advance.observation,
                role=advance.observation_role,
                label=advance.observation_label or advance.observation_role,
            )

        if advance.terminal is not None:
            self._terminal(slot, advance.terminal)
            if advance.pending is None:
                episode.state = "ready"
            return

        if advance.pending is not None:
            episode.state = "tool"
            self._track(slot, episode, "step", advance.pending)
            return
        episode.state = "ready"

    def _append_observation(
        self, slot: _Slot, observation: str, *, role: ObservationRole, label: str
    ) -> None:
        """Append an observation in the role the driver asked for.

        The role is not cosmetic. `tool` renders `<tool_response>`-wrapped, which
        the template's reverse scan skips, so earlier reasoning survives the next
        render. `user` renders plainly and moves `last_query_index`, which drops it.
        A benchmark's next question is genuinely a user turn and must render as one.
        """
        episode = slot.episode
        episode.observations.append(observation)
        message = Message(role=role, content=observation)
        episode.messages.append(message)
        self._mirror_scripted(episode, message)
        self._log(episode.episode_id, f"observation:{label}")

    # --- scoring and close ---

    def _dispatch_scoring(self, slots: Sequence[_Slot]) -> None:
        for slot in slots:
            episode = slot.episode
            if not slot.admitted or episode.terminal_reason is None:
                continue
            if episode.state == "done":
                continue
            if episode.state == "tool":
                # A step or open is still in flight. The pool's own timeout layers
                # resolve it; scoring waits rather than queueing behind a call that
                # may already be dead.
                continue
            episode.state = "tool"
            future = self._driver.score(episode)
            if future is None:
                self._driver.scored(episode, None)
                self._capture_mask(episode)
                self._begin_close(slot, episode)
                continue
            self._track(slot, episode, "score", future)

    def _begin_close(self, slot: _Slot, episode: Episode) -> None:
        # Destruction is part of completion: returning while the environment is
        # still live can exhaust the pool on the next call.
        episode.state = "tool"
        future = self._driver.close(episode)
        if future is None:
            self._live_episode_ids.discard(episode.episode_id)
            episode.state = "done"
            self._log(episode.episode_id, "done")
            self._log(episode.episode_id, "destroyed")
            return
        self._track(slot, episode, "destroy", future)

    def _capture_mask(self, episode: Episode) -> None:
        builder = self._builders.get(episode.episode_id)
        if builder is None:
            return
        episode.drift_tally = builder.tally
        episode.prompt_ids = list(builder.prompt_ids)
        episode.completion_ids = list(builder.completion_ids)
        episode.logprobs = list(builder.logprobs)
        episode.prompt_completion_boundary = builder.boundary
        episode.mask_spans = list(builder.spans)

    # --- replacement, timeouts, failure ---

    def replace(self, episode_ids: Sequence[str], reason: TerminalReason) -> None:
        """Retire the named episodes and re-admit their scenarios fresh.

        Called by a driver that discovers an infrastructure failure -- a crashed
        worker takes every episode it held. A crashed episode scored as a low reward
        would teach the model to avoid a failure it did not cause.
        """
        for episode_id in sorted(set(episode_ids)):
            slot = self._slots_by_episode_id.get(episode_id)
            if slot is None or slot.episode.episode_id != episode_id:
                continue
            self._log(episode_id, "worker_crash")
            self._replace(slot, reason)

    def _replace(self, slot: _Slot, reason: TerminalReason) -> None:
        old = slot.episode
        old.terminal_reason = reason
        old.state = "done"
        self._slots_by_episode_id.pop(old.episode_id, None)
        self._builders.pop(old.episode_id, None)
        self._live_episode_ids.discard(old.episode_id)
        self._log(old.episode_id, f"replaced:{reason}")
        if slot.replacements >= MAX_REPLACEMENTS_PER_POSITION:
            raise TurnEngineError(
                f"position {slot.position} exceeded {MAX_REPLACEMENTS_PER_POSITION} "
                f"replacements ({reason}); a deterministic failure is a bug, not a crash"
            )
        slot.replacements += 1
        slot.episode = slot.fresh_episode()
        slot.tools = tuple(getattr(slot.binding, "tool_schemas", ()) or ())
        slot.admitted = False
        self._log(slot.episode.episode_id, "replacement_open")
        # Re-admission goes through the window, so a replacement cannot push the
        # in-flight count past `max_in_flight`.
        self._open_slot(slot)

    def _mark_timeouts(self, slots: Sequence[_Slot]) -> None:
        now = self._clock()
        for slot in slots:
            episode = slot.episode
            if not slot.admitted or episode.terminal_reason is not None:
                continue
            if episode.state == "done":
                continue
            if now - slot.started_at > self._config.episode_timeout_s:
                self._terminal(slot, "timeout")

    def _cleanup_after_failure(self) -> None:
        """Drain bounded calls and destroy every environment still believed live.

        Liveness comes from `_live_episode_ids`, not from the mask-builder
        dictionary: with `build_masks=False` that dictionary is empty, so the
        previous derivation issued no destroys, leaked every environment, and made
        the *next* run fail on a pool at capacity rather than this one.

        A create may still be running when generation or scheduling raises, so
        pending futures are drained first to discover late successes.
        """
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
                    future.result()
                except Exception:
                    continue
                if action == "create":
                    self._live_episode_ids.add(episode.episode_id)
                elif action == "destroy":
                    self._live_episode_ids.discard(episode.episode_id)

        self._futures.clear()
        self._submitted_at.clear()
        cleanup: set[Future[Any]] = set()
        for episode_id in sorted(self._live_episode_ids):
            episode = Episode(episode_id=episode_id, scenario_id=episode_id, group_index=0)
            try:
                closing = self._driver.close(episode)
            except Exception:
                # A worker crash may already have removed the owner. Preserve the
                # original error; pool shutdown is the fallback.
                continue
            if closing is not None:
                cleanup.add(closing)
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
        self._live_episode_ids.clear()

    # --- helpers ---

    def _terminal(self, slot: _Slot, reason: TerminalReason) -> None:
        slot.episode.terminal_reason = reason
        self._log(slot.episode.episode_id, f"terminal:{reason}")

    def _mirror_scripted(self, episode: Episode, message: Message) -> None:
        observe = getattr(self._backend, "observe", None)
        if observe is not None:
            observe(episode.episode_id, message.to_template_dict())

    def _token_budget(self, prefix: Sequence[int]) -> int:
        room = self._config.max_model_len - len(prefix) - LENGTH_MARGIN
        return max(1, min(self._config.max_new_tokens_per_step, room))

    def _track(self, slot: _Slot, episode: Episode, action: str, future: Future[Any]) -> None:
        self._futures[future] = (slot, episode, action)
        self._submitted_at[future] = self._clock()

    def _wait_for_any(self) -> None:
        if self._futures:
            self._wait(self._futures.keys(), timeout=POLL_INTERVAL_S)

    def _assert_progress(self, slots: Sequence[_Slot]) -> None:
        if self._futures:
            return
        unfinished = [slot for slot in slots if slot.episode.state != "done"]
        if not unfinished:
            return
        if any(not slot.admitted for slot in unfinished):
            return
        if any(
            slot.episode.state == "ready" and slot.episode.terminal_reason is None
            for slot in unfinished
        ):
            return
        raise TurnEngineError(
            "turn engine has unfinished episodes but no ready work or in-flight calls"
        )

    def _log(self, episode_id: str, event: str) -> None:
        self.events.append((self._clock(), episode_id, event))


def _default_wait(futures: Any, timeout: float | None = None) -> set[Future[Any]]:
    done, _ = wait(list(futures), timeout=timeout)
    return done


def completed(value: Any) -> Future[Any]:
    """An already-resolved future, for a driver whose work is synchronous.

    A benchmark whose `step` runs in this process still has to express "here is
    the payload" in the shape the engine awaits. This keeps that adaptation in one
    place instead of every synchronous driver inventing it -- and it runs on the
    engine thread, never a pool thread, which matters because BFCL's adapter
    mutates `sys.path` on import.
    """
    future: Future[Any] = Future()
    future.set_result(value)
    return future
