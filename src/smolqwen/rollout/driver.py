"""Rollout's `TurnDriver`: the worker pool behind the shared turn engine.

Everything specific to training rollout that the engine must not know: tool-call
XML parsing, dispatch to the isolated worker pool, verifier scoring, and the
crash-replacement path. The engine sees only `Advance` objects.

One asymmetry worth naming. A pool `Result` can report `worker_crash`, which is
not a terminal reason but an instruction to replace the episode -- a crashed
episode scored as a low reward would teach the model to avoid an infrastructure
failure it did not cause. The driver holds a back-reference to the engine for
exactly that, and for nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

from smolqwen.data.loader import Message
from smolqwen.env.parse import parse_turn
from smolqwen.env.pool import Result
from smolqwen.env.scenarios import Scenario
from smolqwen.inference.episode import Episode
from smolqwen.inference.turn_engine import Advance, TurnEngine, TurnEngineError

__all__ = ["EnvDispatcher", "RolloutDriver", "ScenarioBinding"]


@dataclass(frozen=True)
class ScenarioBinding:
    """Everything one rollout position needs beside its prompt.

    `tool_schemas` are the parent-side JSON schemas from `EnvSpec.tools` -- no env
    code is compiled in this process to obtain them.
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


class RolloutDriver:
    """Drives training rollout: XML tool calls, the worker pool, verifier reward."""

    def __init__(self, dispatcher: EnvDispatcher) -> None:
        self._dispatcher = dispatcher
        self._bindings: dict[str, ScenarioBinding] = {}
        self._engine: TurnEngine | None = None

    def attach(self, engine: TurnEngine) -> None:
        """Give the driver the engine it asks to replace crashed episodes on."""
        self._engine = engine

    # --- lifecycle ---

    def open(self, episode: Episode, binding: Any) -> Future[Result]:
        self._bindings[episode.episode_id] = binding
        return self._dispatcher.submit_create(episode.episode_id, binding)

    def opened(self, episode: Episode, payload: Any) -> Advance:
        result = self._require(episode, payload, "create")
        if self._replaced(episode, result):
            return Advance()
        if not result.ok:
            raise TurnEngineError(
                f"{episode.episode_id}: environment creation failed "
                f"({result.reason}): {result.detail}"
            )
        return Advance()

    def interpret(self, episode: Episode, text: str) -> Advance:
        binding = self._bindings[episode.episode_id]
        turn = parse_turn(text, available_tools=binding.tool_names)
        if turn.outcome == "no_call":
            return Advance(terminal="final_answer")
        if turn.is_invalid_call:
            # An invalid call becomes an observation the model can retry from, as
            # the released trajectories behave. It never reaches the reward.
            return Advance(
                observation=turn.observation(),
                observation_label="invalid",
                invalid_call=True,
            )
        assert turn.name is not None and turn.arguments is not None
        return Advance(
            pending=self._dispatcher.submit_step(
                episode.episode_id, turn.name, dict(turn.arguments)
            )
        )

    def resolve(self, episode: Episode, payload: Any) -> Advance:
        result = self._require(episode, payload, "step")
        if self._replaced(episode, result):
            return Advance()
        if result.reason == "timeout":
            return Advance(env_steps=1, terminal="timeout")
        if result.reason == "error" and (result.detail or "").startswith("unrecoverable:"):
            return Advance(env_steps=1, terminal="unrecoverable")
        if not result.ok:
            # An env exception becomes an observation the model can act on; the
            # step cap and the episode timeout bound the loop it could open.
            return Advance(
                observation=f"Error: env: {result.detail}",
                observation_label="error",
                env_steps=1,
                invalid_call=True,
            )
        return Advance(observation=str(result.value), observation_label="ok", env_steps=1)

    def score(self, episode: Episode) -> Future[Result]:
        return self._dispatcher.submit_score(episode.episode_id)

    def scored(self, episode: Episode, payload: Any) -> None:
        result = self._require(episode, payload, "score")
        if not result.ok:
            raise TurnEngineError(
                f"{episode.episode_id}: scoring failed ({result.reason}): {result.detail}"
            )
        value = result.value
        episode.reward = float(value["reward"])
        checks = value.get("checks") or []
        episode.per_check_bools = tuple(bool(check.get("passed")) for check in checks)

    def close(self, episode: Episode) -> Future[Result]:
        return self._dispatcher.submit_destroy(episode.episode_id)

    # --- helpers ---

    def _require(self, episode: Episode, payload: Any, action: str) -> Result:
        if not isinstance(payload, Result):
            raise TurnEngineError(
                f"{episode.episode_id}: {action} returned {type(payload).__name__}, not a Result"
            )
        return payload

    def _replaced(self, episode: Episode, result: Result) -> bool:
        """Ask the engine to replace every episode a crashed worker took with it."""
        if not result.is_infrastructure_failure:
            return False
        if self._engine is None:
            raise TurnEngineError(
                f"{episode.episode_id}: worker crash with no engine attached to replace it"
            )
        lost: Sequence[str] = (*result.lost_episode_ids, episode.episode_id)
        self._engine.replace(lost, "worker_crash")
        return True
