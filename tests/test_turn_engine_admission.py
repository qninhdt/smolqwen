"""Two properties that only appear at evaluation scale, and one leak.

**Admission.** Opening every position up front is right for rollout: TRL requires
one returned row per prompt, positionally, so no position may wait. It is fatal for
evaluation. The shipped configs hold 80 held-out tasks (10 environments x 8
scenarios) against a 32-episode pool (4 workers x 8), so the 33rd `create` raises
`PoolError` and the whole run scores zero -- not a degraded number, a zero.

**Cleanup liveness.** Failure cleanup derived the set of environments to destroy
from the mask-builder dictionary. Evaluation does not build masks, so that
dictionary is empty, cleanup issued no destroys, and every environment leaked --
which surfaces on the *next* run as a pool at capacity rather than on this one.

Both are engine-level, so both are tested against a fake pool that enforces the
same capacity rule the real one does.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest

from smolqwen.inference.episode import Episode
from smolqwen.inference.turn_engine import Advance, TurnEngine, TurnEngineConfig, completed
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, initial_messages_for
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    fixture_bindings,
    observation_for,
    script_policy_texts,
    text_list_policy,
)


class CapacityError(RuntimeError):
    """What the real pool raises when every worker is full."""


class BoundedPoolDriver:
    """A synchronous driver over a pool with a hard capacity, like the real one.

    `WorkerPool.create` raises once every worker holds `episodes_per_worker`
    episodes, deliberately -- silent queueing would break the ready-queue model. So
    a driver that admits more positions than capacity must fail here, which is what
    makes the admission test meaningful rather than a restatement of the window.
    """

    def __init__(
        self, capacity: int, *, tool_names: frozenset[str], fail_at: int | None = None
    ) -> None:
        self.capacity = capacity
        self._tool_names = tool_names
        self._fail_at = fail_at
        self.live: set[str] = set()
        self.peak_live = 0
        self.open_calls = 0
        self.destroyed: list[str] = []

    def open(self, episode: Episode, binding: Any) -> Future[Any]:
        if len(self.live) >= self.capacity:
            raise CapacityError(
                f"every worker is full at {self.capacity} episodes; "
                f"{episode.episode_id} cannot be created"
            )
        self.open_calls += 1
        self.live.add(episode.episode_id)
        self.peak_live = max(self.peak_live, len(self.live))
        return completed(None)

    def opened(self, episode: Episode, payload: Any) -> Advance:
        return Advance()

    def interpret(self, episode: Episode, text: str) -> Advance:
        from smolqwen.env.parse import parse_turn

        if self._fail_at is not None and self.open_calls >= self._fail_at:
            raise RuntimeError("injected failure mid-run")
        turn = parse_turn(text, available_tools=self._tool_names)
        if turn.outcome == "no_call":
            return Advance(terminal="final_answer")
        assert turn.name is not None and turn.arguments is not None
        return Advance(pending=completed((turn.name, dict(turn.arguments))))

    def resolve(self, episode: Episode, payload: Any) -> Advance:
        name, arguments = payload
        return Advance(
            observation=observation_for(name, arguments), observation_label="ok", env_steps=1
        )

    def score(self, episode: Episode) -> Future[Any]:
        return completed(1.0)

    def scored(self, episode: Episode, payload: Any) -> None:
        episode.reward = float(payload)

    def close(self, episode: Episode) -> Future[Any]:
        self.live.discard(episode.episode_id)
        self.destroyed.append(episode.episode_id)
        return completed(None)


def build_engine(
    *, driver: Any, config: TurnEngineConfig, tokenizer: OfflineTokenizer | None = None
) -> TurnEngine:
    from smolqwen.data.render import render_prefix
    from smolqwen.inference.decoding import decode_completion

    tokenizer = tokenizer or OfflineTokenizer(token_size=1)

    def render_prefix_ids(messages: Any, tools: Any) -> list[int]:
        text = render_prefix(
            tokenizer, messages, tools=[dict(tool) for tool in tools], add_generation_prompt=True
        )
        return encode_ids(tokenizer, text)

    return TurnEngine(
        backend=ScriptedPolicyBackend(
            text_list_policy(script_policy_texts()), lambda text: encode_ids(tokenizer, text)
        ),
        driver=driver,
        initial_messages=initial_messages_for,
        render_prefix_ids=render_prefix_ids,
        decode=lambda ids: decode_completion(tokenizer, list(ids)),
        config=config,
    )


def engine_config(**overrides: Any) -> TurnEngineConfig:
    defaults: dict[str, Any] = {
        "generation_concurrency": 4,
        "max_env_steps": 8,
        "max_generation_turns": 12,
        "max_new_tokens_per_step": 2048,
        "max_model_len": 1_000_000,
        "build_masks": False,
    }
    defaults.update(overrides)
    return TurnEngineConfig(**defaults)


CAPACITY = 4
TASKS = 12


def test_more_tasks_than_pool_capacity_completes_under_a_window() -> None:
    """The shipped ratio, scaled down: 12 tasks against a 4-episode pool."""
    bindings = fixture_bindings(episodes=TASKS)
    driver = BoundedPoolDriver(CAPACITY, tool_names=bindings[0].tool_names)
    engine = build_engine(driver=driver, config=engine_config(max_in_flight=CAPACITY))

    episodes = engine.run(bindings)

    assert len(episodes) == TASKS
    assert all(episode.state == "done" for episode in episodes)
    assert all(episode.reward == 1.0 for episode in episodes)
    assert driver.peak_live <= CAPACITY, (
        f"the window admitted {driver.peak_live} concurrent episodes against a "
        f"capacity of {CAPACITY}"
    )
    assert not driver.live


def test_the_same_workload_without_a_window_hits_the_capacity_error() -> None:
    """The negative control. Without it the test above proves nothing about the
    window -- a pool with no real limit would pass either way."""
    bindings = fixture_bindings(episodes=TASKS)
    driver = BoundedPoolDriver(CAPACITY, tool_names=bindings[0].tool_names)
    engine = build_engine(driver=driver, config=engine_config(max_in_flight=None))

    with pytest.raises(CapacityError, match="every worker is full"):
        engine.run(bindings)


def test_rollout_keeps_every_position_live_when_the_window_is_absent() -> None:
    """`max_in_flight=None` is rollout's invariant: TRL wants one row per prompt,
    so every position must be admitted on the first cycle."""
    bindings = fixture_bindings(episodes=6)
    driver = BoundedPoolDriver(len(bindings), tool_names=bindings[0].tool_names)
    engine = build_engine(driver=driver, config=engine_config(max_in_flight=None))

    episodes = engine.run(bindings)

    assert driver.peak_live == len(bindings)
    assert [episode.episode_id for episode in episodes] == [
        f"{binding.scenario.task_id}@{position}" for position, binding in enumerate(bindings)
    ]


def test_failure_cleanup_releases_environments_with_mask_building_off() -> None:
    """The leak that surfaced one run later.

    Cleanup derived liveness from the mask-builder dictionary, which is empty when
    `build_masks=False`. So it destroyed nothing, every environment stayed open, and
    the *next* run failed on a pool at capacity -- a symptom one full run removed
    from its cause.
    """
    bindings = fixture_bindings(episodes=4)
    driver = BoundedPoolDriver(
        len(bindings), tool_names=bindings[0].tool_names, fail_at=len(bindings)
    )
    engine = build_engine(driver=driver, config=engine_config(build_masks=False))

    with pytest.raises(RuntimeError, match="injected failure"):
        engine.run(bindings)

    assert not driver.live, f"{len(driver.live)} environment(s) leaked: {sorted(driver.live)}"
    assert not engine.live_episode_ids
    assert sorted(driver.destroyed) == sorted(
        f"{binding.scenario.task_id}@{position}" for position, binding in enumerate(bindings)
    )


def test_rendering_happens_once_per_generated_turn_not_once_per_ready_episode() -> None:
    """Rendering every ready episode and slicing to the width wasted the difference.

    At 12 tasks against a width of 4, the eager form applied the chat template to
    every ready episode each cycle to use four of them. On the single vCPU the
    environment workers share, that reads as environment latency in a profile rather
    than as tokenization -- which is how it would be misdiagnosed.

    The bound: one render per generation, plus one per episode at open time for the
    mask builder's seed (off here), plus at most one wasted render per episode that
    turns out to be over its context budget.
    """
    bindings = fixture_bindings(episodes=TASKS)
    driver = BoundedPoolDriver(CAPACITY, tool_names=bindings[0].tool_names)
    engine = build_engine(driver=driver, config=engine_config(max_in_flight=CAPACITY))

    episodes = engine.run(bindings)

    generations = sum(episode.generation_count for episode in episodes)
    assert generations > 0
    assert engine.render_calls == generations, (
        f"{engine.render_calls} renders for {generations} generations; rendering is no longer lazy"
    )
