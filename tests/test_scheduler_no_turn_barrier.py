"""A slow tool call blocks only its own episode, never the ready queue."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smolqwen.env.pool import Result
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, make_scheduler
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    FakeDispatcher,
    TimedBackend,
    VirtualClock,
    fast_config,
    fixture_bindings,
    observation_for,
    ok_result,
    script_policy_texts,
    text_list_policy,
)


def test_fast_episode_regenerates_before_slow_peer_finishes() -> None:
    clock = VirtualClock()

    def step_behavior(
        episode_id: str, name: str, arguments: Mapping[str, Any]
    ) -> tuple[float, Result]:
        delay = 0.200 if "@0" in episode_id else 0.010
        return delay, ok_result(episode_id, observation_for(name, arguments))

    dispatcher = FakeDispatcher(clock, step_behavior=step_behavior)
    tokenizer = OfflineTokenizer(token_size=1)
    turns = script_policy_texts()[:1] + [script_policy_texts()[-1]]
    backend = TimedBackend(
        ScriptedPolicyBackend(text_list_policy(turns), lambda text: encode_ids(tokenizer, text)),
        clock,
        duration_s=0.005,
    )
    scheduler = make_scheduler(
        backend=backend,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=fast_config(generation_concurrency=2),
        wait_for=dispatcher.wait,
    )
    scheduler.run(fixture_bindings(episodes=2))

    fast_generations = [
        at
        for at, episode_id, event in scheduler.events
        if episode_id.endswith("@1") and event == "generate"
    ]
    slow_observation = next(
        at
        for at, episode_id, event in scheduler.events
        if episode_id.endswith("@0") and event == "observation:ok"
    )
    assert len(fast_generations) == 2
    assert fast_generations[1] < slow_observation


def test_polling_a_valid_slow_create_does_not_trip_a_cycle_guard() -> None:
    clock = VirtualClock()
    dispatcher = FakeDispatcher(
        clock,
        create_behavior=lambda episode_id: (
            2.0,
            ok_result(episode_id, {"tools": []}),
        ),
    )

    def polling_wait(futures: Any, timeout: float | None = None) -> set[Any]:
        waiting = set(futures)
        clock.advance_by(timeout or 0.0)
        completed: set[Any] = set()
        retained = []
        for due, future, result in dispatcher.pending:
            if future in waiting and due <= clock():
                future.set_result(result)
                completed.add(future)
            else:
                retained.append((due, future, result))
        dispatcher.pending = retained
        return completed

    tokenizer = OfflineTokenizer(token_size=1)
    backend = TimedBackend(
        ScriptedPolicyBackend(
            text_list_policy([script_policy_texts()[-1]]),
            lambda text: encode_ids(tokenizer, text),
        ),
        clock,
    )
    scheduler = make_scheduler(
        backend=backend,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=fast_config(max_env_steps=1, episode_timeout_s=10.0),
        wait_for=polling_wait,
    )

    episodes = scheduler.run(fixture_bindings(episodes=1))

    assert clock() >= 2.0
    assert episodes[0].terminal_reason == "final_answer"
