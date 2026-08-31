"""Infrastructure crashes are replaced; policy timeouts are scored."""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from smolqwen.env.pool import Result
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, make_scheduler
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    FakeDispatcher,
    TimedBackend,
    VirtualClock,
    default_score_payload,
    fast_config,
    fixture_bindings,
    observation_for,
    ok_result,
    script_policy_texts,
    text_list_policy,
)


def _run(step_behavior: Any, *, episodes: int = 1) -> tuple[list[Any], Any]:
    clock = VirtualClock()
    dispatcher = FakeDispatcher(
        clock,
        step_behavior=step_behavior,
        score_behavior=lambda episode_id: (
            0.001,
            ok_result(episode_id, default_score_payload(0.25)),
        ),
    )
    tokenizer = OfflineTokenizer(token_size=1)
    turns = script_policy_texts()[:1] + [script_policy_texts()[-1]]
    backend = TimedBackend(
        ScriptedPolicyBackend(text_list_policy(turns), lambda text: encode_ids(tokenizer, text)),
        clock,
    )
    scheduler = make_scheduler(
        backend=backend,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=fast_config(),
        wait_for=dispatcher.wait,
    )
    return scheduler.run(fixture_bindings(episodes=episodes)), scheduler


def test_worker_crash_is_replaced_without_changing_the_return_position() -> None:
    crashed = False

    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        nonlocal crashed
        if not crashed:
            crashed = True
            return 0.001, Result(episode_id, "worker_crash", detail="fixture crash")
        return 0.001, ok_result(episode_id, observation_for(name, arguments))

    episodes, scheduler = _run(behavior)
    episode = episodes[0]
    assert episode.episode_id.endswith("@0~1")
    assert episode.replaced_episode_id is not None
    assert episode.group_index == 0
    assert episode.reward == 0.25
    assert any(event.startswith("replaced:worker_crash") for _, _, event in scheduler.events)


def test_worker_timeout_is_terminal_and_still_scored() -> None:
    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        return 0.001, Result(episode_id, "timeout", detail="tool exceeded budget")

    episodes, _ = _run(behavior)
    episode = episodes[0]
    assert episode.terminal_reason == "timeout"
    assert episode.reward == 0.25
    assert episode.replaced_episode_id is None


def test_replacement_preserves_grouped_multirow_alignment() -> None:
    crashed = False

    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        nonlocal crashed
        if episode_id.endswith("@1") and not crashed:
            crashed = True
            return 0.001, Result(episode_id, "worker_crash", detail="group fixture")
        return 0.001, ok_result(episode_id, observation_for(name, arguments))

    episodes, _ = _run(behavior, episodes=4)
    assert [episode.group_index for episode in episodes] == [0, 0, 1, 1]
    assert episodes[1].episode_id.endswith("@1~1")
    assert all(episode.reward == 0.25 for episode in episodes)


def test_worker_crash_replaces_every_episode_in_the_reported_blast_radius() -> None:
    crashed = False

    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        nonlocal crashed
        if episode_id.endswith("@0") and not crashed:
            crashed = True
            prefix = episode_id.rsplit("@", 1)[0]
            return 0.001, Result(
                episode_id,
                "worker_crash",
                detail="shared worker fixture",
                lost_episode_ids=(f"{prefix}@0", f"{prefix}@1"),
            )
        return 0.001, ok_result(episode_id, observation_for(name, arguments))

    episodes, scheduler = _run(behavior, episodes=2)

    assert [episode.episode_id.rsplit("@", 1)[1] for episode in episodes] == ["0~1", "1~1"]
    assert all(episode.reward == 0.25 for episode in episodes)
    replaced = [event for _, _, event in scheduler.events if event == "replaced:worker_crash"]
    assert len(replaced) == 2


def test_failure_cleanup_awaits_and_destroys_a_late_successful_create() -> None:
    class ThreadedDispatcher:
        def __init__(self) -> None:
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.live: set[str] = set()

        def submit_create(self, episode_id: str, binding: Any) -> Future[Result]:
            def create() -> Result:
                if episode_id.endswith("@1"):
                    time.sleep(0.1)
                self.live.add(episode_id)
                return ok_result(episode_id, {"tools": []})

            return self.executor.submit(create)

        def submit_step(self, episode_id: str, name: str, arguments: Any) -> Future[Result]:
            raise AssertionError("generation fails before a step")

        def submit_score(self, episode_id: str) -> Future[Result]:
            raise AssertionError("generation fails before scoring")

        def submit_destroy(self, episode_id: str) -> Future[Result]:
            def destroy() -> Result:
                self.live.discard(episode_id)
                return ok_result(episode_id, True)

            return self.executor.submit(destroy)

    class FailingBackend:
        def generate(self, requests: Any) -> list[Any]:
            raise RuntimeError("fixture generation failure")

    dispatcher = ThreadedDispatcher()
    tokenizer = OfflineTokenizer(token_size=1)
    scheduler = make_scheduler(
        backend=FailingBackend(),
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=fast_config(generation_concurrency=1),
    )
    try:
        with pytest.raises(RuntimeError, match="fixture generation failure"):
            scheduler.run(fixture_bindings(episodes=2))
        assert dispatcher.live == set()
    finally:
        dispatcher.executor.shutdown(wait=True)
