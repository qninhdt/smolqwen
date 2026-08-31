"""Every bounded scheduler exit reaches scoring and cleanup."""

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


def _episode(turns: list[str], *, step_behavior: Any = None, **config_overrides: Any) -> Any:
    clock = VirtualClock()
    dispatcher = FakeDispatcher(clock, step_behavior=step_behavior)
    tokenizer = OfflineTokenizer(token_size=1)
    backend = TimedBackend(
        ScriptedPolicyBackend(text_list_policy(turns), lambda text: encode_ids(tokenizer, text)),
        clock,
    )
    scheduler = make_scheduler(
        backend=backend,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=fast_config(**config_overrides),
        wait_for=dispatcher.wait,
    )
    return scheduler.run(fixture_bindings(episodes=1))[0]


def test_plain_answer_terminates_as_final_answer() -> None:
    episode = _episode([script_policy_texts()[-1]])
    assert episode.terminal_reason == "final_answer"
    assert episode.reward == 1.0


def test_step_cap_terminates_and_scores() -> None:
    episode = _episode([script_policy_texts()[0]], max_env_steps=1)
    assert episode.terminal_reason == "step_cap"
    assert episode.step_count == 1
    assert episode.reward == 1.0


def test_unrecoverable_step_terminates_and_scores() -> None:
    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        return 0.001, Result(episode_id, "error", detail="unrecoverable: state cannot continue")

    episode = _episode([script_policy_texts()[0]], step_behavior=behavior)
    assert episode.terminal_reason == "unrecoverable"
    assert episode.reward == 1.0


def test_episode_wall_clock_timeout_terminates_and_scores() -> None:
    def behavior(episode_id: str, name: str, arguments: Mapping[str, Any]) -> tuple[float, Result]:
        return 1.0, ok_result(episode_id, observation_for(name, arguments))

    episode = _episode(
        [script_policy_texts()[0]],
        step_behavior=behavior,
        episode_timeout_s=0.05,
    )
    assert episode.terminal_reason == "timeout"
    assert episode.reward == 1.0
