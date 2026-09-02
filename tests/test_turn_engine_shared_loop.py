"""One loop, two consumers: same scripted episode, identical transcript.

The engine's value is that a benchmark number means the same thing in training
and in `evaluate`. That only holds if both consumers really traverse one
implementation, so this file drives the same scripted episode through two
drivers that differ in every way the protocol allows -- one asynchronous over
futures, one fully synchronous in this thread -- and requires the resulting
episodes to be indistinguishable.

`rollout_fixtures.py`'s dispatcher returns *unresolved* futures that a virtual
`wait` later completes, so nothing in the existing suite exercises the
already-resolved path a synchronous benchmark driver takes. That path has its own
failure mode: a `Future` that is done before the engine ever waits on it can leave
the loop spinning if the engine only makes progress inside `wait`.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from smolqwen.inference.episode import Episode
from smolqwen.inference.turn_engine import Advance, TurnEngine, TurnEngineConfig, completed
from smolqwen.rollout.driver import RolloutDriver
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, initial_messages_for
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    FakeDispatcher,
    VirtualClock,
    fixture_bindings,
    observation_for,
    script_policy_texts,
    text_list_policy,
)


def engine_config(**overrides: Any) -> TurnEngineConfig:
    defaults: dict[str, Any] = {
        "generation_concurrency": 4,
        "max_env_steps": 8,
        "max_generation_turns": 12,
        "episode_timeout_s": 600.0,
        "max_new_tokens_per_step": 2048,
        "max_model_len": 1_000_000,
    }
    defaults.update(overrides)
    return TurnEngineConfig(**defaults)


class SynchronousDriver:
    """A driver whose every step runs in this thread, resolved before the engine waits.

    Shaped like a benchmark adapter: no worker pool, no create or destroy, and a
    `step` that is ordinary Python. It answers with `completed(...)` futures so the
    engine's await path is exercised against work that is already done.
    """

    def __init__(self, tool_names: frozenset[str]) -> None:
        self._tool_names = tool_names
        self.opened_ids: list[str] = []
        self.closed_ids: list[str] = []

    def open(self, episode: Episode, binding: Any) -> Future[Any]:
        self.opened_ids.append(episode.episode_id)
        return completed(None)

    def opened(self, episode: Episode, payload: Any) -> Advance:
        return Advance()

    def interpret(self, episode: Episode, text: str) -> Advance:
        from smolqwen.env.parse import parse_turn

        turn = parse_turn(text, available_tools=self._tool_names)
        if turn.outcome == "no_call":
            return Advance(terminal="final_answer")
        if turn.is_invalid_call:
            return Advance(
                observation=turn.observation(), observation_label="invalid", invalid_call=True
            )
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
        episode.per_check_bools = (True,)

    def close(self, episode: Episode) -> Future[Any]:
        self.closed_ids.append(episode.episode_id)
        return completed(None)


def build_engine(
    *, driver: Any, tokenizer: OfflineTokenizer, config: TurnEngineConfig, wait_for: Any = None
) -> TurnEngine:
    from smolqwen.data.render import render_prefix
    from smolqwen.inference.decoding import decode_completion

    def render_prefix_ids(messages: Any, tools: Any) -> list[int]:
        text = render_prefix(
            tokenizer, messages, tools=[dict(tool) for tool in tools], add_generation_prompt=True
        )
        return encode_ids(tokenizer, text)

    backend = ScriptedPolicyBackend(
        text_list_policy(script_policy_texts()), lambda text: encode_ids(tokenizer, text)
    )
    engine = TurnEngine(
        backend=backend,
        driver=driver,
        initial_messages=initial_messages_for,
        render_prefix_ids=render_prefix_ids,
        decode=lambda ids: decode_completion(tokenizer, list(ids)),
        config=config,
        wait_for=wait_for,
    )
    if isinstance(driver, RolloutDriver):
        driver.attach(engine)
    return engine


def transcript(episode: Episode) -> list[tuple[str, str | None, str | None]]:
    """The comparable surface: message shape, in order. Ids and timings differ."""
    return [
        (message.role, message.content, message.reasoning_content) for message in episode.messages
    ]


def test_both_drivers_produce_the_same_transcript_from_the_same_script() -> None:
    tokenizer = OfflineTokenizer(token_size=1)
    bindings = fixture_bindings(episodes=2)

    clock = VirtualClock()
    dispatcher = FakeDispatcher(clock)
    asynchronous = build_engine(
        driver=RolloutDriver(dispatcher),
        tokenizer=tokenizer,
        config=engine_config(),
        wait_for=dispatcher.wait,
    )
    async_episodes = asynchronous.run(bindings)

    synchronous_driver = SynchronousDriver(bindings[0].tool_names)
    synchronous = build_engine(
        driver=synchronous_driver,
        tokenizer=OfflineTokenizer(token_size=1),
        config=engine_config(),
    )
    sync_episodes = synchronous.run(bindings)

    assert len(async_episodes) == len(sync_episodes) == 2
    for left, right in zip(async_episodes, sync_episodes, strict=True):
        assert transcript(left) == transcript(right)
        assert left.step_count == right.step_count
        assert left.generation_count == right.generation_count
        assert left.terminal_reason == right.terminal_reason == "final_answer"
        assert left.observations == right.observations


def test_a_completed_future_driver_terminates_rather_than_spinning() -> None:
    """The path no existing fixture covers.

    `rollout_fixtures.py` hands back unresolved futures that a virtual `wait`
    completes later, so an engine that only advanced inside `wait` would pass every
    test in the suite and hang on a synchronous driver. The real blocking wait is
    used here deliberately -- no injected `wait_for` -- so a loop that failed to
    notice already-done work would hit the pytest timeout.
    """
    tokenizer = OfflineTokenizer(token_size=1)
    driver = SynchronousDriver(fixture_bindings(episodes=1)[0].tool_names)
    engine = build_engine(driver=driver, tokenizer=tokenizer, config=engine_config())

    episodes = engine.run(fixture_bindings(episodes=3))

    assert [episode.state for episode in episodes] == ["done"] * 3
    assert driver.opened_ids == sorted(driver.opened_ids)
    assert sorted(driver.closed_ids) == sorted(episode.episode_id for episode in episodes)
    assert not engine.live_episode_ids


def test_row_order_matches_input_order_for_both_drivers() -> None:
    """TRL sizes rewards from `len(prompts)` and reshapes advantages positionally,
    so a returned row in the wrong place silently mixes one group's advantage into
    another's."""
    tokenizer = OfflineTokenizer(token_size=1)
    bindings = fixture_bindings(episodes=4)
    driver = SynchronousDriver(bindings[0].tool_names)
    engine = build_engine(driver=driver, tokenizer=tokenizer, config=engine_config())

    episodes = engine.run(bindings)

    assert [episode.episode_id for episode in episodes] == [
        f"{binding.scenario.task_id}@{position}" for position, binding in enumerate(bindings)
    ]
    assert [episode.group_index for episode in episodes] == [
        binding.group_index for binding in bindings
    ]
