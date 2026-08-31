"""Scheduler results retain TRL's positional and group-contiguous contracts."""

from __future__ import annotations

import math
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.rollout.generation import (
    ScriptedPolicyBackend,
    VllmColocateBackend,
    _sampling_logprobs,
)
from smolqwen.rollout.rollout_func import (
    assemble_output,
    attach_prompt_messages,
    encode_ids,
    initial_messages_for,
    make_rollout_func,
    make_scheduler,
)
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    FakeDispatcher,
    TimedBackend,
    VirtualClock,
    default_score_payload,
    fast_config,
    fixture_bindings,
    ok_result,
    script_policy_texts,
    text_list_policy,
)


def test_returned_rows_and_groups_match_input_positions() -> None:
    clock = VirtualClock()
    dispatcher = FakeDispatcher(clock)
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
        config=fast_config(generation_concurrency=2),
        wait_for=dispatcher.wait,
    )
    bindings = fixture_bindings(episodes=4, num_generations=2)
    episodes = scheduler.run(bindings)
    output = assemble_output(episodes, scheduler)

    assert [episode.group_index for episode in episodes] == [0, 0, 1, 1]
    assert [episode.scenario_id for episode in episodes] == [
        binding.scenario.task_id for binding in bindings
    ]
    assert all(len(rows) == len(bindings) for rows in output.values())


def test_binding_uses_the_exact_prompt_row_trl_passed() -> None:
    binding = fixture_bindings(episodes=1)[0]
    prompts = [
        [
            {"role": "system", "content": "caller-owned system prompt"},
            {"role": "user", "content": binding.scenario.task},
        ]
    ]
    attached = attach_prompt_messages(prompts, [binding])[0]
    messages = initial_messages_for(attached)
    assert [message.content for message in messages] == [
        "caller-owned system prompt",
        binding.scenario.task,
    ]


def test_actual_rollout_func_closure_returns_rows_and_logs_profile() -> None:
    class ImmediateDispatcher:
        @staticmethod
        def _done(value: Any) -> Future[Any]:
            future: Future[Any] = Future()
            future.set_result(value)
            return future

        def submit_create(self, episode_id: str, binding: Any) -> Future[Any]:
            return self._done(ok_result(episode_id, {"tools": []}))

        def submit_step(self, episode_id: str, name: str, arguments: Any) -> Future[Any]:
            raise AssertionError("final-answer policy must not call a tool")

        def submit_score(self, episode_id: str) -> Future[Any]:
            return self._done(ok_result(episode_id, default_score_payload(0.5)))

        def submit_destroy(self, episode_id: str) -> Future[Any]:
            return self._done(ok_result(episode_id, True))

    bindings = fixture_bindings(episodes=2)
    prompts = [
        [
            {"role": "system", "content": f"system-{index}"},
            {"role": "user", "content": binding.scenario.task},
        ]
        for index, binding in enumerate(bindings)
    ]
    tokenizer = OfflineTokenizer(token_size=1)
    logs: list[dict[str, float | int]] = []
    trainer = SimpleNamespace(
        tools=[], environment_factories=None, log=lambda payload: logs.append(payload)
    )
    rollout = make_rollout_func(
        resolve_bindings=lambda received: bindings,
        config=fast_config(generation_concurrency=2),
        dispatcher=ImmediateDispatcher(),
        tokenizer=tokenizer,
        backend_factory=lambda _: ScriptedPolicyBackend(
            text_list_policy([script_policy_texts()[-1]]),
            lambda text: encode_ids(tokenizer, text),
        ),
    )
    output = rollout(prompts, trainer)

    assert all(len(rows) == len(prompts) for rows in output.values())
    assert logs and logs[0]["rollout/episodes_per_hour"] >= 0.0
    assert "rollout/timeline_scheduling_s" in logs[0]


def test_sampling_logprobs_select_the_sampled_candidate_and_keep_missing_nan() -> None:
    values = _sampling_logprobs(
        [7, 8, 9],
        [[-2.0, -0.2], [None], None],
        [[3, 7], [8], None],
    )
    assert values[:2] == [-0.2, math.nan]
    assert math.isnan(values[1]) and math.isnan(values[2])


def test_vllm_backend_uses_trls_generation_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trl.extras.profiling.profiling_context", lambda trainer, name: nullcontext()
    )

    class FakeGeneration:
        max_completion_length = 99

        def generate(self, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            assert kwargs["prompts"] == [[1, 2]]
            assert kwargs["images"] is None
            assert kwargs["num_generations"] == 1
            assert self.max_completion_length == 2
            return [[1, 2]], [[7, 8]], [[[-0.1], [-0.2]]], [[[7], [8]]]

    generation = FakeGeneration()
    trainer = SimpleNamespace(vllm_generation=generation)
    trainer.__dict__.update(
        state=SimpleNamespace(global_step=0),
        args=SimpleNamespace(report_to=[]),
        accelerator=SimpleNamespace(is_main_process=True),
    )
    backend = VllmColocateBackend(trainer)
    from smolqwen.rollout.generation import GenerationRequest

    result = backend.generate([GenerationRequest("e", (1, 2), 2)])[0]
    assert result.token_ids == (7, 8)
    assert result.logprobs == (-0.1, -0.2)
    assert generation.max_completion_length == 99
