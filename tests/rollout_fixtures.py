"""Fakes and workload builders for the Phase 6 scheduler tests.

Three pieces the scheduler's unit tests need that must not spawn anything:

- `VirtualClock` + `FakeDispatcher` — environment results on a simulated
  timeline. The dispatcher's futures complete when the scheduler's injected
  `wait` advances the clock to their due time, so ordering and non-blocking
  assertions run against virtual milliseconds, not real sleeps.
- `TimedBackend` — charges virtual time for each generation call.
- `fixture_bindings` — real `ScenarioBinding`s over the pool tests' fixture
  scenarios, with the tool schemas the real metadata carries.

The real-template tokenizer comes from `tests/helpers.py` (`OfflineTokenizer`),
which now also decodes: the scheduler decodes sampled ids back to text through
the same instance that encoded them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from smolqwen.data.loader import ToolCall
from smolqwen.data.tool_call_xml import serialize_tool_call
from smolqwen.env.pool import Result
from smolqwen.env.registry import load_env_specs
from smolqwen.env.scenarios import Scenario, load_scenarios
from smolqwen.rollout.scheduler import ScenarioBinding, SchedulerConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"

FIXTURE_TASK = "env_151_rl-task_7"

# The correct action sequence for FIXTURE_TASK, derived from its checklist and
# init_config (it mirrors the selftest scenario). Each step satisfies exactly
# one check, so prefixes score exact fractions.
FIXTURE_SCRIPT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"}),
    (
        "update_enrollment_status",
        {
            "enrollment_id": "e1f02ad0-0e36-482d-aeee-6d170601b657",
            "new_status": "completed",
            "requesting_user_account_id": "acc-4211a7d1-cf6e-46bc-b781-c34d46e85508",
        },
    ),
    (
        "update_enrollment_status",
        {
            "enrollment_id": "0cc3569c-689c-4961-b957-30765e14a8e7",
            "new_status": "completed",
            "requesting_user_account_id": "acc-12ac0bab-bff6-4b61-b328-96d8761dac88",
        },
    ),
)


def xml_call(name: str, arguments: Mapping[str, Any]) -> str:
    """A well-formed tool-call turn, in the template's own serialization."""
    return serialize_tool_call(ToolCall(name=name, arguments=dict(arguments)))


def script_policy_texts(
    script: Sequence[tuple[str, Mapping[str, Any]]] = FIXTURE_SCRIPT,
    *,
    final_text: str = "All required updates are complete.",
) -> list[str]:
    """The turn texts a deterministic policy emits: script calls, then an answer."""
    # The Qwen3.5 generation prompt already ends in `<think>\n`, so these are
    # continuations, not standalone assistant messages. An empty reasoning body
    # still has to close the block before emitting the tool call / answer.
    return [f"\n</think>\n\n{xml_call(name, arguments)}" for name, arguments in script] + [
        f"\n</think>\n\n{final_text}"
    ]


def text_list_policy(texts: Sequence[str]) -> Callable[[str, int, Sequence[Any]], str]:
    """A policy whose turn k text is `texts[k]`, clamped to the last."""

    def policy(episode_id: str, turn_index: int, messages: Sequence[Any]) -> str:
        return texts[min(turn_index, len(texts) - 1)]

    return policy


class VirtualClock:
    """A monotonic clock the tests advance explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance_to(self, moment: float) -> None:
        self.now = max(self.now, moment)

    def advance_by(self, seconds: float) -> None:
        self.now += seconds


StepBehavior = Callable[[str, str, Mapping[str, Any]], tuple[float, Result]]


def ok_result(episode_id: str, value: Any) -> Result:
    return Result(episode_id, "ok", value=value)


def observation_for(name: str, arguments: Mapping[str, Any]) -> str:
    return f"result of {name}({json.dumps(arguments, sort_keys=True, default=str)})"


def default_step_behavior(
    episode_id: str, name: str, arguments: Mapping[str, Any]
) -> tuple[float, Result]:
    return 0.003, ok_result(episode_id, observation_for(name, arguments))


class FakeDispatcher:
    """`EnvDispatcher` on a virtual timeline; nothing here touches a process.

    `step_behavior(episode_id, name, arguments) -> (delay_s, Result)` decides
    every tool call's latency and outcome. `create_behavior` and
    `score_behavior` follow the same shape. The virtual `wait` completes the
    earliest due future among those the scheduler is waiting on and advances
    the clock to it — which is exactly what a real blocking wait does, minus
    the wall time.
    """

    def __init__(
        self,
        clock: VirtualClock,
        *,
        step_behavior: StepBehavior | None = None,
        create_behavior: Callable[[str], tuple[float, Result]] | None = None,
        score_behavior: Callable[[str], tuple[float, Result]] | None = None,
    ) -> None:
        self.clock = clock
        self.step_behavior = step_behavior or default_step_behavior
        self.create_behavior = create_behavior or (
            lambda episode_id: (0.005, ok_result(episode_id, {"tools": []}))
        )
        self.score_behavior = score_behavior or (
            lambda episode_id: (0.002, ok_result(episode_id, default_score_payload(1.0)))
        )
        self.pending: list[tuple[float, Future[Result], Result]] = []
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    # --- EnvDispatcher surface ---

    def submit_create(self, episode_id: str, binding: ScenarioBinding) -> Future[Result]:
        delay, result = self.create_behavior(episode_id)
        return self._schedule(delay, result)

    def submit_step(
        self, episode_id: str, name: str, arguments: Mapping[str, Any]
    ) -> Future[Result]:
        self.calls.append((episode_id, name, dict(arguments)))
        delay, result = self.step_behavior(episode_id, name, arguments)
        return self._schedule(delay, result)

    def submit_score(self, episode_id: str) -> Future[Result]:
        delay, result = self.score_behavior(episode_id)
        return self._schedule(delay, result)

    def submit_destroy(self, episode_id: str) -> Future[Result]:
        return self._schedule(0.0, ok_result(episode_id, True))

    # --- virtual time ---

    def wait(self, futures: Any, timeout: float | None = None) -> set[Future[Result]]:
        waiting = set(futures)
        due = sorted(
            (entry for entry in self.pending if entry[1] in waiting),
            key=lambda entry: entry[0],
        )
        if not due:
            return set()
        moment, future, result = due[0]
        self.pending = [entry for entry in self.pending if entry is not due[0]]
        self.clock.advance_to(moment)
        future.set_result(result)
        return {future}

    def _schedule(self, delay: float, result: Result) -> Future[Result]:
        future: Future[Result] = Future()
        self.pending.append((self.clock() + delay, future, result))
        return future


def default_score_payload(reward: float) -> dict[str, Any]:
    return {
        "reward": reward,
        "passed": 1 if reward >= 1.0 else 0,
        "total": 1,
        "name_errors": 0,
        "checks": [
            {"check_item": "fixture", "passed": reward >= 1.0, "reason": None, "error_type": None}
        ],
    }


class TimedBackend:
    """Wrap a backend so each generate call charges virtual time."""

    def __init__(self, inner: Any, clock: VirtualClock, *, duration_s: float = 0.010) -> None:
        self._inner = inner
        self._clock = clock
        self._duration_s = duration_s
        self.generate_calls: list[int] = []

    def generate(self, requests: Sequence[Any]) -> list[Any]:
        self.generate_calls.append(len(requests))
        self._clock.advance_by(self._duration_s)
        return list(self._inner.generate(requests))

    def bind(self, episode_id: str, messages: Sequence[Any]) -> None:
        if hasattr(self._inner, "bind"):
            self._inner.bind(episode_id, messages)

    def observe(self, episode_id: str, message: Mapping[str, Any]) -> None:
        if hasattr(self._inner, "observe"):
            self._inner.observe(episode_id, message)


def load_fixture_workload() -> tuple[dict[str, Any], dict[str, Scenario]]:
    """Env specs and scenarios from the pool tests' fixture files."""
    env_specs = load_env_specs(ENV_METADATA)
    scenarios = {s.task_id: s for s in load_scenarios(SCENARIOS)}
    return env_specs, scenarios


def fixture_bindings(
    *,
    episodes: int = 4,
    task_id: str = FIXTURE_TASK,
    num_generations: int = 2,
) -> list[ScenarioBinding]:
    """Group-contiguous bindings over the fixture scenario, RepeatSampler-style."""
    env_specs, scenarios = load_fixture_workload()
    scenario = scenarios[task_id]
    spec = env_specs[scenario.env_id]
    return [
        ScenarioBinding(
            scenario=scenario,
            group_index=position // num_generations,
            tool_schemas=tuple(spec.tools),
            env_introduction=spec.introduction(),
        )
        for position in range(episodes)
    ]


def fast_config(**overrides: Any) -> SchedulerConfig:
    """A scheduler config sized for tests; every field overridable.

    `max_model_len` is set far above any real render because the offline test
    tokenizer ids are per-character: a fixture prompt with fifteen tool
    schemas is tens of thousands of ids, and the budget guard must not fire on
    the tokenizer's accounting unit. The budget-exhausted terminal has its own
    test with an explicit small value.
    """
    defaults: dict[str, Any] = dict(
        generation_concurrency=4,
        max_env_steps=8,
        episode_timeout_s=600.0,
        max_new_tokens_per_step=2048,
        max_model_len=1_000_000,
        temperature=1.0,
        top_p=1.0,
    )
    defaults.update(overrides)
    return SchedulerConfig(**defaults)
