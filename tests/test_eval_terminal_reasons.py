"""A score computed from episodes that never generated is not a measurement.

Measured on a T4, not hypothesised. With `max_seq_length` set below EnvScaler's
rendered prompt -- its tool schemas alone run to ~4k tokens, ~6.8k at the widest env
-- every episode hit the window check at admission, terminated with zero generations,
and was still scored: the verifier grades the environment's *final state*, and an
untouched initial state is a valid state that scores whatever it scores.

The report read `score: 0.25255` beside `average_generated_tokens: 0.0`. The zero is
the tell, but it only implies the failure; nothing named it and nothing refused. So
two things exist now: `terminal_<reason>_rate` in every aggregate, and a WARNING when
no episode generated at all.

Not an exception. `evaluate` is also how a genuinely mute model gets measured, and
refusing to report that would be its own wrong answer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult
from smolqwen.eval.batched import evaluate_batched
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.trajectories import TrajectoryRecord
from smolqwen.rollout.rollout_func import encode_ids
from tests.helpers import OfflineTokenizer
from tests.test_eval_batched_agreement import scripted_backend


def _metric(reason: str | None, *, score: float = 0.5) -> TaskMetrics:
    return TaskMetrics(
        category="fixture",
        score=score,
        invalid_calls=0,
        steps=1,
        generated_tokens=4,
        truncated=False,
        terminal_reason=reason,
    )


def test_terminal_reasons_become_rates_with_their_denominator() -> None:
    aggregated = aggregate(
        [
            _metric("final_answer"),
            _metric("final_answer"),
            _metric("turn_cap"),
            _metric("step_cap"),
        ]
    )["fixture"]
    assert aggregated["terminal_final_answer_rate"] == 0.5
    assert aggregated["terminal_turn_cap_rate"] == 0.25
    assert aggregated["terminal_step_cap_rate"] == 0.25
    assert aggregated["terminal_reason_denominator"] == 4.0


def test_a_run_that_never_generated_says_so_in_the_metrics() -> None:
    """The T4 shape: every episode dead at admission, every one still scored.

    `terminal_step_cap_rate: 1.0` is the number that makes the run's own report say
    it. A zero token average only lets a reader infer it.
    """
    aggregated = aggregate([_metric("step_cap", score=0.38), _metric("step_cap", score=0.27)])[
        "fixture"
    ]
    assert aggregated["terminal_step_cap_rate"] == 1.0
    assert aggregated["score"] == pytest.approx(0.325)
    assert "terminal_final_answer_rate" not in aggregated


def test_reasons_are_absent_rather_than_zero_when_nothing_reported_one() -> None:
    """The serial path predates terminal reasons, so `None` must not become a rate."""
    aggregated = aggregate([_metric(None), _metric(None)])["fixture"]
    assert not [key for key in aggregated if key.startswith("terminal_")]


class _StalledAdapter:
    """An adapter whose prompt is longer than the window the engine is given.

    Not a contrived fake: this is the real EnvScaler shape at a small
    `max_seq_length`. The system message carries the environment introduction and the
    tool schemas render into the prefix, so the prompt is thousands of tokens before
    the model has said anything.
    """

    def __init__(self, *, prompt_tokens: int = 4000) -> None:
        self.prompt = "x " * prompt_tokens
        self.scored = 0

    def load_tasks(self) -> list[EvalTask]:
        return [EvalTask(f"task-{index}", "fixture", self.prompt, ()) for index in range(2)]

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if history:
            return [dict(message) for message in history]
        return [{"role": "system", "content": task.prompt}, {"role": "user", "content": "go"}]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        raise AssertionError("nothing should have been generated, so nothing should step")

    def score(self, task: EvalTask) -> AdapterResult:
        # The verifier grades final state. Nothing ran, so it grades the initial
        # state -- and returns a plausible number, which is the whole problem.
        self.scored += 1
        return AdapterResult(0.38, False)

    def invalid_call_count(self, task: EvalTask) -> int:
        return 0

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        return {}

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        return aggregate(tasks)


def test_the_batched_path_warns_when_no_episode_generated_anything(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tokenizer = OfflineTokenizer(token_size=1)
    adapter = _StalledAdapter()
    rendered = len(encode_ids(tokenizer, adapter.prompt))
    # The offline tokenizer is per-character, so the window is set from the measured
    # render rather than a guessed token count.
    config = EvalConfig(
        adapters=("fixture",),
        max_steps_per_task=4,
        profile=EvalConfig().profile.model_copy(update={"max_seq_length": rendered // 2}),
    )
    records: list[TrajectoryRecord] = []
    with caplog.at_level(logging.WARNING, logger="smolqwen.eval.batched"):
        metrics = evaluate_batched(
            config,
            adapter,
            backend=scripted_backend(tokenizer),
            tokenizer=tokenizer,
            records=records,
        )

    # The failure reproduces: scored, non-zero, nothing generated.
    assert adapter.scored == 2
    assert metrics["fixture"]["score"] == pytest.approx(0.38)
    assert metrics["fixture"]["average_generated_tokens"] == 0.0
    assert metrics["fixture"]["terminal_step_cap_rate"] == 1.0
    assert all(record.generation_turns == 0 for record in records)

    assert "no episode generated a single turn" in caplog.text
    # The message has to name the cause, or the next person re-derives it on a card.
    assert "max_seq_length" in caplog.text


def test_no_warning_when_generation_happened(caplog: pytest.LogCaptureFixture) -> None:
    """The negative control: a healthy run must stay quiet."""
    from tests.test_eval_batched_agreement import ScriptedAdapter, config

    tokenizer = OfflineTokenizer(token_size=1)
    with caplog.at_level(logging.WARNING, logger="smolqwen.eval.batched"):
        metrics = evaluate_batched(
            config(), ScriptedAdapter(), backend=scripted_backend(tokenizer), tokenizer=tokenizer
        )
    assert metrics["fixture"]["average_generated_tokens"] > 0
    assert metrics["fixture"]["terminal_final_answer_rate"] == 1.0
    assert "no episode generated" not in caplog.text
