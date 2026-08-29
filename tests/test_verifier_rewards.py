"""Verifier rewards: exact fractions, `initial_state` bound, K as the denominator.

Reward correctness is not testable by inequality. A verifier stuck at 0.0 satisfies
"a partial state scores below a correct one" while measuring nothing, so every
assertion here pins the exact value `passed / K` rounded to 4 places — upstream's
arithmetic (`base_env.py:301-305`).

The `initial_state` group is the one that gates the phase. `env_183_rl-task_29`
carries a check that reads `initial_state["meetings"]` as a global; called without
it the check raises `NameError`, which the design swallows as `False`. Measured
across the vendored release: 86 of 40,231 checks behave that way, spanning 81 of
2,550 scenarios. Silent reward depression on those is exactly the failure this
module's binding rules exist to prevent.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from smolqwen.env.instance import EnvInstance
from smolqwen.env.registry import EnvRegistry
from smolqwen.env.scenarios import load_scenarios
from smolqwen.env.verifier import (
    VerifierError,
    compile_checklist,
    score,
    score_raw,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"

CORRECT_SEQUENCE: tuple[tuple[str, dict[str, Any]], ...] = (
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


@pytest.fixture(scope="module")
def registry() -> EnvRegistry:
    return EnvRegistry.from_metadata(ENV_METADATA)


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Any]:
    return {scenario.task_id: scenario for scenario in load_scenarios(SCENARIOS)}


def _instance(registry: EnvRegistry, scenario: Any) -> EnvInstance:
    return EnvInstance.create(
        env_id=scenario.env_id,
        env_class=registry.env_class(scenario.env_id),
        env_class_name=scenario.env_class_name,
        init_config=scenario.init_config,
        tools=registry.tools(scenario.env_id),
    )


def test_a_correct_sequence_scores_exactly_one(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    scenario = scenarios["env_151_rl-task_7"]
    instance = _instance(registry, scenario)
    for name, arguments in CORRECT_SEQUENCE:
        instance.step(name, arguments)

    result = score_raw(
        scenario.task_id, scenario.checklist, instance.initial_state, instance.final_state()
    )
    assert result.reward == 1.0
    assert result.passed == result.total == 3
    assert all(check.passed for check in result.checks)


def test_a_partial_sequence_scores_the_exact_fraction(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    """One of three checks satisfied -> 1/3 rounded to 4 places, not "less than 1"."""
    scenario = scenarios["env_151_rl-task_7"]
    instance = _instance(registry, scenario)
    instance.step(*CORRECT_SEQUENCE[0][0:1], CORRECT_SEQUENCE[0][1])

    result = score_raw(
        scenario.task_id, scenario.checklist, instance.initial_state, instance.final_state()
    )
    assert result.reward == round(1 / 3, 4) == 0.3333
    assert result.passed == 1


def test_the_initial_state_scores_low(registry: EnvRegistry, scenarios: dict[str, Any]) -> None:
    scenario = scenarios["env_151_rl-task_7"]
    instance = _instance(registry, scenario)
    result = score_raw(
        scenario.task_id, scenario.checklist, instance.initial_state, instance.final_state()
    )
    assert result.reward == 0.0
    # Zero because the checks are unsatisfied, not because they errored -- the two
    # are indistinguishable in the reward and must not be in the detail.
    assert result.name_error_count == 0
    assert not result.reasons()


def test_a_check_reading_initial_state_resolves_it(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    scenario = scenarios["env_183_rl-task_29"]
    instance = _instance(registry, scenario)
    result = score_raw(
        scenario.task_id, scenario.checklist, instance.initial_state, instance.final_state()
    )
    assert result.name_error_count == 0, result.reasons()


def test_omitting_initial_state_is_what_a_name_error_looks_like(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    """The negative control: without it, the check fails silently as `False`.

    This is the defect the phase gates on, reproduced deliberately so the positive
    test above is known to be testing something.
    """
    scenario = scenarios["env_183_rl-task_29"]
    instance = _instance(registry, scenario)
    result = score_raw(scenario.task_id, scenario.checklist, None, instance.final_state())

    assert result.name_error_count == 1
    assert any("NameError" in reason for reason in result.reasons())
    # And the reward is depressed rather than erroring -- silently, which is why it
    # needs a dedicated counter.
    assert result.reward < 1.0


def test_the_same_scenario_scores_identically_across_two_episodes(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    """A GRPO group runs G rollouts of one scenario; they must be comparable."""
    scenario = scenarios["env_183_rl-task_29"]
    compiled = compile_checklist(scenario.task_id, scenario.checklist)

    rewards = []
    for _ in range(2):
        instance = _instance(registry, scenario)
        rewards.append(score(compiled, instance.initial_state, instance.final_state()).reward)
    assert rewards[0] == rewards[1]


def test_a_check_mutating_initial_state_cannot_affect_the_next_call() -> None:
    """Compile-once fixes `__globals__`, so the binding must be refreshed per call."""
    checklist = [
        {
            "check_item": "mutates initial_state then reads it",
            "check_func": (
                "def check_func(final_state):\n"
                "    initial_state['rows'].clear()\n"
                "    return len(initial_state['rows']) == 0\n"
            ),
        },
        {
            "check_item": "requires initial_state to still hold its rows",
            "check_func": (
                "def check_func(final_state):\n    return len(initial_state['rows']) == 2\n"
            ),
        },
    ]
    initial = {"rows": {"a": 1, "b": 2}}
    compiled = compile_checklist("mutation", checklist)

    first = score(compiled, initial, {})
    assert [check.passed for check in first.checks] == [True, True]
    # The caller's dict is untouched, and a second episode sees the same thing.
    assert initial == {"rows": {"a": 1, "b": 2}}
    second = score(compiled, initial, {})
    assert [check.passed for check in second.checks] == [True, True]


def test_each_check_gets_its_own_globals_dict() -> None:
    """11 released checks define module-level helpers that resolve through globals.

    One shared dict would let a helper defined by check 1 leak into check 2's
    namespace -- and a name collision between two scenarios' helpers would make the
    second silently call the first's.
    """
    checklist = [
        {
            "check_item": "defines a helper",
            "check_func": (
                "def helper():\n    return 1\n\n"
                "def check_func(final_state):\n    return helper() == 1\n"
            ),
        },
        {
            "check_item": "must not see the other check's helper",
            "check_func": ("def check_func(final_state):\n    return 'helper' not in globals()\n"),
        },
    ]
    result = score(compile_checklist("helpers", checklist), {}, {})
    assert [check.passed for check in result.checks] == [True, True]


def test_a_module_level_helper_still_sees_initial_state() -> None:
    """A helper resolving `initial_state` through its own globals must work."""
    checklist = [
        {
            "check_item": "helper reads initial_state",
            "check_func": (
                "def original_count():\n    return len(initial_state['rows'])\n\n"
                "def check_func(final_state):\n"
                "    return len(final_state['rows']) > original_count()\n"
            ),
        }
    ]
    result = score(
        compile_checklist("helper-initial", checklist),
        {"rows": {"a": 1}},
        {"rows": {"a": 1, "b": 2}},
    )
    assert result.reward == 1.0
    assert result.name_error_count == 0


def test_final_state_mutation_by_a_check_does_not_reach_the_caller() -> None:
    checklist = [
        {
            "check_item": "clears final_state",
            "check_func": (
                "def check_func(final_state):\n    final_state.clear()\n    return True\n"
            ),
        },
        {
            "check_item": "final_state must still be intact",
            "check_func": "def check_func(final_state):\n    return 'rows' in final_state\n",
        },
    ]
    final = {"rows": {"a": 1}}
    result = score(compile_checklist("final-mutation", checklist), {}, final)
    assert [check.passed for check in result.checks] == [True, True]
    assert final == {"rows": {"a": 1}}


def test_an_empty_checklist_raises_rather_than_returning_zero() -> None:
    """`0/0` has no reward; returning 0.0 would be an unearned low score."""
    with pytest.raises(VerifierError, match="reward is undefined"):
        score(compile_checklist("empty", []), {}, {})


def test_scoring_does_not_recompile(registry: EnvRegistry, scenarios: dict[str, Any]) -> None:
    """Compilation belongs at scenario load, never in the reward path."""
    scenario = scenarios["env_151_rl-task_7"]
    compiled = compile_checklist(scenario.task_id, scenario.checklist)
    assert compiled.exec_count == scenario.check_count

    instance = _instance(registry, scenario)
    for _ in range(3):
        score(compiled, instance.initial_state, instance.final_state())
    assert compiled.exec_count == scenario.check_count


def test_the_caller_s_initial_state_is_never_mutated(
    registry: EnvRegistry, scenarios: dict[str, Any]
) -> None:
    scenario = scenarios["env_183_rl-task_29"]
    instance = _instance(registry, scenario)
    before = deepcopy(instance.initial_state)
    score_raw(scenario.task_id, scenario.checklist, instance.initial_state, instance.final_state())
    assert instance.initial_state == before
