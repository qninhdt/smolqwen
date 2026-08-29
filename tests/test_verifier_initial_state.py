"""`initial_state` reaches every check that needs it — across the real release.

The dedicated gate for the failure that produces no error: a check reading
`initial_state` as a global, called without it, raises `NameError`, the design
swallows that as `False`, and reward is silently depressed. Phase 7's difficulty
profiler then classifies the affected scenarios `always_zero` and the curriculum
drops them, shrinking and skewing the RL training set.

Marked `dataset` so CI (which has no release files) deselects it: the assertion
that matters most here is over the *whole* vendored scenario file, not a fixture,
because a fixture built by the same hand that wrote the binding proves nothing
about the 2,550 scenarios that will actually train.

Measured while writing this: 86 of the 40,231 released checks raise `NameError`
without `initial_state`, spanning 81 of 2,550 scenarios (3.2%). An AST pass finds
105 checks / 97 scenarios that reference the name on some branch. The plan's
"1,208 / 779 / 30.5%" is a text match over sources including comments — the
mitigation is required either way, but the true blast radius is about 8x smaller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.env.scenarios import load_scenarios
from smolqwen.env.verifier import compile_checklist, score

pytestmark = pytest.mark.dataset

VENDORED_SCENARIOS = Path(
    "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/"
    "envscaler_rl_scenario_metadata.json"
)
VENDORED_ENVS = Path(
    "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/191_env_metadata.json"
)

# Enough to cover the affected population several times over without making the
# suite slow: the NameError rate is ~3.2% of scenarios, so 400 samples would
# contain roughly a dozen if the binding were broken.
SAMPLE_SIZE = 400


@pytest.fixture(scope="module")
def scenarios() -> list[Any]:
    if not VENDORED_SCENARIOS.is_file():
        pytest.skip(f"{VENDORED_SCENARIOS} not present")
    return load_scenarios(VENDORED_SCENARIOS)


def test_zero_checks_fail_with_name_error_across_a_scenario_sample(
    scenarios: list[Any],
) -> None:
    """The phase gate. Scored against each scenario's own initial state.

    Using `initial_state` as the final state too is deliberate: it makes every
    "did X change" check fail, so any `NameError` that surfaces is about name
    resolution rather than about the state being wrong.
    """
    import random

    sample = random.Random(1234).sample(scenarios, min(SAMPLE_SIZE, len(scenarios)))
    offenders: list[tuple[str, tuple[str, ...]]] = []
    for scenario in sample:
        compiled = compile_checklist(scenario.task_id, scenario.checklist)
        result = score(compiled, scenario.init_config, dict(scenario.init_config))
        if result.name_error_count:
            offenders.append((scenario.task_id, result.reasons()))

    assert not offenders, f"{len(offenders)} scenario(s) raised NameError: {offenders[:3]}"


def test_the_affected_population_is_what_the_code_claims(scenarios: list[Any]) -> None:
    """Pins the measured rate, so a data refresh that changes it is visible.

    Not a round number by choice: it is what executing every released check with
    and without `initial_state` reports. A future release changing this should fail
    here rather than quietly shift what the verifier docstring claims.
    """
    from copy import deepcopy

    unbound_failures = 0
    affected_scenarios = 0
    for scenario in scenarios:
        hit = False
        for entry in scenario.checklist:
            namespace: dict[str, Any] = {"__builtins__": __builtins__}
            try:
                exec(entry["check_func"], namespace)
                function = namespace.get("check_func")
                if not callable(function):
                    continue
                function(deepcopy(dict(scenario.init_config)))
            except NameError as exc:
                if "initial_state" in str(exc):
                    unbound_failures += 1
                    hit = True
            except Exception:  # noqa: BLE001 - any other error is not this test's subject
                continue
        affected_scenarios += hit

    assert unbound_failures == 86
    assert affected_scenarios == 81


def test_two_consecutive_episodes_of_an_affected_scenario_score_identically(
    scenarios: list[Any],
) -> None:
    """A GRPO group's G rollouts must be comparable, including on these scenarios."""
    affected = [
        scenario
        for scenario in scenarios
        if any("initial_state" in entry["check_func"] for entry in scenario.checklist)
    ][:20]
    assert affected, "no scenario references initial_state; the fixture assumption changed"

    for scenario in affected:
        compiled = compile_checklist(scenario.task_id, scenario.checklist)
        first = score(compiled, scenario.init_config, dict(scenario.init_config))
        second = score(compiled, scenario.init_config, dict(scenario.init_config))
        assert first.reward == second.reward, scenario.task_id
        assert first.name_error_count == 0, (scenario.task_id, first.reasons())


def test_every_env_id_in_the_scenario_file_exists_in_the_env_metadata(
    scenarios: list[Any],
) -> None:
    """A scenario naming an absent environment cannot be run and must not be silent."""
    if not VENDORED_ENVS.is_file():
        pytest.skip(f"{VENDORED_ENVS} not present")
    known = set(json.loads(VENDORED_ENVS.read_text(encoding="utf-8")))
    missing = sorted({s.env_id for s in scenarios} - known)
    assert not missing, f"scenarios reference unknown env_id(s): {missing}"
