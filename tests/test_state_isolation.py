"""State isolation: two episodes from one `init_config` must not see each other.

The highest-impact silent failure in the project. A GRPO group runs G rollouts of
the *same* scenario from the *same* `init_config`. If rollout 1's mutations bleed
into rollout 2, the group's rewards become correlated garbage: advantages go
wrong, training learns noise, and no error is raised anywhere.

So these assert the leak directions specifically, on the real released classes:

- mutating one instance's state leaves the other's untouched;
- mutating one instance cannot reach the *scenario dict* either, which is the
  sharing path a `deepcopy` at the wrong level would leave open;
- `initial_state` is a snapshot, not a live view of the instance;
- a nested container reached through `initial_state` cannot be mutated by a check
  into the next episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.env.instance import EnvInstance, construct, state_of
from smolqwen.env.registry import EnvRegistry
from smolqwen.env.scenarios import load_scenarios

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"


@pytest.fixture(scope="module")
def registry() -> EnvRegistry:
    return EnvRegistry.from_metadata(ENV_METADATA)


@pytest.fixture(scope="module")
def scenario() -> Any:
    return load_scenarios(SCENARIOS)[0]


def _instance(registry: EnvRegistry, scenario: Any) -> EnvInstance:
    return EnvInstance.create(
        env_id=scenario.env_id,
        env_class=registry.env_class(scenario.env_id),
        env_class_name=scenario.env_class_name,
        init_config=scenario.init_config,
        tools=registry.tools(scenario.env_id),
    )


def test_two_instances_from_one_config_do_not_share_state(
    registry: EnvRegistry, scenario: Any
) -> None:
    first = _instance(registry, scenario)
    second = _instance(registry, scenario)

    first.step("update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"})

    assert first.final_state()["clinical_trials"]["CT-101"]["status"] == "completed"
    # The whole point: the second episode still sees the scenario's own value.
    assert second.final_state()["clinical_trials"]["CT-101"]["status"] == "active"


def test_mutating_an_instance_cannot_reach_the_scenario_config(
    registry: EnvRegistry, scenario: Any
) -> None:
    """The sharing path a shallow copy would leave open.

    If the instance held the scenario's own nested dicts, episode 1 would rewrite
    the config every later episode is built from -- and every rollout after the
    first would start from a different state than the profiler measured.
    """
    before = json.dumps(scenario.init_config, sort_keys=True)
    instance = _instance(registry, scenario)
    instance.step("update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"})
    assert json.dumps(scenario.init_config, sort_keys=True) == before


def test_initial_state_is_a_snapshot_not_a_live_view(registry: EnvRegistry, scenario: Any) -> None:
    instance = _instance(registry, scenario)
    captured = instance.initial_state["clinical_trials"]["CT-101"]["status"]
    instance.step("update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"})

    assert captured == "active"
    # Still the captured value: a live view here would make the verifier compare
    # the final state against itself, and every "did X change" check would fail.
    assert instance.initial_state["clinical_trials"]["CT-101"]["status"] == "active"
    assert instance.final_state()["clinical_trials"]["CT-101"]["status"] == "completed"


def test_state_of_deep_copies_nested_containers(registry: EnvRegistry, scenario: Any) -> None:
    instance = _instance(registry, scenario)
    snapshot = state_of(instance.instance)
    snapshot["clinical_trials"]["CT-101"]["status"] = "tampered"
    assert instance.final_state()["clinical_trials"]["CT-101"]["status"] != "tampered"


def test_construct_falls_back_to_a_no_argument_constructor() -> None:
    """Upstream's `init_env_instance` behaviour, which the release depends on."""

    class NoArgs:
        def __init__(self) -> None:
            self.items: dict[str, int] = {}

    instance = construct(NoArgs, {"items": {"a": 1}})
    assert instance.items == {"a": 1}


def test_construct_gives_each_instance_its_own_nested_containers() -> None:
    class Holder:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self.rows: dict[str, Any] = {}

    config = {"rows": {"r1": {"n": 0}}}
    first = construct(Holder, config)
    second = construct(Holder, config)
    first.rows["r1"]["n"] = 99

    assert second.rows["r1"]["n"] == 0
    assert config["rows"]["r1"]["n"] == 0


def test_step_counts_and_rejects_private_attributes(registry: EnvRegistry, scenario: Any) -> None:
    instance = _instance(registry, scenario)
    assert instance.step_count == 0
    instance.step("get_clinical_trial_by_id", {"trial_id": "CT-101"})
    assert instance.step_count == 1

    # A dunder or private name is not a tool even though `getattr` would find it.
    assert not instance.has_tool("__init__")
    assert not instance.has_tool("_private")


def test_declared_tool_surface_rejects_an_undeclared_public_method() -> None:
    class Environment:
        def allowed(self) -> str:
            return "ok"

        def undeclared(self) -> str:
            return "must not run"

    instance = EnvInstance.create(
        env_id="test",
        env_class=Environment,
        env_class_name="Environment",
        init_config={},
        tools=(
            {
                "type": "function",
                "function": {"name": "allowed", "parameters": {"type": "object"}},
            },
        ),
    )
    assert instance.has_tool("allowed")
    assert not instance.has_tool("undeclared")
