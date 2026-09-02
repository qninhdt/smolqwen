"""The dev set is unreachable from training, and the test benchmark from both.

The pipeline is `data -> baseline -> SFT + dev eval -> RL + dev eval -> test
benchmark -> serving`. EnvScaler held-out is dev: it selects checkpoints. BFCL is
test: it runs once, at the end, and influences no decision.

Nothing enforced that. `grpo.py:435` sizes its held-out exclusion from
`curriculum.heldout_env_count` / `heldout_scenarios_per_env`; the evaluation
adapter sizes its dev slice from `adapter_options.envscaler_heldout.env_count` /
`scenarios_per_env`. Both pairs are 10/8 in the shipped configs, so the two sets
coincide -- but they are independent keys in independent files. Raising the eval
side to 12 environments leaves two environments both trained on and scored, with
no error and no symptom beyond a dev score that flatters the run.

The assertion is a subset, not an equality: every task dev scores must be one
training excluded. A dev set smaller than the exclusion is a smaller measurement;
a dev set larger is contamination. Sizes are read from the shipped configs, never
written down here -- a test asserting `10` would pass while the leak it exists to
catch is happening.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import EvalConfig, GrpoConfig, SftConfig
from smolqwen.env.scenarios import Scenario, build_scenario_set
from smolqwen.eval.adapters.envscaler_heldout import (
    ADAPTER_NAME as ENVSCALER_ADAPTER,
)
from smolqwen.eval.adapters.envscaler_heldout import (
    EnvScalerAdapterOptions,
    select_heldout_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# The benchmark that must stay unreachable from any in-training path, spelled the
# way a leak would spell it. `test_adapter_protocol.py` uses the same idiom to keep
# benchmark names out of the generic runner.
TEST_BENCHMARK_TOKENS = ("bfcl", "gorilla", "berkeley")


def shipped(stage: str) -> Any:
    """A stage config as a run would resolve it, without budgets.json overlaying."""
    return resolve(stage, config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))


def dev_sizing(config: EvalConfig) -> EnvScalerAdapterOptions:
    """What the dev adapter would select, validated by the adapter's own model."""
    return EnvScalerAdapterOptions.model_validate(config.adapter_options.get(ENVSCALER_ADAPTER, {}))


def synthetic_population(*, env_count: int, per_env: int) -> tuple[Scenario, ...]:
    """Enough environments and scenarios that neither sizing is truncated.

    Env ids are zero-padded because `iter_by_env` sorts them as strings, and the
    leak this file guards against is about which environments land in the first N.
    """
    return tuple(
        Scenario(
            task_id=f"env_{env:03d}_rl:task_{index:03d}",
            env_id=f"env_{env:03d}_rl",
            env_class_name="Demo",
            task="do the task",
            init_config={},
            checklist=({"description": "check"},),
        )
        for env in range(env_count)
        for index in range(per_env)
    )


def test_dev_tasks_are_a_subset_of_what_training_excludes() -> None:
    """The property, proven on a population large enough for either sizing to grow.

    Both sides select through the same `select_heldout_scenarios`, so this is not a
    re-assertion that one function equals itself: the sizes come from two unrelated
    config keys, and it is their relationship that is under test.
    """
    grpo = shipped("grpo")
    evaluation = shipped("eval")
    assert isinstance(grpo, GrpoConfig)
    assert isinstance(evaluation, EvalConfig)
    dev = dev_sizing(evaluation)

    # Headroom on both axes so a raised config value changes the selection rather
    # than hitting the end of the population and silently agreeing.
    population = synthetic_population(
        env_count=max(grpo.curriculum.heldout_env_count, dev.env_count) + 4,
        per_env=max(grpo.curriculum.heldout_scenarios_per_env, dev.scenarios_per_env) + 4,
    )

    excluded_from_training = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            population,
            env_count=grpo.curriculum.heldout_env_count,
            per_env=grpo.curriculum.heldout_scenarios_per_env,
        )
    }
    scored_as_dev = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            population,
            env_count=dev.env_count,
            per_env=dev.scenarios_per_env,
        )
    }

    leaked = sorted(scored_as_dev - excluded_from_training)
    assert not leaked, (
        f"{len(leaked)} dev task(s) are inside GRPO's training candidates: {leaked[:5]}. "
        f"eval env_count={dev.env_count}/{dev.scenarios_per_env} exceeds "
        f"grpo heldout {grpo.curriculum.heldout_env_count}/"
        f"{grpo.curriculum.heldout_scenarios_per_env}"
    )


def test_a_widened_dev_slice_is_detected_as_contamination() -> None:
    """The negative control: without it, the subset assertion could be vacuous.

    Selecting one more environment than training excludes must produce tasks the
    exclusion does not cover. This is what proves the test above reads the configs
    rather than restating a constant that happens to agree.
    """
    grpo = shipped("grpo")
    assert isinstance(grpo, GrpoConfig)
    per_env = grpo.curriculum.heldout_scenarios_per_env
    population = synthetic_population(
        env_count=grpo.curriculum.heldout_env_count + 4, per_env=per_env + 4
    )

    excluded = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            population, env_count=grpo.curriculum.heldout_env_count, per_env=per_env
        )
    }
    widened = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            population, env_count=grpo.curriculum.heldout_env_count + 1, per_env=per_env
        )
    }
    assert widened - excluded, "widening env_count must add tasks, or the guard is vacuous"


@pytest.mark.dataset
def test_dev_selection_is_disjoint_from_training_on_the_real_release() -> None:
    """Same property against the file that actually trains.

    The synthetic population proves the relationship between the two config keys.
    This proves the shipped values on the real 2,550-scenario release, where
    per-environment counts are uneven and truncation could hide a discrepancy the
    synthetic case is built to expose.
    """
    grpo = shipped("grpo")
    evaluation = shipped("eval")
    assert isinstance(grpo, GrpoConfig)
    assert isinstance(evaluation, EvalConfig)
    scenarios = build_scenario_set(
        grpo.env.vendored_rl_scenarios,
        env_split_manifest=grpo.env.env_split_manifest,
        sha256=grpo.env.vendored_rl_scenarios_sha256,
    ).scenarios

    dev = dev_sizing(evaluation)
    excluded = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            scenarios,
            env_count=grpo.curriculum.heldout_env_count,
            per_env=grpo.curriculum.heldout_scenarios_per_env,
        )
    }
    scored = {
        scenario.task_id
        for scenario in select_heldout_scenarios(
            scenarios, env_count=dev.env_count, per_env=dev.scenarios_per_env
        )
    }
    training = {scenario.task_id for scenario in scenarios} - excluded

    assert scored, "dev selection is empty; the eval slice would measure nothing"
    leaked = sorted(scored & training)
    assert not leaked, (
        f"{len(leaked)} of {len(scored)} dev tasks are trained on: {leaked[:5]}. "
        f"eval {dev.env_count}/{dev.scenarios_per_env} vs grpo heldout "
        f"{grpo.curriculum.heldout_env_count}/{grpo.curriculum.heldout_scenarios_per_env}"
    )


def test_the_training_stages_cannot_name_the_test_benchmark() -> None:
    """BFCL is the test set: an in-training path must not be able to reach it.

    No in-training eval path exists yet, so this passes trivially today -- and
    fails the moment one is added that scores BFCL every checkpoint, which would
    turn the test set into a dev set and make the final Base | SFT | SFT+RL table
    meaningless. Source inspection is the same idiom `test_adapter_protocol.py`
    uses to keep benchmark names out of the generic runner.

    Dev-adapter selection therefore belongs on `GrpoConfig`/`SftConfig` or inside
    the already-opaque `adapter_options`, naming the dev adapter only.
    """
    from smolqwen.training import grpo as grpo_module
    from smolqwen.training import sft as sft_module

    for module in (grpo_module, sft_module):
        source = inspect.getsource(module).casefold()
        for token in TEST_BENCHMARK_TOKENS:
            assert token not in source, f"{module.__name__} names the test benchmark {token!r}"

    for model in (GrpoConfig, SftConfig):
        named = {
            field
            for field in model.model_fields
            for token in TEST_BENCHMARK_TOKENS
            if token in field.casefold()
        }
        assert not named, f"{model.__name__} carries test-benchmark field(s) {sorted(named)}"


def test_the_dev_adapter_is_the_one_the_training_stages_may_score() -> None:
    """Fixes which adapter name an in-training callback is allowed to resolve.

    Both adapters are configured (`eval.yaml` lists both), so a callback reading
    that list wholesale would score BFCL. The dev adapter is named here so a later
    phase wires a specific name rather than iterating `config.adapters`.
    """
    evaluation = shipped("eval")
    assert isinstance(evaluation, EvalConfig)
    assert ENVSCALER_ADAPTER in evaluation.adapters
    assert dev_sizing(evaluation).env_count >= 1
    assert not any(token in ENVSCALER_ADAPTER.casefold() for token in TEST_BENCHMARK_TOKENS), (
        "the dev adapter name must not be the test benchmark"
    )
