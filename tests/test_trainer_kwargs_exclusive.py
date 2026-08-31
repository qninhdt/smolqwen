"""The production rollout trainer must stay separate from the factory oracle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from smolqwen.rollout.rollout_func import RolloutFuncError, assert_trainer_exclusive
from smolqwen.training.grpo import _cleanup_failed_assembly, _trainer_rollout_kwargs


@pytest.mark.parametrize(
    "trainer",
    [
        SimpleNamespace(tools=[object()], environment_factories=None),
        SimpleNamespace(tools=[], environment_factories={None: object()}),
    ],
)
def test_tools_or_an_environment_factory_fail_loudly(trainer: object) -> None:
    with pytest.raises(RolloutFuncError, match="env_mask is read only"):
        assert_trainer_exclusive(trainer)


def test_production_trainer_shape_is_accepted() -> None:
    assert_trainer_exclusive(SimpleNamespace(tools=[], environment_factories=None))


def test_async_and_factory_trainers_are_constructed_exclusively() -> None:
    reward = object()
    rollout = object()
    factories = {"env_1_rl": object()}

    async_kwargs = _trainer_rollout_kwargs(
        "async",
        reward_func=reward,
        rollout_func=rollout,
        environment_factories=factories,
    )
    assert async_kwargs == {
        "reward_funcs": reward,
        "tools": None,
        "rollout_func": rollout,
        "environment_factory": None,
    }

    oracle_kwargs = _trainer_rollout_kwargs(
        "factory_oracle",
        reward_func=reward,
        rollout_func=rollout,
        environment_factories=factories,
    )
    assert oracle_kwargs == {
        "reward_funcs": None,
        "tools": None,
        "rollout_func": None,
        "environment_factory": factories,
    }


def test_failed_assembly_attempts_every_cleanup_even_when_one_raises() -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def shutdown(self) -> None:
            calls.append(self.name)
            raise RuntimeError(self.name)

    class Run:
        def finish(self) -> None:
            calls.append("tracker")
            raise RuntimeError("tracker")

    _cleanup_failed_assembly(
        cast(Any, Resource("dispatcher")),
        cast(Any, Resource("pool")),
        cast(Any, Run()),
    )
    assert calls == ["dispatcher", "pool", "tracker"]
