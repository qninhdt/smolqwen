"""80 tasks against a 32-episode pool: the shipped ratio, and what it used to do.

`eval.yaml` selects 10 environments x 8 scenarios; `l4.yaml` sizes the pool at
4 workers x 8 episodes. The serial loop never noticed, because it held one live
episode at a time. Any concurrent path that opens every task up front makes the
33rd `create` raise `PoolError` and the whole run score **zero** -- not a degraded
number, a zero, because a run that raises produces no report at all.

So the window is `min(generation_concurrency, pool_capacity)`, and these tests pin
both the arithmetic and the completion it buys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import EvalConfig
from smolqwen.eval.batched import admission_window, engine_config, pool_capacity_of

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class PoolBackedAdapter:
    """Shaped like `EnvScalerHeldoutAdapter`: a `_pool` with the real attributes."""

    def __init__(self, worker_count: int, episodes_per_worker: int) -> None:
        self._pool = type(
            "Pool",
            (),
            {"worker_count": worker_count, "episodes_per_worker": episodes_per_worker},
        )()


class InProcessAdapter:
    """Shaped like `BfclMultiTurnAdapter`: no environment layer at all."""

    _pool = None


def shipped_eval(profile: str = "l4") -> EvalConfig:
    config = resolve("eval", profile, config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))
    assert isinstance(config, EvalConfig)
    return config


def test_the_shipped_configs_really_do_hold_more_tasks_than_pool_capacity() -> None:
    """The premise, read from the configs rather than asserted as a constant.

    If a future edit made the pool larger than the held-out set, the window would
    stop mattering and this test says so instead of passing quietly.
    """
    config = shipped_eval()
    options = config.adapter_options["envscaler_heldout"]
    tasks = int(str(options["env_count"])) * int(str(options["scenarios_per_env"]))
    capacity = config.profile.env_worker_count * config.profile.env_episodes_per_worker

    assert tasks > capacity, (
        f"{tasks} held-out tasks against {capacity} pool episodes; the admission "
        "window is no longer load-bearing and this test should be re-derived"
    )


def test_the_window_is_the_smaller_of_generation_width_and_pool_capacity() -> None:
    config = shipped_eval()
    concurrency = config.profile.generation_concurrency

    # Pool smaller than the generation width: the pool binds.
    narrow = PoolBackedAdapter(worker_count=1, episodes_per_worker=2)
    assert pool_capacity_of(narrow) == 2  # type: ignore[arg-type]
    assert admission_window(config, narrow) == 2  # type: ignore[arg-type]

    # Pool larger: generation width binds, because extra episodes only queue.
    wide = PoolBackedAdapter(worker_count=64, episodes_per_worker=8)
    assert admission_window(config, wide) == concurrency  # type: ignore[arg-type]


def test_an_adapter_with_no_pool_is_bounded_only_by_generation_width() -> None:
    """BFCL's steps are in-process Python; there is no capacity to respect."""
    config = shipped_eval()
    adapter = InProcessAdapter()
    assert pool_capacity_of(adapter) is None  # type: ignore[arg-type]
    assert admission_window(config, adapter) == config.profile.generation_concurrency  # type: ignore[arg-type]


def test_the_engine_config_carries_the_window_and_disables_mask_building() -> None:
    """Evaluation trains nothing, and mask-off is what made the engine's liveness
    tracking load-bearing rather than incidental."""
    config = shipped_eval()
    engine = engine_config(config, max_in_flight=6)

    assert engine.max_in_flight == 6
    assert engine.build_masks is False
    # The generation-turn cap is the same field the serial loop bounded, so the two
    # paths remain comparable.
    assert engine.max_generation_turns == config.max_steps_per_task
    assert engine.max_model_len == config.profile.max_seq_length
    assert engine.max_new_tokens_per_step == config.decoding.max_new_tokens


@pytest.mark.parametrize("gpu", ["l4", "a100"])
def test_switching_profile_moves_the_window(gpu: str) -> None:
    """`--profile` must reach the window; a hardcoded default would not."""
    config = shipped_eval(gpu)
    adapter: Any = PoolBackedAdapter(
        worker_count=config.profile.env_worker_count,
        episodes_per_worker=config.profile.env_episodes_per_worker,
    )
    window = admission_window(config, adapter)
    assert window == min(
        config.profile.generation_concurrency,
        config.profile.env_worker_count * config.profile.env_episodes_per_worker,
    )
    assert window >= 1
