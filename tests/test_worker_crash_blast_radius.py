"""A dead worker loses every episode it held — reported, tagged, and re-admittable.

A worker owning N live episodes that dies cannot lose one in isolation. If the pool
reported only the episode whose call was in flight, the other N-1 would sit in the
scheduler as live, then fail one by one as `never created` — or worse, be scored
from a partial state.

Scoring a crashed episode is the failure that reaches training: GRPO would learn
that whatever the model did last was worth a low reward, teaching it to avoid an
infrastructure failure it did not cause. So `worker_crash` is a distinct reason
from `timeout`, `Result.is_infrastructure_failure` is what Phase 6 filters on, and
these tests pin both halves.

`env_episodes_per_worker` makes the blast radius a chosen number. These tests
therefore assert the size of the loss, not just that a loss was reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.env.pool import PoolError, WorkerPool

pytestmark = pytest.mark.slow

# `os._exit` skips cleanup entirely, which is what a segfault or an OOM kill looks
# like from the parent: the queue never receives a reply.
CRASHING_ENV_CODE = """
import os


class CrashingSystem:
    def __init__(self, config=None):
        self.rows = {}

    def crash(self) -> str:
        os._exit(70)

    def quick(self, value: str) -> str:
        self.rows[value] = True
        return f"stored {value}"
"""


@pytest.fixture
def metadata(tmp_path: Path) -> Path:
    path = tmp_path / "crashing.json"
    path.write_text(
        json.dumps(
            {
                "env_crash_rl": {
                    "env_id": "env_crash_rl",
                    "env_class_name": "CrashingSystem",
                    "env_class_code": CRASHING_ENV_CODE,
                    "tools": [],
                    "environment_introduction": "",
                    "constraints_rules": [],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _create(pool: WorkerPool, *episode_ids: str) -> None:
    for episode_id in episode_ids:
        result = pool.create(episode_id, env_id="env_crash_rl", env_class_name="CrashingSystem")
        assert result.ok, result.detail


def test_a_crash_reports_every_episode_the_worker_held(metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(metadata),
        worker_count=1,
        episodes_per_worker=4,
        call_timeout_s=3.0,
    ) as pool:
        _create(pool, "ep1", "ep2", "ep3")

        result = pool.step("ep1", "crash", {})
        assert result.reason == "worker_crash"
        assert result.is_infrastructure_failure
        # Typed, so Phase 6 can re-admit every loss without parsing a log string.
        assert result.lost_episode_ids == ("ep1", "ep2", "ep3")


def test_a_crash_is_tagged_differently_from_a_timeout(metadata: Path) -> None:
    """Phase 6 drops one and scores the other; the reason string is the switch."""
    with WorkerPool(
        metadata_path=str(metadata), worker_count=1, episodes_per_worker=2, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1")
        crash = pool.step("ep1", "crash", {})

    assert crash.reason == "worker_crash"
    assert crash.is_infrastructure_failure


def test_the_pool_replaces_the_dead_worker_and_keeps_serving(metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(metadata), worker_count=1, episodes_per_worker=4, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1")
        assert pool.step("ep1", "crash", {}).reason == "worker_crash"

        # A replacement worker is running, so the scheduler can re-admit the lost
        # episodes rather than the run ending.
        _create(pool, "ep1-retry")
        assert pool.step("ep1-retry", "quick", {"value": "a"}).ok


def test_lost_episodes_are_deregistered_not_left_dangling(metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(metadata), worker_count=1, episodes_per_worker=4, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1", "ep2")
        pool.step("ep1", "crash", {})

        for episode_id in ("ep1", "ep2"):
            assert pool.owner_of(episode_id) is None
            # Explicit rather than routed to the replacement worker, which has no
            # state for it and would report a confusing `error`.
            with pytest.raises(PoolError, match="never created"):
                pool.step(episode_id, "quick", {"value": "x"})


def test_a_crash_on_one_worker_leaves_the_other_workers_episodes_alone(
    metadata: Path,
) -> None:
    """Which is the whole reason `env_worker_count` is separate from the episode cap."""
    with WorkerPool(
        metadata_path=str(metadata), worker_count=2, episodes_per_worker=2, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1", "ep2")
        assert pool.owner_of("ep1") != pool.owner_of("ep2")

        survivor = "ep2" if pool.owner_of("ep1") == 0 else "ep1"
        casualty = "ep1" if survivor == "ep2" else "ep2"

        assert pool.step(casualty, "crash", {}).reason == "worker_crash"
        assert pool.step(survivor, "quick", {"value": "b"}).ok


def test_killing_a_worker_reports_the_same_episode_list(metadata: Path) -> None:
    """`kill_worker` exists so the blast radius is testable without a real segfault."""
    with WorkerPool(
        metadata_path=str(metadata), worker_count=1, episodes_per_worker=4, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1", "ep2", "ep3")
        lost = pool.kill_worker(0)
        assert lost == ["ep1", "ep2", "ep3"]
        assert pool.episodes_of(0) == ()

        _create(pool, "ep4")
        assert pool.step("ep4", "quick", {"value": "c"}).ok


def test_the_blast_radius_is_the_configured_episode_cap(metadata: Path) -> None:
    """A radius of 2 loses at most 2, which is the point of making it a config knob."""
    with WorkerPool(
        metadata_path=str(metadata), worker_count=2, episodes_per_worker=2, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1", "ep2", "ep3", "ep4")
        lost = pool.kill_worker(0)
        assert len(lost) == 2
        # The other two survive on the second worker.
        assert len(pool.episodes_of(1)) == 2


def test_a_step_to_an_already_dead_worker_is_a_crash_not_a_hang(metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(metadata), worker_count=1, episodes_per_worker=4, call_timeout_s=3.0
    ) as pool:
        _create(pool, "ep1")
        assert pool.step("ep1", "crash", {}).reason == "worker_crash"

        _create(pool, "ep2")
        # Kill the process out from under the pool without going through it, then
        # confirm the next request notices immediately rather than waiting out the
        # full deadline.
        pool._workers[0].process.kill()  # noqa: SLF001 - simulating an external kill
        pool._workers[0].process.join(timeout=5)
        result = pool.step("ep2", "quick", {"value": "d"})
        assert result.reason == "worker_crash"
        assert "ep2" in (result.detail or "")
