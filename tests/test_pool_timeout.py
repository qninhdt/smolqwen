"""Pool: a hanging call is bounded, a crash loses everything that worker held.

Two failures with different consequences downstream, which is why the pool reports
them as different reasons:

- **`timeout`** — the call ran too long. The in-worker `SIGALRM` fires, that one
  episode fails, and the worker keeps serving its other episodes. Phase 6 scores a
  timed-out episode: the model chose the action that hung.
- **`worker_crash`** — nobody answered. The worker is wedged or dead, so *every*
  episode it held is lost and the scheduler must re-admit them. Phase 6 drops these
  from the training buffer entirely; scoring a crashed episode as a low reward
  teaches the model to avoid an infrastructure failure it did not cause.

Marked `slow` because each test spawns real interpreters. `spawn` is not a
portability choice here — it is the security boundary (see `pool.py`), so the tests
exercise the real mechanism rather than a threaded stand-in.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from smolqwen.env.pool import PoolError, WorkerPool, minimal_environment
from smolqwen.env.scenarios import load_scenarios

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"

# An environment whose only tool blocks forever. Written as `env_class_code` so it
# travels through the same load-and-exec path the released environments use.
HANGING_ENV_CODE = """
import time


class HangingSystem:
    def __init__(self, config=None):
        self.rows = {}

    def hang(self, seconds: int) -> str:
        time.sleep(seconds)
        return "returned"

    def quick(self, value: str) -> str:
        self.rows[value] = True
        return f"stored {value}"
"""

# One whose tool kills its own process, so the parent sees a real dead worker.
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


def _metadata(path: Path, entries: dict[str, tuple[str, str]]) -> Path:
    payload = {
        env_id: {
            "env_id": env_id,
            "env_class_name": class_name,
            "env_class_code": code,
            "tools": [],
            "environment_introduction": "",
            "constraints_rules": [],
        }
        for env_id, (class_name, code) in entries.items()
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def hanging_metadata(tmp_path: Path) -> Path:
    return _metadata(
        tmp_path / "hanging.json",
        {
            "env_hang_rl": ("HangingSystem", HANGING_ENV_CODE),
            "env_crash_rl": ("CrashingSystem", CRASHING_ENV_CODE),
        },
    )


def test_a_hanging_call_times_out_and_the_pool_survives(hanging_metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(hanging_metadata),
        worker_count=1,
        episodes_per_worker=4,
        call_timeout_s=2.0,
    ) as pool:
        assert pool.create("ep1", env_id="env_hang_rl", env_class_name="HangingSystem").ok

        result = pool.step("ep1", "hang", {"seconds": 30})
        assert result.reason == "timeout"
        assert not result.is_infrastructure_failure

        # The worker is still alive and the episode is still usable: only the call
        # was cancelled.
        after = pool.step("ep1", "quick", {"value": "a"})
        assert after.ok
        assert after.value == "stored a"


def test_a_timeout_does_not_affect_the_workers_other_episodes(
    hanging_metadata: Path,
) -> None:
    with WorkerPool(
        metadata_path=str(hanging_metadata),
        worker_count=1,
        episodes_per_worker=4,
        call_timeout_s=2.0,
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(episode, env_id="env_hang_rl", env_class_name="HangingSystem").ok

        assert pool.step("ep1", "hang", {"seconds": 30}).reason == "timeout"
        assert pool.step("ep2", "quick", {"value": "b"}).ok


def test_an_error_inside_a_tool_is_an_error_not_a_crash(hanging_metadata: Path) -> None:
    """A buggy environment method must not look like lost infrastructure."""
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=1, call_timeout_s=5.0
    ) as pool:
        assert pool.create("ep1", env_id="env_hang_rl", env_class_name="HangingSystem").ok
        result = pool.step("ep1", "quick", {"wrong_parameter": 1})
        assert result.reason == "error"
        assert not result.is_infrastructure_failure
        assert result.detail


def test_calling_an_unknown_tool_is_an_error_the_worker_survives(
    hanging_metadata: Path,
) -> None:
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=1, call_timeout_s=5.0
    ) as pool:
        assert pool.create("ep1", env_id="env_hang_rl", env_class_name="HangingSystem").ok
        assert pool.step("ep1", "no_such_tool", {}).reason == "error"
        assert pool.step("ep1", "quick", {"value": "c"}).ok


def test_an_unknown_env_id_fails_the_create_without_leaking_the_episode(
    hanging_metadata: Path,
) -> None:
    with WorkerPool(metadata_path=str(hanging_metadata), worker_count=1) as pool:
        result = pool.create("ep1", env_id="env_absent_rl", env_class_name="Nope")
        assert not result.ok
        # The episode was never registered, so a later step names the real problem
        # rather than routing to an arbitrary worker.
        assert pool.owner_of("ep1") is None
        with pytest.raises(PoolError, match="never created"):
            pool.step("ep1", "quick", {"value": "x"})


def test_episodes_per_worker_bounds_what_one_worker_holds(hanging_metadata: Path) -> None:
    """The blast radius is a chosen number, so it has to actually be enforced."""
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=1, episodes_per_worker=2
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(episode, env_id="env_hang_rl", env_class_name="HangingSystem").ok
        with pytest.raises(PoolError, match="env_episodes_per_worker"):
            pool.create("ep3", env_id="env_hang_rl", env_class_name="HangingSystem")


def test_destroy_frees_a_slot(hanging_metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=1, episodes_per_worker=1
    ) as pool:
        assert pool.create("ep1", env_id="env_hang_rl", env_class_name="HangingSystem").ok
        assert pool.destroy("ep1").ok
        assert pool.create("ep2", env_id="env_hang_rl", env_class_name="HangingSystem").ok


def test_episodes_are_spread_across_workers(hanging_metadata: Path) -> None:
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=2, episodes_per_worker=4
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(episode, env_id="env_hang_rl", env_class_name="HangingSystem").ok
        # Least-loaded placement, so two episodes land on two workers -- which is
        # what keeps one crash from taking both.
        assert pool.owner_of("ep1") != pool.owner_of("ep2")


def test_concurrent_callers_on_one_worker_receive_their_own_results(
    hanging_metadata: Path,
) -> None:
    """The parent serializes one worker's result queue before Phase 6 goes async."""
    with WorkerPool(
        metadata_path=str(hanging_metadata), worker_count=1, episodes_per_worker=4
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(episode, env_id="env_hang_rl", env_class_name="HangingSystem").ok

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(pool.step, "ep1", "quick", {"value": "one"})
            second = executor.submit(pool.step, "ep2", "quick", {"value": "two"})
            results = (first.result(timeout=10), second.result(timeout=10))

    assert [(result.episode_id, result.value) for result in results] == [
        ("ep1", "stored one"),
        ("ep2", "stored two"),
    ]


def test_scoring_a_real_scenario_through_the_pool() -> None:
    """The pool's own path, on real released data, not just the toy environments."""
    scenario = load_scenarios(SCENARIOS)[0]
    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1, call_timeout_s=30.0) as pool:
        created = pool.create(
            "ep1",
            env_id=scenario.env_id,
            env_class_name=scenario.env_class_name,
            init_config=scenario.init_config,
            checklist=scenario.checklist,
        )
        assert created.ok
        assert len(created.value["tools"]) == 15

        assert pool.score("ep1").value["reward"] == 0.0
        assert pool.step(
            "ep1", "update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"}
        ).ok
        scored = pool.score("ep1")
        assert scored.value["reward"] == 0.3333
        assert scored.value["name_errors"] == 0


def test_the_minimal_environment_carries_no_credentials() -> None:
    """An allow-list, so a credential invented later is not silently included."""
    import os

    original = dict(os.environ)
    os.environ.update(
        {
            "HF_TOKEN": "secret-hf",
            "HUGGING_FACE_HUB_TOKEN": "secret-hub",
            "WANDB_API_KEY": "secret-wandb",
            "AWS_SECRET_ACCESS_KEY": "secret-aws",
        }
    )
    try:
        env = minimal_environment(home="/tmp/does-not-matter")
    finally:
        os.environ.clear()
        os.environ.update(original)

    for leaked in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        assert leaked not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HOME"] == "/tmp/does-not-matter"
    assert env["HF_HOME"].startswith("/tmp/does-not-matter")


def test_a_pool_needs_at_least_one_worker() -> None:
    with pytest.raises(PoolError, match="at least 1"):
        WorkerPool(metadata_path=str(ENV_METADATA), worker_count=0)
