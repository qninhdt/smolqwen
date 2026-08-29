"""Inside a worker: no credentials reachable, and the parent never `exec`ed anything.

The claim "the worker holds no secrets" is what makes the `exec()` posture
acceptable, and env-var scrubbing alone does not deliver it.
`huggingface_hub.get_token()` reads three sources in order — the Colab secrets
vault (cached in a **module global**, so a `fork`ed child inherits the plaintext
token in memory no matter what `os.environ` says), then the env var, then
`~/.cache/huggingface/token` on disk. W&B credentials additionally sit in
`~/.netrc`.

So the assertion is `get_token() is None` *asked from inside the worker*, not an
inspection of `os.environ` — the env var is the one source that was never the
problem. `spawn` handles the module-global cache, and the redirected `HOME`/`HF_HOME`
handle the on-disk copies.

The second half is where the `exec` happened. Compiling in the trainer process
would run 191 dataset-supplied module bodies inside the process holding the token,
the W&B session, the CUDA context and the model weights, with no timeout and no
scrub. The exec counters are therefore read *per worker*: a global count would be
satisfied by compiling in the parent, which is precisely the design being forbidden.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import EnvRegistry
from smolqwen.env.scenarios import load_scenarios

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"

CREDENTIAL_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture
def with_credentials_set() -> Iterator[None]:
    """Put plausible credentials in the parent so their absence downstream means something.

    A test that passes with no credentials anywhere proves nothing about scrubbing.
    """
    original = {name: os.environ.get(name) for name in CREDENTIAL_VARS}
    for name in CREDENTIAL_VARS:
        os.environ[name] = f"fake-{name.lower()}"
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_no_hub_token_is_reachable_inside_a_worker(with_credentials_set: None) -> None:
    """Asked the way the Hub asks it, not by reading `os.environ`."""
    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1, call_timeout_s=30.0) as pool:
        stats = pool.worker_stats(0)
        assert stats.ok, stats.detail
        assert stats.value["hf_token_visible"] is False


def test_the_worker_runs_in_a_different_process(with_credentials_set: None) -> None:
    """`spawn`, so no module-global token cache is inherited."""
    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1) as pool:
        assert pool.worker_stats(0).value["pid"] != os.getpid()


def test_the_parent_process_never_compiled_dataset_source() -> None:
    """Loading scenarios and metadata in the parent must not `exec` anything.

    The parent holds raw JSON strings; compilation is the worker's job. This asserts
    the parent-side load path specifically, since that is where the shortcut would
    be introduced.
    """
    registry = EnvRegistry.from_metadata(ENV_METADATA)
    scenarios = load_scenarios(SCENARIOS)

    assert registry.exec_count == 0
    assert registry.compiled_count() == 0
    # And the scenarios' check sources are still text, not callables.
    assert all(isinstance(entry["check_func"], str) for entry in scenarios[0].checklist)


def test_each_worker_compiles_for_itself_exactly_once() -> None:
    """Compile-once means once per worker; repeated calls must not recompile."""
    scenario = load_scenarios(SCENARIOS)[0]
    with WorkerPool(
        metadata_path=str(ENV_METADATA), worker_count=1, episodes_per_worker=4, call_timeout_s=30.0
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(
                episode,
                env_id=scenario.env_id,
                env_class_name=scenario.env_class_name,
                init_config=scenario.init_config,
                checklist=scenario.checklist,
            ).ok

        # Several steps and several scorings across two episodes of one environment.
        for episode in ("ep1", "ep2"):
            for _ in range(3):
                assert pool.step(episode, "get_clinical_trial_by_id", {"trial_id": "CT-101"}).ok
                assert pool.score(episode).ok

        stats = pool.worker_stats(0).value

    # One environment class, compiled once, despite 2 episodes x 3 steps x 3 scores.
    assert stats["exec_count"] == 1
    assert stats["compiled_envs"] == 1
    # K checks compile once per scenario per worker, not once per episode or score.
    assert stats["checklist_exec_count"] == scenario.check_count


def test_two_workers_each_compile_their_own_copy() -> None:
    """Not a global cache: a per-worker count is the whole point of the assertion."""
    scenario = load_scenarios(SCENARIOS)[0]
    with WorkerPool(
        metadata_path=str(ENV_METADATA), worker_count=2, episodes_per_worker=1, call_timeout_s=30.0
    ) as pool:
        for episode in ("ep1", "ep2"):
            assert pool.create(
                episode,
                env_id=scenario.env_id,
                env_class_name=scenario.env_class_name,
                init_config=scenario.init_config,
            ).ok

        for index in (0, 1):
            assert pool.worker_stats(index).value["exec_count"] == 1


def test_the_worker_home_is_a_private_directory(with_credentials_set: None) -> None:
    """`~/.cache/huggingface/token` and `~/.netrc` must be somewhere the worker is not."""
    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1) as pool:
        pool.start()
        home = Path(pool._workers[0].home)  # noqa: SLF001 - inspecting the boundary under test
        assert home.is_dir()
        assert home != Path.home()
        assert not (home / ".netrc").exists()


def test_no_env_module_imports_an_llm_or_http_client() -> None:
    """A reward that depends on a network call is not a reward from the episode."""
    forbidden = (
        "import requests",
        "import httpx",
        "import openai",
        "from openai",
        "import aiohttp",
        "from vllm",
        "import vllm",
    )
    env_dir = Path("src/smolqwen/env")
    offenders: list[tuple[str, str]] = []
    for module in sorted(env_dir.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                if stripped.startswith(pattern):
                    offenders.append((module.name, stripped))
    assert not offenders, offenders
