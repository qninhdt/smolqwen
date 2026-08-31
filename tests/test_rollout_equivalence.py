"""The ready queue and factory oracle compute identical deterministic rewards."""

from __future__ import annotations

import pytest

from smolqwen.env.pool import WorkerPool
from smolqwen.rollout.bench import check_equivalence, run_serial_factory_oracle_scripted
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, make_scheduler
from smolqwen.rollout.scheduler import PoolDispatcher
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    ENV_METADATA,
    FIXTURE_SCRIPT,
    fast_config,
    fixture_bindings,
    load_fixture_workload,
    script_policy_texts,
    text_list_policy,
)

pytestmark = pytest.mark.slow


def test_scripted_policy_rewards_match_factory_oracle() -> None:
    env_specs, _ = load_fixture_workload()
    bindings = fixture_bindings(episodes=2)
    texts = script_policy_texts(FIXTURE_SCRIPT)
    policy = text_list_policy(texts)
    tokenizer = OfflineTokenizer(token_size=1)

    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=2, episodes_per_worker=4) as pool:
        oracle = run_serial_factory_oracle_scripted(
            pool=pool,
            bindings=bindings,
            env_specs=env_specs,
            policy=policy,
        )
        dispatcher = PoolDispatcher(pool)
        try:
            scheduler = make_scheduler(
                backend=ScriptedPolicyBackend(policy, lambda text: encode_ids(tokenizer, text)),
                dispatcher=dispatcher,
                tokenizer=tokenizer,
                config=fast_config(),
            )
            episodes = scheduler.run(bindings)
        finally:
            dispatcher.shutdown()

    assert check_equivalence(oracle_outcomes=oracle, episodes=episodes) == []
