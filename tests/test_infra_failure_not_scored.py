from __future__ import annotations

import pytest

from smolqwen.training.reward import RewardContractError, verifier_reward


def test_worker_crash_is_never_converted_to_a_low_policy_reward() -> None:
    with pytest.raises(RewardContractError, match="worker_crash"):
        verifier_reward(
            rollout_reward=[0.0],
            terminal_reason=["worker_crash"],
            group_index=[0],
            trajectory=[{}],
        )
