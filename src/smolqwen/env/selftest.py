"""`smolqwen env-selftest`: drive a real scenario end to end and check the reward.

The end-to-end proof the layer works before any model touches it. A scripted,
hardcoded tool sequence runs through the real pool against a real released
scenario, and the reward is asserted twice:

- the **correct** sequence must score exactly 1.0;
- a **partial** prefix of it must score the exact fraction its checks imply, not
  merely "less than 1".

Asserting the exact fraction is the point. A verifier that silently returns 0.0 for
everything also satisfies "partial is lower than correct", and that bug is
invisible in a training curve — reward just looks hard-won.

The default scenario is `env_151_rl-task_7` (K=3): three independent status
updates, so each prefix has a known reward, and the whole episode fits in three
tool calls with no user turn. The tool sequence and the expected rewards were
derived by executing the released class, not by reading the task text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolqwen.config_models import GrpoConfig
from smolqwen.env.pool import WorkerPool
from smolqwen.env.scenarios import Scenario, load_scenarios


class SelfTestError(RuntimeError):
    """Raised when the scripted episode does not produce the expected reward."""


DEFAULT_SCENARIO_ID = "env_151_rl-task_7"

# The correct sequence for `DEFAULT_SCENARIO_ID`, one call per check. Ids are
# spelled out rather than looked up so the script itself cannot drift into
# re-deriving the answer from the state it is supposed to be verifying.
DEFAULT_SCRIPT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("update_clinical_trial_status", {"trial_id": "CT-101", "new_status": "completed"}),
    (
        "update_enrollment_status",
        {
            "enrollment_id": "e1f02ad0-0e36-482d-aeee-6d170601b657",
            "new_status": "completed",
            "requesting_user_account_id": "acc-4211a7d1-cf6e-46bc-b781-c34d46e85508",
        },
    ),
    (
        "update_enrollment_status",
        {
            "enrollment_id": "0cc3569c-689c-4961-b957-30765e14a8e7",
            "new_status": "completed",
            "requesting_user_account_id": "acc-12ac0bab-bff6-4b61-b328-96d8761dac88",
        },
    ),
)


@dataclass(frozen=True)
class EpisodeOutcome:
    task_id: str
    steps: tuple[str, ...]
    reward: float
    passed: int
    total: int
    name_errors: int
    observations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": list(self.steps),
            "reward": self.reward,
            "passed": self.passed,
            "total": self.total,
            "name_errors": self.name_errors,
        }


def run_episode(
    pool: WorkerPool,
    scenario: Scenario,
    script: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    episode_id: str,
) -> EpisodeOutcome:
    """Create, step through `script`, score, destroy. Failures raise rather than log."""
    created = pool.create(
        episode_id,
        env_id=scenario.env_id,
        env_class_name=scenario.env_class_name,
        init_config=scenario.init_config,
        checklist=scenario.checklist,
        checklist_id=scenario.task_id,
    )
    if not created.ok:
        raise SelfTestError(
            f"{scenario.task_id}: create failed ({created.reason}): {created.detail}"
        )

    observations: list[str] = []
    try:
        for name, arguments in script:
            result = pool.step(episode_id, name, arguments)
            if not result.ok:
                raise SelfTestError(
                    f"{scenario.task_id}: step {name} failed ({result.reason}): {result.detail}"
                )
            observations.append(str(result.value))

        scored = pool.score(episode_id)
        if not scored.ok:
            raise SelfTestError(
                f"{scenario.task_id}: scoring failed ({scored.reason}): {scored.detail}"
            )
        payload = scored.value
    finally:
        pool.destroy(episode_id)

    return EpisodeOutcome(
        task_id=scenario.task_id,
        steps=tuple(name for name, _ in script),
        reward=float(payload["reward"]),
        passed=int(payload["passed"]),
        total=int(payload["total"]),
        name_errors=int(payload["name_errors"]),
        observations=tuple(observations),
    )


def run_selftest(
    config: GrpoConfig,
    *,
    scenario_id: str | None = None,
    script: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
) -> int:
    """Run the scripted episode twice — full and partial — and check both rewards."""
    metadata_path = Path(config.env.vendored_env_metadata)
    scenario_path = Path(config.env.vendored_rl_scenarios)
    target_id = scenario_id or DEFAULT_SCENARIO_ID
    sequence = tuple(script or DEFAULT_SCRIPT)

    scenarios = {
        s.task_id: s
        for s in load_scenarios(scenario_path, sha256=config.env.vendored_rl_scenarios_sha256)
    }
    scenario = scenarios.get(target_id)
    if scenario is None:
        raise SelfTestError(f"scenario {target_id!r} not found in {scenario_path}")

    with WorkerPool(
        metadata_path=str(metadata_path),
        metadata_sha256=config.env.vendored_env_metadata_sha256,
        scenario_path=str(scenario_path),
        scenario_sha256=config.env.vendored_rl_scenarios_sha256,
        worker_count=config.profile.env_worker_count,
        episodes_per_worker=config.profile.env_episodes_per_worker,
        create_timeout_s=config.env.create_timeout_s,
        step_timeout_s=config.env.step_timeout_s,
        verify_timeout_s=config.env.verify_timeout_s,
    ) as pool:
        initial = run_episode(pool, scenario, (), episode_id="selftest-initial")
        partial = run_episode(pool, scenario, sequence[:1], episode_id="selftest-partial")
        full = run_episode(pool, scenario, sequence, episode_id="selftest-full")

    # Exact fractions, not inequalities: a verifier stuck at 0.0 satisfies
    # "partial < full" while measuring nothing.
    expected_partial = round(1 / scenario.check_count, 4)
    problems: list[str] = []
    if full.reward != 1.0:
        problems.append(f"correct sequence scored {full.reward}, expected 1.0")
    if partial.reward != expected_partial:
        problems.append(f"partial sequence scored {partial.reward}, expected {expected_partial}")
    if initial.reward >= full.reward:
        problems.append(f"initial state scored {initial.reward}, not below the correct sequence")
    for outcome in (initial, partial, full):
        if outcome.name_errors:
            problems.append(
                f"{outcome.name_errors} check(s) failed with NameError -- "
                "initial_state did not reach the verifier"
            )

    report = {
        "scenario": scenario.task_id,
        "env_id": scenario.env_id,
        "K": scenario.check_count,
        "initial": initial.to_dict(),
        "partial": partial.to_dict(),
        "full": full.to_dict(),
        "ok": not problems,
    }
    print(json.dumps(report, indent=2))
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    return 0
