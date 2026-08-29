"""Held-out EnvScaler evaluation through the isolated Phase 4 worker pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from smolqwen.config_models import EnvRuntimeConfig, EvalConfig, ProfileConfig, StrictModel
from smolqwen.env.parse import parse_turn
from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import load_env_specs
from smolqwen.env.scenarios import Scenario, build_scenario_set, iter_by_env
from smolqwen.eval.adapters.base import AdapterResult, BenchmarkAdapter, EvalTask, StepResult
from smolqwen.eval.manifest import hash_json, sha256_file
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.tool_calls import (
    is_completion_signal,
    is_error_signal,
    parse_normalized_json_calls,
)
from smolqwen.prompts import build_system_prompt

ADAPTER_NAME = "envscaler_heldout"


class EnvScalerAdapterOptions(StrictModel):
    """Held-out environment settings owned and validated by this adapter."""

    env_count: int = Field(default=10, ge=1)
    scenarios_per_env: int = Field(default=8, ge=1)
    runtime: EnvRuntimeConfig = EnvRuntimeConfig()


@dataclass
class _Episode:
    scenario: Scenario
    episode_id: str
    created: bool = False
    failed: bool = False
    invalid_calls: int = 0


def select_heldout_scenarios(
    scenarios: Sequence[Scenario], *, env_count: int, per_env: int
) -> tuple[Scenario, ...]:
    """Stable held-out slice: sorted environments, then sorted task IDs per environment."""
    selected: list[Scenario] = []
    for _, grouped in list(iter_by_env(scenarios))[:env_count]:
        selected.extend(sorted(grouped, key=lambda scenario: scenario.task_id)[:per_env])
    return tuple(selected)


class EnvScalerHeldoutAdapter:
    """One pool-backed episode per held-out RL scenario; no dataset code runs in the parent."""

    def __init__(
        self,
        runtime_config: EnvRuntimeConfig,
        profile: ProfileConfig,
        *,
        env_count: int,
        scenarios_per_env: int,
    ) -> None:
        self.runtime_config = runtime_config
        self.profile = profile
        scenario_set = build_scenario_set(
            runtime_config.vendored_rl_scenarios,
            env_split_manifest=runtime_config.env_split_manifest,
            sha256=runtime_config.vendored_rl_scenarios_sha256,
        )
        self.scenarios = select_heldout_scenarios(
            scenario_set.scenarios, env_count=env_count, per_env=scenarios_per_env
        )
        self._specs = load_env_specs(
            runtime_config.vendored_env_metadata,
            sha256=runtime_config.vendored_env_metadata_sha256,
        )
        self._episodes: dict[str, _Episode] = {}
        self._pool: WorkerPool | None = None

    @property
    def heldout_env_ids(self) -> tuple[str, ...]:
        return tuple(sorted({scenario.env_id for scenario in self.scenarios}))

    @property
    def invalid_calls(self) -> dict[str, int]:
        return {task_id: episode.invalid_calls for task_id, episode in self._episodes.items()}

    def load_tasks(self) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        for scenario in self.scenarios:
            spec = self._specs[scenario.env_id]
            system_prompt = build_system_prompt(
                conversational=False,
                env_introduction=spec.introduction(),
            )
            task = EvalTask(
                task_id=scenario.task_id,
                category="envscaler_heldout",
                prompt=scenario.task,
                tools=spec.tools,
                metadata={
                    "env_id": scenario.env_id,
                    "system_prompt": system_prompt,
                },
            )
            self._episodes[task.task_id] = _Episode(scenario=scenario, episode_id=task.task_id)
            tasks.append(task)
        return tasks

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        self._ensure_created(task)
        if history:
            return [dict(message) for message in history]
        return [
            {"role": "system", "content": task.metadata["system_prompt"]},
            {"role": "user", "content": task.prompt},
        ]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        episode = self._episode(task)
        parsed = parse_turn(
            completion,
            available_tools=frozenset(
                str(tool.get("function", {}).get("name", "")) for tool in task.tools
            ),
        )
        if parsed.outcome == "no_call":
            json_calls = parse_normalized_json_calls(completion)
            if json_calls:
                return self._execute_calls(task, json_calls)
            if is_error_signal(completion):
                episode.failed = True
                return StepResult("Trajectory failed.", complete=True)
            if is_completion_signal(completion):
                return StepResult("Trajectory finished.", complete=True)
            return StepResult("Error: Function call or completion signal not found.")
        if not parsed.ok:
            episode.invalid_calls += int(parsed.is_invalid_call)
            return StepResult(parsed.observation())
        assert parsed.name is not None and parsed.arguments is not None
        return self._execute_calls(task, [(parsed.name, parsed.arguments)])

    def score(self, task: EvalTask) -> AdapterResult:
        episode = self._episode(task)
        if not episode.created:
            self._ensure_created(task)
        try:
            if episode.failed:
                return AdapterResult(0.0, False)
            result = self._pool_or_raise().score(episode.episode_id)
            if not result.ok or not isinstance(result.value, Mapping):
                return AdapterResult(0.0, False)
            reward = float(result.value["reward"])
            return AdapterResult(reward, reward == 1.0)
        finally:
            self._pool_or_raise().destroy(episode.episode_id)

    def invalid_call_count(self, task: EvalTask) -> int:
        return self._episode(task).invalid_calls

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        ordered = sorted(tasks, key=lambda task: task.task_id)
        config = self.runtime_config
        return {
            "env_metadata_path": config.vendored_env_metadata,
            "env_metadata_sha256": sha256_file(config.vendored_env_metadata),
            "rl_scenarios_path": config.vendored_rl_scenarios,
            "rl_scenarios_sha256": sha256_file(config.vendored_rl_scenarios),
            "env_split_path": config.env_split_manifest,
            "env_split_sha256": sha256_file(config.env_split_manifest),
            "heldout_env_ids": list(self.heldout_env_ids),
            "heldout_task_ids": [task.task_id for task in ordered],
            "task_count": len(ordered),
            "task_ids_hash": hash_json([task.task_id for task in ordered]),
            "system_prompt_hash": hash_json([task.metadata["system_prompt"] for task in ordered]),
            "tool_schema_hash": hash_json([task.tools for task in ordered]),
        }

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        return aggregate(tasks)

    def _execute_calls(
        self,
        task: EvalTask,
        calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> StepResult:
        episode = self._episode(task)
        available = frozenset(str(tool.get("function", {}).get("name", "")) for tool in task.tools)
        observations: list[str] = []
        env_steps = 0
        for name, arguments in calls:
            if name not in available:
                episode.invalid_calls += 1
                observations.append(f"Error: unknown_tool: {name!r} is not available")
                continue
            result = self._pool_or_raise().step(episode.episode_id, name, arguments)
            env_steps += 1
            if not result.ok:
                episode.invalid_calls += 1
                observations.append(f"Error: {result.reason}: {result.detail or ''}")
            else:
                observations.append(str(result.value))
        rendered = observations[0] if len(observations) == 1 else str(observations)
        return StepResult(
            rendered,
            env_steps=env_steps,
            tool_observations=tuple(observations),
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _ensure_created(self, task: EvalTask) -> None:
        episode = self._episode(task)
        if episode.created:
            return
        scenario = episode.scenario
        result = self._pool_or_raise().create(
            episode.episode_id,
            env_id=scenario.env_id,
            env_class_name=scenario.env_class_name,
            init_config=scenario.init_config,
            checklist=scenario.checklist,
            checklist_id=scenario.task_id,
        )
        if not result.ok:
            raise RuntimeError(f"{scenario.task_id}: environment create failed: {result.detail}")
        episode.created = True

    def _pool_or_raise(self) -> WorkerPool:
        if self._pool is None:
            config = self.runtime_config
            self._pool = WorkerPool(
                metadata_path=str(Path(config.vendored_env_metadata)),
                metadata_sha256=config.vendored_env_metadata_sha256,
                scenario_path=str(Path(config.vendored_rl_scenarios)),
                scenario_sha256=config.vendored_rl_scenarios_sha256,
                worker_count=self.profile.env_worker_count,
                episodes_per_worker=self.profile.env_episodes_per_worker,
                create_timeout_s=config.create_timeout_s,
                step_timeout_s=config.step_timeout_s,
                verify_timeout_s=config.verify_timeout_s,
            )
        return self._pool

    def _episode(self, task: EvalTask) -> _Episode:
        try:
            return self._episodes[task.task_id]
        except KeyError as exc:
            raise KeyError(f"EnvScaler task was not loaded: {task.task_id}") from exc


def create_adapter(config: EvalConfig) -> BenchmarkAdapter:
    """Factory discovered by :mod:`smolqwen.eval.adapters`."""

    options = EnvScalerAdapterOptions.model_validate(config.adapter_options.get(ADAPTER_NAME, {}))
    return EnvScalerHeldoutAdapter(
        options.runtime,
        config.profile,
        env_count=options.env_count,
        scenarios_per_env=options.scenarios_per_env,
    )
