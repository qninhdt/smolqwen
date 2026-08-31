"""`smolqwen rollout-bench`: the equivalence gate and the A/B measurement.

Two jobs, in a fixed order because the second is meaningless without the
first:

1. **Equivalence** — the same scenarios and deterministic scripted policy run
   through a serial factory-adapter semantic oracle and through the async
   scheduler. Rewards and observations must match exactly.
2. **Diagnostic comparison** — a scripted scheduler-and-environment ceiling.
   The serial oracle is not TRL's batched `_tool_call_loop`, so its throughput
   is never labeled as the accepted turn-synchronous baseline. The
   real colocated-vLLM rows require the Phase 7 trainer construction (including
   weight sync) and are left explicitly pending rather than mislabeled as a GPU
   measurement merely because CUDA happens to be visible.

The scripted policy is a policy, not a stub: it emits real
`serialize_tool_call` XML — byte-compatible with the chat template's own
serialization — so parsing, observation appending, and mask construction run
the production code path. Scenarios without a known script terminate after
their first turn, which is a valid deterministic episode (the initial state's
reward) and exercises the single-turn path of the mask builder.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from smolqwen.data.loader import ToolCall
from smolqwen.data.tool_call_xml import serialize_tool_call
from smolqwen.env.parse import parse_turn
from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import EnvSpec, load_env_specs
from smolqwen.env.scenarios import Scenario, load_scenarios
from smolqwen.env.selftest import DEFAULT_SCENARIO_ID, DEFAULT_SCRIPT
from smolqwen.rollout.episode import Episode
from smolqwen.rollout.factory_env import make_environment_factories
from smolqwen.rollout.scheduler import PoolDispatcher, ScenarioBinding, SchedulerConfig


class BenchError(RuntimeError):
    """Raised when the bench cannot construct its workload."""


def scripted_policy(
    scripts: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
) -> Any:
    """A deterministic policy: known scripts by task, immediate answer otherwise."""

    def policy(episode_id: str, turn_index: int, messages: Sequence[Mapping[str, Any]]) -> str:
        task_id = episode_id.split("@", 1)[0]
        script = scripts.get(task_id, ())
        if turn_index < len(script):
            name, arguments = script[turn_index]
            return serialize_tool_call(ToolCall(name=name, arguments=dict(arguments)))
        return "All required actions are complete."

    return policy


def build_bindings(
    *,
    scenarios: Sequence[Scenario],
    env_specs: Mapping[str, EnvSpec],
    episodes: int,
    num_generations: int,
) -> list[ScenarioBinding]:
    """Expand scenarios into `episodes` positions in group-contiguous order.

    The grouping mirrors TRL's `RepeatSampler`: G consecutive positions per
    scenario, which is the ordering `rewards.view(-1, num_generations)` assumes.
    """
    bindings: list[ScenarioBinding] = []
    position = 0
    while len(bindings) < episodes:
        for scenario in scenarios:
            if len(bindings) >= episodes:
                break
            env_spec = env_specs.get(scenario.env_id)
            if env_spec is None:
                raise BenchError(f"{scenario.task_id}: env {scenario.env_id} not in metadata")
            bindings.append(
                ScenarioBinding(
                    scenario=scenario,
                    group_index=len(bindings) // max(1, num_generations),
                    tool_schemas=tuple(env_spec.tools),
                    env_introduction=env_spec.introduction(),
                )
            )
            position += 1
    return bindings


def run_serial_factory_oracle_scripted(
    *,
    pool: WorkerPool,
    bindings: Sequence[ScenarioBinding],
    env_specs: Mapping[str, EnvSpec],
    policy: Any,
) -> list[dict[str, Any]]:
    """A semantic oracle: serial factory-adapter episodes, one at a time.

    This deliberately does not claim TRL baseline throughput: `_tool_call_loop`
    owns batching and pooling behavior that a handwritten loop cannot reproduce.
    Its narrower job is semantic correctness, and every environment still
    executes in a spawned worker through the same adapter used by the real
    `environment_factory` trainer.
    """
    scenarios = [binding.scenario for binding in bindings]
    factories = make_environment_factories(env_specs=env_specs, scenarios=scenarios, pool=pool)
    outcomes: list[dict[str, Any]] = []
    for position, binding in enumerate(bindings):
        factory = factories[binding.scenario.env_id]
        env = factory()
        env.reset(
            task_id=binding.scenario.task_id,
            prompt=[{"role": "user", "content": binding.scenario.task}],
        )
        episode_id = f"{binding.scenario.task_id}@{position}"
        messages: list[dict[str, Any]] = []
        observations: list[str] = []
        steps = 0
        terminal = "final_answer"
        while True:
            text = policy(episode_id, steps, messages)
            messages.append({"role": "assistant", "content": text})
            turn = parse_turn(text, available_tools=binding.tool_names)
            if turn.outcome == "no_call":
                break
            if turn.is_invalid_call:
                observations.append(turn.observation())
                messages.append({"role": "tool", "content": observations[-1]})
                steps += 1
                if steps > 32:
                    terminal = "step_cap"
                    break
                continue
            observation = getattr(env, str(turn.name))(**dict(turn.arguments or {}))
            observations.append(str(observation))
            messages.append({"role": "tool", "content": observations[-1]})
            steps += 1
            if steps > 32:
                terminal = "step_cap"
                break
        outcomes.append(
            {
                "episode_id": episode_id,
                "task_id": binding.scenario.task_id,
                "reward": env.get_reward(),
                "observations": list(observations),
                "steps": steps,
                "terminal": terminal,
            }
        )
        env._destroy()
    return outcomes


def check_equivalence(
    *,
    oracle_outcomes: Sequence[Mapping[str, Any]],
    episodes: Sequence[Episode],
) -> list[str]:
    """Compare oracle and async results position by position. Empty list = pass."""
    problems: list[str] = []
    if len(oracle_outcomes) != len(episodes):
        return [f"row count mismatch: oracle {len(oracle_outcomes)} vs async {len(episodes)}"]
    for oracle, episode in zip(oracle_outcomes, episodes, strict=True):
        if oracle["task_id"] != episode.scenario_id:
            problems.append(
                f"position misaligned: oracle {oracle['task_id']} vs async {episode.scenario_id}"
            )
            continue
        if (episode.reward is None and oracle["reward"] is not None) or (
            episode.reward is not None and abs(episode.reward - float(oracle["reward"])) > 1e-9
        ):
            problems.append(
                f"{episode.episode_id}: reward {episode.reward} != oracle {oracle['reward']}"
            )
        if list(episode.observations) != list(oracle["observations"]):
            problems.append(
                f"{episode.episode_id}: observations diverge — "
                f"async {len(episode.observations)}, oracle {len(oracle['observations'])}"
            )
    return problems


def scheduler_config_for(config: Any) -> SchedulerConfig:
    """Map the resolved GrpoConfig onto the scheduler's semantic knobs."""
    return SchedulerConfig(
        generation_concurrency=config.profile.generation_concurrency,
        max_env_steps=config.profile.max_env_steps,
        episode_timeout_s=config.episode_timeout_s,
        max_new_tokens_per_step=config.profile.max_new_tokens_per_step,
        max_model_len=config.vllm_max_model_len,
        temperature=config.temperature,
        top_p=config.top_p,
        fork_threshold_tokens=config.fork_threshold_tokens,
    )


def load_workload(config: Any) -> tuple[dict[str, EnvSpec], dict[str, Scenario]]:
    """Env specs (parent-side, never exec'd) and the RL scenario table."""
    env_specs = load_env_specs(
        config.env.vendored_env_metadata, sha256=config.env.vendored_env_metadata_sha256
    )
    scenarios = {
        scenario.task_id: scenario
        for scenario in load_scenarios(
            config.env.vendored_rl_scenarios, sha256=config.env.vendored_rl_scenarios_sha256
        )
    }
    return env_specs, scenarios


def default_bench_scenarios(scenarios: Mapping[str, Scenario]) -> list[Scenario]:
    """The bench workload: the selftest scenario, which has a known script."""
    preferred = [DEFAULT_SCENARIO_ID]
    return [scenarios[task] for task in preferred if task in scenarios]


def gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def write_ab_report(path: Path, sections: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(["# Rollout A/B report", *sections]) + "\n", encoding="utf-8")


def run_equivalence(config: Any, *, episodes: int, verbose: bool = True) -> dict[str, Any]:
    """The CPU gate: both paths scripted, real pool, real scenarios."""
    from smolqwen.rollout.generation import ScriptedPolicyBackend
    from smolqwen.rollout.rollout_func import make_scheduler
    from smolqwen.tokenizer import load_tokenizer

    env_specs, scenarios = load_workload(config)
    workload = default_bench_scenarios(scenarios)
    if not workload:
        raise BenchError("no bench scenario with a known script found in the scenario table")
    bindings = build_bindings(
        scenarios=workload,
        env_specs=env_specs,
        episodes=episodes,
        num_generations=config.profile.num_generations,
    )
    policy = scripted_policy({DEFAULT_SCENARIO_ID: DEFAULT_SCRIPT})
    tokenizer = load_tokenizer(config.model_id)

    with _pool_for(config) as pool:
        oracle_started = time.monotonic()
        oracle = run_serial_factory_oracle_scripted(
            pool=pool, bindings=bindings, env_specs=env_specs, policy=policy
        )
        oracle_wall = time.monotonic() - oracle_started

        dispatcher = PoolDispatcher(pool)
        backend = ScriptedPolicyBackend(policy, _encode_for(tokenizer))
        scheduler = make_scheduler(
            backend=backend,
            dispatcher=dispatcher,
            tokenizer=tokenizer,
            config=scheduler_config_for(config),
        )
        async_started = time.monotonic()
        async_episodes = scheduler.run(bindings)
        async_wall = time.monotonic() - async_started
        dispatcher.shutdown()

    problems = check_equivalence(oracle_outcomes=oracle, episodes=async_episodes)
    report = {
        "episodes": len(bindings),
        "oracle_wall_s": round(oracle_wall, 3),
        "async_wall_s": round(async_wall, 3),
        "oracle_rewards": [float(o["reward"]) for o in oracle],
        "async_rewards": [e.reward for e in async_episodes],
        "ok": not problems,
        "problems": problems,
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def run_bench(config: Any, *, args: Any) -> int:
    """The `rollout-bench` CLI entry: gate on equivalence, then measure.

    Without a CUDA device the A/B table is written with its GPU rows pending
    rather than fabricated — the equivalence gate still runs, because it needs
    only the pool and a tokenizer.
    """
    from smolqwen.rollout.metrics import (
        AB_TABLE_HEADER,
    )

    sections: list[str] = []
    equivalence = run_equivalence(config, episodes=args.episodes)
    sections.append(
        "## Equivalence (scripted policy, real pool)\n\n"
        f"{'PASS' if equivalence['ok'] else 'FAIL'} — {equivalence['episodes']} episodes, "
        f"oracle {equivalence['oracle_wall_s']}s, async {equivalence['async_wall_s']}s."
    )
    if not equivalence["ok"]:
        sections.append("Problems:\n" + "\n".join(f"- {p}" for p in equivalence["problems"]))
        write_ab_report(REPORT_PATH, sections)
        print(f"equivalence FAILED; wrote {REPORT_PATH}", file=sys.stderr)
        return 1

    rows = _run_scripted_ab(config, args)
    sections.append(
        "## Scripted diagnostic comparison\n\n"
        "Not the accepted TRL turn-synchronous A/B baseline.\n\n"
        + AB_TABLE_HEADER
        + "\n"
        + "\n".join(rows)
    )
    cuda_note = "CUDA is visible" if gpu_available() else "no CUDA device is visible"
    sections.append(
        "## Colocated-vLLM A/B\n\n"
        f"PENDING — {cuda_note}, but a card alone is not sufficient. These rows\n"
        "must run through the separately constructed Phase 7 trainers so the\n"
        "timeline includes real LoRA weight sync and prefix-cache invalidation.\n"
        "The scripted rows above are not presented as GPU measurements."
    )
    write_ab_report(REPORT_PATH, sections)
    print(f"equivalence and scripted A/B passed; GPU trainer A/B pending — wrote {REPORT_PATH}")
    return 0


REPORT_PATH = Path("artifacts/rollout/ab_report.md")


def _run_scripted_ab(config: Any, args: Any) -> list[str]:
    """Both scheduling paths with one deterministic policy, honestly labeled."""
    import time

    from smolqwen.rollout.generation import ScriptedPolicyBackend
    from smolqwen.rollout.metrics import ABReportRow, summarize_episodes
    from smolqwen.rollout.profiler import format_timeline, profile_rollout
    from smolqwen.rollout.rollout_func import make_scheduler
    from smolqwen.rollout.scheduler import PoolDispatcher
    from smolqwen.tokenizer import load_tokenizer

    env_specs, scenarios = load_workload(config)
    workload = default_bench_scenarios(scenarios)
    bindings = build_bindings(
        scenarios=workload,
        env_specs=env_specs,
        episodes=args.episodes,
        num_generations=config.profile.num_generations,
    )
    policy = scripted_policy({DEFAULT_SCENARIO_ID: DEFAULT_SCRIPT})
    tokenizer = load_tokenizer(config.model_id)
    profile_name = getattr(args, "profile", None) or "default"
    rows: list[str] = []
    extra: list[str] = []

    with _pool_for(config) as pool:
        for path in [part.strip() for part in args.paths.split(",")]:
            if path == "serial_oracle":
                started = time.monotonic()
                outcomes = run_serial_factory_oracle_scripted(
                    pool=pool, bindings=bindings, env_specs=env_specs, policy=policy
                )
                wall = time.monotonic() - started
                rewards = [float(outcome["reward"]) for outcome in outcomes]
                rows.append(
                    ABReportRow(
                        path=path,
                        profile=profile_name,
                        episodes_per_hour=len(outcomes) / wall * 3600,
                        tokens_per_s=0.0,
                        gpu_util_mean=0.0,
                        gpu_util_peak=0.0,
                        mean_reward=sum(rewards) / len(rewards),
                        notes="serial semantic oracle; not TRL baseline",
                    ).as_markdown()
                )
            elif path == "async":
                dispatcher = PoolDispatcher(pool)
                scheduler = make_scheduler(
                    backend=ScriptedPolicyBackend(policy, _encode_for(tokenizer)),
                    dispatcher=dispatcher,
                    tokenizer=tokenizer,
                    config=scheduler_config_for(config),
                )
                started = time.monotonic()
                episodes_run = scheduler.run(bindings)
                wall = time.monotonic() - started
                dispatcher.shutdown()
                summary = summarize_episodes(episodes=episodes_run, wall_s=wall)
                timeline = profile_rollout(
                    episodes=episodes_run,
                    wall_s=wall,
                    events=scheduler.events,
                    queue_depth=scheduler.queue_depth_samples,
                    stage_intervals=scheduler.stage_intervals,
                )
                rows.append(
                    ABReportRow(
                        path=path,
                        profile=profile_name,
                        episodes_per_hour=float(cast(float | int, summary["episodes_per_hour"])),
                        tokens_per_s=float(cast(float | int, summary["tokens_per_s"])),
                        gpu_util_mean=0.0,
                        gpu_util_peak=0.0,
                        mean_reward=float(cast(float | int, summary["mean_reward"])),
                        notes="scripted backend; the vLLM row is the trainer run",
                    ).as_markdown()
                )
                extra.append(
                    f"## Async timeline ({profile_name})\n\n```\n{format_timeline(timeline)}\n```"
                )
            else:
                raise BenchError(f"unknown rollout path {path!r}; expected serial_oracle or async")
    rows.extend(extra)
    return rows


def _encode_for(tokenizer: Any) -> Any:
    from smolqwen.rollout.rollout_func import encode_ids

    return lambda text: encode_ids(tokenizer, text)


def _pool_for(config: Any) -> WorkerPool:
    return WorkerPool(
        metadata_path=str(config.env.vendored_env_metadata),
        metadata_sha256=config.env.vendored_env_metadata_sha256,
        scenario_path=str(config.env.vendored_rl_scenarios),
        scenario_sha256=config.env.vendored_rl_scenarios_sha256,
        worker_count=config.profile.env_worker_count,
        episodes_per_worker=config.profile.env_episodes_per_worker,
        create_timeout_s=config.env.create_timeout_s,
        step_timeout_s=config.env.step_timeout_s,
        verify_timeout_s=config.env.verify_timeout_s,
    )
