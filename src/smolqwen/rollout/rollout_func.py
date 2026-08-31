"""TRL's `rollout_func` entry point: the production path's boundary.

The callable TRL invokes receives `(prompts, trainer)` and must return, per
input row and positionally aligned, `prompt_ids`, `completion_ids`, `logprobs`
(required keys, verified by TRL) plus `env_mask` marking model tokens 1 and
appended-observation tokens 0. Three boundary rules this module enforces
because nothing downstream would:

1. **`trainer.tools` must be empty.** `env_mask` is read only on the `else`
   branch of TRL's `if self.tools:`; a trainer that also holds tools or an
   `environment_factory` silently discards the mask and trains on
   observations. The entry assertion makes that construction fail loudly here
   instead of silently there.
2. **Row count and order match the prompts exactly.** `_calculate_rewards`
   sizes from `len(prompts)` and zips `strict=True`; advantages come from a
   positional `view(-1, num_generations)`. The scheduler guarantees this; the
   boundary re-asserts it before returning.
3. **`len(logprobs) == len(completion_ids) == len(env_mask)` per row, NaN at
   observation positions.** A shorter `logprobs` array is right-padded with
   0.0 — probability 1 — and shifts every model token after the first
   observation against the wrong position.

Bindings are resolved per call: TRL's `RepeatSampler` reshuffles between steps,
so scenario identity cannot be bound once at construction. The caller supplies
`resolve_bindings(prompts)` returning a positionally aligned
`ScenarioBinding` list; the production trainer builds it from the dataset, the
bench from its scenario sample.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from smolqwen.data.loader import Message, parse_message
from smolqwen.prompts import build_system_prompt
from smolqwen.rollout.episode import Episode
from smolqwen.rollout.generation import GenerationBackend, VllmColocateBackend
from smolqwen.rollout.metrics import (
    GpuUtilizationSampler,
    drift_distribution,
    summarize_episodes,
    wandb_log_payload,
)
from smolqwen.rollout.profiler import profile_rollout
from smolqwen.rollout.scheduler import (
    EnvDispatcher,
    RolloutScheduler,
    ScenarioBinding,
    SchedulerConfig,
)

# prompt rows are conversational message lists; TRL passes them through as-is
Prompts = Sequence[Sequence[Mapping[str, Any]]]
BindingResolver = Callable[[Prompts], Sequence[Any]]
BackendFactory = Callable[[Any], GenerationBackend]


class RolloutFuncError(RuntimeError):
    """Raised when the TRL boundary contract is violated."""


def assert_trainer_exclusive(trainer: Any) -> None:
    """Reject a trainer that would discard the rollout function's env mask."""
    tools = getattr(trainer, "tools", None)
    factories = getattr(trainer, "environment_factories", None)
    if tools or factories is not None:
        raise RolloutFuncError(
            "rollout_func requires trainer.tools and environment_factories to be "
            "empty: env_mask is read only on TRL's no-tools branch. Construct "
            "the production trainer with tools=None, environment_factory=None."
        )


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    """Token ids for rendered text, no special tokens — as `render.py` encodes."""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def initial_messages_for(binding: Any) -> list[Message]:
    """The exact TRL prompt, or the canonical dataset prompt for direct benches."""
    if binding.initial_messages:
        return list(binding.initial_messages)
    scenario = binding.scenario
    system = build_system_prompt(conversational=True, env_introduction=binding.env_introduction)
    return [
        Message(role="system", content=system),
        Message(role="user", content=scenario.task),
    ]


def attach_prompt_messages(
    prompts: Prompts, bindings: Sequence[ScenarioBinding]
) -> list[ScenarioBinding]:
    """Bind each resolved scenario to the exact conversational row TRL supplied."""
    if len(prompts) != len(bindings):
        raise RolloutFuncError(
            f"resolver returned {len(bindings)} bindings for {len(prompts)} prompts"
        )
    return [
        replace(
            binding,
            initial_messages=tuple(parse_message(dict(message)) for message in prompt),
        )
        for prompt, binding in zip(prompts, bindings, strict=True)
    ]


def make_scheduler(
    *,
    backend: GenerationBackend,
    dispatcher: EnvDispatcher,
    tokenizer: Any,
    config: SchedulerConfig,
    wait_for: Callable[..., Any] | None = None,
) -> RolloutScheduler:
    """Wire the scheduler's render/decode seams to one tokenizer.

    `wait_for` overrides the blocking wait — the simulated-clock tests inject
    their dispatcher's virtual wait here.
    """
    from smolqwen.data.render import render_prefix

    def render_prefix_ids(messages: Sequence[Message], binding: ScenarioBinding) -> list[int]:
        text = render_prefix(
            tokenizer, messages, tools=list(binding.tool_schemas), add_generation_prompt=True
        )
        return encode_ids(tokenizer, text)

    def decode(ids: Sequence[int]) -> str:
        return str(tokenizer.decode(list(ids), skip_special_tokens=False))

    return RolloutScheduler(
        backend=backend,
        dispatcher=dispatcher,
        initial_messages=initial_messages_for,
        render_prefix_ids=render_prefix_ids,
        decode=decode,
        config=config,
        wait_for=wait_for,
    )


def make_rollout_func(
    *,
    resolve_bindings: BindingResolver,
    config: SchedulerConfig,
    dispatcher: EnvDispatcher,
    tokenizer: Any,
    backend_factory: BackendFactory | None = None,
) -> Callable[[Prompts, Any], dict[str, list[Any]]]:
    """Build the `rollout_func` TRL accepts.

    `backend_factory(trainer)` defaults to the vLLM colocate backend — the
    production setting. Tests and the equivalence gate pass
    `ScriptedPolicyBackend` through a factory that ignores the trainer.
    """
    backend_builder = backend_factory or (lambda trainer: VllmColocateBackend(trainer))

    def rollout_func(prompts: Prompts, trainer: Any) -> dict[str, list[Any]]:
        assert_trainer_exclusive(trainer)
        bindings = attach_prompt_messages(prompts, resolve_bindings(prompts))
        for prompt, binding in zip(prompts, bindings, strict=True):
            task = binding.scenario.task
            content = _last_user_content(prompt)
            if task not in (content or ""):
                raise RolloutFuncError(
                    "prompt row does not carry its binding's task text; positional "
                    "alignment between prompts and scenarios is broken"
                )

        scheduler = make_scheduler(
            backend=backend_builder(trainer),
            dispatcher=dispatcher,
            tokenizer=tokenizer,
            config=config,
        )
        gpu_sampler = GpuUtilizationSampler()
        gpu_sampler.start()
        started = time.monotonic()
        try:
            episodes = scheduler.run(bindings)
        finally:
            gpu = gpu_sampler.stop()
        wall_s = time.monotonic() - started
        timeline = profile_rollout(
            episodes=episodes,
            wall_s=wall_s,
            events=scheduler.events,
            queue_depth=scheduler.queue_depth_samples,
            stage_intervals=scheduler.stage_intervals,
        )
        log = getattr(trainer, "log", None)
        if callable(log):
            log(
                wandb_log_payload(
                    summarize_episodes(episodes=episodes, wall_s=wall_s),
                    drift_distribution(episodes),
                    gpu,
                    timeline,
                )
            )
        return assemble_output(episodes, scheduler)

    return rollout_func


def assemble_output(
    episodes: Sequence[Episode], scheduler: RolloutScheduler
) -> dict[str, list[Any]]:
    """The TRL return dict, assembled from each episode's mask builder.

    Asserted here, not trusted: the three lengths and the NaN-at-observation
    contract are the silent-corruption boundary, so a violation fails before
    TRL ever sees the batch.
    """
    prompt_rows: list[list[Any]] = []
    completion_rows: list[list[Any]] = []
    logprob_rows: list[list[Any]] = []
    mask_rows: list[list[Any]] = []
    rewards: list[float] = []
    check_rows: list[list[bool]] = []
    terminal_reasons: list[str | None] = []
    scenario_ids: list[str] = []
    group_indices: list[int] = []
    trajectories: list[dict[str, Any]] = []
    for episode in episodes:
        builder = scheduler.episode_builder(episode.episode_id)
        prompt_ids = list(builder.prompt_ids)
        completion_ids = list(builder.completion_ids)
        logprobs = list(builder.logprobs)
        env_mask = list(builder.env_mask)
        if not (len(logprobs) == len(completion_ids) == len(env_mask)):
            raise RolloutFuncError(
                f"{episode.episode_id}: lengths diverge — completion "
                f"{len(completion_ids)}, logprobs {len(logprobs)}, mask "
                f"{len(env_mask)}; TRL would right-pad and misalign the IS ratio"
            )
        if episode.observations and not any(math.isnan(value) for value in logprobs):
            raise RolloutFuncError(
                f"{episode.episode_id}: {len(episode.observations)} observations "
                "but no NaN logprob; observation positions must be NaN so TRL "
                "maps them to ratio 1"
            )
        if episode.observations and all(env_mask):
            raise RolloutFuncError(
                f"{episode.episode_id}: observations present but env_mask is "
                "all ones; the mask would train the model on tool output"
            )
        episode.prompt_ids = prompt_ids
        episode.completion_ids = completion_ids
        episode.logprobs = logprobs
        prompt_rows.append(prompt_ids)
        completion_rows.append(completion_ids)
        logprob_rows.append(logprobs)
        mask_rows.append(env_mask)
        if episode.reward is None:
            raise RolloutFuncError(f"{episode.episode_id}: scored rollout has no verifier reward")
        rewards.append(float(episode.reward))
        check_rows.append(list(episode.per_check_bools or ()))
        terminal_reasons.append(episode.terminal_reason)
        scenario_ids.append(episode.scenario_id)
        group_indices.append(episode.group_index)
        trajectories.append(episode.to_row())
    return {
        "prompt_ids": prompt_rows,
        "completion_ids": completion_rows,
        "logprobs": logprob_rows,
        "env_mask": mask_rows,
        # TRL forwards every extra field to reward functions. Phase 7 consumes
        # the verifier result already computed inside the isolated environment;
        # it never re-scores decoded text or adds shaping terms.
        "rollout_reward": rewards,
        "per_check_bools": check_rows,
        "terminal_reason": terminal_reasons,
        "scenario_id": scenario_ids,
        "group_index": group_indices,
        "trajectory": trajectories,
    }


def _last_user_content(prompt: Sequence[Mapping[str, Any]]) -> str | None:
    for message in reversed(list(prompt)):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
    return None
