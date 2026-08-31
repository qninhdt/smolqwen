"""Agentic GRPO assembly over the Phase 6 asynchronous rollout function."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator, Mapping, Sized
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from torch.utils.data import Sampler
from transformers import TrainerCallback

from smolqwen.artifacts import CheckpointStore, ResumeState
from smolqwen.config_models import GrpoConfig
from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import EnvSpec, load_env_specs
from smolqwen.env.scenarios import Scenario, build_scenario_set
from smolqwen.eval.adapters.envscaler_heldout import select_heldout_scenarios
from smolqwen.prompts import build_system_prompt
from smolqwen.rollout.bench import scheduler_config_for
from smolqwen.rollout.factory_env import make_environment_factories
from smolqwen.rollout.metrics import LogpDifferenceStopCallback
from smolqwen.rollout.rollout_func import Prompts, make_rollout_func
from smolqwen.rollout.scheduler import PoolDispatcher, ScenarioBinding
from smolqwen.tokenizer import assert_text_only_processing_class, load_tokenizer
from smolqwen.tracking import Tracker
from smolqwen.training.difficulty import (
    DifficultyProfile,
    profile_rewards,
    read_profile,
    weighted_scenario_order,
    write_profile,
)
from smolqwen.training.optim import (
    Toggle,
    cast_adapters,
    format_ledger,
    ledger,
    resolve_attn_implementation,
    resolve_liger,
)
from smolqwen.training.reward import verifier_reward


class GrpoError(RuntimeError):
    """Raised when a GRPO run cannot preserve its experiment contracts."""


@dataclass(frozen=True)
class CurriculumCursor:
    """Convert completed optimizer steps into consumed scenario groups."""

    start: int
    dataset_size: int
    groups_per_generation: int
    gradient_accumulation_steps: int
    steps_per_generation: int

    def at_step(self, global_step: int) -> int:
        if self.dataset_size <= 0:
            return 0
        micro_steps = max(0, global_step) * self.gradient_accumulation_steps
        generation_batches = micro_steps // self.steps_per_generation
        return (self.start + generation_batches * self.groups_per_generation) % self.dataset_size


class CursorRepeatSampler(Sampler[int]):
    """TRL's structured repeat sampler, rotated to a persisted curriculum cursor."""

    def __init__(
        self,
        data_source: Sized,
        *,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        cursor: int = 0,
    ) -> None:
        if batch_size < 1 or mini_repeat_count < 1 or repeat_count < 1:
            raise GrpoError("sampler repeat counts and batch size must be positive")
        self.num_samples = len(data_source)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.cursor = cursor % self.num_samples if self.num_samples else 0

    def __iter__(self) -> Iterator[int]:
        indexes = list(range(self.num_samples))
        indexes = indexes[self.cursor :] + indexes[: self.cursor]
        for offset in range(0, len(indexes), self.batch_size):
            chunk = indexes[offset : offset + self.batch_size]
            if len(chunk) != self.batch_size:
                continue
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        complete = (self.num_samples // self.batch_size) * self.batch_size
        return complete * self.mini_repeat_count * self.repeat_count


class CurriculumGRPOTrainerMixin:
    """Sampler override mixed into TRL's trainer at module import time below."""

    curriculum_cursor: CurriculumCursor

    def _get_train_sampler(self, dataset: Any | None = None) -> Sampler[int]:
        trainer = cast(Any, self)
        source = dataset if dataset is not None else trainer.train_dataset
        args = trainer.args
        num_generations = int(trainer.num_generations)
        num_iterations = int(trainer.num_iterations)
        return CursorRepeatSampler(
            source,
            mini_repeat_count=num_generations,
            batch_size=int(args.generation_batch_size) // num_generations,
            repeat_count=num_iterations * int(args.steps_per_generation),
            cursor=self.curriculum_cursor.start,
        )


class GroupVarianceStopCallback(TrainerCallback):
    """Stop when observed degenerate groups materially exceed the profile."""

    def __init__(self, *, after_steps: int, multiplier: float, margin: float) -> None:
        self.after_steps = after_steps
        self.multiplier = multiplier
        self.margin = margin

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        if logs is None or int(state.global_step) < self.after_steps:
            return control
        actual = logs.get("group_reward_variance/zero_fraction")
        predicted = logs.get("group_reward_variance/predicted_zero_fraction")
        if actual is not None and predicted is not None:
            threshold = min(1.0, float(predicted) * self.multiplier + self.margin)
            if float(actual) > threshold:
                control.should_training_stop = True
        return control


class GrpoCheckpointCallback(TrainerCallback):
    """Push adapter, optimizer checkpoint, run id, and sampler cursor together."""

    def __init__(self, store: CheckpointStore, tracker: Tracker, cursor: CurriculumCursor) -> None:
        self.store = store
        self.tracker = tracker
        self.cursor = cursor

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        if checkpoint.is_dir():
            self.store.save_adapter(checkpoint)
            self.store.write_resume_state(
                ResumeState(
                    wandb_run_id=self.tracker.run_id,
                    global_step=step,
                    sampler_cursor=self.cursor.at_step(step),
                )
            )
            self.store.push(commit_message=f"adapter and trainer state at step {step}")
        return control


@dataclass
class GrpoAssembly:
    trainer: Any
    tracker: Tracker
    store: CheckpointStore
    dispatcher: PoolDispatcher
    pool: WorkerPool
    toggles: tuple[Toggle, ...]
    resume_from: str | None
    cursor: CurriculumCursor
    train_task_ids: tuple[str, ...]
    eval_task_ids: tuple[str, ...]

    def shutdown(self) -> None:
        self.dispatcher.shutdown()
        self.pool.shutdown()
        self.tracker.finish()


def _cleanup_failed_assembly(
    dispatcher: PoolDispatcher | None, pool: WorkerPool | None, run: Tracker
) -> None:
    """Best-effort teardown that never masks the assembly failure being handled."""
    if dispatcher is not None:
        with suppress(Exception):
            dispatcher.shutdown()
    if pool is not None:
        with suppress(Exception):
            pool.shutdown()
    with suppress(Exception):
        run.finish()


def _resolve_resume(
    store: CheckpointStore, *, resume: bool
) -> tuple[str | None, ResumeState | None]:
    if not resume:
        return None, None
    state = store.read_resume_state()
    if state is not None and store.local_dir.is_dir():
        return str(store.local_dir), state
    if not store.enabled:
        raise GrpoError("--resume needs a local resume_state.json or configured hub_repo_id")
    revision = store.latest_revision()
    if revision is None:
        raise GrpoError("--resume found no pushed GRPO revision")
    local = store.pull(revision)
    state = store.read_resume_state()
    if state is None:
        raise GrpoError("pushed GRPO checkpoint has no resume_state.json")
    return str(local), state


def _lora_config(config: GrpoConfig) -> Any:
    from peft import LoraConfig as PeftLoraConfig

    target = config.lora.target_modules
    return PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=target if isinstance(target, str) else list(target),
        task_type="CAUSAL_LM",
        bias="none",
    )


def _grpo_args(
    config: GrpoConfig, *, attn: Toggle, use_liger: bool, report_to: list[str], use_vllm: bool
) -> Any:
    from trl import GRPOConfig as TrlGrpoConfig  # type: ignore[attr-defined]

    profile = config.profile
    training = config.training
    generation_batch_size = profile.generation_batch_size
    if generation_batch_size % profile.num_generations:
        raise GrpoError("generation_batch_size must be divisible by num_generations")
    return TrlGrpoConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=profile.micro_batch,
        per_device_eval_batch_size=profile.micro_batch,
        gradient_accumulation_steps=profile.grad_accum,
        generation_batch_size=generation_batch_size,
        num_generations=profile.num_generations,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        warmup_steps=training.warmup_ratio,
        weight_decay=training.weight_decay,
        lr_scheduler_type=training.lr_scheduler_type,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        eval_steps=training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=training.seed,
        bf16=config.optimization.bf16,
        gradient_checkpointing=config.optimization.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=use_liger,
        model_init_kwargs={
            "dtype": "bfloat16" if config.optimization.bf16 else "float32",
            "attn_implementation": attn.name,
        },
        beta=config.beta,
        loss_type=config.loss_type,
        temperature=config.temperature,
        top_p=config.top_p,
        max_completion_length=config.vllm_max_model_len,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=profile.vllm_kv_fraction,
        vllm_max_model_length=config.vllm_max_model_len,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=True,
        log_completions=True,
        num_completions_to_print=config.curriculum.trajectory_samples_per_log,
        remove_unused_columns=False,
        ignore_data_skip=True,
        report_to=report_to,
        run_name=config.tracking.run_name,
        push_to_hub=False,
        save_total_limit=2,
    )


def _dataset_row(
    scenario: Scenario, spec: EnvSpec, difficulty_success_rate: float
) -> dict[str, Any]:
    return {
        "prompt": [
            {
                "role": "system",
                "content": build_system_prompt(
                    conversational=True, env_introduction=spec.introduction()
                ),
            },
            {"role": "user", "content": scenario.task},
        ],
        "task_id": scenario.task_id,
        # TRL reads this control field only for a dict environment_factory. It is
        # harmless on the async path and keeps both trainers on identical rows.
        "environment": scenario.env_id,
        "difficulty_success_rate": difficulty_success_rate,
    }


def _trainer_rollout_kwargs(
    rollout_path: str,
    *,
    reward_func: Any,
    rollout_func: Any,
    environment_factories: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct exactly one of TRL's mutually-exclusive rollout boundaries."""
    if rollout_path == "async":
        return {
            "reward_funcs": reward_func,
            "tools": None,
            "rollout_func": rollout_func,
            "environment_factory": None,
        }
    if rollout_path == "factory_oracle":
        if not environment_factories:
            raise GrpoError("factory_oracle resolved no environment factories")
        return {
            # FactoryEnvBase.get_reward owns the unmodified verifier reward.
            "reward_funcs": None,
            "tools": None,
            "rollout_func": None,
            "environment_factory": dict(environment_factories),
        }
    raise GrpoError(f"unknown rollout path: {rollout_path!r}")


def _make_resolver(
    scenarios: Mapping[str, Scenario], env_specs: Mapping[str, EnvSpec], *, num_generations: int
) -> Any:
    by_task_text: dict[str, Scenario] = {}
    for scenario in scenarios.values():
        if scenario.task in by_task_text:
            raise GrpoError("two scenarios have identical task text; prompt binding is ambiguous")
        by_task_text[scenario.task] = scenario

    def resolve(prompts: Prompts) -> list[ScenarioBinding]:
        bindings: list[ScenarioBinding] = []
        for position, prompt in enumerate(prompts):
            user = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(prompt)
                    if message.get("role") == "user"
                ),
                "",
            )
            scenario = by_task_text.get(user)
            if scenario is None:
                raise GrpoError("TRL prompt does not map to a known scenario")
            spec = env_specs[scenario.env_id]
            bindings.append(
                ScenarioBinding(
                    scenario=scenario,
                    group_index=position // num_generations,
                    tool_schemas=tuple(spec.tools),
                    env_introduction=spec.introduction(),
                )
            )
        return bindings

    return resolve


def _load_catalog(config: GrpoConfig) -> tuple[dict[str, EnvSpec], tuple[Scenario, ...]]:
    scenario_set = build_scenario_set(
        config.env.vendored_rl_scenarios,
        env_split_manifest=config.env.env_split_manifest,
        sha256=config.env.vendored_rl_scenarios_sha256,
    )
    specs = load_env_specs(
        config.env.vendored_env_metadata, sha256=config.env.vendored_env_metadata_sha256
    )
    missing = sorted({scenario.env_id for scenario in scenario_set.scenarios} - specs.keys())
    if missing:
        raise GrpoError(f"scenario environments missing from metadata: {missing[:5]}")
    return specs, scenario_set.scenarios


def _assert_prefix_caching(trainer: Any) -> None:
    generation = getattr(trainer, "vllm_generation", None)
    engine = getattr(getattr(generation, "llm", None), "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    enabled = getattr(cache_config, "enable_prefix_caching", None)
    if enabled is not True:
        raise GrpoError(
            "colocated vLLM prefix caching is not observably enabled; "
            "refusing a changed engine default"
        )


def build_grpo_trainer(
    config: GrpoConfig,
    *,
    resume: bool = False,
    require_difficulty: bool = True,
    use_vllm: bool = True,
    tracker: Tracker | None = None,
    store: CheckpointStore | None = None,
) -> GrpoAssembly:
    """Assemble the production trainer without starting optimization."""
    from datasets import Dataset
    from trl import GRPOTrainer  # type: ignore[attr-defined]

    class CurriculumGRPOTrainer(CurriculumGRPOTrainerMixin, GRPOTrainer):
        pass

    specs, all_scenarios = _load_catalog(config)
    heldout = select_heldout_scenarios(
        all_scenarios,
        env_count=config.curriculum.heldout_env_count,
        per_env=config.curriculum.heldout_scenarios_per_env,
    )
    heldout_ids = {scenario.task_id for scenario in heldout}
    candidates = [scenario for scenario in all_scenarios if scenario.task_id not in heldout_ids]

    profile: DifficultyProfile | None = None
    if config.curriculum.enabled and require_difficulty:
        profile = read_profile(
            config.curriculum.difficulty_profile_path,
            model_id=config.model_id,
            model_revision=config.model_revision,
        )
        candidates = [scenario for scenario in candidates if scenario.task_id in profile.by_task]
    if not candidates:
        raise GrpoError("no training scenarios remain after held-out and curriculum selection")
    order = weighted_scenario_order(
        [scenario.task_id for scenario in candidates],
        profile,
        seed=config.training.seed,
        band_weight=config.curriculum.band_weight,
        always_zero_weight=config.curriculum.always_zero_weight,
        always_one_weight=config.curriculum.always_one_weight,
    )
    if not order:
        raise GrpoError("curriculum weights excluded every training scenario")
    groups_per_generation = config.profile.generation_batch_size // config.profile.num_generations
    usable_scenarios = (len(order) // groups_per_generation) * groups_per_generation
    if usable_scenarios == 0:
        raise GrpoError(
            f"curriculum has {len(order)} scenarios but one generation batch needs "
            f"{groups_per_generation} distinct scenarios"
        )
    # TRL drops a final incomplete generation chunk. Trim it once here so the
    # persisted modulo cursor describes the sampler's real cycle exactly.
    order = order[:usable_scenarios]
    by_id = {scenario.task_id: scenario for scenario in all_scenarios}
    train_scenarios = [by_id[task_id] for task_id in order]
    rates = profile.by_task if profile is not None else {}
    train_dataset = Dataset.from_list(
        [
            _dataset_row(
                scenario,
                specs[scenario.env_id],
                rates[scenario.task_id].success_rate if scenario.task_id in rates else 0.5,
            )
            for scenario in train_scenarios
        ]
    )
    eval_dataset = Dataset.from_list(
        [_dataset_row(scenario, specs[scenario.env_id], 0.5) for scenario in heldout]
    )

    tokenizer = assert_text_only_processing_class(
        load_tokenizer(config.model_id, revision=config.model_revision)
    )
    attn = resolve_attn_implementation(config.optimization.attn_implementation)
    liger = resolve_liger(config.optimization.liger_fused_linear_cross_entropy)
    checkpoint_store = store or CheckpointStore(
        config.tracking.hub_repo_id, Path(config.output_dir) / "adapter"
    )
    resume_from, resume_state = _resolve_resume(checkpoint_store, resume=resume)
    run = tracker or Tracker(
        project=config.tracking.wandb_project,
        entity=config.tracking.wandb_entity,
        run_name=config.tracking.run_name,
        config=config.model_dump(mode="json"),
        resume_run_id=resume_state.wandb_run_id if resume_state else None,
    )
    pool: WorkerPool | None = None
    dispatcher: PoolDispatcher | None = None
    try:
        run.start()
        args = _grpo_args(
            config,
            attn=attn,
            use_liger=liger.enabled,
            report_to=["wandb"] if run.enabled else [],
            use_vllm=use_vllm,
        )
        start_cursor = resume_state.sampler_cursor if resume_state else 0
        cursor = CurriculumCursor(
            start=start_cursor,
            dataset_size=len(train_dataset),
            groups_per_generation=args.generation_batch_size // config.profile.num_generations,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            steps_per_generation=args.steps_per_generation,
        )
        capacity = config.profile.env_worker_count * config.profile.env_episodes_per_worker
        if capacity < config.profile.generation_batch_size:
            raise GrpoError(
                f"environment pool capacity {capacity} < generation batch "
                f"{config.profile.generation_batch_size}"
            )
        pool = WorkerPool(
            metadata_path=config.env.vendored_env_metadata,
            metadata_sha256=config.env.vendored_env_metadata_sha256,
            scenario_path=config.env.vendored_rl_scenarios,
            scenario_sha256=config.env.vendored_rl_scenarios_sha256,
            worker_count=config.profile.env_worker_count,
            episodes_per_worker=config.profile.env_episodes_per_worker,
            call_timeout_s=config.env.step_timeout_s,
            create_timeout_s=config.env.create_timeout_s,
            step_timeout_s=config.env.step_timeout_s,
            verify_timeout_s=config.env.verify_timeout_s,
        )
        pool.start()
        dispatcher = PoolDispatcher(pool, max_workers=config.profile.env_worker_count)
        rollout_func = None
        environment_factories: Mapping[str, Any] = {}
        if config.rollout_path == "async":
            rollout_func = make_rollout_func(
                resolve_bindings=_make_resolver(
                    by_id, specs, num_generations=config.profile.num_generations
                ),
                config=scheduler_config_for(config),
                dispatcher=dispatcher,
                tokenizer=tokenizer,
            )
        else:
            environment_factories = make_environment_factories(
                env_specs=specs,
                scenarios=[*train_scenarios, *heldout],
                pool=pool,
            )
        reward = functools.partial(
            verifier_reward,
            num_generations=config.profile.num_generations,
            trajectory_sample_limit=config.curriculum.trajectory_samples_per_log,
        )
        rollout_kwargs = _trainer_rollout_kwargs(
            config.rollout_path,
            reward_func=reward,
            rollout_func=rollout_func,
            environment_factories=environment_factories,
        )
        trainer = CurriculumGRPOTrainer(
            model=config.model_id,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=_lora_config(config),
            **rollout_kwargs,
        )
        trainer.curriculum_cursor = cursor
        if use_vllm:
            _assert_prefix_caching(trainer)
        adapters = cast_adapters(trainer.model, config.lora.adapter_dtype)
        toggles = (attn, liger, adapters)
        trainer.add_callback(GrpoCheckpointCallback(checkpoint_store, run, cursor))
        trainer.add_callback(
            cast(Any, LogpDifferenceStopCallback(config.logp_difference_stop_threshold))
        )
        trainer.add_callback(
            GroupVarianceStopCallback(
                after_steps=config.curriculum.zero_variance_stop_after_steps,
                multiplier=config.curriculum.zero_variance_stop_multiplier,
                margin=config.curriculum.zero_variance_stop_margin,
            )
        )
        run.config.update(ledger(toggles))
    except Exception:
        _cleanup_failed_assembly(dispatcher, pool, run)
        raise
    assert dispatcher is not None
    assert pool is not None
    return GrpoAssembly(
        trainer=trainer,
        tracker=run,
        store=checkpoint_store,
        dispatcher=dispatcher,
        pool=pool,
        toggles=toggles,
        resume_from=resume_from,
        cursor=cursor,
        train_task_ids=tuple(order),
        eval_task_ids=tuple(scenario.task_id for scenario in heldout),
    )


def run_train_grpo(config: GrpoConfig, *, resume: bool = False) -> int:
    assembled = build_grpo_trainer(config, resume=resume)
    try:
        print(format_ledger(list(assembled.toggles)))
        print(
            f"train {len(assembled.train_task_ids)} curriculum scenarios / "
            f"eval {len(assembled.eval_task_ids)} held-out scenarios"
        )
        assembled.trainer.train(resume_from_checkpoint=assembled.resume_from)
        assembled.trainer.save_model(config.output_dir)
        return 0
    finally:
        assembled.shutdown()


def run_profile_difficulty(config: GrpoConfig) -> int:
    """Run the SFT policy repeatedly and persist verifier-derived difficulty."""
    if config.rollout_path != "async":
        raise GrpoError("profile-difficulty requires rollout_path=async")
    assembled = build_grpo_trainer(config, require_difficulty=False)
    try:
        trainer = assembled.trainer
        sync = getattr(getattr(trainer, "vllm_generation", None), "sync_weights", None)
        if callable(sync):
            sync()
        dataset = trainer.train_dataset
        sample_count = min(config.curriculum.profile_scenario_sample, len(dataset))
        task_ids = tuple(assembled.train_task_ids[:sample_count])
        rows_by_id = {str(row["task_id"]): row for row in dataset}
        rewards: dict[str, list[float]] = {task_id: [] for task_id in task_ids}
        batch_width = config.profile.generation_batch_size
        prompts: list[Any] = []
        owners: list[str] = []
        for task_id in task_ids:
            for _ in range(config.curriculum.profile_rollouts):
                prompts.append(rows_by_id[task_id]["prompt"])
                owners.append(task_id)
        for offset in range(0, len(prompts), batch_width):
            batch_prompts = prompts[offset : offset + batch_width]
            batch_owners = owners[offset : offset + batch_width]
            output = trainer.rollout_func(batch_prompts, trainer)
            for task_id, value in zip(batch_owners, output["rollout_reward"], strict=True):
                rewards[task_id].append(float(value))
        profile = profile_rewards(
            rewards,
            model_id=config.model_id,
            model_revision=config.model_revision,
            seed=config.training.seed,
        )
        path = write_profile(profile, config.curriculum.difficulty_profile_path)
        print(json.dumps(profile.to_dict()["counts"], sort_keys=True))
        print(f"wrote {path}")
        return 0
    finally:
        assembled.shutdown()
