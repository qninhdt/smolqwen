"""Agentic GRPO assembly over the Phase 6 asynchronous rollout function."""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Iterator, Mapping, Sized
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from smolqwen.artifacts import CheckpointStore, ResumeState
from smolqwen.config_models import EvalConfig, GrpoConfig
from smolqwen.console import console, logger, phase
from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import EnvSpec, load_env_specs
from smolqwen.env.scenarios import Scenario, load_scenarios
from smolqwen.inference.profiles import turn_engine_config
from smolqwen.prompts import build_system_prompt
from smolqwen.rollout.factory_env import make_environment_factories
from smolqwen.rollout.generation import VllmColocateBackend
from smolqwen.rollout.metrics import LogpDifferenceStopCallback
from smolqwen.rollout.rollout_func import Prompts, make_rollout_func
from smolqwen.rollout.scheduler import PoolDispatcher, ScenarioBinding
from smolqwen.tokenizer import assert_text_only_processing_class, load_tokenizer
from smolqwen.tracking import Tracker
from smolqwen.training.optim import (
    TEXT_ONLY_PEFT_EXCLUDE_MODULES,
    Toggle,
    cast_adapters,
    format_ledger,
    ledger,
    resolve_attn_implementation,
    resolve_liger,
    resolve_precision,
)
from smolqwen.training.reward import verifier_reward

LOG = logger(__name__)


class GrpoError(RuntimeError):
    """Raised when a GRPO run cannot preserve its experiment contracts."""


@dataclass(frozen=True)
class ScenarioCursor:
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
    """TRL's structured repeat sampler, rotated to a persisted scenario cursor."""

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


class ScenarioGRPOTrainerMixin:
    """Sampler override mixed into TRL's trainer."""

    scenario_cursor: ScenarioCursor
    _pending_sampling_logprobs: Any = None

    def _generate(self, prompts: Any) -> Any:
        """Use vLLM's sampled logprobs as old-policy logprobs when requested."""
        output = cast(Any, super())._generate(prompts)
        if getattr(self, "use_vllm", False) and not getattr(
            self, "vllm_importance_sampling_correction", True
        ):
            logprobs = output[4]
            if logprobs is not None:
                self._pending_sampling_logprobs = logprobs
        return output

    def _get_per_token_logps_and_entropies(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        logits_to_keep: int,
        batch_size: int | None = None,
        compute_entropy: bool = False,
        compute_aux_loss: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Avoid a dense old-policy forward when sampler logprobs are available."""
        pending = self._pending_sampling_logprobs
        if pending is not None and not compute_entropy:
            self._pending_sampling_logprobs = None
            rows = [
                torch.tensor(
                    [
                        0.0 if value is None or not math.isfinite(float(value)) else float(value)
                        for value in row
                    ],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                for row in pending
            ]
            old_logprobs = torch.nn.utils.rnn.pad_sequence(
                rows, batch_first=True, padding_value=0.0
            )
            if old_logprobs.size(1) < logits_to_keep:
                old_logprobs = torch.nn.functional.pad(
                    old_logprobs, (0, logits_to_keep - old_logprobs.size(1))
                )
            return old_logprobs[:, :logits_to_keep], None, None
        return cast(
            tuple[Any, Any, Any],
            cast(Any, super())._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                **kwargs,
            ),
        )

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
            cursor=self.scenario_cursor.start,
        )


class GrpoCheckpointCallback(TrainerCallback):
    """Push adapter, optimizer checkpoint, run id, and sampler cursor together."""

    def __init__(self, store: CheckpointStore, tracker: Tracker, cursor: ScenarioCursor) -> None:
        self.store = store
        self.tracker = tracker
        self.cursor = cursor

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        LOG.info("train-grpo checkpoint step %d: saving adapter and trainer state", step)
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
    cursor: ScenarioCursor
    train_task_ids: tuple[str, ...]

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
        # Keep the all-linear contract on the language branch only: this run is
        # text-only and the vision tower is not part of the vLLM LoRA surface.
        exclude_modules=(
            TEXT_ONLY_PEFT_EXCLUDE_MODULES
            if any(marker in config.model_id.casefold() for marker in ("qwen3.5", "qwen3_5"))
            else None
        ),
        task_type="CAUSAL_LM",
        bias="none",
    )


def _grpo_args(
    config: GrpoConfig,
    *,
    attn: Toggle,
    precision: Toggle,
    use_liger: bool,
    report_to: list[str],
    use_vllm: bool,
) -> Any:
    from trl import GRPOConfig as TrlGrpoConfig  # type: ignore[attr-defined]

    profile = config.profile
    training = config.training
    generation_batch_size = profile.generation_batch_size
    if generation_batch_size % profile.num_generations:
        raise GrpoError("generation_batch_size must be divisible by num_generations")
    # TRL evaluates a whole prompt group at once, so it requires the global eval
    # batch to be a multiple of `num_generations`. `micro_batch` is an SFT-side
    # sizing field and is 1 or 2 in every profile, which is never a multiple of the
    # group size -- so eval batch size is derived from the group size instead of
    # reusing `micro_batch`. MEASURED: with `micro_batch` here, `train-grpo` raised
    # `ValueError: The global eval batch size (1 * 1) must be divisible by the
    # number of generations used for evaluation (2)` at config construction on
    # every profile, before any weight loaded.
    eval_batch_size = profile.num_generations
    return TrlGrpoConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=profile.micro_batch,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=profile.grad_accum,
        generation_batch_size=generation_batch_size,
        num_generations=profile.num_generations,
        num_iterations=config.num_iterations,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        warmup_steps=training.warmup_ratio,
        weight_decay=training.weight_decay,
        lr_scheduler_type=training.lr_scheduler_type,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        # GRPO's dev benchmark is driven by `BenchEvalCallback`, which uses
        # the shared turn engine and records `grpo/bench_*`. Native Trainer eval
        # instead runs the GRPO/Liger loss over `eval_dataset`, which is a separate
        # loss path, so keep that duplicate path disabled.
        eval_strategy="no",
        save_strategy="steps",
        seed=training.seed,
        bf16=precision.name == "bfloat16",
        fp16=precision.name == "float16",
        gradient_checkpointing=config.optimization.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=use_liger,
        model_init_kwargs={
            "dtype": precision.name,
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
        vllm_importance_sampling_correction=config.vllm_importance_sampling_correction,
        log_completions=False,
        num_completions_to_print=config.trajectory_samples_per_log,
        remove_unused_columns=False,
        ignore_data_skip=True,
        report_to=report_to,
        run_name=config.tracking.run_name,
        push_to_hub=False,
        save_total_limit=2,
    )


def _sync_weights(trainer: Any) -> None:
    """Push the trainer's current weights into the colocated engine.

    Required, not optional. At a callback boundary the optimizer step has already
    been applied, and generation runs once per accumulation window, so the engine
    can hold weights up to `grad_accum` optimizer steps old. That drift is invisible
    in a score and would look exactly like a parity bug when `bench_*` is later
    compared against `evaluate` at "the same revision".

    Asserted rather than skipped when absent, matching `_assert_prefix_caching`: a
    silently unsynced eval reports a number for the wrong weights.
    """
    generation = getattr(trainer, "vllm_generation", None)
    sync = getattr(generation, "sync_weights", None)
    if not callable(sync):
        raise GrpoError(
            "in-training benchmark eval requires the colocated vLLM engine's "
            "sync_weights; without it the scored weights are up to grad_accum "
            "optimizer steps stale and the number is unattributable"
        )
    sync()


def _weight_version(trainer: Any, syncs: list[int]) -> str:
    """Global step plus a sync counter: what weights a `bench_*` row measured.

    The step alone is not enough -- two evals at the same step (a save boundary that
    coincides with an interval) would be indistinguishable, and the whole point is
    that the comparison against `evaluate` is falsifiable.
    """
    step = int(getattr(getattr(trainer, "state", None), "global_step", 0))
    syncs.append(step)
    return f"step-{step}.sync-{len(syncs)}"


def bench_eval_config(config: GrpoConfig) -> EvalConfig:
    """The eval stage's config, resized by this GRPO run's own engine.

    Resolved from the same YAML `smolqwen evaluate` reads, so the adapter's options
    and decoding are identical -- that is what makes one number mean one thing in
    both places. Its `profile` subtree is then replaced, for two reasons:

    - `resolve("eval")` takes no `--profile`, so it would otherwise carry
      `ProfileConfig` defaults. A bench eval at the default generation width while
      the trainer was sized by `--profile t4` is two sets of numbers for one card.
    - The context bound is `vllm_max_model_len`, **not** `profile.max_seq_length`.
      The colocated engine TRL built is sized by the former (`_grpo_args`), while
      `EvalProfile.max_model_len` reads the latter. Left unequal, the turn engine
      admits a prefix wider than the engine can accept and vLLM raises
      `The decoder prompt (length N) is longer than the maximum model length`,
      turning every boundary into `bench_failed`.

    The SFT path has no in-training benchmark callback; only GRPO owns this
    checkpoint-independent boundary.
    """
    from smolqwen.config import resolve

    resolved = resolve("eval")
    if not isinstance(resolved, EvalConfig):
        raise GrpoError(f"expected EvalConfig from the eval stage, got {type(resolved).__name__}")
    profile = config.profile.model_copy(update={"max_seq_length": config.vllm_max_model_len})
    # The boundary scores the same distribution the rollout trains on: a
    # thinking-rendered dev set would measure a capability the rollout never
    # exercises (and the reverse), silently bending the checkpoint-selection curve.
    return resolved.model_copy(
        update={"profile": profile, "enable_thinking": config.enable_thinking}
    )


def build_bench_eval_callback(
    config: GrpoConfig,
    trainer: Any,
    tokenizer: Any,
    *,
    tracker: Tracker | None = None,
) -> Any | None:
    """The in-training dev-eval callback, or None when it is disabled.

    The benchmark is named in `bench_eval.adapter` and resolved against the eval
    stage's config.
    """
    if not config.bench_eval.enabled:
        return None
    from smolqwen.training.bench_eval import BenchEvalCallback, BenchEvalRunner

    syncs: list[int] = []
    runner = BenchEvalRunner(
        eval_config=bench_eval_config(config),
        bench_config=config.bench_eval,
        engine_source=lambda: VllmColocateBackend(trainer),
        tokenizer_source=lambda: tokenizer,
        metric_prefix="grpo",
        sink=(lambda payload: trainer.log(dict(payload))) if hasattr(trainer, "log") else None,
        weight_version=lambda: _weight_version(trainer, syncs),
        artifact_dir=config.output_dir,
        outcome_path=lambda step: Path(config.output_dir) / "bench-eval" / f"step-{step}.json",
    )
    return BenchEvalCallback(runner, before_each=lambda _step: _sync_weights(trainer))


def _dataset_row(scenario: Scenario, spec: EnvSpec) -> dict[str, Any]:
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
    scenarios = load_scenarios(
        config.env.vendored_rl_scenarios,
        sha256=config.env.vendored_rl_scenarios_sha256,
    )
    specs = load_env_specs(
        config.env.vendored_env_metadata, sha256=config.env.vendored_env_metadata_sha256
    )
    missing = sorted({scenario.env_id for scenario in scenarios} - specs.keys())
    if missing:
        raise GrpoError(f"scenario environments missing from metadata: {missing[:5]}")
    return specs, tuple(scenarios)


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


@contextmanager
def _force_trl_prefix_caching() -> Iterator[None]:
    """Pass the required cache flag through TRL versions that omit it.

    TRL 1.12 constructs its colocated ``vllm.LLM`` without an
    ``enable_prefix_caching`` argument. vLLM therefore resolves the flag to false
    for hybrid models such as Qwen3.5, even though this project's GRPO contract
    requires prefix caching. Keep the compatibility bridge limited to the
    synchronous trainer construction and restore TRL's module global immediately
    afterwards.
    """
    try:
        from trl.generation import vllm_generation
    except ImportError as exc:
        raise GrpoError(
            "colocated GRPO requires TRL's vLLM generation module to wire enable_prefix_caching"
        ) from exc

    original_llm = getattr(vllm_generation, "LLM", None)
    if original_llm is None:
        raise GrpoError("TRL's vLLM generation module exposes no LLM constructor")

    def prefix_cached_llm(*args: Any, **kwargs: Any) -> Any:
        kwargs["enable_prefix_caching"] = True
        # This project sends text only. Keep the Qwen3.5 wrapper contract while
        # avoiding a resident vision encoder in the colocated training engine.
        kwargs["language_model_only"] = True
        return original_llm(*args, **kwargs)

    patch_target: Any = vllm_generation
    patch_target.LLM = prefix_cached_llm
    try:
        yield
    finally:
        patch_target.LLM = original_llm


def build_grpo_trainer(
    config: GrpoConfig,
    *,
    resume: bool = False,
    use_vllm: bool = True,
    tracker: Tracker | None = None,
    store: CheckpointStore | None = None,
) -> GrpoAssembly:
    """Assemble the production trainer without starting optimization."""
    from datasets import Dataset
    from trl import GRPOTrainer  # type: ignore[attr-defined]

    class ScenarioGRPOTrainer(ScenarioGRPOTrainerMixin, GRPOTrainer):
        pass

    with phase("train-grpo: load environment catalog and scenarios"):
        specs, all_scenarios = _load_catalog(config)
    order = [scenario.task_id for scenario in all_scenarios]
    random.Random(config.training.seed).shuffle(order)
    groups_per_generation = config.profile.generation_batch_size // config.profile.num_generations
    usable_scenarios = (len(order) // groups_per_generation) * groups_per_generation
    if usable_scenarios == 0:
        raise GrpoError(
            f"{len(order)} scenarios cannot fill one generation batch of "
            f"{groups_per_generation} distinct scenarios"
        )
    # TRL drops a final incomplete generation chunk. Trim it once here so the
    # persisted modulo cursor describes the sampler's real cycle exactly.
    order = order[:usable_scenarios]
    by_id = {scenario.task_id: scenario for scenario in all_scenarios}
    train_scenarios = [by_id[task_id] for task_id in order]
    train_dataset = Dataset.from_list(
        [_dataset_row(scenario, specs[scenario.env_id]) for scenario in train_scenarios]
    )

    with phase("train-grpo: load tokenizer"):
        tokenizer = assert_text_only_processing_class(
            load_tokenizer(config.model_id, revision=config.model_revision)
        )
    attn = resolve_attn_implementation(config.optimization.attn_implementation)
    precision = resolve_precision(config.optimization.bf16)
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
            precision=precision,
            use_liger=liger.enabled,
            report_to=["wandb"] if run.enabled else [],
            use_vllm=use_vllm,
        )
        start_cursor = resume_state.sampler_cursor if resume_state else 0
        cursor = ScenarioCursor(
            start=start_cursor,
            dataset_size=len(train_dataset),
            groups_per_generation=args.generation_batch_size // config.profile.num_generations,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            steps_per_generation=args.steps_per_generation * int(args.num_iterations),
        )
        capacity = config.profile.env_worker_count * config.profile.env_episodes_per_worker
        if capacity < config.profile.generation_batch_size:
            raise GrpoError(
                f"environment pool capacity {capacity} < generation batch "
                f"{config.profile.generation_batch_size}"
            )
        with phase("train-grpo: start environment worker pool"):
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
                config=turn_engine_config(config),
                dispatcher=dispatcher,
                tokenizer=tokenizer,
            )
        else:
            environment_factories = make_environment_factories(
                env_specs=specs,
                scenarios=train_scenarios,
                pool=pool,
            )
        reward = functools.partial(
            verifier_reward,
            trajectory_sample_limit=config.trajectory_samples_per_log,
        )
        rollout_kwargs = _trainer_rollout_kwargs(
            config.rollout_path,
            reward_func=reward,
            rollout_func=rollout_func,
            environment_factories=environment_factories,
        )
        with phase("train-grpo: build trainer and colocated vLLM"):
            with _force_trl_prefix_caching() if use_vllm else nullcontext():
                trainer = ScenarioGRPOTrainer(
                    model=config.model_id,
                    args=args,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                    peft_config=_lora_config(config),
                    **rollout_kwargs,
                )
        if liger.enabled:
            # ponytail: keep Liger's fused loss, skip only its torch.compile guard
            # path; torch 2.11/Liger 0.8.2 raises on dynamic sequence shapes.
            trainer.liger_loss.compiled = False
        trainer.scenario_cursor = cursor
        if use_vllm:
            _assert_prefix_caching(trainer)
        # FP16 runs must keep trainable adapters in FP32: GradScaler rejects FP16
        # gradients. `cast_adapters` owns that rule; give it the effective dtype
        # rather than the configured one so a downgraded card is handled.
        adapter_dtype = precision.name if precision.name == "float16" else config.lora.adapter_dtype
        adapters = cast_adapters(trainer.model, adapter_dtype)
        dtype = Toggle(
            "training_dtype",
            precision.enabled,
            f"effective model dtype: {precision.name}; {precision.detail}",
        )
        toggles = (attn, dtype, liger, adapters)
        trainer.add_callback(GrpoCheckpointCallback(checkpoint_store, run, cursor))
        trainer.add_callback(
            cast(Any, LogpDifferenceStopCallback(config.logp_difference_stop_threshold))
        )
        bench_eval = build_bench_eval_callback(config, trainer, tokenizer, tracker=run)
        if bench_eval is not None:
            trainer.add_callback(cast(Any, bench_eval))
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
    )


def run_train_grpo(config: GrpoConfig, *, resume: bool = False) -> int:
    with phase("train-grpo: assemble run"):
        assembled = build_grpo_trainer(config, resume=resume)
    try:
        console().print(format_ledger(list(assembled.toggles)))
        LOG.info(
            "vLLM: %.0f%% VRAM budget, context=%d, concurrency=%d, "
            "per_turn_sleep=%s, post_rollout_sleep=%s, is_correction=%s",
            config.profile.vllm_kv_fraction * 100,
            config.vllm_max_model_len,
            config.profile.generation_concurrency,
            config.vllm_enable_sleep_mode,
            config.rollout_path == "async",
            config.vllm_importance_sampling_correction,
        )
        LOG.info(
            "train %d scenarios",
            len(assembled.train_task_ids),
        )
        with phase("train-grpo: optimizer training"):
            assembled.trainer.train(resume_from_checkpoint=assembled.resume_from)
        with phase("train-grpo: save final checkpoint"):
            assembled.trainer.save_model(config.output_dir)
        return 0
    finally:
        assembled.shutdown()
