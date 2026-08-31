"""Assemble full-trajectory, token-budget, padding-free LoRA SFT.

The shards already carry exact full-render ``input_ids`` and ``labels``, so this
module deliberately does **not** let TRL re-tokenize or re-derive assistant
ownership. ``skip_prepare_dataset=True`` keeps those persisted labels authoritative.

Two more decisions worth stating because they are not defaults:

- **W&B is driven through the Phase 1 `Tracker`, not `report_to="wandb"`.** The
  trainer's own integration starts a fresh run, which forks a resumed run into a
  second curve. `Tracker` resumes with `resume="must"` off the persisted run id.
- **The adapter is pushed on every save, and the resume cursor is pushed with
  it.** Colab reclaims VMs without warning; an adapter that only exists locally
  does not exist. Restoring weights without the step/run-id cursor would replay
  the schedule and fork the curve.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Imported at module scope, unlike inside the CLI handlers: this module is itself
# only imported from inside `_cmd_train_sft`, so `--dry-run` still never pays for
# transformers.
from transformers import TrainerCallback

from smolqwen.artifacts import CheckpointStore, ResumeState
from smolqwen.config_models import SftConfig
from smolqwen.tokenizer import assert_text_only_processing_class, load_tokenizer
from smolqwen.tracking import Tracker
from smolqwen.training.collate import (
    IGNORE_INDEX,
    CollateError,
    padding_free_collator,
    record_to_sequence,
)
from smolqwen.training.optim import (
    Toggle,
    apply_regional_compile,
    cast_adapters,
    format_ledger,
    ledger,
    resolve_attn_implementation,
    resolve_liger,
)
from smolqwen.training.token_batching import TokenBudgetBatchSampler


class SftError(RuntimeError):
    """Raised when the run cannot be assembled from the resolved config."""


def iter_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream one shard line by line, never holding the whole file.

    The real `train.jsonl` is ~2 GB of JSON; parsed into Python lists it is
    several times that, which is more than a Colab VM has. So validation streams
    and the trainer reads through Arrow (see `load_shard_dataset`).
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SftError(f"{path}:{line_number}: not valid JSON: {exc}") from exc
            yield record


@dataclass(frozen=True)
class ShardStats:
    """What a validation pass measured, so the run can report it before step 1."""

    path: str
    samples: int
    total_tokens: int
    supervised_tokens: int


def validate_records(records: Iterable[dict[str, Any]], *, label: str) -> ShardStats:
    """Run every record through the mask builder before a step is taken.

    `record_to_sequence` is the same function the collator calls, so a shard whose
    mask disagrees with its completion fails here -- at load, naming the record --
    rather than mid-epoch or, worse, not at all.
    """
    samples = 0
    tokens = 0
    supervised = 0
    for index, record in enumerate(records):
        try:
            input_ids, labels = record_to_sequence(record)
        except CollateError as exc:
            raise SftError(f"{label}[{index}]: {exc}") from exc
        samples += 1
        tokens += len(input_ids)
        supervised += sum(1 for value in labels if value != IGNORE_INDEX)
    return ShardStats(
        path=label, samples=samples, total_tokens=tokens, supervised_tokens=supervised
    )


def validate_shard(path: Path | str, *, label: str) -> ShardStats:
    return validate_records(iter_records(path), label=label)


# Only these reach a training step. The metadata columns (`trajectory_id`,
# `env_id`, ...) are dropped at load rather than by `remove_unused_columns`,
# because the collator is handed raw feature dicts and a column it ignores is
# only weight in the Arrow table.
FEATURE_COLUMNS = (
    "schema_version",
    "semantics",
    "trajectory_uid",
    "input_ids",
    "labels",
    "seq_length",
    "supervised_tokens",
)

# Length-grouped sampling reads this column. It is materialized at load rather
# than left for the sampler to probe: without it `LengthGroupedSampler` tokenizes
# the whole shard to measure it, which on a 2 GB shard is minutes of startup for a
# number the records already determine.
LENGTH_COLUMN = "length"


def load_shard_dataset(path: Path | str) -> Any:
    """Memory-map one shard through Arrow, keeping only what a step needs.

    `Dataset.from_dict` over parsed Python lists would need the whole 2 GB shard
    resident; `load_dataset("json", ...)` writes an Arrow cache once and then
    memory-maps it, so the trainer's resident set does not scale with the shard.
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train")
    keep = (*FEATURE_COLUMNS, LENGTH_COLUMN)
    extra = [column for column in dataset.column_names if column not in keep]
    dataset = dataset.remove_columns(extra) if extra else dataset
    if LENGTH_COLUMN not in dataset.column_names:
        dataset = dataset.add_column(
            LENGTH_COLUMN,
            [len(row) for row in dataset["input_ids"]],
        )
    return dataset


@dataclass(frozen=True)
class Shards:
    """Both splits, each already validated against the mask contract."""

    train: Any
    eval: Any | None
    train_stats: ShardStats
    eval_stats: ShardStats | None


def load_shards(dataset_dir: Path | str) -> Shards:
    """Validate `train.jsonl` / `val.jsonl`, then hand back Arrow-backed datasets."""
    directory = Path(dataset_dir)
    train_path = directory / "train.jsonl"
    val_path = directory / "val.jsonl"
    if not train_path.is_file():
        raise SftError(f"{train_path} missing -- run `smolqwen prepare-sft` first")

    train_stats = validate_shard(train_path, label="train")
    if not train_stats.samples:
        raise SftError(f"{train_path} is empty")
    eval_stats = validate_shard(val_path, label="val") if val_path.is_file() else None

    return Shards(
        train=load_shard_dataset(train_path),
        eval=load_shard_dataset(val_path) if eval_stats and eval_stats.samples else None,
        train_stats=train_stats,
        eval_stats=eval_stats,
    )


@dataclass(frozen=True)
class Assembled:
    """What `build_trainer` produced, so callers can inspect it without training."""

    trainer: Any
    toggles: tuple[Toggle, ...]
    train_stats: ShardStats
    eval_stats: ShardStats | None
    resume_from: str | None

    @property
    def train_size(self) -> int:
        return self.train_stats.samples

    @property
    def eval_size(self) -> int:
        return self.eval_stats.samples if self.eval_stats else 0


def _lora_config(config: SftConfig) -> Any:
    from peft import LoraConfig as PeftLoraConfig

    lora = config.lora
    target = lora.target_modules
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=target if isinstance(target, str) else list(target),
        task_type="CAUSAL_LM",
        bias="none",
    )


def _sft_config(config: SftConfig, *, attn: Toggle, use_liger: bool, report_to: list[str]) -> Any:
    """Build `SFTConfig`, with TRL's own preprocessing switched off.

    `max_length=None` is deliberate: truncation happens in the Phase 3 collator
    against the profile cap, and TRL's truncation path is unreachable anyway once
    dataset preparation is skipped. Leaving the default 1024 in place would be a
    silent claim that samples are 1024 tokens long.
    """
    from trl import SFTConfig as TrlSftConfig  # type: ignore[attr-defined]

    profile = config.profile
    training = config.training
    optimization = config.optimization
    return TrlSftConfig(
        output_dir=config.output_dir,
        # A custom batch_sampler owns row count; TrainingArguments still requires
        # positive compatibility values here.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=profile.grad_accum,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        # transformers 5.x dropped `warmup_ratio`; `warmup_steps` takes a float in
        # [0, 1) as a ratio of total steps, which is the same quantity.
        warmup_steps=training.warmup_ratio,
        weight_decay=training.weight_decay,
        lr_scheduler_type=training.lr_scheduler_type,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        eval_steps=training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=training.seed,
        bf16=optimization.bf16,
        gradient_checkpointing=optimization.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=use_liger,
        model_init_kwargs={
            "dtype": "bfloat16" if optimization.bf16 else "float32",
            "attn_implementation": attn.name,
        },
        # Ours: the Phase 2 mask is authoritative, so nothing re-derives it.
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
        assistant_only_loss=False,
        packing=False,
        # Project-owned flattening emits Qwen3.5 GDN/conv boundaries that native
        # TRL padding-free does not provide.
        padding_free=False,
        max_length=None,
        remove_unused_columns=False,
        report_to=report_to,
        push_to_hub=False,
        save_total_limit=2,
    )


def _token_budget_trainer_class() -> Any:
    """Create the TRL subclass lazily so CLI dry-runs stay lightweight."""
    from torch.utils.data import DataLoader
    from trl import SFTTrainer  # type: ignore[attr-defined]

    class TokenBudgetSFTTrainer(SFTTrainer):
        def __init__(self, *args: Any, max_tokens_per_microbatch: int, **kwargs: Any) -> None:
            self.max_tokens_per_microbatch = max_tokens_per_microbatch
            self.token_batch_sampler: TokenBudgetBatchSampler | None = None
            super().__init__(*args, **kwargs)

        def _token_dataloader(self, dataset: Any, *, training: bool) -> Any:
            sampler = TokenBudgetBatchSampler(
                [int(length) for length in dataset[LENGTH_COLUMN]],
                max_tokens=self.max_tokens_per_microbatch,
                seed=int(self.args.seed),
                shuffle=training,
            )
            if training:
                self.token_batch_sampler = sampler
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return self.accelerator.prepare(dataloader)

        def get_train_dataloader(self) -> Any:
            if self.train_dataset is None:
                raise SftError("training requires a train dataset")
            return self._token_dataloader(self.train_dataset, training=True)

        def get_eval_dataloader(self, eval_dataset: Any | None = None) -> Any:
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if dataset is None:
                raise SftError("evaluation requires an eval dataset")
            return self._token_dataloader(dataset, training=False)

    return TokenBudgetSFTTrainer


def assert_padding_free_runtime() -> None:
    """Fail closed unless all target-kernel boundary paths are available."""
    import importlib.util

    import torch

    missing = [
        package
        for package in ("flash_attn", "fla", "causal_conv1d")
        if importlib.util.find_spec(package) is None
    ]
    if not torch.cuda.is_available() or missing:
        detail = f"missing kernels: {', '.join(missing)}" if missing else "CUDA unavailable"
        raise SftError(
            "padding-free Qwen3.5 requires real CUDA FlashAttention, FLA GDN, "
            f"and causal-conv boundary kernels; {detail}"
        )


class ThroughputCallback(TrainerCallback):
    """s/Mtok and peak VRAM per step, through the Phase 1 meter.

    Seconds per million tokens rather than tokens/s: the two profiles run
    different batch sizes, so tokens/s is not comparable across them.
    """

    def __init__(self, tracker: Tracker, *, tokens_per_step: int) -> None:
        self.tracker = tracker
        self.tokens_per_step = tokens_per_step

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.tracker.meter.start_step()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.tracker.log_step(tokens=self.tokens_per_step, step=int(state.global_step))

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logs = kwargs.get("logs")
        if logs:
            self.tracker.log(dict(logs), step=int(state.global_step))


class CheckpointPushCallback(TrainerCallback):
    """Push the adapter and the resume cursor on every save.

    Both, together: restoring weights without the step and run id replays the
    schedule from zero and forks the W&B curve, and neither shows up in the loss.
    """

    def __init__(self, store: CheckpointStore, tracker: Tracker) -> None:
        self.store = store
        self.tracker = tracker

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        if not checkpoint.is_dir():
            return
        self.store.save_adapter(checkpoint)
        self.store.write_resume_state(
            ResumeState(
                revision=None,
                wandb_run_id=self.tracker.run_id,
                global_step=int(state.global_step),
            )
        )
        self.store.push(commit_message=f"adapter at step {int(state.global_step)}")


def _resolve_resume(
    store: CheckpointStore, *, resume: bool
) -> tuple[str | None, ResumeState | None]:
    """Materialise the newest pushed revision, or report there is nothing to resume.

    `latest_revision` is reachable only from here: an eval that resolved it could
    have the model swapped under it by a concurrent training push.
    """
    if not resume:
        return None, None
    state = store.read_resume_state()
    if state is not None and Path(store.local_dir).is_dir():
        return str(store.local_dir), state
    if not store.enabled:
        raise SftError(
            "--resume needs either a local resume_state.json or a configured hub_repo_id"
        )
    revision = store.latest_revision()
    if revision is None:
        raise SftError("--resume found no pushed revision to continue from")
    local = store.pull(revision)
    return str(local), store.read_resume_state()


def build_trainer(
    config: SftConfig,
    *,
    resume: bool = False,
    tracker: Tracker | None = None,
    store: CheckpointStore | None = None,
    dataset_dir: Path | str | None = None,
) -> Assembled:
    """Assemble the trainer without starting it, so a smoke test can inspect it."""
    shards = load_shards(dataset_dir or config.dataset_dir)

    tokenizer = assert_text_only_processing_class(
        load_tokenizer(config.model_id, revision=config.model_revision)
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise SftError(f"{config.model_id} tokenizer has neither a pad nor an eos token id")

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

    trainer_class = _token_budget_trainer_class()
    trainer = trainer_class(
        model=config.model_id,
        args=_sft_config(config, attn=attn, use_liger=liger.enabled, report_to=[]),
        data_collator=padding_free_collator(config.profile.max_tokens_per_microbatch),
        train_dataset=shards.train,
        eval_dataset=shards.eval,
        processing_class=tokenizer,
        peft_config=_lora_config(config),
        max_tokens_per_microbatch=config.profile.max_tokens_per_microbatch,
    )

    adapters = cast_adapters(trainer.model, config.lora.adapter_dtype)
    compiled = apply_regional_compile(
        trainer.model,
        exclude_patterns=config.optimization.compile_exclude_patterns,
        enabled=config.optimization.regional_torch_compile,
    )
    toggles = (attn, liger, adapters, compiled)

    tokens_per_step = config.profile.max_tokens_per_microbatch
    trainer.add_callback(ThroughputCallback(run, tokens_per_step=tokens_per_step))
    trainer.add_callback(CheckpointPushCallback(checkpoint_store, run))
    run.config.update(ledger(toggles))

    return Assembled(
        trainer=trainer,
        toggles=toggles,
        train_stats=shards.train_stats,
        eval_stats=shards.eval_stats,
        resume_from=resume_from,
    )


def run_train_sft(config: SftConfig, *, resume: bool = False) -> int:
    """`smolqwen train-sft`: train, evaluate once at the end, record the ledger."""
    assert_padding_free_runtime()
    assembled = build_trainer(config, resume=resume)
    trainer = assembled.trainer
    print(format_ledger(list(assembled.toggles)))
    train = assembled.train_stats
    print(
        f"train {train.samples} samples / {train.total_tokens} tokens "
        f"({train.supervised_tokens} supervised)  val {assembled.eval_size} samples"
    )

    trainer.train(resume_from_checkpoint=assembled.resume_from)
    trainer.save_model(config.output_dir)
    if assembled.eval_size:
        metrics = trainer.evaluate()
        print(
            json.dumps({key: float(value) for key, value in metrics.items() if _is_number(value)})
        )
    return 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
