"""Assemble full-trajectory, token-budget LoRA SFT.

The shards already carry exact full-render ``input_ids`` and ``labels``, so this
module deliberately does **not** let TRL re-tokenize or re-derive assistant
ownership. ``skip_prepare_dataset=True`` keeps those persisted labels authoritative.

Ampere-plus runs use FlashAttention-2 and the padding-free boundary path. Older
cards fall back to ``sdpa`` with padded batches and FP16, selected by
``resolve_sft_runtime``.

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
from smolqwen.console import console, logger
from smolqwen.tokenizer import assert_text_only_processing_class, load_tokenizer
from smolqwen.tracking import Tracker
from smolqwen.training.collate import (
    IGNORE_INDEX,
    CollateError,
    collator,
    padding_free_collator,
    record_to_sequence,
    supervised_positions,
)
from smolqwen.training.optim import (
    FLASH_ATTENTION_2_MIN_CAPABILITY,
    TEXT_ONLY_PEFT_EXCLUDE_MODULES,
    Toggle,
    apply_regional_compile,
    cast_adapters,
    cuda_available,
    format_ledger,
    ledger,
    resolve_attn_implementation,
    resolve_liger,
    resolve_precision,
    resolve_selective_logits,
)
from smolqwen.training.token_batching import TokenBudgetBatchSampler

LOG = logger(__name__)

# sm75 (Turing) is the oldest card this project runs: below it, `sdpa`'s
# memory-efficient backend and the FLA/causal-conv Triton kernels are not
# validated, so a production run refuses rather than silently degrading.
TURING_MIN_CAPABILITY: tuple[int, int] = (7, 5)


class SftError(RuntimeError):
    """Raised when the run cannot be assembled from the resolved config."""


@dataclass(frozen=True)
class SftRuntime:
    """The effective device-specific execution path for one SFT run.

    `precision` carries the dtype decision's reason into the ledger; it is None
    for the CPU assembly path, which resolves its dtype from config alone.
    """

    attention: Toggle
    dtype_name: str
    bf16: bool
    fp16: bool
    padding_free: bool
    precision: Toggle | None = None


def resolve_sft_runtime(
    config: SftConfig,
    *,
    capability: tuple[int, int] | None = None,
    has_cuda: bool | None = None,
    has_flash_attn: bool | None = None,
    require_cuda: bool = False,
    require_kernels: bool = False,
) -> SftRuntime:
    """Resolve attention, dtype, and batch shape from the actual GPU.

    Two attention implementations exist, and the card picks between them:
    FlashAttention-2 on Ampere-plus, `sdpa` below that.  FA2 is what carries the
    `cu_seq_lens` boundary contract, so the padding-free path is available exactly
    when FA2 is, and a pre-Ampere card runs padded batches with a real attention
    mask under FP16.
    """
    # `require_kernels=True` is the production-run guard.  Assembly tests may run
    # with a CUDA-enabled host but without the optional binary kernels, and must
    # retain the old downgrade-only inspection behavior.
    strict = require_kernels
    cuda = cuda_available() if has_cuda is None else has_cuda
    if not cuda:
        if require_cuda or require_kernels:
            raise SftError(
                "train-sft requires a CUDA GPU; use build_trainer for CPU assembly tests"
            )
        # CPU assembly keeps the padding-free shapes so the boundary contract is
        # still what the tests inspect, even though no kernel consumes them here.
        attention = resolve_attn_implementation(
            config.optimization.attn_implementation,
            has_flash_attn=has_flash_attn,
            has_cuda=False,
        )
        dtype_name = "bfloat16" if config.optimization.bf16 else "float32"
        return SftRuntime(
            attention=attention,
            dtype_name=dtype_name,
            bf16=config.optimization.bf16,
            fp16=False,
            padding_free=True,
        )

    if capability is None:
        import torch

        major, minor = torch.cuda.get_device_capability()
        capability = (int(major), int(minor))
    if capability < TURING_MIN_CAPABILITY and strict:
        raise SftError(
            "Qwen3.5 SFT requires compute capability "
            f"{TURING_MIN_CAPABILITY[0]}.{TURING_MIN_CAPABILITY[1]} or newer; "
            f"found sm{capability[0]}{capability[1]}"
        )

    requested = config.optimization.attn_implementation
    attention = resolve_attn_implementation(
        requested,
        has_flash_attn=has_flash_attn,
        has_cuda=True,
        capability=capability,
    )
    precision = resolve_precision(config.optimization.bf16, capability=capability)
    # Only FlashAttention-2 consumes the `cu_seq_lens` boundary metadata, so a
    # production run's batch shape follows its attention decision rather than a
    # separate switch.  Assembly inspection (`require_kernels=False`) keeps the
    # padding-free shapes either way, so the boundary contract stays testable on a
    # dev box that has no kernel wheels.
    padding_free = attention.name == "flash_attention_2" or not strict
    if (
        strict
        and capability >= FLASH_ATTENTION_2_MIN_CAPABILITY
        and requested == "flash_attention_2"
        and attention.name != "flash_attention_2"
    ):
        raise SftError(
            "Ampere-plus padding-free SFT requires the official flash_attn wheel; "
            f"resolved {attention.name}"
        )
    return SftRuntime(
        attention=attention,
        dtype_name=precision.name,
        bf16=precision.name == "bfloat16",
        fp16=precision.name == "float16",
        padding_free=padding_free,
        precision=precision,
    )


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


def validate_records(
    records: Iterable[dict[str, Any]],
    *,
    label: str,
    max_sequence_length: int | None = None,
    max_tokens_per_microbatch: int | None = None,
) -> ShardStats:
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
        if max_sequence_length is not None and len(input_ids) > max_sequence_length:
            raise SftError(
                f"{label}[{index}]: trajectory length {len(input_ids)} exceeds "
                f"profile.max_seq_length={max_sequence_length}; regenerate with "
                "`smolqwen prepare-sft` and the matching cap"
            )
        if max_tokens_per_microbatch is not None and len(input_ids) > max_tokens_per_microbatch:
            raise SftError(
                f"{label}[{index}]: trajectory length {len(input_ids)} exceeds "
                f"profile.max_tokens_per_microbatch={max_tokens_per_microbatch}; "
                "a trajectory cannot be split"
            )
        samples += 1
        tokens += len(input_ids)
        supervised += sum(1 for value in labels if value != IGNORE_INDEX)
    return ShardStats(
        path=label, samples=samples, total_tokens=tokens, supervised_tokens=supervised
    )


def validate_shard(
    path: Path | str,
    *,
    label: str,
    max_sequence_length: int | None = None,
    max_tokens_per_microbatch: int | None = None,
) -> ShardStats:
    return validate_records(
        iter_records(path),
        label=label,
        max_sequence_length=max_sequence_length,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )


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
    """The train shard, already validated against the mask contract."""

    train: Any
    train_stats: ShardStats

    @property
    def supervised_fraction(self) -> float | None:
        """Share of train positions carrying a target, or None for an empty shard.

        How much a selective head saves is a property of the shard rather than of
        the card: a reasoning shard supervises most of its positions, while a
        non-reasoning one supervises the answers only. Recorded in the toggle ledger
        so the measured win can be read against it.
        """
        total = self.train_stats.total_tokens
        return self.train_stats.supervised_tokens / total if total else None


def load_shards(
    dataset_dir: Path | str,
    *,
    max_sequence_length: int | None = None,
    max_tokens_per_microbatch: int | None = None,
) -> Shards:
    """Validate `train.jsonl`, then hand back the Arrow-backed dataset."""
    directory = Path(dataset_dir)
    train_path = directory / "train.jsonl"
    if not train_path.is_file():
        raise SftError(f"{train_path} missing -- run `smolqwen prepare-sft` first")

    train_stats = validate_shard(
        train_path,
        label="train",
        max_sequence_length=max_sequence_length,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )
    if not train_stats.samples:
        raise SftError(f"{train_path} is empty")

    return Shards(
        train=load_shard_dataset(train_path),
        train_stats=train_stats,
    )


@dataclass(frozen=True)
class Assembled:
    """What `build_trainer` produced, so callers can inspect it without training."""

    trainer: Any
    toggles: tuple[Toggle, ...]
    train_stats: ShardStats
    resume_from: str | None

    @property
    def train_size(self) -> int:
        return self.train_stats.samples


def _lora_config(config: SftConfig) -> Any:
    from peft import LoraConfig as PeftLoraConfig

    lora = config.lora
    target = lora.target_modules
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=target if isinstance(target, str) else list(target),
        # `all-linear` is the project contract; exclude Qwen3.5's unused visual
        # tower so a text-only adapter remains loadable by vLLM.
        exclude_modules=(
            TEXT_ONLY_PEFT_EXCLUDE_MODULES
            if any(marker in config.model_id.casefold() for marker in ("qwen3.5", "qwen3_5"))
            else None
        ),
        task_type="CAUSAL_LM",
        bias="none",
    )


def _sft_config(
    config: SftConfig,
    *,
    runtime: SftRuntime,
    use_liger: bool,
    report_to: list[str],
) -> Any:
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
        # Train-only, matching upstream EnvScaler's SFT (LlamaFactory, no
        # validation set): overfitting is not gated on a val curve, and the
        # capability curve is GRPO's bench_eval boundary.
        eval_strategy="no",
        save_strategy="steps",
        seed=training.seed,
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        gradient_checkpointing=optimization.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=use_liger,
        model_init_kwargs={
            "dtype": runtime.dtype_name,
            "attn_implementation": runtime.attention.name,
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
        def __init__(
            self,
            *args: Any,
            max_tokens_per_microbatch: int,
            padding_free: bool,
            selective_logits: bool = False,
            **kwargs: Any,
        ) -> None:
            self.max_tokens_per_microbatch = max_tokens_per_microbatch
            self._project_padding_free = padding_free
            self._selective_logits = selective_logits
            self.token_batch_sampler: TokenBudgetBatchSampler | None = None
            self._token_dataloaders: dict[str, Any] = {}
            self._step_tokens = 0
            self._step_supervised_tokens = 0
            self._step_trajectories = 0
            self._step_microbatches = 0
            self._step_max_trajectory_length = 0
            self._step_padding_saved = 0
            self._step_projected_positions = 0
            super().__init__(*args, **kwargs)

        def reset_step_metrics(self) -> None:
            self._step_tokens = 0
            self._step_supervised_tokens = 0
            self._step_trajectories = 0
            self._step_microbatches = 0
            self._step_max_trajectory_length = 0
            self._step_padding_saved = 0
            self._step_projected_positions = 0

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            """Narrow the head's input to supervised positions before delegating.

            `logits_to_keep` as an index tensor is upstream's own contract for this
            (`modeling_qwen3_5.py:1643`), and Liger's fused head forwards it
            unchanged -- so the 248,320-wide projection runs on the rows that carry
            a target instead of all of them. `shift_labels` must accompany it: the
            rows are no longer contiguous, so the internal `labels[..., 1:]` shift
            would align targets against the wrong positions.

            `num_items_in_batch` is untouched. `Trainer._get_num_items_in_batch`
            already prefers a collator-supplied `shift_labels`, and either way it
            counts non-`-100` targets -- a quantity selection does not change. That
            is what keeps the loss numerically identical rather than merely close.
            """
            if self._selective_logits and "labels" in inputs:
                inputs = dict(inputs)
                labels = inputs.pop("labels")
                index, shift_labels = supervised_positions(labels)
                inputs["logits_to_keep"] = index
                inputs["shift_labels"] = shift_labels
                self._step_projected_positions += int(index.numel())
            elif "labels" in inputs:
                self._step_projected_positions += int(inputs["labels"].shape[-1])
            return super().compute_loss(  # type: ignore[no-untyped-call]
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        def training_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            num_items_in_batch: Any = None,
        ) -> Any:
            if self._project_padding_free:
                # The flattened collator has no padding, so the final dimension
                # is the exact valid-token count.  Counting shape metadata avoids
                # a device synchronization just for telemetry.
                tokens = int(inputs["input_ids"].shape[-1])
                trajectories = int(inputs["cu_seq_lens_q"].shape[0]) - 1
                max_trajectory = int(inputs["max_length_q"])
                padding_saved = trajectories * max_trajectory - tokens
            else:
                # Padded mode is budgeted by the dense tensor it executes, not
                # by the sum of the unpadded source rows.
                tokens = int(inputs["input_ids"].numel())
                trajectories = int(inputs["input_ids"].shape[0])
                max_trajectory = int(inputs["input_ids"].shape[-1])
                padding_saved = 0
            self._step_tokens += tokens
            self._step_trajectories += trajectories
            self._step_microbatches += 1
            self._step_max_trajectory_length = max(self._step_max_trajectory_length, max_trajectory)
            self._step_padding_saved += padding_saved
            # Transformers computes this once across the whole accumulation
            # window and passes the same total into every micro-step.
            if num_items_in_batch is not None and not self._step_supervised_tokens:
                self._step_supervised_tokens = int(num_items_in_batch)
            elif num_items_in_batch is None:
                labels = inputs.get("labels")
                if labels is not None:
                    self._step_supervised_tokens += int((labels[..., 1:] != IGNORE_INDEX).sum())
            return super().training_step(  # type: ignore[no-untyped-call]
                model, inputs, num_items_in_batch
            )

        def _token_dataloader(self, dataset: Any, *, training: bool) -> Any:
            key = "train" if training else f"eval:{id(dataset)}"
            if key in self._token_dataloaders:
                return self._token_dataloaders[key]
            sampler = TokenBudgetBatchSampler(
                [int(length) for length in dataset[LENGTH_COLUMN]],
                max_tokens=self.max_tokens_per_microbatch,
                seed=int(self.args.seed),
                shuffle=training,
                padding_free=self._project_padding_free,
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
            prepared = self.accelerator.prepare(dataloader)
            self._token_dataloaders[key] = prepared
            return prepared

        def get_train_dataloader(self) -> Any:
            if self.train_dataset is None:
                raise SftError("training requires a train dataset")
            return self._token_dataloader(self.train_dataset, training=True)

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


def assert_sft_runtime(runtime: SftRuntime) -> None:
    """Fail before model loading if the selected SFT kernels are absent."""
    import importlib.util

    import torch

    if not torch.cuda.is_available():
        raise SftError("train-sft requires a CUDA GPU")

    # The GDN mixer and its causal convolution have no usable fallback at this
    # model's sizes, so they are required on every card. Attention does have one:
    # `sdpa` needs no wheel.
    missing = [
        package for package in ("fla", "causal_conv1d") if importlib.util.find_spec(package) is None
    ]
    if (
        runtime.attention.name == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        missing.append("flash_attn")
    if missing:
        raise SftError(f"selected SFT runtime is missing required kernels: {', '.join(missing)}")


class ThroughputCallback(TrainerCallback):
    """s/Mtok and peak VRAM per step, through the Phase 1 meter.

    Seconds per million tokens rather than tokens/s: the two profiles run
    different batch sizes, so tokens/s is not comparable across them.
    """

    def __init__(self, tracker: Tracker, *, tokens_per_step: int | None = None) -> None:
        self.tracker = tracker
        self.tokens_per_step = tokens_per_step
        self.trainer: Any | None = None

    def bind_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.trainer is not None:
            self.trainer.reset_step_metrics()
        self.tracker.meter.start_step()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        tokens = (
            self.trainer._step_tokens if self.trainer is not None else self.tokens_per_step or 0
        )
        extra = {}
        if self.trainer is not None:
            extra = {
                "sft/supervised_tokens": self.trainer._step_supervised_tokens,
                "sft/trajectories": self.trainer._step_trajectories,
                "sft/microbatches": self.trainer._step_microbatches,
                "sft/max_trajectory_length": self.trainer._step_max_trajectory_length,
                "sft/padding_tokens_saved": self.trainer._step_padding_saved,
                # Rows the 248,320-wide head actually projected. Equals the token
                # count when the head is dense, and the supervised count when it is
                # selective -- which is what makes the toggle's effect a measured
                # number rather than a claim.
                "sft/projected_positions": self.trainer._step_projected_positions,
            }
        self.tracker.log_step(tokens=tokens, step=int(state.global_step), **extra)

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logs = kwargs.get("logs")
        if logs:
            self.tracker.log(dict(logs), step=int(state.global_step))


class CheckpointPushCallback(TrainerCallback):
    """Push the complete Trainer checkpoint and run identity on every save.

    `save_adapter` copies the checkpoint directory, including Trainer's optimizer,
    scheduler, RNG and `trainer_state.json`. Trainer uses that state to skip data
    on resume; applying a second custom sampler cursor would skip batches twice.
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
    runtime: SftRuntime | None = None,
) -> Assembled:
    """Assemble the trainer without starting it, so a smoke test can inspect it."""
    shards = load_shards(
        dataset_dir or config.dataset_dir,
        max_sequence_length=config.profile.max_seq_length,
        max_tokens_per_microbatch=config.profile.max_tokens_per_microbatch,
    )

    tokenizer = assert_text_only_processing_class(
        load_tokenizer(config.model_id, revision=config.model_revision)
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise SftError(f"{config.model_id} tokenizer has neither a pad nor an eos token id")

    runtime = runtime or resolve_sft_runtime(config)

    liger = resolve_liger(config.optimization.liger_fused_linear_cross_entropy)
    selective = resolve_selective_logits(
        config.optimization.selective_logit_loss,
        liger_enabled=liger.enabled,
        padding_free=runtime.padding_free,
        supervised_fraction=shards.supervised_fraction,
    )

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
    data_collator = (
        padding_free_collator(
            config.profile.max_tokens_per_microbatch,
            max_sequence_length=config.profile.max_seq_length,
        )
        if runtime.padding_free
        else collator(
            pad_token_id,
            max_length=config.profile.max_seq_length,
        )
    )
    trainer = trainer_class(
        model=config.model_id,
        args=_sft_config(config, runtime=runtime, use_liger=liger.enabled, report_to=[]),
        data_collator=data_collator,
        train_dataset=shards.train,
        processing_class=tokenizer,
        peft_config=_lora_config(config),
        max_tokens_per_microbatch=config.profile.max_tokens_per_microbatch,
        padding_free=runtime.padding_free,
        selective_logits=selective.enabled,
    )

    # FP16 runs must keep trainable adapters in FP32: GradScaler rejects FP16
    # gradients. On the BF16 path, honor the explicit adapter dtype setting so a
    # caller can retain PEFT's FP32 adapters when that trade-off is intentional.
    adapter_dtype = runtime.dtype_name if runtime.fp16 else config.lora.adapter_dtype
    adapters = cast_adapters(trainer.model, adapter_dtype)
    compiled = apply_regional_compile(
        trainer.model,
        exclude_patterns=config.optimization.compile_exclude_patterns,
        enabled=config.optimization.regional_torch_compile,
    )
    dtype = Toggle(
        "training_dtype",
        runtime.precision.enabled if runtime.precision is not None else True,
        f"effective model dtype: {runtime.dtype_name}; "
        f"{'padding-free' if runtime.padding_free else 'padded'} SFT path"
        + (f"; {runtime.precision.detail}" if runtime.precision is not None else ""),
    )
    toggles = (runtime.attention, dtype, liger, selective, adapters, compiled)

    throughput = ThroughputCallback(run)
    throughput.bind_trainer(trainer)
    trainer.add_callback(throughput)
    trainer.add_callback(CheckpointPushCallback(checkpoint_store, run))
    run.config.update(ledger(toggles))

    return Assembled(
        trainer=trainer,
        toggles=toggles,
        train_stats=shards.train_stats,
        resume_from=resume_from,
    )


def run_train_sft(config: SftConfig, *, resume: bool = False) -> int:
    """`smolqwen train-sft`: train and record the ledger. No evaluation.

    Train-only matches upstream EnvScaler's SFT: LlamaFactory with no validation
    set, loss curve on the training loss alone. Capability is measured by GRPO's
    benchmark boundary after SFT.
    """
    runtime = resolve_sft_runtime(config, require_cuda=True, require_kernels=True)
    assert_sft_runtime(runtime)
    assembled = build_trainer(config, resume=resume, runtime=runtime)
    trainer = assembled.trainer
    console().print(format_ledger(list(assembled.toggles)))
    train = assembled.train_stats
    LOG.info(
        "train %d samples / %d tokens (%d supervised)",
        train.samples,
        train.total_tokens,
        train.supervised_tokens,
    )

    trainer.train(resume_from_checkpoint=assembled.resume_from)
    trainer.save_model(config.output_dir)
    return 0
