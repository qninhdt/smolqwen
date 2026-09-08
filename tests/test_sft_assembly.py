"""Trainer assembly: the Phase 2 mask reaches the step, and resume does not fork.

`build_trainer` is where the two silent failures of this phase would live:

- TRL re-deriving labels from text instead of using the stored mask. So this
  asserts `skip_prepare_dataset` is on, the dataset keeps its raw id columns, and
  the collator is ours -- if any of the three flipped, TRL would re-tokenize
  through the chat template and re-infer the completion boundary.
- a resumed run forking the W&B curve. So this asserts the persisted run id is
  what the `Tracker` resumes with.

The model is a 4-layer random-weight Qwen3.5 saved locally, so nothing downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.artifacts import CheckpointStore, ResumeState
from smolqwen.config_models import (
    LoraConfig,
    ProfileConfig,
    SftConfig,
    TrackingConfig,
    TrainingConfig,
)
from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS
from smolqwen.tracking import Tracker
from smolqwen.training.optim import Toggle
from smolqwen.training.sft import SftError, _lora_config, _parameter_summary, build_trainer
from tests.helpers import write_tiny_checkpoint

pytestmark = pytest.mark.slow

VOCAB = 256


def _record(index: int, *, length: int = 11) -> dict[str, Any]:
    prompt_length = length - 5
    prompt = [(index + position) % VOCAB for position in range(prompt_length)]
    completion = [(index + position + 30) % VOCAB for position in range(5)]
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": f"t{index}:non_conversation",
        "task_id": f"t{index}",
        "env_id": "env_1_sft",
        "mode": "non_conversation",
        "input_ids": prompt + completion,
        "labels": [-100] * len(prompt) + completion[:3] + [-100, -100],
        "seq_length": length,
        "supervised_tokens": 3,
    }


def _shards(directory: Path, *, length: int = 11) -> Path:
    shard_dir = directory / "sft"
    shard_dir.mkdir(parents=True)
    rows = [json.dumps(_record(index, length=length)) for index in range(3)]
    (shard_dir / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return shard_dir


def _tiny_checkpoint(directory: Path) -> Path:
    return write_tiny_checkpoint(directory, vocab_size=VOCAB)


def _config(model_dir: Path, output_dir: Path, **overrides: Any) -> SftConfig:
    payload: dict[str, Any] = {
        "model_id": str(model_dir),
        "model_revision": None,
        "output_dir": str(output_dir),
        "merged_dir": str(output_dir / "merged"),
        "profile": ProfileConfig(
            micro_batch=1,
            grad_accum=1,
            max_seq_length=512,
            max_tokens_per_microbatch=512,
        ),
        "training": TrainingConfig(max_steps=1, save_steps=1, logging_steps=1),
        "tracking": TrackingConfig(hub_repo_id=None, local_artifact_dir=str(output_dir)),
    }
    payload.update(overrides)
    return SftConfig(**payload)


@pytest.fixture
def assembled(tmp_path: Path) -> Any:
    config = _config(_tiny_checkpoint(tmp_path / "base"), tmp_path / "out")
    # No CUDA and no flash_attn on CI, so the attention toggle downgrades; that is
    # the recorded-reason path, not a failure.
    return build_trainer(
        config, dataset_dir=_shards(tmp_path), tracker=Tracker(project="t", enabled=False)
    )


def test_trl_never_re_derives_the_mask(assembled: Any) -> None:
    args = assembled.trainer.args
    assert args.dataset_kwargs == {"skip_prepare_dataset": True}
    # With preparation skipped, `max_length` truncation is unreachable; leaving the
    # 1024 default would be a silent claim about sample length.
    assert args.max_length is None
    assert args.packing is False
    assert args.assistant_only_loss is False


def test_the_dataset_keeps_the_stored_ids_not_rendered_text(assembled: Any) -> None:
    columns = set(assembled.trainer.train_dataset.column_names)
    # `length` is materialized at load for the length-grouped sampler; without it
    # the sampler tokenizes the whole shard to measure what the records determine.
    assert columns == {
        "schema_version",
        "semantics",
        "trajectory_uid",
        "input_ids",
        "labels",
        "seq_length",
        "supervised_tokens",
        "length",
    }
    assert "text" not in columns


def test_the_length_column_matches_the_stored_ids(assembled: Any) -> None:
    dataset = assembled.trainer.train_dataset
    for row in dataset:
        assert row["length"] == len(row["input_ids"])


def test_the_collator_is_the_phase_three_one(assembled: Any) -> None:
    import torch

    batch = assembled.trainer.data_collator([_record(0), _record(1)])
    assert set(batch) == {
        "input_ids",
        "labels",
        "position_ids",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "seq_idx",
    }
    assert batch["input_ids"].shape == (1, 22)
    assert isinstance(batch["labels"], torch.Tensor)
    # 3 supervised positions per record, exactly as the stored mask says.
    assert int((batch["labels"] != -100).sum()) == 6


def test_train_dataloader_uses_variable_row_token_sampler(assembled: Any) -> None:
    dataloader = assembled.trainer.get_train_dataloader()
    assert assembled.trainer.get_train_dataloader() is dataloader
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"].shape[1] <= 512
    assert assembled.trainer.token_batch_sampler is not None


def test_token_budget_trainer_executes_one_flattened_step(assembled: Any) -> None:
    import torch

    result = assembled.trainer.train()

    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert assembled.trainer._step_tokens == 33
    assert assembled.trainer._step_supervised_tokens == 9
    assert assembled.trainer._step_trajectories == 3
    assert assembled.trainer._step_microbatches == 1
    assert assembled.trainer._step_max_trajectory_length == 11
    assert assembled.trainer._step_padding_saved == 0


def test_gradient_accumulation_uses_the_window_supervised_token_count(tmp_path: Path) -> None:
    config = _config(
        _tiny_checkpoint(tmp_path / "base"),
        tmp_path / "out",
        profile=ProfileConfig(
            micro_batch=1,
            grad_accum=2,
            max_seq_length=512,
            max_tokens_per_microbatch=256,
        ),
    )
    assembled = build_trainer(
        config,
        dataset_dir=_shards(tmp_path, length=256),
        tracker=Tracker(project="t", enabled=False),
    )

    result = assembled.trainer.train()

    assert result.global_step == 1
    assert assembled.trainer._step_microbatches == 2
    assert assembled.trainer._step_tokens == 512
    assert assembled.trainer._step_supervised_tokens == 6


def test_lora_is_attached_and_only_adapters_train(assembled: Any) -> None:
    model = assembled.trainer.model
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable), "a base weight is trainable"


def test_parameter_summary_reports_trainable_frozen_and_total(assembled: Any) -> None:
    summary = _parameter_summary(assembled.trainer.model)
    trainable, total = assembled.trainer.model.get_nb_trainable_parameters()

    assert f"trainable={trainable:,}" in summary
    assert f"frozen={total - trainable:,}" in summary
    assert f"total={total:,}" in summary


def test_qwen35_all_linear_targets_leave_the_unused_visual_tower_alone() -> None:
    config = SftConfig()
    lora = _lora_config(config)

    assert lora.target_modules == "all-linear"
    assert lora.exclude_modules == r".*\.visual(?:\..*)?$"


def test_adapters_are_cast_to_bf16_not_left_in_fp32(assembled: Any) -> None:
    import torch

    names = {toggle.name for toggle in assembled.toggles}
    assert "adapter_dtype" in names
    dtypes = {p.dtype for p in assembled.trainer.model.parameters() if p.requires_grad}
    assert dtypes == {torch.bfloat16}


def test_explicit_fp32_adapter_dtype_is_honored_on_the_bf16_path(tmp_path: Path) -> None:
    config = _config(
        _tiny_checkpoint(tmp_path / "base"),
        tmp_path / "out",
        lora=LoraConfig(adapter_dtype="float32"),
    )
    assembled = build_trainer(
        config,
        dataset_dir=_shards(tmp_path),
        tracker=Tracker(project="t", enabled=False),
    )

    import torch

    dtypes = {p.dtype for p in assembled.trainer.model.parameters() if p.requires_grad}
    assert dtypes == {torch.float32}


def test_only_the_train_shard_is_loaded(assembled: Any) -> None:
    assert assembled.train_size == 3
    # SFT is intentionally train-only: no eval dataset or eval loss exists.
    assert assembled.trainer.eval_dataset is None


def test_resume_without_anything_pushed_fails_loudly(tmp_path: Path) -> None:
    config = _config(_tiny_checkpoint(tmp_path / "base"), tmp_path / "out")
    store = CheckpointStore(None, tmp_path / "missing-adapter")
    with pytest.raises(SftError, match="--resume"):
        build_trainer(
            config,
            resume=True,
            dataset_dir=_shards(tmp_path),
            store=store,
            tracker=Tracker(project="t", enabled=False),
        )


def test_resume_continues_the_same_wandb_run(tmp_path: Path) -> None:
    """A resume that forks the run splits one training curve across two charts."""
    config = _config(_tiny_checkpoint(tmp_path / "base"), tmp_path / "out")
    store = CheckpointStore(None, tmp_path / "adapter")
    store.write_resume_state(ResumeState(wandb_run_id="run-abc", global_step=42))

    result = build_trainer(
        config,
        resume=True,
        dataset_dir=_shards(tmp_path),
        store=store,
    )
    assert result.resume_from == str(store.local_dir)
    assert store.read_resume_state() is not None
    assert store.read_resume_state().wandb_run_id == "run-abc"  # type: ignore[union-attr]


def test_the_ledger_is_recorded_in_the_run_config(tmp_path: Path) -> None:
    config = _config(_tiny_checkpoint(tmp_path / "base"), tmp_path / "out")
    tracker = Tracker(project="t", enabled=False)
    build_trainer(config, dataset_dir=_shards(tmp_path), tracker=tracker)
    keys = [key for key in tracker.config if key.startswith("optimization/")]
    assert {"optimization/adapter_dtype"} <= set(keys)
    assert all(tracker.config[key].startswith(("on: ", "off: ")) for key in keys)


def test_run_train_sft_starts_and_finishes_the_tracker(tmp_path: Path, monkeypatch: Any) -> None:
    """W&B metrics are lost silently when the run is never started.

    `run_train_sft` owns the tracker lifecycle (mirroring GRPO): without an
    explicit `start()` every `log_step` no-ops and a resumed run forks the curve.
    """
    from smolqwen.training import sft as sft_module

    calls: list[str] = []
    runtime = sft_module.SftRuntime(
        attention=Toggle("sdpa", True, "test"),
        dtype_name="bfloat16",
        bf16=True,
        fp16=False,
        padding_free=True,
    )

    class SpyTracker(Tracker):
        def start(self) -> None:
            calls.append("start")

        def finish(self) -> None:
            calls.append("finish")

    class SpyModel:
        @staticmethod
        def get_nb_trainable_parameters() -> tuple[int, int]:
            return 1, 2

    class SpyTrainer:
        model = SpyModel()

        def train(self, resume_from_checkpoint: str | None = None) -> None:
            calls.append(f"train:{resume_from_checkpoint}")

        def save_model(self, output_dir: str) -> None:
            calls.append("save")

    def fake_build_trainer(config: Any, **kwargs: Any) -> Any:
        return sft_module.Assembled(
            trainer=SpyTrainer(),
            toggles=(),
            train_stats=build_trainer(
                config,
                dataset_dir=_shards(tmp_path),
                tracker=SpyTracker(project="t", enabled=False),
            ).train_stats,
            resume_from=None,
            tracker=SpyTracker(project="t", enabled=False),
        )

    monkeypatch.setattr(sft_module, "resolve_sft_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(sft_module, "assert_sft_runtime", lambda runtime: None)
    monkeypatch.setattr(sft_module, "build_trainer", fake_build_trainer)

    exit_code = sft_module.run_train_sft(_config(_tiny_checkpoint(tmp_path / "b"), tmp_path / "o"))

    assert exit_code == 0
    assert calls[0] == "start"
    assert calls[-1] == "finish"
    assert any(call.startswith("train:") for call in calls)
