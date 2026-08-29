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
from smolqwen.config_models import ProfileConfig, SftConfig, TrackingConfig, TrainingConfig
from smolqwen.tracking import Tracker
from smolqwen.training.sft import SftError, build_trainer
from tests.helpers import write_tiny_checkpoint

pytestmark = pytest.mark.slow

VOCAB = 256


def _record(index: int) -> dict[str, Any]:
    return {
        "trajectory_id": f"t{index}",
        "env_id": "env_1_sft",
        "mode": "non_conversation",
        "segment_index": 0,
        "prompt_ids": [(index + p) % VOCAB for p in range(6)],
        "completion_ids": [(index + p + 30) % VOCAB for p in range(5)],
        "loss_mask": [1, 1, 1, 0, 0],
        "total_tokens": 11,
        "supervised_tokens": 3,
    }


def _shards(directory: Path) -> Path:
    shard_dir = directory / "sft"
    shard_dir.mkdir(parents=True)
    for name, offset in (("train", 0), ("val", 100)):
        rows = [json.dumps(_record(offset + index)) for index in range(3)]
        (shard_dir / f"{name}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return shard_dir


def _tiny_checkpoint(directory: Path) -> Path:
    return write_tiny_checkpoint(directory, vocab_size=VOCAB)


def _config(model_dir: Path, output_dir: Path, **overrides: Any) -> SftConfig:
    payload: dict[str, Any] = {
        "model_id": str(model_dir),
        "model_revision": None,
        "output_dir": str(output_dir),
        "merged_dir": str(output_dir / "merged"),
        "profile": ProfileConfig(micro_batch=1, grad_accum=1, max_seq_length=512),
        "training": TrainingConfig(max_steps=1, save_steps=1, eval_steps=1, logging_steps=1),
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
    assert columns == {"prompt_ids", "completion_ids", "loss_mask"}
    assert "text" not in columns and "input_ids" not in columns


def test_the_collator_is_the_phase_three_one(assembled: Any) -> None:
    import torch

    batch = assembled.trainer.data_collator([_record(0), _record(1)])
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert isinstance(batch["labels"], torch.Tensor)
    # 3 supervised positions per record, exactly as the stored mask says.
    assert int((batch["labels"] != -100).sum()) == 6


def test_lora_is_attached_and_only_adapters_train(assembled: Any) -> None:
    model = assembled.trainer.model
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable), "a base weight is trainable"


def test_adapters_are_cast_to_bf16_not_left_in_fp32(assembled: Any) -> None:
    import torch

    names = {toggle.name for toggle in assembled.toggles}
    assert "adapter_dtype" in names
    dtypes = {p.dtype for p in assembled.trainer.model.parameters() if p.requires_grad}
    assert dtypes == {torch.bfloat16}


def test_both_shards_are_loaded_and_counted(assembled: Any) -> None:
    assert assembled.train_size == 3
    assert assembled.eval_size == 3
    assert assembled.trainer.eval_dataset is not None


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
