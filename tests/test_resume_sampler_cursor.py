from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from smolqwen.artifacts import CheckpointStore
from smolqwen.tracking import Tracker
from smolqwen.training.grpo import (
    CurriculumCursor,
    CursorRepeatSampler,
    GrpoCheckpointCallback,
)


def _group_sequence(sampler: CursorRepeatSampler, group_size: int) -> list[int]:
    rows = list(sampler)
    return [rows[index] for index in range(0, len(rows), group_size)]


def test_resume_rotates_to_the_next_unseen_curriculum_group() -> None:
    source = list(range(8))
    uninterrupted = CursorRepeatSampler(
        source, mini_repeat_count=2, batch_size=2, repeat_count=1, cursor=0
    )
    cursor = CurriculumCursor(
        start=0,
        dataset_size=8,
        groups_per_generation=2,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
    )
    resumed = CursorRepeatSampler(
        source,
        mini_repeat_count=2,
        batch_size=2,
        repeat_count=1,
        cursor=cursor.at_step(1),
    )
    assert _group_sequence(uninterrupted, 2)[:2] == [0, 1]
    assert _group_sequence(resumed, 2)[:2] == [2, 3]


def test_cursor_accounts_for_gradient_accumulation_and_generation_reuse() -> None:
    cursor = CurriculumCursor(
        start=3,
        dataset_size=20,
        groups_per_generation=4,
        gradient_accumulation_steps=8,
        steps_per_generation=4,
    )
    assert cursor.at_step(2) == 19


def test_checkpoint_persists_cursor_and_wandb_run_id(tmp_path: Path) -> None:
    class Run:
        id = "same-run"

        def log(self, data: Any, *, step: int | None = None) -> None:
            return None

        def finish(self) -> None:
            return None

    output = tmp_path / "out"
    checkpoint = output / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    store = CheckpointStore(None, tmp_path / "adapter")
    tracker = Tracker(project="test", run=Run())
    cursor = CurriculumCursor(0, 20, 4, 8, 4)
    callback = GrpoCheckpointCallback(store, tracker, cursor)

    callback.on_save(
        SimpleNamespace(output_dir=str(output)),
        SimpleNamespace(global_step=2),
        SimpleNamespace(),
    )

    state = store.read_resume_state()
    assert state is not None
    assert state.wandb_run_id == "same-run"
    assert state.sampler_cursor == 16
