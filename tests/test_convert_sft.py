"""Full-trajectory conversion, cap filtering, schema, and accounting."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from smolqwen.data.convert_sft import (
    SFT_SCHEMA_VERSION,
    SFT_SEMANTICS,
    SKIP_TOO_LONG,
    ConversionEvent,
    ConversionReport,
    Converted,
    Skipped,
    convert_trajectories,
    sample_to_record,
)
from smolqwen.data.loader import LoadStats, Message, iter_trajectories
from smolqwen.data.render import RenderedSample, render_training_sample
from smolqwen.data.splits import split_trajectory_ids
from tests.helpers import OfflineTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _render(tokenizer: OfflineTokenizer) -> Callable[..., RenderedSample]:
    def render(messages: Sequence[Message], **kwargs: Any) -> RenderedSample:
        return render_training_sample(tokenizer, messages, **kwargs)

    return render


def _events(cap: int) -> tuple[list[ConversionEvent], LoadStats]:
    stats = LoadStats()
    events = list(
        convert_trajectories(
            iter_trajectories(FIXTURES / "trajectories.json", stats=stats),
            render=_render(OfflineTokenizer()),
            max_seq_length=cap,
        )
    )
    return events, stats


def test_one_valid_row_emits_exactly_one_sample() -> None:
    events, stats = _events(cap=10_000_000)
    converted = [event for event in events if isinstance(event, Converted)]
    assert len(converted) == 2
    assert stats.total == 3
    assert stats.malformed == 1
    assert all(event.sample.trajectory_uid == event.trajectory_uid for event in converted)
    assert len({event.trajectory_uid for event in converted}) == 2


def test_over_cap_trajectory_is_skipped_whole_not_truncated() -> None:
    events, _ = _events(cap=1)
    assert all(isinstance(event, Skipped) for event in events)
    assert {event.reason for event in events if isinstance(event, Skipped)} == {SKIP_TOO_LONG}
    report = ConversionReport()
    for event in events:
        assert isinstance(event, Skipped)
        report.note_skipped(event)
    assert report.skipped == 2
    assert (report.skip_reasons or {})[SKIP_TOO_LONG] == 2


def test_record_is_versioned_input_ids_and_labels() -> None:
    events, _ = _events(cap=10_000_000)
    for event in events:
        if not isinstance(event, Converted):
            continue
        record = sample_to_record(event.sample)
        assert record["schema_version"] == SFT_SCHEMA_VERSION
        assert record["semantics"] == SFT_SEMANTICS
        assert record["trajectory_uid"] == f"{record['task_id']}:{record['mode']}"
        assert not {"segment_index", "prompt_ids", "completion_ids", "loss_mask"} & record.keys()
        assert len(record["input_ids"]) == len(record["labels"]) == record["seq_length"]
        assert record["supervised_tokens"] == sum(label != -100 for label in record["labels"])
        json.loads(json.dumps(record))


def test_report_accounts_for_every_input_row_and_sample_equals_row() -> None:
    events, stats = _events(cap=10_000_000)
    report = ConversionReport()
    for event in events:
        report.note_skipped(event) if isinstance(event, Skipped) else report.note_converted(event)
    payload = report.to_dict(input_shas={}, load_stats=stats)
    assert payload["accounted"] is True
    assert payload["converted"] == payload["samples"] == 2
    assert payload["by_mode"]["conversation"] == {"trajectories": 1, "samples": 1}
    assert payload["by_mode"]["non_conversation"] == {"trajectories": 1, "samples": 1}


def test_paired_variants_route_by_task_id_not_unique_uid() -> None:
    task_ids = ["task-a", "task-b"]
    split = split_trajectory_ids(task_ids, seed=5, val_fraction=0.5)
    for task_id in task_ids:
        variants = [f"{task_id}:conversation", f"{task_id}:non_conversation"]
        assert len({split.partition(task_id) for _ in variants}) == 1
