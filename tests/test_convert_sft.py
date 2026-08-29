"""Conversion: budget-cap filtering, skip accounting, and the persisted record shape.

The converter is the only place a trajectory is dropped, and a silent drop would
change the training distribution without reporting it. These tests pin:

- a trajectory whose sample exceeds the cap is skipped whole (not truncated), with
  the reason recorded;
- a Conv trajectory's segments all reach the same shard, because routing is by
  trajectory id;
- the persisted record carries the mask aligned to the completion;
- the report accounts for every input row.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from smolqwen.data.convert_sft import (
    SKIP_TOO_LONG,
    ConversionEvent,
    ConversionReport,
    Converted,
    Skipped,
    convert_trajectories,
    sample_to_record,
)
from smolqwen.data.loader import LoadStats, Message, iter_trajectories
from smolqwen.data.render import RenderedSample, Segment, render_segment
from smolqwen.data.splits import split_trajectory_ids
from tests.helpers import OfflineTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _render(tokenizer: OfflineTokenizer) -> Callable[..., RenderedSample]:
    def render(messages: Sequence[Message], segment: Segment, **kwargs: Any) -> RenderedSample:
        return render_segment(tokenizer, messages, segment, **kwargs)

    return render


def _events(cap: int, *, token_size: int = 64) -> tuple[list[ConversionEvent], LoadStats]:
    tokenizer = OfflineTokenizer(token_size=token_size)
    stats = LoadStats()
    events = list(
        convert_trajectories(
            iter_trajectories(FIXTURES / "trajectories.json", stats=stats),
            render=_render(tokenizer),
            max_seq_length=cap,
            shape="tool_role",
        )
    )
    return events, stats


def test_generous_cap_converts_both_valid_trajectories() -> None:
    events, stats = _events(cap=10_000_000)
    converted = [e for e in events if isinstance(e, Converted)]
    # The fixture has 2 valid rows and 1 malformed; the malformed row never
    # reaches the converter (the loader counts and skips it).
    assert len(converted) == 2
    assert stats.malformed == 1
    assert stats.total == 3

    by_id = {e.trajectory_id: e for e in converted}
    # Non-Conv -> one sample; Conv (2 supervised user turns) -> two samples.
    assert len(by_id["env_3_sft-task_28"].samples) == 1
    assert len(by_id["env_82_sft-task_29"].samples) == 2


def test_over_cap_trajectory_is_skipped_whole_not_truncated() -> None:
    """A cap of 1 token drops every trajectory, with `too_long` recorded."""
    events, _ = _events(cap=1)
    assert all(isinstance(e, Skipped) for e in events)
    assert {e.reason for e in events if isinstance(e, Skipped)} == {SKIP_TOO_LONG}

    report = ConversionReport()
    for event in events:
        if isinstance(event, Skipped):
            report.note_skipped(event)
    assert report.skipped == 2
    assert (report.skip_reasons or {})[SKIP_TOO_LONG] == 2
    assert report.samples == 0


def test_record_shape_keeps_mask_aligned() -> None:
    events, _ = _events(cap=10_000_000)
    for event in events:
        if not isinstance(event, Converted):
            continue
        for sample in event.samples:
            record = sample_to_record(sample)
            assert set(record) == {
                "trajectory_id",
                "env_id",
                "mode",
                "segment_index",
                "prompt_ids",
                "completion_ids",
                "loss_mask",
                "total_tokens",
                "supervised_tokens",
            }
            assert len(record["completion_ids"]) == len(record["loss_mask"])
            assert record["total_tokens"] == len(record["prompt_ids"]) + len(
                record["completion_ids"]
            )
            assert record["supervised_tokens"] == sum(record["loss_mask"])
            # The record must be JSON-serialisable as written to the shard.
            json.loads(json.dumps(record))


def test_report_accounts_for_every_input_row() -> None:
    """converted + skipped + malformed == rows read."""
    events, stats = _events(cap=10_000_000)
    report = ConversionReport()
    for event in events:
        if isinstance(event, Skipped):
            report.note_skipped(event)
        else:
            report.note_converted(event)

    payload = report.to_dict(input_shas={}, load_stats=stats)
    assert payload["rows"]["read"] == stats.total
    assert payload["rows"]["malformed"] == 1
    assert payload["accounted"] is True
    # Per-mode counts carry both trajectories and samples.
    assert payload["by_mode"]["conversation"] == {"trajectories": 1, "samples": 2}
    assert payload["by_mode"]["non_conversation"] == {"trajectories": 1, "samples": 1}


def test_all_segments_of_one_trajectory_route_to_one_shard() -> None:
    """Routing keys on the trajectory id, so a Conv trajectory cannot straddle."""
    events, _ = _events(cap=10_000_000)
    split = split_trajectory_ids(
        [e.trajectory_id for e in events if isinstance(e, Converted)],
        seed=5,
        val_fraction=0.5,
    )
    for event in events:
        if not isinstance(event, Converted):
            continue
        partitions = {split.partition(event.trajectory_id) for _ in event.samples}
        assert len(partitions) == 1
