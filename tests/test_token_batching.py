"""Token-budget sampling and padding-free boundary metadata."""

from __future__ import annotations

import pytest

from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS
from smolqwen.training.collate import CollateError, padding_free_collator
from smolqwen.training.token_batching import TokenBatchingError, TokenBudgetBatchSampler


def _record(uid: str, length: int) -> dict[str, object]:
    input_ids = list(range(1, length + 1))
    labels = [-100] + input_ids[1:]
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "semantics": SFT_SEMANTICS,
        "trajectory_uid": uid,
        "task_id": uid,
        "input_ids": input_ids,
        "labels": labels,
        "seq_length": length,
        "supervised_tokens": length - 1,
    }


def test_batches_visit_every_row_once_without_crossing_budget() -> None:
    lengths = [7, 3, 5, 2, 9, 4]
    sampler = TokenBudgetBatchSampler(lengths, max_tokens=12, seed=7)
    batches = sampler.batches()
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in batch) <= 12 for batch in batches)
    assert len({len(batch) for batch in batches}) > 1


def test_sampler_is_seeded_and_resume_cursor_is_exact() -> None:
    lengths = [index % 7 + 1 for index in range(30)]
    uninterrupted = TokenBudgetBatchSampler(lengths, max_tokens=16, seed=11)
    iterator = iter(uninterrupted)
    consumed = [next(iterator), next(iterator)]
    state = uninterrupted.state_dict()
    expected_remaining = list(iterator)

    resumed = TokenBudgetBatchSampler(lengths, max_tokens=16, seed=11)
    resumed.load_state_dict(state)
    assert list(resumed) == expected_remaining
    assert (
        consumed + expected_remaining
        == TokenBudgetBatchSampler(lengths, max_tokens=16, seed=11).batches()
    )


def test_row_over_budget_fails_instead_of_splitting() -> None:
    with pytest.raises(TokenBatchingError, match="cannot fit"):
        TokenBudgetBatchSampler([5, 17], max_tokens=16, seed=1)


def test_padding_free_collator_flattens_and_reconstructs_boundaries() -> None:
    records = [_record("a", 3), _record("b", 5), _record("c", 2)]
    batch = padding_free_collator(10)(records)
    assert batch["input_ids"].shape == (1, 10)
    assert batch["labels"].shape == (1, 10)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 8, 10]
    assert batch["cu_seq_lens_k"].tolist() == [0, 3, 8, 10]
    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]]
    assert batch["max_length_q"] == batch["max_length_k"] == 5


def test_padding_free_collator_rejects_batch_above_budget() -> None:
    with pytest.raises(CollateError, match="above token budget"):
        padding_free_collator(7)([_record("a", 4), _record("b", 4)])
