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


def test_sampler_epoch_hook_preserves_same_epoch_and_resets_next_epoch() -> None:
    sampler = TokenBudgetBatchSampler(list(range(1, 9)), max_tokens=8, seed=11)
    iterator = iter(sampler)
    next(iterator)
    assert sampler.cursor == 1

    sampler.set_epoch(0)
    assert sampler.cursor == 1
    sampler.set_epoch(1)
    assert sampler.cursor == 0


def test_accelerate_resume_skips_batches_once_in_the_same_epoch() -> None:
    from accelerate.data_loader import skip_first_batches
    from torch.utils.data import DataLoader, Dataset

    class IntegerDataset(Dataset[int]):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> int:
            return index

    def keep_rows(rows: list[int]) -> list[int]:
        return rows

    sampler = TokenBudgetBatchSampler([4, 4, 4, 4, 4], max_tokens=8, seed=3, shuffle=False)
    dataloader = DataLoader(IntegerDataset(), batch_sampler=sampler, collate_fn=keep_rows)
    resumed = skip_first_batches(dataloader, 1)

    assert next(iter(resumed)) == [2, 3]
    assert sampler.state_dict() == {"epoch": 0, "cursor": 2}


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


def test_padding_free_masks_each_document_start_for_global_causal_shift() -> None:
    first = _record("a", 3)
    second = _record("b", 2)
    first["labels"] = first["input_ids"]
    second["labels"] = second["input_ids"]
    first["supervised_tokens"] = 3
    second["supervised_tokens"] = 2
    batch = padding_free_collator(10)([first, second])

    assert batch["labels"].tolist() == [[-100, 2, 3, -100, 2]]


def test_flattened_loss_uses_only_within_document_targets() -> None:
    import torch
    from transformers.loss.loss_utils import ForCausalLMLoss

    batch = padding_free_collator(10)([_record("a", 3), _record("b", 2)])
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 16)
    item_count = (batch["labels"][..., 1:] != -100).sum()
    expected = (
        torch.nn.functional.cross_entropy(
            logits[..., :-1, :].float().reshape(-1, 16),
            batch["labels"][..., 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / item_count
    )

    actual = ForCausalLMLoss(
        logits,
        batch["labels"],
        vocab_size=16,
        num_items_in_batch=item_count,
    )
    assert int(item_count) == 3
    assert actual == pytest.approx(expected)


def test_padding_free_rejects_a_row_above_the_sequence_cap() -> None:
    with pytest.raises(CollateError, match="max_sequence_length"):
        padding_free_collator(20, max_sequence_length=4)([_record("a", 5)])


def test_padding_free_collator_rejects_batch_above_budget() -> None:
    with pytest.raises(CollateError, match="above token budget"):
        padding_free_collator(7)([_record("a", 4), _record("b", 4)])
