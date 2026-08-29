"""Collate Phase 2 records into padded batches with labels built from the loss mask.

The records carry `prompt_ids`, `completion_ids` and a `loss_mask` aligned to the
completion. Training needs one flat `input_ids` per sample plus `labels` where
every non-supervised position is `IGNORE_INDEX`. Two failure modes this module
exists to prevent:

- **A mask off by one token trains the model on tool output** and nothing reports
  it. So labels are derived positionally from the stored mask, never re-derived by
  searching for markers in decoded text.
- **Padding silently shifts the mask.** Padded positions must be ignored in the
  loss and masked in attention. Right-padding is used so the prompt/completion
  boundary stays at a fixed offset from position 0.

Causal-LM label shifting is the model's job (`transformers` shifts internally), so
`labels[i]` corresponds to `input_ids[i]` here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IGNORE_INDEX = -100


class CollateError(ValueError):
    """Raised when a record's mask does not line up with its completion."""


@dataclass(frozen=True)
class Batch:
    """A padded batch, framework-agnostic (lists of ints, not tensors).

    Kept as plain lists so the alignment tests run on CPU with no torch import;
    `to_torch` converts once, at the boundary.
    """

    input_ids: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    @property
    def seq_length(self) -> int:
        return len(self.input_ids[0]) if self.input_ids else 0

    def supervised_tokens(self) -> int:
        return sum(1 for row in self.labels for label in row if label != IGNORE_INDEX)

    def to_torch(self) -> Any:
        """Convert to a dict of tensors. Imported lazily so CPU tests stay torch-free."""
        import torch

        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
        }


def record_to_sequence(record: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Flatten one record into `(input_ids, labels)` before padding.

    The prompt is entirely masked: loss covers the current segment's assistant
    tokens only, which is exactly what `loss_mask` marks over the completion.
    """
    prompt_ids = list(record["prompt_ids"])
    completion_ids = list(record["completion_ids"])
    loss_mask = list(record["loss_mask"])

    if len(loss_mask) != len(completion_ids):
        raise CollateError(
            f"{record.get('trajectory_id', '?')}#{record.get('segment_index', '?')}: "
            f"loss_mask length {len(loss_mask)} != completion length {len(completion_ids)}"
        )
    if not any(loss_mask):
        raise CollateError(
            f"{record.get('trajectory_id', '?')}#{record.get('segment_index', '?')}: "
            "no supervised token; the record should never have been written"
        )

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids)
    labels += [
        token if flag else IGNORE_INDEX
        for token, flag in zip(completion_ids, loss_mask, strict=True)
    ]
    return input_ids, labels


def collate(
    records: Iterable[Mapping[str, Any]],
    *,
    pad_token_id: int,
    max_length: int | None = None,
) -> Batch:
    """Right-pad a batch of records, ignoring padded positions in the loss.

    `max_length` truncates from the right when given. Truncation here is a
    last-resort guard, not the budget mechanism: Phase 2 already skips over-cap
    trajectories whole, so a batch reaching this path with an over-length record
    means the cap and the shard disagree.
    """
    sequences = [record_to_sequence(record) for record in records]
    if not sequences:
        return Batch((), (), ())

    if max_length is not None:
        sequences = [(ids[:max_length], labels[:max_length]) for ids, labels in sequences]

    width = max(len(ids) for ids, _ in sequences)
    input_rows: list[tuple[int, ...]] = []
    label_rows: list[tuple[int, ...]] = []
    mask_rows: list[tuple[int, ...]] = []
    for ids, labels in sequences:
        pad = width - len(ids)
        input_rows.append(tuple(ids + [pad_token_id] * pad))
        # Padded positions carry IGNORE_INDEX so they contribute no loss, and 0 in
        # the attention mask so they are not attended to.
        label_rows.append(tuple(labels + [IGNORE_INDEX] * pad))
        mask_rows.append(tuple([1] * len(ids) + [0] * pad))

    return Batch(tuple(input_rows), tuple(label_rows), tuple(mask_rows))


def collator(pad_token_id: int, *, max_length: int | None = None) -> Any:
    """A `DataCollator`-shaped callable for `Trainer`, returning tensors."""

    def call(features: Sequence[Mapping[str, Any]]) -> Any:
        return collate(features, pad_token_id=pad_token_id, max_length=max_length).to_torch()

    return call
