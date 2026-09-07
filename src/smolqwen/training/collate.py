"""Validate full-trajectory records and build the padded and padding-free batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS_TAGS

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
    """Validate and return the already aligned ids/labels from schema v2."""
    schema_matches = record.get("schema_version") == SFT_SCHEMA_VERSION
    semantics_match = record.get("semantics") in SFT_SEMANTICS_TAGS
    if not schema_matches or not semantics_match:
        raise CollateError(
            "incompatible SFT shard; regenerate full-trajectory schema v2 with "
            "`smolqwen prepare-sft`"
        )
    input_ids = [int(token) for token in record["input_ids"]]
    labels = [int(label) for label in record["labels"]]
    uid = record.get("trajectory_uid", "?")
    if len(input_ids) != len(labels):
        raise CollateError(f"{uid}: labels length {len(labels)} != input length {len(input_ids)}")
    if record.get("seq_length") != len(input_ids):
        raise CollateError(f"{uid}: seq_length does not match input_ids")
    if not any(label != IGNORE_INDEX for label in labels):
        raise CollateError(f"{uid}: no supervised token; the record should never have been written")
    for token, label in zip(input_ids, labels, strict=True):
        if label not in (IGNORE_INDEX, token):
            raise CollateError(f"{uid}: label must be -100 or equal its input token")
    if record.get("supervised_tokens") != sum(label != IGNORE_INDEX for label in labels):
        raise CollateError(f"{uid}: supervised_tokens does not match labels")
    return input_ids, labels


def collate(
    records: Iterable[Mapping[str, Any]],
    *,
    pad_token_id: int,
    max_length: int | None = None,
) -> Batch:
    """Right-pad a batch of records, ignoring padded positions in the loss.

    Over-cap records are rejected rather than truncated; conversion owns the cap.
    """
    sequences = [record_to_sequence(record) for record in records]
    if not sequences:
        return Batch((), (), ())

    if max_length is not None:
        over = [len(ids) for ids, _ in sequences if len(ids) > max_length]
        if over:
            raise CollateError(
                f"record length {max(over)} exceeds configured max_length {max_length}; "
                "regenerate shards with the matching cap"
            )

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
    """A `DataCollator`-shaped callable for `Trainer`, returning tensors.

    The padded path an `sdpa` run takes: right-padded rows plus the attention mask
    that tells the kernel which positions are real. `sdpa` reads a mask natively,
    so nothing here has to be omitted or reshaped for it.
    """

    def call(features: Sequence[Mapping[str, Any]]) -> Any:
        return collate(features, pad_token_id=pad_token_id, max_length=max_length).to_torch()

    return call


def padding_free_collator(max_tokens: int, *, max_sequence_length: int | None = None) -> Any:
    """Flatten complete rows and emit boundaries for FA2, GDN, and causal conv."""

    def call(features: Sequence[Mapping[str, Any]]) -> Any:
        import torch

        sequences = [record_to_sequence(feature) for feature in features]
        if not sequences:
            raise CollateError("padding-free collator received an empty batch")
        lengths = [len(input_ids) for input_ids, _ in sequences]
        if max_sequence_length is not None and max(lengths) > max_sequence_length:
            raise CollateError(
                f"trajectory length {max(lengths)} exceeds max_sequence_length "
                f"{max_sequence_length}; regenerate shards with the matching cap"
            )
        total = sum(lengths)
        if total > max_tokens:
            raise CollateError(f"batch has {total} tokens, above token budget {max_tokens}")
        input_ids = [token for row, _ in sequences for token in row]
        # A causal LM shifts one flattened label array globally.  Masking the
        # first label of every document prevents the first token of document B
        # from being trained against the last logit of document A.  This is the
        # same boundary rule as padded execution, where each row is shifted
        # independently.
        labels = [label for _, row in sequences for label in ([IGNORE_INDEX, *row[1:]])]
        position_ids = [position for length in lengths for position in range(length)]
        sequence_ids = [index for index, length in enumerate(lengths) for _ in range(length)]
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        max_length = max(lengths)
        cu_seqlens = torch.tensor(cumulative, dtype=torch.int32)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "position_ids": torch.tensor([position_ids], dtype=torch.long),
            # FlashAttention and FLA use cumulative boundaries; causal-conv1d
            # uses seq_idx to reset its depthwise convolution state.
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_length,
            "max_length_k": max_length,
            "seq_idx": torch.tensor([sequence_ids], dtype=torch.int32),
        }

    return call


def supervised_positions(labels: Any) -> Any:
    """Positions whose *next-token prediction* is supervised, in the shifted frame.

    A causal head at position `i` predicts `labels[i + 1]`, so the rows the head
    must project are one left of the supervised labels -- and the last position is
    never a prediction target. Returns `(index, shift_labels)` where `index` selects
    those rows out of the hidden states and `shift_labels` is the aligned target for
    exactly those rows.

    This is the whole of "selective": the fused head then sees `index.numel()` rows
    instead of `T`. Everything downstream (loss value, `num_items_in_batch`
    normalization, gradient) is unchanged, because dropping a row whose target is
    `-100` removes a term that contributed zero.
    """
    import torch

    shifted = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    if shifted.dim() != 2 or shifted.shape[0] != 1:
        raise CollateError(
            f"selective logits need one flattened row, got shape {tuple(shifted.shape)}; "
            "a shared position index cannot describe per-row supervision"
        )
    index = shifted[0].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    return index, shifted[:, index]
