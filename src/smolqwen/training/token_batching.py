"""Deterministic variable-row batches bounded by complete-trajectory tokens."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class TokenBatchingError(ValueError):
    """Raised when a complete row cannot fit the configured token envelope."""


class TokenBudgetBatchSampler:
    """Greedily pack whole rows under a fixed token budget.

    Rows are shuffled reproducibly per epoch, then locally sorted by length to
    reduce shape churn. No row is split, duplicated, or omitted.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_tokens: int,
        seed: int,
        shuffle: bool = True,
        bucket_size: int = 128,
    ) -> None:
        if max_tokens <= 0:
            raise TokenBatchingError("max_tokens must be positive")
        self.lengths = tuple(int(length) for length in lengths)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.bucket_size = max(1, int(bucket_size))
        self.epoch = 0
        self.cursor = 0
        over = [length for length in self.lengths if length <= 0 or length > self.max_tokens]
        if over:
            raise TokenBatchingError(
                f"row length {max(over)} cannot fit token budget {self.max_tokens}"
            )

    def _order(self) -> list[int]:
        order = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        bucketed: list[int] = []
        for start in range(0, len(order), self.bucket_size):
            bucket = order[start : start + self.bucket_size]
            bucketed.extend(sorted(bucket, key=self.lengths.__getitem__, reverse=True))
        return bucketed

    def batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        tokens = 0
        for index in self._order():
            length = self.lengths[index]
            if batch and tokens + length > self.max_tokens:
                batches.append(batch)
                batch = []
                tokens = 0
            batch.append(index)
            tokens += length
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.batches()
        while self.cursor < len(batches):
            batch = batches[self.cursor]
            self.cursor += 1
            yield batch
        self.cursor = 0
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.batches()) - self.cursor

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0 or cursor < 0:
            raise TokenBatchingError("sampler state cannot be negative")
        self.epoch = epoch
        batch_count = len(self.batches())
        if cursor > batch_count:
            raise TokenBatchingError(
                f"sampler cursor {cursor} exceeds epoch batch count {batch_count}"
            )
        self.cursor = cursor
