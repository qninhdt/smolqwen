"""Mask construction by drift classification, ported from upstream's semantics.

Each turn re-renders the whole conversation through the chat template, and the
re-tokenized prefix can differ from the tokens already held — a seam-merge in
BPE tokenization, or a rewritten turn wherever the template strips reasoning.
Span arithmetic that assumes concatenation is stable will place the boundary
wrong exactly there, so instead every turn's re-rendered prefix is compared
against the accumulated tokens and classified, porting
`_SampleBuilder.classify_token_drift` from `trl.experimental.async_grpo`:

- `CLEAN` — held tokens are a prefix of the new render; append the new tail
  (the observation) as masked context.
- `REALIGN` — drift is confined to the last response and under the fork
  threshold; overwrite the drifted tail as context (mask 0).
- `FORK` — bigger drift. Upstream starts a new training row here because its
  rollout buffer is row-per-sequence. This phase's contract is one returned row
  per input prompt, so a FORK degrades to a forced realign *and is counted as a
  fork in the tally*: the metric stays honest about how often the one-row
  assumption was stressed, which is the only way the threshold stays tunable.

Two deliberate departures from upstream, both load-bearing:

1. **Observation tokens carry NaN logprobs, not 0.0.** Upstream writes 0.0 into
   realigned/context positions; under TRL's importance-sampling correction a
   returned 0.0 is probability 1, and a wrong-length array gets right-padded
   with it. NaN is the value TRL maps to a zero difference — ratio exactly 1 —
   so "NaN marks a position the sampler never saw" holds in the returned arrays
   themselves, not just in TRL's downstream handling.
2. **The prompt is seeded into the builder**, so the accumulated sequence is
   the full conversation exactly as the final render sees it. `prompt_ids` and
   `completion_ids` split at the builder's boundary, which a prompt-crossing
   FORK can move — the split follows the fresh render, never the stale one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smolqwen.rollout.episode import DriftKind, DriftTally, MaskSpan

FORK_THRESHOLD_TOKENS = 1024  # upstream `fork_threshold_tokens` default


def common_prefix_length(held: Sequence[int], rendered: Sequence[int]) -> int:
    """How many leading tokens the two sequences share."""
    limit = min(len(held), len(rendered))
    index = 0
    while index < limit and held[index] == rendered[index]:
        index += 1
    return index


class EpisodeMaskBuilder:
    """One episode's accumulated conversation, split into prompt and completion.

    Seeded with the initial prompt's token ids. The per-turn protocol, matching
    the scheduler's cycle:

    1. `open_turn(rendered_prefix_ids)` — classify this turn's *full* re-rendered
       conversation (generation prompt on) against the held tokens, then fold
       the new tail in as masked context.
    2. `append_response(token_ids, logprobs)` — this turn's sampled assistant
       tokens: the supervised span, carrying the sampler's logprobs.

    After every `open_turn`, the held sequence equals the fresh render. The
    returned `prompt_ids`/`completion_ids` therefore satisfy the invariant the
    differential alignment test pins: concatenating them reproduces the
    template's own render of the final conversation.
    """

    def __init__(
        self,
        prompt_ids: Sequence[int],
        *,
        fork_threshold_tokens: int = FORK_THRESHOLD_TOKENS,
    ) -> None:
        self._tokens: list[int] = list(prompt_ids)
        self._logprobs: list[float] = [math.nan] * len(prompt_ids)
        self._spans: list[MaskSpan] = [MaskSpan(start=0, end=len(prompt_ids), supervised=False)]
        self._boundary = len(prompt_ids)
        self._response_start: int | None = None
        self._fork_threshold = fork_threshold_tokens
        self.tally = DriftTally()

    # --- write side ---

    def open_turn(self, rendered_prefix_ids: Sequence[int]) -> DriftKind:
        matched = common_prefix_length(self._tokens, rendered_prefix_ids)
        drift = len(self._tokens) - matched

        if drift == 0:
            self.tally.observe("clean", 0)
            kind: DriftKind = "clean"
        else:
            start = self._response_start
            if start is not None and matched >= start and drift < self._fork_threshold:
                kind = "realign"
            else:
                kind = "fork"
            self.tally.observe(kind, drift)
            self._truncate_to(matched)
            if matched < self._boundary:
                # A FORK that reaches into the prompt: the prompt itself was
                # re-rendered differently, so the boundary follows the fresh
                # render. Returning the stale prompt would misalign every
                # logprob TRL recomputes against it.
                self._boundary = matched

        self._append_context(list(rendered_prefix_ids[matched:]))
        return kind

    def append_response(self, token_ids: Sequence[int], logprobs: Sequence[float] | None) -> None:
        if logprobs is not None and len(token_ids) != len(logprobs):
            raise ValueError(f"response has {len(token_ids)} tokens but {len(logprobs)} logprobs")
        self._response_start = len(self._tokens)
        values = list(logprobs) if logprobs is not None else [math.nan] * len(token_ids)
        self._spans.append(
            MaskSpan(
                start=len(self._tokens),
                end=len(self._tokens) + len(token_ids),
                supervised=True,
            )
        )
        self._tokens.extend(token_ids)
        self._logprobs.extend(float(value) for value in values)

    def _append_context(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self._spans.append(
            MaskSpan(
                start=len(self._tokens),
                end=len(self._tokens) + len(token_ids),
                supervised=False,
            )
        )
        # Context positions (observations, realigned tails) were never sampled.
        # NaN, not 0.0 — see the module docstring.
        self._tokens.extend(token_ids)
        self._logprobs.extend(math.nan for _ in token_ids)

    def _truncate_to(self, length: int) -> None:
        del self._tokens[length:]
        del self._logprobs[length:]
        kept: list[MaskSpan] = []
        for span in self._spans:
            if span.end <= length:
                kept.append(span)
            elif span.start < length:
                kept.append(MaskSpan(span.start, length, span.supervised))
        self._spans = kept

    # --- read side ---

    @property
    def prompt_ids(self) -> tuple[int, ...]:
        return tuple(self._tokens[: self._boundary])

    @property
    def completion_ids(self) -> tuple[int, ...]:
        return tuple(self._tokens[self._boundary :])

    @property
    def logprobs(self) -> tuple[float, ...]:
        """Same length as `completion_ids`, NaN at every unsampled position."""
        return tuple(self._logprobs[self._boundary :])

    @property
    def env_mask(self) -> tuple[int, ...]:
        """1 on sampled assistant tokens, 0 on prompt-side and context tokens."""
        return tuple(
            1 if span.supervised else 0
            for span in self._spans
            if span.end > self._boundary
            for _ in range(span.end - max(span.start, self._boundary))
        )

    @property
    def boundary(self) -> int:
        return self._boundary

    @property
    def spans(self) -> tuple[MaskSpan, ...]:
        """Every accumulated span, including the prompt's. Tests inspect this."""
        return tuple(self._spans)
