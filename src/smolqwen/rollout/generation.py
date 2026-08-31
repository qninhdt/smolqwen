"""Generation backends: one interface, two implementations, one honesty rule.

The scheduler speaks only to `GenerationBackend`. Which implementation it gets
decides whether a run is real:

- `VllmColocateBackend` drives TRL's colocated vLLM engine through the trainer.
  It is the only implementation that produces policy tokens, and it requires a
  GPU plus a trainer whose weights were synced — TRL calls `sync_weights` before
  `rollout_func` at every step boundary, so all generation within one call runs
  under one weight version. No mixed-policy episodes is a property of that
  boundary, not of anything here.
- `ScriptedPolicyBackend` replays a deterministic policy for tests and the
  equivalence gate. It exists because "the async path computes the same rewards
  as the baseline" is only provable when generation itself is deterministic;
  a sampled model makes both paths noisy and the comparison vacuous.

The backend returns *assistant tokens for one turn only*. Observations are
appended by the scheduler as messages and their tokens enter the completion
through the mask builder's re-render — never through generation, because vLLM
produces no logprob for text it did not sample. That is where the NaN contract
in `Episode.logprobs` comes from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# The scripted backend's logprob for a sampled token. Any constant works — the
# equivalence gate compares rewards, and TRL neutralizes masked positions — but
# it must not be NaN: the scripted policy's tokens are model tokens, and a NaN
# there would blur the "NaN means observation" contract the alignment test pins.
SCRIPTED_LOGPROB = -0.5


class GenerationError(RuntimeError):
    """Raised when a backend cannot produce tokens for a request."""


@dataclass(frozen=True)
class GenerationRequest:
    """One episode's need for one turn of assistant tokens.

    `prompt_ids` is the full rendered conversation prefix for this turn — the
    re-render, not an increment — because vLLM prefix caching and the drift
    classifier both want to see the same view.
    """

    episode_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(frozen=True)
class GenerationResult:
    """One turn's sampled assistant tokens, aligned 1:1 with `logprobs`."""

    episode_id: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    duration_s: float = 0.0
    prompt_tokens: int = 0


@dataclass(frozen=True)
class SamplingParams:
    """The trainer's sampling state, snapshotted per call by `rollout_func`."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024


class GenerationBackend(Protocol):
    """Generate one turn for each request in the sub-batch, as one call."""

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...


# A deterministic policy: called with the episode's current messages and turn
# index, returns the assistant text for that turn. The same (messages, turn)
# must always produce the same text — that is what makes the equivalence gate a
# gate rather than a coin flip.
ScriptedPolicy = Callable[[str, int, Sequence[Mapping[str, Any]]], str]


class ScriptedPolicyBackend:
    """Deterministic generation from a policy callable, for tests and the gate.

    The text is tokenized with the same tokenizer the mask builder re-renders
    with, so drift classification is exercised for real: a scripted turn whose
    re-render differs from its held tokens must still classify correctly.
    """

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for request in requests:
            context = self._contexts.get(request.episode_id)
            if context is None:
                raise GenerationError(f"{request.episode_id}: no scripted context bound")
            text = self._policy(request.episode_id, context.turn_index, context.messages)
            token_ids = tuple(self._encode(text))
            if len(token_ids) > request.max_new_tokens:
                token_ids = token_ids[: request.max_new_tokens]
            results.append(
                GenerationResult(
                    episode_id=request.episode_id,
                    token_ids=token_ids,
                    logprobs=tuple(SCRIPTED_LOGPROB for _ in token_ids),
                    prompt_tokens=len(request.prompt_ids),
                )
            )
            context.turn_index += 1
        return results

    @dataclass
    class _TurnContext:
        turn_index: int = 0
        messages: list[dict[str, Any]] = field(default_factory=list)

    def __init__(self, policy: ScriptedPolicy, encode: Callable[[str], list[int]]) -> None:
        self._policy = policy
        self._encode = encode
        self._contexts: dict[str, ScriptedPolicyBackend._TurnContext] = {}

    def bind(self, episode_id: str, messages: Sequence[Mapping[str, Any]]) -> None:
        """Register an episode the scheduler will drive through this backend.

        Accepts template dicts or `Message` records — whichever the scheduler
        holds at create time — and stores the template-dict form the policy
        reads.
        """
        normalized = [
            dict(message.to_template_dict())
            if hasattr(message, "to_template_dict")
            else dict(message)
            for message in messages
        ]
        self._contexts[episode_id] = ScriptedPolicyBackend._TurnContext(
            turn_index=0, messages=normalized
        )

    def observe(self, episode_id: str, message: Mapping[str, Any]) -> None:
        """Mirror a message the scheduler appended, so the policy sees history."""
        context = self._contexts.get(episode_id)
        if context is None:
            raise GenerationError(f"{episode_id}: no scripted context bound")
        context.messages.append(dict(message))


class VllmColocateBackend:
    """The production backend: TRL's colocated vLLM engine, one call per cycle.

    Constructed from the trainer inside `rollout_func` — never at config time —
    because the engine's existence is a trainer property (`use_vllm` plus
    colocate mode), and the KV budget (`vllm_max_model_len`), prefix caching and
    sleep mode were already set when the trainer was built. This backend adds
    no policy of its own; it forwards sampling params per request.
    """

    def __init__(self, trainer: Any) -> None:
        generation = getattr(trainer, "vllm_generation", None)
        if generation is None:
            raise GenerationError(
                "trainer has no colocated vllm_generation; build the trainer with "
                "use_vllm=True in colocate mode before using VllmColocateBackend"
            )
        self._trainer = trainer
        self._generation = generation

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        import time

        started = time.monotonic()
        from trl.extras.profiling import profiling_context

        # TRL's wrapper owns sampling params and returns four aligned arrays. It
        # accepts one max-completion budget for the sub-batch, so generate at the
        # largest request budget and truncate each row back to its own remaining
        # context room below.
        original_max = self._generation.max_completion_length
        self._generation.max_completion_length = max(request.max_new_tokens for request in requests)
        try:
            _, completion_rows, logprob_rows, logprob_token_rows = self._generation.generate(
                prompts=[list(request.prompt_ids) for request in requests],
                images=None,
                num_generations=1,
                profiler=profiling_context(self._trainer, "rollout.generate"),
            )
        finally:
            self._generation.max_completion_length = original_max
        results: list[GenerationResult] = []
        if logprob_rows is None:
            logprob_rows = [None] * len(completion_rows)
        if logprob_token_rows is None:
            logprob_token_rows = [None] * len(completion_rows)
        for request, token_ids, raw_logprobs, raw_logprob_ids in zip(
            requests,
            completion_rows,
            logprob_rows,
            logprob_token_rows,
            strict=True,
        ):
            token_ids = [int(token) for token in token_ids]
            logprobs = _sampling_logprobs(token_ids, raw_logprobs, raw_logprob_ids)
            if len(token_ids) > request.max_new_tokens:
                token_ids, logprobs = (
                    token_ids[: request.max_new_tokens],
                    logprobs[: request.max_new_tokens],
                )
            results.append(
                GenerationResult(
                    episode_id=request.episode_id,
                    token_ids=tuple(token_ids),
                    logprobs=tuple(logprobs),
                    duration_s=time.monotonic() - started,
                    prompt_tokens=len(request.prompt_ids),
                )
            )
        return results


def _sampling_logprobs(
    token_ids: Sequence[int],
    raw_logprobs: Sequence[Any] | None,
    raw_logprob_token_ids: Sequence[Any] | None,
) -> list[float]:
    """Select each sampled token's logprob from TRL's candidate arrays.

    `VLLMGeneration.generate` returns one candidate list per position plus the
    corresponding candidate token ids. A missing value remains NaN rather than
    shortening the row: TRL would right-pad a short row with 0.0 and shift all
    later importance-sampling positions.
    """
    rows = list(raw_logprobs or ())
    id_rows = list(raw_logprob_token_ids or ())
    selected: list[float] = []
    for position, sampled_id in enumerate(token_ids):
        candidates = rows[position] if position < len(rows) else None
        candidate_ids = id_rows[position] if position < len(id_rows) else None
        if candidates is None:
            selected.append(math.nan)
            continue
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            candidates = [candidates]
        if candidate_ids is not None and (
            not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, (str, bytes))
        ):
            candidate_ids = [candidate_ids]
        index = 0
        if candidate_ids is not None:
            normalized_ids = [int(candidate_id) for candidate_id in candidate_ids]
            try:
                index = normalized_ids.index(int(sampled_id))
            except ValueError:
                selected.append(math.nan)
                continue
        if index >= len(candidates) or candidates[index] is None:
            selected.append(math.nan)
            continue
        value = getattr(candidates[index], "logprob", candidates[index])
        assert value is not None
        selected.append(float(value))
    return selected
