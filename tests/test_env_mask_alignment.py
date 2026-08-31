"""The mask is 1 exactly on model tokens — verified differentially, not by fixture.

The test builds the expected spans through an *independent* construction: the
same incremental-render diff `data/render.py` uses for SFT samples (render the
conversation after each message, require the previous render to be a literal
prefix, attribute the delta to the message that produced it). `mask.py`
classifies drift instead. The two agree only if the rollout path really placed
the boundary where the template places it — a fixture built by the same
concatenation rule as the code would pass while the real mask walks off the
boundary, which is the failure this file exists to catch.

Also pinned here, because nothing downstream would raise on them:

- `len(logprobs) == len(completion_ids) == len(env_mask)`, with NaN exactly on
  unsampled positions — a shorter array is right-padded with 0.0 (probability
  1) by TRL and shifts every later model token against the wrong position;
- supervised spans decode back to the policy's exact turn texts;
- every transition classified `clean` — the rollout message shape keeps
  renders incremental, so a non-zero drift tally is a re-render instability
  that must be seen, not absorbed.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from smolqwen.data.render import render_prefix
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, make_scheduler
from tests.helpers import OfflineTokenizer
from tests.rollout_fixtures import (
    FakeDispatcher,
    TimedBackend,
    VirtualClock,
    fast_config,
    fixture_bindings,
    script_policy_texts,
    text_list_policy,
)


def _tokenizer() -> OfflineTokenizer:
    """Prefix-stable offline tokenizer: per-character ids.

    With 64-char chunks the id grid restarts at every encode call, so
    `encode(prefix + suffix)` is not `encode(prefix) + encode(suffix)` even when
    the text is a literal prefix — systematic seam drift that is an artifact of
    the test tokenizer, not of the template. Per-character ids make encoding
    prefix-stable by construction, which is the property the mask's CLEAN path
    asserts. Real-BPE seam drift is a runtime phenomenon the classifier exists
    to absorb; the builder unit tests cover REALIGN/FORK directly.
    """
    return OfflineTokenizer(token_size=1)


def _run_episode(
    *, step_texts: list[str] | None, config: Any = None
) -> tuple[Any, Any, OfflineTokenizer]:
    """Drive one episode through the real scheduler seams, all fakes virtual."""
    clock = VirtualClock()
    dispatcher = FakeDispatcher(clock)
    tokenizer = _tokenizer()
    texts = script_policy_texts() if step_texts is None else step_texts
    backend = TimedBackend(
        ScriptedPolicyBackend(text_list_policy(texts), lambda text: encode_ids(tokenizer, text)),
        clock,
    )
    scheduler = make_scheduler(
        backend=backend,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        config=config or fast_config(),
        wait_for=dispatcher.wait,
    )
    bindings = fixture_bindings(episodes=1)
    episodes = scheduler.run(bindings)
    return episodes[0], scheduler, tokenizer


def _expected_spans_via_render_diff(
    tokenizer: OfflineTokenizer,
    messages: list[Any],
    binding: Any,
    generated_turns: list[str],
) -> list[tuple[int, int, bool]]:
    """Independent span construction, measured in token positions.

    Starts from the same initial prompt the scheduler seeds its builder with.
    For every assistant turn, independently render the semantic message and
    attribute its delta to the model; then render the following tool result plus
    next generation prompt and attribute that delta to the environment. This is
    the actual inference boundary. Rendering after *every* message with a new
    generation prompt would append a second assistant prompt after an assistant
    and then remove it when the tool result arrives, creating artificial drift.
    """
    tools = list(binding.tool_schemas)
    spans: list[tuple[int, int, bool]] = []
    previous_text = render_prefix(tokenizer, messages[:2], tools=tools, add_generation_prompt=True)
    previous_ids = encode_ids(tokenizer, previous_text)
    index = 2
    turn_index = 0
    while index < len(messages):
        assert messages[index].role == "assistant"
        semantic_text = render_prefix(
            tokenizer, messages[: index + 1], tools=tools, add_generation_prompt=False
        )
        generated = generated_turns[turn_index]
        held_text = previous_text + generated
        assert semantic_text.startswith(held_text), (
            "the semantic assistant render does not preserve the exact sampled "
            "continuation after the generation prompt"
        )
        held_ids = previous_ids + encode_ids(tokenizer, generated)
        assert encode_ids(tokenizer, held_text) == held_ids
        spans.append((len(previous_ids), len(held_ids), True))
        previous_text, previous_ids = held_text, held_ids
        turn_index += 1
        index += 1
        if index < len(messages) and messages[index].role == "tool":
            current_text = render_prefix(
                tokenizer, messages[: index + 1], tools=tools, add_generation_prompt=True
            )
            assert current_text.startswith(previous_text)
            current_ids = encode_ids(tokenizer, current_text)
            assert current_ids[: len(previous_ids)] == previous_ids
            spans.append((len(previous_ids), len(current_ids), False))
            previous_text, previous_ids = current_text, current_ids
            index += 1
    return spans


def test_mask_marks_exactly_the_model_tokens_and_nan_marks_observations() -> None:
    episode, scheduler, _ = _run_episode(step_texts=None)  # 3 calls + final answer

    builder = scheduler.episode_builder(episode.episode_id)
    completion = list(builder.completion_ids)
    logprobs = list(builder.logprobs)
    mask = list(builder.env_mask)

    assert len(logprobs) == len(completion) == len(mask)
    assert sum(mask) > 0, "no supervised token; the episode never generated"
    # Supervised positions carry the sampler's logprob; every masked position
    # is NaN — the value TRL maps to ratio exactly 1.
    for flag, logprob in zip(mask, logprobs, strict=True):
        if flag:
            assert not math.isnan(logprob)
        else:
            assert math.isnan(logprob)

    # At least one observation is present, and none of them is supervised.
    assert len(episode.observations) == 3
    assert sum(mask) < len(mask)


def test_mask_boundaries_agree_with_an_independent_render_diff() -> None:
    episode, scheduler, tokenizer = _run_episode(step_texts=None)

    builder = scheduler.episode_builder(episode.episode_id)
    binding = fixture_bindings(episodes=1)[0]

    expected = _expected_spans_via_render_diff(
        tokenizer, episode.messages, binding, script_policy_texts()
    )
    boundary = len(
        encode_ids(
            tokenizer,
            render_prefix(
                tokenizer,
                episode.messages[:2],
                tools=list(binding.tool_schemas),
                add_generation_prompt=True,
            ),
        )
    )
    assert builder.boundary == boundary

    actual = [(span.start, span.end, span.supervised) for span in builder.spans]
    actual_completion = [span for span in actual if span[1] > boundary]
    assert actual_completion == expected, "mask spans disagree with the template's own render"


def test_supervised_tokens_decode_back_to_the_policys_exact_texts() -> None:
    texts = script_policy_texts()
    episode, scheduler, tokenizer = _run_episode(step_texts=None)

    builder = scheduler.episode_builder(episode.episode_id)
    supervised = [
        token for token, flag in zip(builder.completion_ids, builder.env_mask, strict=True) if flag
    ]
    # The scripted backend encodes each turn text independently, and the
    # tokenizer decodes chunk-by-chunk, so the supervised span is exactly the
    # policy's turns joined — byte for byte, not approximately.
    assert tokenizer.decode(list(supervised)) == "".join(texts)


def test_transitions_are_clean_under_the_committed_message_shape() -> None:
    episode, _, _ = _run_episode(step_texts=None)
    tally = episode.drift_tally
    assert tally.transitions == 4  # one per generation after the first
    assert tally.clean == 4
    assert tally.realign == 0
    assert tally.fork == 0


def test_a_truncated_drift_realigns_the_mask_not_just_the_tokens() -> None:
    """A REALIGN demotes the drifted tail to context and NaN logprobs."""
    from smolqwen.rollout.mask import EpisodeMaskBuilder

    builder = EpisodeMaskBuilder([1, 2, 3, 4])
    builder.append_response([5, 6, 7, 8], [-1.0, -1.1, -1.2, -1.3])
    # The next render disagrees on the last two sampled tokens and appends one
    # observation token.
    kind = builder.open_turn([1, 2, 3, 4, 5, 6, 99, 77])
    assert kind == "realign"
    assert builder.tally.realign == 1

    completion = list(builder.completion_ids)
    mask = list(builder.env_mask)
    logprobs = list(builder.logprobs)
    assert len(logprobs) == len(completion) == len(mask)
    # Tokens 99 and 77 replaced the drifted tail as context: masked, NaN.
    assert mask[-2:] == [0, 0]
    assert math.isnan(logprobs[-2]) and math.isnan(logprobs[-1])
    assert mask[:2] == [1, 1]  # the surviving response head stays supervised


def test_a_fork_is_forced_to_realign_and_counted_as_fork() -> None:
    from smolqwen.rollout.mask import EpisodeMaskBuilder

    builder = EpisodeMaskBuilder([1, 2, 3], fork_threshold_tokens=2)
    builder.append_response([4, 5], [-1.0, -1.0])
    kind = builder.open_turn([1, 2, 3, 9, 9, 9, 9])
    assert kind == "fork"
    assert builder.tally.fork == 1
    # Still one row: held tokens equal the fresh render, tail masked.
    assert list(builder.completion_ids) == [9, 9, 9, 9]
    assert list(builder.env_mask) == [0, 0, 0, 0]


def test_assemble_output_rejects_a_mask_that_would_train_on_observations() -> None:
    """The boundary asserts all-ones masks and missing NaNs before TRL sees them."""
    from smolqwen.rollout.episode import Episode
    from smolqwen.rollout.rollout_func import RolloutFuncError, assemble_output

    episode = Episode(episode_id="e", scenario_id="s", group_index=0)
    episode.completion_ids = [1, 2, 3]
    episode.observations = ["an observation"]

    class _AllOnesBuilder:
        prompt_ids = (0,)
        completion_ids = (1, 2, 3)
        logprobs = (0.0, math.nan, 0.0)
        env_mask = (1, 1, 1)  # all ones despite an observation

    class _FakeScheduler:
        builder: Any = _AllOnesBuilder()

        def episode_builder(self, episode_id: str) -> Any:
            return self.builder

    scheduler = _FakeScheduler()
    with pytest.raises(RolloutFuncError, match="env_mask is all ones"):
        assemble_output([episode], scheduler)  # type: ignore[arg-type]

    class _MissingNanBuilder:
        prompt_ids = (0,)
        completion_ids = (1, 2, 3)
        logprobs = (0.0, 0.0, 0.0)
        env_mask = (1, 0, 1)

    scheduler.builder = _MissingNanBuilder()
    with pytest.raises(RolloutFuncError, match="no NaN logprob"):
        assemble_output([episode], scheduler)  # type: ignore[arg-type]
