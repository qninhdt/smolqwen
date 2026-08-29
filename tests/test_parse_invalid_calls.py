"""Invalid tool calls are classified, not collapsed — and never raised.

A rollout has to react differently to four failures, and one "invalid" bucket
loses the information Phase 7 reports:

- `unknown_tool` means the model invented a name, which for an SFT checkpoint that
  trained cleanly points at a tool-schema mismatch between train and rollout;
- `bad_arguments` means it picked the right tool and got the call wrong, which is a
  reasoning problem;
- `malformed_syntax` usually means generation hit the token cap mid-call, which is
  a `max_new_tokens_per_step` problem;
- `no_call` is not a failure at all in a Conv episode — it is a message to the user.

Nothing here raises: an invalid call becomes an observation the model reads and
retries from, matching how the released trajectories behave. Invalid-call rates are
reported in Phase 7 and never enter the reward, because a model penalised for
malformed syntax learns to stop emitting tool calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from smolqwen.env.instance import EnvInstance
from smolqwen.env.parse import parse_turn, split_reasoning
from smolqwen.env.registry import EnvRegistry
from smolqwen.env.scenarios import load_scenarios

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
SCENARIOS = FIXTURES / "scenarios.json"

WELL_FORMED = (
    "<think>\nCheck the trial first.\n</think>\n\n"
    "<tool_call>\n<function=get_clinical_trial_by_id>\n"
    "<parameter=trial_id>\nCT-101\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def instance() -> EnvInstance:
    registry = EnvRegistry.from_metadata(ENV_METADATA)
    scenario = load_scenarios(SCENARIOS)[0]
    return EnvInstance.create(
        env_id=scenario.env_id,
        env_class=registry.env_class(scenario.env_id),
        env_class_name=scenario.env_class_name,
        init_config=scenario.init_config,
        tools=registry.tools(scenario.env_id),
    )


def _parse(text: str, instance: EnvInstance) -> Any:
    return parse_turn(
        text,
        available_tools=instance.tool_names(),
        signature_lookup=lambda name: getattr(instance.instance, name, None),
    )


def test_a_well_formed_call_parses_with_its_reasoning(instance: EnvInstance) -> None:
    turn = _parse(WELL_FORMED, instance)
    assert turn.outcome == "ok"
    assert turn.name == "get_clinical_trial_by_id"
    assert turn.arguments == {"trial_id": "CT-101"}
    assert turn.reasoning == "Check the trial first."
    assert not turn.is_invalid_call


def test_an_unknown_tool_is_classified_as_such(instance: EnvInstance) -> None:
    text = (
        "<tool_call>\n<function=delete_the_database>\n"
        "<parameter=confirm>\nyes\n</parameter>\n</function>\n</tool_call>"
    )
    turn = _parse(text, instance)
    assert turn.outcome == "unknown_tool"
    assert turn.name == "delete_the_database"
    assert turn.is_invalid_call
    # The observation names the tool, so the model can pick a real one next step.
    assert "delete_the_database" in turn.observation()


def test_bad_arguments_are_distinguished_from_an_unknown_tool(instance: EnvInstance) -> None:
    text = (
        "<tool_call>\n<function=update_clinical_trial_status>\n"
        "<parameter=trial_id>\nCT-101\n</parameter>\n"
        "<parameter=state>\ncompleted\n</parameter>\n</function>\n</tool_call>"
    )
    turn = _parse(text, instance)
    assert turn.outcome == "bad_arguments"
    assert turn.name == "update_clinical_trial_status"
    assert "state" in (turn.reason or "")
    assert turn.is_invalid_call


def test_a_missing_required_argument_is_bad_arguments(instance: EnvInstance) -> None:
    text = (
        "<tool_call>\n<function=update_clinical_trial_status>\n"
        "<parameter=trial_id>\nCT-101\n</parameter>\n</function>\n</tool_call>"
    )
    turn = _parse(text, instance)
    assert turn.outcome == "bad_arguments"
    assert "new_status" in (turn.reason or "")


def test_an_unclosed_tool_call_tag_is_malformed_syntax(instance: EnvInstance) -> None:
    """What a call truncated by the per-step token cap looks like."""
    text = "<tool_call>\n<function=get_clinical_trial_by_id>\n<parameter=trial_id>\nCT-101"
    turn = _parse(text, instance)
    assert turn.outcome == "malformed_syntax"
    assert "unclosed" in (turn.reason or "")


def test_an_unclosed_function_tag_is_malformed_syntax(instance: EnvInstance) -> None:
    text = "<tool_call>\n<function=get_clinical_trial_by_id>\n</tool_call>"
    turn = _parse(text, instance)
    assert turn.outcome == "malformed_syntax"


def test_a_tool_call_block_with_no_function_is_malformed(instance: EnvInstance) -> None:
    turn = _parse('<tool_call>\n{"name": "x"}\n</tool_call>', instance)
    assert turn.outcome == "malformed_syntax"
    assert "no <function=" in (turn.reason or "")


def test_an_empty_function_name_is_malformed_not_no_call(instance: EnvInstance) -> None:
    turn = _parse("<tool_call><function=></function></tool_call>", instance)
    assert turn.outcome == "malformed_syntax"
    assert turn.reason == "empty function name"


def test_plain_text_is_no_call_and_not_an_invalid_call(instance: EnvInstance) -> None:
    """In a Conv episode this is a message to the user, not a failure."""
    turn = _parse("<think>\nAsk them.\n</think>\n\nWhich trial did you mean?", instance)
    assert turn.outcome == "no_call"
    assert turn.content == "Which trial did you mean?"
    assert turn.reasoning == "Ask them."
    assert not turn.is_invalid_call


def test_content_outside_the_call_is_kept(instance: EnvInstance) -> None:
    turn = _parse("Let me look that up.\n" + WELL_FORMED, instance)
    assert turn.outcome == "ok"
    assert turn.content == "Let me look that up."


def test_observation_refuses_to_describe_a_clean_turn(instance: EnvInstance) -> None:
    turn = _parse(WELL_FORMED, instance)
    with pytest.raises(ValueError, match="for invalid calls"):
        turn.observation()


def test_an_unclosed_think_block_keeps_the_reasoning() -> None:
    """Truncated generation: the reasoning is still usable, the remainder is not."""
    reasoning, remainder = split_reasoning("<think>\nI was cut off mid-thought")
    assert reasoning == "I was cut off mid-thought"
    assert remainder == ""


def test_a_turn_with_no_think_block_is_all_content() -> None:
    reasoning, remainder = split_reasoning("Just an answer.")
    assert reasoning == ""
    assert remainder == "Just an answer."


def test_the_first_call_wins_when_several_are_emitted(instance: EnvInstance) -> None:
    """Matches upstream and the system prompt's "no parallel calls" instruction."""
    text = WELL_FORMED + (
        "\n<tool_call>\n<function=get_enrollment_status>\n"
        "<parameter=enrollment_id>\nE1\n</parameter>\n</function>\n</tool_call>"
    )
    turn = _parse(text, instance)
    assert turn.outcome == "ok"
    assert turn.name == "get_clinical_trial_by_id"


def test_parsing_without_a_tool_list_never_reports_unknown_tool() -> None:
    """A metrics pass over logged turns has no environment to check against."""
    turn = parse_turn(
        "<tool_call>\n<function=whatever>\n<parameter=x>\n1\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert turn.outcome == "ok"
    assert turn.name == "whatever"


def test_string_arguments_reach_the_method_coerced(instance: EnvInstance) -> None:
    """The XML parser may hand back an int for a str parameter; coercion fixes it."""
    text = (
        "<tool_call>\n<function=get_clinical_trial_by_id>\n"
        "<parameter=trial_id>\n101\n</parameter>\n</function>\n</tool_call>"
    )
    turn = _parse(text, instance)
    assert turn.outcome == "ok"
    assert turn.arguments == {"trial_id": "101"}


def test_a_parsed_call_actually_executes(instance: EnvInstance) -> None:
    """The end of the chain: classified `ok` means the environment can run it."""
    turn = _parse(WELL_FORMED, instance)
    assert turn.name is not None and turn.arguments is not None
    observation = instance.step(turn.name, turn.arguments)
    assert "CT-101" in observation
