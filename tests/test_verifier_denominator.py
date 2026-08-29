"""Verifier denominator: a raising check lowers the reward, it is not dropped.

Upstream computes `round(sum(results) / len(checklist), 4)` where a failed check
contributes `None`-filtered-to-nothing in the numerator but still counts in the
denominator (`base_env.py:301-305`). The tempting "skip the broken ones" variant
quietly *raises* the reward of a scenario whose checks are buggy, which would make
those scenarios look easy to Phase 7's difficulty profiler.

`K` ranges 2 to 445 across the release with median 14, so this is not a rounding
curiosity: one raising check out of two is half the reward.
"""

from __future__ import annotations

from smolqwen.env.verifier import compile_checklist, score, score_raw


def _check(item: str, body: str) -> dict[str, str]:
    return {"check_item": item, "check_func": body}


PASSING = _check("passes", "def check_func(final_state):\n    return True\n")
FAILING = _check("fails", "def check_func(final_state):\n    return False\n")
RAISING = _check("raises", "def check_func(final_state):\n    raise RuntimeError('boom')\n")


def test_a_raising_check_counts_in_the_denominator() -> None:
    result = score_raw("t", [PASSING, RAISING], {}, {})
    # 1 of 2, not 1 of 1: dropping the raising check would report 1.0 for a
    # scenario that half-failed.
    assert result.reward == 0.5
    assert result.total == 2
    assert result.passed == 1
    assert any("RuntimeError" in reason for reason in result.reasons())


def test_the_error_type_is_recorded_not_just_the_message() -> None:
    result = score_raw("t", [RAISING], {}, {})
    assert result.checks[0].error_type == "RuntimeError"
    # `NameError` is singled out because it is the one error that means the
    # verifier was called wrong rather than the state being wrong.
    assert result.name_error_count == 0


def test_a_check_that_does_not_compile_still_counts() -> None:
    broken = _check("syntax error", "def check_func(final_state)\n    return True\n")
    result = score_raw("t", [PASSING, broken], {}, {})
    assert result.total == 2
    assert result.reward == 0.5
    assert any(check.error_type == "CompileError" for check in result.checks)


def test_a_source_without_check_func_counts_as_a_failed_check() -> None:
    misnamed = _check("wrong name", "def verify(final_state):\n    return True\n")
    result = score_raw("t", [PASSING, misnamed], {}, {})
    assert result.total == 2
    assert result.reward == 0.5


def test_a_non_boolean_return_is_a_failure_not_truthiness() -> None:
    """Matches upstream: a message string is not evidence the state is correct."""
    stringy = _check("returns a string", "def check_func(final_state):\n    return 'yes'\n")
    result = score_raw("t", [PASSING, stringy], {}, {})
    assert result.reward == 0.5
    assert result.checks[1].error_type == "NonBoolResult"


def test_a_truthy_non_bool_does_not_sneak_through() -> None:
    truthy = _check("returns 1", "def check_func(final_state):\n    return 1\n")
    assert score_raw("t", [truthy], {}, {}).reward == 0.0


def test_rounding_matches_upstream_to_four_places() -> None:
    checklist = [PASSING] + [FAILING] * 2
    assert score_raw("t", checklist, {}, {}).reward == round(1 / 3, 4) == 0.3333

    seven = [PASSING] * 2 + [FAILING] * 5
    assert score_raw("t", seven, {}, {}).reward == round(2 / 7, 4) == 0.2857


def test_all_passing_is_exactly_one_and_all_failing_exactly_zero() -> None:
    assert score_raw("t", [PASSING] * 4, {}, {}).reward == 1.0
    assert score_raw("t", [FAILING] * 4, {}, {}).reward == 0.0


def test_broken_checks_are_reported_alongside_the_runnable_ones() -> None:
    """A reward with no explanation is not something a run can be debugged from."""
    broken = _check("syntax error", "def check_func(final_state)\n")
    compiled = compile_checklist("t", [PASSING, broken, RAISING])
    assert len(compiled) == 3
    assert compiled.exec_count == 2  # the broken source never compiled

    result = score(compiled, {}, {})
    assert result.total == 3
    assert [check.check_item for check in result.checks] == ["passes", "raises", "syntax error"]
