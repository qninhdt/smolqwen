"""Score a final state against a scenario's Python checklist. `R = passed / K`.

Three details decide whether the reward means what the paper's reward means, and
each is easy to get wrong in a way nothing reports.

**`initial_state` must reach the check.** Every released `check_func` has arity 1
with parameter `final_state`, so `initial_state` can only arrive through the
callable's `__globals__`. Measured against the vendored file by executing all
40,231 checks with and without it: **86 checks raise `NameError` without it,
spanning 81 of 2,550 scenarios (3.2%)**; an AST pass finds 105 checks / 97
scenarios (3.8%) that reference it on some branch. (The plan's 1,208 / 779 /
30.5% is a text match that counts comments and docstrings mentioning
`initial_state` — the mitigation is required either way, but the magnitude is ~8x
smaller than stated.) Without it the design swallows the `NameError` as `False`
and reward is silently depressed on those scenarios; Phase 7's difficulty profiler
would then classify some of them `always_zero` and the curriculum would drop them.
Hence `name_error_count` on the result and a test that requires it to be zero.

**Compile-once and per-episode binding are in tension.** Compiling at scenario
load fixes each callable's `__globals__` dict, so a naive implementation binds
whatever `initial_state` existed at load. The resolution: one dedicated globals
dict *per check*, and a fresh `deepcopy(initial_state)` written into it before
each episode's scoring. Never one dict shared across a scenario's K checks — 11
released checks also define module-level helpers that resolve through it, and a
check that mutates a container reached via `initial_state` would corrupt every
later check and every later episode. That defect biases reward *downward* without
collapsing group variance, so Phase 7's zero-variance tripwire would not catch it.

**The denominator is K, not the number of checks that ran.** A check that raises
counts as a failed check, and the result is `round(..., 4)` — matching upstream's
`base_env.py:301-305` arithmetic exactly. Dropping a raising check from the
average would quietly raise the reward of a scenario whose checks are buggy.

A check returning a non-`bool` also counts as failed, again matching upstream
(`env_util.py:120-122`): a truthy string is not evidence the state is correct.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class VerifierError(Exception):
    """Raised when a checklist cannot be compiled."""


CHECK_FUNCTION_NAME = "check_func"


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome, with the reason when it did not pass."""

    check_item: str
    passed: bool
    reason: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """A scenario's reward plus everything needed to explain it."""

    reward: float
    checks: tuple[CheckResult, ...]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def name_error_count(self) -> int:
        """Checks that failed with `NameError`.

        Above zero means a global a check expected was not supplied —
        overwhelmingly `initial_state`. It is the one signal that distinguishes
        "the state is wrong" from "the verifier is wrong", and they are
        indistinguishable in the reward itself.
        """
        return sum(1 for check in self.checks if check.error_type == "NameError")

    def reasons(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if check.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "passed": self.passed,
            "total": self.total,
            "name_errors": self.name_error_count,
            "checks": [
                {
                    "check_item": check.check_item,
                    "passed": check.passed,
                    "reason": check.reason,
                    "error_type": check.error_type,
                }
                for check in self.checks
            ],
        }


@dataclass
class CompiledCheck:
    """One `check_func`, compiled once, with its own private globals dict.

    `globals_` is per check by design — see the module docstring. Rebinding
    happens through `bind_initial_state`, which is called per episode.
    """

    check_item: str
    source: str
    function: Any
    globals_: dict[str, Any]

    def bind_initial_state(self, initial_state: Mapping[str, Any] | None) -> None:
        """Write a fresh deep copy of `initial_state` into this check's globals.

        Fresh per episode and per check: a check that mutates a container reached
        through `initial_state` must not be able to affect the next call.

        `None` *removes* the name rather than binding `{}`. Binding an empty dict
        would turn the "caller forgot to supply it" failure into a `KeyError`,
        which `name_error_count` does not see — the tripwire would go quiet exactly
        when it is needed. An episode always has an initial state (captured at
        reset), so `None` can only mean a wiring mistake, and it should look like
        one.
        """
        if initial_state is None:
            self.globals_.pop("initial_state", None)
            return
        self.globals_["initial_state"] = deepcopy(dict(initial_state))

    def run(self, final_state: Mapping[str, Any]) -> CheckResult:
        """Execute the check, converting any failure into a `False` with a reason."""
        try:
            result = self.function(deepcopy(dict(final_state)))
        except Exception as exc:  # noqa: BLE001 - dataset source; any error is data
            return CheckResult(
                self.check_item,
                False,
                reason=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
        if not isinstance(result, bool):
            # Upstream treats a non-bool as a failure rather than as truthiness: a
            # check that returns a message string is not evidence of correctness.
            return CheckResult(
                self.check_item,
                False,
                reason=f"check_func returned {type(result).__name__}, not bool",
                error_type="NonBoolResult",
            )
        return CheckResult(self.check_item, result)


@dataclass
class CompiledChecklist:
    """A scenario's K compiled checks. Built once at scenario load, inside a worker."""

    task_id: str
    checks: tuple[CompiledCheck, ...]
    exec_count: int = 0
    # Checks whose source did not define `check_func`. They still count in K:
    # unrunnable is a failed check, not an absent one.
    broken: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.checks) + len(self.broken)


def compile_checklist(task_id: str, checklist: Sequence[Mapping[str, Any]]) -> CompiledChecklist:
    """Compile every `check_func` in a scenario's checklist, once.

    Each check gets its own module-level namespace with full `__builtins__` — the
    accepted posture, because a restricted-builtins sandbox that silently broke a
    legitimate check would corrupt rewards, which is worse than the threat it
    prevents. The `spawn`ed, credential-free worker plus the per-call timeout is
    what makes that acceptable.
    """
    compiled: list[CompiledCheck] = []
    broken: list[str] = []
    exec_count = 0

    for entry in checklist:
        item = str(entry.get("check_item", ""))
        source = str(entry.get("check_func", ""))
        # No `initial_state` seeded here: `score` binds it per episode, and a
        # placeholder left in the namespace would let a check that runs before the
        # first bind read an empty state instead of failing loudly.
        namespace: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(source, namespace)  # noqa: S102 - see registry.py's module docstring
            exec_count += 1
        except Exception:  # noqa: BLE001 - a check that will not compile is a failed check
            broken.append(item)
            continue
        function = namespace.get(CHECK_FUNCTION_NAME)
        if not callable(function):
            broken.append(item)
            continue
        compiled.append(
            CompiledCheck(check_item=item, source=source, function=function, globals_=namespace)
        )

    return CompiledChecklist(
        task_id=task_id,
        checks=tuple(compiled),
        exec_count=exec_count,
        broken=tuple(broken),
    )


def score(
    checklist: CompiledChecklist,
    initial_state: Mapping[str, Any] | None,
    final_state: Mapping[str, Any],
) -> ScoreResult:
    """`R = passed / K`, rounded to 4 places, with `initial_state` bound per episode."""
    results: list[CheckResult] = []
    for check in checklist.checks:
        check.bind_initial_state(initial_state)
        results.append(check.run(final_state))

    for item in checklist.broken:
        results.append(
            CheckResult(
                item,
                False,
                reason="check_func did not compile or was not defined",
                error_type="CompileError",
            )
        )

    total = len(results)
    if not total:
        raise VerifierError(f"{checklist.task_id}: checklist is empty; reward is undefined")
    passed = sum(1 for result in results if result.passed)
    return ScoreResult(reward=round(passed / total, 4), checks=tuple(results))


def score_raw(
    task_id: str,
    checklist: Sequence[Mapping[str, Any]],
    initial_state: Mapping[str, Any] | None,
    final_state: Mapping[str, Any],
) -> ScoreResult:
    """Compile and score in one call.

    For a one-off (`env-selftest`, a test) — never in the rollout hot path, where
    compilation belongs at scenario load. `test_verifier_exec_once` is what keeps
    this from creeping into the loop.
    """
    return score(compile_checklist(task_id, checklist), initial_state, final_state)
