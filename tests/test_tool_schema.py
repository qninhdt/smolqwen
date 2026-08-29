"""Tool schemas: the declared block is authoritative, and it matches the class.

Two claims, both verified against the real release rather than asserted:

1. **All 191 environments' declared `tools` blocks exactly match their compiled
   classes' public methods** — no extra, no missing. Verified by executing every
   `env_class_code` and introspecting it. That is what makes it safe for the
   rollout to show the model the declared block (which Phase 2 also rendered into
   the SFT prompts) rather than an introspected one: they agree, so using the
   declared block keeps train and rollout byte identical without giving up
   coverage.
2. **Reserved lifecycle names never become tools**, so a model cannot be offered
   `reset` or `step` as an environment operation.

Argument coercion is tested for the decision made once in `tools.py`: a JSON
scalar that converts losslessly to the annotated type is coerced, anything else is
`bad_arguments`. Coercing is right because the model emits XML parameter values
where everything arrives as text, so `"3"` is what an int argument looks like.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.env.registry import EnvRegistry
from smolqwen.env.tools import (
    RESERVED_METHODS,
    coerce_arguments,
    introspect_tools,
    json_type_of,
    tool_names,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENV_METADATA = FIXTURES / "env_metadata.json"
VENDORED_ENVS = Path(
    "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/191_env_metadata.json"
)


@pytest.fixture(scope="module")
def registry() -> EnvRegistry:
    return EnvRegistry.from_metadata(ENV_METADATA)


def test_the_declared_block_matches_the_compiled_class(registry: EnvRegistry) -> None:
    for env_id in registry.specs:
        declared = set(tool_names(registry.tools(env_id)))
        introspected = set(tool_names(introspect_tools(registry.env_class(env_id))))
        assert declared == introspected, env_id


@pytest.mark.dataset
def test_all_191_declared_blocks_match_their_classes() -> None:
    """The claim that lets the rollout trust the declared block, over the release."""
    if not VENDORED_ENVS.is_file():
        pytest.skip(f"{VENDORED_ENVS} not present")
    full = EnvRegistry.from_metadata(VENDORED_ENVS)
    assert len(full) == 191

    mismatches: list[tuple[str, list[str], list[str]]] = []
    for env_id in full.specs:
        declared = set(tool_names(full.tools(env_id)))
        introspected = set(tool_names(introspect_tools(full.env_class(env_id))))
        if declared != introspected:
            mismatches.append(
                (env_id, sorted(introspected - declared), sorted(declared - introspected))
            )
    assert not mismatches, mismatches[:3]
    # Compile-once: 191 classes, 191 execs, no matter how many lookups happened.
    assert full.exec_count == 191
    assert full.compiled_count() == 191


def test_reserved_lifecycle_methods_are_never_tools() -> None:
    class WithLifecycle:
        def reset(self) -> None: ...
        def step(self, action: str) -> None: ...
        def close(self) -> None: ...
        def real_operation(self, item_id: str) -> dict[str, Any]:
            """A genuine tool."""
            return {}

    names = set(tool_names(introspect_tools(WithLifecycle)))
    assert names == {"real_operation"}
    assert not (names & RESERVED_METHODS)


def test_private_methods_and_self_are_excluded() -> None:
    class Env:
        def _helper(self) -> None: ...
        def do(self, item_id: str) -> None: ...

    schemas = introspect_tools(Env)
    assert tool_names(schemas) == ("do",)
    assert "self" not in schemas[0]["function"]["parameters"]["properties"]


def test_required_reflects_the_absence_of_a_default() -> None:
    class Env:
        def do(self, needed: str, optional: int = 3) -> None: ...

    parameters = introspect_tools(Env)[0]["function"]["parameters"]
    assert parameters["required"] == ["needed"]
    assert set(parameters["properties"]) == {"needed", "optional"}


def test_annotations_map_to_json_schema_types() -> None:
    assert json_type_of(str) == "string"
    assert json_type_of(int) == "integer"
    assert json_type_of(float) == "number"
    assert json_type_of(bool) == "boolean"
    assert json_type_of(list[str]) == "array"
    assert json_type_of(dict[str, Any]) == "object"
    # `X | None` is X for schema purposes; the optionality is in `required`.
    assert json_type_of(int | None) == "integer"
    # An unannotated parameter defaults to string rather than being omitted: the
    # model has to be told something, and the XML values are text.
    assert json_type_of(Any) == "string"


def test_the_schema_is_json_serialisable(registry: EnvRegistry) -> None:
    for env_id in registry.specs:
        json.loads(json.dumps(list(registry.tools(env_id))))


class Sample:
    def act(
        self,
        name: str,
        count: int,
        ratio: float,
        flag: bool,
        items: list[str],
        note: str | None = None,
    ) -> None: ...

    def strict(self, count: int) -> None: ...


def test_a_string_integer_is_coerced_not_rejected() -> None:
    """The model emits XML text, so `"3"` is what an int argument looks like."""
    coerced = coerce_arguments(Sample.strict, {"count": "3"})
    assert coerced.ok
    assert coerced.arguments == {"count": 3}
    assert isinstance(coerced.arguments["count"], int)


def test_a_non_numeric_string_for_an_int_is_bad_arguments() -> None:
    coerced = coerce_arguments(Sample.strict, {"count": "many"})
    assert not coerced.ok
    assert "count" in (coerced.reason or "")
    assert "integer" in (coerced.reason or "")


def test_a_boolean_is_not_silently_an_integer() -> None:
    """`int(True)` is 1, which would let a bool satisfy an int parameter."""
    coerced = coerce_arguments(Sample.strict, {"count": True})
    assert not coerced.ok


def test_string_booleans_and_numbers_coerce_as_specified() -> None:
    coerced = coerce_arguments(
        Sample.act,
        {
            "name": 7,
            "count": "12",
            "ratio": "0.5",
            "flag": "true",
            "items": ["a"],
        },
    )
    assert coerced.ok
    assert coerced.arguments == {
        "name": "7",
        "count": 12,
        "ratio": 0.5,
        "flag": True,
        "items": ["a"],
    }


def test_a_missing_required_parameter_is_reported() -> None:
    coerced = coerce_arguments(Sample.act, {"name": "x"})
    assert not coerced.ok
    assert "missing required" in (coerced.reason or "")
    assert "count" in (coerced.reason or "")


def test_an_unknown_parameter_is_reported_with_the_expected_set() -> None:
    coerced = coerce_arguments(Sample.strict, {"count": 1, "typo": 2})
    assert not coerced.ok
    assert "typo" in (coerced.reason or "")
    assert "count" in (coerced.reason or "")


def test_an_omitted_optional_parameter_is_fine() -> None:
    coerced = coerce_arguments(
        Sample.act,
        {"name": "x", "count": 1, "ratio": 1.0, "flag": False, "items": []},
    )
    assert coerced.ok
    assert "note" not in coerced.arguments


def test_a_scalar_where_an_array_is_expected_is_reported() -> None:
    coerced = coerce_arguments(
        Sample.act,
        {"name": "x", "count": 1, "ratio": 1.0, "flag": False, "items": "not-a-list"},
    )
    assert not coerced.ok
    assert "array" in (coerced.reason or "")


def test_coercion_against_a_real_environment_method(registry: EnvRegistry) -> None:
    env_class = registry.env_class("env_151_rl")
    # The class is dataset-compiled, so its static type is only `type`.
    method = env_class.__dict__["update_clinical_trial_status"]
    assert callable(method)
    coerced = coerce_arguments(method, {"trial_id": "CT-101", "new_status": "completed"})
    assert coerced.ok
    assert coerced.arguments == {"trial_id": "CT-101", "new_status": "completed"}

    wrong = coerce_arguments(method, {"trial_id": "CT-101", "state": "completed"})
    assert not wrong.ok
