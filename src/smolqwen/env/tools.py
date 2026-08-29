"""Tool schemas from the environment class, and argument coercion for a call.

Two jobs, both about the same failure: the model calling something that does not
exist or passing something the method cannot take.

**Schemas.** Every environment ships a declared `tools` block, and Phase 2
rendered *that* block into the SFT prompts. So the declared block is what
`registry.tools()` returns and what the rollout must show the model — introspection
is not an alternative source of truth here. `introspect_tools` exists to *check*
the declared block against the compiled class (verified equal for all 191
environments), and to build a schema for a class that has no declared block, which
is what the tests construct.

**Coercion.** The decision, made once: a JSON scalar that converts losslessly to
the annotated type is coerced (`"3"` for an `int` parameter becomes `3`), and
anything else is a `bad_arguments` invalid call rather than a `TypeError` inside
the environment. Coercing is the right default because the model emits XML
parameter values, where everything arrives as text and `"3"` is what an int
argument looks like; refusing it would classify correct behaviour as invalid.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

# Lifecycle and bookkeeping names that are never tools even when public. Kept
# explicit rather than derived: the released classes define no common base, so
# there is nothing to subtract a base class's surface from.
RESERVED_METHODS: frozenset[str] = frozenset(
    {
        "reset",
        "step",
        "close",
        "render",
        "seed",
        "state",
        "get_state",
        "get_state_info",
        "final_state",
        "initial_state",
        "tools",
    }
)

# JSON Schema type names for the annotations the released signatures use.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolError(Exception):
    """Raised when a tool schema cannot be derived."""


def _unwrap_optional(annotation: Any) -> Any:
    """`Optional[X]` / `X | None` -> `X`; anything else unchanged."""
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def json_type_of(annotation: Any) -> str:
    """The JSON Schema type name for a Python annotation, defaulting to string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    base = _unwrap_optional(annotation)
    origin = get_origin(base)
    if origin is not None:
        base = origin
    return _JSON_TYPES.get(base, "string")


def signature_of(function: Any) -> inspect.Signature:
    """`inspect.signature` with string annotations resolved where possible.

    A module using `from __future__ import annotations` makes every annotation a
    string, and a string annotation coerces nothing — `"int"` is not `int`, so
    every parameter would fall through to "no annotation" and the coercion rules
    would silently stop applying. The released `env_class_code` bodies do not use
    the future import, but a caller's module might, so resolution is attempted and
    falls back rather than raising on a name the annotation references but the
    module does not define.
    """
    try:
        return inspect.signature(function, eval_str=True)
    except (NameError, TypeError, ValueError):
        return inspect.signature(function)


def introspect_tools(env_class: type) -> tuple[dict[str, Any], ...]:
    """Build OpenAI-format schemas from a class's public methods.

    Used to verify a declared block rather than to replace it — see the module
    docstring. Descriptions come from the docstring, which is what the released
    declared blocks carry too.
    """
    schemas: list[dict[str, Any]] = []
    for name, function in inspect.getmembers(env_class, predicate=inspect.isfunction):
        if name.startswith("_") or name in RESERVED_METHODS:
            continue
        signature = signature_of(function)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name == "self":
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            properties[parameter_name] = {"type": json_type_of(parameter.annotation)}
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": inspect.getdoc(function) or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return tuple(schemas)


def tool_names(schemas: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(schema.get("function", {}).get("name", "")) for schema in schemas)


@dataclass(frozen=True)
class Coerced:
    """The outcome of coercing one call's arguments.

    `reason` is `None` on success. On failure it names the parameter and what was
    wrong, because "bad arguments" with no detail is not something a rollout log
    can act on.
    """

    arguments: dict[str, Any]
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


def _coerce_scalar(value: Any, annotation: Any) -> tuple[Any, str | None]:
    target = _unwrap_optional(annotation)
    if target is inspect.Parameter.empty or target is Any:
        return value, None
    origin = get_origin(target)
    if origin is not None:
        # A container annotation (`List[str]`, `Dict[str, Any]`): check the outer
        # shape only. Element-wise validation would reject data the environment
        # itself accepts, and the environment is the authority on its own inputs.
        if origin in (list, tuple) and not isinstance(value, (list, tuple)):
            return value, f"expected an array, got {type(value).__name__}"
        if origin is dict and not isinstance(value, dict):
            return value, f"expected an object, got {type(value).__name__}"
        return value, None

    if target is bool:
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true", None
        return value, f"expected a boolean, got {value!r}"
    if target is int:
        if isinstance(value, bool):
            return value, "expected an integer, got a boolean"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            try:
                return int(value.strip()), None
            except ValueError:
                return value, f"expected an integer, got {value!r}"
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        return value, f"expected an integer, got {type(value).__name__}"
    if target is float:
        if isinstance(value, bool):
            return value, "expected a number, got a boolean"
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value.strip()), None
            except ValueError:
                return value, f"expected a number, got {value!r}"
        return value, f"expected a number, got {type(value).__name__}"
    if target is str:
        if isinstance(value, str):
            return value, None
        # A model that emits `<parameter=task_id>7</parameter>` produced text; the
        # XML parser coerced it to 7, and the method wants "7" back.
        if isinstance(value, (int, float, bool)):
            return str(value), None
        return value, f"expected a string, got {type(value).__name__}"
    return value, None


def coerce_arguments(method: Any, arguments: Mapping[str, Any]) -> Coerced:
    """Coerce a parsed call's arguments against the method's own signature.

    Reports rather than raises: an unknown parameter, a missing required one, or a
    value that cannot convert all become a `reason` the rollout returns to the
    model as an observation it can react to.
    """
    try:
        signature = signature_of(method)
    except (TypeError, ValueError) as exc:  # pragma: no cover - builtins only
        raise ToolError(f"cannot introspect {method!r}: {exc}") from exc

    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }
    accepts_kwargs = any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )

    unknown = [name for name in arguments if name not in parameters]
    if unknown and not accepts_kwargs:
        return Coerced(
            dict(arguments),
            f"unknown parameter(s) {sorted(unknown)}; expected {sorted(parameters)}",
        )

    coerced: dict[str, Any] = {}
    for name, value in arguments.items():
        parameter = parameters.get(name)
        if parameter is None:
            coerced[name] = value
            continue
        converted, reason = _coerce_scalar(value, parameter.annotation)
        if reason is not None:
            return Coerced(dict(arguments), f"parameter {name!r}: {reason}")
        coerced[name] = converted

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty and name not in coerced
    ]
    if missing:
        return Coerced(coerced, f"missing required parameter(s) {sorted(missing)}")

    return Coerced(coerced)
