"""The `environment_factory` oracle: TRL's own turn-synchronous path over Phase 4.

This adapter exists to establish what the right answer is. It runs inside its
**own** trainer — one constructed with `environment_factory` and *no*
`rollout_func` — because `env_mask` is only read on the `else` branch of
TRL's `if self.tools:`, and `environment_factory` populates `self.tools` from
every public method of the object the factory returns. A trainer holding both
silently discards our mask and rebuilds it as all-ones over completions that
already contain observations: loss descends, the model learns to predict tool
output, nothing errors. `test_trainer_kwargs_exclusive` fails loudly on exactly
that construction.

TRL pools factory instances and re-`reset`s them across batches, and upstream's
`init_env_instance` restores only `init_config` keys — so an attribute an
environment gained in a previous episode can survive into the next
`final_state()` and satisfy a check it should not. Here `reset` never mutates a
live environment at all: it destroys the pool episode and creates a fresh one,
which constructs a brand-new instance inside the worker. The reuse test pins
this by running two consecutive batches of one scenario through the *same*
pooled adapter instance.

Every environment executes inside a `spawn`ed Phase 4 worker, same as the async
path — the oracle differs from production in scheduling only, never in
isolation. Tool schemas for the prompt are built from the metadata's JSON
schemas (`EnvSpec.tools`), and each proxy method carries a matching signature
and Google-style docstring so `transformers.get_json_schema` renders the real
schema, not a `**kwargs` stub.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from smolqwen.env.pool import Result, WorkerPool
from smolqwen.env.registry import EnvSpec
from smolqwen.env.scenarios import Scenario

_JSON_TYPE_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class FactoryEnvError(RuntimeError):
    """Raised when the oracle adapter is misused or its pool call cannot run."""


def _proxy_method(name: str, schema: Mapping[str, Any]) -> Any:
    """Build one public tool method from the env's JSON schema.

    The signature and docstring are reconstructed from the schema because the
    class this method proxies lives inside a worker process — the parent has
    the schema, never the code. A `**kwargs` stub would render as an argument-
    less tool and the model would never see the real parameters.
    """
    parameters = schema.get("parameters", {})
    properties: Mapping[str, Any] = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    signature_parameters: list[inspect.Parameter] = []
    doc_args: list[str] = []
    for parameter_name, spec in properties.items():
        annotation = _JSON_TYPE_TO_PYTHON.get(str(spec.get("type", "")), Any)
        description = str(spec.get("description", "")).strip().replace("\n", " ")
        if parameter_name in required:
            default = inspect.Parameter.empty
        else:
            annotation = annotation
            default = None
        signature_parameters.append(
            inspect.Parameter(
                parameter_name,
                # JSON Schema preserves property order, but that order may put
                # an optional property before a required one. Keyword-only
                # parameters retain the schema order and legally allow that
                # combination in an inspect.Signature.
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )
        doc_args.append(f"    {parameter_name}: {description or 'argument'}")

    def method(self: Any, **kwargs: Any) -> str:
        return str(self._step(name, kwargs))

    raw_description = str(schema.get("description", "")).strip()
    # Released schemas often embed their own typed ``Args: name (type):``
    # section. Transformers' Google-docstring parser does not accept that typed
    # form, so retain only the prose summary and emit one canonical Args block
    # from the JSON-schema parameters below.
    description = raw_description.split("\n\nArgs:", 1)[0].strip() or f"Call {name}."
    lines = [description, ""]
    if doc_args:
        lines.append("Args:")
        lines.extend(doc_args)
    method.__doc__ = "\n".join(lines)
    method.__name__ = name
    method.__qualname__ = name
    # Bound-method inspection removes the first signature parameter. Include a
    # synthetic `self` so `inspect.signature(instance.tool)` and Transformers'
    # schema builder both retain every real tool argument.
    method.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY),
            *signature_parameters,
        ]
    )
    method.__annotations__ = {
        parameter.name: parameter.annotation for parameter in signature_parameters
    }
    return method


class FactoryEnvBase:
    """The pooled oracle instance TRL drives. One per rollout position.

    State on this object is deliberately just the current pool episode id:
    everything live lives in a worker, behind an episode id that `reset`
    recreates from scratch. No attribute can survive a batch boundary because
    none is ever written during stepping.
    """

    _pool: WorkerPool
    _scenarios_by_task: Mapping[str, Scenario]
    _env_id: str

    def __init__(self) -> None:
        self._episode_id: str | None = None
        self._generation = 0

    # --- TRL contract surface ---

    def reset(self, **row: Any) -> None:
        """Rebind this position to a fresh episode of the row's scenario.

        `row` is the dataset row TRL forwards (every column). It must carry
        `task_id`; the prompt itself comes from the dataset, so no observation
        is returned.
        """
        task_id = row.get("task_id")
        if not task_id:
            raise FactoryEnvError(
                "oracle dataset rows must carry a task_id column; reset cannot "
                "guess which scenario this position runs"
            )
        scenario = self._scenarios_by_task.get(str(task_id))
        if scenario is None:
            raise FactoryEnvError(f"task_id {task_id!r} is not in the scenario set")

        self._destroy()
        self._generation += 1
        episode_id = f"oracle-{self._env_id}-{id(self):x}-{self._generation}"
        result = self._pool.create(
            episode_id,
            env_id=scenario.env_id,
            env_class_name=scenario.env_class_name,
            init_config=scenario.init_config,
            checklist=scenario.checklist,
            checklist_id=scenario.task_id,
        )
        if not result.ok:
            raise FactoryEnvError(f"{task_id}: create failed ({result.reason}): {result.detail}")
        self._episode_id = episode_id
        return None

    def get_reward(self) -> float:
        """The verifier's reward for the current episode's final state."""
        result = self._score()
        if not result.ok:
            raise FactoryEnvError(
                f"{self._episode_id}: score failed ({result.reason}): {result.detail}"
            )
        return float(result.value["reward"])

    # --- internals ---

    def _step(self, name: str, arguments: Mapping[str, Any]) -> str:
        if self._episode_id is None:
            raise FactoryEnvError("step before reset; TRL must call reset first")
        result = self._pool.step(self._episode_id, name, dict(arguments))
        if result.ok:
            return str(result.value)
        # Errors and timeouts come back as observations — the same text the
        # async scheduler feeds its episodes — so a failed call costs the
        # oracle a turn instead of killing the batch.
        return f"Error: {result.reason}: {result.detail}"

    def _score(self) -> Result:
        if self._episode_id is None:
            raise FactoryEnvError("get_reward before reset")
        return self._pool.score(self._episode_id)

    def _destroy(self) -> None:
        if self._episode_id is None:
            return
        self._pool.destroy(self._episode_id)
        self._episode_id = None

    def final_state(self) -> dict[str, Any] | None:
        """Expose the worker-side final state, for the reuse test's assertion."""
        if self._episode_id is None:
            return None
        result = self._pool.finalize(self._episode_id)
        if not result.ok:
            raise FactoryEnvError(f"finalize failed ({result.reason}): {result.detail}")
        return dict(result.value)


def build_factory_env_class(
    *,
    env_spec: EnvSpec,
    pool: WorkerPool,
    scenarios_by_task: Mapping[str, Scenario],
) -> type[FactoryEnvBase]:
    """One adapter class per env, public methods mirroring its tool schema.

    Built per env rather than one class for everything because TRL collects
    tools via `inspect.getmembers` — dynamic `__getattr__` proxies are invisible
    to it. The env's real method names must therefore exist as real methods.
    """
    namespace: dict[str, Any] = {
        "_pool": pool,
        "_scenarios_by_task": scenarios_by_task,
        "_env_id": env_spec.env_id,
    }
    for tool in env_spec.tools:
        function = tool.get("function", {})
        name = str(function.get("name", ""))
        if not name or name.startswith("_"):
            continue
        namespace[name] = _proxy_method(name, function)
    return type(f"FactoryEnv_{env_spec.env_id}", (FactoryEnvBase,), namespace)


def make_environment_factories(
    *,
    env_specs: Mapping[str, EnvSpec],
    scenarios: Sequence[Scenario],
    pool: WorkerPool,
) -> dict[str, Any]:
    """The `environment_factory` dict TRL accepts, keyed by env id.

    One factory per env id present in the scenario set. TRL probes each factory
    once at trainer init (`factory()`), pools the instance, and hands pooled
    instances out per batch position — which is exactly the reuse pattern the
    oracle's `reset` is built to survive.
    """
    scenarios_by_task = {scenario.task_id: scenario for scenario in scenarios}
    factories: dict[str, Any] = {}
    for scenario in scenarios:
        if scenario.env_id in factories:
            continue
        env_spec = env_specs.get(scenario.env_id)
        if env_spec is None:
            raise FactoryEnvError(
                f"{scenario.task_id}: env {scenario.env_id!r} missing from metadata"
            )
        factories[scenario.env_id] = build_factory_env_class(
            env_spec=env_spec, pool=pool, scenarios_by_task=scenarios_by_task
        )
    return factories
