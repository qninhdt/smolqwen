"""One live environment per episode, constructed fresh from `init_config`.

Isolation is the correctness hinge of the whole RL stage. A GRPO group runs G
rollouts of the *same* scenario from the *same* `init_config`. If rollout 1's
mutations bleed into rollout 2, the group's rewards become correlated garbage:
advantages go wrong, training learns noise, and **no error is raised anywhere**.

So an instance is built by calling the compiled class again and deep-copying the
config into it — never by copying a shared live object. `initial_state` is captured
at reset and carried for the episode's lifetime rather than reconstructed at
scoring time, because nearly a third of RL scenarios have a check that reads it.

Construction follows upstream's `init_env_instance` semantics (try the dict
constructor, fall back to a no-arg one, then `setattr` every config key), because
the released scenarios were generated against exactly that behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class InstanceError(Exception):
    """Raised when an environment cannot be constructed or stepped."""


def state_of(instance: Any) -> dict[str, Any]:
    """The environment's state as a plain dict, deep-copied.

    Matches upstream `get_state_info`: instance `__dict__` minus dunder keys. The
    deep copy is what makes a captured snapshot immune to later mutation of the
    live instance — without it, `initial_state` would silently track the episode.
    """
    return deepcopy(
        {
            key: value
            for key, value in vars(instance).items()
            if not (key.startswith("__") and key.endswith("__"))
        }
    )


def construct(env_class: type, init_config: Mapping[str, Any] | None) -> Any:
    """Build a fresh instance and apply `init_config` onto it.

    The config is deep-copied before anything touches it, so two instances built
    from one scenario dict cannot share a nested container. That sharing is the
    exact shape the isolation failure would take: a list mutated through instance
    A visible through instance B, with both reporting plausible rewards.
    """
    config = deepcopy(dict(init_config or {}))
    try:
        instance = env_class(config) if config else env_class({})
    except TypeError:
        # Upstream's fallback: some released classes take no constructor argument.
        instance = env_class()

    for key, value in config.items():
        setattr(instance, key, value)
    return instance


@dataclass
class EnvInstance:
    """A live environment plus the initial state its verifier will need."""

    env_id: str
    env_class_name: str
    instance: Any
    initial_state: dict[str, Any]
    step_count: int = 0
    tools: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        *,
        env_id: str,
        env_class: type,
        env_class_name: str,
        init_config: Mapping[str, Any] | None,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> EnvInstance:
        instance = construct(env_class, init_config)
        return cls(
            env_id=env_id,
            env_class_name=env_class_name,
            instance=instance,
            initial_state=state_of(instance),
            tools=tools,
        )

    def tool_names(self) -> frozenset[str]:
        return frozenset(str(tool.get("function", {}).get("name", "")) for tool in self.tools) - {
            ""
        }

    def has_tool(self, name: str) -> bool:
        """Whether `name` is callable on this environment.

        Checked against the live instance as well as the declared block: the
        declared tools are what the model was shown, and `getattr` is what will
        actually run, so a name has to satisfy both.
        """
        if name.startswith("_"):
            return False
        # Test-only and caller-supplied classes may have no declared schema. Every
        # released EnvScaler class has one (asserted by `test_tool_schema`), where
        # the declaration is the authoritative surface.
        return (not self.tools or name in self.tool_names()) and callable(
            getattr(self.instance, name, None)
        )

    def step(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Call one environment method and return its result as an observation.

        The return value is stringified, matching what the released trajectories
        put in their `tool` messages — the model was trained on `str(result)`, so
        returning structured JSON here would be a format the model never saw.
        """
        if not self.has_tool(name):
            raise InstanceError(f"{self.env_id}: no callable tool named {name!r}")
        method = getattr(self.instance, name)
        self.step_count += 1
        return f"{method(**dict(arguments))}"

    def final_state(self) -> dict[str, Any]:
        return state_of(self.instance)
