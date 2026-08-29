"""Compile the released environment classes and verifier sources — once, per worker.

`191_env_metadata.json` carries 191 `env_class_code` strings; the RL scenario file
carries 40,231 `check_func` bodies. Both get `exec()`ed, and *where* that happens
is the load-bearing decision:

**The trainer process never execs dataset source.** Compiling there would run
module-level statements from dataset text inside the process holding the Hub
token, the W&B session, the CUDA context and the model weights, with no timeout
and no credential scrub — bypassing every mitigation the accepted `exec()` posture
rests on. So a registry is built inside each worker (see `pool.py`), and
"compile once" means once *per worker*, not once globally.

The accepted posture, stated so it is not re-litigated: the code is MIT-licensed
data from a university lab, the worker holds no secrets (enforced by `spawn` plus
a scrubbed environment, not by hoping), and process workers are needed for async
rollout anyway. The real risk being managed is a buggy `check_func` hanging an
overnight run, not malice. Full `__builtins__` is therefore kept — a
restricted-builtins sandbox that silently broke a legitimate check would corrupt
rewards, which is worse than the threat it prevents.

`exec` calls are counted on the registry so a test can assert compilation never
moved into the hot path. Counting per registry rather than globally is deliberate:
a global counter would be satisfied by compiling in the parent, which is the thing
this module exists to prevent.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolqwen.data.loader import sha256_of, verify_sha256


class RegistryError(Exception):
    """Raised when environment metadata cannot be loaded or compiled."""


@dataclass(frozen=True)
class EnvSpec:
    """One environment's metadata, before anything is compiled."""

    env_id: str
    env_class_name: str
    env_class_code: str
    tools: tuple[dict[str, Any], ...]
    environment_introduction: str = ""
    constraints_rules: tuple[str, ...] = ()

    def introduction(self) -> str:
        """The env-introduction block, matching upstream's wording exactly.

        Reproduced rather than reworded: it is part of the prompt the released
        trajectories were generated against, so a paraphrase would make the SFT
        format and the RL format disagree by exactly the amount of the paraphrase.
        """
        rules = "".join(f"- {rule}\n" for rule in self.constraints_rules)
        return (
            "# Environment Information\n\n"
            f"## Brief Introduction:  \n{self.environment_introduction}\n\n"
            f"## Environment Rules / Constraints:  \n{rules}"
        )


def parse_env_spec(env_id: str, payload: Any) -> EnvSpec:
    if not isinstance(payload, dict):
        raise RegistryError(f"{env_id}: metadata entry is not an object")
    try:
        return EnvSpec(
            env_id=env_id,
            env_class_name=str(payload["env_class_name"]),
            env_class_code=str(payload["env_class_code"]),
            tools=tuple(payload.get("tools") or ()),
            environment_introduction=str(payload.get("environment_introduction") or ""),
            constraints_rules=tuple(payload.get("constraints_rules") or ()),
        )
    except KeyError as exc:
        raise RegistryError(f"{env_id}: metadata entry is missing {exc}") from exc


def load_env_specs(path: Path | str, *, sha256: str | None = None) -> dict[str, EnvSpec]:
    """Read every env spec, verifying the file's sha256 first.

    The hash matters more than the count: 191 classes still compile after an
    `env_class_code` body is modified, the suffix split is unchanged, and the `K`
    counts are unchanged — so a count check passes trivially for the one change
    that would matter.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise RegistryError(f"env metadata not found: {file_path}")
    verify_sha256(file_path, sha256)

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegistryError(f"{file_path}: expected a JSON object keyed by env_id")
    return {env_id: parse_env_spec(env_id, entry) for env_id, entry in payload.items()}


@dataclass
class EnvRegistry:
    """Compiled environment classes, cached by `env_id`.

    Construct with `EnvRegistry.from_metadata(...)` inside a worker. The
    `exec_count` field is what the compile-once test reads.
    """

    specs: dict[str, EnvSpec]
    source_path: str
    source_sha256: str
    _classes: dict[str, type] = field(default_factory=dict)
    exec_count: int = 0

    @classmethod
    def from_metadata(
        cls, path: Path | str, *, sha256: str | None = None, eager: bool = False
    ) -> EnvRegistry:
        file_path = Path(path)
        specs = load_env_specs(file_path, sha256=sha256)
        registry = cls(
            specs=specs,
            source_path=str(file_path),
            source_sha256=sha256 or sha256_of(file_path),
        )
        if eager:
            for env_id in specs:
                registry.env_class(env_id)
        return registry

    def __len__(self) -> int:
        return len(self.specs)

    def env_class(self, env_id: str) -> type:
        """The compiled class for `env_id`, compiling on first request only."""
        cached = self._classes.get(env_id)
        if cached is not None:
            return cached

        spec = self.specs.get(env_id)
        if spec is None:
            raise RegistryError(f"unknown env_id {env_id!r}; {len(self.specs)} known")

        # A fresh module namespace per environment: 148 of the 191 bodies carry
        # their own imports and type aliases, and a shared namespace would let one
        # environment's `TaskInfo` shadow another's.
        module = types.ModuleType(f"smolqwen_env_{env_id}")
        try:
            exec(spec.env_class_code, module.__dict__)  # noqa: S102 - see module docstring
        except Exception as exc:  # noqa: BLE001 - dataset source, any error is data
            raise RegistryError(f"{env_id}: env_class_code failed to compile: {exc}") from exc
        self.exec_count += 1

        compiled = getattr(module, spec.env_class_name, None)
        if compiled is None:
            raise RegistryError(
                f"{env_id}: class {spec.env_class_name!r} not defined by its env_class_code"
            )
        if not isinstance(compiled, type):
            raise RegistryError(f"{env_id}: {spec.env_class_name!r} is not a class")
        self._classes[env_id] = compiled
        return compiled

    def tools(self, env_id: str) -> tuple[dict[str, Any], ...]:
        """The declared OpenAI-format tool schemas for `env_id`.

        Declared, not introspected: verified equal for all 191 environments (see
        `tests/test_tool_schema.py`), and the declared block is what Phase 2
        rendered into the SFT prompts, so using it keeps train and rollout byte
        identical.
        """
        spec = self.specs.get(env_id)
        if spec is None:
            raise RegistryError(f"unknown env_id {env_id!r}")
        return spec.tools

    def compiled_count(self) -> int:
        return len(self._classes)

    def manifest(self) -> dict[str, Any]:
        """What this registry compiled and from where, for a run's provenance."""
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "env_count": len(self.specs),
            "compiled_count": self.compiled_count(),
            "exec_count": self.exec_count,
        }
