"""Run the pinned vLLM OpenAI-compatible server. Argv is owned by `inference`."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping

from smolqwen.config_models import ServeConfig
from smolqwen.inference.engine import disable_telemetry
from smolqwen.inference.profiles import ServeProfile


class ServingError(RuntimeError):
    """Raised when serving would violate an explicit deployment contract."""


def build_serve_command(config: ServeConfig) -> list[str]:
    """Return argv only; the API key stays in the environment, never process args."""
    return ServeProfile(config).command()


def serving_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Require the server/proxy key and mirror it for benchmark clients."""
    environment = dict(source or os.environ)
    key = environment.get("VLLM_API_KEY", "").strip()
    if not key:
        raise ServingError("VLLM_API_KEY must be set; there is no default serving key")
    environment["OPENAI_API_KEY"] = key
    disable_telemetry(environment)
    return environment


def run_server(config: ServeConfig, *, print_command: bool = False) -> int:
    command = build_serve_command(config)
    if print_command:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, env=serving_environment(), check=False)
    return int(completed.returncode)
