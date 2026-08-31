"""Build and run the pinned vLLM OpenAI-compatible server command."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from typing import Any

from smolqwen.config_models import ServeConfig


class ServingError(RuntimeError):
    """Raised when serving would violate an explicit deployment contract."""


def build_serve_command(config: ServeConfig) -> list[str]:
    """Return argv only; the API key stays in the environment, never process args."""
    command = [
        "vllm",
        "serve",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--served-model-name",
        config.served_model_name,
        "--max-model-len",
        str(config.max_model_len),
        "--dtype",
        config.dtype,
        "--reasoning-parser",
        config.reasoning_parser,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        config.tool_call_parser,
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    command.append(
        "--enable-prefix-caching" if config.enable_prefix_caching else "--no-enable-prefix-caching"
    )
    command.append(
        "--enable-chunked-prefill"
        if config.enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    if config.model_revision:
        command.extend(["--revision", config.model_revision])
    if config.quantization:
        command.extend(["--quantization", config.quantization])
    if config.speculative_num_tokens is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": config.speculative_num_tokens,
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    return command


def serving_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Require the server/proxy key and mirror it for benchmark clients."""
    environment = dict(source or os.environ)
    key = environment.get("VLLM_API_KEY", "").strip()
    if not key:
        raise ServingError("VLLM_API_KEY must be set; there is no default serving key")
    environment["OPENAI_API_KEY"] = key
    return environment


def run_server(config: ServeConfig, *, print_command: bool = False) -> int:
    command = build_serve_command(config)
    if print_command:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, env=serving_environment(), check=False)
    return int(completed.returncode)


def safe_command_text(command: list[str]) -> str:
    """A reportable command string; secrets cannot appear because argv has none."""
    return shlex.join(command)


def config_metadata(config: ServeConfig) -> dict[str, Any]:
    return {
        "model": config.served_model_name,
        "dtype": config.dtype,
        "quantization": config.quantization,
        "speculative_decoding": (
            None
            if config.speculative_num_tokens is None
            else {"method": "mtp", "num_speculative_tokens": config.speculative_num_tokens}
        ),
        "kv_budget": config.gpu_memory_utilization,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "chunked_prefill": config.enable_chunked_prefill,
        "prefix_caching": config.enable_prefix_caching,
    }
