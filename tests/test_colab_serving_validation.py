"""Static contracts for the direct Colab serving validation harness."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast


def test_validation_command_uses_measured_profile_without_secret_in_argv() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "colab-serving-validation.py"
    namespace = runpy.run_path(str(script))
    server_command = cast(Any, namespace["_server_command"])
    command = server_command(
        {
            "dtype": "bfloat16",
            "quantization": "fp8",
            "kv_cache_dtype": "fp8",
            "max_num_seqs": 32,
            "max_num_batched_tokens": 2048,
            "max_num_queued_reqs": 64,
            "max_num_queued_tokens": 25856,
            "gpu_memory_utilization": 0.9,
        }
    )
    rendered = " ".join(command)
    assert "Qwen/Qwen3.5-2B" in command
    assert "--quantization fp8" in rendered
    assert "--max-num-queued-reqs 64" in rendered
    assert "--max-num-queued-tokens 25856" in rendered
    assert "--api-key" not in command
    assert "cloudflared" not in rendered
    assert "docker" not in rendered
