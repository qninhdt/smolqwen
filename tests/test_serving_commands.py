from __future__ import annotations

import json

import pytest

from smolqwen.config_models import ServeConfig, ServingProfileConfig
from smolqwen.serving.server import ServingError, build_serve_command, serving_environment


def test_serve_argv_has_parsers_and_mtp_but_no_secret() -> None:
    config = ServeConfig(
        speculative_num_tokens=1,
        profile=ServingProfileConfig(quantization="fp8", kv_cache_dtype="fp8"),
    )
    command = build_serve_command(config)
    assert command[:2] == ["vllm", "serve"]
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {"method": "mtp", "num_speculative_tokens": 1}
    assert "--api-key" not in command


def test_serve_argv_renders_native_admission_limits_once() -> None:
    config = ServeConfig(
        profile=ServingProfileConfig(
            max_num_queued_reqs=96,
            max_num_queued_tokens=16384,
        )
    )
    command = build_serve_command(config)
    assert command[command.index("--max-num-queued-reqs") + 1] == "96"
    assert command[command.index("--max-num-queued-tokens") + 1] == "16384"
    assert command.count("--max-num-queued-reqs") == 1
    assert command.count("--max-num-queued-tokens") == 1


def test_invalid_serving_profile_pairings_are_rejected() -> None:
    with pytest.raises(ValueError, match="FP8 serving weights"):
        ServingProfileConfig(quantization="fp8", dtype="float16")
    with pytest.raises(ValueError, match="must be set together"):
        ServingProfileConfig(max_num_queued_reqs=2)


def test_checkpoint_kv_scales_are_not_silently_dropped() -> None:
    config = ServeConfig(
        profile=ServingProfileConfig(kv_cache_dtype="fp8", kv_cache_scale="checkpoint")
    )
    with pytest.raises(ValueError, match="no independent --kv-cache-scale flag"):
        build_serve_command(config)


def test_serving_environment_requires_key_without_putting_it_in_argv() -> None:
    with pytest.raises(ServingError, match="VLLM_API_KEY"):
        serving_environment({})
    environment = serving_environment({"VLLM_API_KEY": "secret"})
    assert environment["OPENAI_API_KEY"] == "secret"


def test_serving_environment_opts_out_of_vllm_usage_stats() -> None:
    """vLLM's usage-stats collection is default-on; this is the only opt-out.

    CI already sets offline flags for HF, transformers and W&B (`ci.yml:14-18`) and
    set none for vLLM, so the served process was the one path still reporting.
    """
    environment = serving_environment({"VLLM_API_KEY": "secret"})
    assert environment["VLLM_NO_USAGE_STATS"] == "1"
    assert environment["VLLM_DO_NOT_TRACK"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"
