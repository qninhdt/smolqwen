from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PATHS = (
    "/v1/chat/completions",
    "/tokenize",
    "/detokenize",
    "/metrics",
    "/version",
    "/load",
)


def test_proxy_auth_is_server_wide_before_the_catch_all_location() -> None:
    config = (ROOT / "serving/proxy.conf").read_text(encoding="utf-8")
    guard = "if ($smolqwen_authorized = 0)"
    assert config.count(guard) == 1
    assert config.index(guard) < config.index("location /")
    assert "return 401;" in config
    assert config.count("location ") == 1
    # No path is allowlisted around the server-wide bearer check.
    for path in PROTECTED_PATHS:
        assert f"location {path}" not in config


def test_compose_publishes_only_proxy_and_never_places_key_in_argv() -> None:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(text)
    server = compose["services"]["vllm-server"]
    assert server["ports"] == ["${PROXY_PORT:-8080}:8080"]
    assert "8000:8000" not in text
    assert "--api-key" not in text
    assert "VLLM_API_KEY must be set" in text
    assert "Authorization: Bearer $$VLLM_API_KEY" in text
    bench_mounts = compose["services"]["bench"]["volumes"]
    assert any("artifacts/models/qwen3.5-2b-sft-grpo-merged" in mount for mount in bench_mounts)
    assert any("third_party" in mount for mount in bench_mounts)


def test_colab_launcher_creates_tunnel_only_after_authenticated_readiness() -> None:
    script = (ROOT / "scripts/run_colab_serve.sh").read_text(encoding="utf-8")
    readiness = script.index("http://127.0.0.1:8080/v1/models")
    tunnel = script.index("setsid cloudflared tunnel")
    assert readiness < tunnel
    assert 'echo "api_key_file=$key_path"' in script
    assert 'echo "$VLLM_API_KEY"' not in script
