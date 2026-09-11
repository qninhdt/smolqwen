"""Phase 4 observability wiring: Prometheus scrape, Grafana provisioning, dashboard.

These are pure config/asset checks (no Docker, no GPU): the committed dashboard
must reference only verified vLLM 0.29.0 series through the fixed datasource, the
scrape must authenticate through nginx without a plaintext key, and Grafana must
bind loopback-only with no committed password.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from smolqwen.serving.metrics_capture import (
    KV_USAGE,
    PREEMPTIONS,
    PREFIX_HITS,
    PREFIX_QUERIES,
    RUNNING,
    WAITING,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVING = REPO_ROOT / "serving"
DASHBOARD = SERVING / "grafana" / "dashboards" / "smolqwen-serving.json"
DATASOURCE = SERVING / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
DASHBOARD_PROVIDER = SERVING / "grafana" / "provisioning" / "dashboards" / "default.yaml"
PROMETHEUS = SERVING / "prometheus.yml"
COMPOSE = REPO_ROOT / "docker-compose.yml"

DATASOURCE_UID = "smolqwen-prometheus"

# vLLM V1 (0.29.0) Prometheus series this dashboard is allowed to use. The gauge /
# counter names are the same constants the sweep capture depends on; the histogram,
# token-rate, and success names are the documented V1 series. Nothing else may
# appear in a panel query, so the dashboard can never invent an unavailable metric.
VERIFIED_METRICS = {
    RUNNING,
    WAITING,
    KV_USAGE,
    PREEMPTIONS,
    PREFIX_QUERIES,
    PREFIX_HITS,
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:time_to_first_token_seconds",
    # vLLM V1 renamed the per-request TPOT histogram; the pre-0.29 name
    # `vllm:time_per_output_token_seconds` is not exposed by v0.29.0's
    # PrometheusStatLogger (verified against vllm/v1/metrics/loggers.py @ v0.29.0),
    # so querying it would render an empty panel.
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:e2e_request_latency_seconds",
}
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_dicts(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(_iter_dicts(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_dicts(item))
    return found


def _dashboard() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    return data


def _exprs() -> list[str]:
    return [d["expr"] for d in _iter_dicts(_dashboard()) if "expr" in d]


def test_dashboard_is_valid_json_with_stable_identity() -> None:
    dash = _dashboard()
    assert dash["uid"] == "smolqwen-serving"
    assert dash["title"]
    assert dash["templating"]["list"] == [], "no import-time datasource prompt"


def test_dashboard_only_targets_the_provisioned_datasource() -> None:
    uids = {
        d["datasource"]["uid"]
        for d in _iter_dicts(_dashboard())
        if isinstance(d.get("datasource"), dict) and d["datasource"].get("type") == "prometheus"
    }
    assert uids == {DATASOURCE_UID}


def test_every_query_uses_a_verified_vllm_metric() -> None:
    exprs = _exprs()
    assert exprs, "dashboard has no panel queries"
    used: set[str] = set()
    for expr in exprs:
        used.update(re.findall(r"vllm:[a-z0-9_]+", expr))
    assert used, "no vllm series referenced"
    for name in used:
        base = name
        for suffix in _HISTOGRAM_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        assert base in VERIFIED_METRICS, f"unverified series {name!r} in a panel query"


def test_dashboard_has_the_four_required_sections() -> None:
    titles = {p["title"] for p in _dashboard()["panels"] if p.get("type") == "row"}
    assert {"Traffic", "Latency", "Reliability"} <= titles
    assert any("KV" in title for title in titles)


def test_reliability_section_tracks_scrape_health() -> None:
    assert any('up{job="vllm"}' in expr for expr in _exprs())


def test_datasource_provisioning_matches_the_dashboard() -> None:
    ds = _load_yaml(DATASOURCE)["datasources"][0]
    assert ds["uid"] == DATASOURCE_UID
    assert ds["type"] == "prometheus"
    assert ds["url"] == "http://prometheus:9090"


def test_dashboard_provider_points_at_the_mounted_directory() -> None:
    provider = _load_yaml(DASHBOARD_PROVIDER)["providers"][0]
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"


def test_prometheus_scrapes_vllm_through_authenticated_proxy() -> None:
    cfg = _load_yaml(PROMETHEUS)
    jobs = {job["job_name"]: job for job in cfg["scrape_configs"]}
    assert "vllm" in jobs
    job = jobs["vllm"]
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"][0]["targets"] == ["vllm-server:8080"]
    # Bearer auth from a mounted file — never a literal credential in the config.
    assert job["authorization"]["type"] == "Bearer"
    assert "credentials_file" in job["authorization"]
    assert "credentials" not in job["authorization"]


def test_no_inline_bearer_credential_is_committed() -> None:
    # The only place a key is referenced is Prometheus, and only by file path.
    for path in (PROMETHEUS, DATASOURCE, DASHBOARD_PROVIDER, DASHBOARD):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            assert not line.startswith("credentials:"), f"inline credential in {path.name}"


def test_compose_wires_private_observability_without_gpu() -> None:
    services = _load_yaml(COMPOSE)["services"]
    assert "prometheus" in services and "grafana" in services
    for name in ("prometheus", "grafana"):
        svc = services[name]
        assert "observability" in svc.get("profiles", [])
        assert "deploy" not in svc, f"{name} must not reserve the GPU"
        for published in svc.get("ports", []):
            assert str(published).startswith("127.0.0.1:"), f"{name} must bind loopback only"


def test_grafana_requires_a_password_and_has_no_committed_secret() -> None:
    grafana = _load_yaml(COMPOSE)["services"]["grafana"]
    password = grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
    assert password.startswith("${GRAFANA_ADMIN_PASSWORD")
    assert ":?" in password, "admin password must be required, not defaulted"
    volumes = " ".join(grafana["volumes"])
    assert "/etc/grafana/provisioning" in volumes
    assert "/var/lib/grafana/dashboards" in volumes


def test_prometheus_mounts_config_and_secret_key() -> None:
    prometheus = _load_yaml(COMPOSE)["services"]["prometheus"]
    volumes = " ".join(prometheus["volumes"])
    assert "/etc/prometheus/prometheus.yml" in volumes
    assert "/etc/prometheus/secrets/vllm-api-key" in volumes
