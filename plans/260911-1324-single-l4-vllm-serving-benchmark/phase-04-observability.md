---
phase: 4
title: "Add minimal Prometheus and Grafana observability"
status: CPU-complete; runtime proof pending L4
priority: P1
effort: "1 day"
dependencies: [2]
---

# Phase 4: Add minimal Prometheus and Grafana observability

## Context Links

- [`docker-compose.yml`](../../docker-compose.yml)
- [`serving/proxy.conf`](../../serving/proxy.conf)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)

## Overview

Scrape vLLM's native Prometheus endpoint and provision one small Grafana dashboard.
No tracing, Loki, custom telemetry service, or duplicate metrics pipeline.

## Requirements

- Traffic: generation/prompt token rates and running/waiting requests.
- Latency: TTFT/TPOT p50/p95/p99, queue time; ITL secondary.
- KV: usage, preemptions, prefix queries/hits, cached prompt tokens.
- Reliability: native success/failure/status metrics when 0.29.0 exposes them,
  benchmark failures, and container health; never invent unavailable series.
- Grafana binds locally by default and has no committed password.

## Architecture

Prometheus reaches `/metrics` over private Compose networking or authenticated
nginx. Grafana reads only Prometheus. Sweep snapshots use the same counters.

## Related Code Files

- Modify: `docker-compose.yml`, `serving/proxy.conf`, `.gitignore`
- Create: `serving/prometheus.yml`
- Create: `serving/grafana/provisioning/datasources/prometheus.yaml`
- Create: `serving/grafana/provisioning/dashboards/default.yaml`
- Create: `serving/grafana/dashboards/smolqwen-serving.json`
- Create/modify: observability config tests

## Implementation Steps

1. Inventory exact 0.29.0 metric names and labels before writing PromQL.
2. Add one vLLM scrape target with deterministic interval and retention.
3. Keep metrics private without exposing raw vLLM on the host/tunnel.
4. Provision datasource/dashboard; expose Grafana on loopback only by default.
5. Build four sections with histogram quantiles and rate windows.
6. Generate smoke traffic and require every retained panel query to return data.
7. Confirm metrics/logs contain no authorization header or key value.

## Todo

- [x] Prometheus scrapes vLLM. (`serving/prometheus.yml` job `vllm` → `vllm-server:8080/metrics`, Bearer from mounted key file)
- [x] Dashboard uses verified 0.29.0 metrics. (test-enforced allowlist of V1 series in `tests/test_serving_observability.py`)
- [ ] Traffic, latency, KV, and reliability sections render live data. (4 sections + 12 panels built; live-data render pending L4)
- [x] Monitoring is CPU-only and private by default. (no GPU reservation, loopback ports, `observability` profile)

## Success Criteria

- [ ] `up{job="vllm"} == 1` after service readiness. (pending L4 runtime)
- [ ] Quantiles, rates, queue depth, KV, APC, and preemption panels move under load. (pending L4 runtime)
- [x] No GPU consumer, log stack, tracing stack, or custom exporter is added.

## Implementation Notes (CPU-complete 2026-09-11)

Delivered and verified on CPU:
- `docker-compose.yml`: `prometheus` (v3.1.0) + `grafana` (11.4.0) under the opt-in
  `observability` profile, loopback-only ports (`127.0.0.1:9090`, `127.0.0.1:3000`),
  no GPU reservation, Grafana admin password required (`GRAFANA_ADMIN_PASSWORD:?`),
  Grafana state on the Docker-managed `grafana-data` volume.
- `serving/prometheus.yml`: single `vllm` scrape job through the authenticated nginx
  door with the bearer key read from `/etc/prometheus/secrets/vllm-api-key`.
- `serving/grafana/provisioning/{datasources/prometheus.yaml,dashboards/default.yaml}`
  and `serving/grafana/dashboards/smolqwen-serving.json` (Traffic, Latency,
  KV cache & prefix reuse, Reliability — 12 panels, all on `uid=smolqwen-prometheus`).
- `tests/test_serving_observability.py`: 12 config/asset checks, all passing.

No `.gitignore` change was needed: the bearer key already lives under the
gitignored `artifacts/serving/vllm-api-key`, and Grafana state uses a Docker-managed
named volume (no local data directory is written).

Pending L4 (runtime proof only): `up{job="vllm"} == 1` and live-data panel movement,
per the plan's evidence boundary.

## Risk Assessment

- No native HTTP-status metric: retain 503 evidence in benchmark/integration
  artifacts and document the dashboard limitation; exporter scope needs a new decision.
- Prometheus cannot reach loopback: scrape authenticated nginx, never expose vLLM raw.
