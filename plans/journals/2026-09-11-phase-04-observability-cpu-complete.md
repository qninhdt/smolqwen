---
title: Phase 4 observability CPU-complete
date: 2026-09-11
summary: "Prometheus + Grafana observability for the single-L4 vLLM service is CPU-complete and verified; only live-data render remains, gated on the L4."
---

# Phase 4 observability CPU-complete

## What happened

Finalized Phase 4 of `260911-1324-single-l4-vllm-serving-benchmark`. All config
and assets were already on disk from prior sessions; this pass verified them,
reviewed the one shared touchpoint, and synced plan status back.

Delivered:
- `docker-compose.yml`: `prometheus` (v3.1.0) and `grafana` (11.4.0) added under an
  opt-in `observability` profile — loopback-only ports, no GPU reservation, Grafana
  admin password required, Grafana state on a Docker-managed `grafana-data` volume.
- `serving/prometheus.yml`: one `vllm` scrape job through the authenticated nginx
  door (`vllm-server:8080/metrics`), bearer key from a mounted secret file.
- Grafana provisioning (datasource `uid=smolqwen-prometheus`, file-based dashboard
  provider) and `serving/grafana/dashboards/smolqwen-serving.json` — four sections
  (Traffic, Latency, KV cache & prefix reuse, Reliability), 12 panels.
- `tests/test_serving_observability.py`: 12 pure config/asset checks.

## Decision

No `.gitignore` change: the bearer key already sits under the gitignored
`artifacts/serving/vllm-api-key`, and Grafana state uses a Docker-managed named
volume, so nothing new is written to the working tree.

The `docker-compose.yml` observability block is purely additive behind a profile;
the only other edits in that file (`0.26.0`→`0.29.0`, profile `l4`→`balanced`) are
legitimate Phase 1/2 changes, not regressions.

## Verification

- 576 passed, 26 gpu/dataset deselected (`pytest -m "not gpu and not dataset"`).
- Observability suite: 12/12.
- `docker compose --profile observability config`: valid.
- ruff + mypy clean on changed files.
- Secret scan: only env-var/file-path references, no committed credential.

## Next steps

L4 runtime proof only: `up{job="vllm"} == 1` after readiness and live-data panel
movement under load, per the plan's evidence boundary.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
