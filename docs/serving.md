# Serving

This repository has two separate serving surfaces:

1. a completed direct benchmark of official `Qwen/Qwen3.5-2B` on one L4;
2. an optional nginx + vLLM deployment for a local checkpoint.

The separation is deliberate. Docker, nginx, and tunnels are not part of the
benchmark measurement.

## Direct L4 benchmark

The frozen study uses:

- official `Qwen/Qwen3.5-2B` at revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`;
- vLLM 0.29.0 and Torch 2.13.0+cu130;
- one NVIDIA L4;
- 16 selected scheduler/load points in BF16 and runtime FP8;
- one observation and 200 successful requests per point.

Colab was only the remote L4 provider. vLLM and `vllm bench sweep serve` ran
directly inside the VM. If an L4 is available locally, the same remote-side
scripts can run without the Colab controller.

The machine-readable owners are:

- `benchmarks/serving/study-points.json`;
- `benchmarks/serving/sweep.json`;
- `artifacts/serving/workload-manifest.json`.

The controller `scripts/run-selected-colab-serving.py` uploads one point,
downloads its complete bundle, and only then advances. This makes a reclaimed
session resumable without repeating completed measurements.

## Regenerate results

No GPU is needed to regenerate the report, profiles, and figures:

```bash
uv run python benchmarks/serving/analyze.py
```

Outputs:

- `artifacts/serving/summary.json`;
- `artifacts/serving/report.md`;
- four Matplotlib PNG figures;
- `configs/serving/{latency,balanced,throughput}.yaml`.

The profiles use provisional native vLLM queue limits because no production
TTFT/TPOT SLO was supplied.

## Validate one measured profile

```bash
uv run python scripts/run-colab-serving-validation.py \
  --profile balanced \
  --archive /tmp/smolqwen-serving-src.tgz
```

The validation covers:

- authentication on inference routes;
- model and chat completion;
- SSE streaming;
- structured tool calls;
- cancellation;
- native Prometheus metrics;
- bounded overload with HTTP 503;
- health recovery, zero preemption, and teardown.

`VLLM_API_KEY` is passed through the process environment, never argv or a
committed file. The direct endpoint binds to loopback.

APC queries were observed, but Qwen3.5 GDN prefix reuse produced no cache hits in
the validation. The measured profiles therefore do not depend on APC reuse.

## Optional deployment

The deployment path keeps raw vLLM private and exposes only nginx:

```text
client → nginx :8080 → vLLM :8000 → NVIDIA GPU
             │             └→ /metrics
             └→ bearer auth, SSE, request timeouts

/metrics → Prometheus → Grafana
```

Start a profile:

```bash
export VLLM_API_KEY="$(openssl rand -hex 32)"
MODEL_PATH=/absolute/path/to/checkpoint \
SMOLQWEN_PROFILE=balanced \
docker compose up --build vllm-server proxy
```

Add observability:

```bash
export GRAFANA_ADMIN_PASSWORD="[set-locally]"
docker compose --profile observability up
```

Prometheus and Grafana bind to loopback. This repository does not create a public
tunnel.

## Runtime boundary

The training/evaluation environment remains on the pinned vLLM 0.26 + Torch 2.11
ABI required by its GPU kernels. Serving benchmark and deployment run in an
isolated vLLM 0.29 environment. No serving measurement was produced by vLLM 0.26.

For measured values and interpretation, see
[Measured serving performance](performance.md).
