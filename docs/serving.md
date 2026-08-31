# Serving

The serving stack exposes the merged checkpoint as an OpenAI-compatible model
while keeping authentication independent of vLLM's route coverage. Docker
Compose is the deployment owner; direct `smolqwen` commands are the measurement
entry points. The base contract is in
[`configs/base/serve.yaml`](../configs/base/serve.yaml), with target-specific
overlays in [`configs/serving/`](../configs/serving/).

## Start the authenticated endpoint

The checkpoint must exist at the configured `model_path` (or the host
`MODEL_PATH` supplied to Compose). From the repository root:

```sh
export VLLM_API_KEY="$(openssl rand -hex 32)"
export SMOLQWEN_PROFILE=l4
docker compose up --build vllm-server proxy
```

Use `a100` instead of `l4` only on that target. Readiness is an authenticated
model query, not merely an open port:

```sh
curl --fail \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8080/v1/models
```

[`docker-compose.yml`](../docker-compose.yml) owns service wiring and checkpoint
mounts. [`serving/Dockerfile`](../serving/Dockerfile) owns the pinned image, and
[`serving/entrypoint.sh`](../serving/entrypoint.sh) owns the vLLM launch-time
checks. Use `smolqwen serve --profile l4 --print-command` only to inspect the
resolved raw vLLM command; it does not create the proxy boundary.

## Authentication boundary

Only proxy port `8080` is intended for clients or a tunnel. vLLM stays on
loopback at port `8000` in the shared service network namespace. This boundary
exists because vLLM's own API-key middleware does not cover every auxiliary
route; exposing vLLM directly can leave tokenizer, metrics, version, and health
surfaces outside the expected guard.

[`serving/proxy.conf`](../serving/proxy.conf) applies an exact bearer-key check
before its catch-all location, so every current and future path is rejected
unless it carries `Authorization: Bearer <key>`. `VLLM_API_KEY` is required with
no default and remains in the environment rather than process arguments. Never
publish port `8000`, put the key in Compose command arguments, or create a tunnel
to the raw vLLM process.

## Benchmark and sweep entry points

For a live Compose endpoint, keep the same key in the client environment and
point the benchmark wrapper at the proxy:

```sh
export SMOLQWEN_BASE_URL=http://127.0.0.1:8080
smolqwen bench --profile l4 --dataset random --concurrency 1,4,16
```

The Compose-owned default benchmark can instead be launched with:

```sh
docker compose --profile bench run --rm bench
```

[`src/smolqwen/serving/bench.py`](../src/smolqwen/serving/bench.py) is the owner
for accepted datasets, required dataset paths, normalized measurements, and
report generation. In particular, `sharegpt` and `custom` require
`--dataset-path`. Pass the paired HTTP evaluation JSON with `--quality-report`
and at least one Base/SFT report with `--quality-reference`; the wrapper refuses
invariant-manifest drift before joining speed and quality. It also requires the
evaluation row to match the benchmark's explicit dtype, quantization,
speculative-decoding setting, KV budget, batching limits, chunked-prefill
setting, and prefix-cache setting exactly. The evaluation workflow and serving
metadata fields are documented in
[`evaluation.md`](evaluation.md).

Raw and normalized result filenames include a fingerprint of the resolved
serving configuration, so another dataset or configuration does not overwrite a
prior row. Each benchmark invocation rebuilds the configured `report.md` from
all normalized rows already present, producing one aggregate view across
datasets and serving configurations. The fingerprint and aggregation behavior
are owned by the benchmark wrapper; do not infer configuration identity from a
filename by hand.

The agent-shaped traffic entry point is:

```sh
smolqwen bench --profile l4 --dataset bfcl-agentic --concurrency 1,4,16
```

This is a custom workload built from the first agent request of the repository's
pinned BFCL multi-turn tasks, rendered with each task's tool schema. vLLM's
built-in BFCL loader does not replay those multi-turn categories. The workload
therefore measures agent-shaped serving traffic; it is **not a BFCL score or a
multi-turn quality claim**. Its generated composition file records exactly what
was sampled. [`src/smolqwen/serving/workload.py`](../src/smolqwen/serving/workload.py)
owns that definition. The Compose benchmark service mounts the configured
checkpoint and the read-only benchmark checkout for this path.

Run the upstream-owned serving sweep from the target-GPU environment, with no
other process consuming that GPU:

```sh
smolqwen sweep --profile l4 --experiment-name l4 --resume
```

[`src/smolqwen/serving/sweep.py`](../src/smolqwen/serving/sweep.py) owns the
candidate parameters and delegates repetitions, resume behavior, and Pareto
plotting to `vllm bench sweep`. Results live under the configured `output_dir`;
[`src/smolqwen/serving/report.py`](../src/smolqwen/serving/report.py) owns the
joined serving/quality report format.

## Colab

On a GPU Colab runtime with Docker and `cloudflared` available:

```sh
SMOLQWEN_PROFILE=l4 bash scripts/run_colab_serve.sh
```

[`scripts/run_colab_serve.sh`](../scripts/run_colab_serve.sh) starts the same
Compose services and therefore the same image used above; Colab is not a second
image definition. It creates or reuses a credential under `artifacts/serving/`,
waits for authenticated model readiness, and only then creates the public
tunnel. It prints the base URL and the credential file path, never the key.

## Target-profile smoke evidence

The one-shot runner [`scripts/colab-l4-smoke.py`](../scripts/colab-l4-smoke.py)
was executed on a real Colab L4 with `Qwen/Qwen3.5-2B` pinned to revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. The final batch passed source
preparation, the Colab install, authenticated vLLM health/models/chat, the
random-dataset benchmark, and a one-environment held-out EnvScaler evaluation.
The server log recorded HTTP 200 for both benchmark and evaluation chat
requests. The controller downloaded the result file and stopped the session;
`colab sessions` reported no active sessions afterward.

The smoke runner uses `max_model_len=8192` and
`max_num_batched_tokens=8192`: the first held-out environment renders 4,893
input tokens when its 18 tool schemas are included, so the earlier 2,048-token
smoke cap correctly rejected the request with HTTP 400. This is a smoke sizing
choice, not a selected serving profile.

The following non-obvious checks are intentional:

- `sharegpt` and `custom` still require `--dataset-path`; the smoke benchmark
  selects vLLM's self-contained `random` dataset.
- The benchmark passes `--tokenizer` separately from the served alias, because
  `smolqwen` is not a Hub tokenizer identifier.
- HTTP evaluation supplies both the serving key and an explicit checkpoint
  revision; moving `main`/implicit revisions are rejected by the evaluation
  contract.

The two profiles exist because their quantization choices are hardware-bound:
L4 (sm89) has a native FP8 path, while A100 (sm80) does not; A100 must compare
AWQ/GPTQ int4 with BF16. The checked-in overlays intentionally leave
`quantization` and speculative decoding unset. Serving dtype is explicit in the
base config so benchmark and quality rows cannot silently disagree about it.
Values in the base serving config are starting candidates, not measured winners.

The smoke pass is not the optimization evidence: no L4/A100 sweep, Pareto
selection, peak-VRAM row, MTP-1 acceptance/throughput delta, quantized
measurement, or paired BFCL multi-turn quality table has been claimed. Those
profile-selection artifacts remain **pending**, as does a live Compose/tunnel
run against a merged SFT+RL checkpoint.

MTP is bounded to one draft token for this experiment. The state-rollback
rationale for that bound is still a working hypothesis, so acceptance rate and
throughput must decide whether it remains enabled; it is not a performance claim.

## Reproduction constraints

- A missing `VLLM_API_KEY` is a hard failure. An unauthenticated readiness probe
  also fails forever even if model loading succeeded.
- A tunnel must be created only after the proxy and authenticated readiness check
  succeed; otherwise a public unauthenticated window exists.
- The image entrypoint checks the pinned vLLM version and adds pip-installed CUDA
  runtime-library directories to `LD_LIBRARY_PATH`. Diagnose an ABI or
  `libcudart.so` failure there rather than changing the measured environment.
- Prefix caching on a hybrid/GDN model may produce an experimental warning or
  output divergence. Compare it on the prefix-repetition workload and leave it
  disabled if correctness changes.
- Benchmark parsing rejects missing promised latency fields rather than emitting
  a plausible-looking partial row. Missing VRAM, speculative acceptance, or
  paired quality stays visibly pending in the generated report.
