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

## Benchmark and sweep

Benchmarking is upstream's. `vllm bench serve` and `vllm bench sweep serve` own
execution, resume, and the Pareto front; the repo-local wrappers that shelled out
to them only renamed vLLM's result fields into a dataclass, so they are gone.

Point the benchmark at the **proxy**, not the raw vLLM port. The proxy is the only
service that checks the bearer key, so a measurement past it describes an endpoint
nobody can reach. `vllm bench serve` reads the key from `OPENAI_API_KEY`.

```sh
export OPENAI_API_KEY="$(cat artifacts/serving/vllm-api-key)"
vllm bench serve \
  --base-url http://127.0.0.1:8080 \
  --model smolqwen \
  --dataset-name random \
  --num-prompts 100 --random-input-len 1024 --random-output-len 256 \
  --max-concurrency 4 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir artifacts/serving
```

Repeat per concurrency value; `--max-concurrency` takes one number.

### Agent-shaped traffic

`--dataset-name random` measures token throughput on synthetic prompts, which says
nothing about an agentic request's prefill shape or its tool-schema overhead. Build
the real traffic first:

```sh
smolqwen build-workload --profile l4 --output artifacts/serving/bfcl-agentic.jsonl
vllm bench serve \
  --base-url http://127.0.0.1:8080 --model smolqwen \
  --dataset-name custom --dataset-path artifacts/serving/bfcl-agentic.jsonl \
  --skip-chat-template --max-concurrency 4 \
  --save-result --result-dir artifacts/serving
```

The workload is the first agent request of each pinned BFCL multi-turn task,
rendered with that task's tool schema — vLLM's built-in BFCL loader does not replay
those categories. It measures agent-shaped *serving traffic*; it is **not a BFCL
score or a multi-turn quality claim**, and the generated composition file records
exactly what was sampled. `--skip-chat-template` is required because the prompts are
already rendered.

The template comes from the pinned base model in `configs/base/sft.yaml`, at its
recorded revision — not from `EvalConfig.http_model`, which is the name the server
answers to (`--served-model-name smolqwen`) and never a repo id. A merged checkpoint
carries the base tokenizer verbatim, so rendering against the base renders what the
server will see. The workload's render mode follows `enable_thinking` in
`configs/base/eval.yaml` — set it to `false` to benchmark a non-reasoning
endpoint's prompt shape.

### Non-reasoning serving

The server needs no change: `--reasoning-parser qwen3` tolerates the empty think
block, and the render mode is a per-request property. A non-reasoning client
sends `"chat_template_kwargs": {"enable_thinking": false}` in its chat
completion request (the eval HTTP path does exactly this when its eval config
says so); a thinking client sends `{"enable_thinking": true}`. The mode a score
was measured under lives in the eval manifest's invariant set, not in the server
configuration.

The Compose service runs this same command:

```sh
docker compose --profile bench run --rm bench
```

### Sweep

```sh
vllm bench sweep serve --serve-params <serve.json> --bench-params <bench.json> \
  --resume --strict-params -o artifacts/serving/sweep
```

### Pairing a quality score to a speed row

A throughput number and a quality score belong on one row only if both were measured
under the same serving config. `assert_comparable` cannot establish that: it compares
the manifest's **invariant** set, and two runs at different `max_num_seqs` have
identical invariants while being different experiments.

[`src/smolqwen/eval/serving_pairing.py`](../src/smolqwen/eval/serving_pairing.py)
compares `recorded_free` instead — dtype, quantization, speculative decoding, KV
budget, batching limits, chunked prefill, prefix caching — and refuses the pairing on
any difference. `smolqwen evaluate --require-serving-match <report.json>` applies it.

Those eight fields are read off the in-process engine's resolved `VllmConfig`, which
is the only party that knows what it ran at: vLLM resolves `max_num_batched_tokens`
and the chunked-prefill default itself. An `--endpoint` evaluation records them as
unknown, and the guard compares only what the throughput measurement recorded, so an
unknown makes no claim rather than a false one — but it also means a quality score
worth pairing has to come from a local engine run, not from re-scoring through the
endpoint.
The evaluation workflow and the recorded fields are documented in
[`evaluation.md`](evaluation.md).

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
