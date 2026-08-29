---
phase: 8
title: "vLLM serving optimization"
status: pending
priority: P1
effort: "5d"
dependencies: [7]
---

# Phase 8: vLLM serving optimization

## Overview

Serve the final checkpoint as an ordinary OpenAI-compatible LLM endpoint, then
optimize it with measurement: a config sweep, MTP-1 speculative decoding, and
quantization — with a BFCL re-eval paired to every quantized row so speed claims
never stand alone.

Docker Compose is the source of truth; Colab runs the same image via tunnel.

## Requirements

**Functional**
- `docker compose up` serves the merged SFT+RL checkpoint on an OpenAI-compatible
  endpoint with reasoning and tool-call parsers enabled.
- Colab launcher running the same image definition, exposed via tunnel, with a
  generated API key.
- Benchmark harness over `vllm bench serve` using its documented datasets:
  `sharegpt` and `random` for comparability, `prefix_repetition` for prefix-cache
  measurement. **BFCL cannot supply agent-shaped multi-turn traffic**: selection
  requires `--dataset-path`/`--hf-name` matching `BFCLDataset.SUPPORTED_DATASET_PATHS`
  plus `--backend openai-chat` (`datasets.py:2290-2302`), and the loader takes only
  `question[0]` with the explicit comment "skip multi-turn categories in this loader"
  (`:4687-4688`). So tool-calling traffic shape comes from a custom dataset built
  from our own BFCL multi-turn tasks, or from `random` with a tool schema attached —
  stated as such, not claimed as BFCL multi-turn.
- Config sweep driven by `vllm bench sweep serve` with `--serve-params` /
  `--bench-params` JSON: chunked prefill, prefix caching, `max_num_seqs`,
  `max_num_batched_tokens`, CUDA graph capture, KV budget. Pareto front from
  `vllm bench sweep plot_pareto`. Report TTFT, TPOT, ITL, E2EL at p50/p95/p99 plus
  throughput and VRAM per config.
- MTP-1 speculative decoding enabled, with measured acceptance rate and
  throughput delta on agent-shaped traffic.
- Quantization per profile: FP8 on L4 (sm89), AWQ/GPTQ int4 or BF16 on A100 (sm80,
  no native FP8).
- BFCL `multi_turn_*` re-eval through the Phase 5 HTTP policy for every quantized
  configuration, under a manifest whose **invariant** set matches the training-table
  runs and whose recorded-free set captures the serving config that distinguishes
  the row.
- Two committed serving profiles with the chosen settings and the numbers that
  justified them.

**Non-functional**
- The endpoint requires an API key on **every** path. vLLM's `--api-key` only
  challenges paths under `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`
  (`server_utils.py:27,75`), leaving `/tokenize`, `/detokenize`, `/metrics`,
  `/version`, `/load`, `/health` — and `/start_profile` if profiling is enabled —
  open. On a public tunnel that is a free CPU-exhaustion vector plus model and
  throughput disclosure, so vLLM binds to loopback and a key-checking reverse proxy
  is the only thing the tunnel exposes.
- The key is passed via `VLLM_API_KEY` from a gitignored `.env`, never as
  `--api-key` in argv (visible in the process table and in `docker inspect`).
  `run_colab_serve.sh` writes the key to a gitignored file and prints its path, not
  its value. Notebook outputs are stripped in CI.
- One image definition, two run targets. Compose and Colab must not drift.
- Every performance claim traces to a committed benchmark artifact.

## Architecture

Treat it as a normal hosted model endpoint. No bespoke agent harness, no mock
business backend — the deliverable is the endpoint plus the evidence that its
configuration was chosen rather than defaulted.

```
docker compose
  ├── vllm-server   merged checkpoint, reasoning parser, tool-call parser
  │                 bound to loopback; VLLM_API_KEY from .env
  ├── proxy         requires the key on EVERY path; the only thing exposed
  └── bench         vllm bench serve → artifacts/serving/*.json

Colab: same image, tunnel points at the proxy, key written to a gitignored file
```

Two hardware facts shape the whole phase. First, quantization strategy is forced
apart: L4 is sm89 with native FP8, A100 is sm80 without. That is the real reason
two serving profiles exist — not batch size. Second, speculative decoding is capped
at **MTP-1**. Qwen3.5's GDN `conv_states` and `recurrent_states` have no sequence
dimension, so a partially accepted draft cannot be rolled back to the accepted
prefix; a 1-token draft is all-or-nothing and needs no rollback. This is
architectural, not tunable, and it bounds the achievable speedup — measure the
acceptance rate before claiming anything. This reading has not been verified
against a Qwen3.5 GDN implementation; treat it as the working hypothesis the
measurement tests, not an established fact.

The auth boundary is the third shaping fact, and it is not where `--api-key`
suggests. vLLM's middleware challenges only `GUARDED_PREFIX`-matching paths, so
`--api-key` alone leaves the tokenizer and metrics endpoints open on a public
tunnel. Hence loopback binding plus a proxy — and hence the health check must send
the key, because `/v1/models` is guarded and an unauthenticated probe gets 401
forever.

`vllm bench sweep` owns the parameter grid, the repetitions, and the Pareto plot —
`sweep serve` takes `--serve-params` / `--bench-params` JSON and already supports
`--resume` for an interrupted run, and `sweep plot_pareto` draws the front. So our
sweep code is a params-JSON emitter plus a thin invocation, not a grid engine. Do
not reimplement the Cartesian product or write tests asserting upstream's grid
expansion.

Quality pairing is the part that makes the table honest. A quantized row without a
BFCL re-eval is a speed number with an unmeasured quality cost — at 2B, tool-calling
accuracy is exactly what quantization is most likely to damage. The Phase 5 HTTP
policy makes this cheap: the same eval code that produced the training table points
at the endpoint. The manifest's invariant/recorded split (defined in Phase 5) is
what lets that comparison happen at all — strict equality over library versions
would refuse it, and a manifest with no quantization field would let three
differently-quantized rows share one hash.

## Related Code Files

- Create: `serving/Dockerfile` — vLLM image, CUDA runtime library path fix, entrypoint
- Create: `docker-compose.yml` — `vllm-server` on loopback + key-checking `proxy` (the only exposed service) + `bench`, checkpoint volume, env-driven config, no default key
- Create: `serving/entrypoint.sh` — `vllm serve` with model, parsers, KV budget, key from `VLLM_API_KEY`
- Create: `serving/proxy.conf` — requires the key on every path, not just `GUARDED_PREFIX`
- Create: `tests/test_auth_all_paths.py` — every route including `/tokenize` and `/metrics` returns 401 without a key
- Create: `src/smolqwen/serving/bench.py` — wrapper over `vllm bench serve`, result normalization into the report schema
- Create: `src/smolqwen/serving/sweep.py` — emits `--serve-params` / `--bench-params` JSON and invokes `vllm bench sweep serve --resume`; reads back the Pareto output. No grid engine, no Pareto reimplementation
- Create: `src/smolqwen/serving/report.py` — markdown table joining latency, throughput, VRAM, BFCL score, and each row's recorded-free manifest fields
- Create: `configs/serving/{l4,a100}.yaml` — chosen serving settings per GPU, with justifying numbers as comments
- Create: `scripts/run_colab_serve.sh` — launch the same image on Colab, tunnel, print base URL and key
- Create: `tests/test_bench_result_parsing.py` — benchmark JSON parses into the report schema
- Create: `docs/serving.md` — measured numbers, chosen configs, and the non-obvious failures encountered
- Modify: `configs/base/serve.yaml` — model path, parsers, defaults
- Modify: `src/smolqwen/cli.py` — wire `serve`, `bench`, `sweep`
- Create: `notebooks/04-serve.ipynb` — thin wrapper

## Implementation Steps

1. `serving/Dockerfile` + `entrypoint.sh`: pin the vLLM version, set the CUDA
   runtime library path explicitly (the PyPI wheel's expected CUDA runtime may not
   be on the loader path — export the pip-installed `nvidia/cuXX/lib` directory
   rather than downgrading torch), and start `vllm serve` with the reasoning parser,
   tool-call parser, an explicit `--max-model-len`, bound to loopback, reading the
   key from `VLLM_API_KEY` rather than an `--api-key` argv flag.
2. `docker-compose.yml`: `vllm-server` mounting the checkpoint, a key-checking
   `proxy` as the only exposed service, plus a `bench` service. The health check must
   probe **model readiness with the key attached**:
   `curl -H "Authorization: Bearer $VLLM_API_KEY" /v1/models`, or `/health` (which
   does check engine liveness, not just the port) plus one authenticated one-token
   completion. An unauthenticated `/v1/models` probe returns 401 forever and the
   operator debugs a load failure on a server that loaded fine. No default key value
   in `docker-compose.yml`.
3. Bring it up locally or on the target GPU and verify a chat completion with tools
   returns a parsed tool call and a separate reasoning field. This is the acceptance
   gate before any measurement.
4. `bench.py`: wrap `vllm bench serve`. Run `sharegpt` and `random` for
   comparability with published numbers and `prefix_repetition` to measure
   prefix-cache benefit directly. For tool-calling traffic shape, build a custom
   dataset from our own BFCL multi-turn tasks — vLLM's BFCL loader takes only the
   first turn and explicitly skips multi-turn categories, so it cannot produce the
   traffic MTP-1's acceptance rate needs to be measured against. Normalize output
   into one schema.
5. Baseline BF16 measurement at several concurrency levels. Record TTFT, TPOT, ITL,
   E2EL at p50/p95/p99, output throughput, and VRAM. This is the row every later
   claim is relative to.
6. `sweep.py`: emit `--serve-params` / `--bench-params` JSON over chunked prefill,
   prefix caching, `max_num_seqs`, `max_num_batched_tokens`, CUDA graph capture, and
   KV budget; invoke `vllm bench sweep serve` with `--resume` so a reclaimed Colab
   session continues rather than restarts; read the front from
   `vllm bench sweep plot_pareto`. Reduce the grid to the axes the front actually
   moves along.
7. Enable MTP-1. Measure acceptance rate and the throughput delta on the `bfcl`
   workload specifically — structured tool-call syntax is where drafting should help
   most. Record the acceptance rate as a first-class number; a 1-token draft caps the
   ceiling and the report must say so.
8. Quantize per profile: FP8 on L4, AWQ/GPTQ int4 (or BF16 if int4 quality collapses)
   on A100. Measure the same latency and throughput set.
9. Re-eval BFCL `multi_turn_*` through the Phase 5 HTTP policy for every quantized
   configuration. `assert_comparable` checks the invariant set against the
   training-table manifests; the serving config (quantization, dtype, spec-decode,
   KV budget, `max_num_seqs`) lands in the recorded-free set and is printed per row.
   Read the checkpoint by pinned revision sha.
10. `report.py`: join latency, throughput, VRAM, and BFCL score into one table.
    Write `artifacts/serving/report.md` and commit the chosen settings into
    `configs/serving/{l4,a100}.yaml` with the justifying numbers as comments.
11. `scripts/run_colab_serve.sh`: run the same image on Colab, expose the **proxy**
    via tunnel, print the base URL and the path to the gitignored key file — never the
    key itself, since notebook cells get committed. Start the server and proxy with the
    key in place **before** the tunnel is created: a tunnel URL is reachable the moment
    it exists, so creating it first opens an unauthenticated window on a public
    address. Detach with `setsid` so a kernel restart does not kill it (Colab kills the
    kernel's process group, which `nohup` alone does not escape), and poll readiness in
    short authenticated calls rather than one long blocking exec.
12. `docs/serving.md`: the measured numbers, the chosen configs, and the non-obvious
    failures hit along the way. That last part is the most useful section for anyone
    reproducing this.

## Success Criteria

- [ ] `docker compose up` serves the merged SFT+RL checkpoint; a tools request returns a parsed tool call and a separate reasoning field.
- [ ] Every path rejects unauthenticated requests through the tunnel — enumerate `/v1/chat/completions`, `/tokenize`, `/detokenize`, `/metrics`, `/version`, `/load` and assert each returns 401. `--api-key` alone does not achieve this.
- [ ] The key never appears in argv, `docker-compose.yml`, or a committed notebook cell; CI asserts `.ipynb` outputs are empty.
- [ ] The tunnel is created only after server and proxy are up with the key — no unauthenticated window on a public URL.
- [ ] `bench` starts only after an **authenticated** readiness probe succeeds; a server whose weights failed to load fails the health check instead of producing an error table.
- [ ] The tool-calling benchmark workload is documented as a custom dataset, with a note that vLLM's BFCL loader cannot sample multi-turn categories.
- [ ] Colab launcher runs the same image definition and prints a working base URL plus key; the server survives a kernel restart.
- [ ] Baseline BF16 numbers recorded across concurrency levels: TTFT, TPOT, ITL, E2EL at p50/p95/p99, throughput, VRAM.
- [ ] Config sweep completed via `vllm bench sweep serve` with a Pareto front from `plot_pareto`; chosen settings committed to both serving profiles with justifying numbers. No grid or Pareto logic reimplemented.
- [ ] The BFCL workload's actual composition is recorded, including that vLLM's loader skips multi-turn categories and what was used instead.
- [ ] MTP-1 acceptance rate measured on agent-shaped traffic and reported alongside its throughput delta.
- [ ] FP8 measured on L4; int4 or BF16 measured on A100; the sm89-vs-sm80 reason documented.
- [ ] Every quantized row carries a BFCL `multi_turn_*` re-eval whose invariant manifest set matches the training table, with the serving config printed per row.
- [ ] `artifacts/serving/report.md` joins latency, throughput, VRAM, and quality in one table.
- [ ] `docs/serving.md` records the chosen configs and the non-obvious failures encountered.

## Risk Assessment

**CUDA runtime mismatch blocks vLLM import.** Signal: `ImportError:
libcudart.so.N`. Response: export the pip-installed `nvidia/cuXX/lib` onto
`LD_LIBRARY_PATH` in the entrypoint. Downgrading torch to match the wheel would
rebuild the environment the training runs are pinned against — not an option.

**MTP-1 acceptance rate is too low to matter.** Signal: acceptance well under ~50%
with negligible throughput gain. Response: report the measured number and turn it
off. A negative result on a hard architectural limit is a legitimate finding; a
speculative-decoding row that claims a win it did not measure is not.

**Quantization damages tool-calling accuracy.** Most likely at 2B, and the exact
reason the re-eval is mandatory. Signal: BFCL-MT drop beyond noise on the quantized
row. Response: report the quality/speed trade-off explicitly and pick BF16 as the
default if the drop is material. Never ship a quantized config whose quality cost
was not measured.

**Prefix caching flagged experimental for hybrid models.** Enabling it can put the
Mamba/GDN cache into a mode vLLM itself labels experimental. Signal: a startup
warning to that effect, or output divergence with caching on. Response: measure with
and without on the `prefix_repetition` workload; if outputs diverge, report it and
leave caching off in the committed profile.

**Compose and Colab drift apart.** Signal: settings present in one launcher and not
the other. Response: both read the same `configs/serving/*.yaml` and the same image;
the Colab script may only supply the tunnel and the key.

**`--api-key` guards less than its name implies.** vLLM's middleware challenges only
`GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`. Signal: an
unauthenticated `GET /metrics` or `POST /tokenize` succeeding through the tunnel.
Response: loopback binding plus a proxy that requires the key on every path; the
success criterion enumerates routes rather than testing one. The same boundary,
read in the other direction, is why the health check must send the key — an
unauthenticated `/v1/models` probe returns 401 forever, and the tempting fix
(moving the probe to `/health`) is a partial regression toward the liveness-only
check step 2 exists to eliminate.

**BFCL cannot supply the multi-turn benchmark workload.** vLLM's loader takes
`question[0]` and skips multi-turn categories by design. Signal: discovering this at
step 4 rather than now. Response: already resolved in the requirements — a custom
dataset built from our own tasks, documented as such. Never report a number as "BFCL
multi-turn traffic" when the loader flattened it to first turns.

**Sweep exhausts the Colab session.** A full grid across two GPUs is many
server restarts. Signal: sessions timing out mid-sweep. Response: `vllm bench sweep
serve --resume` continues an interrupted sweep — use it rather than writing our own
config-hash resume — and reduce the grid to the axes the Pareto front actually moves
along.

**Health check passes while the model is broken.** A liveness probe on the port
succeeds as soon as the process binds, which can be minutes before weights finish
loading — or never, if loading fails. Signal: `bench` producing a table of
connection or 503 errors rather than latencies. Response: probe a real inference
path with a startup grace period, per step 2.
