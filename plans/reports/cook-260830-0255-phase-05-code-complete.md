# Phase 5 completion report

## Status

Code complete and CPU-verified. Base/SFT GPU measurements pending the Phase 3
checkpoint. Phase 6 not started.

## Delivered

- Benchmark plugins are discovered dynamically. Each adapter owns task
  lifecycle, validation of its opaque config, invalid-call accounting,
  provenance, and metric aggregation. The runner and core eval config contain
  no BFCL/EnvScaler/category-specific branches.
- BFCL supports all four pinned multi-turn categories with static user turns,
  delayed tool visibility, safe AST parsing, state/result scoring, and no
  execution sink in the eval package.
- EnvScaler held-out evaluation reuses the isolated worker runtime and verifier,
  including normalized HTTP tool calls, multiple calls per turn, exact
  completion/error signals, and exact-success reporting.
- Local base, PEFT adapter, merged checkpoint, and OpenAI-compatible HTTP
  policies require 40-character revision SHAs. HTTP requests have a configured
  timeout and preserve usage, finish reason, structured multi-call history, and
  tool-call observation pairing.
- Manifests assert adapter-owned benchmark/data/prompt/schema/task provenance
  plus shared decoding limits. Execution-only checkpoint, endpoint, backend,
  dtype, quantization, speculative decoding, KV, batching, caching, and library
  fields remain recorded but comparison-exempt.
- JSON, Markdown, and cross-run comparison reports are implemented. CLI and
  evaluation documentation expose the required checkpoint and serving fields.

## Verification evidence

- Full test suite: passed with zero failures.
- Ruff over owned `src`, `tests`, and the Phase 5 notebook: passed.
- Strict mypy over `src`: passed.
- BFCL oracle replay: 800/800 tasks passed, 200 in each configured category.
- Real manifest construction: 800 BFCL tasks and 80 held-out EnvScaler tasks;
  both adapters produced pinned source hashes and exact task provenance.
- Safety scan: no `eval(` or `exec(` call under `src/smolqwen/eval/`.

## Pending GPU evidence

No Base or SFT model artifact exists under `artifacts/models/`; Phase 3 records
its training run as GPU-pending. The current RTX 3050 has 4 GB VRAM, insufficient
for the planned 2B checkpoint evaluation. Therefore these remain pending:

- Base and merged-SFT full evaluation under comparable manifests.
- Adapter-on-base versus merged parity run.
- `artifacts/evaluation/base_vs_sft.{json,md}`.
- Confirmation that SFT beats Base on BFCL multi-turn overall, or a score-based
  diagnosis if it does not.

No placeholder reports or invented scores were written.

## Unresolved questions

None. GPU execution is gated by the already-defined Phase 3 checkpoint output.
