---
title: "End-to-end CLI and Colab progress logging"
status: completed
created: 2026-09-07
---

# Outcome

Every long-running CLI and Colab phase reports what it is doing while blocked on
model downloads, tokenizer/data loading, vLLM construction, training, evaluation,
merge, serving, or child processes. Machine-readable stdout remains unchanged;
human progress and logs go to stderr or the Colab controller stream.

## Scope

- Add one stdlib heartbeat/phase helper to the existing console boundary.
- Add lifecycle logs around every long-running stage owner.
- Stream child-process output and emit periodic heartbeats in Colab runners.
- Keep existing stdout JSON/argv contracts byte-compatible.

## Acceptance criteria

- A non-TTY CLI prints phase start, periodic heartbeat, and completion/failure for
  model/data/engine/training/eval/merge/serve work.
- `evaluate` logs before and during checkpoint/tokenizer/vLLM construction, not
  only after the engine exists.
- Colab validation/sweep/smoke scripts stream child output while running and emit
  a heartbeat when a child is quiet.
- Existing stdout contract tests, CPU tests, Ruff, and mypy remain green.

## Validation

- Focused console/CLI and Colab logging tests.
- `make check`, `make smoke`, and `pytest -m "not gpu and not dataset"`.
