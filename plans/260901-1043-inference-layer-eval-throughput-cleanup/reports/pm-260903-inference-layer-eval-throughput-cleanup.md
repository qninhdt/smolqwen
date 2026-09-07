## Session Report: 2026-09-03

### Work Completed

- [x] Enforced the configured Python-level benchmark timeout and preserved nested
      SIGALRM state.
- [x] Added `sft/bench_weight_version` and `grpo/bench_weight_version` to sink
      payloads.
- [x] Persisted SFT boundary outcomes beside `checkpoint-N`, with base/missing-
      checkpoint fallbacks and non-fatal sidecar errors.
- [x] Updated the SFT optimization ledger with the artifact and attribution rules.
- [x] Corrected the vLLM GPU fixtures to use the released Qwen3.5 wrapper shape.
- [x] Moved vLLM memory accounting into the worker process via `collective_rpc`.
- [x] Added deterministic non-zero adapter preflight; recognized vLLM capability
      refusals fall back to Transformers through the shared turn engine while
      OOM/corrupt-adapter errors propagate; zero-init adapters still exercise lazy
      loading.
- [x] Added the SFT adapter fallback backend, explicit dtype propagation, and narrow
      missing-vLLM/capability-error classification.
- [x] Added tensor-parallel worker-sum coverage and narrowed the adapter GPU probe
      to exercise lazy generation rather than treating registration as acceptance.
- [x] Added contract assertions for worker-extension injection and echoed prompt-ID
      alignment, so the two new safety boundaries are checked on every CPU run.
- [x] Isolated metric-sink failures from the eval boundary, so a transient
      Trainer/W&B logging error cannot stop training after evaluation completes.
- [x] Made GPU-only adapter tests skip explicitly when vLLM/CUDA is absent; direct
      `pytest -m gpu` now reports the unavailable environment instead of failing
      during collection, while real-card assertions remain unchanged.
- [x] Routed the local Transformers fallback through the shared token-id turn engine,
      kept HTTP text-native, restored per-task stderr progress, and shut down the
      fallback policy after evaluation.
- [x] Moved batched progress reporting onto the turn engine's episode-completion
      callback, so long runs report tasks while remaining episodes are still active;
      added a regression test for callback timing.

### Verification

| Check | Result |
|---|---|
| Focused eval/inference/training regression slice | 97 passed |
| Latest fallback/turn-engine focused slice | 39 passed in 4.66s |
| `make test-ci` | 495 passed, 18 deselected, 2 known TRL warnings |
| `pytest -m gpu` (local) | 5 skipped: no vLLM/CUDA; target-card tests remain enabled |
| `make check` | Ruff, format, mypy passed |
| `make smoke` | Passed |
| T4 follow-up | vLLM checks `s..`; fp16 fallback, worker RPC, and silent adapter no-op guard verified |
| Latest Colab L4 request | Rejected: accelerator unavailable due quota/entitlement |
| Direct L4 allocation retry | Rejected again by the backend: no quota/entitlement |
| Colab G4 fallback probe | Rejected: accelerator unavailable due quota/entitlement |
| Current Colab session inventory | No active sessions; no VM left running |
| Plan validation | Passed; 5/10 phases done, 85/105 tasks (80%) |

### In Progress / Blocked

- [ ] L4 validation remains blocked: local hardware is an RTX 3050 Laptop GPU
      with 4 GiB, and `vllm` is absent. Colab rejected both L4 and A100 session
      requests, and also rejected a G4 fallback, for unavailable quota/entitlement;
      the available T4 was used only for the rehearsal recorded in
      `reports/t4-rehearsal.md`. Real-weight
      agreement, throughput, batch stability, sleep/envelope, training-cost,
      Colab, and final BFCL evidence remain unmeasured.
- [ ] The deliberately rejected `bench_heldout_minus_train_reward` metric remains
      unimplemented because its source series use mismatched logging cadences.
- [x] The vLLM adapter fixture and parent-process VRAM guard findings from the T4
      rehearsal are fixed in the worktree; their live adapter and L4 measurements
      remain Phase 10 work.

### Next Steps

1. Run the Phase 10 sequence on one L4/A100 with the pinned vLLM/Colab extras.
2. Reconcile the remaining GPU checkboxes and publish the raw measurement report.
3. Do not commit the dirty `third_party/EnvScaler` submodule state; it predates
   this session and is outside scope.
