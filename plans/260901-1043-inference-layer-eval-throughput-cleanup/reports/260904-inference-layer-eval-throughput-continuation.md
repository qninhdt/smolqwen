---
title: "Inference layer throughput cleanup continuation"
status: partial
date: 2026-09-04
---

# Inference layer throughput cleanup continuation

## Summary

The current dirty worktree passes the CPU implementation gates. This continuation
also fixed four boundary issues: SFT/GRPO now select adapter dtype correctly for
FP16/GradScaler and explicit BF16 settings; the training callback deduplicates a
successful step when interval and save hooks coincide while retrying failed
boundaries; the Colab harness auto-selects T4/L4/A100 sizing and records a
run/profile/device/model identity for safe resume; and the non-zero adapter
preflight uses a broader deterministic probe panel. The main plan's stale claim
that T4 SFT refuses to run was corrected to the current padded FP16 SDPA contract.

## Verification

The required CPU boundary is:

- per-card routing: sm75 → padded FP16 + SDPA; sm80+ → BF16 + FA2/padding-free;
- CPU assembly retains its padding-free boundary metadata;
- shipped profiles resolve and dry-run without importing vLLM;
- the validation controller carries its profile selection into each child phase.

Verification completed on 2026-09-04:

- focused regression suite: `88 passed`;
- `make check`: Ruff, format, and mypy all passed (`160` source files);
- `make test-ci`: `520 passed, 18 deselected`, with the two existing TRL
  experimental-feature warnings;
- `make smoke`: all stage/profile dry-runs passed;
- `uv run pytest -m gpu -rs`: `5 skipped, 533 deselected`, because vLLM is not
  installed locally;
- `git diff --check`, script bytecode compilation, and validation-script `--help`
  all passed, including `--profile {auto,t4,l4,a100}`.

## Hardware boundary

No GPU session is active. A new L4 request was rejected for quota/entitlement;
two T4 requests (`4e2549` and `cb9e4c`) reached assignment but returned HTTP
503. A follow-up `colab sessions` check confirmed there was no session left
running. The local RTX 3050 is not a substitute for either target card, and vLLM
is absent locally. The main plan's GPU-only criteria therefore remain open: real adapter
acceptance, worker-side vLLM sleep/memory release, baseline agreement, throughput
and stability, in-training cost/envelope, non-TTY Colab rendering, and the final
BFCL run.

The current checkpoint backend calls `wake_up()` before its cached-adapter branch,
so the review's specific asleep-engine control-flow concern is not present in the
current source. The real duplicate-eval cost was fixed in `BenchEvalCallback` and
covered by focused tests. The adapter preflight remains a conservative guard rather
than a semantic proof; its broader panel reduces accidental equality but does not
replace the required live GPU capability result.

## Next step

When a target card can be assigned, run
`scripts/colab-gpu-validation.py --profile auto` (or an explicit profile),
then complete the ordered Phase 10 measurement sequence. Do not promote the
CPU result or the T4 rehearsal into L4 evidence.
