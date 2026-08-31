# smolqwen post-training and serving progress

## Status

Plan remains **in progress**. Code is CPU-verified through serving orchestration,
and the bounded real-L4 smoke now passes; full target-GPU training/optimization
measurements and production live-serving evidence remain pending.

| Scope | Checked | Total | Progress | Open |
|---|---:|---:|---:|---:|
| Overall acceptance | 7 | 14 | 50.0% | 7 |
| Phase criteria (`ak plan status`) | 64 | 101 | 63.4% | 37 |
| All phase + overall criteria | 71 | 115 | 61.7% | 44 |
| Phase 1 | 7 | 11 | 63.6% | 4 |
| Phase 2 | 11 | 11 | 100.0% | 0 |
| Phase 3 | 3 | 10 | 30.0% | 7 |
| Phase 4 | 15 | 15 | 100.0% | 0 |
| Phase 5 | 7 | 11 | 63.6% | 4 |
| Phase 6 | 12 | 15 | 80.0% | 3 |
| Phase 7 | 5 | 13 | 38.5% | 8 |
| Phase 8 | 4 | 15 | 26.7% | 11 |

Phase files marked complete: 2/8 (Phases 2 and 4). Phase 8 is code complete and
has a real-L4 smoke proof, but remains in progress pending profile-selection
measurements and live Compose/tunnel proof.

## Completed this session

- Full-plan sync-back across `plan.md` and all eight phase files.
- Backfilled 12 verified criteria: 7 overall acceptance, Phase 6's real
  separately constructed factory-oracle trainer, and 4 Phase 8 static gates.
- Factory oracle proof: real `GRPOTrainer`, `environment_factory`, no
  `rollout_func`, complete fixture episodes, two verifier reward calls, one CPU
  optimizer step.
- Phase 8 static proof: secret omitted from argv/Compose/notebook output; tunnel
  ordered after authenticated readiness; benchmark readiness authenticated; and
  the custom agent-shaped workload limitation documented. The chosen-config docs
  gate remains open until target-GPU selection is measured.
- Final real-L4 smoke (`scripts/colab-l4-smoke.py`) passed source preparation,
  Colab install, authenticated vLLM health/models/chat, random benchmark, and a
  one-environment held-out EnvScaler HTTP evaluation against pinned model
  revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The session was stopped
  immediately after artifact download.
- Real-L4 one-step batch probe (`scripts/colab-l4-batch-sweep.py`) measured the
  SFT boundary at micro-batch 4→8 OOM under the 16,384-token cap, the
  Transformers GRPO generation boundary at 64→128 OOM for a 2,118+512-token
  envelope, and trainer-side GRPO micro-batch boundaries at 1→2 OOM (8k
  envelope) and 2→4 OOM (4k envelope). Both probe sessions were stopped after
  downloading the results; see `qa-260830-l4-batch-limits.md`.
- Preserved every measurement/runtime gate as unchecked.

## Verification evidence

- `make check`: PASS; Ruff plus strict mypy over 118 files.
- `make test`: PASS; 308 passed, 3 warnings in 90.24s.
- `make test-ci`: PASS; 302 passed, 6 deselected, 2 expected TRL warnings.
- `make smoke`: PASS.
- Real L4 final5: `install-colab` 63.44s, `vllm-serve-smoke` 376.39s,
  `vllm-bench-smoke` 14.26s, `heldout-eval-smoke` 4.28s; all passed. The vLLM
  log records HTTP 200 for readiness, benchmark, and evaluation requests.
- Earlier full L4 gate batch: CUDA kernel self-test, full pytest, config smoke,
  environment self-test, and rollout benchmark passed after its path argument
  was corrected. That GPU pytest run includes the real two-step SFT optimizer
  smoke and two-step GRPO optimizer smoke (tiny mixed-mixer checkpoint, LoRA,
  forward/backward/step), plus the one-step factory-oracle GRPO smoke.
- Focused GRPO: 3 passed.
- Shell syntax: serving scripts pass `bash -n`.
- Compose: base and bench-profile renders pass.
- Review: no Critical or Important findings after fixes.
- Repository notebooks: 5/5 have zero outputs and zero execution counts.

No full SFT/GRPO training, serving sweep, quantization/MTP measurement, live
Docker/tunnel run, or route-by-route public HTTP audit was performed. The
bounded SFT/GRPO training smoke passed; existing rollout/final-results files
contain pending rows, not target measurements, and are not accepted as
benchmark or quality evidence.

## Blockers and risks

- Credentials: no `HF_TOKEN` or `WANDB_API_KEY`; full training/push/tracking blocked.
- The L4 smoke uses the Hub base checkpoint only; no trained SFT or SFT+RL merged
  checkpoint is available for the product-level run.
- Phase 1 privacy gate: `.env.example` and `.env` ignore remain unchecked because
  the privacy-hook tooling blocker prevents satisfying the full compound criterion.
- CI clean-checkout gate remains unchecked: workflow exists locally but was not
  run remotely.
- No trained SFT or SFT+RL merged checkpoint; downstream eval, rollout A/B, and
  serving-quality pairing cannot be demonstrated.

## Next steps

1. Use the L4 sizing evidence to choose conservative profile values, then run
   Phase 1 probes and Phase 3 SFT on target L4/A100; record checkpoint SHA.
2. Run Phase 6 factory-vs-async A/B and concurrency sweep on both targets.
3. Run difficulty profiling and full GRPO; verify interrupted resume and W&B curves.
4. Evaluate Base/SFT/SFT+RL under comparable manifests.
5. Start the real Compose stack, exercise every guarded route through the tunnel,
   verify restart survival, then collect BF16/sweep/MTP/quantization artifacts and
   paired BFCL quality rows.
6. Run the clean-checkout workflow remotely and resolve the privacy-hook blocker.

## Unresolved questions

- Which target GPU is available first, and is the A100 allocation 40 GB or 80 GB?
- When will HF/W&B credentials and the privacy-hook tooling path be available?
