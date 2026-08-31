# Phase 6 rollout progress

## Status

Phase 6 in progress. Ready-queue code and CPU correctness gates are complete;
the accepted target-hardware A/B is not.

| Gate | Result |
|---|---|
| Plan sync | 2/8 phases complete; 54/101 criteria checked (53%) |
| Full repository suite | 269/269 passed after final cleanup fix (268/268 before its added regression) |
| Phase 6 focused suite | 33/33 passed after final cleanup fix (32/32 before its added regression) |
| Static analysis | Ruff check, Ruff format check, strict mypy green |
| Dataset/tool contract | 191 environments and 3,548 declared tool schemas exhaustively validated; all 191 classes compile in the dataset suite |
| Independent review | Original 5 findings plus late-create cleanup leak fixed; re-audit found no blockers |

The late-create fix drains bounded pool calls and destroys an environment whose
creation completes after failure cleanup begins; its regression is included in
the 33-test focused gate.

## Pending evidence

- Separately constructed TRL `environment_factory` trainer completing real
  episodes with no `rollout_func`.
- Colocated-vLLM factory-vs-ready-queue A/B on L4 and A100.
- Weight-sync timeline and prefix-cache hit rate with per-step invalidation.
- `generation_batch_size` / `generation_concurrency` target-profile sweep and
  committed selections.
- Measured episodes/hour speedup, or measured negative-result diagnosis.

Blockers: local RTX 3050 Laptop has 4 GB VRAM; no Phase 3 SFT checkpoint exists.
The current `artifacts/rollout/ab_report.md` therefore contains honest pending
target rows, not synthetic throughput.

## Next actions

1. Complete Phase 3 SFT on L4 or A100; record merged checkpoint revision SHA.
2. Construct the isolated TRL factory-oracle trainer and run its correctness gate.
3. Run colocated-vLLM factory/async A/B on L4, then A100; capture episodes/hour,
   tokens/s, GPU utilization, weight-sync time, and prefix-cache hit rate.
4. Sweep active-pool size and generation concurrency; write only measured choices
   to the owning profile fields.
5. Fill `artifacts/rollout/ab_report.md`; check the remaining Phase 6 criteria
   only from those target-GPU artifacts.

## Unresolved questions

- Colab A100 allocation remains 40 GB versus 80 GB until the target session probe.
