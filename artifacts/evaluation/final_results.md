# Final results

Status: **PENDING TARGET-GPU RUNS**

This artifact is intentionally incomplete. CPU verification proves the Phase 7
training contracts but cannot produce the merged RL checkpoint, target-profile
OOM/throughput evidence, W&B learning curves, or pinned benchmark manifests.
Values must not be filled from fixture or scripted-policy tests.

The RL profile choice must cite the measured colocated-vLLM episodes/hour rows in
[`../rollout/ab_report.md`](../rollout/ab_report.md), not the Phase 1 capability
probe.

| Metric | Base | SFT | SFT+RL |
|---|---:|---:|---:|
| BFCL-MT overall | PENDING | PENDING | PENDING |
| BFCL-MT simple | PENDING | PENDING | PENDING |
| BFCL-MT multiple | PENDING | PENDING | PENDING |
| BFCL-MT parallel | PENDING | PENDING | PENDING |
| BFCL-MT parallel multiple | PENDING | PENDING | PENDING |
| Held-out EnvScaler mean reward | PENDING | PENDING | PENDING |
| Invalid-call rate | PENDING | PENDING | PENDING |
| Average steps | PENDING | PENDING | PENDING |
| Average tokens | PENDING | PENDING | PENDING |

## Interpretation

Pending the full GRPO run and comparable pinned evaluations. The final text must
explicitly discuss every unchanged or regressed headline and secondary metric;
a mixed or negative result must not be rewritten as a uniform improvement.
