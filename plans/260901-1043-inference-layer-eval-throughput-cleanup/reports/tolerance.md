# Agreement tolerance and disagreement budget

Registered before any phase claims agreement with the baseline, so the number
cannot be chosen after seeing the result it has to accept.

## Why byte-identical was the wrong criterion

Three divergences exist in the current code, independent of any kernel or
batching difference. Each was read out of source, not inferred:

| Divergence | Evidence | Metric it moves |
|---|---|---|
| Decode convention | `policies.py:282` decodes with `skip_special_tokens=True`; `rollout_func.py:138` uses `False` | `score` — a completion whose tool call is wrapped in special tokens parses differently |
| `finish_reason` is synthesized | `policies.py:284` returns `"length" if tokens == max_new_tokens else "stop"` — a token-count comparison, never an observed EOS | `truncation_rate` |
| `generated_tokens` is a tensor width | `policies.py:283` takes `generated.shape[-1]`, which includes EOS | `average_generated_tokens` |

A fourth is arithmetic rather than semantic: `metrics.py:27-33` sums floats in
completion order, so concurrent completion changes the summation order and the
last bits of every averaged metric.

So the criterion is agreement within a stated tolerance, with the per-task
disagreement list published.

## The tolerance, expressed in tasks

Score tolerance has to be stated in tasks and then converted, because BFCL
scoring is all-or-nothing per task: `bfcl.py:195,212,217` each return
`AdapterResult(0.0, False)`, and `:219` is the only `1.0`. One flipped task
moves a category score by exactly `1/N`.

Measured task counts, from the pinned checkout at
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`:

| Set | Tasks | One flip is worth |
|---|---|---|
| `multi_turn_base` | 200 | 0.005 |
| `multi_turn_miss_func` | 200 | 0.005 |
| `multi_turn_miss_param` | 200 | 0.005 |
| `multi_turn_long_context` | 200 | 0.005 |
| `multi_turn_overall` | 800 | 0.00125 |
| `envscaler_heldout` (dev) | 80 | ≤0.0125 |

EnvScaler is **not** all-or-nothing — `verifier.py:246` returns
`round(passed / total, 4)` over K checks, K ranging 2 to 445 — so a single task's
contribution there is bounded by `1/80` rather than equal to it. The rounding to
four places also means a per-task EnvScaler score is quantized at 1e-4, below any
tolerance stated here.

**Registered budget, per category:**

- **Score tolerance: 3 tasks.** `multi_turn_*`: 0.015 absolute. `overall`:
  0.00375. `envscaler_heldout`: 0.0375.
- **Per-task disagreement cap: 3 tasks per category, 8 across the run.** Counted
  at task-id level, not on the aggregate: two flips in opposite directions cancel
  in the mean and must not read as agreement.
- **Clustering rule.** Disagreements concentrated in one category fail the phase
  regardless of the aggregate. Three flips in `miss_func` is a parse regression;
  three spread across four categories is sampling noise at the decode boundary.
- **`truncation_rate` and `average_generated_tokens` are excluded from the
  agreement gate** and reported side by side instead. Both are computed from
  quantities the current code synthesizes, so the vLLM path is expected to differ
  and, on `finish_reason`, to be more correct.
- **Publication rule: the disagreement list is always published, never
  summarized.** Task id, both scores, and the terminal reason on each side. A
  tolerance wide enough to absorb any divergence proves nothing; the list is what
  makes the tolerance auditable.

## Baseline

Not captured. `evaluate` needs weights, and no checkpoint exists — `artifacts/`
holds only `data/` and empty `evaluation/` and `rollout/` directories, and
`artifacts/evaluation/final_results.md` is all `PENDING`.

The local card is a 4 GB RTX 3050, below the plan's single-L4 floor, so the
baseline is deferred to Phase 10 step 2 on a Colab L4. The exact command, to be
run against the first checkpoint that exists:

```sh
smolqwen evaluate \
  --checkpoint artifacts/models/qwen3.5-2b-sft-merged \
  --revision <checkpoint-commit-sha> \
  --tag baseline-preref \
  --adapter envscaler_heldout \
  --profile l4
```

Recorded environment for reproducing it: `torch==2.11.0+cu130`,
`vllm==0.26.0` (absent locally — `serve`/`colab` extras only), CPython 3.13 on
Colab. Phase 4's agreement criterion is therefore settled in Phase 10, not
before, and Phase 10 step 2 re-captures the baseline on the same card as the
vLLM run so the comparison is not across two machines.
