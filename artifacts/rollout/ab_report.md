# Rollout A/B report

## Equivalence (scripted policy, real pool)

PASS — 1 episodes, oracle 0.333s, async 0.065s.

## Scripted diagnostic comparison

Not the accepted TRL turn-synchronous A/B baseline.

| path | profile | episodes/hour | tokens/s | gpu util mean | gpu util peak | mean reward | notes |
|---|---|---|---|---|---|---|---|
| serial_oracle | l4 | 11054.1 | 0 | 0.0% | 0.0% | 1.0000 | serial semantic oracle; not TRL baseline |
| async | l4 | 101702.0 | 11272 | 0.0% | 0.0% | 1.0000 | scripted backend; the vLLM row is the trainer run |
## Async timeline (l4)

```
wall 0.04s over 1 episodes (101702.0 episodes/h if sustained)
  generation        0.00s    2.0%  n=4     mean=   0.18ms max=    0.23ms
  tokenization      0.02s   62.7%  n=4     mean=   5.55ms max=    6.23ms
  env.create        0.00s    1.3%  n=1     mean=   0.45ms max=    0.45ms
  env.step          0.00s    4.1%  n=3     mean=   0.49ms max=    0.52ms
  env.destroy       0.00s    1.4%  n=1     mean=   0.50ms max=    0.50ms
  parse             0.00s    1.0%  n=4     mean=   0.09ms max=    0.10ms
  verifier          0.00s    3.7%  n=1     mean=   1.29ms max=    1.29ms
  scheduling        0.01s   23.9%  n=7     mean=   0.00ms max=    0.00ms
  terminals: {'final_answer': 1}
```

## Colocated-vLLM A/B

PENDING — CUDA is visible, but a card alone is not sufficient. These rows
must run through the separately constructed Phase 7 trainers so the
timeline includes real LoRA weight sync and prefix-cache invalidation.
The scripted rows above are not presented as GPU measurements.
