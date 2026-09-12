# Official Qwen single-L4 serving results

The approved study contains 32 single observations (16 configurations in BF16
and FP8). Values are not medians; no variance claim is made. Goodput is absent
because no production SLO was supplied. Admission limits are provisional.

Workload manifest: `9b26de5440dc986f15c488903f637636d1a5800f93da6671e3dca239fa9769f6`
Summary hash: `853dcc361e6cab71a4cf40465020021df43684a8a76dfcd57ee48a72a5062b58`
Prompt length p95: `808` tokens
Raw point bundles: `artifacts/serving/sweeps/direct-vllm029-official-16x2/`
Runtime validation: `artifacts/serving/validation/validation.json`

## Reproducibility

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.5-2B` (pinned revision) |
| GPU | NVIDIA L4 24 GB |
| Runtime | vLLM 0.29.0, Torch 2.13.0+cu130 |
| Workload | Fixed BFCL serving workload, 200 successful requests/config |
| Study | 16 selected configs × BF16/FP8 × 1 observation |

## Selected profiles

| Profile | Precision | tok | seq | c | output tok/s | p95 TTFT ms | p95 TPOT ms | queued reqs | queued tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latency | fp8 | 2048 | 16 | 1 | 79.85 | 64.1 | 12.22 | 32 | 12928 |
| balanced | fp8 | 2048 | 32 | 32 | 1055.57 | 784.0 | 26.81 | 64 | 25856 |
| throughput | fp8 | 2048 | 128 | 128 | 1490.55 | 4176.7 | 84.89 | 256 | 103424 |

## Matched precision comparison

Across all 16 matched configurations, FP8's median output-throughput
uplift is **25.94%**.

| tok | seq | c | BF16 output tok/s | FP8 output tok/s | FP8 uplift |
|---:|---:|---:|---:|---:|---:|
| 2048 | 16 | 1 | 61.30 | 79.85 | +30.25% |
| 2048 | 16 | 4 | 210.49 | 277.26 | +31.72% |
| 2048 | 16 | 16 | 589.47 | 749.27 | +27.11% |
| 2048 | 32 | 32 | 838.53 | 1055.57 | +25.88% |
| 2048 | 64 | 64 | 1068.67 | 1307.06 | +22.31% |
| 2048 | 128 | 128 | 1210.80 | 1490.55 | +23.10% |
| 4096 | 32 | 32 | 836.90 | 1046.52 | +25.05% |
| 4096 | 64 | 64 | 1052.34 | 1306.30 | +24.13% |
| 8192 | 16 | 1 | 61.30 | 79.61 | +29.87% |
| 8192 | 16 | 4 | 210.63 | 279.70 | +32.79% |
| 8192 | 16 | 16 | 588.67 | 747.06 | +26.91% |
| 8192 | 32 | 32 | 836.79 | 1054.97 | +26.07% |
| 8192 | 64 | 64 | 1064.71 | 1305.30 | +22.60% |
| 8192 | 64 | 128 | 1082.30 | 1363.61 | +25.99% |
| 8192 | 128 | 128 | 1209.18 | 1489.10 | +23.15% |
| 16384 | 64 | 64 | 1053.03 | 1322.64 | +25.60% |

## All measurements

| Precision | tok | seq | c | req/s | output tok/s | p95 TTFT ms | p99 TTFT ms | p95 TPOT ms | p99 TPOT ms | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| bf16 | 2048 | 16 | 1 | 0.4447 | 61.30 | 107.9 | 111.5 | 15.94 | 15.98 | yes |
| bf16 | 2048 | 16 | 4 | 1.5296 | 210.49 | 135.2 | 237.6 | 18.68 | 19.14 | yes |
| bf16 | 2048 | 16 | 16 | 4.2624 | 589.47 | 367.9 | 628.2 | 24.68 | 25.75 | yes |
| bf16 | 2048 | 32 | 32 | 6.0280 | 838.53 | 934.4 | 1191.7 | 34.12 | 40.68 | yes |
| bf16 | 2048 | 64 | 64 | 7.7063 | 1068.67 | 2018.2 | 2370.3 | 56.91 | 70.60 | yes |
| bf16 | 2048 | 128 | 128 | 8.6743 | 1210.80 | 4983.7 | 5294.4 | 104.37 | 120.08 | yes |
| bf16 | 4096 | 32 | 32 | 6.0000 | 836.90 | 984.4 | 1178.6 | 33.26 | 40.19 | yes |
| bf16 | 4096 | 64 | 64 | 7.6540 | 1052.34 | 2144.9 | 2334.3 | 59.47 | 69.75 | yes |
| bf16 | 8192 | 16 | 1 | 0.4448 | 61.30 | 108.3 | 111.4 | 15.93 | 15.95 | yes |
| bf16 | 8192 | 16 | 4 | 1.5272 | 210.63 | 124.4 | 198.6 | 18.75 | 19.39 | yes |
| bf16 | 8192 | 16 | 16 | 4.2884 | 588.67 | 613.1 | 662.7 | 23.89 | 24.53 | yes |
| bf16 | 8192 | 32 | 32 | 6.0262 | 836.79 | 1169.9 | 1176.6 | 33.29 | 35.73 | yes |
| bf16 | 8192 | 64 | 64 | 7.6436 | 1064.71 | 2306.7 | 2314.3 | 55.05 | 69.36 | yes |
| bf16 | 8192 | 64 | 128 | 7.7498 | 1082.30 | 7926.8 | 8674.3 | 57.87 | 75.29 | yes |
| bf16 | 8192 | 128 | 128 | 8.6509 | 1209.18 | 4779.4 | 4782.6 | 115.05 | 132.12 | yes |
| bf16 | 16384 | 64 | 64 | 7.5670 | 1053.03 | 2162.2 | 2298.1 | 56.70 | 69.80 | yes |
| fp8 | 2048 | 16 | 1 | 0.5675 | 79.85 | 64.1 | 68.8 | 12.22 | 12.23 | yes |
| fp8 | 2048 | 16 | 4 | 1.9338 | 277.26 | 115.1 | 200.9 | 14.42 | 15.08 | yes |
| fp8 | 2048 | 16 | 16 | 5.0597 | 749.27 | 371.9 | 540.2 | 19.44 | 21.67 | yes |
| fp8 | 2048 | 32 | 32 | 7.2389 | 1055.57 | 784.0 | 1006.4 | 26.81 | 32.05 | yes |
| fp8 | 2048 | 64 | 64 | 9.2079 | 1307.06 | 1766.3 | 2023.3 | 48.71 | 55.50 | yes |
| fp8 | 2048 | 128 | 128 | 10.5848 | 1490.55 | 4176.7 | 4468.3 | 84.89 | 97.68 | yes |
| fp8 | 4096 | 32 | 32 | 7.3463 | 1046.52 | 881.9 | 1009.1 | 27.42 | 30.71 | yes |
| fp8 | 4096 | 64 | 64 | 9.2374 | 1306.30 | 1727.4 | 1950.3 | 50.25 | 58.23 | yes |
| fp8 | 8192 | 16 | 1 | 0.5605 | 79.61 | 65.1 | 69.1 | 12.26 | 12.27 | yes |
| fp8 | 8192 | 16 | 4 | 1.9145 | 279.70 | 115.8 | 191.3 | 14.34 | 14.93 | yes |
| fp8 | 8192 | 16 | 16 | 5.1165 | 747.06 | 570.3 | 573.9 | 19.74 | 20.67 | yes |
| fp8 | 8192 | 32 | 32 | 7.2895 | 1054.97 | 1049.1 | 1051.5 | 26.35 | 32.06 | yes |
| fp8 | 8192 | 64 | 64 | 9.2427 | 1305.30 | 1654.4 | 1902.4 | 48.48 | 56.39 | yes |
| fp8 | 8192 | 64 | 128 | 9.0871 | 1363.61 | 7034.4 | 7934.3 | 49.69 | 61.84 | yes |
| fp8 | 8192 | 128 | 128 | 10.4678 | 1489.10 | 4019.0 | 4022.5 | 100.02 | 109.56 | yes |
| fp8 | 16384 | 64 | 64 | 9.2308 | 1322.64 | 1935.1 | 1936.8 | 43.33 | 55.84 | yes |

## Findings

- FP8 improves output throughput at every matched point; the median uplift is 25.94%.
- The practical throughput/latency knee appears around concurrency 16–32.
- Concurrency 64→128 adds modest throughput while sharply increasing TTFT and TPOT.
- At concurrency 128, limiting `max_num_seqs` to 64 lowers TPOT but moves delay into
  scheduler wait time, increasing TTFT.

## Limitations

- One observation per point; there is no repetition variance estimate.
- Prometheus files captured after each run prove counters and clean end state;
  they are not peak KV/running/waiting measurements.
- Native vLLM wrote every complete raw JSON before its optional combined-table
  step failed because the first Colab runtime lacked pandas. The installer is
  fixed; no benchmark was rerun.
- FP8 is runtime quantization of the same pinned official checkpoint; serving
  quality comparison is outside this study.
