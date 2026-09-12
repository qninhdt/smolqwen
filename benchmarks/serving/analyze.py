#!/usr/bin/env python
"""CLI for the serving-sweep analyzer.

Thin wrapper around ``smolqwen.serving.analysis``: it only parses arguments and
wires files. All computation, and every unit test, lives in the importable,
type-checked module so this script stays free of logic.

    python benchmarks/serving/analyze.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smolqwen.serving.analysis import (
    SLO,
    analyze_sweep,
    load_vllm_study_runs,
    write_profiles,
    write_report,
    write_summary,
)
from smolqwen.serving.plots import write_figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("artifacts/serving/sweeps/direct-vllm029-official-16x2"),
        help="native vLLM study artifact directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/serving/workload-manifest.json"),
        help="frozen workload manifest",
    )
    parser.add_argument("--slo-ttft-ms", type=float, default=None, help="optional p95 TTFT SLO")
    parser.add_argument("--slo-tpot-ms", type=float, default=None, help="optional p95 TPOT SLO")
    parser.add_argument(
        "--prefill-tokens-per-s",
        type=float,
        default=None,
        help="measured prefill throughput; enables the SLO-based token budget",
    )
    parser.add_argument(
        "--queue-slots-factor",
        type=float,
        default=1.0,
        help="finite queue depth as a multiple of active capacity",
    )
    parser.add_argument("--serving-dir", type=Path, default=Path("configs/serving"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/serving"))
    args = parser.parse_args()

    if (args.slo_ttft_ms is None) != (args.slo_tpot_ms is None):
        parser.error("--slo-ttft-ms and --slo-tpot-ms must be given together")
    slo = (
        None
        if args.slo_ttft_ms is None
        else SLO(p95_ttft_ms=args.slo_ttft_ms, p95_tpot_ms=args.slo_tpot_ms)
    )

    runs, prompt_len_p95 = load_vllm_study_runs(args.runs, manifest_path=args.manifest)
    result = analyze_sweep(
        runs,
        num_runs=1,
        prompt_len_p95=prompt_len_p95,
        slo=slo,
        prefill_tokens_per_s=args.prefill_tokens_per_s,
        queue_slots_factor=args.queue_slots_factor,
    )
    profiles = write_profiles(result, args.serving_dir)
    summary = write_summary(result, args.output_dir / "summary.json")
    figures = write_figures(result, args.output_dir)
    report = write_report(result, args.output_dir / "report.md")
    print(f"analyzed {len(result.points)} points, {len(result.eligible)} eligible")
    for name, point in result.selections.items():
        limits = result.admission[name]
        flag = " (provisional)" if limits.provisional else ""
        print(
            f"  {name}: precision={point.precision} concurrency={point.concurrency} "
            f"max_num_seqs={point.max_num_seqs} "
            f"queued_reqs={limits.max_num_queued_reqs} "
            f"queued_tokens={limits.max_num_queued_tokens}{flag}"
        )
    print(
        f"wrote {summary}, {report}, {len(figures)} figures, and "
        f"{len(profiles)} profiles under {args.serving_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
