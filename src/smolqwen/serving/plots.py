"""Publication-quality Matplotlib figures for the bounded serving study."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from smolqwen.serving.analysis import (
    PROFILE_NAMES,
    AnalysisResult,
    PointStats,
    matched_precision_pairs,
)

_COLORS = {"bf16": "#2563EB", "fp8": "#EA580C"}
_FIGURES = (
    ("p95_ttft_ms", "p95 TTFT (ms)", "throughput-vs-p95-ttft.png"),
    ("p95_tpot_ms", "p95 TPOT (ms)", "throughput-vs-p95-tpot.png"),
)
_SOURCE_NOTE = "Official Qwen3.5-2B · NVIDIA L4 · vLLM 0.29.0 · one observation per point"


def _style(axis: Axes) -> None:
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#334155", labelsize=9)
    axis.title.set_color("#0F172A")
    axis.xaxis.label.set_color("#334155")
    axis.yaxis.label.set_color("#334155")


def _save(figure: Figure, target: Path) -> None:
    figure.set_facecolor("white")
    figure.text(0.01, 0.01, _SOURCE_NOTE, color="#64748B", fontsize=7)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")


def _write_frontiers(result: AnalysisResult, directory: Path) -> list[Path]:
    written: list[Path] = []
    for metric, x_label, filename in _FIGURES:
        figure = Figure(figsize=(9, 5.4))
        axis = figure.subplots()
        for precision, axes in result.frontiers.items():
            points = [point for point in result.eligible if point.precision == precision]
            axis.scatter(
                [point.medians[metric] for point in points],
                [point.medians["output_throughput"] for point in points],
                color=_COLORS[precision],
                alpha=0.28,
                s=45,
                edgecolors="none",
                label=f"{precision.upper()} observations",
            )
            frontier = axes["ttft" if metric == "p95_ttft_ms" else "tpot"]
            axis.plot(
                [point.medians[metric] for point in frontier],
                [point.medians["output_throughput"] for point in frontier],
                color=_COLORS[precision],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label=f"{precision.upper()} frontier",
            )
        for name in PROFILE_NAMES:
            point = result.selections[name]
            axis.annotate(
                name,
                (point.medians[metric], point.medians["output_throughput"]),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=8,
                color="#0F172A",
            )
        axis.set(
            title=f"Throughput–latency frontier: {x_label.removeprefix('p95 ')}",
            xlabel=x_label,
            ylabel="Output throughput (tokens/s)",
        )
        axis.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
        _style(axis)
        figure.subplots_adjust(bottom=0.15)
        target = directory / filename
        _save(figure, target)
        written.append(target)
    return written


def _write_uplift(result: AnalysisResult, directory: Path) -> Path:
    pairs = sorted(matched_precision_pairs(result.points), key=lambda pair: pair[2])
    labels = [
        f"c{bf16.concurrency} · s{bf16.max_num_seqs} · t{bf16.max_num_batched_tokens}"
        for bf16, _, _ in pairs
    ]
    values = [uplift for _, _, uplift in pairs]
    middle = sorted(values)[len(values) // 2 - 1 : len(values) // 2 + 1]
    median_value = sum(middle) / len(middle)

    figure = Figure(figsize=(9, 6.3))
    axis = figure.subplots()
    positions = list(range(len(values)))
    axis.barh(positions, values, color="#F97316", alpha=0.86, height=0.64)
    axis.axvline(median_value, color="#0F172A", linestyle="--", linewidth=1.3)
    axis.text(
        median_value + 0.25,
        len(values) - 0.2,
        f"median {median_value:.2f}%",
        color="#0F172A",
        fontsize=8,
        va="top",
    )
    axis.set_yticks(positions, labels)
    axis.set(
        title="FP8 output-throughput uplift across matched configurations",
        xlabel="Uplift over BF16 (%)",
        ylabel="Concurrency · max sequences · token budget",
    )
    axis.set_xlim(0, max(values) + 4)
    _style(axis)
    figure.subplots_adjust(left=0.24, bottom=0.12)
    target = directory / "matched-fp8-throughput-uplift.png"
    _save(figure, target)
    return target


def _canonical_ladder(points: Sequence[PointStats], precision: str) -> list[PointStats]:
    configs = (
        (2048, 16, 1),
        (2048, 16, 4),
        (2048, 16, 16),
        (2048, 32, 32),
        (2048, 64, 64),
        (2048, 128, 128),
    )
    keyed = {
        (point.max_num_batched_tokens, point.max_num_seqs, point.concurrency): point
        for point in points
        if point.precision == precision and point.valid
    }
    return [keyed[config] for config in configs]


def _write_scaling(result: AnalysisResult, directory: Path) -> Path:
    panels = (
        ("output_throughput", "Output throughput", "tokens/s"),
        ("p95_ttft_ms", "p95 TTFT", "ms"),
        ("p95_tpot_ms", "p95 TPOT", "ms/token"),
    )
    figure = Figure(figsize=(12.5, 4.4))
    axes = figure.subplots(1, 3)
    positions = list(range(6))
    labels = ["1", "4", "16", "32", "64", "128"]
    for axis, (metric, title, unit) in zip(axes, panels, strict=True):
        for precision in ("bf16", "fp8"):
            ladder = _canonical_ladder(result.points, precision)
            axis.plot(
                positions,
                [point.medians[metric] for point in ladder],
                color=_COLORS[precision],
                linewidth=2.3,
                marker="o",
                markersize=5,
                label=precision.upper(),
            )
        axis.set_xticks(positions, labels)
        axis.set(title=title, xlabel="Concurrency", ylabel=unit)
        _style(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Concurrency scaling on the canonical 2,048-token scheduler ladder", y=1.02)
    figure.subplots_adjust(bottom=0.2, wspace=0.34)
    target = directory / "concurrency-scaling.png"
    _save(figure, target)
    return target


def write_figures(result: AnalysisResult, output_dir: Path | str) -> list[Path]:
    """Write the two latency frontiers and two matched-comparison figures."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        *_write_frontiers(result, directory),
        _write_uplift(result, directory),
        _write_scaling(result, directory),
    ]
