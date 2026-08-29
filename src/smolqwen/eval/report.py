"""Stable JSON and Markdown evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smolqwen.eval.manifest import EvalManifest, assert_comparable


def write_report(
    output_dir: Path | str,
    *,
    tag: str,
    manifest: EvalManifest,
    metrics: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "tag": tag,
        "manifest": manifest.to_dict(),
        "metrics": metrics,
    }
    json_path = target / f"{tag}.json"
    markdown_path = target / f"{tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    has_exact = any("exact_success_rate" in values for values in metrics.values())
    header = "| category | score | invalid-call rate | avg steps | avg tokens | truncation rate |"
    divider = "| --- | ---: | ---: | ---: | ---: | ---: |"
    if has_exact:
        header = header[:-1] + "| exact-success rate |"
        divider = divider[:-1] + "| ---: |"
    rows = [
        (
            "| {category} | {score:.4f} | {invalid_call_rate:.4f} | {average_steps:.2f} | "
            "{average_generated_tokens:.2f} | {truncation_rate:.4f} |".format(
                category=category, **values
            )
            + (
                (
                    f" {values['exact_success_rate']:.4f} |"
                    if "exact_success_rate" in values
                    else " — |"
                )
                if has_exact
                else ""
            )
        )
        for category, values in sorted(metrics.items())
    ]
    recorded = json.dumps(dict(manifest.recorded_free), sort_keys=True)
    markdown_path.write_text(
        "\n".join(
            [
                f"# Evaluation: {tag}",
                "",
                header,
                divider,
                *rows,
                "",
                f"Recorded-free: `{recorded}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path


def write_comparison_report(
    output_dir: Path | str,
    *,
    tags: Sequence[str],
    output_tag: str = "base_vs_sft",
) -> tuple[Path, Path]:
    """Join per-checkpoint reports after enforcing manifest comparability.

    The individual reports remain the source of truth for reruns and diffs.  A
    comparison is a derived artifact: it reads ``<tag>.json`` files, refuses an
    invariant mismatch, and prints every recorded-free execution detail beside
    the score columns.
    """

    labels = tuple(str(tag) for tag in tags)
    if len(labels) < 2:
        raise ValueError("a comparison needs at least two evaluation report tags")
    if len(set(labels)) != len(labels):
        raise ValueError("comparison report tags must be unique")

    target = Path(output_dir)
    reports: list[dict[str, Any]] = []
    manifests: list[EvalManifest] = []
    for tag in labels:
        path = target / f"{tag}.json"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation report missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            raise ValueError(f"evaluation report is malformed: {path}")
        manifest_payload = payload.get("manifest")
        if not isinstance(manifest_payload, dict):
            raise ValueError(f"evaluation report has no manifest: {path}")
        manifest = EvalManifest.from_dict(manifest_payload)
        manifests.append(manifest)
        reports.append({"tag": tag, "manifest": manifest.to_dict(), "metrics": payload["metrics"]})

    assert_comparable(*manifests)
    categories = sorted({str(category) for report in reports for category in report["metrics"]})
    metric_names = (
        "score",
        "invalid_call_rate",
        "average_steps",
        "average_generated_tokens",
        "truncation_rate",
        "exact_success_rate",
    )
    comparison_metrics: dict[str, dict[str, dict[str, float]]] = {
        report["tag"]: report["metrics"] for report in reports
    }
    payload = {
        "tag": output_tag,
        "reports": reports,
        "metrics": comparison_metrics,
        "invariant": manifests[0].invariant,
    }
    json_path = target / f"{output_tag}.json"
    markdown_path = target / f"{output_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = ["category"]
    divider = ["---"]
    for tag in labels:
        short = tag.replace("_", " ")
        for metric in metric_names:
            header.append(f"{short} {metric.replace('_', ' ')}")
            divider.append("---:")
    rows: list[str] = []
    for category in categories:
        cells = [category]
        for report in reports:
            values = report["metrics"].get(category, {})
            for metric in metric_names:
                value = values.get(metric)
                cells.append("—" if value is None else f"{float(value):.4f}")
        rows.append("| " + " | ".join(cells) + " |")

    recorded_lines = [
        f"- `{report['tag']}`: `{json.dumps(report['manifest']['recorded_free'], sort_keys=True)}`"
        for report in reports
    ]
    markdown_path.write_text(
        "\n".join(
            [
                f"# Evaluation comparison: {output_tag}",
                "",
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(divider) + " |",
                *rows,
                "",
                "Invariant: `" + json.dumps(dict(manifests[0].invariant), sort_keys=True) + "`",
                "",
                "Recorded-free execution fields:",
                *recorded_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path


# A descriptive alias for callers that already use "compare" terminology.
compare_reports = write_comparison_report
