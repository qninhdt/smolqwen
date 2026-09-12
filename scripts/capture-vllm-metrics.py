#!/usr/bin/env python
"""Save one direct vLLM Prometheus snapshot after a sweep repetition."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("usage: capture-vllm-metrics.py OUTPUT_PREFIX")
    prefix = Path(args[0])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    counter_path = prefix.with_name(prefix.name + ".counter")
    try:
        run_number = int(counter_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        run_number = 0
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=30) as response:
        metrics = response.read()
    output = prefix.with_name(f"{prefix.name}-after-run-{run_number}.prom")
    output.write_bytes(metrics)
    counter_path.write_text(str(run_number + 1) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
