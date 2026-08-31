"""Build a deterministic representative shard from the prepared SFT train set.

The source shard is too large to upload for every Colab measurement.  Reservoir
sampling gives every record the same inclusion probability while keeping memory
bounded.  The selected records are shuffled with the same seeded RNG so the
benchmark does not inherit trajectory-local ordering from the source file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def sample_records(source: Path, *, count: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    rows_seen = 0
    with source.open(encoding="utf-8") as handle:
        for rows_seen, line in enumerate(handle, start=1):
            record = json.loads(line)
            record["benchmark_source_index"] = rows_seen - 1
            if len(reservoir) < count:
                reservoir.append(record)
                continue
            replacement = rng.randrange(rows_seen)
            if replacement < count:
                reservoir[replacement] = record
    if rows_seen < count:
        raise ValueError(f"requested {count} records, but {source} has only {rows_seen}")
    rng.shuffle(reservoir)
    return reservoir, rows_seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/data/sft/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    records, rows_seen = sample_records(args.source, count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for benchmark_id, record in enumerate(records):
            record["benchmark_id"] = benchmark_id
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    lengths = [len(row["prompt_ids"]) + len(row["completion_ids"]) for row in records]
    supervised = [sum(row["loss_mask"]) for row in records]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_rows": rows_seen,
                "sample_rows": len(records),
                "seed": args.seed,
                "valid_tokens": sum(lengths),
                "supervised_tokens": sum(supervised),
                "length_min": min(lengths),
                "length_max": max(lengths),
                "length_mean": sum(lengths) / len(lengths),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
