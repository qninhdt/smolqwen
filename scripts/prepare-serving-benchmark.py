#!/usr/bin/env python
"""Freeze the serving benchmark: regenerate the committed spec files and stamp a manifest.

This is a thin runtime wrapper. The sweep axes, category validation, and manifest
hashing all live in ``smolqwen.serving.benchmark_spec`` and are unit-tested there;
this script only regenerates the committed bounded spec files (``sweep.json``
and ``study-points.json``) and writes the immutable
``workload-manifest.json`` the sweep and the analyzer key on.

The traffic is vLLM's built-in BFCL loader resolved from the Hub at run time, so
there is no local file to render first. Provenance is the dataset repo pinned to a
revision plus the sha256 of each requested category file, resolved here via
``huggingface_hub`` (needs Hub access, no GPU):

    python scripts/prepare-serving-benchmark.py \
        --model Qwen/Qwen3.5-2B --tokenizer-revision <sha>

Pass ``--dataset-revision`` to freeze against a specific dataset commit instead of
the current Hub HEAD, or ``--regenerate-spec-only`` to rewrite just the spec files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from smolqwen.serving.benchmark_spec import (
    BFCL_CATEGORIES,
    DATASET_REPO,
    MODEL_REPO,
    MODEL_REVISION,
    NUM_PROMPTS,
    build_workload_manifest,
    render_study_points,
    render_sweep_json,
    validate_categories,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = _REPO_ROOT / "benchmarks" / "serving"
_SWEEP_JSON = _BENCH_DIR / "sweep.json"
_STUDY_POINTS = _BENCH_DIR / "study-points.json"
_DEFAULT_MANIFEST = _REPO_ROOT / "artifacts" / "serving" / "workload-manifest.json"


def _regenerate_spec_files() -> None:
    """Rewrite every committed, module-derived spec file deterministically."""
    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in (
        (_SWEEP_JSON, render_sweep_json()),
        (_STUDY_POINTS, render_study_points()),
    ):
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_dataset(categories: list[str], revision: str | None) -> tuple[str, dict[str, str]]:
    """Pin the Hub dataset commit and hash each requested category's file.

    Files are named ``BFCL_v3_<category>.json`` in the dataset repo; the manifest
    keys shas by bare category so it stays decoupled from that naming.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    resolved = revision or api.dataset_info(DATASET_REPO).sha
    if not resolved:
        raise SystemExit(f"could not resolve a commit for dataset {DATASET_REPO}")
    shas: dict[str, str] = {}
    for category in categories:
        local = hf_hub_download(
            DATASET_REPO,
            f"BFCL_v3_{category}.json",
            repo_type="dataset",
            revision=resolved,
        )
        shas[category] = _sha256_bytes(Path(local).read_bytes())
    return resolved, shas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-spec-only",
        action="store_true",
        help="only rewrite the committed benchmarks/serving/*.json spec files and exit",
    )
    parser.add_argument("--model", default=MODEL_REPO)
    parser.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument(
        "--categories",
        default=",".join(BFCL_CATEGORIES),
        help="comma-separated BFCL categories (defaults to the frozen set)",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="pin a specific dataset commit; defaults to the current Hub HEAD",
    )
    parser.add_argument("--output", type=Path, default=None, help="manifest output path")
    args = parser.parse_args()

    _regenerate_spec_files()
    if args.regenerate_spec_only:
        return 0

    categories = list(validate_categories(args.categories.split(",")))
    dataset_revision, dataset_file_sha256 = _resolve_dataset(categories, args.dataset_revision)
    manifest = build_workload_manifest(
        dataset_revision=dataset_revision,
        categories=categories,
        num_prompts=args.num_prompts,
        seed=args.seed,
        model=args.model,
        tokenizer_revision=args.tokenizer_revision,
        generation={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "ignore_eos": False},
        dataset_file_sha256=dataset_file_sha256,
    )
    output = args.output or _DEFAULT_MANIFEST
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} (manifest_hash={manifest['manifest_hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
