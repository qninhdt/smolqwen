"""Frozen serving-benchmark specification and workload manifest (Phase 5).

The approved non-Cartesian point map and BFCL category set are owned here so one
source defines `benchmarks/serving/sweep.json`, `study-points.json`, and the
recorded workload manifest. A committed JSON that drifts from these constants is
caught by a test, not discovered mid-sweep.

Everything in this module is pure and deterministic: it never fetches a dataset,
starts a server, or touches a GPU. The prepare script wires the download; this
module validates axes and hashes the workload manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PRECISIONS = ("bf16", "fp8")

# vLLM's built-in BFCL loader replays only these static (non-multi-turn)
# categories. Multi-turn quality stays in the evaluation pipeline, not here.
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
_MULTI_TURN_MARKER = "multi_turn"

# The dataset the sweep freezes traffic against. vLLM's built-in BFCL loader
# resolves it from the Hub at run time (`--dataset-name hf --dataset-path <repo>`),
# carries each sample's `tools` schema and chat history, and applies the chat
# template server-side. There is no pre-rendered local file to hash.
DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
MODEL_REPO = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

# Request count per measurement point. Frozen so every (server config, concurrency)
# point sends the same number of BFCL requests; `--num-prompts` is passed verbatim.
NUM_PROMPTS = 200

DEFAULT_NUM_RUNS = 1
CONTEXT_WINDOW = 32768

# Resource-bounded L4 study requested for the first pass. Each tuple is one
# server configuration plus its paired load; it is deliberately not a Cartesian
# product because the useful load differs by scheduler capacity.
STUDY_POINTS = (
    ("tok2048-seq16-c1", 2048, 16, 1),
    ("tok8192-seq16-c1", 8192, 16, 1),
    ("tok2048-seq16-c4", 2048, 16, 4),
    ("tok8192-seq16-c4", 8192, 16, 4),
    ("tok2048-seq16-c16", 2048, 16, 16),
    ("tok8192-seq16-c16", 8192, 16, 16),
    ("tok2048-seq32-c32", 2048, 32, 32),
    ("tok4096-seq32-c32", 4096, 32, 32),
    ("tok8192-seq32-c32", 8192, 32, 32),
    ("tok2048-seq64-c64", 2048, 64, 64),
    ("tok4096-seq64-c64", 4096, 64, 64),
    ("tok8192-seq64-c64", 8192, 64, 64),
    ("tok16384-seq64-c64", 16384, 64, 64),
    ("tok2048-seq128-c128", 2048, 128, 128),
    ("tok8192-seq128-c128", 8192, 128, 128),
    ("tok8192-seq64-c128", 8192, 64, 128),
)

# precision -> the ServingProfileConfig weight/KV fields it implies. Kept here as
# the single source of truth; `analysis.PRECISION_SETTINGS` must match (a drift
# test asserts it) so a generated profile and a swept server agree on dtype.
PRECISION_SETTINGS: Mapping[str, Mapping[str, object]] = {
    "bf16": {"dtype": "bfloat16", "quantization": None, "kv_cache_dtype": "auto"},
    "fp8": {"dtype": "bfloat16", "quantization": "fp8", "kv_cache_dtype": "fp8"},
}


class SpecError(Exception):
    """Raised when the sweep axes or workload manifest are inconsistent."""


@dataclass(frozen=True)
class ServerConfig:
    """One vLLM server configuration in the sweep (a clean-start boundary)."""

    precision: str
    max_num_seqs: int
    max_num_batched_tokens: int

    def as_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }


def build_server_configs() -> list[ServerConfig]:
    """Unique clean-start configurations required by the approved point map."""
    configs: list[ServerConfig] = []
    seen: set[ServerConfig] = set()
    for precision in PRECISIONS:
        for _, batched, seqs, _ in STUDY_POINTS:
            config = ServerConfig(precision, seqs, batched)
            if config not in seen:
                seen.add(config)
                configs.append(config)
    return configs


def sweep_point_count() -> int:
    """Total measurements in the approved 16x2 study."""
    return len(PRECISIONS) * len(STUDY_POINTS)


def validate_categories(categories: Sequence[str]) -> tuple[str, ...]:
    """Reject unsupported (multi-turn) categories; return a sorted, deduplicated tuple."""
    if not categories:
        raise SpecError("at least one BFCL category is required")
    if any(_MULTI_TURN_MARKER in c for c in categories):
        raise SpecError("multi-turn categories belong to evaluation, not the serving sweep")
    unknown = [c for c in categories if c not in BFCL_CATEGORIES]
    if unknown:
        raise SpecError(f"unsupported BFCL categories for the built-in loader: {sorted(unknown)}")
    return tuple(sorted(set(categories)))


def build_sweep_spec(*, num_runs: int = DEFAULT_NUM_RUNS) -> dict[str, object]:
    """The canonical, committed ``sweep.json`` content (decision-only, deterministic)."""
    if num_runs < 1:
        raise SpecError("num_runs must be at least 1")
    return {
        "points": build_study_points(),
        "categories": list(BFCL_CATEGORIES),
        "context_window": CONTEXT_WINDOW,
        "dataset_name": "hf",
        "dataset_repo": DATASET_REPO,
        "num_prompts": NUM_PROMPTS,
        "num_runs": num_runs,
        "server_config_count": len(build_server_configs()),
        "point_count": sweep_point_count(),
        "workload_manifest": "artifacts/serving/workload-manifest.json",
        "note": "approved non-Cartesian 16x2 study; one observation per point",
    }


def render_sweep_json(*, num_runs: int = DEFAULT_NUM_RUNS) -> str:
    """Deterministic text for the committed ``benchmarks/serving/sweep.json``."""
    return json.dumps(build_sweep_spec(num_runs=num_runs), indent=2, sort_keys=True) + "\n"


def build_study_points() -> list[dict[str, object]]:
    """Expand the requested 16-point load map across BF16 and FP8."""
    return [
        {
            "_benchmark_name": f"{precision}-{name}",
            "precision": precision,
            "max_num_batched_tokens": batched_tokens,
            "max_num_seqs": max_num_seqs,
            "max_concurrency": concurrency,
        }
        for precision in PRECISIONS
        for name, batched_tokens, max_num_seqs, concurrency in STUDY_POINTS
    ]


def render_study_points() -> str:
    return json.dumps(build_study_points(), indent=2, sort_keys=True) + "\n"


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_workload_manifest(
    *,
    dataset_revision: str,
    categories: Sequence[str],
    num_prompts: int,
    seed: int,
    model: str,
    tokenizer_revision: str,
    generation: Mapping[str, object],
    dataset_file_sha256: Mapping[str, str],
    dataset_repo: str = DATASET_REPO,
) -> dict[str, object]:
    """Assemble the immutable workload manifest and stamp it with a content hash.

    The native BFCL loader resolves its rows from the Hub, so provenance is the
    dataset repo at a pinned ``dataset_revision`` plus the sha256 of each requested
    category file (``dataset_file_sha256`` is keyed by category, e.g. ``simple``),
    not a locally rendered file. ``manifest_hash`` covers every field except itself,
    so two prepare runs over the same frozen inputs produce the same hash and the
    analyzer can reject aggregating points recorded under different manifests.
    """
    normalized = validate_categories(categories)
    if num_prompts < 1:
        raise SpecError("num_prompts must be at least 1")
    recorded = set(dataset_file_sha256)
    if recorded != set(normalized):
        raise SpecError(
            "dataset_file_sha256 must record exactly one sha per requested category; "
            f"categories={list(normalized)} shas={sorted(recorded)}"
        )
    body: dict[str, object] = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "categories": list(normalized),
        "num_prompts": int(num_prompts),
        "seed": seed,
        "model": model,
        "tokenizer_revision": tokenizer_revision,
        "generation": dict(sorted(generation.items())),
        "dataset_file_sha256": dict(sorted(dataset_file_sha256.items())),
    }
    manifest_hash = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    return {"manifest_hash": manifest_hash, **body}
