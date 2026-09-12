"""Prometheus names and parser shared by serving analysis and dashboards."""

from __future__ import annotations

# vLLM v1 Prometheus series the capture depends on. Verify against the running
# 0.29.0 engine during Phase 6 preflight and fix here if upstream renamed one.
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PREEMPTIONS = "vllm:num_preemptions_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Sum every sample value by exact series name from a Prometheus exposition body.

    Comment/``# HELP``/``# TYPE`` lines are ignored. Values are summed across label
    sets so a single-model server collapses to one value per counter or gauge;
    histogram ``_bucket``/``_sum``/``_count`` names are preserved verbatim.
    """
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value_part = line.rpartition(" ")
        if not name_part:
            continue
        name = name_part.split("{", 1)[0].strip()
        try:
            value = float(value_part)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals
