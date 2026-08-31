"""Thin vLLM benchmark wrapper with a stable, validated result schema."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from smolqwen.config_models import ServeConfig
from smolqwen.serving.report import load_quality_result, write_serving_report
from smolqwen.serving.server import config_metadata, serving_environment


class BenchError(RuntimeError):
    """Raised when benchmark input or output cannot support an honest row."""


@dataclass(frozen=True)
class BenchResult:
    dataset: str
    concurrency: int
    successful_requests: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    e2el_p50_ms: float
    e2el_p95_ms: float
    e2el_p99_ms: float
    peak_vram_gb: float | None
    speculative_acceptance_rate: float | None
    serving: Mapping[str, Any]
    bfcl_mt_score: float | None = None
    quality_recorded_free: Mapping[str, Any] | None = None
    quality_invariant_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_config_id(serving: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(serving, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def load_normalized_rows(directory: Path | str) -> list[BenchResult]:
    return [
        BenchResult(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(directory).glob("*.json"))
    ]


def _number(payload: Mapping[str, Any], *keys: str, required: bool = True) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    if required:
        raise BenchError(f"vLLM result is missing numeric field; tried {', '.join(keys)}")
    return None


def _percentile(payload: Mapping[str, Any], metric: str, percentile: int) -> float:
    keys = [f"p{percentile}_{metric}_ms", f"{metric}_p{percentile}_ms"]
    if percentile == 50:
        keys.extend([f"median_{metric}_ms", f"{metric}_median_ms"])
    value = _number(payload, *keys)
    assert value is not None
    return value


def parse_bench_payload(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    concurrency: int,
    serving: Mapping[str, Any],
) -> BenchResult:
    """Normalize vLLM's JSON and fail if any promised latency is absent."""
    completed = _number(payload, "completed", "successful_requests")
    request_throughput = _number(payload, "request_throughput", "request_throughput_req_s")
    output_throughput = _number(payload, "output_throughput", "output_throughput_tok_s")
    total_throughput = _number(payload, "total_token_throughput", "total_token_throughput_tok_s")
    assert completed is not None
    assert request_throughput is not None
    assert output_throughput is not None
    assert total_throughput is not None
    return BenchResult(
        dataset=dataset,
        concurrency=concurrency,
        successful_requests=int(completed),
        request_throughput=float(request_throughput),
        output_throughput=float(output_throughput),
        total_token_throughput=float(total_throughput),
        ttft_p50_ms=_percentile(payload, "ttft", 50),
        ttft_p95_ms=_percentile(payload, "ttft", 95),
        ttft_p99_ms=_percentile(payload, "ttft", 99),
        tpot_p50_ms=_percentile(payload, "tpot", 50),
        tpot_p95_ms=_percentile(payload, "tpot", 95),
        tpot_p99_ms=_percentile(payload, "tpot", 99),
        itl_p50_ms=_percentile(payload, "itl", 50),
        itl_p95_ms=_percentile(payload, "itl", 95),
        itl_p99_ms=_percentile(payload, "itl", 99),
        e2el_p50_ms=_percentile(payload, "e2el", 50),
        e2el_p95_ms=_percentile(payload, "e2el", 95),
        e2el_p99_ms=_percentile(payload, "e2el", 99),
        peak_vram_gb=_number(payload, "peak_vram_gb", "peak_gpu_memory_gb", required=False),
        speculative_acceptance_rate=_number(
            payload,
            "spec_decode_acceptance_rate",
            "speculative_acceptance_rate",
            required=False,
        ),
        serving=dict(serving),
    )


def parse_bench_file(
    path: Path | str,
    *,
    dataset: str,
    concurrency: int,
    serving: Mapping[str, Any],
) -> BenchResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise BenchError(f"empty vLLM result list: {path}")
        payload = payload[-1]
    if not isinstance(payload, Mapping):
        raise BenchError(f"vLLM result must be an object: {path}")
    return parse_bench_payload(
        payload,
        dataset=dataset,
        concurrency=concurrency,
        serving=serving,
    )


def parse_concurrency(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise BenchError(f"invalid concurrency list: {raw!r}") from exc
    if not values or any(value < 1 for value in values):
        raise BenchError("concurrency values must be positive integers")
    return values


def wait_for_readiness(
    config: ServeConfig,
    *,
    environment: Mapping[str, str],
    opener: Any = urllib.request.urlopen,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    """Wait for the authenticated model-list path, not merely an open port."""
    base_url = environment.get("SMOLQWEN_BASE_URL", f"http://127.0.0.1:{config.proxy_port}").rstrip(
        "/"
    )
    deadline = monotonic() + config.readiness_timeout_s
    last_error = "endpoint did not respond"
    while monotonic() < deadline:
        request = urllib.request.Request(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {environment['VLLM_API_KEY']}"},
        )
        try:
            with opener(request, timeout=config.readiness_poll_interval_s) as response:
                if int(response.status) == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        sleep(config.readiness_poll_interval_s)
    raise BenchError(
        "authenticated model readiness did not succeed within "
        f"{config.readiness_timeout_s:g}s: {last_error}"
    )


def build_bench_command(
    config: ServeConfig,
    *,
    dataset: str,
    concurrency: int,
    result_path: Path,
    dataset_path: Path | None,
) -> list[str]:
    supported = {"sharegpt", "random", "prefix_repetition", "custom"}
    if dataset not in supported:
        raise BenchError(f"unsupported dataset {dataset!r}; expected one of {sorted(supported)}")
    if dataset in {"sharegpt", "custom"} and dataset_path is None:
        raise BenchError(f"dataset {dataset!r} requires --dataset-path")
    base_url = os.environ.get("SMOLQWEN_BASE_URL", f"http://127.0.0.1:{config.proxy_port}").rstrip(
        "/"
    )
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        config.served_model_name,
        "--tokenizer",
        config.model_path,
        "--dataset-name",
        dataset,
        "--num-prompts",
        str(config.benchmark_num_prompts),
        "--input-len",
        str(config.benchmark_input_len),
        "--output-len",
        str(config.benchmark_output_len),
        "--max-concurrency",
        str(concurrency),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        ",".join(str(value) for value in config.benchmark_percentiles),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    if dataset_path is not None:
        command.extend(["--dataset-path", str(dataset_path)])
    if dataset == "custom":
        command.append("--skip-chat-template")
    return command


def run_benchmarks(config: ServeConfig, *, args: Any) -> int:
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(args.dataset)
    command_dataset = dataset
    dataset_path = args.dataset_path
    if dataset == "bfcl-agentic":
        from smolqwen.config import resolve
        from smolqwen.config_models import EvalConfig
        from smolqwen.serving.workload import build_bfcl_agentic_workload
        from smolqwen.tokenizer import load_tokenizer

        eval_config = resolve("eval")
        if not isinstance(eval_config, EvalConfig):
            raise BenchError("resolved eval config has the wrong type")
        tokenizer = load_tokenizer(config.model_path, revision=config.model_revision)
        dataset_path, composition = build_bfcl_agentic_workload(
            eval_config,
            tokenizer=tokenizer,
            output_path=output_dir / "bfcl-agentic.jsonl",
        )
        print(f"wrote agent-shaped workload and composition: {dataset_path}, {composition}")
        command_dataset = "custom"
    if args.quality_report and not args.quality_reference:
        raise BenchError("--quality-report requires at least one --quality-reference")
    try:
        quality = (
            load_quality_result(args.quality_report, references=args.quality_reference)
            if args.quality_report
            else None
        )
    except (OSError, ValueError) as exc:
        raise BenchError(f"quality pairing failed: {exc}") from exc

    serving = config_metadata(config)
    config_id = benchmark_config_id(serving)
    rows: list[BenchResult] = []
    environment = serving_environment()
    wait_for_readiness(config, environment=environment)
    for concurrency in parse_concurrency(str(args.concurrency)):
        stem = f"{dataset}-c{concurrency}-{config_id}"
        raw_path = raw_dir / f"{stem}.json"
        command = build_bench_command(
            config,
            dataset=command_dataset,
            concurrency=concurrency,
            result_path=raw_path,
            dataset_path=dataset_path,
        )
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            raise BenchError(f"vllm bench serve failed at concurrency {concurrency}")
        result = parse_bench_file(
            raw_path,
            dataset=dataset,
            concurrency=concurrency,
            serving=serving,
        )
        if quality is not None:
            from smolqwen.serving.report import assert_quality_matches_serving

            try:
                assert_quality_matches_serving([result], quality)
            except ValueError as exc:
                raise BenchError(f"quality pairing failed: {exc}") from exc
            result = replace(
                result,
                bfcl_mt_score=quality.score,
                quality_recorded_free=dict(quality.manifest.recorded_free),
                quality_invariant_hash=quality.manifest.invariant_hash,
            )
        normalized_path = normalized_dir / f"{stem}.json"
        normalized_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(result)

    aggregate_rows = load_normalized_rows(normalized_dir)
    report_path = write_serving_report(output_dir, rows=aggregate_rows, quality=None)
    print(
        f"wrote {len(rows)} benchmark rows; report contains "
        f"{len(aggregate_rows)} rows: {report_path}"
    )
    return 0
