#!/usr/bin/env python
"""Run one official-Qwen serving point and leave a downloadable bundle."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path("/content/smolqwen")
ARCHIVE = Path("/content/smolqwen-l4-src.tgz")
POINT_FILE = Path("/content/serving-point.json")
VENV = Path("/content/vllm029")
PYTHON = VENV / "bin/python"
VLLM = VENV / "bin/vllm"
OUTPUT_ROOT = ROOT / "artifacts/serving/sweeps/direct-vllm029-official-16x2"
LOG = Path("/content/serving-point.log")
BUNDLE = Path("/content/serving-point.tgz")
STATUS = Path("/content/serving-point-status.json")
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _write_status(payload: dict[str, object]) -> None:
    STATUS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write("+ " + shlex.join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        log.write(completed.stdout)
        log.write(completed.stderr)
    return completed


def _prepare_source() -> None:
    if (ROOT / "pyproject.toml").is_file():
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")


def _install_runtime() -> None:
    if not PYTHON.exists():
        completed = subprocess.run(["uv", "venv", str(VENV)], text=True, check=False)
        if completed.returncode:
            raise RuntimeError("could not create the vLLM 0.29 environment")
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import pandas, torch, vllm; print(vllm.__version__, torch.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip().startswith("0.29.0 2.13.0"):
        return
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(PYTHON),
        "vllm==0.29.0",
        "torch==2.13.0",
        "pandas",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "--extra-index-url",
        "https://pypi.org/simple",
        "--index-strategy",
        "unsafe-best-match",
    ]
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("could not install vLLM 0.29")


def _params(point: dict[str, object]) -> tuple[Path, Path, str]:
    precision = str(point["precision"])
    batched_tokens = int(str(point["max_num_batched_tokens"]))
    max_num_seqs = int(str(point["max_num_seqs"]))
    concurrency = int(str(point["max_concurrency"]))
    point_name = str(point["_benchmark_name"])
    experiment = point_name.replace("_", "-")
    server: dict[str, object] = {
        "_benchmark_name": point_name,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto" if precision == "bf16" else "fp8",
        "max_model_len": 32768,
        "max_num_batched_tokens": batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
    }
    if precision == "fp8":
        server["quantization"] = "fp8"
    bench = {
        "_benchmark_name": f"concurrency-{concurrency}",
        "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
        "model": "smolqwen",
        "tokenizer": MODEL,
        "dataset_name": "hf",
        "dataset_path": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "bfcl_categories": "simple,multiple,parallel,parallel_multiple",
        "num_prompts": 200,
        "percentile_metrics": "ttft,tpot,itl,e2el",
        "metric_percentiles": "50,95,99",
        "save_result": True,
        "save_detailed": True,
        "temperature": 0.0,
        "max_concurrency": concurrency,
    }
    serve_path = Path("/content/point-serve-params.json")
    bench_path = Path("/content/point-bench-params.json")
    serve_path.write_text(json.dumps([server], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bench_path.write_text(json.dumps([bench], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return serve_path, bench_path, experiment


def _sweep_command(
    serve_path: Path,
    bench_path: Path,
    experiment: str,
    *,
    dry_run: bool,
    resume: bool,
) -> list[str]:
    serve = shlex.join(
        ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "smolqwen"]
    )
    bench = shlex.join(
        [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai-chat",
            "--endpoint",
            "/v1/chat/completions",
            "--model",
            "smolqwen",
            "--tokenizer",
            MODEL,
            "--dataset-name",
            "hf",
            "--dataset-path",
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            "--bfcl-categories",
            "simple,multiple,parallel,parallel_multiple",
            "--num-prompts",
            "200",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
        ]
    )
    metrics = shlex.join(
        [
            "python",
            str(ROOT / "scripts/capture-vllm-metrics.py"),
            str(OUTPUT_ROOT / experiment / "metrics"),
        ]
    )
    command = [
        str(VLLM),
        "bench",
        "sweep",
        "serve",
        "--serve-cmd",
        serve,
        "--bench-cmd",
        bench,
        "--after-bench-cmd",
        metrics,
        "--serve-params",
        str(serve_path),
        "--bench-params",
        str(bench_path),
        "--num-runs",
        "1",
        "--experiment-name",
        experiment,
        "--server-ready-timeout",
        "900",
        "-o",
        str(OUTPUT_ROOT),
    ]
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    return command


def _bundle(point: dict[str, object], experiment: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime = Path("/content/vllm-runtime.txt")
    if PYTHON.exists():
        runtime.write_text(
            subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import torch, vllm; print(vllm.__version__); print(torch.__version__)",
                ],
                text=True,
            ),
            encoding="utf-8",
        )
    point_copy = Path("/content/point.json")
    point_copy.write_text(json.dumps(point, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_root = f"artifacts/serving/sweeps/direct-vllm029-official-16x2/{experiment}"
    with tarfile.open(BUNDLE, "w:gz") as archive:
        result_dir = OUTPUT_ROOT / experiment
        if result_dir.exists():
            archive.add(result_dir, arcname=artifact_root)
        archive.add(point_copy, arcname=f"{artifact_root}/point.json")
        if LOG.exists():
            archive.add(LOG, arcname=f"{artifact_root}/serving-point.log")
        if runtime.exists():
            archive.add(runtime, arcname=f"{artifact_root}/vllm-runtime.txt")
        archive.add(STATUS, arcname=f"{artifact_root}/point-status.json")


def main() -> int:
    point = json.loads(POINT_FILE.read_text(encoding="utf-8"))
    experiment = str(point["_benchmark_name"]).replace("_", "-")
    LOG.write_text("", encoding="utf-8")
    status: dict[str, object] = {
        "point": point,
        "status": "failed",
        "started_at_unix": time.time(),
    }
    returncode = 1
    try:
        _prepare_source()
        _install_runtime()
        serve_path, bench_path, experiment = _params(point)
        resume = (OUTPUT_ROOT / experiment).exists()
        environment = dict(os.environ)
        environment["PATH"] = f"{VENV / 'bin'}:{environment.get('PATH', '')}"
        environment["VLLM_NO_USAGE_STATS"] = "1"
        environment["DO_NOT_TRACK"] = "1"
        dry_run = _run(
            _sweep_command(serve_path, bench_path, experiment, dry_run=True, resume=resume),
            env=environment,
        )
        if dry_run.returncode:
            raise RuntimeError(f"dry-run failed with exit code {dry_run.returncode}")
        result = _run(
            _sweep_command(serve_path, bench_path, experiment, dry_run=False, resume=resume),
            env=environment,
        )
        if result.returncode:
            raise RuntimeError(f"benchmark failed with exit code {result.returncode}")
        status["status"] = "passed"
        returncode = 0
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["returncode"] = 1
    finally:
        status["finished_at_unix"] = time.time()
        status["returncode"] = returncode
        _write_status(status)
        _bundle(point, experiment)
    print(json.dumps(status, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
