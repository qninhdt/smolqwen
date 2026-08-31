"""Emit vLLM sweep inputs and delegate execution/Pareto logic upstream."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from smolqwen.config_models import ServeConfig
from smolqwen.serving.server import serving_environment


class SweepError(RuntimeError):
    """Raised when the upstream vLLM sweep command fails."""


def serve_parameter_rows(config: ServeConfig) -> list[dict[str, Any]]:
    """Explicit candidate combinations; vLLM owns iteration and Pareto selection."""
    baseline = {
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "enable_prefix_caching": config.enable_prefix_caching,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": False,
    }
    return [
        {"_benchmark_name": "baseline", **baseline},
        {"_benchmark_name": "no-prefix-cache", **baseline, "enable_prefix_caching": False},
        {"_benchmark_name": "eager", **baseline, "enforce_eager": True},
        {
            "_benchmark_name": "lower-kv-budget",
            **baseline,
            "gpu_memory_utilization": max(0.1, round(config.gpu_memory_utilization - 0.1, 2)),
        },
        {
            "_benchmark_name": "wider-batch",
            **baseline,
            "max_num_seqs": config.max_num_seqs * 2,
            "max_num_batched_tokens": config.max_num_batched_tokens * 2,
        },
    ]


def bench_parameter_rows(config: ServeConfig) -> list[dict[str, Any]]:
    return [
        {
            "_benchmark_name": f"random-c{concurrency}",
            "max_concurrency": concurrency,
            "input_len": config.benchmark_input_len,
            "output_len": config.benchmark_output_len,
        }
        for concurrency in (1, 4, 16)
    ]


def write_sweep_parameters(config: ServeConfig, directory: Path | str) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    serve_path = target / "serve-params.json"
    bench_path = target / "bench-params.json"
    serve_path.write_text(
        json.dumps(serve_parameter_rows(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bench_path.write_text(
        json.dumps(bench_parameter_rows(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return serve_path, bench_path


def _base_serve_command(config: ServeConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--served-model-name",
        config.served_model_name,
        "--max-model-len",
        str(config.max_model_len),
        "--dtype",
        config.dtype,
        "--reasoning-parser",
        config.reasoning_parser,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        config.tool_call_parser,
    ]
    if config.model_revision:
        command.extend(["--revision", config.model_revision])
    if config.quantization:
        command.extend(["--quantization", config.quantization])
    if config.speculative_num_tokens is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": config.speculative_num_tokens,
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    return command


def _base_bench_command(config: ServeConfig) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        f"http://127.0.0.1:{config.port}",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        config.served_model_name,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(config.benchmark_num_prompts),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        ",".join(str(value) for value in config.benchmark_percentiles),
    ]


def build_sweep_command(
    config: ServeConfig,
    *,
    serve_params: Path,
    bench_params: Path,
    output_dir: Path,
    experiment_name: str,
    resume: bool,
) -> list[str]:
    command = [
        "vllm",
        "bench",
        "sweep",
        "serve",
        "--serve-cmd",
        shlex.join(_base_serve_command(config)),
        "--bench-cmd",
        shlex.join(_base_bench_command(config)),
        "--serve-params",
        str(serve_params),
        "--bench-params",
        str(bench_params),
        "--num-runs",
        str(config.sweep_num_runs),
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        experiment_name,
        "--strict-params",
    ]
    if resume:
        command.append("--resume")
    return command


def run_sweep(config: ServeConfig, *, args: Any) -> int:
    output_dir = Path(config.output_dir) / "sweep"
    experiment = str(args.experiment_name or args.profile or "default")
    params_dir = output_dir / "parameters" / experiment
    serve_params, bench_params = write_sweep_parameters(config, params_dir)
    command = build_sweep_command(
        config,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=output_dir,
        experiment_name=experiment,
        resume=bool(args.resume),
    )
    environment = serving_environment()
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise SweepError("vllm bench sweep serve failed")
    experiment_dir = output_dir / experiment
    plot = subprocess.run(
        [
            "vllm",
            "bench",
            "sweep",
            "plot_pareto",
            str(experiment_dir),
            "--label-by",
            "max_concurrency,max_num_seqs,max_num_batched_tokens",
        ],
        env=environment,
        check=False,
    )
    if plot.returncode != 0:
        raise SweepError("vllm bench sweep plot_pareto failed")
    print(f"wrote sweep results and Pareto plot under {experiment_dir}")
    return 0
