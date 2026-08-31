from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from smolqwen.config_models import EvalConfig, ServeConfig
from smolqwen.eval.adapters.base import EvalTask
from smolqwen.serving.bench import (
    BenchError,
    build_bench_command,
    parse_concurrency,
    wait_for_readiness,
)
from smolqwen.serving.server import ServingError, build_serve_command, serving_environment
from smolqwen.serving.sweep import build_sweep_command, write_sweep_parameters
from smolqwen.serving.workload import build_bfcl_agentic_workload


def test_serve_argv_has_parsers_and_mtp_but_no_secret() -> None:
    config = ServeConfig(speculative_num_tokens=1, quantization="fp8")
    command = build_serve_command(config)
    assert command[:2] == ["vllm", "serve"]
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {"method": "mtp", "num_speculative_tokens": 1}
    assert "--api-key" not in command


def test_serving_environment_requires_key_without_putting_it_in_argv() -> None:
    with pytest.raises(ServingError, match="VLLM_API_KEY"):
        serving_environment({})
    environment = serving_environment({"VLLM_API_KEY": "secret"})
    assert environment["OPENAI_API_KEY"] == "secret"


def test_bench_command_uses_documented_percentiles_and_result_path(tmp_path: Path) -> None:
    result = tmp_path / "raw.json"
    command = build_bench_command(
        ServeConfig(),
        dataset="random",
        concurrency=4,
        result_path=result,
        dataset_path=None,
    )
    assert command[command.index("--percentile-metrics") + 1] == "ttft,tpot,itl,e2el"
    assert command[command.index("--metric-percentiles") + 1] == "50,95,99"
    assert command[command.index("--result-filename") + 1] == "raw.json"
    assert command[command.index("--tokenizer") + 1] == ServeConfig().model_path
    assert "--header" not in command


def test_custom_benchmark_uses_the_pinned_vllm_skip_template_flag(tmp_path: Path) -> None:
    command = build_bench_command(
        ServeConfig(),
        dataset="custom",
        concurrency=1,
        result_path=tmp_path / "raw.json",
        dataset_path=tmp_path / "workload.jsonl",
    )
    assert "--skip-chat-template" in command
    assert "--custom-skip-chat-template" not in command


def test_dataset_path_and_concurrency_validation_fail_early(tmp_path: Path) -> None:
    with pytest.raises(BenchError, match="dataset-path"):
        build_bench_command(
            ServeConfig(),
            dataset="sharegpt",
            concurrency=1,
            result_path=tmp_path / "result.json",
            dataset_path=None,
        )
    with pytest.raises(BenchError, match="positive"):
        parse_concurrency("1,0")


def test_readiness_probe_uses_the_key_on_a_model_path() -> None:
    seen: list[object] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def opener(request: object, *, timeout: float) -> Response:
        seen.extend([request, timeout])
        return Response()

    wait_for_readiness(
        ServeConfig(readiness_timeout_s=1.0, readiness_poll_interval_s=0.1),
        environment={"VLLM_API_KEY": "secret", "SMOLQWEN_BASE_URL": "http://proxy:8080"},
        opener=opener,
        monotonic=lambda: 0.0,
    )
    request = seen[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "http://proxy:8080/v1/models"
    assert request.headers["Authorization"] == "Bearer secret"


def test_sweep_delegates_combinations_resume_and_pareto_inputs_upstream(
    tmp_path: Path,
) -> None:
    config = ServeConfig()
    serve_params, bench_params = write_sweep_parameters(config, tmp_path / "params")
    serve_rows = json.loads(serve_params.read_text(encoding="utf-8"))
    assert {
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enforce_eager",
        "gpu_memory_utilization",
    } <= set(serve_rows[0])
    command = build_sweep_command(
        config,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=tmp_path / "results",
        experiment_name="l4",
        resume=True,
    )
    assert command[:4] == ["vllm", "bench", "sweep", "serve"]
    assert "--resume" in command
    assert "--strict-params" in command


def test_bfcl_agentic_workload_records_shape_without_claiming_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = EvalTask(
        "case-1",
        "multi_turn_base",
        "book it",
        ({"type": "function", "function": {"name": "book"}},),
    )

    class FakeAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def load_tasks(self) -> list[EvalTask]:
            return [task]

        def build_prompt(self, _task: EvalTask, _history: list[object]) -> list[dict[str, str]]:
            return [{"role": "system", "content": "agent"}, {"role": "user", "content": "book it"}]

    class FakeTokenizer:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            assert messages
            assert kwargs["tools"] == list(task.tools)
            return "rendered agent request"

    monkeypatch.setattr("smolqwen.serving.workload.BfclMultiTurnAdapter", FakeAdapter)
    config = EvalConfig(
        adapter_options={
            "bfcl_multi_turn": {
                "categories": ["multi_turn_base"],
                "data_dir": "unused",
                "benchmark_commit": "a" * 40,
            }
        }
    )
    workload, composition = build_bfcl_agentic_workload(
        config,
        tokenizer=FakeTokenizer(),
        output_path=tmp_path / "agentic.jsonl",
    )

    assert json.loads(workload.read_text(encoding="utf-8")) == {"prompt": "rendered agent request"}
    metadata = json.loads(composition.read_text(encoding="utf-8"))
    assert metadata["quality_claim"] is False
    assert metadata["category_counts"] == {"multi_turn_base": 1}
    assert metadata["min_tools"] == metadata["max_tools"] == 1
