from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.config_models import EvalConfig, ServeConfig
from smolqwen.eval.adapters.base import EvalTask
from smolqwen.eval.workload import build_bfcl_agentic_workload
from smolqwen.serving.server import ServingError, build_serve_command, serving_environment


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


def test_serving_environment_opts_out_of_vllm_usage_stats() -> None:
    """vLLM's usage-stats collection is default-on; this is the only opt-out.

    CI already sets offline flags for HF, transformers and W&B (`ci.yml:14-18`) and
    set none for vLLM, so the served process was the one path still reporting.
    """
    environment = serving_environment({"VLLM_API_KEY": "secret"})
    assert environment["VLLM_NO_USAGE_STATS"] == "1"
    assert environment["VLLM_DO_NOT_TRACK"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"


def test_bfcl_agentic_workload_records_shape_without_claiming_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why this survives the wrapper deletion: `--dataset-name random` measures
    token throughput on synthetic prompts, which says nothing about an agentic
    request's prefill shape or its tool-schema overhead."""
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

    monkeypatch.setattr("smolqwen.eval.workload.BfclMultiTurnAdapter", FakeAdapter)
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
