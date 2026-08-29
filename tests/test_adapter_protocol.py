from __future__ import annotations

import inspect

from smolqwen.config_models import EvalConfig
from smolqwen.eval import runner
from smolqwen.eval.adapters import adapter_factories
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult


def test_adapter_value_objects_preserve_benchmark_identity_and_state() -> None:
    task = EvalTask("case-1", "multi_turn_base", "hello", ({"type": "function"},))
    assert task.category == "multi_turn_base"
    assert not StepResult("continue").complete
    assert AdapterResult(1.0, True).exact_success


def test_adapter_modules_are_discovered_without_a_runner_switch() -> None:
    assert {"bfcl_multi_turn", "envscaler_heldout"} <= set(adapter_factories())


def test_core_runner_and_config_do_not_name_specific_benchmarks() -> None:
    runner_source = inspect.getsource(runner).casefold()
    config_source = inspect.getsource(EvalConfig).casefold()
    for benchmark_detail in ("bfcl", "envscaler", "multi_turn", "manifest_tools"):
        assert benchmark_detail not in runner_source
        assert benchmark_detail not in config_source
    assert not {
        "bfcl_categories",
        "bfcl_data_dir",
        "bfcl_commit",
        "heldout_env_count",
        "heldout_scenarios_per_env",
        "env",
    } & set(EvalConfig.model_fields)


def test_eval_config_accepts_options_for_an_unregistered_adapter() -> None:
    config = EvalConfig(
        adapters=("future_benchmark",),
        adapter_options={"future_benchmark": {"dataset": "fixture", "revision": "abc"}},
    )
    assert config.adapter_options["future_benchmark"] == {
        "dataset": "fixture",
        "revision": "abc",
    }
