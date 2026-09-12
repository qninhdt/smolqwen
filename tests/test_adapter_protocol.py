from __future__ import annotations

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters import adapter_factories
from smolqwen.eval.adapters.base import AdapterResult, EvalTask, StepResult


def test_adapter_value_objects_preserve_benchmark_identity_and_state() -> None:
    task = EvalTask("case-1", "multi_turn_base", "hello", ({"type": "function"},))
    assert task.category == "multi_turn_base"
    assert not StepResult("continue").complete
    assert AdapterResult(1.0, True).exact_success


def test_adapter_modules_are_discovered_without_a_runner_switch() -> None:
    assert {"bfcl_multi_turn"} <= set(adapter_factories())


def test_eval_config_accepts_options_for_an_unregistered_adapter() -> None:
    config = EvalConfig(
        adapter_options={"future_benchmark": {"dataset": "fixture", "revision": "abc"}},
    )
    assert config.adapter_options["future_benchmark"] == {
        "dataset": "fixture",
        "revision": "abc",
    }
