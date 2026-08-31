"""TRL-pooled factory adapters reconstruct environment state on every reset."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
from transformers.utils.chat_template_utils import get_json_schema

from smolqwen.env.pool import WorkerPool
from smolqwen.env.registry import load_env_specs
from smolqwen.rollout.factory_env import build_factory_env_class
from tests.rollout_fixtures import ENV_METADATA, FIXTURE_SCRIPT, FIXTURE_TASK, load_fixture_workload

pytestmark = pytest.mark.slow

VENDORED_METADATA = (
    "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/191_env_metadata.json"
)


def test_reusing_one_factory_instance_carries_no_state_between_batches() -> None:
    specs, scenarios = load_fixture_workload()
    scenario = scenarios[FIXTURE_TASK]
    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1, episodes_per_worker=2) as pool:
        adapter_type = build_factory_env_class(
            env_spec=specs[scenario.env_id], pool=pool, scenarios_by_task=scenarios
        )
        adapter = adapter_type()
        adapter.reset(task_id=FIXTURE_TASK)
        name, arguments = FIXTURE_SCRIPT[0]
        getattr(adapter, name)(**arguments)
        first_keys = set((adapter.final_state() or {}).keys())

        adapter.reset(task_id=FIXTURE_TASK)
        second_keys = set((adapter.final_state() or {}).keys())
        adapter._destroy()

    assert first_keys == second_keys


def test_bound_tool_signature_keeps_every_schema_argument() -> None:
    specs, scenarios = load_fixture_workload()
    scenario = scenarios[FIXTURE_TASK]
    adapter_type = build_factory_env_class(
        env_spec=specs[scenario.env_id],
        pool=cast(WorkerPool, object()),
        scenarios_by_task=scenarios,
    )
    signature = inspect.signature(cast(Any, adapter_type()).update_clinical_trial_status)
    assert list(signature.parameters) == ["trial_id", "new_status"]
    schema = get_json_schema(cast(Any, adapter_type()).update_clinical_trial_status)
    assert list(schema["function"]["parameters"]["properties"]) == [
        "trial_id",
        "new_status",
    ]


def test_real_schema_allows_required_properties_after_optional_properties() -> None:
    spec = load_env_specs(VENDORED_METADATA)["env_1_sft"]
    adapter_type = build_factory_env_class(
        env_spec=spec,
        pool=cast(WorkerPool, object()),
        scenarios_by_task={},
    )
    method = cast(Any, adapter_type()).create_task
    signature = inspect.signature(method)

    assert list(signature.parameters) == [
        "task_id",
        "description",
        "status",
        "due_date",
        "priority",
        "assigned_to",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    schema = get_json_schema(method)
    assert set(schema["function"]["parameters"]["required"]) == {
        "task_id",
        "description",
        "status",
        "priority",
    }
