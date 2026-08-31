"""Two real GRPO optimizer steps over the Phase 6 rollout boundary.

The step runs on whichever device `smoke_device` reports: the tiny model has GDN
layers, and with the mandatory `causal_conv1d` kernel installed those have no CPU
path, so a GPU host must place the trainer on the device.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import pytest

from smolqwen.env.pool import Result, WorkerPool
from smolqwen.rollout.factory_env import build_factory_env_class
from smolqwen.rollout.generation import ScriptedPolicyBackend
from smolqwen.rollout.rollout_func import encode_ids, make_rollout_func
from smolqwen.training.reward import verifier_reward
from tests.helpers import load_template, smoke_device, write_tiny_checkpoint
from tests.rollout_fixtures import (
    ENV_METADATA,
    FIXTURE_TASK,
    default_score_payload,
    fast_config,
    fixture_bindings,
    load_fixture_workload,
    ok_result,
)

pytestmark = pytest.mark.slow


class _ImmediateDispatcher:
    @staticmethod
    def _done(value: Result) -> Future[Result]:
        future: Future[Result] = Future()
        future.set_result(value)
        return future

    def submit_create(self, episode_id: str, binding: Any) -> Future[Result]:
        return self._done(ok_result(episode_id, {"tools": []}))

    def submit_step(
        self, episode_id: str, name: str, arguments: Mapping[str, Any]
    ) -> Future[Result]:
        raise AssertionError("the smoke policy terminates without a tool call")

    def submit_score(self, episode_id: str) -> Future[Result]:
        position = int(episode_id.rsplit("@", 1)[1])
        return self._done(ok_result(episode_id, default_score_payload(float(position % 2))))

    def submit_destroy(self, episode_id: str) -> Future[Result]:
        return self._done(ok_result(episode_id, True))


def test_two_grpo_steps_run_against_fixture_scenarios(tmp_path: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[attr-defined]

    checkpoint = write_tiny_checkpoint(tmp_path / "tiny")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    tokenizer.chat_template = load_template()
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.float32)
    binding = fixture_bindings(episodes=1)[0]
    prompt = [
        {"role": "system", "content": "fixture system"},
        {"role": "user", "content": binding.scenario.task},
    ]
    dataset = Dataset.from_list(
        [
            {"prompt": prompt, "task_id": binding.scenario.task_id},
            {"prompt": prompt, "task_id": binding.scenario.task_id},
        ]
    )
    rollout = make_rollout_func(
        resolve_bindings=lambda prompts: fixture_bindings(episodes=len(prompts), num_generations=2),
        config=fast_config(generation_concurrency=2, max_new_tokens_per_step=32),
        dispatcher=_ImmediateDispatcher(),
        tokenizer=tokenizer,
        backend_factory=lambda _: ScriptedPolicyBackend(
            lambda episode_id, turn, messages: "Task Completed",
            lambda text: encode_ids(tokenizer, text),
        ),
    )
    args = GRPOConfig(
        output_dir=str(tmp_path / "out"),
        use_cpu=smoke_device() == "cpu",
        bf16=False,
        gradient_checkpointing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        num_generations=2,
        max_steps=2,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_vllm=False,
        max_completion_length=64,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=cast(Any, verifier_reward),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        tools=None,
        rollout_func=cast(Any, rollout),
        environment_factory=None,
    )
    output = trainer.train()
    assert output.global_step == 2
    assert torch.isfinite(torch.tensor(output.training_loss))


def test_factory_oracle_trainer_runs_complete_episodes_and_verifier_rewards(
    tmp_path: Path,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[attr-defined]

    checkpoint = write_tiny_checkpoint(tmp_path / "tiny-factory")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    tokenizer.chat_template = load_template()
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.float32)
    specs, scenarios = load_fixture_workload()
    scenario = scenarios[FIXTURE_TASK]
    env_id = scenario.env_id
    prompt = [
        {"role": "system", "content": "fixture system"},
        {"role": "user", "content": scenario.task},
    ]
    dataset = Dataset.from_list(
        [
            {"prompt": prompt, "task_id": FIXTURE_TASK, "environment": env_id},
            {"prompt": prompt, "task_id": FIXTURE_TASK, "environment": env_id},
        ]
    )
    args = GRPOConfig(
        output_dir=str(tmp_path / "factory-out"),
        use_cpu=smoke_device() == "cpu",
        bf16=False,
        gradient_checkpointing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        num_generations=2,
        max_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_vllm=False,
        max_completion_length=32,
        max_tool_calling_iterations=4,
        remove_unused_columns=False,
        disable_tqdm=True,
    )

    with WorkerPool(metadata_path=str(ENV_METADATA), worker_count=1, episodes_per_worker=4) as pool:
        base = build_factory_env_class(
            env_spec=specs[env_id],
            pool=pool,
            scenarios_by_task=scenarios,
        )

        class TrackedFactoryEnv(base):  # type: ignore[misc, valid-type]
            reward_calls = 0

            def get_reward(self) -> float:
                type(self).reward_calls += 1
                return float(super().get_reward())

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=None,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
            tools=None,
            rollout_func=None,
            environment_factory={env_id: TrackedFactoryEnv},
        )
        output = trainer.train()

    assert output.global_step == 1
    assert TrackedFactoryEnv.reward_calls == 2
    assert torch.isfinite(torch.tensor(output.training_loss))
