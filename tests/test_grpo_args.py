"""GRPO trainer arguments: what TRL requires that the profile does not state.

`_grpo_args` is the only place the project's sizing profile meets TRL's own
divisibility rules. Those rules fire at config construction, before a weight
loads, so a violated one is a run that cannot start on any card -- which makes it
worth asserting on CPU rather than discovering on a rented GPU.

The in-training eval config is here for the same reason: the bound it has to
respect is the colocated engine's, which `_grpo_args` sets and nothing else knows.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import PROFILES, EvalConfig, GrpoConfig
from smolqwen.eval.batched import engine_config
from smolqwen.training.grpo import (
    GrpoError,
    _force_trl_prefix_caching,
    _grpo_args,
    _lora_config,
    bench_eval_config,
)
from smolqwen.training.optim import Toggle


def _args(config: GrpoConfig) -> Any:
    return _grpo_args(
        config,
        attn=Toggle("sdpa", True, "test"),
        precision=Toggle("float16", True, "test"),
        use_liger=False,
        report_to=[],
        use_vllm=False,
    )


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_assembles_trl_arguments(profile: str) -> None:
    """Each shipped profile must satisfy TRL's constructor, not just our schema."""
    config = resolve("grpo", profile=profile)
    assert isinstance(config, GrpoConfig)

    args = _args(config)

    # TRL evaluates a whole prompt group at once, so the global eval batch must be
    # a multiple of the group size. `micro_batch` is 1 or 2 in every profile and is
    # never a multiple of `num_generations`, so reusing it here raised at
    # construction on all three profiles.
    assert args.per_device_eval_batch_size % config.profile.num_generations == 0
    # Train batch keeps following the profile's own sizing field.
    assert args.per_device_train_batch_size == config.profile.micro_batch


def test_grpo_uses_the_shared_benchmark_callback_instead_of_native_trl_eval() -> None:
    config = resolve("grpo", profile="t4")
    assert isinstance(config, GrpoConfig)

    args = _args(config)

    assert args.eval_strategy == "no"


def test_qwen35_grpo_all_linear_targets_are_text_only() -> None:
    config = resolve("grpo", profile="l4")
    assert isinstance(config, GrpoConfig)

    lora = _lora_config(config)

    assert lora.target_modules == "all-linear"
    assert lora.exclude_modules == r".*\.visual(?:\..*)?$"


def test_a_generation_batch_that_cannot_hold_whole_groups_is_refused() -> None:
    # 3 prompts cannot be split into groups of 2, so the pool would carry a partial
    # group. `active_pool_multiplier` is 1 here so `generation_batch_size` is the
    # concurrency itself; at the profile's default of 2 the product would be even
    # and the check would not fire.
    config = resolve(
        "grpo",
        profile="t4",
        overrides=["profile.generation_concurrency=3", "profile.active_pool_multiplier=1"],
    )
    assert isinstance(config, GrpoConfig)
    assert config.profile.generation_batch_size % config.profile.num_generations != 0

    with pytest.raises(GrpoError, match="divisible by num_generations"):
        _args(config)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_the_bench_eval_engine_cannot_be_asked_for_more_context_than_it_has(
    profile: str,
) -> None:
    """The bound is `vllm_max_model_len`, not `profile.max_seq_length`.

    `resolve("eval")` takes no `--profile`, so the callback used to size its turn
    engine from `ProfileConfig` defaults: 32768 context and width 8 against a
    colocated engine built at 16384, on every profile. The turn engine would admit a
    prefix the engine cannot accept and vLLM raises `The decoder prompt (length N) is
    longer than the maximum model length`, turning every boundary into
    `bench_failed`.
    """
    config = resolve("grpo", profile=profile)
    assert isinstance(config, GrpoConfig)

    resolved = bench_eval_config(config)
    assert isinstance(resolved, EvalConfig)
    engine = engine_config(resolved, max_in_flight=1)

    assert engine.max_model_len == config.vllm_max_model_len
    # And the rest of the sizing is this run's, not the defaults.
    assert engine.generation_concurrency == config.profile.generation_concurrency
    assert engine.max_env_steps == config.profile.max_env_steps
    # The adapter selection and decoding still come from the eval stage, which is
    # what makes one number mean one thing in training and in `evaluate`.
    assert resolved.adapters, "the eval stage config should still name its adapters"
    assert resolved.decoding == resolve("eval").decoding


def test_the_bench_eval_config_leaves_the_run_s_own_profile_untouched() -> None:
    """`ProfileConfig` is frozen; the copy must not be a mutation in disguise."""
    config = resolve("grpo", profile="l4")
    assert isinstance(config, GrpoConfig)

    bench_eval_config(config)

    assert config.profile.max_seq_length != config.vllm_max_model_len


def test_the_bench_eval_inherits_the_run_s_render_mode() -> None:
    """The boundary scores the distribution the rollout trains on: a dev set
    rendered in the other mode would bend the checkpoint-selection curve."""
    config = resolve("grpo", profile="l4")
    assert isinstance(config, GrpoConfig)

    resolved = bench_eval_config(config)
    assert resolved.enable_thinking is False

    config = config.model_copy(update={"enable_thinking": True})
    resolved = bench_eval_config(config)
    assert resolved.enable_thinking is True


def test_trl_vllm_constructor_receives_required_prefix_cache_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = ModuleType("trl.generation.vllm_generation")
    calls: list[dict[str, Any]] = []

    def fake_llm(*_args: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    generation.LLM = fake_llm  # type: ignore[attr-defined]
    trl = ModuleType("trl")
    trl_generation = ModuleType("trl.generation")
    trl.generation = trl_generation  # type: ignore[attr-defined]
    trl_generation.vllm_generation = generation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.generation", trl_generation)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", generation)

    original_llm = generation.LLM
    with _force_trl_prefix_caching():
        generation.LLM(model="qwen", enable_prefix_caching=False)

    assert calls == [{"model": "qwen", "enable_prefix_caching": True, "language_model_only": True}]
    assert generation.LLM is original_llm
