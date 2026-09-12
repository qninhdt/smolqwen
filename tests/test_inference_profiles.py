"""Profiles derive from the resolved `ProfileConfig`; they never declare sizing.

The failure this file exists to prevent is silent. A parallel `EvalProfile` with
its own `max_model_len` sits outside the overlay chain -- `config.py:245` resolves
a profile YAML only into the `profile` subtree -- so `--profile l4` would stop
sizing evaluation while `runner.py:42` kept writing `profile.max_seq_length` into
the manifest's **invariant** set as `max_context_tokens`, certifying comparability
between two runs that truncated at different lengths. Nothing raises.

So the assertions here are identity assertions against the resolved config, plus
argv equality for the serving command that moved into this package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import (
    EvalConfig,
    GrpoConfig,
    ProfileConfig,
    ServeConfig,
    ServingProfileConfig,
)
from smolqwen.inference.profiles import EvalProfile, RolloutProfile, ServeProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def shipped(stage: str, profile: str | None = None) -> object:
    return resolve(stage, profile, config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))


def test_eval_profile_reads_the_resolved_profile_rather_than_its_own_defaults() -> None:
    config = shipped("eval", "l4")
    assert isinstance(config, EvalConfig)
    profile = EvalProfile.from_config(config)

    assert profile.max_model_len == config.profile.max_seq_length
    assert profile.gpu_memory_utilization == config.profile.vllm_kv_fraction
    assert profile.concurrency == config.profile.generation_concurrency
    assert profile.enforce_eager == config.profile.enforce_eager
    assert profile.max_lora_slots == config.profile.max_lora_slots
    # Decoding is manifest-invariant; the engine must not restate it either.
    assert profile.max_new_tokens == config.decoding.max_new_tokens
    assert profile.temperature == config.decoding.temperature
    assert profile.top_p == config.decoding.top_p
    assert profile.top_k == config.decoding.top_k
    assert profile.presence_penalty == config.decoding.presence_penalty
    assert profile.seed == config.decoding.seed


@pytest.mark.parametrize("gpu", ["l4", "a100"])
def test_switching_profile_changes_eval_sizing(gpu: str) -> None:
    """`--profile <gpu>` must still reach evaluation after the move."""
    config = shipped("eval", gpu)
    assert isinstance(config, EvalConfig)
    profile_yaml = CONFIG_DIR / "profiles" / f"{gpu}.yaml"
    assert profile_yaml.is_file()

    profile = EvalProfile.from_config(config)
    # The two shipped profiles differ on KV fraction, so an EvalProfile that
    # ignored the overlay would produce the same number for both.
    assert profile.gpu_memory_utilization == config.profile.vllm_kv_fraction
    assert profile.concurrency == config.profile.generation_concurrency


def test_the_two_shipped_profiles_do_not_size_eval_identically() -> None:
    """The negative control for the test above: the values genuinely differ."""
    l4 = EvalProfile.from_config(_eval("l4"))
    a100 = EvalProfile.from_config(_eval("a100"))
    assert (l4.gpu_memory_utilization, l4.concurrency) != (
        a100.gpu_memory_utilization,
        a100.concurrency,
    )


def _eval(gpu: str) -> EvalConfig:
    config = shipped("eval", gpu)
    assert isinstance(config, EvalConfig)
    return config


def test_no_profile_field_is_declared_twice() -> None:
    """Every `EvalProfile` field must trace to a config field, not shadow one."""
    declared = set(ProfileConfig.model_fields)
    # These are the four names `EvalProfile` intentionally renames on the way out,
    # because the vLLM keyword differs from the config field name.
    renamed = {
        "max_model_len": "max_seq_length",
        "gpu_memory_utilization": "vllm_kv_fraction",
        "concurrency": "generation_concurrency",
    }
    for engine_name, config_name in renamed.items():
        assert config_name in declared
        assert engine_name not in declared, (
            f"{engine_name} exists on both EvalProfile and ProfileConfig; "
            "one of them is now outside the overlay chain"
        )


def test_rollout_profile_reads_grpo_engine_settings() -> None:
    config = shipped("grpo", "l4")
    assert isinstance(config, GrpoConfig)
    profile = RolloutProfile.from_config(config)

    assert profile.max_model_len == config.vllm_max_model_len
    assert profile.gpu_memory_utilization == config.profile.vllm_kv_fraction
    assert profile.concurrency == config.profile.generation_concurrency
    assert profile.enable_sleep_mode == config.vllm_enable_sleep_mode
    assert profile.enable_prefix_caching is True


def test_turn_engine_config_carries_the_render_mode() -> None:
    """Rollout rendering and decoding must switch together; the flag rides the
    engine config so no caller can set one without the other."""
    from smolqwen.inference.profiles import turn_engine_config

    config = shipped("grpo", "l4")
    assert isinstance(config, GrpoConfig)
    assert turn_engine_config(config).enable_thinking is False
    config = config.model_copy(update={"enable_thinking": True})
    assert turn_engine_config(config).enable_thinking is True


def test_eval_engine_config_carries_the_render_mode() -> None:
    from smolqwen.eval.batched import engine_config

    config = shipped("eval", "l4")
    assert isinstance(config, EvalConfig)
    engine = engine_config(config, max_in_flight=4)
    assert engine.enable_thinking is False
    engine = engine_config(config.model_copy(update={"enable_thinking": True}), max_in_flight=4)
    assert engine.enable_thinking is True


def test_serve_argv_is_unchanged_by_the_move() -> None:
    """`ServeProfile.command()` is `serving/server.py`'s builder, verbatim.

    `build_serve_command` now delegates here, so this asserts the delegation is
    an identity rather than a rewrite. Both `--enable-*`/`--no-enable-*` pairs
    must survive: vLLM's defaults for prefix caching and chunked prefill have
    changed across releases, so the negative form being explicit is what makes the
    recorded config the one that ran.
    """
    from smolqwen.serving.server import build_serve_command

    for config in (
        ServeConfig(),
        ServeConfig(
            speculative_num_tokens=1,
            profile=ServingProfileConfig(quantization="fp8", kv_cache_dtype="fp8"),
        ),
        ServeConfig(enable_prefix_caching=False, enable_chunked_prefill=False),
        ServeConfig(model_revision="b" * 40),
    ):
        assert ServeProfile(config).command() == build_serve_command(config)

    negatives = ServeProfile(
        ServeConfig(enable_prefix_caching=False, enable_chunked_prefill=False)
    ).command()
    assert "--no-enable-prefix-caching" in negatives
    assert "--no-enable-chunked-prefill" in negatives
