"""Config resolution: merge order, type coercion, and closed-schema enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smolqwen.config import (
    deep_merge,
    load_budgets,
    parse_override,
    resolve,
    resolved_summary,
)
from smolqwen.config_models import (
    BUDGET_SEEDED_FIELDS,
    PROFILES,
    ConfigError,
    DataConfig,
    GrpoConfig,
    ProfileConfig,
    SftConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def test_deep_merge_nests_and_overlay_wins() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    overlay = {"nested": {"y": 99, "z": 3}}
    assert deep_merge(base, overlay) == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}}


def test_deep_merge_treats_list_as_leaf() -> None:
    # A half-overridden exclusion list is worse than either list, so a list
    # replaces rather than extends.
    merged = deep_merge({"patterns": ["a", "b"]}, {"patterns": ["c"]})
    assert merged["patterns"] == ["c"]


def test_resolve_reads_committed_base_configs() -> None:
    for stage in ("data", "sft", "grpo", "eval", "serve"):
        config = resolve(stage, config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))
        assert config is not None
        # Every resolved config round-trips to JSON, which is what --dry-run and
        # the W&B run config both print.
        json.loads(resolved_summary(config))


def test_profile_overlays_only_sizing_fields(tmp_path: Path) -> None:
    l4 = resolve("sft", profile="l4", config_dir=CONFIG_DIR, budgets_path=tmp_path / "none.json")
    a100 = resolve(
        "sft", profile="a100", config_dir=CONFIG_DIR, budgets_path=tmp_path / "none.json"
    )
    assert isinstance(l4, SftConfig) and isinstance(a100, SftConfig)
    # Sizing differs...
    assert l4.profile.micro_batch != a100.profile.micro_batch
    # ...while every semantic field is identical, which is what makes the
    # L4-vs-A100 comparison the same experiment.
    assert l4.model_dump(exclude={"profile"}) == a100.model_dump(exclude={"profile"})


def test_profile_cannot_carry_a_semantic_field(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "l4.yaml").write_text("micro_batch: 2\nlearning_rate: 3.0e-4\n", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "sft.yaml").write_text("model_id: Qwen/Qwen3.5-2B\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="learning_rate"):
        resolve("sft", profile="l4", config_dir=tmp_path, budgets_path=tmp_path / "none.json")


def test_override_coerces_via_target_annotation() -> None:
    config = resolve(
        "sft",
        config_dir=CONFIG_DIR,
        overrides=["training.learning_rate=1e-4"],
        budgets_path=Path("/nonexistent"),
    )
    assert isinstance(config, SftConfig)
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert isinstance(config.training.learning_rate, float)


def test_override_coerces_int_and_bool() -> None:
    config = resolve(
        "sft",
        config_dir=CONFIG_DIR,
        overrides=["profile.micro_batch=4", "optimization.gradient_checkpointing=false"],
        budgets_path=Path("/nonexistent"),
    )
    assert isinstance(config, SftConfig)
    assert config.profile.micro_batch == 4
    # bool("false") is True, so without annotation-driven coercion this override
    # would silently enable the flag it was asked to disable.
    assert config.optimization.gradient_checkpointing is False


def test_override_rejects_unparseable_bool() -> None:
    with pytest.raises(ConfigError, match="boolean"):
        parse_override(SftConfig, "optimization.bf16=maybe")


def test_unknown_override_key_fails_with_available_keys() -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_override(SftConfig, "training.lerning_rate=1e-4")
    message = str(excinfo.value)
    assert "lerning_rate" in message
    assert "learning_rate" in message  # the available-keys list names the real one


def test_override_naming_a_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="names a section"):
        parse_override(SftConfig, "training=5")


def test_override_requires_equals_sign() -> None:
    with pytest.raises(ConfigError, match="SECTION.KEY=VALUE"):
        parse_override(SftConfig, "training.learning_rate")


def test_unknown_stage_and_profile_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown stage"):
        resolve("nope", config_dir=CONFIG_DIR)
    with pytest.raises(ConfigError, match="unknown profile"):
        resolve("sft", profile="h100", config_dir=CONFIG_DIR)


def test_budgets_is_the_first_layer_of_the_chain(tmp_path: Path) -> None:
    budgets = tmp_path / "budgets.json"
    budgets.write_text(
        json.dumps({"recommended": {"max_seq_length": 4096, "max_env_steps": 12}}),
        encoding="utf-8",
    )
    overlay = load_budgets(budgets)
    assert overlay == {"profile": {"max_seq_length": 4096, "max_env_steps": 12}}

    # A profile may lower a seeded field...
    config = resolve(
        "sft",
        config_dir=CONFIG_DIR,
        overrides=["profile.max_seq_length=2048"],
        budgets_path=budgets,
    )
    assert isinstance(config, SftConfig)
    assert config.profile.max_seq_length == 2048


def test_profile_may_not_exceed_the_budget_cap(tmp_path: Path) -> None:
    budgets = tmp_path / "budgets.json"
    budgets.write_text(json.dumps({"recommended": {"max_seq_length": 4096}}), encoding="utf-8")
    # ...but not raise it above the measured cap: the budget bounds the training
    # distribution, and a sweep that raised it would train on a slice the
    # profiler never characterised.
    with pytest.raises(ConfigError, match="exceeds budgets.json cap"):
        resolve(
            "sft",
            config_dir=CONFIG_DIR,
            overrides=["profile.max_seq_length=16384"],
            budgets_path=budgets,
        )


def test_absent_budgets_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_budgets(tmp_path / "missing.json") == {}


def test_committed_profiles_do_not_override_budget_seeded_fields() -> None:
    """A placeholder in a profile YAML would silently override the measured cap.

    The profile layer merges *after* `budgets.json`, so a `max_seq_length: 8192`
    written as a placeholder wins over a measured 16384 and nothing reports it --
    the run trains under a cap the profiler never chose. A sweep writes one of
    these back only to LOWER it, and then it is a measurement, not a placeholder.
    """
    for profile in PROFILES:
        payload = yaml.safe_load(
            (CONFIG_DIR / "profiles" / f"{profile}.yaml").read_text(encoding="utf-8")
        )
        present = set(payload) & set(BUDGET_SEEDED_FIELDS.values())
        assert not present, (
            f"{profile}.yaml sets budget-seeded field(s) {sorted(present)}; "
            "these come from budgets.json, which merges earlier in the chain"
        )


def test_measured_budgets_reach_the_resolved_profile(tmp_path: Path) -> None:
    """End to end through the committed profiles, not just the overlay function."""
    budgets = tmp_path / "budgets.json"
    budgets.write_text(
        json.dumps({"recommended": {"max_seq_length": 16384, "max_new_tokens_per_step": 5694}}),
        encoding="utf-8",
    )
    config = resolve("sft", profile="l4", config_dir=CONFIG_DIR, budgets_path=budgets)
    assert isinstance(config, SftConfig)
    assert config.profile.max_seq_length == 16384
    assert config.profile.max_new_tokens_per_step == 5694


def test_malformed_budgets_file_raises(tmp_path: Path) -> None:
    budgets = tmp_path / "budgets.json"
    budgets.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_budgets(budgets)


def test_generation_concurrency_and_num_generations_are_separate_fields() -> None:
    # Two different quantities: vLLM batch width inside one rollout call vs G per
    # GRPO group. Phase 6 owns the first, Phase 7 the second.
    fields = set(ProfileConfig.model_fields)
    assert {"generation_concurrency", "num_generations"} <= fields


def test_data_config_pins_dataset_revisions() -> None:
    config = resolve("data", config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))
    assert isinstance(config, DataConfig)
    for pin in (config.sft_trajectories, config.rl_scenarios, config.env_metadata):
        assert pin.revision, f"{pin.filename} must be pinned by revision"
    # The two files whose contents get exec()ed additionally carry a sha256: a
    # count check passes trivially for a modified env_class_code body.
    assert config.env_metadata.sha256
    assert config.rl_scenarios.sha256


def test_grpo_runtime_pins_the_vendored_executable_metadata() -> None:
    config = resolve("grpo", config_dir=CONFIG_DIR, budgets_path=Path("/nonexistent"))
    assert isinstance(config, GrpoConfig)
    assert config.env.env_metadata_source == "vendored"
    assert config.env.vendored_env_metadata_sha256
    assert config.env.vendored_rl_scenarios_sha256


def test_committed_sft_config_does_not_reference_the_pretemplated_file() -> None:
    text = (CONFIG_DIR / "base" / "data.yaml").read_text(encoding="utf-8")
    # It appears only inside the comment explaining why it is excluded, never as
    # a filename value.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "mask_history" not in stripped
