from __future__ import annotations

import pytest

from smolqwen.config import resolve
from smolqwen.config_models import EvalConfig
from smolqwen.eval.manifest import EvalManifest, ManifestMismatchError, assert_comparable
from smolqwen.eval.runner import build_manifest


def _manifest(*, prompt: str = "a", backend: str = "transformers") -> EvalManifest:
    return EvalManifest(
        invariant={"system_prompt_hash": prompt, "temperature": 0.0},
        recorded_free={"backend": backend, "quantization": "fp8", "checkpoint_revision": "abc"},
    )


def test_an_invariant_difference_is_refused_with_the_field_name() -> None:
    with pytest.raises(ManifestMismatchError, match="system_prompt_hash"):
        assert_comparable(_manifest(), _manifest(prompt="changed"))


def test_recorded_free_differences_are_comparable_and_retained() -> None:
    local, served = _manifest(), _manifest(backend="vllm")
    assert_comparable(local, served)
    assert local.to_dict()["recorded_free"]["backend"] == "transformers"
    assert served.to_dict()["recorded_free"]["backend"] == "vllm"


def test_pinned_checkpoint_revisions_can_differ_between_arms() -> None:
    base = _manifest()
    sft = EvalManifest(base.invariant, {**base.recorded_free, "checkpoint_revision": "def"})
    assert_comparable(base, sft)
    assert sft.to_dict()["recorded_free"]["checkpoint_revision"] == "def"


def test_manifest_retains_benchmark_invariants_without_interpreting_them() -> None:
    config = resolve("eval")
    assert isinstance(config, EvalConfig)
    first = build_manifest(
        config,
        revision="abc",
        backend="transformers",
        adapter_invariants={
            "fixture": {
                "system_prompt_hash": "prompt-a",
                "tool_schema_hash": "tools-a",
                "heldout_ids": ["case"],
            }
        },
    )
    second = build_manifest(
        config,
        revision="def",
        backend="transformers",
        adapter_invariants={
            "fixture": {
                "system_prompt_hash": "prompt-b",
                "tool_schema_hash": "tools-b",
                "heldout_ids": ["case"],
            }
        },
    )
    assert first.invariant["benchmark"] != second.invariant["benchmark"]
    assert first.invariant["benchmark"]["fixture"]["heldout_ids"] == ["case"]
    assert first.recorded_free["checkpoint_revision"] == "abc"


def test_render_mode_is_an_invariant_and_a_mode_mismatch_is_refused() -> None:
    """A thinking-prompt run and a non-thinking-prompt run ask the model different
    things, so the two are not comparable even with identical sampling."""
    config = resolve("eval")
    assert isinstance(config, EvalConfig)
    assert config.enable_thinking is False
    thinking = build_manifest(
        config.model_copy(update={"enable_thinking": True}),
        revision="abc",
        backend="transformers",
    )
    non_thinking = build_manifest(
        config.model_copy(update={"enable_thinking": False}),
        revision="abc",
        backend="transformers",
    )
    assert thinking.invariant["enable_thinking"] is True
    assert non_thinking.invariant["enable_thinking"] is False
    with pytest.raises(ManifestMismatchError, match="enable_thinking"):
        assert_comparable(thinking, non_thinking)
