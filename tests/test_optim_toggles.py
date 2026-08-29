"""Optimization toggles: availability is checked, and the mixer stays eager.

The compile exclusion list is the load-bearing one. GDN layers run
`flash-linear-attention` Triton kernels that upstream marks
`torch.compiler.disable`; compiling through them raises inductor errors. So these
tests pin that `Qwen3_5GatedDeltaNet` is never selected, that the pattern list
from config is honoured, and that a downgraded toggle records *why* it downgraded
rather than silently reporting success.
"""

from __future__ import annotations

from smolqwen.training.optim import (
    Toggle,
    format_ledger,
    ledger,
    resolve_attn_implementation,
    resolve_liger,
    select_compile_targets,
)

# What a real Qwen3.5 module walk looks like: layers 0/2 are GDN, 1/3 full attn.
NAMED_CLASSES: list[tuple[str, str]] = [
    ("", "Qwen3_5ForCausalLM"),
    ("model", "Qwen3_5TextModel"),
    ("model.embed_tokens", "Embedding"),
    ("model.layers.0.linear_attn", "Qwen3_5GatedDeltaNet"),
    ("model.layers.0.linear_attn.conv1d", "Conv1d"),
    ("model.layers.0.mlp", "Qwen3_5MLP"),
    ("model.layers.0.input_layernorm", "Qwen3_5RMSNorm"),
    ("model.layers.1.self_attn", "Qwen3_5Attention"),
    ("model.layers.1.mlp", "Qwen3_5MLP"),
    ("model.norm", "Qwen3_5RMSNorm"),
    ("lm_head", "Linear"),
]

EXCLUDE = ("linear_attn", "mixer", "conv1d")


def test_the_gdn_mixer_body_is_never_a_compile_target() -> None:
    targets = select_compile_targets(NAMED_CLASSES, EXCLUDE)
    assert "model.layers.0.linear_attn" not in targets
    assert not any("linear_attn" in name for name in targets)


def test_mlps_norms_and_full_attention_are_compiled() -> None:
    targets = select_compile_targets(NAMED_CLASSES, EXCLUDE)
    assert "model.layers.0.mlp" in targets
    assert "model.layers.1.self_attn" in targets
    assert "model.norm" in targets


def test_the_root_module_is_never_compiled_as_a_whole() -> None:
    """A single `torch.compile(model)` traces straight through the mixer and raises."""
    assert "" not in select_compile_targets(NAMED_CLASSES, EXCLUDE)


def test_the_forbidden_class_is_excluded_even_with_an_empty_pattern_list() -> None:
    # The pattern list is configurable, so the class-name guard is the backstop:
    # an empty list must not become permission to compile the mixer.
    targets = select_compile_targets([("model.layers.0.linear_attn", "Qwen3_5GatedDeltaNet")], ())
    assert targets == []


def test_a_wider_pattern_list_removes_more() -> None:
    narrow = select_compile_targets(NAMED_CLASSES, ("conv1d",))
    wide = select_compile_targets(NAMED_CLASSES, ("conv1d", "mlp"))
    assert set(wide) < set(narrow)


def test_liger_off_by_config_says_what_that_costs() -> None:
    toggle = resolve_liger(False)
    assert not toggle.enabled
    assert "248,320" in toggle.detail


def test_liger_requested_but_absent_names_the_right_lever() -> None:
    """Lower the sequence length, not the batch: the logits tensor scales with seq."""
    toggle = resolve_liger(True, available=False)
    assert not toggle.enabled
    assert "max_seq_length" in toggle.detail
    assert "micro_batch" in toggle.detail


def test_liger_available_reports_what_it_avoids() -> None:
    toggle = resolve_liger(True, available=True)
    assert toggle.enabled
    assert "248320" in toggle.detail


def test_flash_attention_downgrades_with_a_recorded_reason() -> None:
    no_cuda = resolve_attn_implementation("flash_attention_2", has_flash_attn=True, has_cuda=False)
    assert no_cuda.name == "sdpa"
    assert "no CUDA device" in no_cuda.detail

    no_wheel = resolve_attn_implementation("flash_attention_2", has_flash_attn=False, has_cuda=True)
    assert no_wheel.name == "sdpa"
    assert "flash_attn wheel" in no_wheel.detail


def test_flash_attention_selected_when_both_present() -> None:
    toggle = resolve_attn_implementation("flash_attention_2", has_flash_attn=True, has_cuda=True)
    assert toggle.name == "flash_attention_2"
    assert toggle.enabled


def test_an_explicit_non_flash_choice_is_taken_as_given() -> None:
    toggle = resolve_attn_implementation("sdpa", has_flash_attn=False, has_cuda=False)
    assert toggle.name == "sdpa"
    assert toggle.enabled


def test_the_ledger_carries_state_and_reason_for_every_toggle() -> None:
    toggles = [
        resolve_liger(True, available=False),
        resolve_attn_implementation("flash_attention_2", has_flash_attn=False, has_cuda=False),
        Toggle("adapter_dtype", True, "cast 2 trainable tensors to bfloat16"),
    ]
    entries = ledger(toggles)
    assert len(entries) == 3
    # Every entry states on/off, so a run that quietly lost an optimization is
    # visible in the run config rather than only in the throughput number.
    assert all(value.startswith(("on: ", "off: ")) for value in entries.values())

    table = format_ledger(toggles)
    assert table.startswith("| toggle | state | detail |")
    assert table.count("\n") == len(toggles) + 1
