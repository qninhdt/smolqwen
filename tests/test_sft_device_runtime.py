"""Per-card SFT routing: FlashAttention-2 on Ampere-plus, sdpa below it."""

from __future__ import annotations

import pytest

from smolqwen.config_models import SftConfig
from smolqwen.data.convert_sft import SFT_SCHEMA_VERSION, SFT_SEMANTICS
from smolqwen.training.collate import collator
from smolqwen.training.optim import cast_adapters, resolve_attn_implementation, resolve_precision
from smolqwen.training.sft import SftError, _sft_config, resolve_sft_runtime


def test_turing_falls_back_to_padded_fp16_sdpa() -> None:
    runtime = resolve_sft_runtime(
        SftConfig(),
        capability=(7, 5),
        has_cuda=True,
        has_flash_attn=True,
        require_cuda=True,
        require_kernels=True,
    )

    # The wheel is present and still not selected: FA2's kernels need sm80.
    assert runtime.attention.name == "sdpa"
    assert runtime.dtype_name == "float16"
    assert runtime.bf16 is False
    assert runtime.fp16 is True
    assert runtime.padding_free is False


@pytest.mark.parametrize("capability", [(8, 0), (8, 9)])
def test_ampere_default_remains_padding_free_bf16(capability: tuple[int, int]) -> None:
    runtime = resolve_sft_runtime(
        SftConfig(),
        capability=capability,
        has_cuda=True,
        has_flash_attn=True,
        require_cuda=True,
        require_kernels=True,
    )

    assert runtime.attention.name == "flash_attention_2"
    assert runtime.dtype_name == "bfloat16"
    assert runtime.bf16 is True
    assert runtime.fp16 is False
    assert runtime.padding_free is True


def test_ampere_without_the_wheel_is_a_setup_failure_not_a_downgrade() -> None:
    with pytest.raises(SftError, match="official flash_attn wheel"):
        resolve_sft_runtime(
            SftConfig(),
            capability=(8, 9),
            has_cuda=True,
            has_flash_attn=False,
            require_cuda=True,
            require_kernels=True,
        )


def test_cpu_assembly_keeps_its_existing_padding_free_contract() -> None:
    runtime = resolve_sft_runtime(SftConfig(), has_cuda=False)

    assert runtime.padding_free is True
    assert runtime.dtype_name == "bfloat16"


def test_turing_runtime_reaches_transformers_fp16_arguments() -> None:
    runtime = resolve_sft_runtime(
        SftConfig(),
        capability=(7, 5),
        has_cuda=True,
        has_flash_attn=True,
        require_cuda=True,
        require_kernels=True,
    )

    args = _sft_config(SftConfig(), runtime=runtime, use_liger=False, report_to=[])

    assert args.fp16 is True
    assert args.bf16 is False
    assert args.model_init_kwargs == {"dtype": "float16", "attn_implementation": "sdpa"}


def test_pre_turing_production_run_refuses_rather_than_degrading() -> None:
    with pytest.raises(SftError, match="compute capability 7.5"):
        resolve_sft_runtime(
            SftConfig(),
            capability=(7, 0),
            has_cuda=True,
            require_cuda=True,
            require_kernels=True,
        )


def test_bf16_downgrades_to_fp16_below_ampere_with_the_reason_recorded() -> None:
    turing = resolve_precision(True, capability=(7, 5))
    ampere = resolve_precision(True, capability=(8, 6))

    assert turing.name == "float16"
    assert turing.enabled is False
    assert "no bf16 tensor cores" in turing.detail
    assert ampere.name == "bfloat16"
    assert ampere.enabled is True


def test_explicit_sdpa_request_is_not_second_guessed() -> None:
    toggle = resolve_attn_implementation("sdpa", has_flash_attn=True, has_cuda=True)

    assert toggle.name == "sdpa"
    assert toggle.enabled is True


def test_fp16_training_keeps_trainable_adapters_in_fp32() -> None:
    import torch

    adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("lora_adapter", adapter)

    toggle = cast_adapters(model, "float16")

    assert not toggle.enabled
    assert "GradScaler" in toggle.detail
    assert adapter.dtype == torch.float32


@pytest.mark.gpu
def test_padded_sdpa_forward_ignores_padding_from_other_rows() -> None:
    import torch

    from tests.helpers import tiny_qwen35_model

    model = tiny_qwen35_model().to("cuda")
    model.config._attn_implementation = "sdpa"
    records = [
        {
            "schema_version": SFT_SCHEMA_VERSION,
            "semantics": SFT_SEMANTICS,
            "trajectory_uid": "short",
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "seq_length": 3,
            "supervised_tokens": 2,
        },
        {
            "schema_version": SFT_SCHEMA_VERSION,
            "semantics": SFT_SEMANTICS,
            "trajectory_uid": "long",
            "input_ids": [4, 5, 6, 7],
            "labels": [-100, 5, 6, 7],
            "seq_length": 4,
            "supervised_tokens": 3,
        },
    ]
    batch = collator(pad_token_id=0, max_length=32)(records)
    batch = {key: value.to("cuda") for key, value in batch.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        reference = outputs.logits[1, :4].clone()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["input_ids"][0] = torch.tensor([20, 21, 22, 23], device="cuda")
        changed_outputs = model(**changed)

    assert outputs.logits.shape == (2, 4, 256)
    torch.testing.assert_close(changed_outputs.logits[1, :4], reference)
