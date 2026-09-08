"""Prove vLLM loads this project's rank-32 all-linear Qwen3.5 adapters.

Both training configs adapt every linear layer, including the Gated DeltaNet
projections. vLLM 0.26 declares Qwen3.5 LoRA support and maps the separate HF GDN
weights into its fused projections; this test exercises the actual adapter artifact
boundary and rejects a silent base-model fallback.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from smolqwen.config_models import LoraConfig as ProjectLoraConfig
from smolqwen.inference.engine import EngineError, OfflineEngine
from smolqwen.inference.profiles import EvalProfile
from tests.helpers import write_tiny_vllm_checkpoint

# The rank the shipped configs train at, read from the model rather than repeated,
# so raising `lora.r` moves this fixture with it.
ADAPTER_RANK = ProjectLoraConfig().r


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM is not installed"),
    pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available"),
]

PROBE = "Call the lookup tool for id 1."

PROFILE = EvalProfile(
    max_model_len=512,
    # This fixture is a solo capability probe, not the colocated training profile.
    # Reserve enough cache for vLLM's startup profiling before testing LoRA.
    gpu_memory_utilization=0.80,
    concurrency=2,
    enforce_eager=True,
    max_lora_slots=1,
    max_new_tokens=16,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    seed=1234,
)


def trained_adapter(base_dir: Path, adapter_dir: Path) -> Path:
    """An `all-linear` adapter at the project's own rank, with non-zero `lora_B`.

    `get_peft_model` initializes `lora_B` to zeros so the adapted model starts
    equal to base. Filling it makes the adapter's delta observable, which is what
    an output comparison needs in order to distinguish "loaded" from "ignored".
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    # The vLLM fixture is deliberately a multimodal wrapper. Build the PEFT
    # adapter on its nested text config directly; AutoModelForCausalLM may select
    # the wrapper from `model_type=qwen3_5` and then read wrapper fields as text
    # fields on Transformers 5.16/Python 3.13.
    wrapper_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    text_config = Qwen3_5TextConfig.from_dict(wrapper_config["text_config"])
    factory: Any = Qwen3_5ForCausalLM
    base = factory(text_config)
    torch.manual_seed(1234)
    adapted = get_peft_model(
        base,
        LoraConfig(
            r=ADAPTER_RANK,
            lora_alpha=ADAPTER_RANK * 2,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        generator = torch.Generator().manual_seed(5678)
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                delta = torch.randn(
                    parameter.shape,
                    generator=generator,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter.copy_(delta * 0.5)
    adapted.save_pretrained(str(adapter_dir))
    return adapter_dir


@pytest.fixture(scope="module")
def probe_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("adapter-capability")
    # vLLM registers Qwen3.5's released multimodal wrapper (not the text-only
    # Qwen3_5TextConfig that the SFT model loader uses internally).
    base = write_tiny_vllm_checkpoint(root / "base")
    adapter = trained_adapter(base, root / "adapter")
    return base, adapter


def test_an_all_linear_adapter_loads_and_changes_generation(
    probe_checkpoint: tuple[Path, Path],
) -> None:
    base, adapter = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_lora=True, max_lora_rank=ADAPTER_RANK)
    try:
        engine.build()
        engine.load_adapter("probe", str(adapter))
        engine.validate_adapter("probe")
    finally:
        engine.shutdown()


def test_an_adapter_wider_than_the_reserved_rank_is_named_before_generation(
    probe_checkpoint: tuple[Path, Path],
) -> None:
    """vLLM's own rank check fires lazily and names only the reserved value.

    This is the shape of the bug that made every adapter path unusable: the engine
    reserved vLLM's default of 16 while the configs train at 32, and the refusal
    arrived at the first generation as a bare
    `LoRA rank 32 is greater than max_lora_rank 16`. Rejecting at registration says
    which side to change, and it is not a capability refusal -- a fallback to the
    slower policy would hide a fixable sizing mistake.
    """
    base, adapter = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_lora=True, max_lora_rank=1)
    try:
        engine.build()
        with pytest.raises(EngineError, match="above the 1 this engine reserved"):
            engine.load_adapter("too-wide", str(adapter))
    finally:
        engine.shutdown()


def test_generating_with_an_unloaded_adapter_name_raises(
    probe_checkpoint: tuple[Path, Path],
) -> None:
    base, _ = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_lora=True)
    engine.build()
    try:
        with pytest.raises(EngineError, match="never loaded"):
            engine.generate([PROBE], adapter="absent")
    finally:
        engine.shutdown()


def test_sleep_releases_memory_and_waking_restores_generation(
    probe_checkpoint: tuple[Path, Path],
) -> None:
    """The measurement the in-training eval's memory arithmetic depends on.

    `tracking.py:45` reads `max_memory_allocated()`, a monotonic high-water mark,
    and `memory_reserved()`, which does not shrink without `empty_cache()` --
    neither is called anywhere in `src/`. So a release can only be seen through
    `reset_peak_memory_stats()` plus the worker-side driver footprint, because
    `CuMemAllocator` leaves the PyTorch allocation counter high after unmapping;
    the parent-process reading would report nothing under vLLM V1.

    Record both numbers in the phase report. If the release is small, the
    in-training eval takes its subprocess fallback rather than discovering the
    problem against a live trainer.
    """
    base, _ = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_sleep_mode=True)
    engine.build()
    try:
        engine.generate([PROBE])
        before = engine.memory_allocated_bytes(reset_peak=True)

        engine.sleep()
        after = engine.memory_allocated_bytes(reset_peak=True)

        released_mib = (before - after) / (1024**2)
        print(f"sleep released {released_mib:.1f} MiB ({before} -> {after} bytes allocated)")
        assert after < before, (
            f"sleep() released nothing: {before} -> {after} bytes. The in-training "
            "eval must take its subprocess fallback."
        )

        engine.wake_up()
        assert engine.generate([PROBE])
    finally:
        engine.shutdown()
