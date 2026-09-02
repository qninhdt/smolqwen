"""Does vLLM load this project's adapters? A recorded capability, not a gate.

Both training configs use `target_modules: all-linear` (`sft.yaml:18`,
`grpo.yaml:13`), which emits LoRA weights for every linear in the model --
including Qwen3.5's Gated DeltaNet mixer projections. vLLM validates adapter
weights against a per-architecture allowlist, so it may accept them or refuse.

Nothing blocks on the answer. `TransformersPolicy` (`policies.py:201`) is the only
path that evaluates an adapter without merging it, and it stays; the outcome
decides how *fast* adapter evaluation is, not whether it works:

| checkpoint kind | path |
|---|---|
| base or merged | vLLM offline, batched |
| adapter, vLLM accepts | vLLM offline + `LoRARequest` |
| adapter, vLLM refuses | `TransformersPolicy`, unchanged |

The failure this file rules out is the middle row going wrong quietly. A silently
ignored adapter serves base output, which would make an in-training benchmark
curve flat and a cross-check against `evaluate` agree while both sides measured
the wrong weights. So `load_adapter` must either work or raise.

**On comparing outputs.** A freshly initialized LoRA adapter has `lora_B = 0` by
construction, so its delta is exactly zero and its output equals base **by
design**. Comparing adapter output against base only means something with an
adapter whose `lora_B` is non-zero, which is why the probe below perturbs it
rather than trusting `get_peft_model`'s initial state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smolqwen.inference.engine import EngineError, OfflineEngine
from smolqwen.inference.profiles import EvalProfile
from tests.helpers import write_tiny_checkpoint

pytestmark = pytest.mark.gpu

PROBE = "Call the lookup tool for id 1."

PROFILE = EvalProfile(
    max_model_len=512,
    gpu_memory_utilization=0.25,
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
    """An `all-linear` adapter with non-zero `lora_B`, saved to disk.

    `get_peft_model` initializes `lora_B` to zeros so the adapted model starts
    equal to base. Filling it makes the adapter's delta observable, which is what
    an output comparison needs in order to distinguish "loaded" from "ignored".
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(base_dir))
    adapted = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.05)
    adapted.save_pretrained(str(adapter_dir))
    return adapter_dir


@pytest.fixture(scope="module")
def probe_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("adapter-capability")
    base = write_tiny_checkpoint(root / "base")
    adapter = trained_adapter(base, root / "adapter")
    return base, adapter


def test_an_all_linear_adapter_either_loads_or_raises(
    probe_checkpoint: tuple[Path, Path],
) -> None:
    """Record which branch applies. Both are acceptable; silence is not.

    Whoever runs this on a card writes the outcome into the phase report, because
    the later in-training eval reads it when choosing how to load a checkpoint.
    """
    base, adapter = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_lora=True)
    engine.build()
    try:
        try:
            engine.load_adapter("probe", str(adapter))
        except Exception as exc:  # noqa: BLE001 - the refusal is the recorded result
            pytest.skip(f"vLLM refused this project's all-linear adapter: {exc!r}")

        adapted = engine.generate([PROBE], adapter="probe")
        base_output = engine.generate([PROBE])
        assert adapted[0].text != base_output[0].text, (
            "adapter output equals base output on a trained adapter: the adapter was "
            "accepted but not applied, which would make a benchmark curve flat while "
            "every cross-check still agreed"
        )
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
    `reset_peak_memory_stats()` plus `memory_allocated()`, and the obvious reading
    would report success whether or not sleep worked.

    Record both numbers in the phase report. If the release is small, the
    in-training eval takes its subprocess fallback rather than discovering the
    problem against a live trainer.
    """
    import torch

    base, _ = probe_checkpoint
    engine = OfflineEngine(str(base), PROFILE, enable_sleep_mode=True)
    engine.build()
    try:
        engine.generate([PROBE])
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        engine.sleep()
        torch.cuda.reset_peak_memory_stats()
        after = torch.cuda.memory_allocated()

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
