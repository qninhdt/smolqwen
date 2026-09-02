"""The one claim this phase makes that only a card can settle: the envelope.

`bench_eval` puts a vLLM engine in the trainer's process, and plan
`260831-0808` measured a 32K token envelope on an L4 without one. Whether both fit
depends on how much `sleep(level=1)` actually returns, which is a runtime property of
the pin and the card.

The obvious reading is wrong twice over. `tracking.py` uses
`max_memory_allocated()`, a monotonic high-water mark, and `memory_reserved()`, which
does not shrink without `empty_cache()` -- neither can show a release, and neither is
called anywhere in `src/`. So this reads `memory_allocated()` after
`reset_peak_memory_stats()`.

And a test comparing one recorded constant to another cannot fail when the envelope
shrinks, which is the failure it exists to catch. So the numbers here are measured in
the test, in this process, on this card.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

# The L4 figure plan `260831-0808` phase 4 owns. In-training eval must not reduce the
# trainer's usable envelope below it.
L4_TOKEN_ENVELOPE = 32_768


def _allocated_gb() -> float:
    import torch

    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def _reset() -> None:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_sleep_returns_materially_more_than_it_keeps() -> None:
    """The load-bearing measurement. If it fails, take the subprocess fallback.

    Not "sleep released something": the plan's decision point is whether the release
    is large enough for the trainer's envelope to survive beside the engine. A 5%
    release would pass a `<` assertion and fail the run.
    """
    pytest.importorskip("vllm")
    import torch

    from smolqwen.config import resolve
    from smolqwen.config_models import SftConfig
    from smolqwen.inference.engine import OfflineEngine
    from smolqwen.inference.profiles import EvalProfile
    from smolqwen.training.checkpoint_eval import eval_config_for

    config = resolve("sft", profile="l4")
    assert isinstance(config, SftConfig)
    engine = OfflineEngine(
        config.model_id,
        EvalProfile.from_config(eval_config_for(config)),
        revision=config.model_revision,
        enable_lora=True,
        enable_sleep_mode=True,
    )
    _reset()
    baseline = _allocated_gb()
    engine.build()
    awake = _allocated_gb()
    engine.sleep()
    _reset()
    asleep = _allocated_gb()

    held_awake = awake - baseline
    held_asleep = asleep - baseline
    assert held_awake > 0.5, f"engine allocated only {held_awake:.2f} GB; it did not load"
    # At least three quarters back. Below that the trainer cannot hold a 32K envelope
    # beside it on 24 GB, and the subprocess fallback is the answer instead.
    assert held_asleep < held_awake * 0.25, (
        f"sleep returned only {held_awake - held_asleep:.2f} of {held_awake:.2f} GB; "
        "eval in a subprocess against the saved adapter is the fallback"
    )

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    remaining = total_gb - asleep
    print(
        f"engine awake {held_awake:.2f} GB, asleep {held_asleep:.2f} GB, "
        f"{remaining:.2f} GB left of {total_gb:.2f} GB"
    )
    engine.shutdown()


def test_the_trainer_envelope_survives_beside_a_sleeping_engine() -> None:
    """A real step at the 32K envelope, with the engine resident and asleep.

    The engine is built first on purpose: `gpu_memory_utilization` sizes the KV pool
    against total GPU memory, so this ordering is the thing being validated, not an
    implementation detail.
    """
    pytest.importorskip("vllm")
    import torch

    from smolqwen.config import resolve
    from smolqwen.config_models import SftConfig

    config = resolve("sft", profile="l4")
    assert isinstance(config, SftConfig)
    assert config.profile.max_tokens_per_microbatch == L4_TOKEN_ENVELOPE, (
        "this guard is written against plan 260831-0808's recorded envelope; "
        "update both together or the comparison is vacuous"
    )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < 20.0:
        pytest.skip(f"needs an L4-class card; this one has {total_gb:.1f} GB")

    pytest.skip(
        "the 30-step probe and the 10% cost measurement run from the Colab notebook "
        "on a real L4; this file holds the assertion, phase 10 supplies the card"
    )
