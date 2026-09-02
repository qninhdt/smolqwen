"""The engine wrapper's contract, without vLLM installed.

CI has no vllm by construction (`pyproject.toml:33-36`), so these tests drive
`OfflineEngine` against a fake `vllm.LLM` injected into `sys.modules`. That proves
the wrapper -- lifecycle, telemetry, alignment checking -- and proves nothing about
`LLM.generate` itself, which is settled on a card in the GPU-validation phase.

The alignment check is the reason the fake exists at all. vLLM returns outputs in
request order, and if that ever changed, one task's completion would be attributed
to another task's score with every downstream metric still looking plausible.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from smolqwen.inference.engine import (
    TELEMETRY_ENV,
    Completion,
    EngineError,
    OfflineEngine,
    disable_telemetry,
)
from smolqwen.inference.profiles import EvalProfile

PROFILE = EvalProfile(
    max_model_len=4096,
    gpu_memory_utilization=0.25,
    concurrency=8,
    enforce_eager=False,
    max_lora_slots=1,
    max_new_tokens=128,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    seed=1234,
)


@dataclass
class FakeOutput:
    prompt: str
    text: str
    token_ids: tuple[int, ...] = (1, 2, 3)
    finish_reason: str | None = "stop"
    # One `{token_id: Logprob}` map per position, as vLLM returns when the sampling
    # params ask for logprobs. The second position deliberately omits its own token,
    # so the NaN-rather-than-drop path is exercised.
    logprobs: tuple[Any, ...] | None = None

    @property
    def outputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                text=self.text,
                token_ids=self.token_ids,
                finish_reason=self.finish_reason,
                logprobs=self.logprobs,
            )
        ]


@dataclass
class FakeLLM:
    """Records construction kwargs and returns one output per prompt, in order."""

    kwargs: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    slept: list[int] = field(default_factory=list)
    woke: int = 0
    reorder: bool = False

    def generate(self, **request: Any) -> list[FakeOutput]:
        self.calls.append(request)
        prompts = list(request["prompts"])
        outputs = [FakeOutput(prompt=prompt, text=f"out:{prompt}") for prompt in prompts]
        if self.reorder and len(outputs) > 1:
            outputs[0], outputs[1] = outputs[1], outputs[0]
        return outputs

    def sleep(self, level: int = 1) -> None:
        self.slept.append(level)

    def wake_up(self) -> None:
        self.woke += 1


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> list[FakeLLM]:
    """Install a fake `vllm` package, so `import vllm` inside `build()` resolves."""
    built: list[FakeLLM] = []

    def llm_factory(**kwargs: Any) -> FakeLLM:
        instance = FakeLLM(kwargs=kwargs)
        built.append(instance)
        return instance

    @dataclass(frozen=True)
    class FakeSamplingParams:
        temperature: float
        top_p: float
        top_k: int
        max_tokens: int
        seed: int | None
        logprobs: int | None = None

    vllm = ModuleType("vllm")
    vllm.LLM = llm_factory  # type: ignore[attr-defined]
    vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]

    lora = ModuleType("vllm.lora")
    request_module = ModuleType("vllm.lora.request")

    @dataclass(frozen=True)
    class FakeLoRARequest:
        lora_name: str
        lora_int_id: int
        lora_path: str

    request_module.LoRARequest = FakeLoRARequest  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)
    return built


def test_module_import_pulls_in_neither_torch_nor_vllm() -> None:
    """`--dry-run` resolves a config where neither is installed (`cli.py:1-8`)."""
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, smolqwen.inference.engine as e; "
            "print(sorted(m for m in ('torch', 'vllm') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "[]"


def test_telemetry_is_disabled_on_every_variable_vllm_honours() -> None:
    """Usage-stats collection is default-on and needs an explicit opt-out."""
    environment: dict[str, str] = {}
    assert disable_telemetry(environment) == dict(TELEMETRY_ENV)
    assert environment == dict(TELEMETRY_ENV)


def test_telemetry_does_not_overwrite_an_operator_choice() -> None:
    environment = {"VLLM_NO_USAGE_STATS": "0"}
    disable_telemetry(environment)
    assert environment["VLLM_NO_USAGE_STATS"] == "0"
    assert environment["DO_NOT_TRACK"] == "1"


def test_build_sets_telemetry_before_constructing_and_passes_profile_sizing(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in TELEMETRY_ENV:
        monkeypatch.delenv(name, raising=False)

    engine = OfflineEngine("local/checkpoint", PROFILE, revision="a" * 40)
    engine.build()

    assert len(fake_vllm) == 1
    kwargs = fake_vllm[0].kwargs
    assert kwargs["model"] == "local/checkpoint"
    assert kwargs["revision"] == "a" * 40
    assert kwargs["max_model_len"] == PROFILE.max_model_len
    assert kwargs["gpu_memory_utilization"] == PROFILE.gpu_memory_utilization
    assert kwargs["enforce_eager"] is False
    assert kwargs["enable_lora"] is False
    assert kwargs["enable_sleep_mode"] is False
    for name in TELEMETRY_ENV:
        assert name in __import__("os").environ

    # Idempotent: a callback may call build() at every boundary.
    engine.build()
    assert len(fake_vllm) == 1


def test_generation_is_one_batched_call_for_every_prompt(fake_vllm: list[FakeLLM]) -> None:
    """The whole point: `runner.py:69` looped tasks serially at batch size 1."""
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    prompts = [f"prompt-{index}" for index in range(5)]

    completions = engine.generate(prompts)

    assert len(fake_vllm[0].calls) == 1
    assert fake_vllm[0].calls[0]["prompts"] == prompts
    assert [completion.text for completion in completions] == [f"out:{p}" for p in prompts]


def test_sampling_params_come_from_the_resolved_decoding_config(
    fake_vllm: list[FakeLLM],
) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    engine.generate(["one"], max_new_tokens=64)

    params = fake_vllm[0].calls[0]["sampling_params"]
    assert params.temperature == PROFILE.temperature
    assert params.top_p == PROFILE.top_p
    assert params.top_k == PROFILE.top_k
    assert params.seed == PROFILE.seed
    assert params.max_tokens == 64


def test_an_empty_prompt_list_does_not_reach_the_engine(fake_vllm: list[FakeLLM]) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    assert engine.generate([]) == []
    assert engine.generate_ids([]) == []
    assert fake_vllm[0].calls == []


def test_token_generation_passes_ids_through_and_asks_for_logprobs(
    fake_vllm: list[FakeLLM],
) -> None:
    """The turn engine renders and tokenizes itself, so ids go in and ids come out.

    Re-tokenizing a decoded string would let a BPE seam move the prompt/completion
    boundary the mask builder depends on, which is why there is a separate entry
    point rather than a decode-and-re-encode round trip.
    """
    engine = OfflineEngine("m", PROFILE)
    engine.build()

    completions = engine.generate_ids([[1, 2, 3], [4, 5]])

    call = fake_vllm[0].calls[0]
    assert call["prompts"] == [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5]}]
    # `logprobs=0` is the sampled token's own logprob with no alternatives.
    assert call["sampling_params"].logprobs == 0
    assert len(completions) == 2
    assert completions[0].token_ids == (1, 2, 3)


def test_a_position_whose_logprob_is_missing_stays_nan_rather_than_shortening_the_row(
    fake_vllm: list[FakeLLM],
) -> None:
    """A short row is right-padded downstream and shifts every later position.

    vLLM returns one `{token_id: Logprob}` map per position. A position whose own
    sampled token is absent from its map has no logprob to report, and NaN is the
    value that marks "not observed" without changing the row's length.
    """
    logprob_rows = (
        {1: SimpleNamespace(logprob=-0.5)},
        {99: SimpleNamespace(logprob=-9.9)},  # the sampled token 2 is absent
        {3: SimpleNamespace(logprob=-0.25)},
    )

    def generate(**request: Any) -> list[FakeOutput]:
        fake_vllm[0].calls.append(request)
        return [FakeOutput(prompt="", text="t", logprobs=logprob_rows)]

    engine = OfflineEngine("m", PROFILE)
    engine.build()
    fake_vllm[0].generate = generate  # type: ignore[method-assign]

    completion = engine.generate_ids([[7]])[0]

    assert len(completion.logprobs) == len(completion.token_ids) == 3
    assert completion.logprobs[0] == -0.5
    assert math.isnan(completion.logprobs[1])
    assert completion.logprobs[2] == -0.25


def test_a_short_token_batch_raises(fake_vllm: list[FakeLLM]) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    fake_vllm[0].generate = lambda **_: []  # type: ignore[method-assign]

    with pytest.raises(EngineError, match="0 outputs for 2 prompts"):
        engine.generate_ids([[1], [2]])


def test_a_reordered_batch_raises_rather_than_misattributing_a_score(
    fake_vllm: list[FakeLLM],
) -> None:
    """A silent reorder scores task A with task B's completion; every metric still
    looks plausible, which is why this is an error and not a warning."""
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    fake_vllm[0].reorder = True

    with pytest.raises(EngineError, match="positionally aligned"):
        engine.generate(["first", "second"])


def test_a_short_batch_raises(fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    monkeypatch.setattr(fake_vllm[0], "generate", lambda **_: [])

    with pytest.raises(EngineError, match="0 outputs for 2 prompts"):
        engine.generate(["first", "second"])


def test_generation_before_build_is_an_error() -> None:
    with pytest.raises(EngineError, match="not built"):
        OfflineEngine("m", PROFILE).generate(["one"])


def test_sleep_offloads_once_and_refuses_generation_while_asleep(
    fake_vllm: list[FakeLLM],
) -> None:
    engine = OfflineEngine("m", PROFILE, enable_sleep_mode=True)
    engine.build()

    engine.sleep()
    engine.sleep()
    assert engine.is_sleeping
    assert fake_vllm[0].slept == [1]

    with pytest.raises(EngineError, match="asleep"):
        engine.generate(["one"])


def test_waking_restores_generation_and_is_idempotent(fake_vllm: list[FakeLLM]) -> None:
    engine = OfflineEngine("m", PROFILE, enable_sleep_mode=True)
    engine.build()
    engine.sleep()

    engine.wake_up()
    engine.wake_up()
    assert fake_vllm[0].woke == 1
    assert not engine.is_sleeping
    assert engine.generate(["one"])


def test_sleep_without_the_engine_flag_is_an_error_not_a_no_op(
    fake_vllm: list[FakeLLM],
) -> None:
    """vLLM requires `enable_sleep_mode` at construction. A silent no-op here would
    leave a trainer sizing itself against memory the engine never released."""
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    with pytest.raises(EngineError, match="enable_sleep_mode"):
        engine.sleep()


def test_an_adapter_must_be_loaded_before_it_can_be_generated_with(
    fake_vllm: list[FakeLLM],
) -> None:
    engine = OfflineEngine("m", PROFILE, enable_lora=True)
    engine.build()
    assert fake_vllm[0].kwargs["enable_lora"] is True
    assert fake_vllm[0].kwargs["max_loras"] == PROFILE.max_lora_slots

    with pytest.raises(EngineError, match="never loaded"):
        engine.generate(["one"], adapter="sft")

    engine.load_adapter("sft", "artifacts/models/checkpoint-100")
    engine.generate(["one"], adapter="sft")
    assert fake_vllm[0].calls[-1]["lora_request"].lora_path.endswith("checkpoint-100")


def test_loading_an_adapter_without_the_engine_flag_is_an_error(
    fake_vllm: list[FakeLLM],
) -> None:
    """A silently ignored adapter serves base output, which makes an in-training
    benchmark curve flat and a cross-check agree while both sides are wrong."""
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    with pytest.raises(EngineError, match="enable_lora"):
        engine.load_adapter("sft", "somewhere")


def test_finish_reason_is_the_engine_s_own_not_a_token_count() -> None:
    """`policies.py:284` synthesized this by comparing width to `max_new_tokens`,
    so a completion ending exactly at the budget read as truncated."""
    assert Completion("t", 128, "length").truncated
    assert not Completion("t", 128, "stop").truncated
    assert not Completion("t", 128, None).truncated
