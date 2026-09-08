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

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from smolqwen.config_models import LoraConfig as ProjectLoraConfig
from smolqwen.inference.engine import (
    TELEMETRY_ENV,
    AdapterCapabilityError,
    Completion,
    EngineError,
    OfflineEngine,
    OfflineEngineBackend,
    adapter_rank,
    disable_telemetry,
    lora_rank_slot,
    offline_engine_for_eval,
)
from smolqwen.inference.profiles import EvalProfile
from smolqwen.rollout.generation import GenerationRequest

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
    prompt_token_ids: tuple[int, ...] | None = None
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
    prefix_caching: bool = True
    rpc_calls: list[dict[str, Any]] = field(default_factory=list)
    rpc_readings: list[int] | None = None

    @property
    def llm_engine(self) -> SimpleNamespace:
        """The resolved `VllmConfig`, shaped as vLLM 0.26 exposes it.

        `serving_config()` reads its provenance here rather than from `EvalProfile`,
        because vLLM resolves `max_num_batched_tokens` and the chunked-prefill
        default itself and the recorded value has to be the resolved one.
        """
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(dtype="torch.bfloat16", quantization=None),
                cache_config=SimpleNamespace(
                    gpu_memory_utilization=self.kwargs.get("gpu_memory_utilization"),
                    enable_prefix_caching=self.prefix_caching,
                ),
                scheduler_config=SimpleNamespace(
                    max_num_seqs=48,
                    max_num_batched_tokens=8192,
                    enable_chunked_prefill=True,
                ),
                speculative_config=None,
            )
        )

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

    def read_cuda_memory_allocated(self, *, reset_peak: bool = False) -> int:
        return 0

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        call = {"method": method, "timeout": timeout, "args": args, "kwargs": kwargs or {}}
        self.rpc_calls.append(call)
        if self.rpc_readings is not None:
            return list(self.rpc_readings)
        if isinstance(method, str):
            return [getattr(self, method)(*args, **(kwargs or {}))]
        return [method(self, *args, **(kwargs or {}))]


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
    assert kwargs["language_model_only"] is True
    assert kwargs["worker_extension_cls"] == "smolqwen.inference.engine.MemoryWorkerExtension"
    for name in TELEMETRY_ENV:
        assert name in __import__("os").environ

    # Idempotent: a callback may call build() at every boundary.
    engine.build()
    assert len(fake_vllm) == 1


def test_build_resolves_bfloat16_default_for_turing(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("smolqwen.inference.profiles._capability", lambda: (7, 5))

    engine = OfflineEngine("m", PROFILE)
    engine.build()

    assert engine.profile.dtype == "float16"
    assert fake_vllm[0].kwargs["dtype"] == "float16"


def test_build_requests_prefix_caching_and_fails_closed_on_a_change(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-turn eval re-sends the whole conversation every turn.

    vLLM 0.26 resolves `enable_prefix_caching` to false for hybrid models when the
    flag is not passed (the same default the colocated GRPO engine overrides), so
    the request must be explicit and a resolved false must stop the build instead
    of leaving every turn to re-prefill the full prefix.
    """
    engine = OfflineEngine("m", PROFILE)
    engine.build()

    assert fake_vllm[0].kwargs["enable_prefix_caching"] is True

    def refusing_factory(**kwargs: Any) -> FakeLLM:
        return FakeLLM(kwargs=kwargs, prefix_caching=False)

    monkeypatch.setattr(sys.modules["vllm"], "LLM", refusing_factory)
    with pytest.raises(EngineError, match="enable_prefix_caching"):
        OfflineEngine("m", PROFILE).build()


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


def test_token_generation_rejects_an_echoed_prompt_id_mismatch(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    output = FakeOutput(prompt="", text="out", token_ids=(7,), prompt_token_ids=(99,))
    monkeypatch.setattr(fake_vllm[0], "generate", lambda **_: [output])

    with pytest.raises(EngineError, match="different prompt token ids"):
        engine.generate_ids([[1, 2, 3]])


def test_offline_backend_preserves_vllm_finish_reason_and_local_budget(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    backend = OfflineEngineBackend(engine)

    monkeypatch.setattr(
        fake_vllm[0],
        "generate",
        lambda **_: [
            FakeOutput(prompt="", text="first", token_ids=(1, 2), finish_reason="stop"),
            FakeOutput(prompt="", text="second", token_ids=(3,), finish_reason="length"),
        ],
    )
    generated = backend.generate(
        [
            GenerationRequest("episode-0", (1,), max_new_tokens=1),
            GenerationRequest("episode-1", (1,), max_new_tokens=3),
        ]
    )

    assert generated[0].token_ids == (1,)
    assert generated[0].finish_reason == "length"  # local per-row budget
    assert generated[0].truncated
    assert generated[1].finish_reason == "length"  # vLLM's own reason


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


def test_memory_read_is_requested_from_the_vllm_worker(fake_vllm: list[FakeLLM]) -> None:
    """The parent allocator is not the vLLM V1 allocator."""
    engine = OfflineEngine("m", PROFILE)
    engine.build()

    assert engine.memory_allocated_bytes(reset_peak=True) >= 0
    assert len(fake_vllm[0].rpc_calls) == 1
    assert fake_vllm[0].rpc_calls[0]["method"] == "read_cuda_memory_allocated"
    assert fake_vllm[0].rpc_calls[0]["kwargs"] == {"reset_peak": True}


def test_memory_read_sums_all_tensor_parallel_workers(fake_vllm: list[FakeLLM]) -> None:
    engine = OfflineEngine("m", PROFILE)
    engine.build()
    fake_vllm[0].rpc_readings = [17, 29]

    assert engine.memory_allocated_bytes() == 46


def test_qwen35_text_adapter_gets_the_wrapper_namespace(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    save_file(
        {
            "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.ones(4, 2),
        },
        str(adapter / "adapter_model.safetensors"),
    )

    engine = OfflineEngine("m", PROFILE)
    engine.build()
    monkeypatch.setattr(engine, "_is_qwen35_wrapper", lambda: True)
    prepared = Path(engine._prepare_adapter_path(str(adapter)))

    keys = set(load_file(str(prepared / "adapter_model.safetensors")))
    assert prepared != adapter
    assert all(".model.language_model.layers." in key for key in keys)
    engine.shutdown()
    assert not prepared.exists()


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


def test_the_reserved_lora_rank_covers_this_project_s_configured_rank(
    fake_vllm: list[FakeLLM],
) -> None:
    """vLLM defaults `max_lora_rank` to 16 and the configs train at 32.

    Left unset, every adapter load raised `LoRA rank 32 is greater than
    max_lora_rank 16` at its first generation -- fatal for `evaluate --adapter-path`
    and `bench_failed` at every SFT eval boundary. The fixture-only `r: 4` adapter in
    `test_vllm_adapter_capability.py` fit under the default, which is why no test saw
    it.
    """
    engine = OfflineEngine("m", PROFILE, enable_lora=True, max_lora_rank=32)
    engine.build()

    assert fake_vllm[0].kwargs["max_lora_rank"] == 32
    assert fake_vllm[0].kwargs["max_lora_rank"] >= ProjectLoraConfig().r


def test_a_rank_vllm_does_not_offer_is_rounded_up_to_one_it_does(
    fake_vllm: list[FakeLLM],
) -> None:
    """`LoRAConfig.max_lora_rank` is a validated Literal, not a free integer.

    Rounding up keeps the adapter loadable; rounding down would reproduce the
    refusal. The engine is also built without `enable_lora` in the second half,
    where vLLM builds no `LoRAConfig` at all and the argument would be unread.
    """
    assert lora_rank_slot(1) == 1
    assert lora_rank_slot(17) == 32
    assert lora_rank_slot(32) == 32
    with pytest.raises(EngineError, match="exceeds the largest rank"):
        lora_rank_slot(1024)

    OfflineEngine("m", PROFILE, enable_lora=True, max_lora_rank=17).build()
    assert fake_vllm[0].kwargs["max_lora_rank"] == 32

    OfflineEngine("m", PROFILE).build()
    assert "max_lora_rank" not in fake_vllm[1].kwargs


def test_an_adapter_wider_than_the_reserved_rank_is_refused_at_registration(
    fake_vllm: list[FakeLLM], tmp_path: Path
) -> None:
    """Named at registration, and not as a capability refusal.

    vLLM's own check fires lazily and reports only the reserved value. Treating a
    sizing mistake as a capability refusal would silently move the run onto the
    slower policy instead of surfacing a one-line fix.
    """
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")

    engine = OfflineEngine("m", PROFILE, enable_lora=True, max_lora_rank=32)
    engine.build()

    with pytest.raises(EngineError, match="rank 64, above the 32"):
        engine.load_adapter("too-wide", str(adapter))


def test_the_eval_helper_reserves_the_widest_rank_it_is_about_to_register(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read from the artifact, because the artifact is what vLLM validates.

    An adapter trained at one rank and an engine sized from a since-edited config is
    exactly the mismatch that fails at the first generation.
    """
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: False)
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    (narrow / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    wide = tmp_path / "wide"
    wide.mkdir()
    (wide / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")

    offline_engine_for_eval("m", PROFILE, adapter={"a": str(narrow), "b": str(wide)})

    assert fake_vllm[0].kwargs["max_lora_rank"] == 64


def test_an_unreadable_adapter_config_falls_back_to_the_configured_rank(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An adapter without a loadable `adapter_config.json` is not loadable at all.

    So the unreadable case must not lower the ceiling below what the configs train
    at; vLLM remains the party that reports the real problem.
    """
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: False)
    empty = tmp_path / "empty"
    empty.mkdir()

    assert adapter_rank(empty) is None
    offline_engine_for_eval("m", PROFILE, adapter={"a": str(empty)})

    assert fake_vllm[0].kwargs["max_lora_rank"] == ProjectLoraConfig().r


def test_serving_config_is_read_from_the_engine_not_from_the_profile(
    fake_vllm: list[FakeLLM],
) -> None:
    """The provenance a paired speed/quality row compares.

    The nine CLI flags that used to assert these are gone, and nothing replaced them
    as a *source*, so every field stayed `None` and `--require-serving-match` could
    not match any real serving row. vLLM resolves the batching limits itself, which
    is why this reads `vllm_config` rather than `EvalProfile`.
    """
    engine = OfflineEngine("m", PROFILE)
    with pytest.raises(EngineError, match="not built"):
        engine.serving_config()

    engine.build()
    recorded = engine.serving_config()

    assert recorded == {
        # `torch.` stripped, so the engine's spelling matches the manifest's.
        "dtype": "bfloat16",
        "quantization": None,
        "speculative_decoding": None,
        "kv_budget": PROFILE.gpu_memory_utilization,
        "max_num_seqs": 48,
        "max_num_batched_tokens": 8192,
        "chunked_prefill": True,
        "prefix_caching": True,
    }


def test_a_nonzero_adapter_that_matches_base_is_rejected_as_a_silent_noop(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = OfflineEngine("m", PROFILE, enable_lora=True)
    engine.build()
    engine.load_adapter("sft", "adapter")
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: True)

    with pytest.raises(AdapterCapabilityError, match="silent no-op"):
        engine.validate_adapter("sft")

    assert len(fake_vllm[0].calls[0]["prompts"]) == 8
    assert fake_vllm[0].calls[0]["sampling_params"].max_tokens == 4


def test_adapter_probe_uses_greedy_sampling_even_when_profile_samples(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = EvalProfile(**{**PROFILE.__dict__, "temperature": 0.7, "top_p": 0.8, "top_k": 20})
    engine = OfflineEngine("m", profile, enable_lora=True)
    engine.build()
    engine.load_adapter("sft", "adapter")
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: False)

    engine.validate_adapter("sft")

    params = fake_vllm[0].calls[0]["sampling_params"]
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.top_k == -1


def test_a_zero_adapter_still_exercises_vllm_lazy_loading(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = OfflineEngine("m", PROFILE, enable_lora=True)
    engine.build()
    engine.load_adapter("sft", "adapter")
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: False)

    engine.validate_adapter("sft")

    assert len(fake_vllm[0].calls) == 1
    assert fake_vllm[0].calls[0]["lora_request"].lora_name == "sft"


def test_eval_helper_preflights_an_adapter_before_returning_the_engine(
    fake_vllm: list[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("smolqwen.inference.engine._adapter_has_nonzero_lora_b", lambda _: True)

    with pytest.raises(AdapterCapabilityError, match="silent no-op"):
        offline_engine_for_eval("m", PROFILE, adapter={"sft": "adapter"})

    # Registration alone would make this a single call; the second call is the
    # deterministic adapter probe that catches vLLM's lazy no-op.
    assert len(fake_vllm[0].calls) == 2


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
    """A completion ending exactly at its budget can still have stopped normally."""
    assert Completion("t", 128, "length").truncated
    assert not Completion("t", 128, "stop").truncated
    assert not Completion("t", 128, None).truncated
