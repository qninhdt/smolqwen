"""Which generation path `evaluate` takes, and that it takes the vLLM one.

This is the test Phase 4 should have had. `evaluate_batched` was written, tested
against a scripted backend, and then never called by `run_evaluation` -- so
`smolqwen evaluate` kept driving HuggingFace `generate()` at batch size 1 while the
plan's second goal was recorded as met. Nothing failed, because nothing asserted
which path the *command* took.

So the first test here is a wiring assertion: with an engine available, the batched
path runs and the serial one does not. It fails on the pre-fix code.

The remaining cases cover the supported HTTP path and prove that a missing vLLM
runtime or a refused adapter fails local evaluation instead of changing engines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config_models import EvalConfig
from smolqwen.eval import batched as batched_module
from smolqwen.eval import runner
from smolqwen.eval.batched import ADAPTER_SLOT, Generation, generation_for
from smolqwen.eval.checkpoints import ResolvedCheckpoint
from smolqwen.eval.manifest import EvalManifest
from smolqwen.eval.serving_pairing import (
    QualityResult,
    ServingPairingError,
    assert_quality_matches_serving,
)
from smolqwen.inference.engine import AdapterCapabilityError
from smolqwen.inference.profiles import resolve_dtype

SHA = "a" * 40
METRICS = {
    "score": 0.5,
    "invalid_call_rate": 0.0,
    "average_steps": 1.0,
    "average_generated_tokens": 2.0,
    "truncation_rate": 0.0,
}


def _args(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "checkpoint": str(tmp_path),
        "revision": SHA,
        "endpoint": None,
        "adapter_path": None,
        "adapter_revision": None,
        "adapter": "fixture",
        "tag": "wired",
        "serving_backend": None,
        "require_serving_match": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _captured_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[EvalManifest]:
    captured: list[EvalManifest] = []

    def write_report(
        output_dir: Any, *, tag: str, manifest: EvalManifest, metrics: Any
    ) -> tuple[Path, Path]:
        captured.append(manifest)
        return tmp_path / f"{tag}.json", tmp_path / f"{tag}.md"

    monkeypatch.setattr(runner, "write_report", write_report)
    return captured


def test_evaluate_generates_through_the_engine_when_one_can_be_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression this file exists for: the command must use the batched path."""
    calls: list[str] = []

    def refuse_serial(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the serial path ran while an engine was available")

    monkeypatch.setattr(runner, "evaluate_adapter", refuse_serial)
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=None),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())

    def batched(*_a: Any, **_k: Any) -> dict[str, dict[str, float]]:
        calls.append("batched")
        return {"fixture": dict(METRICS)}

    monkeypatch.setattr(runner, "evaluate_batched", batched)
    captured = _captured_manifest(monkeypatch, tmp_path)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    assert runner.run_evaluation(config, _args(tmp_path)) == 0

    assert calls == ["batched"]
    recorded = captured[0].recorded_free
    assert recorded["generation_path"] == "vllm"
    assert recorded["backend"] == "vllm"


def test_no_http_policy_is_loaded_when_the_engine_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse_policy(**_k: Any) -> Any:
        raise AssertionError("an HTTP policy was constructed for local evaluation")

    monkeypatch.setattr(runner, "load_http_policy", refuse_policy)
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=None),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())
    monkeypatch.setattr(runner, "evaluate_batched", lambda *_a, **_k: {"fixture": dict(METRICS)})
    _captured_manifest(monkeypatch, tmp_path)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    assert runner.run_evaluation(config, _args(tmp_path)) == 0


def test_the_engine_is_shut_down_even_when_the_run_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An engine holding VRAM after a failed `evaluate` blocks the next command."""
    shutdowns: list[int] = []
    engine = SimpleNamespace(
        shutdown=lambda: shutdowns.append(1), profile=SimpleNamespace(dtype="float16")
    )
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=engine),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())

    def explode(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "evaluate_batched", explode)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.run_evaluation(config, _args(tmp_path))
    assert shutdowns == [1]


def test_an_endpoint_keeps_the_http_policy(tmp_path: Path) -> None:
    resolved = ResolvedCheckpoint(
        path=None, revision=SHA, source="endpoint", adapter_path=None, adapter_revision=None
    )
    generation = generation_for(EvalConfig(), resolved)
    assert generation.backend is None
    assert generation.path == "http"


def test_a_host_without_vllm_fails_local_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_vllm(*_a: Any, **_k: Any) -> Any:
        raise ModuleNotFoundError("No module named 'vllm'", name="vllm")

    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", no_vllm, raising=True)
    resolved = ResolvedCheckpoint(path=str(tmp_path), revision=SHA, source="local")
    with pytest.raises(ModuleNotFoundError, match="vllm"):
        generation_for(EvalConfig(), resolved)


def test_a_refused_adapter_fails_local_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    seen: list[dict[str, str] | None] = []

    def refuse(
        model: Any, profile: Any, *, revision: Any = None, adapter: Any = None, **_: Any
    ) -> Any:
        seen.append(adapter)
        raise AdapterCapabilityError(
            "LoRA target module model.layers.0.linear_attn (Qwen3_5GatedDeltaNet) matched "
            "the deployment configuration but could not be wrapped by any LoRA layer "
            "implementation. target_modules=['all-linear']"
        )

    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", refuse, raising=True)
    resolved = ResolvedCheckpoint(
        path=str(tmp_path),
        revision=SHA,
        adapter_path=str(adapter_dir),
        adapter_revision="b" * 40,
        source="local",
    )
    with pytest.raises(AdapterCapabilityError, match="could not be wrapped"):
        generation_for(EvalConfig(), resolved)
    # The adapter reached the engine under the one slot `evaluate` uses.
    assert seen == [{ADAPTER_SLOT: str(adapter_dir)}]


def test_a_rank_mismatch_is_not_treated_as_a_capability_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """vLLM's default `max_lora_rank` is 16 and the configs train at 32.

    A rank mismatch must surface directly rather than being hidden behind another
    inference implementation.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise ValueError("LoRA rank 32 is greater than max_lora_rank 16.")

    monkeypatch.setattr(
        batched_module,
        "EvalProfile",
        SimpleNamespace(from_config=lambda _c: SimpleNamespace(dtype="float16")),
    )
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", refuse, raising=True)
    resolved = ResolvedCheckpoint(
        path=str(tmp_path),
        revision=SHA,
        adapter_path=str(adapter_dir),
        adapter_revision="b" * 40,
        source="local",
    )
    with pytest.raises(ValueError, match="max_lora_rank"):
        generation_for(EvalConfig(), resolved)


def test_an_unrelated_import_error_is_not_treated_as_missing_vllm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken_runtime(*_a: Any, **_k: Any) -> Any:
        raise ImportError("cannot import name 'LLM' from a broken vllm dependency")

    monkeypatch.setattr(
        batched_module,
        "EvalProfile",
        SimpleNamespace(from_config=lambda _c: SimpleNamespace(dtype="float16")),
    )
    monkeypatch.setattr(
        "smolqwen.inference.engine.offline_engine_for_eval", broken_runtime, raising=True
    )
    resolved = ResolvedCheckpoint(path=str(tmp_path), revision=SHA, source="local")
    with pytest.raises(ImportError, match="broken vllm dependency"):
        generation_for(EvalConfig(), resolved)


def test_a_corrupt_adapter_is_not_treated_as_a_capability_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def corrupt(*_a: Any, **_k: Any) -> Any:
        raise ValueError("adapter_model.safetensors is corrupt")

    monkeypatch.setattr(
        batched_module,
        "EvalProfile",
        SimpleNamespace(from_config=lambda _c: SimpleNamespace(dtype="float16")),
    )
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", corrupt, raising=True)
    resolved = ResolvedCheckpoint(
        path=str(tmp_path),
        revision=SHA,
        adapter_path=str(adapter_dir),
        adapter_revision="b" * 40,
        source="local",
    )
    with pytest.raises(ValueError, match="corrupt"):
        generation_for(EvalConfig(), resolved)


def test_an_adapter_oom_is_not_treated_as_a_capability_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def oom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA out of memory while loading LoRA")

    monkeypatch.setattr(
        batched_module,
        "EvalProfile",
        SimpleNamespace(from_config=lambda _c: SimpleNamespace(dtype="float16")),
    )
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", oom, raising=True)
    resolved = ResolvedCheckpoint(
        path=str(tmp_path),
        revision=SHA,
        adapter_path=str(adapter_dir),
        adapter_revision="b" * 40,
        source="local",
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        generation_for(EvalConfig(), resolved)


def test_a_failure_with_no_adapter_involved_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OOM must not quietly become a slower run reporting a different number."""

    def oom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        batched_module,
        "EvalProfile",
        SimpleNamespace(from_config=lambda _c: SimpleNamespace(dtype="float16")),
    )
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", oom, raising=True)
    resolved = ResolvedCheckpoint(path=str(tmp_path), revision=SHA, source="local")
    with pytest.raises(RuntimeError, match="out of memory"):
        generation_for(EvalConfig(), resolved)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 5), "float16"),  # T4: Turing, no bf16 tensor cores
        ((8, 0), "bfloat16"),  # A100
        ((8, 6), "bfloat16"),  # RTX 30xx
        ((8, 9), "bfloat16"),  # L4
        ((9, 0), "bfloat16"),  # H100
    ],
)
def test_dtype_follows_the_card(capability: tuple[int, int], expected: str) -> None:
    assert resolve_dtype("bfloat16", capability=capability) == expected


def test_an_explicit_dtype_is_taken_as_given() -> None:
    """Same rule as `resolve_attn_implementation`: an explicit choice is not second-guessed."""
    assert resolve_dtype("float32", capability=(7, 5)) == "float32"


def test_the_downgrade_is_logged_rather_than_silent(caplog: pytest.LogCaptureFixture) -> None:
    """fp16 has a narrower exponent range; a reader comparing two rows needs to know."""
    with caplog.at_level(logging.WARNING, logger="smolqwen.inference.profiles"):
        resolve_dtype("bfloat16", capability=(7, 5))
    assert "no bf16 tensor cores" in caplog.text
    assert "float16" in caplog.text


def test_the_recorded_dtype_comes_from_the_engine_not_a_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A T4 row claiming bf16 would present two numeric regimes as one experiment."""
    engine = SimpleNamespace(shutdown=lambda: None, profile=SimpleNamespace(dtype="float16"))
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=engine),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())
    monkeypatch.setattr(runner, "evaluate_batched", lambda *_a, **_k: {"fixture": dict(METRICS)})
    captured = _captured_manifest(monkeypatch, tmp_path)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    assert runner.run_evaluation(config, _args(tmp_path)) == 0
    assert captured[0].recorded_free["dtype"] == "float16"


SERVING = {
    "dtype": "bfloat16",
    "quantization": None,
    "speculative_decoding": None,
    "kv_budget": 0.25,
    "max_num_seqs": 48,
    "max_num_batched_tokens": 8192,
    "chunked_prefill": True,
    "prefix_caching": True,
}


def test_the_engine_s_serving_config_reaches_the_report_and_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--require-serving-match` had nothing to match against.

    The eight serving-detail flags were deleted in favour of recording what ran, but
    nothing replaced them as a source, so every field stayed `None` and the guard
    refused every real serving row on six of eight fields. The engine is the only
    party that knows what it resolved, so it is the source.
    """
    engine = SimpleNamespace(
        shutdown=lambda: None,
        profile=SimpleNamespace(dtype="bfloat16"),
        serving_config=lambda: dict(SERVING),
    )
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=engine),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())
    monkeypatch.setattr(runner, "evaluate_batched", lambda *_a, **_k: {"fixture": dict(METRICS)})
    captured = _captured_manifest(monkeypatch, tmp_path)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    assert runner.run_evaluation(config, _args(tmp_path)) == 0

    recorded = captured[0].recorded_free
    assert {field: recorded[field] for field in SERVING} == SERVING

    # The point of recording them: a throughput row measured under this config pairs,
    # and one measured under another does not.
    quality = QualityResult(score=1.0, manifest=captured[0])
    assert_quality_matches_serving(dict(SERVING), quality)
    with pytest.raises(ServingPairingError, match="max_num_seqs"):
        assert_quality_matches_serving({**SERVING, "max_num_seqs": 128}, quality)


def test_a_serving_config_read_that_fails_is_logged_not_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Provenance is evidence; a scored run is the deliverable."""

    def refuse() -> dict[str, Any]:
        raise RuntimeError("vllm_config moved")

    engine = SimpleNamespace(
        shutdown=lambda: None,
        profile=SimpleNamespace(dtype="float16"),
        serving_config=refuse,
    )
    monkeypatch.setattr(
        runner,
        "generation_for",
        lambda *_a, **_k: Generation(backend=object(), path="vllm", engine=engine),
    )
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "create_adapter", lambda *_a, **_k: _Adapter())
    monkeypatch.setattr(runner, "evaluate_batched", lambda *_a, **_k: {"fixture": dict(METRICS)})
    captured = _captured_manifest(monkeypatch, tmp_path)

    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="smolqwen.eval.runner"):
        assert runner.run_evaluation(config, _args(tmp_path)) == 0

    assert "serving config" in caplog.text
    # Still says which dtype ran; only the server-side detail is unknown.
    assert captured[0].recorded_free["dtype"] == "float16"
    assert captured[0].recorded_free["max_num_seqs"] is None


class _Adapter:
    """The narrowest adapter the runner's own bookkeeping touches."""

    def load_tasks(self) -> list[Any]:
        from smolqwen.eval.adapters.base import EvalTask

        return [EvalTask("case", "fixture", "prompt", ())]

    def manifest_invariants(self, tasks: Any) -> dict[str, Any]:
        return {"task_ids": ["case"]}

    def close(self) -> None:
        return None
