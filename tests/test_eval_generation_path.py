"""Which generation path `evaluate` takes, and that it takes the vLLM one.

This is the test Phase 4 should have had. `evaluate_batched` was written, tested
against a scripted backend, and then never called by `run_evaluation` -- so
`smolqwen evaluate` kept driving HuggingFace `generate()` at batch size 1 while the
plan's second goal was recorded as met. Nothing failed, because nothing asserted
which path the *command* took.

So the first test here is a wiring assertion: with an engine available, the batched
path runs and the serial one does not. It fails on the pre-fix code.

The rest cover the three recorded fallbacks. Each has a real cause -- an endpoint,
a host without vllm, an adapter vLLM refuses -- and each must be visible in the
report rather than inferred, because a base-model row and an adapter-on-base row
generated on different paths are still being compared in one table.
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


def test_no_policy_is_loaded_when_the_engine_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`TransformersPolicy.__init__` loads weights. Building both wastes the card."""

    def refuse_policy(**_k: Any) -> Any:
        raise AssertionError("a policy was constructed while an engine was available")

    monkeypatch.setattr(runner, "load_policy", refuse_policy)
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


def test_a_host_without_vllm_falls_back_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """vllm lives in the `serve`/`colab` extras and is absent from CI by design."""

    def no_vllm(*_a: Any, **_k: Any) -> Any:
        raise ImportError("No module named 'vllm'")

    monkeypatch.setattr(batched_module, "EvalProfile", SimpleNamespace(from_config=lambda _c: None))
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", no_vllm, raising=True)
    resolved = ResolvedCheckpoint(path=str(tmp_path), revision=SHA, source="local")
    with caplog.at_level(logging.WARNING, logger="smolqwen.eval.batched"):
        generation = generation_for(EvalConfig(), resolved)
    assert (generation.backend, generation.path) == (None, "transformers")
    assert "vllm is not installed" in caplog.text


def test_a_refused_adapter_falls_back_to_adapter_on_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`all-linear` emits LoRA weights for GDN projections; vLLM may refuse them.

    `TransformersPolicy` is the only path that evaluates an adapter without merging
    it, which is the whole reason it was kept.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    seen: list[dict[str, str] | None] = []

    def refuse(
        model: Any, profile: Any, *, revision: Any = None, adapter: Any = None, **_: Any
    ) -> Any:
        seen.append(adapter)
        raise ValueError("unsupported LoRA weight for architecture Qwen3_5ForCausalLM")

    monkeypatch.setattr(batched_module, "EvalProfile", SimpleNamespace(from_config=lambda _c: None))
    monkeypatch.setattr("smolqwen.inference.engine.offline_engine_for_eval", refuse, raising=True)
    resolved = ResolvedCheckpoint(
        path=str(tmp_path),
        revision=SHA,
        adapter_path=str(adapter_dir),
        adapter_revision="b" * 40,
        source="local",
    )
    with caplog.at_level(logging.WARNING, logger="smolqwen.eval.batched"):
        generation = generation_for(EvalConfig(), resolved)
    assert (generation.backend, generation.path) == (None, "transformers")
    assert "refused the adapter" in caplog.text
    # The adapter reached the engine under the one slot `evaluate` uses.
    assert seen == [{ADAPTER_SLOT: str(adapter_dir)}]


def test_a_failure_with_no_adapter_involved_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OOM must not quietly become a slower run reporting a different number."""

    def oom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(batched_module, "EvalProfile", SimpleNamespace(from_config=lambda _c: None))
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


class _Adapter:
    """The narrowest adapter the runner's own bookkeeping touches."""

    def load_tasks(self) -> list[Any]:
        from smolqwen.eval.adapters.base import EvalTask

        return [EvalTask("case", "fixture", "prompt", ())]

    def manifest_invariants(self, tasks: Any) -> dict[str, Any]:
        return {"task_ids": ["case"]}

    def close(self) -> None:
        return None
