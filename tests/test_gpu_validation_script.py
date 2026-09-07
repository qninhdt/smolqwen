"""The Colab validation controller must carry the requested sizing profile."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _script() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "colab-gpu-validation.py"
    spec = importlib.util.spec_from_file_location("smolqwen_gpu_validation", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("capability", "device_name", "expected"),
    [((7, 5), "Tesla T4", "t4"), ((8, 9), "NVIDIA L4", "l4"), ((8, 0), "NVIDIA A100", "a100")],
)
def test_auto_profile_matches_the_assigned_card(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    device_name: str,
    expected: str,
) -> None:
    module = _script()
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _index=0: capability,
        get_device_name=lambda _index=0: device_name,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    assert module._effective_profile("auto") == expected


def test_controller_forwards_explicit_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _script()
    calls: list[tuple[str, list[str], int]] = []
    device_info = {
        "gpu": "NVIDIA A100",
        "compute_capability": "8.0",
        "vram_gb": 40.0,
        "expected_attention": "flash_attention_2",
        "expected_dtype": "bfloat16",
    }

    monkeypatch.setattr(module, "_device_info", lambda: device_info)
    monkeypatch.setattr(module, "_effective_profile", lambda requested: requested)
    monkeypatch.setattr(module, "_load_results", lambda **_kwargs: [])
    monkeypatch.setattr(module, "_write_results", lambda _results: None)

    def run_child(
        _results: list[dict[str, Any]],
        name: str,
        args: list[str],
        *,
        timeout: int,
        run_id: str,
        profile: str,
        device: dict[str, Any],
    ) -> str:
        assert run_id
        assert profile == "a100"
        assert device == device_info
        calls.append((name, args, timeout))
        return "passed"

    monkeypatch.setattr(module, "_run_child", run_child)

    assert module._controller(("probe",), profile="a100") == 0
    assert calls == [("probe", ["--phase", "probe", "--profile", "a100"], 600)]


def test_runtime_environment_exposes_venv_console_scripts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _script()
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(module, "PYTHON", python)
    monkeypatch.setenv("PATH", "/system/bin")

    assert module._runtime_environment()["PATH"].split(os.pathsep) == [
        str(python.parent),
        "/system/bin",
    ]


def test_resume_rejects_rows_from_a_different_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _script()
    device = {
        "gpu": "Tesla T4",
        "compute_capability": "7.5",
        "vram_gb": 16.0,
        "expected_attention": "sdpa",
        "expected_dtype": "float16",
    }
    result = tmp_path / "gpu-validation.json"
    result.write_text(
        json.dumps(
            [
                {
                    "name": "probe",
                    "run_id": "old",
                    "model_id": module.MODEL_ID,
                    "model_revision": module.MODEL_REVISION,
                    "profile": "t4",
                    "device": device,
                    "status": "passed",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RESULT", result)

    with pytest.raises(RuntimeError, match="different profile, device, or model identity"):
        module._load_results(profile="l4", device=device)
