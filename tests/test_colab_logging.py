from __future__ import annotations

import sys

import pytest
from scripts.colab_logging import run_streaming


def test_streaming_runner_forwards_child_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_streaming(
        [sys.executable, "-c", "print('ready', flush=True)"],
        name="child",
        heartbeat_s=0.1,
    )

    assert result.returncode == 0
    assert "ready" in result.output
    assert "[child] ready" in capsys.readouterr().out


def test_streaming_runner_reports_quiet_children(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_streaming(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        name="quiet",
        heartbeat_s=0.01,
    )

    assert result.returncode == 0
    assert "[quiet] still running" in capsys.readouterr().out
