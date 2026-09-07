from __future__ import annotations

import time

import pytest

from smolqwen.console import phase


def test_phase_reports_start_heartbeat_and_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="smolqwen.phase"):
        with phase("test phase", every=0.01):
            time.sleep(0.03)

    assert "test phase: start" in caplog.text
    assert "test phase still running" in caplog.text
    assert "test phase: complete" in caplog.text
