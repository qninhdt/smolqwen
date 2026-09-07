"""Small streaming runner shared by the Colab validation scripts."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast


@dataclass(frozen=True)
class StreamResult:
    returncode: int
    output: str
    duration_s: float

    @property
    def output_tail(self) -> str:
        return self.output[-12_000:]


def run_streaming(
    command: Sequence[str],
    *,
    name: str,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    heartbeat_s: float = 30.0,
    stream: TextIO | None = None,
) -> StreamResult:
    """Run a child while forwarding every output line and quiet heartbeat."""
    import sys

    output_stream = stream or sys.stdout
    started = time.monotonic()
    timeout_value = float(timeout) if timeout is not None else 0.0
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    deadline = started + timeout if timeout is not None else None

    print(f"[{name}] started", file=output_stream, flush=True)
    try:
        while selector.get_map():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                _kill_process(process)
                raise subprocess.TimeoutExpired(list(command), timeout_value)
            wait_for = heartbeat_s
            if deadline is not None:
                wait_for = min(wait_for, max(0.0, deadline - now))
            events = selector.select(wait_for)
            if not events:
                if deadline is not None and time.monotonic() >= deadline:
                    _kill_process(process)
                    error = subprocess.TimeoutExpired(list(command), timeout_value)
                    error.output = "\n".join(lines)
                    raise error
                print(
                    f"[{name}] still running ({time.monotonic() - started:.0f}s elapsed)",
                    file=output_stream,
                    flush=True,
                )
                continue
            for key, _ in events:
                stream_handle = cast(TextIO, key.fileobj)
                line = stream_handle.readline()
                if line == "":
                    selector.unregister(stream_handle)
                    stream_handle.close()
                    continue
                line = line.rstrip("\n")
                lines.append(line)
                print(f"[{name}] {line}", file=output_stream, flush=True)
        returncode = process.wait()
    except subprocess.TimeoutExpired as exc:
        exc.output = "\n".join(lines)
        if process.poll() is None:
            _kill_process(process)
        process.wait()
        raise
    except BaseException:
        if process.poll() is None:
            _kill_process(process)
        process.wait()
        raise
    finally:
        selector.close()
    elapsed = time.monotonic() - started
    print(f"[{name}] exited {returncode} after {elapsed:.1f}s", file=output_stream, flush=True)
    return StreamResult(returncode, "\n".join(lines), elapsed)


def _kill_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - Colab is Linux
        process.kill()
