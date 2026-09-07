"""One console for every command: rich on **stderr**, machine output on stdout.

stdout is a contract here, not a convenience. Machine-readable command output stays
on stdout, while every human-facing line goes to stderr so `smolqwen ... | jq` keeps
working while a run is still legible.

The concrete failure this fixes: `eval/runner.py` used to run for hours and emit one
JSON line at the end, so a stalled run was indistinguishable from a slow one.

**Non-TTY is the normal case.** Colab cells are not terminals, and a rich progress
bar there floods the output with redraw frames. `data/cli_actions.py` already solved
that with a bar plus periodic plain lines; `progress_task` is that pattern promoted
to one place rather than reinvented per module.

Importing this module must not import torch or vllm. `--dry-run` resolves a config
where neither is installed, and logging is configured before dispatch.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Read at `configure_logging` time so a Colab cell can set it without a flag.
LOG_LEVEL_ENV = "SMOLQWEN_LOG_LEVEL"

# Libraries whose own loggers should reach the same handler. Left as strings rather
# than imports: this module must stay free of torch and vllm, and `logging` resolves
# a logger by name whether or not the package is installed.
LIBRARY_LOGGERS = ("transformers", "trl", "vllm", "peft", "accelerate", "datasets")

# How often the non-TTY fallback prints a plain line. Frequent enough that a stalled
# run is visible within a minute of real work, rare enough not to flood a cell.
PLAIN_LINE_EVERY = 250

_console: Console | None = None


def console() -> Console:
    """The one console. On stderr, so stdout stays a machine-readable stream."""
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def is_terminal() -> bool:
    """Whether a redrawing progress bar is appropriate for this output stream."""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def resolve_level(*, verbose: bool = False, quiet: bool = False) -> int:
    """Flags beat the environment; `--quiet` beats `--verbose`.

    `--quiet` wins because it is the one a caller uses to make output parseable in a
    pipeline, and silently upgrading that to DEBUG because a shell profile exported
    the variable would be the wrong surprise.
    """
    if quiet:
        return logging.ERROR
    if verbose:
        return logging.DEBUG
    named = os.environ.get(LOG_LEVEL_ENV, "").strip().upper()
    if named:
        resolved = logging.getLevelName(named)
        if isinstance(resolved, int):
            return resolved
    return logging.INFO


def configure_logging(level: int | None = None, **flags: bool) -> None:
    """Route this package's logs and the ML libraries' through one rich handler.

    Idempotent: the handler is replaced rather than appended, because transformers
    installs its own on first import and a second call would otherwise double every
    line.
    """
    resolved = level if level is not None else resolve_level(**flags)
    handler = RichHandler(
        console=console(),
        rich_tracebacks=True,
        show_path=False,
        # A redrawing bar and a live handler fight over the same lines outside a
        # terminal; plain output is what a notebook cell can actually render.
        markup=False,
        show_time=is_terminal(),
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger("smolqwen")
    root.handlers = [handler]
    root.setLevel(resolved)
    root.propagate = False

    for name in LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.handlers = [handler]
        library.setLevel(max(resolved, logging.INFO))
        library.propagate = False


def logger(name: str) -> logging.Logger:
    """A `smolqwen.<module>` logger, so a level can be set per module."""
    suffix = name.removeprefix("smolqwen.")
    return logging.getLogger(f"smolqwen.{suffix}" if suffix else "smolqwen")


@contextmanager
def progress_task(
    description: str,
    *,
    total: int | None = None,
    unit: str = "items",
    every: int = PLAIN_LINE_EVERY,
) -> Iterator[Callable[..., None]]:
    """A progress bar plus periodic plain lines, both on stderr.

    Yields an `advance(detail=...)` the caller invokes once per unit of work. The
    plain lines are not redundant with the bar: outside a terminal the bar cannot
    redraw, and a Colab cell showing one frozen frame is exactly how a stalled run
    reads as a slow one. `data/cli_actions.py` discovered that; this is the same fix
    in one place.

    `detail` is appended to the plain line when one is emitted, so a caller with a
    running statistic (an evaluation's mean score) can surface it without deciding
    the cadence itself. It is computed by the caller every call and printed rarely,
    which is the right trade for a string built from numbers already in hand.
    """
    surface = console()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=surface,
        transient=False,
        # Without this the bar emits one frame per update into a non-TTY stream.
        disable=not is_terminal(),
    )
    task_id = progress.add_task(description, total=total)
    started = monotonic()
    done = 0
    finished = False

    def advance(detail: str | None = None) -> None:
        nonlocal done
        done += 1
        progress.advance(task_id)
        if done == 1 or done % every == 0:
            elapsed = monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            count = f"{done}/{total} {unit}" if total is not None else f"{done} {unit}"
            suffix = f", {detail}" if detail else ""
            surface.print(f"{description}: {count} ({rate:.1f}/s, {elapsed:.0f}s elapsed{suffix})")

    with progress:
        try:
            yield advance
            finished = True
        finally:
            elapsed = monotonic() - started
            status = "complete" if finished else "failed"
            progress.update(task_id, description=f"{description} {status}")
            surface.print(f"{description} {status}: {done} {unit} in {elapsed:.1f}s")


def status_table(title: str, rows: dict[str, Any]) -> None:
    """Print a two-column summary to stderr. Never a substitute for a JSON emitter."""
    table = Table(title=title, show_header=False, title_justify="left")
    table.add_column("field", style="bold")
    table.add_column("value")
    for name, value in rows.items():
        table.add_row(str(name), str(value))
    console().print(table)


def report_error(message: str, *, exception: BaseException | None = None) -> None:
    """One error path: a rich traceback when there is one, the message either way.

    Replaces four `print(f"...: {exc}", file=sys.stderr)` handlers whose output
    dropped the traceback entirely, so a config error and a bug in the same handler
    looked identical.
    """
    log = logger("cli")
    if exception is not None:
        log.error(message, exc_info=exception)
    else:
        log.error(message)
