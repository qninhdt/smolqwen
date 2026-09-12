"""`--dry-run` must work where torch and vllm are not installed.

The console module is imported by `cli.main` before dispatch, so anything it pulls
in is pulled in by every command including `--dry-run`. CI installs neither vllm nor
a CUDA torch by construction, and config validation must not need a GPU -- so a
transitive `import torch` here would turn a config typo into an ImportError on the
one machine that is supposed to catch the typo.

A subprocess, not `sys.modules`: by the time pytest collects, another test has
already imported transformers, so an in-process assertion would pass for the wrong
reason.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY = ("torch", "vllm", "transformers", "trl", "peft", "datasets")

PROBE = """
import json
import sys

import smolqwen.console
import smolqwen.config
import smolqwen.config_models
import smolqwen.cli

print(json.dumps(sorted(name for name in {heavy!r} if name in sys.modules)))
"""


def _imported_heavy(source: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = completed.stdout.strip().splitlines()[-1]
    return list(json.loads(loaded))


def test_importing_the_console_and_cli_pulls_in_no_ml_library() -> None:
    assert _imported_heavy(PROBE.format(heavy=HEAVY)) == []


def test_configure_logging_routes_library_loggers_without_importing_them() -> None:
    """`logging.getLogger` resolves a name whether or not the package is installed.

    That is why `LIBRARY_LOGGERS` is a tuple of strings rather than imports, and it
    is the property that lets this phase route transformers' and vllm's output
    through one handler without making them a dependency of `--dry-run`.
    """
    source = """
import json
import logging
import sys

from smolqwen.console import LIBRARY_LOGGERS, configure_logging

configure_logging(level=logging.INFO)
routed = [
    name
    for name in LIBRARY_LOGGERS
    if len(logging.getLogger(name).handlers) == 1 and not logging.getLogger(name).propagate
]
assert routed == list(LIBRARY_LOGGERS), routed
print(json.dumps(sorted(name for name in {heavy!r} if name in sys.modules)))
"""
    assert _imported_heavy(source.format(heavy=HEAVY)) == []


def test_dry_run_of_every_stage_needs_no_ml_library() -> None:
    """The end-to-end version: a real `--dry-run`, in a fresh interpreter."""
    source = """
import json
import sys

from smolqwen.cli import SUBCOMMAND_STAGES, main

for command in SUBCOMMAND_STAGES:
    profile = []
    if command != "prepare-sft":
        profile = ["--profile", "balanced" if command == "serve" else "l4"]
    assert main([command, *profile, "--dry-run", "--quiet"]) == 0, command
print(json.dumps(sorted(name for name in {heavy!r} if name in sys.modules)))
"""
    assert _imported_heavy(source.format(heavy=HEAVY)) == []
