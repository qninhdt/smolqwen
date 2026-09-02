"""`--dry-run` must validate config without importing torch CUDA paths."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from smolqwen.cli import SUBCOMMAND_STAGES, build_parser, main
from smolqwen.console import LOG_LEVEL_ENV, resolve_level


def test_every_stage_subcommand_dry_runs(capsys: pytest.CaptureFixture[str]) -> None:
    for command, stage in SUBCOMMAND_STAGES.items():
        exit_code = main([command, "--profile", "l4", "--dry-run"])
        assert exit_code == 0, f"{command} failed to dry-run"
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, dict)
        assert "profile" in payload, f"{command} resolved {stage} without a profile section"


def test_verbose_logging_keeps_the_dry_run_stdout_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug logs are the loudest this gets, and none of it may reach stdout.

    `--verbose` on every subparser rather than the top-level parser, because on the
    top level it would have to precede the subcommand -- the opposite of how anyone
    types it. Which means every subcommand can now emit debug output into a stream
    another program parses, so the purity is asserted at the loudest level.
    """
    for command in SUBCOMMAND_STAGES:
        assert main([command, "--profile", "l4", "--dry-run", "--verbose"]) == 0
        captured = capsys.readouterr()
        assert isinstance(json.loads(captured.out), dict), command


def test_log_level_resolution_prefers_flags_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--quiet` beats `--verbose` beats `SMOLQWEN_LOG_LEVEL` beats the default.

    `--quiet` wins the flag conflict because it is what a caller uses to make output
    parseable in a pipeline; silently upgrading that to DEBUG because a shell profile
    exported the variable would be the wrong surprise. The variable exists for Colab,
    where there is no place to add a flag.
    """
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    assert resolve_level() == logging.INFO
    assert resolve_level(verbose=True) == logging.DEBUG
    assert resolve_level(quiet=True) == logging.ERROR
    assert resolve_level(verbose=True, quiet=True) == logging.ERROR

    monkeypatch.setenv(LOG_LEVEL_ENV, "warning")
    assert resolve_level() == logging.WARNING
    assert resolve_level(verbose=True) == logging.DEBUG

    monkeypatch.setenv(LOG_LEVEL_ENV, "not-a-level")
    assert resolve_level() == logging.INFO


def test_verbose_and_quiet_are_mutually_exclusive_per_subcommand() -> None:
    parser = build_parser()
    assert parser.parse_args(["probe", "--quiet"]).quiet is True
    with pytest.raises(SystemExit):
        parser.parse_args(["train-sft", "--verbose", "--quiet"])


def test_dry_run_does_not_initialise_cuda() -> None:
    # The point of --dry-run is that a typo fails at load rather than thirty
    # minutes into a run, which means it must be runnable on a machine with no
    # GPU at all. If torch was never imported, no CUDA context can exist.
    #
    # Before/after rather than "not initialized": `probe` legitimately queries the
    # device, so on a machine with a card any earlier test that probed leaves a
    # context behind and an absolute assertion would fail on test order rather than
    # on anything this command did. `test_console_no_heavy_imports.py` makes the
    # stronger claim -- torch not imported at all -- in a fresh interpreter.
    before = _cuda_initialized()
    assert main(["train-sft", "--profile", "l4", "--dry-run"]) == 0
    assert _cuda_initialized() == before


def _cuda_initialized() -> bool:
    torch_module = sys.modules.get("torch")
    if torch_module is None:  # never imported, so no context can exist
        return False
    return bool(torch_module.cuda.is_initialized())


def test_override_reaches_the_dry_run_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "train-sft",
                "--profile",
                "l4",
                "--override",
                "training.learning_rate=1e-4",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["training"]["learning_rate"] == pytest.approx(1e-4)


def test_unknown_override_key_exits_nonzero_with_a_readable_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["train-sft", "--override", "training.lerning_rate=1e-4", "--dry-run"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "config error" in captured.err
    assert "lerning_rate" in captured.err
    # The error goes to the logger, so stdout is empty rather than carrying half a
    # config summary a caller would try to parse.
    assert captured.out == ""


def test_unknown_profile_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train-sft", "--profile", "l4", "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        # argparse rejects the choice before config resolution sees it.
        main(["train-sft", "--profile", "h100", "--dry-run"])


def test_probe_subcommand_needs_no_config(capsys: pytest.CaptureFixture[str]) -> None:
    # probe runs on a fresh VM before anything is configured, so it must not
    # resolve a stage config at all.
    assert "probe" not in SUBCOMMAND_STAGES
    assert main(["probe", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "gpu available" in out


def test_train_grpo_without_a_profile_names_the_required_preflight(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["train-grpo", "--profile", "l4"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "profile-difficulty" in captured.err
    assert "GRPO error" in captured.err
    assert captured.out == ""


def test_parser_exposes_pinned_revision_on_evaluate() -> None:
    """The revision stays mandatory; the eight serving-detail flags are gone.

    The in-process engine knows its own dtype, KV budget, batching and caching and
    records what it used, so asserting those on the command line only created a way
    to record something other than what ran. `--serving-backend` survives because
    the served process is a separate one this command cannot inspect.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "org/repo",
            "--revision",
            "abc123",
            "--serving-backend",
            "vllm",
        ]
    )
    assert args.revision == "abc123"
    assert args.serving_backend == "vllm"

    for removed in ("--served-dtype", "--quantization", "--max-num-seqs", "--chunked-prefill"):
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--revision", "abc123", removed, "x"])


def test_prepare_sft_parser_accepts_only_positive_worker_count() -> None:
    parser = build_parser()
    assert parser.parse_args(["prepare-sft", "--workers", "3"]).workers == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare-sft", "--workers", "0"])


def test_serving_reports_a_missing_key_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The assertion that survived the benchmark wrapper's deletion.

    `serve` is the remaining subcommand that needs the key, and a missing key must
    still exit 2 with the variable named rather than raising through argparse.
    """
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    override = f"output_dir={tmp_path}"
    assert main(["serve", "--profile", "l4", "--override", override]) == 2
    captured = capsys.readouterr()
    assert "VLLM_API_KEY" in captured.err
    # `serve --print-command` writes argv to stdout, so a failing `serve` must not
    # write anything there that a shell capture would consume as argv.
    assert captured.out == ""
