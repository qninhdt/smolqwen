"""`--dry-run` must validate config without importing torch CUDA paths."""

from __future__ import annotations

import json
import sys

import pytest

from smolqwen.cli import SUBCOMMAND_STAGES, build_parser, main


def test_every_stage_subcommand_dry_runs(capsys: pytest.CaptureFixture[str]) -> None:
    for command, stage in SUBCOMMAND_STAGES.items():
        exit_code = main([command, "--profile", "l4", "--dry-run"])
        assert exit_code == 0, f"{command} failed to dry-run"
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, dict)
        assert "profile" in payload, f"{command} resolved {stage} without a profile section"


def test_dry_run_does_not_initialise_cuda() -> None:
    # The point of --dry-run is that a typo fails at load rather than thirty
    # minutes into a run, which means it must be runnable on a machine with no
    # GPU at all. If torch was never imported, no CUDA context can exist.
    assert main(["train-sft", "--profile", "l4", "--dry-run"]) == 0
    torch_module = sys.modules.get("torch")
    if torch_module is not None:  # another test may have imported it first
        assert not torch_module.cuda.is_initialized()


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
    stderr = capsys.readouterr().err
    assert "config error" in stderr
    assert "lerning_rate" in stderr


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


def test_unimplemented_stage_exits_nonzero_naming_its_phase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A stub that silently succeeded would be mistaken for a completed run.
    exit_code = main(["train-grpo", "--profile", "l4"])
    assert exit_code == 2
    assert "Phase 7" in capsys.readouterr().err


def test_parser_exposes_pinned_revision_on_evaluate() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "org/repo",
            "--revision",
            "abc123",
            "--served-dtype",
            "float8_e4m3fn",
            "--serving-backend",
            "vllm",
            "--quantization",
            "fp8",
            "--max-num-seqs",
            "64",
            "--chunked-prefill",
            "--no-prefix-caching",
        ]
    )
    assert args.revision == "abc123"
    assert args.served_dtype == "float8_e4m3fn"
    assert args.serving_backend == "vllm"
    assert args.quantization == "fp8"
    assert args.max_num_seqs == 64
    assert args.chunked_prefill is True
    assert args.prefix_caching is False
