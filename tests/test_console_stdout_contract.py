"""stdout is a contract: every machine-readable emitter stays on stdout.

Rich went to stderr so a long run is legible, and the failure mode that introduces
is silent: route one `print()` through the logger and a notebook cell or a
documented `| jq` pipeline starts reading empty stdout. Nothing else would notice.

Two layers, because the emitters are not equally reachable:

- **Structural.** `EMITTERS` declares every `print()` in `src/` with the callable
  whose result it prints. An emitter moved to the logger disappears from the AST
  walk; a human-progress `print()` creeping back in appears as an extra. This is
  the completeness guarantee -- it covers `train-sft`'s final metrics and
  `rollout-bench`'s report, which need a trainer and a live worker pool
  respectively and so have no cheap behavioral test.
- **Behavioral.** Everything reachable without a GPU or a spawned pool runs for
  real, with stdout and stderr captured separately, and stdout is parsed by the
  consumer's own parser (`json.loads`, `shlex.split`) rather than matched loosely.
"""

from __future__ import annotations

import ast
import json
import logging
import shlex
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.cli import SUBCOMMAND_STAGES, main
from smolqwen.console import LOG_STREAM_ENV, configure_logging, logger, progress_task

SRC = Path(__file__).resolve().parents[1] / "src" / "smolqwen"

# module path relative to `src/smolqwen` -> the callables whose results reach
# stdout, sorted. "f-string" is a printed literal rather than a call.
EMITTERS: dict[str, tuple[str, ...]] = {
    "cli.py": ("f-string", "format_probe", "resolved_summary"),
    "env/selftest.py": ("json.dumps",),
    "eval/runner.py": ("json.dumps",),
    "rollout/bench.py": ("json.dumps",),
    "serving/server.py": ("shlex.join",),
    "training/merge.py": ("json.dumps",),
}


def _callee(node: ast.expr) -> str:
    if isinstance(node, ast.JoinedStr):
        return "f-string"
    if isinstance(node, ast.Call):
        return _callee(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_callee(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return type(node).__name__


def _prints_in(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def _observed_emitters() -> dict[str, tuple[str, ...]]:
    observed: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        calls = _prints_in(path)
        if not calls:
            continue
        key = path.relative_to(SRC).as_posix()
        observed[key] = sorted(_callee(call.args[0]) for call in calls if call.args)
    return {name: tuple(values) for name, values in observed.items()}


def test_every_stdout_emitter_is_declared_and_still_prints() -> None:
    """The declared table is the whole set: nothing added, nothing rerouted."""
    assert _observed_emitters() == EMITTERS


def test_no_emitter_writes_to_stderr_through_print() -> None:
    """Human output goes through the logger, not `print(file=sys.stderr)`.

    Four handlers used to do exactly that and dropped the traceback with it, so a
    config error and a bug in the same handler looked identical. `report_error` owns
    that path now, and this keeps the old shape from returning.
    """
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{call.lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for call in _prints_in(path)
        if any(keyword.arg == "file" for keyword in call.keywords)
    ]
    assert offenders == []


def test_probe_report_and_written_path_stay_on_stdout(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`test_cli_dry_run.py` reads the table from stdout; so does the probe notebook."""
    assert main(["probe", "--output-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "gpu available" in captured.out
    # The path is printed, and it is the path that was written: the slug depends on
    # the host, so the file is found rather than named.
    written = [line for line in captured.out.splitlines() if line.startswith("wrote ")]
    assert written and Path(written[0].removeprefix("wrote ")).exists()
    assert list(tmp_path.glob("*.json"))


def test_dry_run_summary_is_parseable_json_on_stdout_for_every_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for command in SUBCOMMAND_STAGES:
        profile = (
            []
            if command == "prepare-sft"
            else ["--profile", "balanced" if command == "serve" else "l4"]
        )
        assert main([command, *profile, "--dry-run"]) == 0
        captured = capsys.readouterr()
        # Parsed, not matched: a stray line on stdout breaks `json.loads` the same
        # way it breaks the notebook cell that pipes this into `jq`.
        assert isinstance(json.loads(captured.out), dict), command


def test_serve_print_command_stays_shell_parseable_on_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`docs/serving.md` captures this argv in a shell; no test guarded it before."""
    assert main(["serve", "--profile", "balanced", "--print-command"]) == 0
    captured = capsys.readouterr()
    argv = shlex.split(captured.out)
    assert argv[:2] == ["vllm", "serve"]
    assert "--api-key" not in argv
    assert captured.err == ""


def test_merge_report_json_stays_on_stdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`notebooks/01-sft.ipynb` runs `merge-adapter`; the merge itself needs weights."""
    from smolqwen.training import merge as merge_module

    result = merge_module.MergeResult(
        adapter_dir="adapter",
        output_dir="merged",
        base_model_id="Qwen/Qwen3.5-2B",
        base_revision="b" * 40,
        merged_parameters=7,
    )
    monkeypatch.setattr(merge_module, "merge_adapter", lambda **_: result)
    assert main(["merge-adapter", "--profile", "l4"]) == 0
    assert json.loads(capsys.readouterr().out) == result.to_dict()


def test_selftest_report_json_stays_on_stdout_and_failures_go_to_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is the stdout contract; `FAIL:` lines were the only other print."""
    from smolqwen.env import selftest as selftest_module

    configure_logging(level=logging.INFO)

    class _Pool:
        def __enter__(self) -> _Pool:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

    outcomes = {
        "selftest-initial": 0.0,
        "selftest-partial": 1 / 3,
        # Deliberately wrong, so the failure branch runs and its destination is
        # asserted rather than assumed.
        "selftest-full": 0.5,
    }

    def run_episode(pool: Any, scenario: Any, script: Any, *, episode_id: str) -> Any:
        reward = outcomes[episode_id]
        return selftest_module.EpisodeOutcome(
            task_id=scenario.task_id,
            steps=tuple(name for name, _ in script),
            reward=round(reward, 4),
            passed=0,
            total=scenario.check_count,
            name_errors=0,
            observations=(),
        )

    monkeypatch.setattr(selftest_module, "WorkerPool", lambda **_: _Pool())
    monkeypatch.setattr(selftest_module, "run_episode", run_episode)
    monkeypatch.setattr(
        selftest_module,
        "load_scenarios",
        lambda *_a, **_k: [
            SimpleNamespace(
                task_id=selftest_module.DEFAULT_SCENARIO_ID,
                env_id="env_151_rl",
                env_class_name="Fixture",
                init_config={},
                checklist=[],
                check_count=3,
            )
        ],
    )
    assert main(["env-selftest", "--profile", "l4"]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["scenario"] == selftest_module.DEFAULT_SCENARIO_ID
    assert report["ok"] is False
    assert "selftest FAIL" in captured.err
    assert "FAIL" not in captured.out


def test_evaluation_report_paths_are_json_while_progress_goes_to_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression this phase could have introduced, asserted directly.

    `evaluate` gained per-task progress. If that progress reached stdout, the JSON
    line another program parses would be preceded by prose and `json.loads` would
    fail -- which is exactly how a Colab cell reading this would break.
    """
    from smolqwen.config_models import EvalConfig
    from smolqwen.eval import bfcl_runner, runner
    from smolqwen.eval.bfcl_runner import BfclTask
    from tests.helpers import OfflineTokenizer

    configure_logging(level=logging.INFO)
    tokenizer = OfflineTokenizer(token_size=1)
    done_ids = tokenizer("done")["input_ids"]

    class _Engine:
        profile = SimpleNamespace(dtype="bfloat16")

        def generate_ids(self, prompts: Sequence[Any], **_: Any) -> list[Any]:
            return [SimpleNamespace(token_ids=done_ids, finish_reason="stop") for _ in prompts]

        def serving_config(self) -> dict[str, str]:
            return {"dtype": "bfloat16"}

        def shutdown(self) -> None:
            pass

    tasks = [
        BfclTask(
            f"case-{index}",
            "multi_turn_base",
            {
                "id": f"case-{index}",
                "question": [[{"role": "user", "content": "prompt"}]],
                "function": [],
                "initial_config": {},
                "involved_classes": [],
            },
            ((),),
        )
        for index in range(3)
    ]
    tracker = SimpleNamespace(
        start=lambda: None,
        log=lambda *_a, **_k: None,
        log_artifact=lambda *_a, **_k: None,
        finish=lambda: None,
    )
    monkeypatch.setattr(runner, "load_bfcl_tasks", lambda *_: (tasks, "b" * 40))
    monkeypatch.setattr(runner, "_engine_for", lambda *_: (_Engine(), None))
    monkeypatch.setattr(runner, "_tokenizer_for", lambda *_: tokenizer)
    monkeypatch.setattr(runner, "tracker_for", lambda *_a, **_k: tracker)
    monkeypatch.setattr(bfcl_runner, "_check", lambda *_a, **_k: {"valid": True})
    args = SimpleNamespace(
        checkpoint=str(tmp_path),
        revision="a" * 40,
        adapter_path=None,
        adapter_revision=None,
        tag="progress",
    )
    config = EvalConfig(output_dir=str(tmp_path))
    assert runner.run_evaluation(config, args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {"json", "markdown"}
    assert "multi_turn_base" in captured.err
    assert "mean" in captured.err


def test_progress_and_logs_never_reach_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """The shared helpers themselves, independent of any command.

    Also the non-TTY fallback: pytest's captured stderr is not a terminal, so the
    redrawing bar is disabled and the plain lines are the only progress output. A
    Colab cell is the same shape, and one frozen bar frame there is exactly how a
    stalled run reads as a slow one.
    """
    configure_logging(level=logging.INFO)
    with progress_task("unit", total=2, unit="things", every=1) as advance:
        advance("detail")
        advance()
    logger("test").info("a log line")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unit: 1/2 things" in captured.err
    assert "detail" in captured.err
    assert "unit: 2/2 things" in captured.err
    assert "unit complete: 2 things" in captured.err
    assert "a log line" in captured.err


def test_colab_logs_and_progress_stream_on_stdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Colab's shell can delay stderr, so its human stream is stdout by design."""
    monkeypatch.setenv("COLAB_GPU", "L4")
    configure_logging(level=logging.INFO)
    with progress_task("unit", total=1, unit="thing", every=1) as advance:
        advance()
    logger("test").info("a Colab log line")
    captured = capsys.readouterr()
    assert "unit: 1/1 thing" in captured.out
    assert "a Colab log line" in captured.out
    assert captured.err == ""


def test_colab_can_force_logs_back_to_stderr_for_a_pipeline(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLAB_GPU", "L4")
    monkeypatch.setenv(LOG_STREAM_ENV, "stderr")
    configure_logging(level=logging.INFO)
    logger("test").info("pipeline log")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pipeline log" in captured.err


def test_a_failing_progress_task_says_so_and_still_closes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A crashed run must not leave a bar reading "complete"."""
    with pytest.raises(RuntimeError, match="boom"):
        with progress_task("unit", total=2, unit="things", every=1) as advance:
            advance()
            raise RuntimeError("boom")
    captured = capsys.readouterr()
    assert "unit failed: 1 things" in captured.err
    assert captured.out == ""
