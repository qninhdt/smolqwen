"""`smolqwen` console entry point.

Subcommand bodies are thin dispatchers: the logic lives in the stage modules, and
the import of a stage module happens *inside* its handler. That keeps
`--dry-run` free of torch and CUDA -- config validation must not need a GPU, and
CI must be able to exercise every parser path with nothing installed but the base
dependencies.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from smolqwen.config import resolve, resolved_summary
from smolqwen.config_models import (
    PROFILES,
    ConfigError,
    DataConfig,
    EvalConfig,
    GrpoConfig,
    ServeConfig,
    SftConfig,
    StrictModel,
)
from smolqwen.console import configure_logging, report_error

# subcommand -> which stage config it resolves. `probe` is absent: it reads no
# config, because it must run on a fresh VM before anything is set up.
SUBCOMMAND_STAGES: dict[str, str] = {
    "profile-data": "data",
    "prepare-sft": "data",
    "train-sft": "sft",
    "merge-adapter": "sft",
    "env-selftest": "grpo",
    "profile-difficulty": "grpo",
    "rollout-bench": "grpo",
    "train-grpo": "grpo",
    "evaluate": "eval",
    "build-workload": "eval",
    "serve": "serve",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_logging(parser: argparse.ArgumentParser) -> None:
    """Verbosity for every command, `probe` included.

    On the top-level parser these would have to precede the subcommand, which is the
    opposite of how anyone types it, so each subparser gets its own pair.
    `SMOLQWEN_LOG_LEVEL` covers the Colab case where there is no place to add a flag.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="debug-level logs on stderr")
    group.add_argument(
        "--quiet", action="store_true", help="errors only; stdout output is unaffected"
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="explicit base config path")
    parser.add_argument("--profile", choices=PROFILES, default=None, help="GPU sizing profile")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate config, print it, and exit without touching the GPU",
    )
    parser.add_argument(
        "--budgets",
        type=Path,
        default=None,
        help="path to budgets.json (defaults to artifacts/data/budgets.json)",
    )
    _add_logging(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smolqwen",
        description="Agentic post-training and optimized serving for Qwen3.5-2B",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser(
        "probe", help="report GPU capability and installed versions"
    )
    probe_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/probe"), help="where to write the JSON"
    )
    probe_parser.add_argument("--no-write", action="store_true", help="print only")
    _add_logging(probe_parser)

    profile_data = subparsers.add_parser(
        "profile-data", help="measure the trajectory distribution and write budgets.json"
    )
    _add_common(profile_data)

    prepare_sft = subparsers.add_parser(
        "prepare-sft", help="convert trajectories into rendered SFT samples"
    )
    _add_common(prepare_sft)
    prepare_sft.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="CPU render workers (default: auto, up to 4)",
    )

    sft = subparsers.add_parser("train-sft", help="LoRA reasoning SFT")
    _add_common(sft)
    sft.add_argument(
        "--resume", action="store_true", help="continue from the newest pushed revision"
    )

    merge = subparsers.add_parser("merge-adapter", help="merge a LoRA adapter into the base model")
    _add_common(merge)
    merge.add_argument("--adapter-dir", type=Path, default=None)
    merge.add_argument("--output-dir", type=Path, default=None)
    merge.add_argument(
        "--push",
        action="store_true",
        help=(
            "upload the merged checkpoint to tracking.merged_hub_repo_id; "
            "several GB, so opt-in rather than automatic"
        ),
    )

    selftest = subparsers.add_parser(
        "env-selftest", help="run a scripted episode end to end against a real scenario"
    )
    _add_common(selftest)
    selftest.add_argument("--scenario-id", default=None)
    selftest.add_argument("--limit", type=int, default=1)

    difficulty = subparsers.add_parser(
        "profile-difficulty", help="classify scenarios into always-zero / band / always-one"
    )
    _add_common(difficulty)
    difficulty.add_argument("--checkpoint", type=Path, default=None)
    difficulty.add_argument("--revision", default=None)

    bench_rollout = subparsers.add_parser(
        "rollout-bench", help="verify rollout equivalence and emit rollout diagnostics"
    )
    _add_common(bench_rollout)
    bench_rollout.add_argument("--episodes", type=int, default=64)
    bench_rollout.add_argument(
        "--paths",
        default="serial_oracle,async",
        help="comma-separated rollout paths to benchmark",
    )

    grpo = subparsers.add_parser("train-grpo", help="agentic GRPO from the SFT checkpoint")
    _add_common(grpo)
    grpo.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="run a benchmark adapter against a checkpoint"
    )
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", default=None, help="local path or Hub repo id")
    evaluate.add_argument(
        "--revision",
        default=None,
        help="checkpoint revision sha; required for Hub reads, never resolved implicitly",
    )
    evaluate.add_argument("--tag", required=False, default=None, help="column label, e.g. base/sft")
    evaluate.add_argument("--adapter", default=None, help="benchmark adapter name")
    evaluate.add_argument(
        "--adapter-path", default=None, help="PEFT adapter directory or pinned Hub revision"
    )
    evaluate.add_argument(
        "--adapter-revision", default=None, help="explicit revision sha for a PEFT adapter"
    )
    evaluate.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL")
    evaluate.add_argument(
        "--serving-backend",
        default=None,
        help="actual serving engine recorded in the manifest, for example vllm",
    )
    # The eight serving-detail flags this used to carry are gone. The in-process
    # engine knows its own dtype, KV budget, batching and caching and records them,
    # so asserting them on the command line only created a way to record something
    # other than what ran. `--serving-backend` stays because the served process is a
    # separate one whose engine this command cannot inspect.
    evaluate.add_argument(
        "--require-serving-match",
        type=Path,
        default=None,
        help=(
            "an evaluation report whose recorded serving config must match this run's; "
            "refuses a paired speed/quality row measured under a different config"
        ),
    )

    serve = subparsers.add_parser("serve", help="launch the vLLM endpoint")
    _add_common(serve)
    serve.add_argument("--print-command", action="store_true", help="print argv and exit")

    workload = subparsers.add_parser(
        "build-workload", help="write BFCL-shaped benchmark traffic for vllm bench serve"
    )
    _add_common(workload)
    workload.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/serving/bfcl-agentic.jsonl"),
        help="where to write the rendered prompts",
    )

    return parser


def _resolve_for(args: argparse.Namespace) -> StrictModel:
    stage = SUBCOMMAND_STAGES[args.command]
    return resolve(
        stage,
        profile=args.profile,
        overrides=args.override,
        config_path=args.config,
        budgets_path=args.budgets,
    )


def _cmd_probe(args: argparse.Namespace) -> int:
    from smolqwen.probe import format_probe, probe, write_probe

    report = probe()
    # The formatted table is what a human reads and what `notebooks/00-probe-gpu`
    # shows, and no program parses it — but `test_cli_dry_run.py:73` captures it from
    # stdout, so it stays there rather than moving to the logger.
    print(format_probe(report))
    if not args.no_write:
        path = write_probe(report, args.output_dir)
        print(f"\nwrote {path}")
    return 0


def _cmd_profile_data(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.data.cli_actions import run_profile_data

    return run_profile_data(_as(config, DataConfig))


def _cmd_prepare_sft(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.data.cli_actions import run_prepare_sft

    return run_prepare_sft(_as(config, DataConfig), workers=args.workers)


ConfigT = TypeVar("ConfigT", bound=StrictModel)


def _as(config: StrictModel, kind: type[ConfigT]) -> ConfigT:
    """Narrow a resolved config to the stage model its handler needs.

    One generic in place of four character-identical functions. The check is real,
    not decorative: `SUBCOMMAND_STAGES` maps a subcommand to a stage name, and a
    wrong entry there would otherwise hand a handler the wrong model and fail deep
    inside it on a missing attribute.
    """
    if not isinstance(config, kind):
        raise TypeError(f"expected {kind.__name__}, got {type(config).__name__}")
    return config


def _cmd_env_selftest(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.env.selftest import run_selftest

    return run_selftest(_as(config, GrpoConfig), scenario_id=args.scenario_id)


def _cmd_train_sft(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.sft import run_train_sft

    return run_train_sft(_as(config, SftConfig), resume=args.resume)


def _cmd_merge_adapter(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.merge import run_merge_adapter

    return run_merge_adapter(
        _as(config, SftConfig),
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        push=args.push,
    )


def _cmd_evaluate(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.config_models import EvalConfig
    from smolqwen.eval.runner import run_evaluation

    if not isinstance(config, EvalConfig):
        raise TypeError(f"expected EvalConfig, got {type(config).__name__}")
    return run_evaluation(config, args)


def _cmd_rollout_bench(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.rollout.bench import run_bench

    if not isinstance(config, GrpoConfig):
        raise TypeError(f"expected GrpoConfig, got {type(config).__name__}")
    return run_bench(config, args=args)


def _cmd_profile_difficulty(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.difficulty import DifficultyError
    from smolqwen.training.grpo import GrpoError, run_profile_difficulty

    grpo = _as(config, GrpoConfig)
    update: dict[str, Any] = {}
    if args.checkpoint is not None:
        update["model_id"] = str(args.checkpoint)
    if args.revision is not None:
        update["model_revision"] = args.revision
    if update:
        grpo = grpo.model_copy(update=update)
    try:
        return run_profile_difficulty(grpo)
    except (DifficultyError, GrpoError) as exc:
        report_error(f"GRPO error: {exc}", exception=exc)
        return 2


def _cmd_train_grpo(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.difficulty import DifficultyError
    from smolqwen.training.grpo import GrpoError, run_train_grpo

    try:
        return run_train_grpo(_as(config, GrpoConfig), resume=args.resume)
    except (DifficultyError, GrpoError) as exc:
        report_error(f"GRPO error: {exc}", exception=exc)
        return 2


def _cmd_serve(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.serving.server import ServingError, run_server

    try:
        return run_server(_as(config, ServeConfig), print_command=args.print_command)
    except ServingError as exc:
        report_error(f"serving error: {exc}", exception=exc)
        return 2


def _cmd_build_workload(args: argparse.Namespace, config: StrictModel) -> int:
    """Render BFCL-shaped traffic for `vllm bench serve --dataset-name custom`.

    Kept because it is the only thing that makes a serving benchmark measure *this*
    workload: `--dataset-name random` measures token throughput on synthetic prompts,
    which says nothing about an agentic request's prefill shape or tool-schema
    overhead. The upstream commands own execution; this owns the traffic.
    """
    from smolqwen.eval.workload import build_bfcl_agentic_workload
    from smolqwen.tokenizer import load_tokenizer

    evaluation = _as(config, EvalConfig)
    tokenizer = load_tokenizer(evaluation.http_model or "Qwen/Qwen3.5-2B")
    workload, composition = build_bfcl_agentic_workload(
        evaluation, tokenizer=tokenizer, output_path=args.output
    )
    print(json.dumps({"workload": str(workload), "composition": str(composition)}, sort_keys=True))
    return 0


DISPATCH: dict[str, Callable[[argparse.Namespace, StrictModel], int]] = {
    "profile-data": _cmd_profile_data,
    "prepare-sft": _cmd_prepare_sft,
    "train-sft": _cmd_train_sft,
    "merge-adapter": _cmd_merge_adapter,
    "env-selftest": _cmd_env_selftest,
    "evaluate": _cmd_evaluate,
    "rollout-bench": _cmd_rollout_bench,
    "profile-difficulty": _cmd_profile_difficulty,
    "train-grpo": _cmd_train_grpo,
    "serve": _cmd_serve,
    "build-workload": _cmd_build_workload,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Before dispatch, so a stage module's own logger is already routed by the time
    # its first import emits anything.
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.command == "probe":
        return _cmd_probe(args)

    try:
        config = _resolve_for(args)
    except ConfigError as exc:
        report_error(f"config error: {exc}", exception=exc)
        return 1

    if args.dry_run:
        # Machine-readable: `notebooks/01-sft.ipynb` and `test_cli_dry_run.py` both
        # read this from stdout.
        print(resolved_summary(config))
        return 0

    return DISPATCH[args.command](args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
