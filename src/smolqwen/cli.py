"""`smolqwen` console entry point.

Subcommand bodies are thin dispatchers: the logic lives in the stage modules, and
the import of a stage module happens *inside* its handler. That keeps
`--dry-run` free of torch and CUDA -- config validation must not need a GPU, and
CI must be able to exercise every parser path with nothing installed but the base
dependencies.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from smolqwen.config import resolve, resolved_summary
from smolqwen.config_models import (
    PROFILES,
    ConfigError,
    DataConfig,
    GrpoConfig,
    ServeConfig,
    SftConfig,
    StrictModel,
)

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
    "serve": "serve",
    "bench": "serve",
    "sweep": "serve",
}


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

    data_help = {
        "profile-data": "measure the trajectory distribution and write budgets.json",
        "prepare-sft": "convert trajectories into rendered SFT samples",
    }
    for name, help_text in data_help.items():
        sub = subparsers.add_parser(name, help=help_text)
        _add_common(sub)

    sft = subparsers.add_parser("train-sft", help="LoRA reasoning SFT")
    _add_common(sft)
    sft.add_argument(
        "--resume", action="store_true", help="continue from the newest pushed revision"
    )

    merge = subparsers.add_parser("merge-adapter", help="merge a LoRA adapter into the base model")
    _add_common(merge)
    merge.add_argument("--adapter-dir", type=Path, default=None)
    merge.add_argument("--output-dir", type=Path, default=None)

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
    evaluate.add_argument("--served-dtype", default=None, help="served dtype recorded in manifest")
    evaluate.add_argument(
        "--quantization", default=None, help="quantization scheme recorded in manifest"
    )
    evaluate.add_argument(
        "--speculative-decoding",
        default=None,
        help="speculative-decoding config recorded in manifest",
    )
    evaluate.add_argument(
        "--kv-budget",
        default=None,
        help="KV-cache budget (for example 0.25 or 8GiB) recorded in manifest",
    )
    evaluate.add_argument("--max-num-seqs", type=int, default=None)
    evaluate.add_argument("--max-num-batched-tokens", type=int, default=None)
    evaluate.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record whether the served endpoint uses chunked prefill",
    )
    evaluate.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record whether the served endpoint uses prefix caching",
    )

    serve = subparsers.add_parser("serve", help="launch the vLLM endpoint")
    _add_common(serve)
    serve.add_argument("--print-command", action="store_true", help="print argv and exit")

    bench = subparsers.add_parser("bench", help="run vllm bench serve against a live endpoint")
    _add_common(bench)
    bench.add_argument("--dataset", default="sharegpt")
    bench.add_argument("--dataset-path", type=Path, default=None)
    bench.add_argument("--concurrency", default="1,4,16")
    bench.add_argument("--quality-report", type=Path, default=None)
    bench.add_argument(
        "--quality-reference",
        type=Path,
        action="append",
        default=[],
        help="reference evaluation report whose invariant manifest must match",
    )

    sweep = subparsers.add_parser("sweep", help="drive vllm bench sweep serve and read the front")
    _add_common(sweep)
    sweep.add_argument("--resume", action="store_true", default=True)
    sweep.add_argument("--experiment-name", default=None)

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


def _not_implemented(phase: int, what: str) -> Callable[..., int]:
    """A handler for a stage whose module has not landed yet.

    Exiting non-zero with the owning phase named beats a stub that silently
    succeeds and gets mistaken for a completed run.
    """

    def handler(*_: Any, **__: Any) -> int:
        print(f"{what} is implemented in Phase {phase}", file=sys.stderr)
        return 2

    return handler


def _cmd_probe(args: argparse.Namespace) -> int:
    from smolqwen.probe import format_probe, probe, write_probe

    report = probe()
    print(format_probe(report))
    if not args.no_write:
        path = write_probe(report, args.output_dir)
        print(f"\nwrote {path}")
    return 0


def _cmd_profile_data(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.data.cli_actions import run_profile_data

    return run_profile_data(_as_data_config(config))


def _cmd_prepare_sft(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.data.cli_actions import run_prepare_sft

    return run_prepare_sft(_as_data_config(config))


def _as_data_config(config: StrictModel) -> DataConfig:
    if not isinstance(config, DataConfig):
        raise TypeError(f"expected DataConfig, got {type(config).__name__}")
    return config


def _as_sft_config(config: StrictModel) -> SftConfig:
    if not isinstance(config, SftConfig):
        raise TypeError(f"expected SftConfig, got {type(config).__name__}")
    return config


def _as_grpo_config(config: StrictModel) -> GrpoConfig:
    if not isinstance(config, GrpoConfig):
        raise TypeError(f"expected GrpoConfig, got {type(config).__name__}")
    return config


def _as_serve_config(config: StrictModel) -> ServeConfig:
    if not isinstance(config, ServeConfig):
        raise TypeError(f"expected ServeConfig, got {type(config).__name__}")
    return config


def _cmd_env_selftest(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.env.selftest import run_selftest

    return run_selftest(_as_grpo_config(config), scenario_id=args.scenario_id)


def _cmd_train_sft(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.sft import run_train_sft

    return run_train_sft(_as_sft_config(config), resume=args.resume)


def _cmd_merge_adapter(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.merge import run_merge_adapter

    return run_merge_adapter(
        _as_sft_config(config),
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
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

    grpo = _as_grpo_config(config)
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
        print(f"GRPO error: {exc}", file=sys.stderr)
        return 2


def _cmd_train_grpo(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.training.difficulty import DifficultyError
    from smolqwen.training.grpo import GrpoError, run_train_grpo

    try:
        return run_train_grpo(_as_grpo_config(config), resume=args.resume)
    except (DifficultyError, GrpoError) as exc:
        print(f"GRPO error: {exc}", file=sys.stderr)
        return 2


def _cmd_serve(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.serving.server import ServingError, run_server

    try:
        return run_server(_as_serve_config(config), print_command=args.print_command)
    except ServingError as exc:
        print(f"serving error: {exc}", file=sys.stderr)
        return 2


def _cmd_bench(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.serving.bench import BenchError, run_benchmarks
    from smolqwen.serving.server import ServingError

    try:
        return run_benchmarks(_as_serve_config(config), args=args)
    except (BenchError, ServingError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


def _cmd_sweep(args: argparse.Namespace, config: StrictModel) -> int:
    from smolqwen.serving.server import ServingError
    from smolqwen.serving.sweep import SweepError, run_sweep

    try:
        return run_sweep(_as_serve_config(config), args=args)
    except (SweepError, ServingError) as exc:
        print(f"sweep error: {exc}", file=sys.stderr)
        return 2


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
    "bench": _cmd_bench,
    "sweep": _cmd_sweep,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "probe":
        return _cmd_probe(args)

    try:
        config = _resolve_for(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(resolved_summary(config))
        return 0

    return DISPATCH[args.command](args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
