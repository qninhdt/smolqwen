"""Run pinned BFCL multi-turn evaluation with an in-process vLLM engine."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.console import logger, phase, status_table
from smolqwen.eval.bfcl_runner import (
    TRAJECTORY_NAME,
    BfclCompletion,
    BfclRequest,
    evaluate_bfcl,
    load_bfcl_tasks,
)
from smolqwen.eval.checkpoints import resolve as resolve_checkpoint
from smolqwen.eval.manifest import EvalManifest
from smolqwen.eval.report import write_report
from smolqwen.eval.trajectories import append_trajectories
from smolqwen.inference.decoding import decode_completion
from smolqwen.inference.profiles import EvalProfile
from smolqwen.tracking import Tracker, tracker_for

LOG = logger(__name__)
ADAPTER_SLOT = "eval-adapter"


def _library_versions() -> dict[str, str | None]:
    """Capture packages that can change BFCL generation or scoring."""

    packages = ("torch", "transformers", "peft", "vllm", "bfcl")
    resolved: dict[str, str | None] = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def build_manifest(
    config: EvalConfig,
    *,
    revision: str,
    backend: str,
    benchmark_invariants: Mapping[str, Any] | None = None,
    adapter_invariants: Mapping[str, Mapping[str, Any]] | None = None,
    recorded_free: Mapping[str, Any] | None = None,
) -> EvalManifest:
    benchmark = benchmark_invariants
    if benchmark is None and adapter_invariants is not None:
        benchmark = {name: dict(values) for name, values in adapter_invariants.items()}
    return EvalManifest(
        invariant={
            "benchmark": dict(benchmark or {}),
            "decoding": config.decoding.model_dump(mode="json"),
            "enable_thinking": config.enable_thinking,
            "max_context_tokens": config.profile.max_seq_length,
            "max_steps_per_turn": config.max_steps_per_task,
        },
        recorded_free={
            "backend": backend,
            "checkpoint_revision": revision,
            **(recorded_free or {}),
        },
    )


def run_evaluation(config: EvalConfig, args: Any) -> int:
    """Evaluate one checkpoint on BFCL multi-turn base and write its artifacts."""

    if getattr(args, "endpoint", None):
        raise ValueError("evaluate accepts checkpoints only; use in-process vLLM")
    with phase("evaluate: resolve pinned checkpoint"):
        resolved = resolve_checkpoint(
            checkpoint=args.checkpoint,
            revision=args.revision,
            adapter=args.adapter_path,
            adapter_revision=args.adapter_revision,
            endpoint=None,
            store=_checkpoint_store(config, args),
        )
    with phase("evaluate: load BFCL multi_turn_base"):
        tasks, benchmark_revision = load_bfcl_tasks(_expected_bfcl_revision(config))
    with phase(f"evaluate: build vLLM engine ({resolved.path})"):
        engine, adapter_name = _engine_for(config, resolved)
    with phase("evaluate: load checkpoint tokenizer"):
        tokenizer = _tokenizer_for(resolved)

    tag = args.tag or "evaluation"
    generation_path = "vllm+lora" if adapter_name else "vllm"
    status_table(
        f"evaluation: {tag}",
        {
            "benchmark": "BFCL multi_turn_base",
            "checkpoint": resolved.path,
            "revision": resolved.revision,
            "generation": generation_path,
            "batch size": config.profile.generation_concurrency,
            "max steps per turn": config.max_steps_per_task,
        },
    )

    tracker = tracker_for(config.tracking, config=config.model_dump(mode="json"))
    tracker.start()
    try:
        with append_trajectories(config.output_dir, tag=tag, adapter=TRAJECTORY_NAME) as (
            trajectory_path,
            append_record,
        ):
            run = evaluate_bfcl(
                config,
                tokenizer=tokenizer,
                generate=_vllm_generator(engine, tokenizer, config, adapter_name),
                tasks=tasks,
                benchmark_revision=benchmark_revision,
                record_sink=append_record,
            )
        manifest = build_manifest(
            config,
            revision=resolved.revision,
            backend=generation_path,
            benchmark_invariants=run.invariants,
            recorded_free={
                **resolved.to_recorded(),
                **_serving_config(engine),
                "library_versions": _library_versions(),
                "generation_path": generation_path,
                "generation_concurrency": config.profile.generation_concurrency,
                "enforce_eager": config.profile.enforce_eager,
                "max_context_tokens_used": config.profile.max_seq_length,
                "trajectory_records": {TRAJECTORY_NAME: str(trajectory_path)},
            },
        )
        json_path, markdown_path = write_report(
            config.output_dir, tag=tag, manifest=manifest, metrics=run.metrics
        )
        _log_report(tracker, tag=tag, paths=(json_path, markdown_path), metrics=run.metrics)
        tracker.log_artifact(
            json_path,
            name=f"eval-{tag}",
            artifact_type="evaluation",
            extra_paths=[markdown_path, trajectory_path],
        )
    finally:
        engine.shutdown()
        tracker.finish()
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True))
    return 0


def _engine_for(config: EvalConfig, resolved: Any) -> tuple[Any, str | None]:
    from smolqwen.inference.engine import offline_engine_for_eval

    adapters = {ADAPTER_SLOT: resolved.adapter_path} if resolved.adapter_path else None
    engine = offline_engine_for_eval(
        resolved.path,
        EvalProfile.from_config(config),
        revision=resolved.revision,
        adapter=adapters,
    )
    return engine, ADAPTER_SLOT if adapters else None


def _vllm_generator(
    engine: Any, tokenizer: Any, config: EvalConfig, adapter_name: str | None
) -> Any:
    def generate(requests: Sequence[BfclRequest]) -> list[BfclCompletion]:
        if not requests:
            return []
        generated = engine.generate_ids(
            [request.prompt_ids for request in requests],
            max_new_tokens=max(request.max_new_tokens for request in requests),
            adapter=adapter_name,
            temperature=config.decoding.temperature,
            top_p=config.decoding.top_p,
            top_k=config.decoding.top_k,
            presence_penalty=config.decoding.presence_penalty,
        )
        completions: list[BfclCompletion] = []
        for request, result in zip(requests, generated, strict=True):
            token_ids = tuple(result.token_ids[: request.max_new_tokens])
            truncated = (
                len(result.token_ids) > request.max_new_tokens or result.finish_reason == "length"
            )
            completions.append(
                BfclCompletion(
                    task_id=request.task_id,
                    text=decode_completion(tokenizer, token_ids),
                    generated_tokens=len(token_ids),
                    finish_reason="length" if truncated else result.finish_reason,
                )
            )
        return completions

    return generate


def _expected_bfcl_revision(config: EvalConfig) -> str | None:
    value = config.adapter_options.get("bfcl_multi_turn", {}).get("benchmark_commit")
    return str(value) if value else None


def _checkpoint_store(config: EvalConfig, args: Any) -> Any:
    checkpoint = getattr(args, "checkpoint", None)
    if not checkpoint or Path(checkpoint).is_dir():
        return None
    repo_id = config.tracking.hub_repo_id or checkpoint
    from smolqwen.artifacts import CheckpointStore

    return CheckpointStore(repo_id, Path(config.tracking.local_artifact_dir) / "eval-checkpoints")


def _tokenizer_for(resolved: Any) -> Any:
    from smolqwen.tokenizer import load_tokenizer

    return load_tokenizer(resolved.path, revision=resolved.revision)


def _serving_config(engine: Any) -> dict[str, Any]:
    read = getattr(engine, "serving_config", None)
    if callable(read):
        try:
            return dict(read())
        except Exception as exc:  # noqa: BLE001 - provenance must not fail a scored run
            LOG.warning("could not read vLLM serving config (%s: %s)", type(exc).__name__, exc)
    return {"dtype": str(engine.profile.dtype)}


def _log_report(
    tracker: Tracker,
    *,
    tag: str,
    paths: tuple[Path, Path],
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    tracker.log(
        {
            f"eval/{tag}/{category}/{name}": float(value)
            for category, values in metrics.items()
            for name, value in values.items()
        }
    )
    LOG.info("wrote %s and %s", *paths)
