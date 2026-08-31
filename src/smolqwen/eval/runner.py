"""Evaluation orchestration shared by local and HTTP policies."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from time import monotonic
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters import create_adapter
from smolqwen.eval.adapters.base import BenchmarkAdapter, EvalTask
from smolqwen.eval.manifest import EvalManifest
from smolqwen.eval.metrics import TaskMetrics
from smolqwen.eval.policies import Policy, load_policy
from smolqwen.eval.report import write_report

EvaluationProgress = Callable[[EvalTask, TaskMetrics | None], None]


def _library_versions() -> dict[str, str | None]:
    """Capture versions that can affect generation without requiring imports."""

    packages = ("torch", "transformers", "peft", "trl", "vllm")
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
    adapter_invariants: Mapping[str, Mapping[str, Any]] | None = None,
    recorded_free: Mapping[str, Any] | None = None,
) -> EvalManifest:
    invariant = {
        "decoding": config.decoding.model_dump(mode="json"),
        "max_context_tokens": config.profile.max_seq_length,
        "max_steps": config.max_steps_per_task,
        "seed": config.decoding.seed,
        # The checkpoint revision is intentionally recorded, not invariant:
        # Base, SFT, and SFT+RL must be comparable while using different pinned
        # weights. The benchmark and decoding fields above remain strict.
        "adapters": {
            name: dict(values) for name, values in sorted((adapter_invariants or {}).items())
        },
    }
    return EvalManifest(
        invariant=invariant,
        recorded_free={
            "backend": backend,
            "checkpoint_revision": revision,
            **(recorded_free or {}),
        },
    )


def evaluate_adapter(
    config: EvalConfig,
    policy: Policy,
    adapter: BenchmarkAdapter,
    tasks: Sequence[EvalTask] | None = None,
    progress: EvaluationProgress | None = None,
) -> dict[str, dict[str, float]]:
    task_metrics: list[TaskMetrics] = []
    for task in tasks if tasks is not None else adapter.load_tasks():
        if progress is not None:
            progress(task, None)
        history: list[dict[str, Any]] = adapter.build_prompt(task, [])
        tools = task.tools
        generated_tokens = 0
        truncated = False
        generation_turns = 0
        env_steps = 0
        while generation_turns < config.max_steps_per_task:
            result = policy.generate(history, tools)
            generated_tokens += result.generated_tokens
            truncated = truncated or result.truncated
            step = adapter.step(task, result.completion)
            generation_turns += 1
            env_steps += step.env_steps
            history.append({"role": "assistant", "content": result.completion})
            if step.tool_observations is not None:
                history.extend(
                    {"role": "tool", "content": observation}
                    for observation in step.tool_observations
                )
            elif step.observation:
                history.append({"role": step.observation_role, "content": step.observation})
            if step.tools is not None:
                tools = step.tools
            # A completion signal may have advanced the adapter to a new user
            # turn.  Adapter-owned prompt construction appends that turn once.
            history = adapter.build_prompt(task, history)
            if step.complete:
                break
        score = adapter.score(task)
        metrics = TaskMetrics(
            category=task.category,
            score=score.score,
            invalid_calls=adapter.invalid_call_count(task),
            steps=env_steps,
            generated_tokens=generated_tokens,
            truncated=truncated,
            exact_success=score.exact_success,
        )
        task_metrics.append(metrics)
        if progress is not None:
            progress(task, metrics)
    return adapter.summarize(task_metrics)


@contextmanager
def _evaluation_progress(adapter_name: str, *, total: int) -> Iterator[EvaluationProgress]:
    """Show live Rich progress plus durable notebook logs for one adapter."""
    rich_progress = Progress(
        TextColumn("{task.description}", markup=False),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )
    progress_id = rich_progress.add_task(f"evaluate {adapter_name}", total=total)
    started = monotonic()
    completed = 0
    log_every = max(1, (total + 19) // 20)
    succeeded = False

    def update(task: EvalTask, metrics: TaskMetrics | None) -> None:
        nonlocal completed
        label = f"evaluate {adapter_name} [{task.category}] {task.task_id}"
        if metrics is None:
            rich_progress.update(progress_id, description=label)
            return

        completed += 1
        rich_progress.update(progress_id, description=label, advance=1)
        if completed == 1 or completed % log_every == 0 or completed == total:
            elapsed = monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (total - completed) / rate if rate else 0.0
            print(
                f"evaluation {adapter_name}: {completed}/{total} tasks "
                f"({rate:.2f}/s, ETA {eta:.0f}s) — {task.task_id} "
                f"score={metrics.score:.4f} steps={metrics.steps} "
                f"tokens={metrics.generated_tokens}",
                flush=True,
            )

    with rich_progress:
        try:
            yield update
            succeeded = True
        finally:
            elapsed = monotonic() - started
            status = "complete" if succeeded else "failed"
            rich_progress.update(
                progress_id,
                description=f"evaluate {adapter_name} {status}",
            )
            print(
                f"evaluation {adapter_name} {status}: {completed}/{total} tasks in {elapsed:.1f}s",
                flush=True,
            )


def _evaluate_named_adapter(
    config: EvalConfig,
    policy: Policy,
    name: str,
) -> tuple[dict[str, dict[str, float]], Mapping[str, Any]]:
    print(f"evaluation: creating adapter {name}", flush=True)
    adapter = create_adapter(name, config)
    try:
        print(f"evaluation {name}: loading tasks", flush=True)
        tasks = adapter.load_tasks()
        print(f"evaluation {name}: loaded {len(tasks)} tasks", flush=True)
        print(f"evaluation {name}: computing manifest invariants", flush=True)
        invariants = adapter.manifest_invariants(tasks)
        with _evaluation_progress(name, total=len(tasks)) as progress:
            metrics = evaluate_adapter(
                config,
                policy,
                adapter,
                tasks=tasks,
                progress=progress,
            )
        print(
            f"evaluation {name}: summarized {len(metrics)} metric categories",
            flush=True,
        )
        return metrics, invariants
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            print(f"evaluation {name}: closing adapter", flush=True)
            close()


def run_evaluation(config: EvalConfig, args: Any) -> int:
    transport_backend = "http" if args.endpoint else "transformers"
    locator = args.endpoint or args.checkpoint or config.http_model
    print(
        f"evaluation: loading {transport_backend} policy from {locator}",
        flush=True,
    )
    policy = load_policy(
        checkpoint=args.checkpoint,
        revision=args.revision,
        endpoint=args.endpoint,
        adapter=args.adapter_path,
        adapter_revision=args.adapter_revision,
        model=config.http_model,
        max_new_tokens=config.decoding.max_new_tokens,
        temperature=config.decoding.temperature,
        top_p=config.decoding.top_p,
        top_k=config.decoding.top_k,
        seed=config.decoding.seed,
        http_timeout_s=config.http_timeout_s,
    )
    print(f"evaluation: policy ready at revision {policy.revision}", flush=True)
    adapter_invariants: dict[str, Mapping[str, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    adapter_names = (args.adapter,) if args.adapter else tuple(config.adapters)
    if not adapter_names:
        raise ValueError("evaluation requires at least one benchmark adapter")
    for adapter_name in adapter_names:
        adapter_metrics, invariants = _evaluate_named_adapter(config, policy, adapter_name)
        duplicates = sorted(set(metrics) & set(adapter_metrics))
        if duplicates:
            raise ValueError(
                f"evaluation adapters produced duplicate metric categories: {duplicates}"
            )
        metrics.update(adapter_metrics)
        adapter_invariants[adapter_name] = invariants
    backend = getattr(args, "serving_backend", None) or transport_backend
    manifest = build_manifest(
        config,
        revision=policy.revision,
        backend=backend,
        adapter_invariants=adapter_invariants,
        recorded_free={
            "checkpoint": args.checkpoint,
            "endpoint": args.endpoint,
            "served_model": config.http_model if args.endpoint else None,
            "dtype": getattr(args, "served_dtype", None)
            or ("bfloat16" if transport_backend == "transformers" else None),
            "quantization": getattr(args, "quantization", None),
            "speculative_decoding": getattr(args, "speculative_decoding", None),
            "kv_budget": getattr(args, "kv_budget", None),
            "max_num_seqs": getattr(args, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
            "chunked_prefill": getattr(args, "chunked_prefill", None),
            "prefix_caching": getattr(args, "prefix_caching", None),
            "library_versions": _library_versions(),
            "adapter_revision": getattr(policy, "adapter_revision", None),
        },
    )
    print(f"evaluation: writing report to {config.output_dir}", flush=True)
    json_path, markdown_path = write_report(
        config.output_dir, tag=args.tag or "evaluation", manifest=manifest, metrics=metrics
    )
    print(f"evaluation: report complete with {len(metrics)} metric categories", flush=True)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True))
    return 0
