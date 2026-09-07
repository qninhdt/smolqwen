"""Evaluation orchestration shared by local and HTTP policies."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.console import logger, phase, progress_task, status_table
from smolqwen.eval.adapters import create_adapter
from smolqwen.eval.adapters.base import BenchmarkAdapter, EvalTask
from smolqwen.eval.batched import evaluate_batched, generation_for
from smolqwen.eval.checkpoints import resolve as resolve_checkpoint
from smolqwen.eval.manifest import EvalManifest
from smolqwen.eval.metrics import TaskMetrics
from smolqwen.eval.policies import Policy, load_http_policy
from smolqwen.eval.report import write_report
from smolqwen.eval.serving_pairing import load_quality_result
from smolqwen.eval.trajectories import TrajectoryRecord, write_trajectories
from smolqwen.tracking import Tracker, tracker_for

LOG = logger(__name__)


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
        # The render mode belongs beside decoding: a thinking-prompt run and a
        # non-thinking-prompt run ask the model different things, so the two are
        # not comparable even with identical sampling parameters.
        "enable_thinking": config.enable_thinking,
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
    tasks: Sequence[Any] | None = None,
    records: list[TrajectoryRecord] | None = None,
    label: str = "evaluation",
    progress: Callable[[EvalTask, TaskMetrics | None], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Advance every task to terminal through the text-native baseline path.

    `records` collects one `TrajectoryRecord` per task when supplied. The history
    was previously built and discarded, which left an all-or-nothing `0.0`
    unattributable and a re-grade impossible.

    The command uses `evaluate_batched` for local vLLM; this helper remains the
    text-policy path used for agreement checks and HTTP compatibility.

    Progress goes to stderr per task. This loop runs for hours and used to emit
    nothing until the final JSON line, so a stalled run and a slow one looked
    identical; `every=1` is right here because a task is a whole multi-turn episode,
    not a cheap unit.
    """
    task_list = list(tasks if tasks is not None else adapter.load_tasks())
    task_metrics: list[TaskMetrics] = []
    with progress_task(label, total=len(task_list), unit="tasks", every=1) as advance:
        for task in task_list:
            if progress is not None:
                progress(task, None)
            history: list[dict[str, Any]] = adapter.build_prompt(task, [])
            tools = task.tools
            generated_tokens = 0
            truncated = False
            generation_turns = 0
            env_steps = 0
            terminal_reason = "turn_cap"
            # Collected as they are appended rather than filtered out of `history`
            # afterwards: the opening user prompt is a `user` message too, and a
            # filter by role would silently count it as the episode's first
            # observation.
            observations: list[str] = []
            started = time.monotonic()
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
                    observations.extend(step.tool_observations)
                elif step.observation:
                    history.append({"role": step.observation_role, "content": step.observation})
                    observations.append(step.observation)
                if step.tools is not None:
                    tools = step.tools
                # A completion signal may have advanced the adapter to a new user
                # turn.  Adapter-owned prompt construction appends that turn once.
                history = adapter.build_prompt(task, history)
                if step.complete:
                    terminal_reason = "final_answer"
                    break
            wall_s = time.monotonic() - started
            score = adapter.score(task)
            invalid_calls = adapter.invalid_call_count(task)
            task_metrics.append(
                TaskMetrics(
                    category=task.category,
                    score=score.score,
                    invalid_calls=invalid_calls,
                    steps=env_steps,
                    generated_tokens=generated_tokens,
                    truncated=truncated,
                    exact_success=score.exact_success,
                    diagnostics=dict(score.diagnostics),
                    terminal_reason=terminal_reason,
                )
            )
            if records is not None:
                records.append(
                    TrajectoryRecord(
                        task_id=task.task_id,
                        category=task.category,
                        messages=[dict(message) for message in history],
                        observations=observations,
                        score=score.score,
                        exact_success=score.exact_success,
                        completed=score.completed,
                        failure_reason=score.failure_reason,
                        failed_check_names=list(score.failed_check_names),
                        diagnostics=dict(score.diagnostics),
                        terminal_reason=terminal_reason,
                        generation_turns=generation_turns,
                        env_steps=env_steps,
                        generated_tokens=generated_tokens,
                        truncated=truncated,
                        invalid_calls=invalid_calls,
                        wall_s=wall_s,
                    )
                )
            running = sum(metric.score for metric in task_metrics) / len(task_metrics)
            advance(f"{task.category} mean {running:.3f}")
            if progress is not None:
                progress(task, task_metrics[-1])
    return adapter.summarize(task_metrics)


def _evaluate_named_adapter(
    config: EvalConfig,
    policy: Policy | None,
    name: str,
    records: list[TrajectoryRecord] | None = None,
    backend: Any | None = None,
    tokenizer: Any | None = None,
) -> tuple[dict[str, dict[str, float]], Mapping[str, Any]]:
    """Score one benchmark through local vLLM or the HTTP compatibility path."""
    with phase(f"evaluate {name}: create adapter and load tasks"):
        adapter = create_adapter(name, config)
        tasks = adapter.load_tasks()
    try:
        with phase(f"evaluate {name}: compute benchmark manifest"):
            invariants = adapter.manifest_invariants(tasks)
        if backend is not None:
            metrics = evaluate_batched(
                config,
                adapter,
                backend=backend,
                tokenizer=tokenizer,
                tasks=tasks,
                records=records,
                label=name,
            )
        else:
            if policy is None:
                raise ValueError("evaluation needs either a generation backend or a policy")
            metrics = evaluate_adapter(
                config, policy, adapter, tasks=tasks, records=records, label=name
            )
        return metrics, invariants
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def _checkpoint_store(config: EvalConfig, args: Any) -> Any:
    """A store only when the checkpoint is a Hub id and a repo is configured.

    Constructed lazily so a local-directory evaluation never touches the Hub API,
    and never needs a token to run.
    """
    checkpoint = getattr(args, "checkpoint", None)
    if not checkpoint or Path(checkpoint).is_dir() or args.endpoint:
        return None
    repo_id = config.tracking.hub_repo_id or checkpoint
    from smolqwen.artifacts import CheckpointStore

    return CheckpointStore(repo_id, Path(config.tracking.local_artifact_dir) / "eval-checkpoints")


def run_evaluation(config: EvalConfig, args: Any) -> int:
    # Config validation before weight resolution: an empty adapter list is a typo,
    # and discovering it after a multi-gigabyte checkpoint download wastes the
    # download. Nothing here touches the network or the GPU.
    adapter_names = (args.adapter,) if args.adapter else tuple(config.adapters)
    if not adapter_names:
        raise ValueError("evaluation requires at least one benchmark adapter")

    with phase("evaluate: resolve pinned checkpoint"):
        resolved = resolve_checkpoint(
            checkpoint=args.checkpoint,
            revision=args.revision,
            adapter=args.adapter_path,
            adapter_revision=args.adapter_revision,
            endpoint=args.endpoint,
            store=_checkpoint_store(config, args),
        )
    generation = generation_for(config, resolved)
    policy = (
        load_http_policy(
            revision=resolved.revision,
            endpoint=args.endpoint,
            model=config.http_model,
            max_new_tokens=config.decoding.max_new_tokens,
            temperature=config.decoding.temperature,
            top_p=config.decoding.top_p,
            top_k=config.decoding.top_k,
            seed=config.decoding.seed,
            http_timeout_s=config.http_timeout_s,
            enable_thinking=config.enable_thinking,
        )
        if generation.path == "http"
        else None
    )
    inference_backend = generation.backend
    if inference_backend is not None:
        with phase("evaluate: load checkpoint tokenizer"):
            tokenizer = _tokenizer_for(resolved)
    else:
        tokenizer = None
    adapter_invariants: dict[str, Mapping[str, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    tag = args.tag or "evaluation"
    trajectory_paths: dict[str, str] = {}
    status_table(
        f"evaluation: {tag}",
        {
            "adapters": ", ".join(adapter_names),
            "checkpoint": resolved.path or args.endpoint or "(config default)",
            "revision": resolved.revision,
            "source": resolved.source,
            "generation": generation.path,
            "max steps per task": config.max_steps_per_task,
        },
    )
    # `evaluate` had no tracker at all, so the report it spends hours producing
    # existed only on a VM that gets reclaimed. Disabled without `WANDB_API_KEY`,
    # which is the same degradation every other stage already relies on.
    tracker = tracker_for(config.tracking, config=config.model_dump(mode="json"))
    tracker.start()
    try:
        for adapter_name in adapter_names:
            records: list[TrajectoryRecord] = []
            adapter_metrics, invariants = _evaluate_named_adapter(
                config,
                policy,
                adapter_name,
                records=records,
                backend=inference_backend,
                tokenizer=tokenizer,
            )
            duplicates = sorted(set(metrics) & set(adapter_metrics))
            if duplicates:
                raise ValueError(
                    f"evaluation adapters produced duplicate metric categories: {duplicates}"
                )
            metrics.update(adapter_metrics)
            adapter_invariants[adapter_name] = invariants
            trajectory_paths[adapter_name] = str(
                write_trajectories(
                    config.output_dir, tag=tag, adapter=adapter_name, records=records
                )
            )
            LOG.info(
                "%s: %d categories scored, %d trajectories written",
                adapter_name,
                len(adapter_metrics),
                len(records),
            )
        transport_backend = "http" if args.endpoint else generation.path
        backend = getattr(args, "serving_backend", None) or transport_backend
        manifest = build_manifest(
            config,
            revision=resolved.revision,
            backend=backend,
            adapter_invariants=adapter_invariants,
            recorded_free={
                **resolved.to_recorded(),
                "endpoint": args.endpoint,
                "served_model": config.http_model if args.endpoint else None,
                # The serving config generation actually ran under, read off the
                # engine. The nine flags that used to assert these are gone; with
                # nothing recording them the eight fields stayed None and
                # `--require-serving-match` could not match any real serving row.
                **_serving_config(generation),
                "library_versions": _library_versions(),
                # What generation actually used, so a row is self-describing without
                # the caller having asserted it on the command line.
                "generation_path": generation.path,
                "generation_concurrency": config.profile.generation_concurrency,
                "enforce_eager": config.profile.enforce_eager,
                "max_context_tokens_used": config.profile.max_seq_length,
                "trajectory_records": trajectory_paths,
            },
        )
        json_path, markdown_path = write_report(
            config.output_dir, tag=tag, manifest=manifest, metrics=metrics
        )
        reference = getattr(args, "require_serving_match", None)
        if reference is not None:
            _assert_serving_match(reference, manifest)
        _log_report(tracker, tag=tag, paths=(json_path, markdown_path), metrics=metrics)
        # The trajectory files ride along because the report's `trajectory_records`
        # field names them: uploading the report alone leaves those pointers dangling
        # at paths on a reclaimed VM.
        tracker.log_artifact(
            json_path,
            name=f"eval-{tag}",
            artifact_type="evaluation",
            extra_paths=[markdown_path, *trajectory_paths.values()],
        )
    finally:
        generation.shutdown()
        tracker.finish()
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True))
    return 0


def _serving_config(generation: Any) -> dict[str, Any]:
    """The eight serving fields a paired speed/quality row compares.

    From the in-process engine when there is one -- it is the only party that knows
    what it resolved. An endpoint's serving fields stay `None`: the server is a
    separate process this command cannot inspect. `assert_quality_matches_serving`
    compares only fields the measurement recorded, so a `None` makes no claim.
    """
    engine = getattr(generation, "engine", None)
    read = getattr(engine, "serving_config", None)
    if callable(read):
        try:
            return dict(read())
        except Exception as exc:  # noqa: BLE001 - provenance must not fail a scored run
            LOG.warning(
                "could not read the engine's serving config (%s: %s); "
                "the report records it as unknown",
                type(exc).__name__,
                exc,
            )
    return {"dtype": _recorded_dtype(generation)}


def _recorded_dtype(generation: Any) -> str | None:
    """The dtype generation actually ran at, from the engine when there is one.

    Previously hardcoded `"bfloat16"` whenever the backend was not HTTP, which is
    now wrong in a way that matters: a T4 run downgrades to fp16, and a report
    claiming bf16 would present two different numeric regimes as one experiment.
    """
    engine = getattr(generation, "engine", None)
    if engine is not None:
        return str(engine.profile.dtype)
    return None


def _tokenizer_for(resolved: Any) -> Any:
    """The tokenizer the turn engine renders and tokenizes with.

    The same checkpoint the engine loaded, at the same revision: the mask builder and
    the prefix renderer must agree with the weights, and a tokenizer from elsewhere
    would move a BPE seam under them.
    """
    from smolqwen.tokenizer import load_tokenizer

    return load_tokenizer(resolved.path, revision=resolved.revision)


def _log_report(
    tracker: Tracker,
    *,
    tag: str,
    paths: tuple[Path, Path],
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    """Send the headline scalars to the run, so a report has a chart beside it."""
    tracker.log(
        {
            f"eval/{tag}/{category}/{name}": float(value)
            for category, values in metrics.items()
            for name, value in values.items()
        }
    )
    LOG.info("wrote %s and %s", *paths)


def _assert_serving_match(reference: Any, manifest: EvalManifest) -> None:
    """Refuse a paired speed/quality row measured under a different serving config.

    Compares `recorded_free`, not `invariant`. Two runs at different `max_num_seqs`
    have identical invariants and are still different experiments, so the check
    `assert_comparable` performs cannot establish this one.
    """
    from smolqwen.eval.serving_pairing import assert_quality_matches_serving

    assert_quality_matches_serving(
        dict(manifest.recorded_free),
        load_quality_result(reference),
        label=str(reference),
    )
