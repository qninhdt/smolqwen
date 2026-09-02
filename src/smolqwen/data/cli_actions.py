"""Thin CLI dispatchers for the data pipeline.

`cli.py` imports these to dispatch `profile-data` and `prepare-sft`. They own the
"where do the release files live" question so the data modules stay agnostic, and
resolve the config's pinned datasets (revision + sha256) into local paths the
streamers can read -- a vendored copy when configured, otherwise a download from
the pinned Hub revision (served from cache when present).
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import get_ident, local
from typing import Any

from smolqwen.config_models import DataConfig, DatasetPin
from smolqwen.console import console, logger, progress_task, status_table
from smolqwen.data.convert_sft import (
    ConversionEvent,
    ConversionReport,
    Skipped,
    convert_trajectories,
    convert_trajectory,
    sample_to_record,
)
from smolqwen.data.loader import LoadStats, Trajectory, iter_trajectories, verify_sha256
from smolqwen.data.profiler import format_profile_table, profile_dataset, write_profile
from smolqwen.data.render import render_training_sample, training_chat_template
from smolqwen.data.splits import Split, build_env_split_manifest, split_trajectory_ids
from smolqwen.tokenizer import load_tokenizer

LOG = logger(__name__)


def _tokenizer(config: DataConfig) -> Any:
    """The text tokenizer whose chat template produces the rendered samples.

    Loaded lazily inside the subcommand handlers so config validation and
    `--dry-run` never touch transformers (see the `cli.py` docstring).
    """
    return load_tokenizer(config.model_id)


def _resolve_dataset(pin: DatasetPin) -> Path:
    """Resolve a pinned dataset to a local file, preferring the vendored copy.

    The env metadata and RL scenario files are vendored in `third_party/EnvScaler`
    and are pinned by sha256 in config; the 701 MB SFT trajectory file is not
    vendored and comes from the Hub at its pinned revision. The download uses the
    standard `HF_HOME` cache rather than a project-local one, so an already-cached
    revision is reused instead of pulling another copy per checkout.
    """
    if pin.local_path and Path(pin.local_path).is_file():
        verify_sha256(pin.local_path, pin.sha256)
        return Path(pin.local_path)

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=pin.repo_id,
            filename=pin.filename,
            revision=pin.revision,
            repo_type=pin.repo_type,
        )
    )
    verify_sha256(path, pin.sha256)
    return path


def run_profile_data(config: DataConfig) -> int:
    """`smolqwen profile-data`: profile trajectories, write budgets and env split."""
    output_dir = Path(config.output_dir)

    LOG.info("resolving pinned datasets")
    sft_path = _resolve_dataset(config.sft_trajectories)
    tokenizer = _tokenizer(config)
    LOG.info("rendering/tokenizing trajectories")
    with progress_task("profile-data") as advance:
        result = profile_dataset(
            tokenizer,
            sft_path,
            revision=config.sft_trajectories.revision,
            progress=advance,
        )
    profile_path, budgets_path = write_profile(result, output_dir)

    # The env-split manifest depends only on the static metadata, so it is written
    # in the same pass. The RL scenario env-ids come from the RL scenario file.
    LOG.info("building environment split")
    env_path = _resolve_dataset(config.env_metadata)
    rl_env_ids = _rl_scenario_env_ids(_resolve_dataset(config.rl_scenarios))
    manifest = build_env_split_manifest(
        env_path,
        rl_scenario_env_ids=rl_env_ids,
        input_sha256=config.env_metadata.sha256,
        input_revision=config.env_metadata.revision,
    )
    (output_dir / "env_split.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The table is a human summary, so it goes to stderr with the rest of the
    # progress. Every downstream reader takes the JSON files, not this text.
    console().print(format_profile_table(result))
    status_table(
        "profile-data",
        {
            "profile": profile_path,
            "budgets": budgets_path,
            "env split": output_dir / "env_split.json",
        },
    )
    return 0


def _rl_scenario_env_ids(rl_path: Path) -> list[str]:
    from smolqwen.data.loader import iter_json_array

    ids: list[str] = []
    for row in iter_json_array(rl_path):
        if isinstance(row, dict):
            ids.append(str(row.get("env_id") or ""))
    return ids


def run_prepare_sft(config: DataConfig, *, workers: int | None = None) -> int:
    """`smolqwen prepare-sft`: render trajectories into train/val SFT shards.

    Converts in two passes: the first collects task ids for the seeded train/val
    split, the second renders and routes each sample to its shard. Profiling is
    optional analysis, not a prerequisite for conversion.
    """
    output_dir = Path(config.output_dir)
    cap = config.max_seq_length

    LOG.info("resolving pinned dataset")
    sft_path = _resolve_dataset(config.sft_trajectories)
    tokenizer = _tokenizer(config)
    shape = config.tool_result_shape
    worker_count = _prepare_worker_count(workers)

    # Pass one: task groups for the seeded split. Paired row variants must stay together.
    LOG.info("pass 1/2: collecting split ids")
    ids: list[str] = []
    with progress_task("prepare-sft split") as advance:
        for trajectory in iter_trajectories(sft_path):
            ids.append(trajectory.task_id)
            advance()
    split = split_trajectory_ids(ids, seed=config.split_seed, val_fraction=config.val_fraction)
    LOG.info("pass 1/2 complete: %d trajectories", len(ids))

    # Pass two: render and route.
    report = ConversionReport()
    train_path = output_dir / "sft" / "train.jsonl"
    val_path = output_dir / "sft" / "val.jsonl"
    LOG.info("pass 2/2: rendering and writing shards with %d workers", worker_count)
    with progress_task("prepare-sft render/write", total=len(ids)) as advance:
        stats = _write_shards(
            sft_path,
            split,
            cap,
            tokenizer,
            shape,
            train_path,
            val_path,
            report,
            progress=advance,
            workers=worker_count,
        )

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(
            report.to_dict(
                input_shas=_input_shas(config, sft_path),
                input_revisions=_input_revisions(config),
                load_stats=stats,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status_table(
        "prepare-sft",
        {
            "train": train_path,
            "val": val_path,
            "report": report_path,
            "converted": report.converted,
            "skipped": report.skipped,
            "malformed": stats.malformed,
            "samples": report.samples,
        },
    )
    return 0


def _input_revisions(config: DataConfig) -> dict[str, str]:
    return {
        "sft_trajectories": config.sft_trajectories.revision,
        "rl_scenarios": config.rl_scenarios.revision,
        "env_metadata": config.env_metadata.revision,
    }


def _input_shas(config: DataConfig, sft_path: Path) -> dict[str, str]:
    """The sha256 of every input file, alongside its pinned revision.

    A count check does not detect a modified `env_class_code` body, so the hash is
    what lets Phases 4 and 7 assert they are executing the same dataset this
    conversion was built against.
    """
    from smolqwen.data.loader import sha256_of

    shas: dict[str, str] = {"sft_trajectories": sha256_of(sft_path)}
    for name, pin in (
        ("rl_scenarios", config.rl_scenarios),
        ("env_metadata", config.env_metadata),
    ):
        if pin.local_path and Path(pin.local_path).is_file():
            shas[name] = sha256_of(pin.local_path)
    return shas


def _write_shards(
    sft_path: Path,
    split: Split,
    cap: int,
    tokenizer: Any,
    shape: str,
    train_path: Path,
    val_path: Path,
    report: ConversionReport,
    *,
    progress: Callable[[], None] | None = None,
    workers: int = 1,
) -> LoadStats:
    """Render every trajectory and route its samples to one shard.

    Routing is by trajectory id, so a Conv trajectory's segments never straddle
    the train/val split. Returns the load stats so the report can account for
    malformed input rows as well as converted and skipped ones.
    """
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stats = LoadStats()
    if workers < 1:
        raise ValueError("workers must be at least 1")

    render = _build_renderer(tokenizer, workers=workers)

    with (
        train_path.open("w", encoding="utf-8") as train_handle,
        val_path.open("w", encoding="utf-8") as val_handle,
    ):
        trajectories = iter_trajectories(sft_path, stats=stats)
        events = (
            convert_trajectories(
                trajectories,
                render=render,
                max_seq_length=cap,
                shape=shape,
            )
            if workers == 1
            else _convert_parallel(
                trajectories,
                render=render,
                max_seq_length=cap,
                shape=shape,
                workers=workers,
            )
        )
        handles = {"train": train_handle, "val": val_handle}
        for event in events:
            if isinstance(event, Skipped):
                report.note_skipped(event)
            else:
                report.note_converted(event)
                partition = split.partition(event.task_id)
                handles[partition].write(json.dumps(sample_to_record(event.sample)) + "\n")
            if progress is not None:
                progress()

    return stats


def _build_renderer(tokenizer: Any, *, workers: int) -> Callable[..., Any]:
    """Build a renderer with one immutable compiled-template cache per worker."""
    canonical_template = training_chat_template(tokenizer)
    worker_state = local()

    def render(messages: Any, **kwargs: Any) -> Any:
        selected_template = canonical_template
        if workers > 1:
            # Transformers caches compiled Jinja templates globally by source.
            # A generation tracker is mutable and cannot be entered concurrently,
            # so a no-op worker comment gives each thread an isolated cache entry.
            worker_template: tuple[str, str] | None = getattr(
                worker_state, "training_template", None
            )
            if worker_template is None:
                source, fingerprint = canonical_template
                worker_template = (
                    f"{source}\n{{# smolqwen worker {get_ident()} #}}",
                    fingerprint,
                )
                worker_state.training_template = worker_template
            selected_template = worker_template
        return render_training_sample(
            tokenizer,
            messages,
            training_template=selected_template,
            **kwargs,
        )

    return render


def _prepare_worker_count(requested: int | None) -> int:
    """Choose a conservative CPU default while allowing an explicit CLI override."""
    workers = min(4, os.cpu_count() or 1) if requested is None else requested
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    return workers


def _convert_parallel(
    trajectories: Iterator[Trajectory],
    *,
    render: Callable[..., Any],
    max_seq_length: int,
    shape: str,
    workers: int,
) -> Iterator[ConversionEvent]:
    """Render concurrently with bounded memory and deterministic input order."""
    max_pending = workers * 2
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prepare-sft") as executor:
        pending: deque[Future[ConversionEvent]] = deque()

        def submit_next() -> bool:
            try:
                trajectory = next(trajectories)
            except StopIteration:
                return False
            pending.append(
                executor.submit(
                    convert_trajectory,
                    trajectory,
                    render=render,
                    max_seq_length=max_seq_length,
                    shape=shape,
                )
            )
            return True

        while len(pending) < max_pending and submit_next():
            pass
        while pending:
            yield pending.popleft().result()
            submit_next()
