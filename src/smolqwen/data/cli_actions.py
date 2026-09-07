"""Thin CLI dispatcher for the SFT data pipeline.

`cli.py` imports this module to dispatch `prepare-sft`. It owns the
"where do the release files live" question so the data modules stay agnostic, and
resolves the pinned release (served from cache when present).
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
from smolqwen.console import logger, phase, progress_task, status_table
from smolqwen.data.convert_sft import (
    SFT_SEMANTICS,
    SFT_SEMANTICS_NON_REASONING,
    ConversionEvent,
    ConversionReport,
    Skipped,
    convert_trajectories,
    convert_trajectory,
    sample_to_record,
)
from smolqwen.data.loader import LoadStats, Trajectory, iter_trajectories, verify_sha256
from smolqwen.data.render import render_training_sample, training_chat_template
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

    The 701 MB SFT trajectory file is not vendored and comes from the Hub at its
    pinned revision. The download uses the standard `HF_HOME` cache rather than a
    project-local one, so an already-cached revision is reused.
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
            repo_type="dataset",
        )
    )
    verify_sha256(path, pin.sha256)
    return path


def run_prepare_sft(config: DataConfig, *, workers: int | None = None) -> int:
    """`smolqwen prepare-sft`: render every accepted trajectory into one shard."""
    output_dir = Path(config.output_dir)
    cap = config.max_seq_length

    with phase("prepare-sft: resolve pinned dataset"):
        sft_path = _resolve_dataset(config.sft_trajectories)
    with phase("prepare-sft: load tokenizer"):
        tokenizer = _tokenizer(config)
    shape = config.tool_result_shape
    reasoning = config.enable_thinking
    worker_count = _prepare_worker_count(workers)

    report = ConversionReport()
    train_path = output_dir / "sft" / "train.jsonl"
    LOG.info("rendering and writing with %d workers", worker_count)
    with phase("prepare-sft: render and write train shard"):
        with progress_task("prepare-sft render/write") as advance:
            stats = _write_shards(
                sft_path,
                cap,
                tokenizer,
                shape,
                train_path,
                report,
                reasoning=reasoning,
                progress=advance,
                workers=worker_count,
            )

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(
            report.to_dict(
                input_shas=_input_shas(sft_path),
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
            "report": report_path,
            "converted": report.converted,
            "skipped": report.skipped,
            "malformed": stats.malformed,
            "samples": report.samples,
        },
    )
    return 0


def _input_revisions(config: DataConfig) -> dict[str, str]:
    return {"sft_trajectories": config.sft_trajectories.revision}


def _input_shas(sft_path: Path) -> dict[str, str]:
    """The sha256 of every input file, alongside its pinned revision.

    The hash ties the rendered shard to the exact downloaded release file.
    """
    from smolqwen.data.loader import sha256_of

    return {"sft_trajectories": sha256_of(sft_path)}


def _write_shards(
    sft_path: Path,
    cap: int,
    tokenizer: Any,
    shape: str,
    train_path: Path,
    report: ConversionReport,
    *,
    reasoning: bool = True,
    progress: Callable[[], None] | None = None,
    workers: int = 1,
) -> LoadStats:
    """Render every trajectory into one train shard.

    Returns load stats so the report accounts for malformed input rows as well as
    converted and skipped ones.

    `reasoning=False` strips teacher reasoning at render time and tags the
    written records with the non-reasoning semantics.
    """
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stats = LoadStats()
    if workers < 1:
        raise ValueError("workers must be at least 1")

    semantics = SFT_SEMANTICS if reasoning else SFT_SEMANTICS_NON_REASONING
    render = _build_renderer(tokenizer, workers=workers, reasoning=reasoning)

    with train_path.open("w", encoding="utf-8") as train_handle:
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
        for event in events:
            if isinstance(event, Skipped):
                report.note_skipped(event)
            else:
                report.note_converted(event)
                record = sample_to_record(event.sample, semantics=semantics)
                train_handle.write(json.dumps(record) + "\n")
            if progress is not None:
                progress()

    return stats


def _build_renderer(tokenizer: Any, *, workers: int, reasoning: bool = True) -> Callable[..., Any]:
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
            reasoning=reasoning,
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
