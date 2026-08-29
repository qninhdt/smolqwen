"""Checkpoint persistence. The Hub is durable; local `artifacts/` is a cache.

Colab VMs are reclaimed without warning, so an adapter that only exists locally
does not exist. Adapters are tens to low hundreds of MB, which makes a push on
every `save_steps` affordable and makes VM loss a restart rather than a loss.

The read path is deliberately asymmetric to the write path: `pull` requires an
explicit revision sha and `latest_revision` is reachable only from `--resume`.
`latest_revision` resolves at call time, so a training run still pushing adapters
can change what "SFT" means between two eval invocations -- the comparison would
drift with nothing in the output showing it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

RESUME_MARKER = "resume_state.json"


class CheckpointStoreError(Exception):
    """Raised when a checkpoint cannot be saved, pushed, or resolved."""


class HubClient(Protocol):
    """The slice of `huggingface_hub` this module uses. Injectable, so tests mock it."""

    def create_repo(
        self, repo_id: str, *, private: bool, exist_ok: bool, repo_type: str
    ) -> Any: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str,
        commit_message: str,
        repo_type: str,
    ) -> Any: ...

    def list_repo_refs(self, repo_id: str, *, repo_type: str) -> Any: ...

    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        repo_type: str,
    ) -> str: ...


@dataclass(frozen=True)
class ResumeState:
    """Everything a run needs to continue that is not the weights.

    Restoring only the adapter makes a GRPO run replay the curriculum from the
    top, re-training on scenarios it already saw and over-weighting whatever
    sorts first. Nothing in the loss curve shows it, so the cursor is part of the
    payload rather than something to reconstruct.
    """

    revision: str | None = None
    wandb_run_id: str | None = None
    global_step: int = 0
    sampler_cursor: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "wandb_run_id": self.wandb_run_id,
            "global_step": self.global_step,
            "sampler_cursor": self.sampler_cursor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResumeState:
        return cls(
            revision=payload.get("revision"),
            wandb_run_id=payload.get("wandb_run_id"),
            global_step=int(payload.get("global_step", 0)),
            sampler_cursor=int(payload.get("sampler_cursor", 0)),
        )


class CheckpointStore:
    """Save/push/resolve/pull for one repo, so no training code touches the Hub API."""

    def __init__(
        self,
        repo_id: str | None,
        local_dir: Path | str,
        *,
        client: HubClient | None = None,
        private: bool = True,
        repo_type: str = "model",
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.private = private
        self.repo_type = repo_type
        self._client = client
        self._repo_created = False

    @property
    def enabled(self) -> bool:
        """False when no repo is configured: local-only runs must still work."""
        return self.repo_id is not None

    def _hub(self) -> HubClient:
        if self._client is None:
            import huggingface_hub  # imported lazily so CPU-only CI needs no network

            self._client = huggingface_hub.HfApi()
        assert self._client is not None
        return self._client

    def save_adapter(self, source_dir: Path | str, *, subdir: str | None = None) -> Path:
        """Copy an adapter directory into the local cache, returning its path."""
        source = Path(source_dir)
        if not source.is_dir():
            raise CheckpointStoreError(f"adapter directory not found: {source}")
        target = self.local_dir / subdir if subdir else self.local_dir
        if source.resolve() == target.resolve():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target

    def push(self, *, commit_message: str, folder: Path | str | None = None) -> None:
        """Upload the local cache. A no-op when no repo is configured."""
        if not self.enabled:
            return
        assert self.repo_id is not None
        folder_path = Path(folder) if folder is not None else self.local_dir
        if not folder_path.is_dir():
            raise CheckpointStoreError(f"nothing to push, not a directory: {folder_path}")
        hub = self._hub()
        if not self._repo_created:
            hub.create_repo(
                self.repo_id, private=self.private, exist_ok=True, repo_type=self.repo_type
            )
            self._repo_created = True
        hub.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(folder_path),
            commit_message=commit_message,
            repo_type=self.repo_type,
        )

    def latest_revision(self) -> str | None:
        """The newest revision sha on the main branch.

        Resume-only by contract. `pull` refuses to be called without an explicit
        revision, and `assert_not_eval_path` is what keeps this out of an eval.
        """
        if not self.enabled:
            return None
        assert self.repo_id is not None
        refs = self._hub().list_repo_refs(self.repo_id, repo_type=self.repo_type)
        branches = getattr(refs, "branches", None) or []
        for branch in branches:
            if getattr(branch, "name", None) == "main":
                sha = getattr(branch, "target_commit", None)
                return str(sha) if sha else None
        return None

    def pull(self, revision: str, *, local_dir: Path | str | None = None) -> Path:
        """Download one pinned revision. The sha is mandatory, not defaulted."""
        if not self.enabled:
            raise CheckpointStoreError("cannot pull without a configured repo_id")
        if not revision or not isinstance(revision, str):
            raise CheckpointStoreError(
                "pull requires an explicit revision sha; latest_revision() is for --resume only"
            )
        assert self.repo_id is not None
        target = Path(local_dir) if local_dir is not None else self.local_dir
        target.mkdir(parents=True, exist_ok=True)
        path = self._hub().snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            local_dir=str(target),
            repo_type=self.repo_type,
        )
        return Path(path)

    def write_resume_state(self, state: ResumeState) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        path = self.local_dir / RESUME_MARKER
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
        return path

    def read_resume_state(self) -> ResumeState | None:
        path = self.local_dir / RESUME_MARKER
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as handle:
            return ResumeState.from_dict(json.load(handle))


def resolve_eval_checkpoint(store: CheckpointStore, revision: str | None) -> Path:
    """Materialise a checkpoint for evaluation, refusing an unpinned read.

    An eval that resolves `latest_revision` can have the model swapped under it by
    a concurrent training push, making two runs of the same tag disagree with
    nothing in the manifest to show why.
    """
    if revision is None:
        raise CheckpointStoreError(
            "evaluation requires an explicit checkpoint revision sha; "
            "latest_revision() must not be reachable from an eval path"
        )
    return store.pull(revision)
