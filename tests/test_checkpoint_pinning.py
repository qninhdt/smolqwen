"""Checkpoint reads must be pinned by sha; `latest_revision` is resume-only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from smolqwen.artifacts import (
    CheckpointStore,
    CheckpointStoreError,
    ResumeState,
    resolve_eval_checkpoint,
)


@dataclass
class _Branch:
    name: str
    target_commit: str


@dataclass
class _Refs:
    branches: list[_Branch]


@dataclass
class FakeHub:
    """Records calls instead of touching the network."""

    latest: str = "cafe1234"
    created: list[str] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)

    def create_repo(self, repo_id: str, *, private: bool, exist_ok: bool, repo_type: str) -> None:
        self.created.append(repo_id)

    def upload_folder(
        self, *, repo_id: str, folder_path: str, commit_message: str, repo_type: str
    ) -> None:
        self.uploads.append({"repo_id": repo_id, "folder": folder_path, "message": commit_message})

    def list_repo_refs(self, repo_id: str, *, repo_type: str) -> _Refs:
        return _Refs(branches=[_Branch(name="main", target_commit=self.latest)])

    def snapshot_download(
        self, *, repo_id: str, revision: str, local_dir: str, repo_type: str
    ) -> str:
        self.downloads.append({"repo_id": repo_id, "revision": revision})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "adapter_model.safetensors").write_bytes(b"weights")
        return local_dir


@pytest.fixture()
def adapter_dir(tmp_path: Path) -> Path:
    source = tmp_path / "run" / "checkpoint-100"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    return source


def test_save_push_and_pull_round_trip(tmp_path: Path, adapter_dir: Path) -> None:
    hub = FakeHub()
    store = CheckpointStore("org/smolqwen-sft", tmp_path / "cache", client=hub)

    saved = store.save_adapter(adapter_dir)
    assert (saved / "adapter_model.safetensors").is_file()

    store.push(commit_message="step 100")
    assert hub.created == ["org/smolqwen-sft"]
    assert hub.uploads[0]["message"] == "step 100"

    pulled = store.pull("cafe1234", local_dir=tmp_path / "restored")
    assert (pulled / "adapter_model.safetensors").is_file()
    assert hub.downloads == [{"repo_id": "org/smolqwen-sft", "revision": "cafe1234"}]


def test_pull_requires_an_explicit_revision(tmp_path: Path) -> None:
    store = CheckpointStore("org/smolqwen-sft", tmp_path / "cache", client=FakeHub())
    for bad in ("", None):
        with pytest.raises(CheckpointStoreError, match="explicit revision sha"):
            store.pull(bad)  # type: ignore[arg-type]


def test_eval_path_cannot_resolve_latest_revision(tmp_path: Path) -> None:
    hub = FakeHub()
    store = CheckpointStore("org/smolqwen-sft", tmp_path / "cache", client=hub)

    # latest_revision() resolves at call time, so a training run still pushing
    # adapters can change what "SFT" means between two eval invocations.
    with pytest.raises(CheckpointStoreError, match="must not be reachable from an eval path"):
        resolve_eval_checkpoint(store, None)
    assert hub.downloads == []

    resolve_eval_checkpoint(store, "deadbeef")
    assert hub.downloads[-1]["revision"] == "deadbeef"


def test_latest_revision_is_available_for_resume(tmp_path: Path) -> None:
    store = CheckpointStore("org/smolqwen-sft", tmp_path / "cache", client=FakeHub(latest="abc999"))
    assert store.latest_revision() == "abc999"


def test_local_only_store_is_a_noop_on_push(tmp_path: Path, adapter_dir: Path) -> None:
    # A run without a configured repo must still execute the same code path.
    store = CheckpointStore(None, tmp_path / "cache", client=FakeHub())
    assert store.enabled is False
    store.save_adapter(adapter_dir)
    store.push(commit_message="ignored")
    assert store.latest_revision() is None


def test_resume_state_carries_more_than_weights(tmp_path: Path) -> None:
    store = CheckpointStore(None, tmp_path / "cache")
    # Restoring only the adapter makes a GRPO run replay the curriculum from the
    # top, over-weighting whatever sorts first with nothing in the loss curve
    # showing it.
    state = ResumeState(
        revision="abc123", wandb_run_id="run-7", global_step=42, sampler_cursor=1337
    )
    store.write_resume_state(state)
    restored = store.read_resume_state()
    assert restored == state
    assert restored is not None and restored.sampler_cursor == 1337


def test_read_resume_state_absent_is_none(tmp_path: Path) -> None:
    assert CheckpointStore(None, tmp_path / "cache").read_resume_state() is None


def test_save_adapter_rejects_a_missing_source(tmp_path: Path) -> None:
    store = CheckpointStore(None, tmp_path / "cache")
    with pytest.raises(CheckpointStoreError, match="adapter directory not found"):
        store.save_adapter(tmp_path / "nope")
