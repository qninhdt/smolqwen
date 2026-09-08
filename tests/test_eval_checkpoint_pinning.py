"""Which weights an evaluation reads, and the two places pinning was missing.

`resolve_eval_checkpoint` refuses an unpinned Hub read and had a test, but nothing
called it from `evaluate` — so a Hub-hosted checkpoint reached `from_pretrained`
with whatever `--revision` held, and a concurrent training push could change what a
tag meant between two runs of it. Adapter pinning had the same gap from the other
direction: the vLLM adapter path used to load a mutable reference without enforcing
a commit SHA first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from smolqwen.eval.checkpoints import CheckpointResolutionError, resolve

SHA = "a" * 40
OTHER_SHA = "b" * 40


class Store:
    """A `CheckpointStore` stand-in that records what it was asked to pull."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.pulled: list[str] = []

    def pull(self, revision: str, *, local_dir: object = None) -> Path:
        self.pulled.append(revision)
        return self.target


def test_a_local_directory_needs_no_download_or_revision(tmp_path: Path) -> None:
    resolved = resolve(checkpoint=str(tmp_path), revision=None)
    assert resolved.source == "local"
    assert resolved.path == str(tmp_path)
    assert resolved.revision is None
    assert resolved.to_recorded()["checkpoint_revision"] is None

    pinned = resolve(checkpoint=str(tmp_path), revision=SHA)
    assert pinned.revision == SHA


def test_a_hub_checkpoint_goes_through_the_pinned_pull(tmp_path: Path) -> None:
    store = Store(tmp_path / "pulled")
    resolved = resolve(checkpoint="org/smolqwen-sft", revision=SHA, store=cast(Any, store))
    assert store.pulled == [SHA]
    assert resolved.source == "hub"
    assert resolved.path == str(tmp_path / "pulled")


def test_an_unpinned_revision_is_refused_before_any_weights_load() -> None:
    """The failure this exists to prevent: two runs of one tag reading different
    weights, with nothing in the manifest to show why."""
    for revision in (None, "", "main", "v1.0", "a" * 39):
        with pytest.raises(CheckpointResolutionError, match="checkpoint revision sha"):
            resolve(checkpoint="org/smolqwen-sft", revision=revision)


def test_an_adapter_without_its_own_revision_is_refused(tmp_path: Path) -> None:
    """The vLLM adapter path must pin the adapter before loading it."""
    local_adapter = tmp_path / "adapter"
    local_adapter.mkdir()

    resolved_local = resolve(
        checkpoint=str(tmp_path), revision=None, adapter=str(local_adapter)
    )
    assert resolved_local.adapter_revision is None

    with pytest.raises(CheckpointResolutionError, match="adapter revision sha"):
        resolve(checkpoint=str(tmp_path), revision=None, adapter="org/smolqwen-adapter")

    resolved = resolve(
        checkpoint=str(tmp_path),
        revision=None,
        adapter="org/smolqwen-adapter",
        adapter_revision=OTHER_SHA,
    )
    assert resolved.adapter_revision == OTHER_SHA
    assert resolved.to_recorded()["adapter_revision"] == OTHER_SHA


def test_an_endpoint_has_no_local_weights_but_still_pins_the_served_revision() -> None:
    resolved = resolve(checkpoint=None, revision=SHA, endpoint="http://127.0.0.1:8000/v1")
    assert resolved.source == "endpoint"
    assert resolved.path is None
    assert resolved.revision == SHA


def test_a_missing_local_path_with_no_store_fails_locally_rather_than_over_the_network() -> None:
    """Letting `from_pretrained` attempt the read turns a typo into a 404 from the
    Hub, several seconds and one confusing traceback later."""
    with pytest.raises(CheckpointResolutionError, match="not a local directory"):
        resolve(checkpoint="artifacts/models/typo", revision=SHA)


def test_a_checkpoint_is_required_unless_an_endpoint_is_given() -> None:
    with pytest.raises(CheckpointResolutionError, match="--checkpoint is required"):
        resolve(checkpoint=None, revision=SHA)
