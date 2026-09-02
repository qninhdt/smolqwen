"""Which weights an evaluation actually reads, resolved once and pinned.

Two pinning holes existed. `resolve_eval_checkpoint` (`artifacts.py:208`) refuses an
unpinned checkpoint read and was tested but never wired to the `evaluate` command,
so a Hub-hosted checkpoint reached `from_pretrained` with whatever `--revision`
happened to be, and a concurrent training push could change what a tag meant
between two runs. And adapter-revision pinning lived only in
`TransformersPolicy.__init__` (`policies.py:217-219`), so the vLLM adapter path had
none at all.

Both checks belong at the boundary where weights enter, which is here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from smolqwen.artifacts import CheckpointStore, resolve_eval_checkpoint

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


class CheckpointResolutionError(RuntimeError):
    """Raised when the weights an evaluation would read are not pinned."""


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Where the weights are, and how they got there. Recorded in the manifest.

    `source` is `local`, `hub`, or `endpoint`. A run whose numbers disagree with
    another run's is answerable from this field plus the two revisions -- without
    it, "the same tag" is an assumption.
    """

    path: str | None
    revision: str
    adapter_path: str | None = None
    adapter_revision: str | None = None
    source: str = "local"

    def to_recorded(self) -> dict[str, str | None]:
        return {
            "checkpoint": self.path,
            "checkpoint_revision": self.revision,
            "checkpoint_source": self.source,
            "adapter": self.adapter_path,
            "adapter_revision": self.adapter_revision,
        }


def require_sha(value: str | None, *, label: str) -> str:
    if not value or _COMMIT_SHA.fullmatch(value) is None:
        raise CheckpointResolutionError(
            f"evaluation requires an explicit {label} revision sha; got {value!r}"
        )
    return value


def resolve(
    *,
    checkpoint: str | None,
    revision: str | None,
    adapter: str | None = None,
    adapter_revision: str | None = None,
    endpoint: str | None = None,
    store: CheckpointStore | None = None,
) -> ResolvedCheckpoint:
    """Pin every weight source an evaluation will read, before it reads any.

    A local directory needs no download, but still needs its revision recorded --
    that is what makes two reports comparable. A Hub repo id goes through
    `resolve_eval_checkpoint`, which refuses to resolve a branch tip. An endpoint
    has no local weights at all, and the served revision is the operator's claim.
    """
    revision = require_sha(revision, label="checkpoint")
    if adapter:
        adapter_revision = require_sha(adapter_revision, label="adapter")

    if endpoint:
        return ResolvedCheckpoint(
            path=None,
            revision=revision,
            adapter_path=adapter,
            adapter_revision=adapter_revision,
            source="endpoint",
        )
    if not checkpoint:
        raise CheckpointResolutionError("--checkpoint is required unless --endpoint is supplied")

    if Path(checkpoint).is_dir():
        return ResolvedCheckpoint(
            path=checkpoint,
            revision=revision,
            adapter_path=adapter,
            adapter_revision=adapter_revision,
            source="local",
        )
    if store is None:
        # Not a directory and no store to pull from. Reporting this as a missing
        # path rather than letting `from_pretrained` attempt a network read keeps
        # the failure local and legible.
        raise CheckpointResolutionError(
            f"checkpoint {checkpoint!r} is not a local directory and no checkpoint "
            "store is configured to pull it from"
        )
    pulled = resolve_eval_checkpoint(store, revision)
    return ResolvedCheckpoint(
        path=str(pulled),
        revision=revision,
        adapter_path=adapter,
        adapter_revision=adapter_revision,
        source="hub",
    )
