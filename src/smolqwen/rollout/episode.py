"""The per-episode record every other rollout module writes into.

One `Episode` is the contract between the scheduler (state and terminal reason),
generation (token spans and logprobs), mask building (spans and drift tally), and
the Phase 7 readers (trajectory table, reward). Its shape is tabulated in the
phase plan and must not grow ad-hoc fields: a later phase reading a field that
another writer never sets fails silently in training, which is the failure mode
this record exists to make loud.

Two invariants worth restating because nothing downstream enforces them:

- `logprobs`, `completion_ids`, and the assembled `env_mask` are equal length,
  with NaN at every appended-observation position. A shorter `logprobs` array
  would be right-padded with 0.0 by TRL — probability 1 — and shift every model
  token after the first observation against the wrong position.
- `terminal_reason` decides exclusion, not the reward: `worker_crash` episodes
  are replaced rather than scored; `timeout` is the model's own doing under a
  stated budget and is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from smolqwen.data.loader import Message

EpisodeState = Literal["ready", "generating", "tool", "done"]

TerminalReason = Literal[
    "final_answer",  # the model stopped calling tools
    "step_cap",  # `max_env_steps` reached
    "unrecoverable",  # the environment cannot continue (create/score failed)
    "timeout",  # episode wall-clock budget spent
    "worker_crash",  # infrastructure failure; the episode is replaced, not scored
]

DriftKind = Literal["clean", "realign", "fork"]


@dataclass
class DriftTally:
    """Turn-transition drift accounting, ported from upstream's `_SampleBuilder`.

    Logged as a first-class metric rather than thresholded silently: the fork
    threshold is otherwise tuned blind. `transitions` is the number of turn
    boundaries classified, so `clean + realign + fork == transitions`.
    """

    clean: int = 0
    realign: int = 0
    fork: int = 0
    transitions: int = 0
    drift_tokens: int = 0
    drift_max: int = 0

    def observe(self, kind: DriftKind, drift: int) -> None:
        if kind == "clean":
            self.clean += 1
        elif kind == "realign":
            self.realign += 1
        elif kind == "fork":
            self.fork += 1
        else:  # pragma: no cover - Literal exhausts this
            raise ValueError(f"unknown drift kind {kind!r}")
        self.transitions += 1
        self.drift_tokens += drift
        self.drift_max = max(self.drift_max, drift)

    def to_dict(self) -> dict[str, int]:
        return {
            "clean": self.clean,
            "realign": self.realign,
            "fork": self.fork,
            "transitions": self.transitions,
            "drift_tokens": self.drift_tokens,
            "drift_max": self.drift_max,
        }


@dataclass(frozen=True)
class MaskSpan:
    """A contiguous completion range and whether the model produced its tokens.

    Spans are the stored form; `rollout_func` flattens them into the positional
    `env_mask`. Kept as spans so the differential alignment test can name the
    boundary it inspects rather than inferring it from a flat run.
    """

    start: int
    end: int
    supervised: bool


@dataclass
class Episode:
    """One rollout position's full history. Writers and readers per plan table."""

    # --- scheduler ---
    episode_id: str
    scenario_id: str
    group_index: int
    state: EpisodeState = "ready"
    messages: list[Message] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    step_count: int = 0
    invalid_call_count: int = 0
    terminal_reason: TerminalReason | None = None
    # The episode that this one replaced (`worker_crash`), for lineage only.
    replaced_episode_id: str | None = None

    # --- generation ---
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    prompt_completion_boundary: int = 0
    stage_timings: dict[str, list[float]] = field(default_factory=dict)

    # --- mask ---
    mask_spans: list[MaskSpan] = field(default_factory=list)
    drift_tally: DriftTally = field(default_factory=DriftTally)

    # --- reward (Phase 7 owns the writer; the field exists so the record is total) ---
    reward: float | None = None
    per_check_bools: tuple[bool, ...] | None = None

    def record_timing(self, stage: str, seconds: float) -> None:
        self.stage_timings.setdefault(stage, []).append(seconds)

    def env_mask(self) -> list[int]:
        """Flatten the spans into the positional mask TRL expects.

        1 on model tokens, 0 on appended observations. Any gap between spans is
        a bug in span bookkeeping, so it raises rather than padding silently.
        """
        mask = [0] * len(self.completion_ids)
        boundary = self.prompt_completion_boundary
        covered = 0
        for span in self.mask_spans:
            if span.end <= boundary:
                continue
            start = max(span.start, boundary) - boundary
            end = span.end - boundary
            if start != covered:
                raise RuntimeError(
                    f"{self.episode_id}: mask span starts at {start}, "
                    f"expected {covered}; span bookkeeping drifted"
                )
            if end > len(self.completion_ids):
                raise RuntimeError(
                    f"{self.episode_id}: mask span ends at {end} beyond "
                    f"completion length {len(self.completion_ids)}"
                )
            for position in range(start, end):
                mask[position] = 1 if span.supervised else 0
            covered = end
        if covered != len(self.completion_ids):
            raise RuntimeError(
                f"{self.episode_id}: mask spans cover {covered} of "
                f"{len(self.completion_ids)} completion tokens"
            )
        return mask

    def to_row(self) -> dict[str, Any]:
        """The trajectory-table row Phase 7 reads. Field names are its contract."""
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "group_index": self.group_index,
            "messages": [message.to_template_dict() for message in self.messages],
            "observations": list(self.observations),
            "step_count": self.step_count,
            "invalid_call_count": self.invalid_call_count,
            "terminal_reason": self.terminal_reason,
            "replaced_episode_id": self.replaced_episode_id,
            "reward": self.reward,
            "per_check_bools": list(self.per_check_bools or ()),
            "drift": self.drift_tally.to_dict(),
            "stage_timings": {stage: list(v) for stage, v in self.stage_timings.items()},
        }
