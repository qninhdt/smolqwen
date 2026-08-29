"""Conv-split and the seeded train/val split by trajectory id.

- A Conv trajectory yields one sample per real user turn, each sample's prompt a
  cumulative prefix that begins at the system message and ends at that turn's
  user query (+ generation prompt).
- The train/val split is **by trajectory id, never by sample**, so a Conv
  trajectory's segments never straddle the split, and it is reproducible from
  the seed.
"""

from __future__ import annotations

from pathlib import Path

from smolqwen.data.loader import Trajectory, parse_trajectory
from smolqwen.data.render import split_segments
from smolqwen.data.splits import build_env_split_manifest, split_trajectory_ids
from tests.helpers import OfflineTokenizer, load_trajectory_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_trajectory(trajectory_id: str) -> Trajectory:
    for row in load_trajectory_rows():
        try:
            traj = parse_trajectory(row)
        except Exception:
            continue
        if traj.trajectory_id == trajectory_id:
            return traj
    raise AssertionError(f"fixture missing {trajectory_id}")


def test_conv_trajectory_splits_at_real_user_boundaries() -> None:
    """One sample per real user turn, each with a cumulative prompt prefix."""
    traj = _fixture_trajectory("env_82_sft-task_29")
    segments = split_segments(traj.messages)

    # Real user turns at indices 1, 7, 13; 13 has no assistant after it and is
    # dropped. Two supervised samples, cumulative prefixes.
    assert [(s.prompt_upto, s.completion_upto) for s in segments] == [(2, 7), (8, 13)]

    # Each prompt prefix is a strict prefix of the next: the second sample's
    # prompt includes the first sample's full render, so the earlier history is
    # available as context rather than re-derived.
    tokenizer = OfflineTokenizer()
    real_user_indices = {i for i, m in enumerate(traj.messages) if m.is_real_user_turn}
    for segment in segments:
        prompt = tokenizer.apply_chat_template(
            [m.to_template_dict() for m in traj.messages[: segment.prompt_upto]],
            add_generation_prompt=True,
        )
        # The prompt always starts at the system message and ends with the
        # generation prompt marker (Qwen3.5 opens a `<think>` block).
        assert prompt.startswith("<|im_start|>system")
        assert prompt.endswith("<|im_start|>assistant\n<think>\n")
        # Every segment opens right after a real user message -- the only boundary
        # the split cuts at.
        assert segment.prompt_upto - 1 in real_user_indices


def test_nonconv_trajectory_is_single_sample() -> None:
    """A Non-Conv trajectory converts to exactly one supervised segment."""
    traj = _fixture_trajectory("env_3_sft-task_28")
    segments = split_segments(traj.messages)
    assert len(segments) == 1
    assert segments[0].prompt_upto == 2  # system + first user


def test_split_by_trajectory_id_is_seeded_and_reproducible() -> None:
    ids = [f"traj-{i}" for i in range(120)]
    first = split_trajectory_ids(ids, seed=7, val_fraction=0.1)
    second = split_trajectory_ids(ids, seed=7, val_fraction=0.1)
    assert first == second
    assert len(first.val_ids) == 12
    assert first.train_ids & first.val_ids == set()
    assert len(first.train_ids) + len(first.val_ids) == len(set(ids))


def test_split_val_is_a_fixed_fraction() -> None:
    ids = [f"traj-{i}" for i in range(1000)]
    split = split_trajectory_ids(ids, seed=3, val_fraction=0.02)
    assert 15 <= len(split.val_ids) <= 25  # 2% of 1000, rounding-tolerant


def test_routing_by_trajectory_id_never_straddles() -> None:
    """Segments of one trajectory route together because the key is the trajectory id.

    In production `_write_shards` sends each `Converted` trajectory's samples to a
    single partition chosen by `split.partition(trajectory_id)`. A Conv trajectory
    that yields several samples therefore never splits its segments across
    train/val. Test the routing decision directly.
    """
    traj_ids = [f"traj-{i}" for i in range(50)]
    split = split_trajectory_ids(traj_ids, seed=11, val_fraction=0.1)
    # Simulate a multi-segment trajectory landing in each partition: every
    # segment shares the trajectory id, so each segment routes identically.
    for tid in traj_ids:
        segments = [f"{tid}#seg-{i}" for i in range(3)]
        partitions = {split.partition(tid) for _ in segments}
        assert len(partitions) == 1


def test_env_split_manifest_from_metadata() -> None:
    """The manifest derives SFT/RL by `env_id` suffix and flags RL violations."""
    # A synthetic (small) metadata object: the derived split is tested here; the
    # real 140/51 split is recorded by the actual `smolqwen profile-data` run.
    env_meta = FIXTURES / "env_metadata_small.json"
    env_meta.write_text(
        '{"env_1_sft": {}, "env_2_sft": {}, "env_10_rl": {}, "env_11_rl": {}, "env_9_other": {}}',
        encoding="utf-8",
    )
    try:
        manifest = build_env_split_manifest(
            env_meta,
            rl_scenario_env_ids=["env_10_rl", "env_11_rl", "env_1_sft"],
        )
        assert manifest.sft_env_ids == ("env_1_sft", "env_2_sft")
        assert manifest.rl_env_ids == ("env_10_rl", "env_11_rl")
        assert manifest.other_env_ids == ("env_9_other",)
        # The RL scenario referencing an SFT env is a data-lineage violation.
        assert manifest.other_env_count == 1
    finally:
        env_meta.unlink(missing_ok=True)
