"""The system prompt we present is the one the release trained under.

Phase 5's first diagnosis step when SFT fails to beat Base is "the prompt matches
between training and rollout". That check is worth nothing if it compares our
prompt against our own copy of it, so these tests compare against two independent
sources: upstream's `system_prompt_util.py` module, and the released trajectory
file's own `messages[0].content`.

The release-file test is `dataset`-marked (CPU CI has no 701 MB download); the
upstream-module test needs only the submodule, so it runs everywhere.
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import pytest

from smolqwen.data.loader import iter_json_array
from smolqwen.prompts import (
    CONVERSATIONAL,
    INTRODUCTION_SEPARATOR,
    NON_CONVERSATIONAL,
    build_system_prompt,
    system_prompt_hash,
)

UPSTREAM_AGENT = Path("third_party/EnvScaler/interact_with_env")
RELEASE_GLOB = (
    "~/.cache/huggingface/hub/datasets--XXHStudyHard--EnvScaler-SFT-Traj-9K/"
    "snapshots/*/envscaler_sft_traj_9k_metadata.json"
)


def test_both_prompts_are_byte_identical_to_upstreams_own_literals() -> None:
    """A paraphrase here silently shifts the whole evaluation distribution."""
    if not UPSTREAM_AGENT.is_dir():
        pytest.skip(f"{UPSTREAM_AGENT} not present")
    sys.path.insert(0, str(UPSTREAM_AGENT))
    try:
        from agent.system_prompt_util import (  # type: ignore[import-not-found]
            conversational_system_prompt,
            non_conversational_system_prompt,
        )
    finally:
        sys.path.remove(str(UPSTREAM_AGENT))
    assert CONVERSATIONAL == conversational_system_prompt
    assert NON_CONVERSATIONAL == non_conversational_system_prompt


def test_the_introduction_join_is_exact() -> None:
    assembled = build_system_prompt(conversational=False, env_introduction="INTRO")
    assert assembled == NON_CONVERSATIONAL + INTRODUCTION_SEPARATOR + "INTRO"
    # BFCL supplies no introduction, so its prompt is the bare mode string.
    assert build_system_prompt(conversational=False) == NON_CONVERSATIONAL


def test_the_hash_distinguishes_the_two_modes() -> None:
    assert system_prompt_hash(CONVERSATIONAL) != system_prompt_hash(NON_CONVERSATIONAL)


@pytest.mark.dataset
def test_released_trajectories_start_with_the_assembled_prompt() -> None:
    """The trained prompt is the released one, separator included."""
    matches = glob.glob(str(Path(RELEASE_GLOB).expanduser()))
    if not matches:
        pytest.skip("EnvScaler SFT release not downloaded")
    checked = 0
    for row in itertools.islice(iter_json_array(matches[0]), 200):
        messages = json.loads(row["messages"])
        assert messages[0]["role"] == "system"
        conversational = row["traj_type"] == "conversation"
        base = CONVERSATIONAL if conversational else NON_CONVERSATIONAL
        content = messages[0]["content"]
        assert content.startswith(base), row["traj_id"]
        remainder = content[len(base) :]
        assert not remainder or remainder.startswith(INTRODUCTION_SEPARATOR)
        checked += 1
    assert checked > 0
