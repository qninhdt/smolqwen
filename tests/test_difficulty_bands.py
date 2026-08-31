from __future__ import annotations

from smolqwen.training.difficulty import profile_rewards, weighted_scenario_order


def test_verifier_outcomes_define_the_three_difficulty_bands() -> None:
    profile = profile_rewards(
        {
            "impossible": [0.0, 0.25, 0.0, 0.5],
            "band": [0.0, 1.0, 0.5, 1.0],
            "easy": [1.0, 1.0, 1.0, 1.0],
        },
        model_id="checkpoint",
        model_revision="abc",
        seed=7,
    )
    assert profile.by_task["impossible"].band == "always_zero"
    assert profile.by_task["band"].band == "band"
    assert profile.by_task["easy"].band == "always_one"
    assert profile.by_task["impossible"].mean_reward == 0.1875


def test_zero_weight_bands_are_excluded_from_the_curriculum() -> None:
    profile = profile_rewards(
        {"zero": [0.0, 0.0], "band": [0.0, 1.0], "one": [1.0, 1.0]},
        model_id="checkpoint",
        model_revision=None,
        seed=7,
    )
    order = weighted_scenario_order(
        ["zero", "band", "one"],
        profile,
        seed=7,
        band_weight=1.0,
        always_zero_weight=0.0,
        always_one_weight=0.0,
    )
    assert order == ["band"]
