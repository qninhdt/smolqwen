"""A paired speed/quality row is one experiment, and `invariant` cannot prove it.

`assert_comparable` compares the manifest's **invariant** set — decoding, step
limits, benchmark revision — which is what makes Base, SFT and SFT+RL belong in one
table. It says nothing about serving config, because serving config is
`recorded_free` by design: two runs of the same checkpoint at different
`max_num_seqs` are still comparable *as quality measurements*.

That is exactly why it is the wrong check for a paired row. A throughput number
measured at `max_num_seqs: 128` and a quality score measured at 8 have identical
invariants and are not the same experiment, so the guard has to read
`recorded_free`. The negative control below is the important test: it shows an
`invariant`-only comparison accepting a pairing this guard refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolqwen.eval.manifest import EvalManifest, ManifestMismatchError, assert_comparable
from smolqwen.eval.serving_pairing import (
    PAIRED_SERVING_FIELDS,
    QualityResult,
    ServingPairingError,
    assert_quality_matches_serving,
    load_quality_result,
)

INVARIANT = {"decoding": {"temperature": 0.0}, "max_steps": 20}


def quality(**recorded: object) -> QualityResult:
    return QualityResult(score=0.5, manifest=EvalManifest(INVARIANT, dict(recorded)))


def test_a_matching_serving_config_pairs() -> None:
    observed = {"dtype": "bfloat16", "quantization": None, "max_num_seqs": 64}
    assert_quality_matches_serving(observed, quality(**observed))


def test_a_quantization_difference_refuses_the_pairing() -> None:
    with pytest.raises(ServingPairingError, match="quantization"):
        assert_quality_matches_serving(
            {"dtype": "bfloat16", "quantization": "fp8"},
            quality(dtype="bfloat16", quantization="awq"),
        )


def test_a_batching_difference_refuses_the_pairing_and_invariant_would_not() -> None:
    """The negative control this guard exists for.

    Both manifests carry the same invariant, so `assert_comparable` accepts them --
    and a quality score measured at `max_num_seqs: 8` would be attached to a
    throughput row measured at 128.
    """
    served_wide = quality(max_num_seqs=128, dtype="bfloat16")
    served_narrow = quality(max_num_seqs=8, dtype="bfloat16")

    # `invariant`-only comparison: no objection.
    assert_comparable(served_wide.manifest, served_narrow.manifest)

    # `recorded_free` comparison: refused.
    with pytest.raises(ServingPairingError, match="max_num_seqs"):
        assert_quality_matches_serving({"max_num_seqs": 128, "dtype": "bfloat16"}, served_narrow)


def test_a_field_the_measurement_did_not_record_makes_no_claim() -> None:
    """Inventing `None` on the measurement's behalf would fail every paired row
    against a quantized score, which is a false negative rather than a guard."""
    assert_quality_matches_serving(
        {"dtype": "bfloat16"}, quality(dtype="bfloat16", quantization="fp8")
    )


def test_a_stringified_value_compares_equal_to_its_decoded_form() -> None:
    """Serving config crosses a CLI boundary as text and a manifest as JSON, so the
    same fact arrives in two shapes; a raw comparison would report a mismatch that
    is only serialization."""
    assert_quality_matches_serving(
        {"kv_budget": "0.9", "chunked_prefill": "true"},
        quality(kv_budget=0.9, chunked_prefill=True),
    )


def test_every_paired_field_is_actually_compared() -> None:
    """A field silently dropped from the tuple would let that dimension differ."""
    for field in PAIRED_SERVING_FIELDS:
        with pytest.raises(ServingPairingError, match=field):
            assert_quality_matches_serving({field: "measured"}, quality(**{field: "scored"}))


def write_report(path: Path, *, temperature: float, score: float = 0.5) -> Path:
    path.write_text(
        json.dumps(
            {
                "manifest": EvalManifest({"decoding": {"temperature": temperature}}, {}).to_dict(),
                "metrics": {"multi_turn_overall": {"score": score}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loading_a_quality_report_refuses_an_invariant_mismatch(tmp_path: Path) -> None:
    """Both checks are needed, in that order: comparable as a measurement first,
    then measured under the same serving config."""
    reference = write_report(tmp_path / "base.json", temperature=0.0)
    served = write_report(tmp_path / "served.json", temperature=0.1)

    with pytest.raises(ManifestMismatchError, match="temperature"):
        load_quality_result(served, references=[reference])

    matching = write_report(tmp_path / "sft.json", temperature=0.0, score=0.7)
    assert load_quality_result(matching, references=[reference]).score == 0.7


def test_a_report_without_a_headline_score_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "manifest": EvalManifest(INVARIANT, {}).to_dict(),
                "metrics": {"multi_turn_base": {"score": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multi_turn_overall"):
        load_quality_result(path)
