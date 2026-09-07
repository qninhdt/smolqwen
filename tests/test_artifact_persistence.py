"""What survives VM loss, and what must never leave the machine in a report.

`artifacts.py:1-5` states the premise -- "Colab VMs are reclaimed without warning,
so an adapter that only exists locally does not exist" -- and applies it to
checkpoint artifacts.

Two properties are asserted here, and the second is the one with teeth:

- Each producing command logs its artifact, and every path still works with no
  `WANDB_API_KEY` and no `HF_TOKEN` -- the degradation `tracking.py:1-8` promises.
- **No artifact contains a routable ingress or any userinfo.** The Colab serving
  path is a public `trycloudflare.com` hostname (`scripts/run_colab_serve.sh:53`),
  so a report recording it verbatim publishes a route to a GPU box the moment the
  report is uploaded. `test_auth_all_paths.py:78-79` enforces the same posture for
  the API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smolqwen.config_models import EvalConfig, SftConfig
from smolqwen.eval.manifest import REDACTED_HOST, EvalManifest, redact_endpoint
from smolqwen.eval.report import write_report
from smolqwen.tracking import Tracker, tracker_for


class FakeRun:
    """The `Run` protocol slice, recording what a real W&B run would receive."""

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[Any] = []
        self.finished = False

    def log(self, data: Any, *, step: int | None = None) -> None:
        self.logged.append(dict(data))

    def log_artifact(self, artifact_or_path: Any, **_: Any) -> Any:
        self.artifacts.append(artifact_or_path)
        return artifact_or_path

    def finish(self) -> None:
        self.finished = True


def _artifact_files(run: FakeRun) -> list[str]:
    """Every filename across every logged artifact, in log order."""
    return [entry for artifact in run.artifacts for entry in artifact.manifest.entries]


def test_a_disabled_tracker_logs_nothing_and_raises_nothing(tmp_path: Path) -> None:
    """The no-credential path must not crash or upload."""
    path = tmp_path / "budgets.json"
    path.write_text("{}", encoding="utf-8")
    tracker = Tracker(project="t", enabled=False)
    tracker.start()
    tracker.log_artifact(path, name="data-budgets", artifact_type="dataset-profile")
    tracker.finish()
    # Nothing to assert positively -- the property is that no wandb call happened and
    # no exception escaped. `enabled=False` leaves `_run` None, which is the guard.
    assert tracker.run_id is None


def test_a_missing_file_is_reported_rather_than_uploaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A vanished report must not take the run down with it."""
    run = FakeRun()
    tracker = Tracker(project="t", run=run)
    with caplog.at_level("WARNING", logger="smolqwen.tracking"):
        tracker.log_artifact(tmp_path / "absent.json", name="eval-x", artifact_type="evaluation")
    assert run.artifacts == []
    assert "missing" in caplog.text


def test_evaluate_logs_the_report_with_its_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report names its trajectory files, so uploading it alone dangles them."""
    from smolqwen.eval import runner

    run = FakeRun()
    monkeypatch.setattr(
        runner,
        "load_http_policy",
        lambda **_: SimpleNamespace(revision="a" * 40, adapter_revision=None),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_named_adapter",
        lambda *_a, **_k: ({"fixture": _metrics()}, {"dataset_hash": "hash"}),
    )
    monkeypatch.setattr(runner, "tracker_for", lambda *_a, **_k: Tracker(project="t", run=run))
    config = EvalConfig(adapters=("fixture",), output_dir=str(tmp_path))
    args = SimpleNamespace(
        checkpoint=None,
        revision="a" * 40,
        endpoint="http://127.0.0.1:8000/v1",
        adapter_path=None,
        adapter_revision=None,
        adapter=None,
        tag="sft",
        serving_backend="vllm",
        require_serving_match=None,
    )
    assert runner.run_evaluation(config, args) == 0

    files = _artifact_files(run)
    assert "sft.json" in files
    assert "sft.md" in files
    assert "sft-fixture.jsonl" in files
    # Headline scalars reach the run too, so a report has a chart beside it.
    assert any("eval/sft/fixture/score" in payload for payload in run.logged)
    # The run is closed even though the upload happened inside the try block.
    assert run.finished


def test_the_merged_push_is_opt_in_and_uses_its_own_repo(tmp_path: Path) -> None:
    """Sharing the adapter repo would interleave two kinds of revision in one history."""
    from smolqwen.training.merge import MergeResult, push_merged

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x" * 1024)
    result = MergeResult(
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(merged),
        base_model_id="Qwen/Qwen3.5-2B",
        base_revision="b" * 40,
        merged_parameters=7,
    )

    unconfigured = SftConfig()
    assert unconfigured.tracking.merged_hub_repo_id is None
    assert push_merged(unconfigured, result) is None

    configured = SftConfig(
        tracking=unconfigured.tracking.model_copy(
            update={"hub_repo_id": "org/adapters", "merged_hub_repo_id": "org/merged"}
        )
    )
    assert configured.tracking.merged_hub_repo_id != configured.tracking.hub_repo_id

    pushes: list[dict[str, Any]] = []
    store = SimpleNamespace(
        enabled=True,
        repo_id="org/merged",
        push=lambda **kwargs: pushes.append(kwargs),
    )
    assert push_merged(configured, result, store=store) == "org/merged"
    # Pushed from the merged directory directly: `save_adapter` would rmtree the
    # adapter cache and copy gigabytes first.
    assert pushes[0]["folder"] == str(merged)
    assert "b" * 40 in pushes[0]["commit_message"]


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        # The one that matters: a public tunnel hostname.
        ("https://calm-river-1234.trycloudflare.com/v1", f"https://{REDACTED_HOST}/v1"),
        # Loopback and private survive: not reachable off-host, and real provenance.
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("http://10.0.0.4:8000/v1", "http://10.0.0.4:8000/v1"),
        # A public IP is as routable as a hostname.
        ("http://34.12.5.6:8000/v1", f"http://{REDACTED_HOST}:8000/v1"),
        # Userinfo goes unconditionally, including on the loopback branch.
        ("https://user:secret@127.0.0.1:8080/v1", "https://127.0.0.1:8080/v1"),
        ("https://user:secret@public.example.com/v1", f"https://{REDACTED_HOST}/v1"),
        # Not a URL this can reason about, so nothing can be asserted non-routable.
        ("public.example.com:8000", REDACTED_HOST),
        (None, None),
    ],
)
def test_endpoint_redaction(endpoint: str | None, expected: str | None) -> None:
    assert redact_endpoint(endpoint) == expected


def test_redaction_is_idempotent_so_a_rehydrated_report_is_unchanged() -> None:
    once = redact_endpoint("https://calm-river.trycloudflare.com/v1")
    assert redact_endpoint(once) == once


def test_no_written_report_contains_a_tunnel_hostname_or_userinfo(tmp_path: Path) -> None:
    """End to end, on the bytes: both report files and the rehydrated manifest."""
    secret_host = "calm-river-1234.trycloudflare.com"
    manifest = EvalManifest(
        invariant={"temperature": 0.0},
        recorded_free={
            "backend": "vllm",
            "endpoint": f"https://operator:hunter2@{secret_host}/v1",
            "checkpoint_revision": "a" * 40,
        },
    )
    json_path, markdown_path = write_report(
        tmp_path, tag="served", manifest=manifest, metrics={"multi_turn_overall": _metrics()}
    )
    for path in (json_path, markdown_path):
        text = path.read_text(encoding="utf-8")
        assert secret_host not in text
        assert "hunter2" not in text
        assert "operator" not in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["recorded_free"]["endpoint"] == f"https://{REDACTED_HOST}/v1"
    # Rehydration goes through the same constructor, so it cannot reintroduce one.
    assert EvalManifest.from_dict(payload["manifest"]).recorded_free["endpoint"] == (
        f"https://{REDACTED_HOST}/v1"
    )


def _metrics() -> dict[str, float]:
    return {
        "score": 0.5,
        "invalid_call_rate": 0.0,
        "average_steps": 1.0,
        "average_generated_tokens": 2.0,
        "truncation_rate": 0.0,
    }


def test_tracker_for_reads_the_tracking_config_it_is_given() -> None:
    """One answer to "which run does this command attach to", not four spellings."""
    config = EvalConfig()
    tracker = tracker_for(
        config.tracking.model_copy(update={"wandb_project": "p", "run_name": "r"}),
        config={"a": 1},
    )
    assert (tracker.project, tracker.run_name, tracker.config) == ("p", "r", {"a": 1})
    # No key in this environment, so it is disabled rather than half-configured.
    assert tracker.enabled is bool(__import__("os").environ.get("WANDB_API_KEY"))
