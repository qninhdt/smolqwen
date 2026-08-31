from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serving_key_file_is_ignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "artifacts/serving/vllm-api-key" in ignored


def test_committed_notebooks_have_no_outputs() -> None:
    for notebook_path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("outputs", []) == [], notebook_path
            assert cell.get("execution_count") is None, notebook_path


def test_ci_runs_without_gpu_or_remote_model_access() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES: ""' in workflow
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert "make test-ci" in workflow
