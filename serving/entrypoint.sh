#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_API_KEY:?VLLM_API_KEY must be set}"

python - <<'PY'
from importlib.metadata import version

expected = "0.26.0"
actual = version("vllm")
if actual != expected:
    raise SystemExit(f"vLLM version mismatch: expected {expected}, found {actual}")
PY

# PyPI CUDA packages place runtime libraries below site-packages/nvidia/*/lib.
# Add every existing directory without assuming one CUDA minor version.
cuda_library_path="$({ python - <<'PY'
import site
from pathlib import Path

paths = []
for root in map(Path, site.getsitepackages()):
    paths.extend(str(path) for path in sorted((root / "nvidia").glob("*/lib")) if path.is_dir())
print(":".join(paths))
PY
} )"
if [[ -n "$cuda_library_path" ]]; then
  export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

profile="${SMOLQWEN_PROFILE:-l4}"
exec smolqwen serve --profile "$profile" "$@"
