#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required; install it before running setup_colab.sh" >&2
  exit 1
}

# Kernel and CUDA packages are declared `no-build-package` in pyproject.toml, so a
# resolution that would have to invoke nvcc fails as unsatisfiable rather than
# burning an hour of a reclaimable VM compiling flash-attn. `--locked` additionally
# refuses to re-resolve, so the exact prebuilt wheel URLs in uv.lock are what lands.
#
# Install the locked base first. The kernel wheels are built against one exact
# torch ABI, so validate the vLLM-owned anchor before installing them.
uv sync --locked --no-dev
uv run --no-sync python - <<'PY'
from importlib.metadata import version

expected = "2.11.0"
actual = version("torch")
if actual != expected:
    raise SystemExit(f"torch pin mismatch: expected {expected}, resolved {actual}")
print(f"verified torch=={actual}")
PY

uv sync --locked --no-dev --extra colab

git submodule update --init --recursive --checkout third_party/EnvScaler
git submodule status --recursive third_party/EnvScaler

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# The kernel libraries are mandatory, not opportunistic. Falling back to sdpa and
# an unfused loss head materializes a dense [batch, seq, 248320] logits tensor and
# the unfused GDN recurrence, neither of which fits in 24 GB -- so a missing
# kernel is a setup failure, not a slower run. Each is exercised on the device,
# because importable and ABI-compatible are different claims.
uv run --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable after Colab setup")
major, minor = torch.cuda.get_device_capability(0)
properties = torch.cuda.get_device_properties(0)
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"compute_capability={major}.{minor}")
print(f"vram_gb={properties.total_memory / 1024**3:.2f}")

from flash_attn import flash_attn_func

qkv = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(qkv, qkv, qkv, causal=True)
print("verified flash_attn")

from causal_conv1d import causal_conv1d_fn

causal_conv1d_fn(
    torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16),
    torch.randn(16, 4, device="cuda", dtype=torch.bfloat16),
)
print("verified causal_conv1d")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

shape = (1, 64, 4, 64)
chunk_gated_delta_rule(
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    g=torch.rand(*shape[:3], device="cuda", dtype=torch.float32).log(),
    beta=torch.rand(*shape[:3], device="cuda", dtype=torch.bfloat16),
)
print("verified flash_linear_attention")

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

LigerFusedLinearCrossEntropyLoss()(
    torch.randn(1000, 128, device="cuda", dtype=torch.bfloat16),
    torch.randn(8, 128, device="cuda", dtype=torch.bfloat16),
    torch.randint(0, 1000, (8,), device="cuda"),
)
print("verified liger_kernel")
PY
