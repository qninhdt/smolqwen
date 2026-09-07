#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

heartbeat_s="${SMOLQWEN_SETUP_HEARTBEAT_S:-30}"
log() {
  printf '[setup %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2
}

run_step() {
  local label="$1"
  shift
  local started=$SECONDS
  log "START: $label"
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$heartbeat_s"
    if kill -0 "$pid" 2>/dev/null; then
      log "WAITING: $label ($((SECONDS - started))s elapsed)"
    fi
  done
  if wait "$pid"; then
    log "DONE: $label ($((SECONDS - started))s)"
  else
    local code=$?
    log "FAILED: $label (exit $code after $((SECONDS - started))s)"
    return "$code"
  fi
}

log "setup started in $project_root"

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
run_step "install locked base dependencies" uv sync --locked --no-dev
run_step "verify torch ABI" uv run --no-sync python - <<'PY'
from importlib.metadata import version

expected = "2.11.0"
actual = version("torch")
if actual != expected:
    raise SystemExit(f"torch pin mismatch: expected {expected}, resolved {actual}")
print(f"verified torch=={actual}")
PY

run_step "install locked Colab GPU dependencies" uv sync --locked --no-dev --extra colab

if git -C "$project_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  run_step "restore EnvScaler submodule" git submodule update --init --recursive --checkout third_party/EnvScaler
  git submodule status --recursive third_party/EnvScaler >&2
elif [[ -f "$project_root/third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/191_env_metadata.json" ]]; then
  echo "using archived EnvScaler sources (no Git metadata)"
else
  echo "EnvScaler sources are missing from the source archive" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# The GDN mixer, causal convolution and fused loss kernels are mandatory, not
# opportunistic. Falling back to an unfused loss head materializes a dense
# [batch, seq, 248320] logits tensor and the unfused GDN recurrence, neither of
# which fits in 24 GB -- so a missing kernel there is a setup failure, not a
# slower run. Each is exercised on the device, because importable and
# ABI-compatible are different claims.
#
# Attention is the exception: FlashAttention-2's kernels are Ampere-and-newer, and
# `sdpa` is a correct fallback, so on Turing the wheel is expected to be unusable
# and only its absence-of-crash matters. FP16 is used below sm80 because bf16 has
# no tensor cores there.
run_step "run GPU and kernel self-tests" uv run --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable after Colab setup")
major, minor = torch.cuda.get_device_capability(0)
properties = torch.cuda.get_device_properties(0)
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"compute_capability={major}.{minor}")
print(f"vram_gb={properties.total_memory / 1024**3:.2f}")

if (major, minor) < (7, 5):
    raise SystemExit(f"smolqwen requires sm75 or newer, found sm{major}{minor}")

ampere = (major, minor) >= (8, 0)
attention_dtype = torch.bfloat16 if ampere else torch.float16

if ampere:
    from flash_attn import flash_attn_func

    qkv = torch.randn(1, 128, 8, 64, device="cuda", dtype=attention_dtype)
    flash_attn_func(qkv, qkv, qkv, causal=True)
    print("verified flash_attn")
else:
    query = torch.randn(1, 8, 128, 256, device="cuda", dtype=attention_dtype)
    key = torch.randn(1, 8, 128, 256, device="cuda", dtype=attention_dtype)
    torch.nn.functional.scaled_dot_product_attention(query, key, key, is_causal=True)
    print(f"verified sdpa (sm{major}{minor} predates flash_attention_2)")

from causal_conv1d import causal_conv1d_fn

causal_conv1d_fn(
    torch.randn(2, 16, 64, device="cuda", dtype=attention_dtype),
    torch.randn(16, 4, device="cuda", dtype=attention_dtype),
)
print("verified causal_conv1d")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# `use_qk_l2norm_in_kernel=True` matches how modeling_qwen3_5 calls this at both
# of its call sites, and it is load-bearing in FP16: it bounds ||q||/||k|| to 1
# before the kernel's intra-chunk accumulation. Omitting it makes unit-variance
# FP16 keys overflow to NaN under a weak decay -- measured 0/5 seeds finite on a
# T4 without it, 5/5 with it -- which would fail a card the model runs fine on.
shape = (1, 64, 4, 64)
output, _state = chunk_gated_delta_rule(
    torch.randn(*shape, device="cuda", dtype=attention_dtype),
    torch.randn(*shape, device="cuda", dtype=attention_dtype),
    torch.randn(*shape, device="cuda", dtype=attention_dtype),
    g=torch.rand(*shape[:3], device="cuda", dtype=torch.float32).log(),
    beta=torch.rand(*shape[:3], device="cuda", dtype=attention_dtype),
    use_qk_l2norm_in_kernel=True,
)
if not torch.isfinite(output).all():
    raise SystemExit("flash_linear_attention produced non-finite values on this card")
print("verified flash_linear_attention")

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

LigerFusedLinearCrossEntropyLoss()(
    torch.randn(1000, 128, device="cuda", dtype=attention_dtype),
    torch.randn(8, 128, device="cuda", dtype=attention_dtype),
    torch.randint(0, 1000, (8,), device="cuda"),
)
print("verified liger_kernel")
PY

log "setup complete"
