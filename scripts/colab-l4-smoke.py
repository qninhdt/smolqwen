"""One-shot real-L4 verification driver used through ``colab exec``.

The local controller owns session teardown.  This script owns in-VM process
cleanup and writes a machine-readable result even when a verification step
fails, so the controller can download evidence before releasing the VM.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/content/smolqwen")
ARCHIVE = Path("/content/smolqwen-l4-src.tgz")
RESULT = Path("/content/smolqwen-l4-smoke-results.json")
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SUBMODULES = (
    (
        "https://github.com/RUC-NLPIR/EnvScaler.git",
        "third_party/EnvScaler",
        "87e667397abacf274858c0964796beb8f984aafe",
    ),
    (
        "https://github.com/ShishirPatil/gorilla.git",
        "third_party/gorilla",
        "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    ),
)

KERNEL_SELFTEST = r"""
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.__version__.split("+", 1)[0] != "2.11.0":
    raise SystemExit(f"expected torch 2.11.0, got {torch.__version__}")

from flash_attn import flash_attn_func

qkv = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(qkv, qkv, qkv, causal=True)
print("verified flash_attn", flush=True)

from causal_conv1d import causal_conv1d_fn

causal_conv1d_fn(
    torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16),
    torch.randn(16, 4, device="cuda", dtype=torch.bfloat16),
)
print("verified causal_conv1d", flush=True)

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

shape = (1, 64, 4, 64)
chunk_gated_delta_rule(
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    torch.randn(*shape, device="cuda", dtype=torch.bfloat16),
    g=torch.rand(*shape[:3], device="cuda", dtype=torch.float32).log(),
    beta=torch.rand(*shape[:3], device="cuda", dtype=torch.bfloat16),
)
print("verified flash_linear_attention", flush=True)

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

LigerFusedLinearCrossEntropyLoss()(
    torch.randn(1000, 128, device="cuda", dtype=torch.bfloat16),
    torch.randn(8, 128, device="cuda", dtype=torch.bfloat16),
    torch.randint(0, 1000, (8,), device="cuda"),
)
print("verified liger_kernel", flush=True)
"""

EVAL_CONFIG_SMOKE = r"""
from pathlib import Path

import yaml

source = Path("configs/base/eval.yaml")
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
payload["adapters"] = ["envscaler_heldout"]
heldout = payload["adapter_options"]["envscaler_heldout"]
heldout["env_count"] = 1
heldout["scenarios_per_env"] = 1
payload["decoding"]["max_new_tokens"] = 32
payload["max_steps_per_task"] = 1
Path("/content/eval-smoke.yaml").write_text(
    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
)
"""


class StepFailure(RuntimeError):
    """A verification command returned a non-zero exit status."""


results: list[dict[str, Any]] = []


def record(name: str, status: str, **details: Any) -> None:
    results.append({"name": name, "status": status, **details})
    RESULT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def run(name: str, command: list[str], *, timeout: int = 1800) -> None:
    print(f"\n=== {name} ===", flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            timeout=timeout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 2)
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        tail = output[-12000:]
        if tail:
            print(tail, file=sys.stderr, flush=True)
        record(
            name,
            "failed",
            returncode=None,
            duration_s=duration,
            output_tail=tail,
            timeout_s=timeout,
        )
        raise StepFailure(f"{name} timed out after {timeout}s") from exc
    duration = round(time.monotonic() - started, 2)
    output = completed.stdout or ""
    if completed.returncode != 0:
        tail = output[-12000:]
        if tail:
            print(tail, file=sys.stderr, flush=True)
        record(
            name,
            "failed",
            returncode=completed.returncode,
            duration_s=duration,
            output_tail=tail,
        )
        raise StepFailure(f"{name} failed with exit code {completed.returncode}")
    record(name, "passed", returncode=0, duration_s=duration)


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    request.add_header("Authorization", f"Bearer {os.environ['VLLM_API_KEY']}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode()
    return json.loads(body) if body else None


def wait_for_server(process: subprocess.Popen[Any], log_path: Path, *, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
            record("vllm-serve-log", "failed", returncode=process.returncode, tail=tail)
            raise StepFailure(f"vLLM exited before readiness with {process.returncode}:\n{tail}")
        try:
            request = urllib.request.Request("http://127.0.0.1:8000/health")
            request.add_header("Authorization", f"Bearer {os.environ['VLLM_API_KEY']}")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise StepFailure(f"vLLM readiness timed out: {last_error}")


def terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def prepare() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    for url, relative_path, revision in SUBMODULES:
        destination = ROOT / relative_path
        destination.mkdir(parents=True)
        subprocess.run(["git", "-C", str(destination), "init"], check=True)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
            check=True,
        )
        subprocess.run(["git", "-C", str(destination), "checkout", "FETCH_HEAD"], check=True)
    record("prepare-source", "passed")


def serve_smoke() -> None:
    api_key = secrets.token_urlsafe(32)
    os.environ["VLLM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["SMOLQWEN_BASE_URL"] = "http://127.0.0.1:8000"
    command = [
        "uv",
        "run",
        "smolqwen",
        "serve",
        "--profile",
        "l4",
        "--override",
        "model_path=Qwen/Qwen3.5-2B",
        "--override",
        f"model_revision={MODEL_REVISION}",
        "--override",
        "max_model_len=8192",
        "--override",
        "max_num_seqs=4",
        "--override",
        "max_num_batched_tokens=8192",
        "--override",
        "gpu_memory_utilization=0.80",
    ]
    print("\n=== vllm-serve-smoke ===", flush=True)
    started = time.monotonic()
    log_path = Path("/content/vllm-smoke.log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            start_new_session=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_server(process, log_path)
            models = request_json("http://127.0.0.1:8000/v1/models")
            completion = request_json(
                "http://127.0.0.1:8000/v1/chat/completions",
                {
                    "model": "smolqwen",
                    "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                    "temperature": 0,
                    "max_tokens": 16,
                },
            )
            if not models.get("data"):
                raise StepFailure("vLLM models response is empty")
            if not completion.get("choices"):
                raise StepFailure("vLLM chat completion response is empty")
            record(
                "vllm-serve-smoke",
                "passed",
                duration_s=round(time.monotonic() - started, 2),
                model=models["data"][0].get("id"),
                finish_reason=completion["choices"][0].get("finish_reason"),
            )

            run(
                "vllm-bench-smoke",
                [
                    "uv",
                    "run",
                    "smolqwen",
                    "bench",
                    "--profile",
                    "l4",
                    "--dataset",
                    "random",
                    "--override",
                    "model_path=Qwen/Qwen3.5-2B",
                    "--override",
                    f"model_revision={MODEL_REVISION}",
                    "--override",
                    "benchmark_num_prompts=2",
                    "--override",
                    "benchmark_input_len=16",
                    "--override",
                    "benchmark_output_len=8",
                    "--concurrency",
                    "1",
                ],
                timeout=600,
            )
            run(
                "prepare-eval-smoke-config",
                ["uv", "run", "python", "-c", EVAL_CONFIG_SMOKE],
                timeout=60,
            )
            run(
                "heldout-eval-smoke",
                [
                    "uv",
                    "run",
                    "smolqwen",
                    "evaluate",
                    "--profile",
                    "l4",
                    "--config",
                    "/content/eval-smoke.yaml",
                    "--checkpoint",
                    "Qwen/Qwen3.5-2B",
                    "--revision",
                    MODEL_REVISION,
                    "--tag",
                    "l4-smoke",
                    "--adapter",
                    "envscaler_heldout",
                    "--endpoint",
                    "http://127.0.0.1:8000",
                    "--serving-backend",
                    "vllm",
                    "--served-dtype",
                    "bfloat16",
                    "--kv-budget",
                    "0.80",
                    "--max-num-seqs",
                    "4",
                    "--max-num-batched-tokens",
                    "8192",
                    "--chunked-prefill",
                    "--prefix-caching",
                ],
                timeout=900,
            )
        finally:
            terminate(process)


def main() -> int:
    RESULT.write_text("[]\n", encoding="utf-8")
    try:
        prepare()
        run(
            "install-colab",
            ["uv", "sync", "--locked", "--no-dev", "--extra", "colab"],
            timeout=2400,
        )
        serve_smoke()
    except Exception as exc:
        record("run", "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    record("run", "passed")
    print(f"\nRESULT_FILE={RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
