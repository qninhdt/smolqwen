#!/usr/bin/env python
"""Validate one measured direct-vLLM profile and leave a downloadable bundle."""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

ROOT = Path("/content/smolqwen")
ARCHIVE = Path("/content/smolqwen-l4-src.tgz")
CONFIG = Path("/content/serving-validation-config.json")
VENV = Path("/content/vllm029")
PYTHON = VENV / "bin/python"
VLLM = VENV / "bin/vllm"
LOG = Path("/content/serving-validation.log")
BUNDLE = Path("/content/serving-validation.tgz")
OUTPUT = ROOT / "artifacts/serving/validation"
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BASE_URL = "http://127.0.0.1:8000"


class ValidationError(RuntimeError):
    """Validation completed with one or more failed checks."""

    def __init__(self, checks: list[dict[str, object]]) -> None:
        self.checks = checks
        super().__init__(
            "failed checks: "
            + ", ".join(str(check["name"]) for check in checks if not check["passed"])
        )


def _prepare_source() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")


def _install_runtime() -> None:
    if not PYTHON.exists():
        subprocess.run(["uv", "venv", str(VENV)], check=True)
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import pandas, torch, vllm; print(vllm.__version__, torch.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip().startswith("0.29.0 2.13.0"):
        return
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(PYTHON),
            "vllm==0.29.0",
            "torch==2.13.0",
            "pandas",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
            "--extra-index-url",
            "https://pypi.org/simple",
            "--index-strategy",
            "unsafe-best-match",
        ],
        check=True,
    )


def _server_command(profile: dict[str, object]) -> list[str]:
    command = [
        str(VLLM),
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--served-model-name",
        "smolqwen",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "32768",
        "--dtype",
        str(profile["dtype"]),
        "--kv-cache-dtype",
        str(profile["kv_cache_dtype"]),
        "--max-num-seqs",
        str(profile["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(profile["max_num_batched_tokens"]),
        "--max-num-queued-reqs",
        str(profile["max_num_queued_reqs"]),
        "--max-num-queued-tokens",
        str(profile["max_num_queued_tokens"]),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if profile.get("quantization"):
        command.extend(["--quantization", str(profile["quantization"])])
    return command


def _request(
    path: str,
    *,
    api_key: str | None = None,
    payload: dict[str, object] | None = None,
    timeout: float = 180,
) -> tuple[int, bytes]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(BASE_URL + path, data=body)
    if api_key is not None:
        request.add_header("Authorization", f"Bearer {api_key}")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _json_request(
    path: str,
    *,
    api_key: str,
    payload: dict[str, object] | None = None,
    timeout: float = 180,
) -> tuple[int, dict[str, Any]]:
    status, body = _request(path, api_key=api_key, payload=payload, timeout=timeout)
    return status, json.loads(body) if body else {}


def _chat_payload(prompt: str, *, max_tokens: int = 32) -> dict[str, object]:
    return {
        "model": "smolqwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _metrics() -> tuple[str, dict[str, float]]:
    status, body = _request("/metrics")
    if status != 200:
        raise RuntimeError(f"metrics returned HTTP {status}")
    text = body.decode()
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        name = sample.split("{", 1)[0]
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return text, totals


def _wait_ready(process: subprocess.Popen[bytes], timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with {process.returncode}")
        try:
            status, _ = _request("/health", timeout=5)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM readiness timed out")


def _wait_idle(timeout: float = 90) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        _, last = _metrics()
        if (
            last.get("vllm:num_requests_running", 0) == 0
            and last.get("vllm:num_requests_waiting", 0) == 0
        ):
            return last
        time.sleep(1)
    raise RuntimeError(f"request gauges did not return to zero: {last}")


def _stream_once(api_key: str, *, cancel: bool) -> int:
    payload = _chat_payload(
        "Count upward forever, one integer per line."
        if cancel
        else "Reply with exactly: stream-ready",
        max_tokens=2048 if cancel else 32,
    )
    payload["stream"] = True
    if cancel:
        payload["ignore_eos"] = True
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    chunks = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            if raw.startswith(b"data: "):
                chunks += 1
                if cancel or raw.strip() == b"data: [DONE]":
                    break
    return chunks


def _validate(profile: dict[str, object], api_key: str) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, *, required: bool = True, **details: object) -> None:
        checks.append({"name": name, "passed": passed, "required": required, **details})
        if required and not passed:
            failures.append(name)

    bad_status, _ = _request("/v1/models", api_key="invalid")
    good_status, models = _json_request("/v1/models", api_key=api_key)
    record(
        "auth-and-models",
        bad_status == 401
        and good_status == 200
        and any(item.get("id") == "smolqwen" for item in models.get("data", [])),
        bad=bad_status,
        good=good_status,
    )

    chat_status, chat = _json_request(
        "/v1/chat/completions", api_key=api_key, payload=_chat_payload("Reply exactly: ready")
    )
    record("chat", chat_status == 200 and bool(chat.get("choices")), status=chat_status)

    tool_payload = _chat_payload("What is the weather in Paris?", max_tokens=64)
    tool_payload.update(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )
    tool_status, tool = _json_request("/v1/chat/completions", api_key=api_key, payload=tool_payload)
    tool_calls = tool.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    record("tool-call", tool_status == 200 and bool(tool_calls), status=tool_status)

    record("streaming", _stream_once(api_key, cancel=False) > 0)

    required_metrics = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:request_success_total",
        "vllm:time_to_first_token_seconds_count",
        "vllm:inter_token_latency_seconds_count",
        "vllm:request_queue_time_seconds_count",
    }
    before_text, before = _metrics()
    missing = sorted(required_metrics - before.keys())
    record("metrics", not missing, missing=missing)

    prefix_payload = _chat_payload(
        "A stable validation prefix with repeated context. " * 128, max_tokens=8
    )
    prefix_payload["cache_salt"] = "smolqwen-validation"
    for suffix in ("first", "second"):
        payload = dict(prefix_payload)
        payload["cache_salt"] = "smolqwen-validation"
        status, response = _json_request("/v1/chat/completions", api_key=api_key, payload=payload)
        record(
            f"apc-request-{suffix}", status == 200 and bool(response.get("choices")), status=status
        )
    _, after_apc = _metrics()
    hit_delta = after_apc.get("vllm:prefix_cache_hits_total", 0) - before.get(
        "vllm:prefix_cache_hits_total", 0
    )
    record(
        "apc-hit",
        hit_delta > 0,
        required=False,
        hit_delta=hit_delta,
        limitation=(
            "Qwen3.5 GDN align-mode prefix reuse was not observed; do not rely on APC"
            if hit_delta == 0
            else None
        ),
    )

    cancel_chunks = _stream_once(api_key, cancel=True)
    idle_after_cancel = _wait_idle()
    record("cancellation", cancel_chunks > 0, chunks=cancel_chunks, gauges=idle_after_cancel)

    overload_prompt = "Long admission-control validation prompt. " * 320
    barrier = Barrier(81)

    def overload_request(_: int) -> int:
        barrier.wait()
        payload = _chat_payload(overload_prompt, max_tokens=64)
        payload["ignore_eos"] = True
        status, _ = _request("/v1/chat/completions", api_key=api_key, payload=payload, timeout=300)
        return status

    with ThreadPoolExecutor(max_workers=80) as pool:
        futures = [pool.submit(overload_request, index) for index in range(80)]
        barrier.wait()
        statuses = [future.result() for future in futures]
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    record("native-overload-503", status_counts.get(503, 0) > 0, statuses=status_counts)

    final_metrics = _wait_idle()
    health_status, _ = _request("/health")
    recovery_status, recovery = _json_request(
        "/v1/chat/completions", api_key=api_key, payload=_chat_payload("Reply: recovered")
    )
    preemption_delta = final_metrics.get("vllm:num_preemptions_total", 0) - before.get(
        "vllm:num_preemptions_total", 0
    )
    record(
        "recovery",
        health_status == 200 and recovery_status == 200 and bool(recovery.get("choices")),
        health=health_status,
        chat=recovery_status,
    )
    record("no-preemption", preemption_delta == 0, delta=preemption_delta)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics-before.prom").write_text(before_text, encoding="utf-8")
    final_text, _ = _metrics()
    (OUTPUT / "metrics-after.prom").write_text(final_text, encoding="utf-8")
    if failures:
        raise ValidationError(checks)
    return checks


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    _prepare_source()
    _install_runtime()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    profile = config["profile"]
    api_key = secrets.token_urlsafe(32)
    environment = dict(os.environ)
    environment["PATH"] = f"{VENV / 'bin'}:{environment.get('PATH', '')}"
    environment["VLLM_API_KEY"] = api_key
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["DO_NOT_TRACK"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    status: dict[str, object] = {
        "profile_name": config["profile_name"],
        "profile": profile,
        "status": "failed",
        "started_at_unix": time.time(),
    }
    returncode = 1
    try:
        with LOG.open("wb") as log:
            process = subprocess.Popen(
                _server_command(profile),
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_ready(process)
            status["runtime"] = subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import torch, vllm; print(vllm.__version__, torch.__version__)",
                ],
                text=True,
            ).strip()
            status["checks"] = _validate(profile, api_key)
            status["status"] = "passed"
            returncode = 0
    except Exception as error:
        if isinstance(error, ValidationError):
            status["checks"] = error.checks
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        _terminate(process)
        status["finished_at_unix"] = time.time()
        status["returncode"] = returncode
        (OUTPUT / "validation.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if LOG.exists():
            (OUTPUT / "server.log").write_bytes(LOG.read_bytes())
        with tarfile.open(BUNDLE, "w:gz") as archive:
            archive.add(OUTPUT, arcname="artifacts/serving/validation")
    print(json.dumps(status, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
