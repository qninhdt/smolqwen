"""Run the smolqwen pipeline validation matrix on a Colab GPU.

The local controller uploads this source tree, then invokes this file once with
``colab exec``.  The invocation only starts a detached controller because the
Colab websocket times out while a model is loading or training.  Each phase runs
in a fresh child process, so an expected CUDA OOM cannot poison the next
measurement.

The phases validate the wiring the CPU suite cannot: a real SFT step, a real
GRPO step over the rollout boundary, a real batched evaluation through the
in-process vLLM engine, and the GPU-marked tests.  On a card below sm80 the run
takes the ``sdpa`` + FP16 padded path, which is a different execution path but
the same pipeline.

``--only`` restricts the matrix to named phases.  ``--profile auto`` selects the
T4, L4, or A100 sizing profile from the assigned device; pass a profile explicitly
when the card and sizing target intentionally differ.  A free Colab T4 has been
observed to be reclaimed about an hour after assignment regardless of activity,
which is less than the whole matrix takes on 2 vCPUs, so the run has to be
resumable across sessions.  ``--only`` is how the next session picks up the
phases the previous one did not reach; results are merged into the existing
``artifacts/gpu-validation.json`` rather than replacing it.  Every phase row carries
its run id, model revision, resolved profile, and device identity; a resume refuses
to merge rows from a different validation identity.

Expected remote layout::

    /content/smolqwen/.venv/bin/python
    /content/smolqwen/scripts/colab-gpu-validation.py

Use ``colab download`` to retrieve ``artifacts/gpu-validation.json`` and the
matching log after the controller finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from colab_logging import run_streaming
except ModuleNotFoundError:  # imported from the repository root in tests
    from scripts.colab_logging import run_streaming

ROOT = Path("/content/smolqwen")
PYTHON = ROOT / ".venv" / "bin" / "python"
CLI = ROOT / ".venv" / "bin" / "smolqwen"
SCRIPT = ROOT / "scripts" / "colab-gpu-validation.py"
ARTIFACTS = ROOT / "artifacts"
RESULT = ARTIFACTS / "gpu-validation.json"
LOG = ARTIFACTS / "gpu-validation.log"
DATASET = ARTIFACTS / "data" / "sft-nonreasoning"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SFT_OUTPUT = ARTIFACTS / "gpu-validation-sft" / "production"
SELECTION = ARTIFACTS / "gpu-validation-sft" / "selection.json"
MERGED_OUTPUT = ARTIFACTS / "gpu-validation-sft" / "selected-merged"
GRPO_OUTPUT = ARTIFACTS / "gpu-validation-grpo" / "production"
PROFILE_CHOICES = ("auto", "t4", "l4", "a100")

# Every phase, in the order a failure invalidates the least.  `required` phases
# stop the run; the rest are recorded and the matrix continues, because a
# throughput or envelope result is still worth having when a later phase fails.
#
# The CPU suite runs locally in the final verification step. The remote phase only
# runs GPU-marked tests, because duplicating 500+ CPU tests on a 2-vCPU billable VM
# spends the session without adding GPU evidence.
PHASES: tuple[tuple[str, int, bool], ...] = (
    ("probe", 600, True),
    ("kernels", 900, True),
    ("data", 600, True),
    ("sft", 3_000, True),
    ("merge", 2_400, True),
    ("eval-base", 3_000, True),
    ("eval-sft", 3_000, True),
    ("stability", 6_000, False),
    ("grpo", 3_600, True),
    ("non-tty", 600, False),
    ("gpu-tests", 2_400, False),
)


def _write_results(results: list[dict[str, Any]]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _device_identity(device: dict[str, Any]) -> tuple[Any, ...]:
    """Fields that must match before two GPU phase attempts can share an artifact."""
    return (
        device.get("gpu"),
        device.get("compute_capability"),
        device.get("vram_gb"),
        device.get("expected_attention"),
        device.get("expected_dtype"),
    )


def _load_results(*, profile: str, device: dict[str, Any]) -> list[dict[str, Any]]:
    """Load prior rows only when their hardware and model identity still match."""
    if not RESULT.is_file():
        return []
    try:
        loaded = json.loads(RESULT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    # The `run` summary describes one invocation's verdict, so it is dropped on
    # resume; the phase rows are the evidence and are kept.
    rows = [item for item in loaded if isinstance(item, dict) and item.get("name") != "run"]
    expected_device = _device_identity(device)
    for item in rows:
        actual_device = item.get("device")
        if (
            item.get("profile") != profile
            or item.get("model_id") != MODEL_ID
            or item.get("model_revision") != MODEL_REVISION
            or not isinstance(actual_device, dict)
            or _device_identity(actual_device) != expected_device
        ):
            raise RuntimeError(
                "cannot resume gpu-validation.json: existing rows have a different "
                "profile, device, or model identity; move the artifact before starting "
                "a new validation run"
            )
    return rows


def _tail(output: str, limit: int = 12_000) -> str:
    return output[-limit:]


def _is_oom(returncode: int, output: str) -> bool:
    return returncode in {-9, 137} or "out of memory" in output.lower()


def _runtime_python() -> str:
    """Use the synced venv, falling back to the Colab kernel for bootstrap."""
    return str(PYTHON if PYTHON.is_file() else Path(sys.executable))


def _runtime_environment() -> dict[str, str]:
    """Expose the synced venv's console scripts to child processes."""
    environment = os.environ.copy()
    if PYTHON.is_file():
        environment["PATH"] = os.pathsep.join((str(PYTHON.parent), environment.get("PATH", "")))
    return environment


def _run_command(command: list[str], *, timeout: int) -> str:
    """Run one phase command, streaming output and quiet heartbeats."""
    result = run_streaming(
        command,
        name=Path(command[0]).name,
        cwd=ROOT,
        timeout=timeout,
        env=_runtime_environment(),
    )
    if result.returncode != 0:
        executable = command[1] if len(command) > 1 else command[0]
        raise RuntimeError(f"{executable} exited {result.returncode}")
    return result.output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_child(
    results: list[dict[str, Any]],
    name: str,
    args: list[str],
    *,
    timeout: int,
    run_id: str,
    profile: str,
    device: dict[str, Any],
) -> str:
    print(f"\n=== {name} ===", flush=True)
    started = time.monotonic()
    try:
        result = run_streaming(
            [_runtime_python(), str(SCRIPT), *args],
            name=name,
            cwd=ROOT,
            timeout=timeout,
            env=_runtime_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        item = {
            "name": name,
            "run_id": run_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "profile": profile,
            "device": device,
            "status": "timeout",
            "timeout_s": timeout,
            "duration_s": round(time.monotonic() - started, 2),
            "output_tail": _tail(output),
        }
        results.append(item)
        _write_results(results)
        print(item["output_tail"], flush=True)
        return "timeout"

    output = result.output
    if result.returncode == 0:
        status = "passed"
    elif _is_oom(result.returncode, output):
        status = "oom"
    else:
        status = "failed"
    item = {
        "name": name,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "profile": profile,
        "device": device,
        "status": status,
        "returncode": result.returncode,
        "duration_s": round(result.duration_s, 2),
        "output_tail": _tail(output),
    }
    results.append(item)
    _write_results(results)
    return status


def _controller(only: tuple[str, ...] = (), *, profile: str = "auto") -> int:
    device = _device_info()
    profile_name = _effective_profile(profile)
    run_id = f"{time.time_ns()}-{os.getpid()}"
    results = _load_results(profile=profile_name, device=device)
    _write_results(results)
    failures: list[str] = []
    selected = [entry for entry in PHASES if not only or entry[0] in only]

    for phase, timeout, required in selected:
        status = _run_child(
            results,
            phase,
            ["--phase", phase, "--profile", profile_name],
            timeout=timeout,
            run_id=run_id,
            profile=profile_name,
            device=device,
        )
        if status == "passed":
            continue
        failures.append(phase)
        if required:
            results.append(
                {
                    "name": "run",
                    "run_id": run_id,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "profile": profile_name,
                    "device": device,
                    "status": "failed",
                    "reason": f"required phase failed: {phase}",
                }
            )
            _write_results(results)
            print(f"\nRESULT_FILE={RESULT}", flush=True)
            return 1

    results.append(
        {
            "name": "run",
            "run_id": run_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "profile": profile_name,
            "device": device,
            "status": "passed" if not failures else "passed-with-failures",
            "phases": [phase for phase, _, _ in selected],
            "failed_phases": failures,
        }
    )
    _write_results(results)
    print(f"\nRESULT_FILE={RESULT}", flush=True)
    return 0 if not failures else 1


def _launch_controller(only: tuple[str, ...] = (), *, profile: str = "auto") -> int:
    if not ROOT.is_dir() or not SCRIPT.is_file():
        print(f"remote source is not prepared at {ROOT}", file=sys.stderr)
        return 1
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_handle = LOG.open("a", encoding="utf-8")
    command = [sys.executable, str(SCRIPT), "--controller"]
    if only:
        command += ["--only", ",".join(only)]
    command += ["--profile", profile]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"},
    )
    log_handle.close()
    print(f"GPU_VALIDATION_PID={process.pid}")
    print(f"GPU_VALIDATION_RESULT={RESULT}")
    print(f"GPU_VALIDATION_LOG={LOG}")
    return 0


def _device_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    major, minor = torch.cuda.get_device_capability(0)
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "vram_gb": round(properties.total_memory / 1024**3, 3),
        "torch": torch.__version__,
        # Below sm80 there is no FA2 and no bf16 tensor core, so the pipeline runs
        # its padded FP16 sdpa path.  Recorded here so every later phase's numbers
        # are read against the path that produced them.
        "expected_attention": "flash_attention_2" if (major, minor) >= (8, 0) else "sdpa",
        "expected_dtype": "bfloat16" if (major, minor) >= (8, 0) else "float16",
    }


def _effective_profile(requested: str) -> str:
    """Resolve the sizing profile for this card, unless the caller chose one."""
    if requested != "auto":
        return requested
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot auto-select a GPU profile")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 0):
        return "t4"
    if "a100" in torch.cuda.get_device_name(0).casefold():
        return "a100"
    return "l4"


def _phase_probe(requested_profile: str) -> int:
    import importlib.metadata

    info = _device_info()
    profile_name = _effective_profile(requested_profile)
    from smolqwen.config import resolve
    from smolqwen.config_models import SftConfig
    from smolqwen.training.sft import resolve_sft_runtime

    config = resolve("sft", profile=profile_name)
    assert isinstance(config, SftConfig)
    runtime = resolve_sft_runtime(config, require_cuda=True, require_kernels=True)
    if runtime.attention.name != info["expected_attention"]:
        raise RuntimeError(
            f"runtime chose {runtime.attention.name}, expected {info['expected_attention']}"
        )
    if runtime.dtype_name != info["expected_dtype"]:
        raise RuntimeError(f"runtime chose {runtime.dtype_name}, expected {info['expected_dtype']}")

    versions: dict[str, str] = {}
    for package in (
        "torch",
        "vllm",
        "transformers",
        "trl",
        "causal-conv1d",
        "flash-linear-attention",
        "liger-kernel",
        "flash-attn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    print(
        json.dumps(
            {
                "device": info,
                "profile": profile_name,
                "versions": versions,
                "runtime": {
                    "attention": runtime.attention.name,
                    "dtype": runtime.dtype_name,
                    "padding_free": runtime.padding_free,
                    "attention_detail": runtime.attention.detail,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _finite(tensor: Any, label: str) -> None:
    import torch

    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{label} produced non-finite values")


def _phase_kernels() -> int:
    """Exercise each mandatory kernel on the device, forward and backward.

    Importable and ABI-compatible are different claims, and a kernel that only
    forwards is useless to a trainer.
    """
    import torch

    info = _device_info()
    dtype = torch.bfloat16 if info["expected_dtype"] == "bfloat16" else torch.float16

    if info["expected_attention"] == "flash_attention_2":
        from flash_attn import flash_attn_func

        query = torch.randn((1, 128, 8, 64), device="cuda", dtype=dtype, requires_grad=True)
        output = flash_attn_func(query, query, query, causal=True)
        _finite(output, "flash_attn")
        output.float().square().mean().backward()
    else:
        query = torch.randn((1, 8, 128, 256), device="cuda", dtype=dtype, requires_grad=True)
        key = torch.randn((1, 8, 128, 256), device="cuda", dtype=dtype, requires_grad=True)
        output = torch.nn.functional.scaled_dot_product_attention(query, key, key, is_causal=True)
        _finite(output, "sdpa")
        output.float().square().mean().backward()
    torch.cuda.synchronize()
    print(f"{info['expected_attention']} fwd+bwd ok", flush=True)

    from causal_conv1d import causal_conv1d_fn

    features = torch.randn((1, 16, 128), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((16, 4), device="cuda", dtype=dtype, requires_grad=True)
    conv = causal_conv1d_fn(features, weight)
    _finite(conv, "causal_conv1d")
    conv.float().square().mean().backward()
    torch.cuda.synchronize()
    print("causal_conv1d fwd+bwd ok", flush=True)

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # `use_qk_l2norm_in_kernel=True` is not optional decoration: `modeling_qwen3_5`
    # passes it at both gated-delta-rule call sites, and it is what bounds ||q|| and
    # ||k|| to 1 before the kernel's intra-chunk accumulation. Without it, FP16 keys
    # of unit variance combined with a weak decay (exp(g) near 1, so the recurrence
    # keeps its state across the whole chunk) overflow FP16 and the output is NaN.
    # MEASURED on a T4: 0/5 seeds finite without it at weak decay, 5/5 with it, and
    # 5/5 in FP32 either way. So probing without it tests a shape the model never
    # executes and fails a card the model runs fine on.
    shape = (1, 128, 4, 64)
    query = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    gate = torch.rand(shape[:3], device="cuda", dtype=torch.float32).log()
    beta = torch.rand(shape[:3], device="cuda", dtype=dtype, requires_grad=True)
    gdn = chunk_gated_delta_rule(query, key, value, g=gate, beta=beta, use_qk_l2norm_in_kernel=True)
    tensors = gdn if isinstance(gdn, (tuple, list)) else (gdn,)
    loss = sum(item.float().mean() for item in tensors if isinstance(item, torch.Tensor))
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError("flash_linear_attention returned no tensor")
    _finite(loss, "flash_linear_attention")
    loss.backward()
    torch.cuda.synchronize()
    print("flash_linear_attention fwd+bwd ok", flush=True)

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    hidden = torch.randn((8, 128), device="cuda", dtype=dtype, requires_grad=True)
    projection = torch.randn((1_000, 128), device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 1_000, (8,), device="cuda")
    ce = LigerFusedLinearCrossEntropyLoss()(projection, hidden, labels)
    _finite(ce, "liger_kernel")
    ce.backward()
    torch.cuda.synchronize()
    print("liger_kernel fwd+bwd ok", flush=True)
    return 0


def _phase_data(requested_profile: str) -> int:
    """Validate the real non-reasoning shard against the resolved train envelope."""
    from smolqwen.config import resolve
    from smolqwen.config_models import SftConfig
    from smolqwen.data.convert_sft import SFT_SEMANTICS_NON_REASONING
    from smolqwen.training.sft import iter_records, validate_shard

    profile_name = _effective_profile(requested_profile)
    config = resolve("sft", profile=profile_name)
    if not isinstance(config, SftConfig):
        raise RuntimeError(f"expected SftConfig, got {type(config).__name__}")
    if not (DATASET / "train.jsonl").is_file():
        raise RuntimeError(f"non-reasoning smoke shard is incomplete: {DATASET}")
    train = validate_shard(
        DATASET / "train.jsonl",
        label="train",
        max_sequence_length=config.profile.max_seq_length,
        max_tokens_per_microbatch=config.profile.max_tokens_per_microbatch,
    )
    semantics = {str(record.get("semantics")) for record in iter_records(DATASET / "train.jsonl")}
    if semantics != {SFT_SEMANTICS_NON_REASONING}:
        raise RuntimeError(f"expected only non-reasoning samples, found {sorted(semantics)}")
    payload = {
        "dataset": str(DATASET),
        "semantics": sorted(semantics),
        "profile": profile_name,
        "max_seq_length": config.profile.max_seq_length,
        "max_tokens_per_microbatch": config.profile.max_tokens_per_microbatch,
        "micro_batch": config.profile.micro_batch,
        "grad_accum": config.profile.grad_accum,
        "train": train.__dict__,
    }
    _write_json(ARTIFACTS / "data-validation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _phase_gpu_tests() -> int:
    """Run only tests that require the assigned CUDA/vLLM runtime."""
    import importlib.util

    if importlib.util.find_spec("pytest") is None:
        _run_command(["uv", "sync", "--locked", "--extra", "colab"], timeout=900)
    _run_command(
        [str(PYTHON), "-m", "pytest", "-m", "gpu", "-q", "--timeout", "600"],
        timeout=2_200,
    )
    return 0


def _training_runtime(output: Path, output_text: str) -> float | None:
    """Read Trainer's measured runtime, with wall time as the fallback."""
    states = [output / "trainer_state.json"]
    if output.is_dir():
        states.extend(output.rglob("trainer_state.json"))
    for state in states:
        try:
            payload = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in reversed(payload.get("log_history", [])):
            value = row.get("train_runtime") if isinstance(row, dict) else None
            if isinstance(value, (int, float)):
                return float(value)
    matches = re.findall(r"train_runtime[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)", output_text)
    return float(matches[-1]) if matches else None


def _phase_sft_candidate(
    requested_profile: str, *, compile_enabled: bool, label: str, output: Path
) -> dict[str, Any]:
    """Run one full-shape SFT step for a tunable optimization candidate."""
    info = _device_info()
    profile_name = _effective_profile(requested_profile)
    from smolqwen.config import resolve

    resolved = resolve("sft", profile=profile_name)
    profile = resolved.profile
    dataset = DATASET
    if not (dataset / "train.jsonl").is_file():
        raise RuntimeError(f"non-reasoning smoke shard is missing: {dataset}")
    lengths = [
        len(json.loads(line)["input_ids"])
        for line in (dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if output.exists():
        shutil.rmtree(output)
    command = [
        str(CLI),
        "train-sft",
        "--profile",
        profile_name,
        "--override",
        f"model_id={MODEL_ID}",
        "--override",
        f"model_revision={MODEL_REVISION}",
        "--override",
        f"dataset_dir={dataset}",
        "--override",
        f"output_dir={output}",
        "--override",
        "training.max_steps=1",
        "--override",
        f"optimization.regional_torch_compile={'true' if compile_enabled else 'false'}",
    ]
    started = time.monotonic()
    output_text = _run_command(command, timeout=2_700)
    wall_s = time.monotonic() - started
    if info["expected_attention"] not in output_text:
        raise RuntimeError(f"SFT output did not record {info['expected_attention']}")
    if info["expected_dtype"] not in output_text:
        raise RuntimeError(f"SFT output did not record {info['expected_dtype']}")
    saved = list(output.rglob("*.safetensors")) if output.is_dir() else []
    if not saved:
        raise RuntimeError(f"train-sft did not save an adapter under {output}")
    payload = {
        "label": label,
        "regional_torch_compile": compile_enabled,
        "dataset": str(dataset),
        "lengths": lengths,
        "output": str(output),
        "saved_files": len(saved),
        "profile": profile_name,
        "max_seq_length": profile.max_seq_length,
        "max_tokens_per_microbatch": profile.max_tokens_per_microbatch,
        "micro_batch": profile.micro_batch,
        "grad_accum": profile.grad_accum,
        "wall_s": round(wall_s, 2),
        "train_runtime_s": _training_runtime(output, output_text),
    }
    _write_json(output / "validation-timing.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def _phase_sft(requested_profile: str) -> int:
    """Run the selected actual-shape SFT configuration.

    The compile-on comparison was already measured on the same L4 workload and
    took longer; keeping it out of this rerun avoids spending another VM on a known
    losing candidate whose Inductor workers can outlive the Colab websocket.
    """
    result = _phase_sft_candidate(
        requested_profile,
        compile_enabled=False,
        label="compile-off",
        output=SFT_OUTPUT,
    )
    payload = {
        "candidates": [result],
        "selected_label": result["label"],
        "selected_output": result["output"],
        "selection_metric": "train_runtime_s_or_wall_s",
        "selection_reason": "compile-off won the prior same-shape L4 comparison; "
        "compile-on rerun was stopped by VM reclaim before writing an artifact",
    }
    _write_json(SELECTION, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _small_eval_config() -> Path:
    """Write a trimmed copy of the base eval config.

    `adapter_options` is an opaque mapping by design, so `--override` cannot reach
    into it (`config.py` refuses a path through a non-section). A written config is
    the supported way to shrink the adapter's own selection, and `--config` replaces
    the base file, so this starts from that file rather than from defaults.
    """
    import yaml

    base = yaml.safe_load((ROOT / "configs" / "base" / "eval.yaml").read_text(encoding="utf-8"))
    # The shipped benchmark is BFCL multi-turn base; keep the config's own options
    # and shrink the run instead (2 turns, tiny generation budget).
    base["adapters"] = ["bfcl_multi_turn"]
    base["adapter_options"] = {"bfcl_multi_turn": dict(base["adapter_options"]["bfcl_multi_turn"])}
    base["max_steps_per_task"] = 2
    base["enable_thinking"] = False
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "gpu-validation-eval.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    return path


def _selected_output() -> Path:
    """Return the adapter selected by the measured SFT candidate comparison."""
    if not SELECTION.is_file():
        raise RuntimeError(f"SFT selection is missing: {SELECTION}")
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    output = Path(str(payload["selected_output"]))
    if not output.is_dir():
        raise RuntimeError(f"selected SFT output is missing: {output}")
    return output


def _evaluate_checkpoint(
    requested_profile: str,
    *,
    checkpoint: str | Path,
    tag: str,
    output: Path,
    extra_overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run the same non-reasoning, two-turn eval against one checkpoint."""
    profile_name = _effective_profile(requested_profile)
    command = [
        str(CLI),
        "evaluate",
        "--profile",
        profile_name,
        "--config",
        str(_small_eval_config()),
        "--adapter",
        "bfcl_multi_turn",
        "--checkpoint",
        str(checkpoint),
        "--revision",
        MODEL_REVISION,
        "--tag",
        tag,
        "--override",
        f"output_dir={output}",
        # Solo evaluation needs a larger KV reservation than the colocated trainer.
        "--override",
        "profile.vllm_kv_fraction=0.80",
    ]
    for override in extra_overrides:
        command.extend(("--override", override))
    output_text = _run_command(command, timeout=3_000)

    report_line = next(
        (line for line in reversed(output_text.splitlines()) if line.startswith('{"json"')),
        None,
    )
    if report_line is None:
        raise RuntimeError("evaluate printed no report path")
    report_path = Path(json.loads(report_line)["json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded = report["manifest"]["recorded_free"]
    if recorded.get("generation_path") != "vllm":
        raise RuntimeError(f"evaluate ran the {recorded.get('generation_path')} path, not vllm")
    missing = [
        field
        for field in ("dtype", "kv_budget", "max_num_seqs", "max_num_batched_tokens")
        if recorded.get(field) is None
    ]
    if missing:
        raise RuntimeError(f"the engine recorded no serving config for {missing}")
    metrics = report["metrics"]["bfcl_multi_turn"]
    if not metrics.get("average_generated_tokens"):
        raise RuntimeError("every episode generated nothing; the prompt exceeded the window")
    result = {
        "tag": tag,
        "report": str(report_path),
        "invariant": report["manifest"]["invariant"],
        "recorded_free": recorded,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _phase_eval_base(requested_profile: str) -> int:
    """Run the re-captured Base reference on the real non-reasoning path."""
    result = _evaluate_checkpoint(
        requested_profile,
        checkpoint=MODEL_ID,
        tag="base",
        output=ARTIFACTS / "evaluation-production",
    )
    _write_json(ARTIFACTS / "evaluation-production" / "base-summary.json", result)
    return 0


def _phase_eval_sft(requested_profile: str) -> int:
    """Run the selected merged SFT checkpoint with the same eval manifest."""
    result = _evaluate_checkpoint(
        requested_profile,
        checkpoint=MERGED_OUTPUT,
        tag="sft",
        output=ARTIFACTS / "evaluation-production",
    )
    _write_json(ARTIFACTS / "evaluation-production" / "sft-summary.json", result)
    return 0


def _phase_stability(requested_profile: str) -> int:
    """Compare the same task at concurrency 1 and the actual L4 width."""
    base = MODEL_ID
    output = ARTIFACTS / "evaluation-stability"
    serial = _evaluate_checkpoint(
        requested_profile,
        checkpoint=base,
        tag="base-c1",
        output=output,
        extra_overrides=("profile.generation_concurrency=1",),
    )
    batched = _evaluate_checkpoint(
        requested_profile,
        checkpoint=base,
        tag="base-c8",
        output=output,
        extra_overrides=("profile.generation_concurrency=8",),
    )
    payload = {
        "serial": serial,
        "batched": batched,
        "score_delta": batched["metrics"].get("score", 0.0) - serial["metrics"].get("score", 0.0),
    }
    _write_json(ARTIFACTS / "evaluation-stability.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _phase_merge(requested_profile: str) -> int:
    """Merge the selected adapter while preserving Qwen3.5's vLLM wrapper shape."""
    selected = _selected_output()
    if MERGED_OUTPUT.exists():
        shutil.rmtree(MERGED_OUTPUT)
    command = [
        str(CLI),
        "merge-adapter",
        "--profile",
        _effective_profile(requested_profile),
        "--adapter-dir",
        str(selected),
        "--output-dir",
        str(MERGED_OUTPUT),
        "--override",
        f"model_id={MODEL_ID}",
        "--override",
        f"model_revision={MODEL_REVISION}",
    ]
    _run_command(command, timeout=2_300)
    config_path = MERGED_OUTPUT / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"merged config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError(f"merged checkpoint has wrong model_type: {config.get('model_type')}")
    required = (
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    missing = [name for name in required if not (MERGED_OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"merged multimodal wrapper metadata missing: {missing}")
    payload = {
        "selected_adapter": str(selected),
        "merged_output": str(MERGED_OUTPUT),
        "model_type": config.get("model_type"),
        "processor_files": required,
    }
    _write_json(ARTIFACTS / "merge-validation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _phase_grpo(requested_profile: str) -> int:
    """One actual GRPO optimizer step with the real rollout group and dev callback."""
    if GRPO_OUTPUT.exists():
        shutil.rmtree(GRPO_OUTPUT)
    command = [
        str(CLI),
        "train-grpo",
        "--profile",
        _effective_profile(requested_profile),
        "--override",
        f"model_id={MERGED_OUTPUT}",
        "--override",
        f"model_revision={MODEL_REVISION}",
        "--override",
        f"output_dir={GRPO_OUTPUT}",
        "--override",
        "training.max_steps=1",
        "--override",
        "training.save_steps=1",
        "--override",
        "enable_thinking=false",
        "--override",
        "vllm_enable_sleep_mode=true",
        "--override",
        "bench_eval.enabled=true",
        "--override",
        "bench_eval.task_limit=1",
        "--override",
        "bench_eval.timeout_s=900",
        "--override",
        # The first actual-profile attempt OOMed in TRL's old-policy logits at
        # micro_batch=2. Keep all sequence and rollout widths fixed; lower only the
        # registered GRPO fallback axis so the step can be measured end to end.
        "profile.micro_batch=1",
    ]
    _run_command(command, timeout=3_500)
    outcomes = []
    if GRPO_OUTPUT.is_dir():
        outcomes.extend(GRPO_OUTPUT.rglob("bench_eval.json"))
        outcomes.extend((GRPO_OUTPUT / "bench-eval").glob("step-*.json"))
    outcomes.sort()
    if not outcomes:
        raise RuntimeError("GRPO produced no bench_eval.json sidecar")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in outcomes]
    failed = [row for row in rows if row.get("failed_reason")]
    if failed:
        raise RuntimeError(f"GRPO benchmark callback failed: {failed[0]['failed_reason']}")
    payload = {"outcomes": rows, "sidecars": [str(path) for path in outcomes]}
    _write_json(ARTIFACTS / "grpo-validation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _phase_non_tty() -> int:
    """Exercise the plain-line fallback with stdout connected to a pipe."""
    result = run_streaming(
        [str(CLI), "probe", "--no-write"],
        name="non-tty probe",
        cwd=ROOT,
        timeout=300,
        env=_runtime_environment(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"non-TTY probe exited {result.returncode}: {result.output_tail}")
    if "\r" in result.output:
        raise RuntimeError("non-TTY output contains redraw carriage returns")
    payload = {"stdout_lines": len(result.output.splitlines()), "redraw_returns": False}
    _write_json(ARTIFACTS / "non-tty-validation.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--phase", choices=tuple(name for name, _, _ in PHASES))
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="auto",
        help="sizing profile; auto selects from the assigned GPU (default: auto)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated phase names to run; the rest are left untouched",
    )
    args, _unknown = parser.parse_known_args()
    known = {name for name, _, _ in PHASES}
    only = tuple(name for name in args.only.split(",") if name)
    unknown = sorted(set(only) - known)
    if unknown:
        parser.error(f"unknown phase(s): {', '.join(unknown)}")
    if args.controller:
        return _controller(only, profile=args.profile)
    phases = {
        "probe": lambda: _phase_probe(args.profile),
        "kernels": _phase_kernels,
        "data": lambda: _phase_data(args.profile),
        "sft": lambda: _phase_sft(args.profile),
        "merge": lambda: _phase_merge(args.profile),
        "eval-base": lambda: _phase_eval_base(args.profile),
        "eval-sft": lambda: _phase_eval_sft(args.profile),
        "stability": lambda: _phase_stability(args.profile),
        "grpo": lambda: _phase_grpo(args.profile),
        "non-tty": _phase_non_tty,
        "gpu-tests": _phase_gpu_tests,
    }
    if args.phase is not None:
        return phases[args.phase]()
    return _launch_controller(only, profile=args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
