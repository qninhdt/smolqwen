"""`smolqwen probe`: what this GPU can do. Capability only.

This is the machine-readable answer to the A100-variant question, and it is the
one thing every later sizing decision reads. It reports **no throughput and no
cost**: those do not exist until Phase 6's `rollout-bench` measures episodes/hour,
and a probe that guessed at them would get cited as if it had measured.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Kernel libraries whose availability changes which optimizations Phase 3 can
# enable. Import success is reported, never assumed.
KERNEL_LIBRARIES = ("flash_attn", "causal_conv1d", "fla", "liger_kernel")
PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm")

# FP8 is native from sm89 (L4, H100). sm80 (A100) has no FP8 path, which is why
# the two serving profiles differ in quantization strategy rather than batch size.
FP8_MIN_CAPABILITY = (8, 9)


@dataclass
class GpuInfo:
    name: str | None = None
    compute_capability: str | None = None
    total_memory_gb: float | None = None
    multi_processor_count: int | None = None
    supports_fp8: bool = False
    supports_bf16: bool = False


@dataclass
class ProbeReport:
    gpu_available: bool = False
    gpu_count: int = 0
    gpu: GpuInfo = field(default_factory=GpuInfo)
    python_version: str = ""
    platform: str = ""
    package_versions: dict[str, str | None] = field(default_factory=dict)
    kernel_imports: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _kernel_importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def probe() -> ProbeReport:
    """Collect capability facts about the current host."""
    report = ProbeReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_versions={name: _package_version(name) for name in PACKAGES},
        kernel_imports={module: _kernel_importable(module) for module in KERNEL_LIBRARIES},
    )

    try:
        import torch
    except ImportError:
        return report

    if not torch.cuda.is_available():
        return report

    report.gpu_available = True
    report.gpu_count = torch.cuda.device_count()
    major, minor = torch.cuda.get_device_capability(0)
    properties = torch.cuda.get_device_properties(0)
    report.gpu = GpuInfo(
        name=torch.cuda.get_device_name(0),
        compute_capability=f"{major}.{minor}",
        total_memory_gb=round(properties.total_memory / 1024**3, 2),
        multi_processor_count=getattr(properties, "multi_processor_count", None),
        supports_fp8=(major, minor) >= FP8_MIN_CAPABILITY,
        supports_bf16=bool(torch.cuda.is_bf16_supported()),
    )
    return report


def probe_slug(report: ProbeReport) -> str:
    """A stable filename stem, so `artifacts/probe/l4.json` lands predictably."""
    name = report.gpu.name
    if not name:
        return "cpu"
    lowered = name.lower()
    for known in ("a100", "h100", "h200", "l4", "l40s", "l40", "t4", "v100", "a10g"):
        if known in lowered:
            # An A100 exists in 40 GB and 80 GB variants and the two imply
            # different RL batch ceilings, so the variant is part of the name.
            if known == "a100" and report.gpu.total_memory_gb:
                return f"a100-{int(round(report.gpu.total_memory_gb / 10) * 10)}gb"
            return known
    return "".join(ch if ch.isalnum() else "-" for ch in lowered).strip("-")


def write_probe(report: ProbeReport, output_dir: Path | str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{probe_slug(report)}.json"
    path.write_text(report.to_json() + "\n", encoding="utf-8")
    return path


def format_probe(report: ProbeReport) -> str:
    lines = [
        f"python            {report.python_version}",
        f"platform          {report.platform}",
        f"gpu available     {report.gpu_available}",
    ]
    if report.gpu_available:
        lines += [
            f"gpu               {report.gpu.name} (x{report.gpu_count})",
            f"compute cap       {report.gpu.compute_capability}",
            f"total vram        {report.gpu.total_memory_gb} GB",
            f"fp8 native        {report.gpu.supports_fp8}",
            f"bf16              {report.gpu.supports_bf16}",
        ]
    lines.append("packages")
    for name, value in sorted(report.package_versions.items()):
        lines.append(f"  {name:16} {value or 'not installed'}")
    lines.append("kernels")
    for name, importable in sorted(report.kernel_imports.items()):
        lines.append(f"  {name:16} {'ok' if importable else 'missing'}")
    return "\n".join(lines)


def probe_payload() -> dict[str, Any]:
    return asdict(probe())
