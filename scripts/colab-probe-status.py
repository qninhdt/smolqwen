"""Read-only status probe for a running Colab benchmark.

Prints GPU utilization, the live child process (if any), and the tail of the
results file so a detached controller can tell "still computing" from "hung"
without opening a second Python kernel on the VM.
"""

from __future__ import annotations

import subprocess

COMMANDS = (
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
    "ps -eo pid,etime,pcpu,cmd | grep 'sft-speed.py --child' | grep -v grep || echo NO_CHILD",
    "ps -eo pid,etime,cmd | grep 'sft-speed.py' | grep -v grep | grep -v child || echo NO_PARENT",
    "ls -la /content/*.json 2>/dev/null || echo NO_JSON",
)

for command in COMMANDS:
    completed = subprocess.run(  # noqa: S602 - fixed read-only probe strings
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"$ {command}")
    print((completed.stdout or "") + (completed.stderr or ""), flush=True)
