"""Every vLLM interaction this project makes, behind one boundary.

Three code paths generated tokens before this package existed and none shared an
implementation: `eval/policies.py` drove HuggingFace `generate()` one sequence at a
time, `rollout/generation.py` drove TRL's colocated engine, and
`serving/server.py` shelled out to `vllm serve`. `grep "import vllm" src/`
returned nothing -- the evaluation path never touched vLLM at all.

Importing this package must not import torch or vllm. `--dry-run` resolves and
validates a config on a machine where neither is installed (`cli.py:1-8`), and CI
installs neither by construction, so every heavy import lives inside the function
that needs it.
"""

from __future__ import annotations

from smolqwen.inference.client import ChatClient, ReadinessError, wait_for_readiness
from smolqwen.inference.profiles import EvalProfile, RolloutProfile, ServeProfile

__all__ = [
    "ChatClient",
    "EvalProfile",
    "ReadinessError",
    "RolloutProfile",
    "ServeProfile",
    "wait_for_readiness",
]
