"""The system prompts the released trajectories were generated against.

Reproduced byte-for-byte from `third_party/EnvScaler/interact_with_env/agent/
system_prompt_util.py` rather than reworded, and asserted against the release file
itself by `tests/test_system_prompt_parity.py`.

Why this module exists at package root rather than inside `eval/`: the same two
strings are the prompt SFT trained under (every trajectory's `messages[0]`), the
prompt evaluation must present, and the prompt Phase 6 rollout must present. Three
copies would let the training distribution and the evaluation distribution drift
by exactly the amount of the paraphrase — which is the first item on Phase 5's
diagnosis list when SFT fails to beat Base, and it would be invisible in any
number the harness prints.

Mode selection follows upstream's `task_solve_agent.reset`:

- `bfcl` and `envscaler_non_conversation_*` -> `NON_CONVERSATIONAL`
- `envscaler_conversation_*`, tau-bench, ACEBench multi-turn -> `CONVERSATIONAL`

BFCL contributes no environment introduction (`bfcl_env/env.py:328` sets it to
`""`), so its system prompt is the bare non-conversational string.
"""

from __future__ import annotations

import hashlib

# Trailing whitespace in both literals is upstream's and is load-bearing: it is in
# the bytes the model was trained on, so `ruff` must not be allowed to trim it.
# fmt: off
NON_CONVERSATIONAL = (
    "You are a helpful assistant. When given a specific task, your goal is to "
    "complete it in an interactive environment by making step-by-step use of "
    "available tools. \n"
    "- Before completing the task, at each step, select a tool from the tool list "
    "and fill in all required parameters, making sure that the values are valid. "
    "Avoid making parallel tool calls in one step.\n"
    "- When you believe the task has been completed, respond only with 'Task "
    "Completed' to end the trajectory, without adding any other content or making "
    "any tool calls.\n"
    "- It is recommended to first call query tools to gather sufficient "
    "information, then use modification tools to complete the task. Adjust actions "
    "promptly based on the feedback from the environment, i.e., the tool results.\n"
)

CONVERSATIONAL = (
    "You are a helpful assistant. Your goal is to fulfill the user's requests in "
    "an interactive environment by step-by-step use of available tools, while "
    "proactively communicating with the user when necessary, until the user ends "
    "the conversation.\n"
    "At each step, you will receive either the user's task/reply or the "
    "environment's tool call result.\n"
    "- If you lack essential information to complete the task or perform a tool "
    "call, and it cannot be obtained through the existing tool set, actively ask "
    "the user for specific details.  \n"
    "- If you can proceed with the current information, select one tool from the "
    "tool set and provide complete, valid parameters. Avoid making parallel tool "
    "calls or calling a tool while interacting with the user in one step.\n"
    "- It is recommended to first call query tools to gather sufficient "
    "information, then use modification tools to complete the task. Adjust actions "
    "promptly based on the feedback from the environment, i.e., the tool results.\n"
    "- When you believe the task is completed, clearly inform the user of the "
    "result and ask whether there are any new tasks or follow-up requests. \n"
)
# fmt: on

# The exact join between the mode prompt and the environment introduction. One
# newline out and every rendered prompt differs from the trained one.
INTRODUCTION_SEPARATOR = "\n\nThe following is an introduction to the current environment:\n"


def build_system_prompt(*, conversational: bool, env_introduction: str = "") -> str:
    """Assemble the system prompt exactly as upstream's agent assembles it."""
    prompt = CONVERSATIONAL if conversational else NON_CONVERSATIONAL
    if env_introduction:
        return f"{prompt}{INTRODUCTION_SEPARATOR}{env_introduction}"
    return prompt


def system_prompt_hash(prompt: str) -> str:
    """Stable hash of a rendered system prompt, for the manifest's invariant set."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
