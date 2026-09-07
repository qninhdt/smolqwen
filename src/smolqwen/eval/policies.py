"""Policy boundary shared by local checkpoints and OpenAI-compatible serving."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import urlopen

from smolqwen.inference.client import ChatClient

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def _require_revision_sha(value: str | None, *, label: str = "checkpoint") -> str:
    if not value or _COMMIT_SHA.fullmatch(value) is None:
        raise ValueError(f"evaluation requires an explicit {label} revision sha")
    return value


@dataclass(frozen=True)
class GenerationResult:
    completion: str
    generated_tokens: int
    finish_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class Policy(Protocol):
    revision: str

    def generate(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> GenerationResult: ...


class HttpPolicy:
    """Minimal OpenAI chat-completions policy; metrics come from server usage."""

    def __init__(
        self,
        endpoint: str,
        *,
        revision: str,
        model: str = "smolqwen",
        api_key: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = 1234,
        timeout_s: float = 60.0,
        enable_thinking: bool = True,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        revision = _require_revision_sha(revision)
        if not model:
            raise ValueError("HTTP evaluation requires a served model name")
        self._client = ChatClient(
            endpoint,
            api_key=api_key,
            timeout_s=timeout_s,
            opener=opener,
        )
        self.endpoint = endpoint.rstrip("/")
        self.revision = revision
        self.adapter_revision: str | None = None
        self.model = model
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.timeout_s = timeout_s
        self.enable_thinking = enable_thinking

    @property
    def completion_url(self) -> str:
        """The chat-completions URL, for either a root or a ``/v1`` base."""
        return self._client.completion_url

    def generate(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._openai_messages(messages),
            "tools": list(tools),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k >= 0:
            payload["top_k"] = self.top_k
        if self.seed is not None:
            payload["seed"] = self.seed
        if not self.enable_thinking:
            # Sent only when disabled: strict OpenAI-compatible endpoints reject
            # unknown body fields, and the default (thinking) needs no override.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = self._client.chat(payload)
        choice = body["choices"][0]
        message = choice["message"]
        completion = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            normalized_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": arguments}
                if not isinstance(name, str) or not isinstance(arguments, Mapping):
                    raise ValueError("OpenAI response contains a malformed tool call")
                normalized_calls.append({"name": name, "arguments": dict(arguments)})
            tool_text = json.dumps(
                normalized_calls[0] if len(normalized_calls) == 1 else normalized_calls,
                ensure_ascii=False,
            )
            completion = f"{completion}\n{tool_text}" if completion else tool_text
        return GenerationResult(
            completion=completion,
            generated_tokens=int(body.get("usage", {}).get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason"),
        )

    @staticmethod
    def _openai_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Restore structured tool-call history for strict OpenAI endpoints.

        The runner stores a backend-neutral assistant completion string so local
        Qwen XML and HTTP policies share one trajectory.  On the wire, OpenAI
        requires a tool observation to reference a preceding structured call;
        rebuild that shape from HttpPolicy's final-line JSON representation.
        """

        normalized: list[dict[str, Any]] = []
        pending_ids: list[str] = []
        call_index = 0
        for source in messages:
            message = dict(source)
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                content = str(message["content"])
                prefix, separator, final_line = content.rpartition("\n")
                candidate = final_line.strip() if separator else content.strip()
                try:
                    parsed = json.loads(candidate) if candidate.startswith(("[", "{")) else None
                except json.JSONDecodeError:
                    parsed = None
                calls = parsed if isinstance(parsed, list) else [parsed]
                valid_calls: list[Mapping[str, Any]] = []
                for call in calls:
                    if (
                        not isinstance(call, Mapping)
                        or not isinstance(call.get("name"), str)
                        or not isinstance(call.get("arguments"), Mapping)
                    ):
                        valid_calls = []
                        break
                    valid_calls.append(call)
                if valid_calls:
                    tool_calls: list[dict[str, Any]] = []
                    pending_ids = []
                    for call in valid_calls:
                        call_id = f"call_{call_index}"
                        call_index += 1
                        pending_ids.append(call_id)
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(
                                        dict(call["arguments"]), ensure_ascii=False
                                    ),
                                },
                            }
                        )
                    message["content"] = prefix if separator and prefix else None
                    message["tool_calls"] = tool_calls
            elif message.get("role") == "tool" and pending_ids:
                message["tool_call_id"] = pending_ids.pop(0)
            normalized.append(message)
        return normalized


def load_http_policy(
    *,
    revision: str | None,
    endpoint: str | None,
    model: str = "smolqwen",
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None = 1234,
    http_timeout_s: float = 60.0,
    enable_thinking: bool = True,
) -> Policy:
    """Build the HTTP policy for an explicitly served vLLM endpoint."""
    revision = _require_revision_sha(revision)
    if not endpoint:
        raise ValueError("local evaluation requires the in-process vLLM engine")
    return HttpPolicy(
        endpoint,
        revision=revision,
        model=model,
        api_key=os.environ.get("VLLM_API_KEY"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        timeout_s=http_timeout_s,
        enable_thinking=enable_thinking,
    )
