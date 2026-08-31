"""Policy boundary shared by local checkpoints and OpenAI-compatible serving."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

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
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        revision = _require_revision_sha(revision)
        if not model:
            raise ValueError("HTTP evaluation requires a served model name")
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
        self._opener = opener

    @property
    def completion_url(self) -> str:
        """Return the chat-completions URL for either a root or ``/v1`` base."""

        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}/chat/completions"
        return f"{self.endpoint}/v1/chat/completions"

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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.completion_url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode())
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


class TransformersPolicy:
    """A local base, merged, or adapter-on-base checkpoint with a pinned revision."""

    def __init__(
        self,
        checkpoint: str,
        *,
        revision: str,
        adapter: str | None = None,
        adapter_revision: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = 1234,
    ) -> None:
        revision = _require_revision_sha(revision)
        if adapter:
            adapter_revision = _require_revision_sha(adapter_revision, label="adapter")
        self.revision = revision
        self.adapter_revision = adapter_revision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("local Transformers evaluation requires CUDA")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model: Any = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision)
        mode_name = "eval"
        self._model = getattr(model, mode_name)()

    def generate(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> GenerationResult:
        rendered: Any = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tools=[dict(tool) for tool in tools],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(rendered, Mapping):
            raise TypeError("chat template output must be a mapping of model inputs")
        model_inputs = {
            str(name): value.to(self._model.device) if hasattr(value, "to") else value
            for name, value in rendered.items()
        }
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("chat template output is missing input_ids")
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "top_p": self.top_p,
        }
        if self.top_k >= 0:
            kwargs["top_k"] = self.top_k
        if self.temperature > 0.0:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            self._torch.manual_seed(self.seed)
        with self._torch.inference_mode():
            output = self._model.generate(**model_inputs, **kwargs)
        generated = output[0, input_ids.shape[-1] :]
        tokens = int(generated.shape[-1])
        return GenerationResult(
            completion=str(self._tokenizer.decode(generated, skip_special_tokens=True)),
            generated_tokens=tokens,
            finish_reason="length" if tokens == self.max_new_tokens else "stop",
        )


def load_policy(
    *,
    checkpoint: str | None,
    revision: str | None,
    endpoint: str | None,
    adapter: str | None,
    adapter_revision: str | None = None,
    model: str = "smolqwen",
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None = 1234,
    http_timeout_s: float = 60.0,
) -> Policy:
    """Select the HTTP, base/merged, or adapter-on-base policy without implicit revisions."""
    revision = _require_revision_sha(revision)
    if endpoint:
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
        )
    if not checkpoint:
        raise ValueError("--checkpoint is required unless --endpoint is supplied")
    return TransformersPolicy(
        checkpoint,
        revision=revision,
        adapter=adapter,
        adapter_revision=adapter_revision,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
