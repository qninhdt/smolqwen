"""The offline `vllm.LLM` lifecycle: build, generate, sleep, wake, load adapter.

vllm and torch are imported inside the methods that need them. Importing this
module on a machine with neither installed must work -- `--dry-run` depends on it,
and CI has no vllm by construction (`pyproject.toml:33-36`).

**Telemetry.** vLLM's usage-stats collection is on by default and disabled only by
`VLLM_NO_USAGE_STATS`, `VLLM_DO_NOT_TRACK` or `DO_NOT_TRACK`. Zero occurrences
existed repo-wide while CI already set offline flags for HF, transformers and W&B
(`ci.yml:14-18`). An in-training eval puts this engine inside the process holding
the HF token and the W&B session -- the boundary
`tests/test_worker_isolation_secrets.py` exists to defend -- so the flag is set at
construction, before the import, rather than left to the environment.

**Sleep mode.** Level 1 offloads weights to CPU and discards the KV cache, which
is what lets a training process hold an engine across steps without paying for it.
Whether that release is large enough for the trainer's envelope is a runtime
property, measured on a card, not asserted here.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smolqwen.inference.profiles import EvalProfile
from smolqwen.rollout.generation import GenerationRequest, TurnTokens

# Set before `import vllm`, because usage-stats collection is decided at import.
TELEMETRY_ENV = {
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "DO_NOT_TRACK": "1",
}


class EngineError(RuntimeError):
    """Raised when the engine cannot produce a usable, aligned result."""


def disable_telemetry(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Set every variable vLLM honours for opting out. Returns what was set."""
    target = environment if environment is not None else os.environ
    for name, value in TELEMETRY_ENV.items():
        target.setdefault(name, value)
    return {name: target[name] for name in TELEMETRY_ENV}


@dataclass(frozen=True)
class Completion:
    """One prompt's generated text, with the metrics a manifest records.

    `finish_reason` is vLLM's own, not synthesized from a token count. The
    HuggingFace path had to guess (`policies.py:284` compares the generated width
    against `max_new_tokens`), which reports `length` for a completion that
    happened to end exactly at the budget. `truncation_rate` is built on this
    field, so the two paths are expected to differ here and the vLLM one is right.
    """

    text: str
    generated_tokens: int
    finish_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass(frozen=True)
class TokenCompletion:
    """One prompt's sampled token ids and their logprobs, for the turn engine.

    Shaped like `rollout/generation.py`'s `TurnTokens` because the turn engine
    consumes exactly that: token ids to hand the mask builder, and one logprob per
    token. A missing logprob stays NaN rather than shortening the row -- a short row
    would be right-padded downstream and shift every later position.
    """

    episode_id: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class OfflineEngine:
    """An in-process `vllm.LLM`, batched, with sleep/wake and adapter loading.

    Construction order matters wherever a trainer shares the card:
    `gpu_memory_utilization` sizes the KV pool against **total** GPU memory, not
    against what is free. Built after a resident trainer, the engine either OOMs or
    reserves against a figure it cannot honour; built before, the reservation is
    held through training. So a trainer-side caller builds this first, sleeps it,
    and lets the trainer size itself against what remains.
    """

    def __init__(
        self,
        model: str,
        profile: EvalProfile,
        *,
        revision: str | None = None,
        enable_lora: bool = False,
        enable_sleep_mode: bool = False,
    ) -> None:
        self.model = model
        self.profile = profile
        self.revision = revision
        self.enable_lora = enable_lora
        self.enable_sleep_mode = enable_sleep_mode
        self._llm: Any | None = None
        self._sleeping = False
        self._adapters: dict[str, Any] = {}

    def build(self) -> None:
        """Construct the engine. Idempotent, so a callback may call it freely."""
        if self._llm is not None:
            return
        disable_telemetry()
        from vllm import LLM

        self._llm = LLM(
            model=self.model,
            revision=self.revision,
            max_model_len=self.profile.max_model_len,
            gpu_memory_utilization=self.profile.gpu_memory_utilization,
            enforce_eager=self.profile.enforce_eager,
            enable_lora=self.enable_lora,
            max_loras=self.profile.max_lora_slots if self.enable_lora else 1,
            enable_sleep_mode=self.enable_sleep_mode,
        )

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    def _engine(self) -> Any:
        if self._llm is None:
            raise EngineError("engine is not built; call build() first")
        if self._sleeping:
            raise EngineError("engine is asleep; call wake_up() before generating")
        return self._llm

    def sampling_params(self, *, max_new_tokens: int | None = None) -> Any:
        """Greedy by default, from the resolved decoding config.

        `logprobs=0` asks vLLM for the sampled token's own logprob and no
        alternatives, which is what the turn engine needs and the cheapest form of
        the request.
        """
        from vllm import SamplingParams

        profile = self.profile
        return SamplingParams(
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            max_tokens=max_new_tokens or profile.max_new_tokens,
            seed=profile.seed,
            logprobs=0,
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int | None = None,
        adapter: str | None = None,
    ) -> list[Completion]:
        """Generate every prompt in one batched call, positionally aligned.

        vLLM returns outputs in request order, but nothing obliges a caller to
        trust that silently: a misalignment would attribute one task's completion
        to another task's score, and every downstream metric would still look
        plausible. So the count is checked and each output's echoed prompt is
        compared against the prompt at its index.
        """
        if not prompts:
            return []
        engine = self._engine()
        request: dict[str, Any] = {
            "prompts": list(prompts),
            "sampling_params": self.sampling_params(max_new_tokens=max_new_tokens),
        }
        if adapter is not None:
            request["lora_request"] = self._adapter_request(adapter)
        outputs = engine.generate(**request)

        if len(outputs) != len(prompts):
            raise EngineError(
                f"engine returned {len(outputs)} outputs for {len(prompts)} prompts; "
                "positional alignment with tasks cannot be assumed"
            )
        return [_completion(output, index, prompts[index]) for index, output in enumerate(outputs)]

    def generate_ids(
        self,
        prompt_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int | None = None,
        adapter: str | None = None,
    ) -> list[TokenCompletion]:
        """Generate from pre-tokenized prompts, returning token ids and logprobs.

        The turn engine renders and tokenizes itself -- the mask builder needs the
        exact prefix ids the template produced, and re-tokenizing a decoded string
        would let a BPE seam move the boundary. So the engine takes ids in and gives
        ids back; text decoding stays at the one seam `inference/decoding.py` owns.
        """
        if not prompt_ids:
            return []
        engine = self._engine()
        request: dict[str, Any] = {
            "prompts": [{"prompt_token_ids": list(ids)} for ids in prompt_ids],
            "sampling_params": self.sampling_params(max_new_tokens=max_new_tokens),
        }
        if adapter is not None:
            request["lora_request"] = self._adapter_request(adapter)
        outputs = engine.generate(**request)

        if len(outputs) != len(prompt_ids):
            raise EngineError(
                f"engine returned {len(outputs)} outputs for {len(prompt_ids)} prompts; "
                "positional alignment with tasks cannot be assumed"
            )
        return [_token_completion(output, index) for index, output in enumerate(outputs)]

    def sleep(self, level: int = 1) -> None:
        """Offload weights to CPU and discard the KV cache.

        Level 1 keeps the weights on the host, so waking does not re-read the
        checkpoint. How much VRAM this actually returns is a runtime property of
        the pin and the card; a caller that depends on the number measures it with
        `reset_peak_memory_stats()` plus `memory_allocated()`. `memory_reserved()`
        does not shrink without `empty_cache()`, and `max_memory_allocated()` is a
        monotonic high-water mark -- neither can show a release.
        """
        if self._llm is None or self._sleeping:
            return
        if not self.enable_sleep_mode:
            raise EngineError("engine was built without enable_sleep_mode; sleep() is unavailable")
        self._llm.sleep(level=level)
        self._sleeping = True

    def wake_up(self) -> None:
        if self._llm is None or not self._sleeping:
            return
        self._llm.wake_up()
        self._sleeping = False

    def load_adapter(self, name: str, path: str) -> Any:
        """Register a PEFT adapter directory for `generate(adapter=name)`.

        Whether vLLM accepts this project's adapters is a capability, not a gate:
        both training configs use `target_modules: all-linear`, which emits LoRA
        weights for Qwen3.5's Gated DeltaNet mixer projections, and vLLM validates
        adapter weights against a per-architecture allowlist. If it refuses,
        `TransformersPolicy` still evaluates the adapter -- slower, unchanged.

        The raising is deliberate. A silently-ignored adapter serves base output,
        which would make an in-training benchmark curve flat and a cross-check
        agree while both sides measured the wrong weights.
        """
        if not self.enable_lora:
            raise EngineError("engine was built without enable_lora; adapters cannot be loaded")
        from vllm.lora.request import LoRARequest

        # vLLM keys adapters by a positive integer id; the name is for callers.
        request = LoRARequest(name, len(self._adapters) + 1, path)
        self._adapters[name] = request
        return request

    def _adapter_request(self, name: str) -> Any:
        try:
            return self._adapters[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._adapters)) or "none"
            raise EngineError(f"adapter {name!r} was never loaded; loaded: {known}") from exc

    def shutdown(self) -> None:
        self._llm = None
        self._sleeping = False
        self._adapters.clear()

    def __enter__(self) -> OfflineEngine:
        self.build()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


def _completion(output: Any, index: int, prompt: str) -> Completion:
    """One vLLM `RequestOutput` as a `Completion`, with its alignment checked.

    vLLM echoes the prompt it generated for on each output, so comparing that
    against the prompt at this index catches a reordering directly rather than
    trusting a request id whose format is vLLM's to change.
    """
    echoed = getattr(output, "prompt", None)
    if echoed is not None and str(echoed) != prompt:
        raise EngineError(
            f"output at position {index} echoes a different prompt; "
            "prompts and completions are not positionally aligned"
        )
    candidates = getattr(output, "outputs", ())
    if not candidates:
        raise EngineError(f"output at position {index} carries no completion")
    first = candidates[0]
    return Completion(
        text=str(first.text),
        generated_tokens=len(first.token_ids),
        finish_reason=None if first.finish_reason is None else str(first.finish_reason),
    )


def _token_completion(output: Any, index: int) -> TokenCompletion:
    """One vLLM `RequestOutput` as token ids plus per-token logprobs.

    vLLM returns `logprobs` as a list of `{token_id: Logprob}` maps, one per
    position, and only when the sampling params asked for them. A position whose
    sampled token is absent from its own map stays NaN: the alternative, dropping
    it, would shorten the row and shift every later logprob against the wrong token.
    """
    candidates = getattr(output, "outputs", ())
    if not candidates:
        raise EngineError(f"output at position {index} carries no completion")
    first = candidates[0]
    token_ids = tuple(int(token) for token in first.token_ids)
    rows = getattr(first, "logprobs", None) or ()
    logprobs: list[float] = []
    for position, token_id in enumerate(token_ids):
        row = rows[position] if position < len(rows) else None
        entry = row.get(token_id) if isinstance(row, Mapping) else None
        value = getattr(entry, "logprob", entry)
        logprobs.append(float(value) if isinstance(value, int | float) else math.nan)
    return TokenCompletion(
        episode_id=str(index),
        token_ids=token_ids,
        logprobs=tuple(logprobs),
        finish_reason=None if first.finish_reason is None else str(first.finish_reason),
    )


class OfflineEngineBackend:
    """`GenerationBackend` over an `OfflineEngine`, for the shared turn engine.

    The engine speaks prompts-and-completions; the turn engine speaks
    `GenerationRequest`/`TurnTokens`. This is the whole adaptation, and it is a
    class rather than a lambda because it also carries the adapter name: an
    in-training eval scores the checkpoint TRL just wrote, which reaches vLLM as a
    `LoRARequest` registered under a name.

    `max_new_tokens` comes from each request, not from the profile: the turn engine
    computes a per-episode budget from how much context that episode has left, and
    a batch of requests at different depths must not all be generated at the widest
    one's budget.
    """

    def __init__(self, engine: OfflineEngine, *, adapter: str | None = None) -> None:
        self.engine = engine
        self.adapter = adapter

    def generate(self, requests: Sequence[GenerationRequest]) -> list[TurnTokens]:
        if not requests:
            return []
        started = time.monotonic()
        # One batched call at the widest budget, then each row truncated back to its
        # own -- the same trade `VllmColocateBackend` makes, for the same reason: the
        # engine takes one budget per call.
        completions = self.engine.generate_ids(
            [request.prompt_ids for request in requests],
            max_new_tokens=max(request.max_new_tokens for request in requests),
            adapter=self.adapter,
        )
        elapsed = time.monotonic() - started
        return [
            TurnTokens(
                episode_id=request.episode_id,
                token_ids=completion.token_ids[: request.max_new_tokens],
                logprobs=completion.logprobs[: request.max_new_tokens],
                duration_s=elapsed,
                prompt_tokens=len(request.prompt_ids),
            )
            for request, completion in zip(requests, completions, strict=True)
        ]
