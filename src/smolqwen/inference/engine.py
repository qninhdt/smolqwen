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

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from smolqwen.config_models import LoraConfig
from smolqwen.inference.profiles import EvalProfile
from smolqwen.rollout.generation import GenerationRequest, TurnTokens

# Set before `import vllm`, because usage-stats collection is decided at import.
TELEMETRY_ENV = {
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "DO_NOT_TRACK": "1",
}

# The values vLLM's `LoRAConfig.max_lora_rank` accepts (`vllm/config/lora.py:29`).
# It is a validated pydantic field, so an arbitrary rank is rejected at engine
# construction and a rank has to be rounded up to one of these.
VLLM_LORA_RANKS: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256, 320, 512)

# Used when no adapter directory is available to read a rank from -- SFT's first
# boundary, before TRL has written a checkpoint. Read from the model rather than
# repeated, so raising `lora.r` moves the reservation with it; vLLM's own default of
# 16 refuses the 32 both training configs set.
DEFAULT_LORA_RANK = LoraConfig().r

# A non-zero adapter can leave one short greedy continuation unchanged by chance.
# Keep this panel deterministic, small, and varied in both token values and prompt
# lengths so the lazy accepted-but-no-op case is less likely to be mistaken for a
# valid adapter. This remains a guard, not a proof of semantic quality.
_ADAPTER_PROBE_PROMPTS = (
    (1, 2, 3, 4),
    (1, 3, 5, 7, 9),
    (2, 4, 8, 16, 32, 64),
    (3, 9, 27, 81),
    (5, 10, 15, 20, 25, 30, 35),
    (6, 12, 18, 24, 30, 36, 42, 48),
    (7, 14, 21, 28, 35),
    (8, 16, 24, 32, 40, 48, 56),
)


class EngineError(RuntimeError):
    """Raised when the engine cannot produce a usable, aligned result."""


class AdapterCapabilityError(EngineError):
    """Raised when vLLM cannot apply an adapter without silently ignoring it."""


class MemoryWorkerExtension:
    """Small worker-side RPC surface for reading the vLLM CUDA allocator."""

    def read_cuda_memory_allocated(self, *, reset_peak: bool = False) -> int:
        import torch

        torch.cuda.synchronize()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        # vLLM sleep mode uses CuMemAllocator, a pluggable CUDA allocator. Its
        # unmapped allocations remain visible to `torch.cuda.memory_allocated()`
        # even after sleep, so that counter cannot measure the release. Driver
        # free-memory accounting observes the actual mapped footprint instead.
        free, total = torch.cuda.mem_get_info()
        return int(total - free)


def disable_telemetry(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Set every variable vLLM honours for opting out. Returns what was set."""
    target = environment if environment is not None else os.environ
    for name, value in TELEMETRY_ENV.items():
        target.setdefault(name, value)
    return {name: target[name] for name in TELEMETRY_ENV}


def lora_rank_slot(rank: int) -> int:
    """The smallest `max_lora_rank` vLLM accepts that still holds `rank`.

    `LoRAConfig.max_lora_rank` is a validated `Literal`, not a free integer, so a
    trained rank cannot be passed through unrounded. Rounding *up* is what keeps
    the adapter loadable; rounding down would reproduce the failure this exists to
    prevent, which is vLLM's default of 16 refusing this project's `r: 32`.
    """
    if rank < 1:
        raise EngineError(f"LoRA rank must be positive; got {rank}")
    for slot in VLLM_LORA_RANKS:
        if slot >= rank:
            return slot
    raise EngineError(
        f"LoRA rank {rank} exceeds the largest rank vLLM supports ({VLLM_LORA_RANKS[-1]})"
    )


def adapter_rank(path: str | Path) -> int | None:
    """The `r` a PEFT adapter directory records, or None when it cannot be read.

    Read from the artifact rather than taken from a config field, because the
    artifact is what vLLM validates: an adapter trained at one rank and an engine
    sized from a since-edited config is exactly the mismatch that fails at the
    first generation. None means "unreadable", which is left to vLLM to report --
    an adapter without a loadable `adapter_config.json` is not loadable at all.
    """
    config_path = Path(path) / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rank = payload.get("r") if isinstance(payload, Mapping) else None
    return int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else None


@dataclass(frozen=True)
class Completion:
    """One prompt's generated text, with the metrics a manifest records.

    `finish_reason` is vLLM's own, not synthesized from a token count. A completion
    can end exactly at the budget without being truncated, so `truncation_rate` must
    use the engine's reason rather than infer one from width.
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
        max_lora_rank: int | None = None,
    ) -> None:
        self.model = model
        self.profile = profile
        self.revision = revision
        self.enable_lora = enable_lora
        self.enable_sleep_mode = enable_sleep_mode
        # The rank the engine reserves adapter slots for. vLLM's own default is 16
        # and this project trains at 32, so leaving it unset makes every adapter
        # load raise `LoRA rank 32 is greater than max_lora_rank 16`. Sized from
        # the adapter that will be loaded when one is known; otherwise from the
        # largest rank the configs use.
        self.max_lora_rank = max_lora_rank
        self._llm: Any | None = None
        self._sleeping = False
        self._adapters: dict[str, Any] = {}
        self._adapter_tempdirs: list[Any] = []

    def build(self) -> None:
        """Construct the engine. Idempotent, so a callback may call it freely."""
        if self._llm is not None:
            return
        disable_telemetry()
        from vllm import LLM

        from smolqwen.inference.profiles import resolve_dtype

        # `EvalProfile.from_config()` already resolves this, but direct callers
        # can construct the dataclass with its bf16 default. Resolve at the final
        # hardware boundary too: vLLM rejects bf16 on Turing rather than emulating
        # it, so a direct `OfflineEngine` must be as portable as the CLI path.
        resolved_dtype = resolve_dtype(self.profile.dtype)
        if resolved_dtype != self.profile.dtype:
            self.profile = replace(self.profile, dtype=resolved_dtype)

        lora_kwargs: dict[str, Any] = {}
        if self.enable_lora:
            # Passed only on the LoRA path: vLLM builds no `LoRAConfig` without
            # `enable_lora`, and `max_lora_rank` would then be an unread argument
            # whose value nothing validates.
            lora_kwargs["max_lora_rank"] = lora_rank_slot(
                self.max_lora_rank if self.max_lora_rank is not None else DEFAULT_LORA_RANK
            )

        self._llm = LLM(
            model=self.model,
            revision=self.revision,
            max_model_len=self.profile.max_model_len,
            gpu_memory_utilization=self.profile.gpu_memory_utilization,
            enforce_eager=self.profile.enforce_eager,
            enable_lora=self.enable_lora,
            max_loras=self.profile.max_lora_slots if self.enable_lora else 1,
            enable_sleep_mode=self.enable_sleep_mode,
            # Resolved from the card, not configured: vLLM refuses bf16 below sm80
            # rather than emulating it, so a T4 run has to ask for fp16 or fail at
            # construction. `EvalProfile.dtype` carries the decision and its reason.
            dtype=self.profile.dtype,
            # vLLM resolves the flag to false for hybrid models such as Qwen3.5
            # when it is not passed (the same default the GRPO colocated engine
            # overrides). Multi-turn evaluation sends each turn as a fresh request
            # holding the whole conversation; without block reuse every turn
            # re-prefills the entire prefix.
            enable_prefix_caching=True,
            worker_extension_cls=f"{__name__}.MemoryWorkerExtension",
            # This project is text-only. Keep Qwen3.5's wrapper checkpoint shape,
            # but do not initialize its vision encoder or multimodal cache.
            language_model_only=True,
            **lora_kwargs,
        )
        resolved_caching = self._llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
        if resolved_caching is not True:
            # Fail closed on a changed engine default rather than serving a
            # multi-turn eval that silently prefill-bombs every turn.
            raise EngineError(
                "vLLM resolved enable_prefix_caching to "
                f"{resolved_caching!r}; multi-turn evaluation requires block reuse"
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

    def sampling_params(
        self,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
    ) -> Any:
        """Greedy by default, from the resolved decoding config.

        `logprobs=0` asks vLLM for the sampled token's own logprob and no
        alternatives, which is what the turn engine needs and the cheapest form of
        the request.
        """
        from vllm import SamplingParams

        profile = self.profile
        return SamplingParams(
            temperature=profile.temperature if temperature is None else temperature,
            top_p=profile.top_p if top_p is None else top_p,
            top_k=profile.top_k if top_k is None else top_k,
            presence_penalty=(
                profile.presence_penalty if presence_penalty is None else presence_penalty
            ),
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
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
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
            "sampling_params": self.sampling_params(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
            ),
        }
        if adapter is not None:
            request["lora_request"] = self._adapter_request(adapter)
        outputs = engine.generate(**request)

        if len(outputs) != len(prompt_ids):
            raise EngineError(
                f"engine returned {len(outputs)} outputs for {len(prompt_ids)} prompts; "
                "positional alignment with tasks cannot be assumed"
            )
        return [
            _token_completion(output, index, prompt_ids[index])
            for index, output in enumerate(outputs)
        ]

    def sleep(self, level: int = 1) -> None:
        """Offload weights to CPU and discard the KV cache.

        Level 1 keeps the weights on the host, so waking does not re-read the
        checkpoint. How much VRAM this actually returns is a runtime property of
        the pin and the card; a caller that depends on the number measures it with
        `reset_peak_memory_stats()` plus the worker's driver-level memory reading.
        `CuMemAllocator` can leave ``torch.cuda.memory_allocated()`` unchanged after
        unmapping, while `memory_reserved()` does not shrink without `empty_cache()`
        and `max_memory_allocated()` is a monotonic high-water mark.
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

    def memory_allocated_bytes(self, *, reset_peak: bool = False) -> int:
        """Read vLLM's physical CUDA footprint from its worker process.

        vLLM V1 runs the model in a spawned ``EngineCore``/worker process, so a
        ``torch.cuda.memory_allocated()`` call in this process reports the
        trainer's allocator (usually zero), not the engine's. ``collective_rpc``
        executes the read where the model owns its CUDA allocations. Sleep mode
        uses a pluggable allocator whose unmapped allocations are not reflected by
        ``torch.cuda.memory_allocated()``, so the worker reports the driver-level
        mapped footprint from ``mem_get_info`` instead. ``reset_peak`` is retained
        for the same-process measurement contract and compatibility with callers.
        """
        llm = self._llm
        if llm is None:
            raise EngineError("engine is not built; call build() first")
        try:
            readings = llm.collective_rpc(
                "read_cuda_memory_allocated",
                kwargs={"reset_peak": reset_peak},
            )
        except AttributeError as exc:
            raise EngineError("vLLM runtime does not expose collective_rpc") from exc
        if not readings:
            raise EngineError("vLLM returned no CUDA memory readings")
        return sum(int(reading) for reading in readings)

    def serving_config(self) -> dict[str, Any]:
        """What the built engine actually runs at, read from its own `VllmConfig`.

        This is the provenance a paired speed/quality row compares. The nine CLI
        flags that used to assert these values are gone, and correctly so -- they
        were a way to record something other than what ran -- but nothing replaced
        them as a *source*, so every field stayed `None` and
        `assert_quality_matches_serving` could never match a real serving row.

        Read from `vllm_config` rather than from `EvalProfile`: vLLM resolves
        `max_num_batched_tokens` and the chunked-prefill default itself, and the
        recorded value has to be the resolved one. Field names are vLLM 0.26's
        (`config/{model,cache,scheduler}.py`); an absent attribute records `None`
        rather than raising, because a report is not worth failing a scored run for.
        """
        llm = self._llm
        if llm is None:
            raise EngineError("engine is not built; call build() first")
        config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        model = getattr(config, "model_config", None)
        cache = getattr(config, "cache_config", None)
        scheduler = getattr(config, "scheduler_config", None)
        speculative = getattr(config, "speculative_config", None)
        return {
            "dtype": _dtype_name(getattr(model, "dtype", None)) or self.profile.dtype,
            "quantization": getattr(model, "quantization", None),
            "speculative_decoding": _speculative_name(speculative),
            "kv_budget": getattr(cache, "gpu_memory_utilization", None),
            "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
            "chunked_prefill": getattr(scheduler, "enable_chunked_prefill", None),
            "prefix_caching": getattr(cache, "enable_prefix_caching", None),
        }

    def load_adapter(self, name: str, path: str) -> Any:
        """Register a PEFT adapter directory for `generate(adapter=name)`.

        vLLM validates the adapter against the architecture when it is first used.
        Any refusal is fatal: serving base output under an adapter name would make an
        in-training benchmark curve flat while appearing valid.
        """
        if not self.enable_lora:
            raise EngineError("engine was built without enable_lora; adapters cannot be loaded")
        reserved = lora_rank_slot(
            self.max_lora_rank if self.max_lora_rank is not None else DEFAULT_LORA_RANK
        )
        vllm_path = self._prepare_adapter_path(path)
        found = adapter_rank(vllm_path)
        if found is not None and found > reserved:
            # Caught here rather than left to vLLM: its own check fires lazily, on
            # the first request carrying the `LoRARequest`, and reports the reserved
            # rank without naming what to change.
            raise EngineError(
                f"adapter {name!r} at {path} has rank {found}, above the {reserved} this "
                "engine reserved; build the engine with max_lora_rank at or above the "
                "rank the adapter was trained at"
            )
        try:
            from vllm.lora.request import LoRARequest
        except (ImportError, AttributeError) as exc:
            raise AdapterCapabilityError(
                "vLLM does not expose LoRARequest support for this runtime"
            ) from exc

        request = LoRARequest(name, len(self._adapters) + 1, vllm_path)
        self._adapters[name] = request
        return request

    def _prepare_adapter_path(self, path: str) -> str:
        """Add Qwen3.5's wrapper prefix to a text-only PEFT adapter when needed."""
        if not self._is_qwen35_wrapper():
            return path
        source = Path(path)
        weights = source / "adapter_model.safetensors"
        if not weights.is_file():
            return path

        from safetensors.torch import load_file, save_file

        tensors = load_file(str(weights), device="cpu")
        text_prefix = "base_model.model.model."
        wrapper_prefix = "base_model.model.model.language_model."
        if not any(
            key.startswith(text_prefix) and not key.startswith(wrapper_prefix) for key in tensors
        ):
            return path

        # ponytail: rewrite one known Qwen3.5 namespace seam; use vLLM's native
        # mapper unchanged for every other architecture.
        temporary = tempfile.TemporaryDirectory(prefix="smolqwen-vllm-adapter-")
        target = Path(temporary.name)
        shutil.copy2(source / "adapter_config.json", target / "adapter_config.json")
        save_file(
            {key.replace(text_prefix, wrapper_prefix, 1): value for key, value in tensors.items()},
            str(target / "adapter_model.safetensors"),
        )
        self._adapter_tempdirs.append(temporary)
        return str(target)

    def _is_qwen35_wrapper(self) -> bool:
        """Whether the built vLLM model uses the multimodal wrapper namespace."""
        llm = self._llm
        config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        model = getattr(config, "model_config", None)
        hf_config = getattr(model, "hf_config", None)
        return (
            getattr(model, "model_type", None) == "qwen3_5"
            or getattr(hf_config, "model_type", None) == "qwen3_5"
        )

    def validate_adapter(self, name: str) -> None:
        """Prove a non-zero adapter changes at least one deterministic probe.

        vLLM loads LoRA weights lazily, during the first request that carries the
        ``LoRARequest``. Some wrapper architectures can therefore accept the
        request while matching no runtime module and silently serving base output.
        A trained adapter must not take that path: a flat benchmark curve would
        otherwise look valid. Zero-initialized LoRA adapters are intentionally
        allowed because their zero delta is correct by construction.
        """
        request = self._adapter_request(name)
        path = getattr(request, "lora_path", None)
        if path is not None and not _adapter_has_nonzero_lora_b(str(path)):
            # Zero-initialized adapters correctly produce the base output, so there
            # is no useful equality comparison. Still issue one real request: LoRA
            # loading is lazy, and an unsupported zero adapter must not be allowed to
            # fail later outside the fallback boundary.
            self.generate_ids(
                ((1, 2, 3, 4),),
                max_new_tokens=1,
                adapter=name,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
            )
            return

        probes = _ADAPTER_PROBE_PROMPTS
        adapted = self.generate_ids(
            probes,
            max_new_tokens=4,
            adapter=name,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        base = self.generate_ids(
            probes,
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        if any(
            _token_completions_differ(left, right)
            for left, right in zip(adapted, base, strict=True)
        ):
            return
        raise AdapterCapabilityError(
            f"vLLM accepted adapter {name!r}, but its non-zero LoRA weights had no "
            "observable effect on deterministic probes; refusing a silent no-op"
        )

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
        self._adapter_tempdirs.clear()

    def __enter__(self) -> OfflineEngine:
        self.build()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


def _dtype_name(value: Any) -> str | None:
    """A torch dtype or string as the name a manifest records.

    `ModelConfig.dtype` is resolved to a `torch.dtype`, whose `str()` is
    `torch.float16`. The manifest's own vocabulary is vLLM's spelling, so the
    `torch.` prefix comes off and the two paths compare equal.
    """
    if value is None:
        return None
    text = str(value)
    return text.removeprefix("torch.") or None


def _speculative_name(config: Any) -> str | None:
    """One string for a speculative-decoding setup, or None when there is none.

    Shaped `<method>-<tokens>` so it survives the string comparison
    `assert_quality_matches_serving` performs; the pairing guard needs two rows to
    disagree visibly, not a nested object to diff.
    """
    if config is None:
        return None
    method = getattr(config, "method", None)
    tokens = getattr(config, "num_speculative_tokens", None)
    if method is None and tokens is None:
        return None
    return f"{method}-{tokens}"


def _adapter_has_nonzero_lora_b(path: str) -> bool:
    """Return whether an adapter appears trained rather than zero-initialized.

    PEFT writes ``adapter_model.safetensors``. If an artifact cannot be inspected,
    return ``True`` and let vLLM perform the authoritative load rather than treating
    an unreadable adapter as harmless.
    """
    root = Path(path)
    if not root.is_dir():
        return True
    weight_files = sorted(root.glob("adapter_model*.safetensors"))
    if not weight_files:
        return True

    found_lora_b = False
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if "lora_B" not in key:
                            continue
                        found_lora_b = True
                        if _tensor_has_nonzero_value(handle.get_tensor(key)):
                            return True
            except Exception:
                return True
    return not found_lora_b


def _tensor_has_nonzero_value(value: Any) -> bool:
    """Inspect one tensor without importing torch at module scope."""
    try:
        tensor = value.detach() if hasattr(value, "detach") else value
        return bool(tensor.ne(0).any().item())
    except Exception:
        # An unfamiliar tensor wrapper should not turn a real adapter into an
        # assumed zero delta; the subsequent vLLM probe remains authoritative.
        return True


def _token_completions_differ(left: TokenCompletion, right: TokenCompletion) -> bool:
    """Compare sampled ids and observed logprobs, tolerating NaN sentinels."""
    if left.token_ids != right.token_ids:
        return True
    for adapted, base in zip(left.logprobs, right.logprobs, strict=True):
        if (
            math.isfinite(adapted)
            and math.isfinite(base)
            and not math.isclose(adapted, base, rel_tol=1e-5, abs_tol=1e-6)
        ):
            return True
    return False


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


def _token_completion(output: Any, index: int, prompt_ids: Sequence[int]) -> TokenCompletion:
    """One vLLM `RequestOutput` as token ids plus per-token logprobs.

    vLLM returns `logprobs` as a list of `{token_id: Logprob}` maps, one per
    position, and only when the sampling params asked for them. A position whose
    sampled token is absent from its own map stays NaN: the alternative, dropping
    it, would shorten the row and shift every later logprob against the wrong token.
    """
    echoed = getattr(output, "prompt_token_ids", None)
    if echoed is not None and tuple(int(token) for token in echoed) != tuple(prompt_ids):
        raise EngineError(
            f"output at position {index} echoes different prompt token ids; "
            "prompts and completions are not positionally aligned"
        )
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
                finish_reason=(
                    "length"
                    if len(completion.token_ids) > request.max_new_tokens
                    or completion.finish_reason == "length"
                    else completion.finish_reason
                ),
            )
            for request, completion in zip(requests, completions, strict=True)
        ]


def offline_engine_for_eval(
    model: str,
    profile: EvalProfile,
    *,
    revision: str | None = None,
    adapter: Mapping[str, str] | None = None,
    enable_sleep_mode: bool = False,
) -> OfflineEngine:
    """Build an engine and register any adapters, in the order the engine needs.

    `enable_lora` has to be decided at construction -- vLLM allocates adapter slots
    then -- so it is derived from whether any adapter was named rather than being a
    second flag a caller can set inconsistently. `max_lora_rank` is decided there
    too, and for the same reason it is read from the adapters themselves: the widest
    rank about to be registered is the only value that keeps every one of them
    loadable.

    This existed once and was deleted as unreferenced. It had no references because
    `run_evaluation` was never rewired onto it, which is the opposite of dead: an
    audit reporting zero consumers for a helper written to be a command's entry point
    is reporting missing wiring, not dead code.
    """
    engine = OfflineEngine(
        model,
        profile,
        revision=revision,
        enable_lora=bool(adapter),
        enable_sleep_mode=enable_sleep_mode,
        max_lora_rank=widest_adapter_rank(adapter) if adapter else None,
    )
    try:
        engine.build()
        for name, path in (adapter or {}).items():
            engine.load_adapter(name, path)
            engine.validate_adapter(name)
        return engine
    except BaseException:
        # A failed build or adapter preflight may already have spawned vLLM's
        # EngineCore. Release it before propagating the failure.
        engine.shutdown()
        raise


def widest_adapter_rank(adapters: Mapping[str, str]) -> int:
    """The largest rank among the adapter directories about to be registered.

    An engine holding several adapters validates each against one `max_lora_rank`,
    so the widest is what has to fit. Directories whose rank cannot be read
    contribute the configured default rather than silently lowering the ceiling.
    """
    ranks = [adapter_rank(path) or DEFAULT_LORA_RANK for path in adapters.values()]
    return max(ranks) if ranks else DEFAULT_LORA_RANK
