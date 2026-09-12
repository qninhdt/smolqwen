"""vLLM profiles derived from resolved stage configuration.

Evaluation and rollout profiles do not declare sizing of their own:
`max_model_len`, `gpu_memory_utilization` and batch width come from the resolved
`ProfileConfig`. Serving uses its separate closed `ServingProfileConfig`, so
`latency`, `balanced`, and `throughput` cannot leak hardware/training fields into
the serving command. Declaring either set again outside the overlay chain would
make a profile silently stop sizing its stage while manifests still record the
old value.

So: read the resolved profile, add only genuinely new knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from smolqwen.config_models import EvalConfig, GrpoConfig, ServeConfig
from smolqwen.console import logger
from smolqwen.inference.turn_engine import TurnEngineConfig

LOG = logger(__name__)

# The dtype names vLLM's `LLM` accepts. Spelled out rather than left as `str` so a
# typo in a serving overlay fails at type-check instead of at engine construction.
VllmDtype = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]

# bf16 tensor cores arrive with Ampere. Below that vLLM refuses bf16 outright
# rather than emulating it, so a T4 (sm75) run must ask for fp16.
BF16_MIN_CAPABILITY = (8, 0)


def resolve_dtype(
    requested: VllmDtype = "bfloat16", *, capability: tuple[int, int] | None = None
) -> VllmDtype:
    """The dtype this card can actually run, with the downgrade logged.

    Same shape as `resolve_attn_implementation`: state the request, state what the
    host supports, pick the workable one and say why. A silent choice here is the
    wrong kind of quiet -- fp16 has a narrower exponent range than bf16, so a reader
    comparing a T4 number against an L4 number needs to know which ran.

    `torch.cuda.is_bf16_supported()` is not the check: it returns True on sm75
    because it counts emulation, and vLLM's own guard is the capability number.
    Probed here rather than imported at module scope so `--dry-run` stays free of
    torch.
    """
    if requested != "bfloat16":
        return requested
    if capability is None:
        capability = _capability()
    if capability is None or capability >= BF16_MIN_CAPABILITY:
        return "bfloat16"
    LOG.warning(
        "compute capability %d.%d has no bf16 tensor cores; using float16. "
        "Numerics differ from a bf16 run -- do not compare the two as one experiment.",
        *capability,
    )
    return "float16"


def _capability() -> tuple[int, int] | None:
    """This host's compute capability, or None when there is no CUDA device."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


@dataclass(frozen=True)
class EvalProfile:
    """What an offline `vllm.LLM` needs to score a benchmark.

    `max_model_len` is `ProfileConfig.max_seq_length` and must stay that way:
    `runner.py:42` records that field into the manifest invariant as
    `max_context_tokens`. If the two ever need to diverge, that line changes in
    the same commit -- otherwise a run truncating at one length certifies
    comparability with a run that truncated at another.

    `dtype` is resolved from the card rather than configured. It is sizing in the
    same sense `enforce_eager` is: it does not change what a benchmark measures, and
    a config field for it would let a T4 run be asked for bf16 and fail at engine
    construction instead of downgrading with a recorded reason. The manifest records
    what ran.
    """

    max_model_len: int
    gpu_memory_utilization: float
    concurrency: int
    enforce_eager: bool
    max_lora_slots: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    seed: int | None
    dtype: VllmDtype = "bfloat16"

    @classmethod
    def from_config(cls, config: EvalConfig) -> EvalProfile:
        profile = config.profile
        decoding = config.decoding
        return cls(
            max_model_len=profile.max_seq_length,
            gpu_memory_utilization=profile.vllm_kv_fraction,
            concurrency=profile.generation_concurrency,
            enforce_eager=profile.enforce_eager,
            max_lora_slots=profile.max_lora_slots,
            max_new_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            presence_penalty=decoding.presence_penalty,
            seed=decoding.seed,
            dtype=resolve_dtype(),
        )


@dataclass(frozen=True)
class RolloutProfile:
    """The training-side engine's sizing, for the in-training eval boundary.

    GRPO's engine is TRL's, constructed by the trainer -- this profile does not
    build it. It exists so an in-training eval can be sized from the same numbers
    the trainer used, rather than from a second set that drifts.
    """

    max_model_len: int
    gpu_memory_utilization: float
    concurrency: int
    enable_sleep_mode: bool
    enable_prefix_caching: bool

    @classmethod
    def from_config(cls, config: GrpoConfig) -> RolloutProfile:
        return cls(
            max_model_len=config.vllm_max_model_len,
            gpu_memory_utilization=config.profile.vllm_kv_fraction,
            concurrency=config.profile.generation_concurrency,
            enable_sleep_mode=config.vllm_enable_sleep_mode,
            enable_prefix_caching=config.vllm_enable_prefix_caching,
        )


def turn_engine_config(config: GrpoConfig, **overrides: Any) -> TurnEngineConfig:
    """Map a resolved `GrpoConfig` onto the shared engine's semantic knobs.

    Lives here rather than in `training/grpo.py` because `rollout/bench.py` also
    builds it, and `bench.py` already imports from `grpo.py` -- putting the mapping
    there would close an import cycle.

    `max_generation_turns` is derived from `max_env_steps` rather than given its own
    config field: a well-behaved episode issues one generation per environment step
    plus a final answer, and the extra head-room covers invalid calls that consume a
    generation without executing anything. A model emitting only prose is what this
    bound exists for, and it hits at `max_env_steps + 4` instead of running to the
    wall clock.
    """
    profile = config.profile
    defaults: dict[str, Any] = {
        "generation_concurrency": profile.generation_concurrency,
        "max_env_steps": profile.max_env_steps,
        "max_generation_turns": profile.max_env_steps + 4,
        "episode_timeout_s": config.episode_timeout_s,
        "max_new_tokens_per_step": profile.max_new_tokens_per_step,
        "max_model_len": config.vllm_max_model_len,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "fork_threshold_tokens": config.fork_threshold_tokens,
        "enable_thinking": config.enable_thinking,
    }
    defaults.update(overrides)
    return TurnEngineConfig(**defaults)


@dataclass(frozen=True)
class ServeProfile:
    """The `vllm serve` argv for one measured operating profile.

    `command()` is `serving/server.py:19` verbatim, including both
    `--enable-*`/`--no-enable-*` pairs -- vLLM's defaults for prefix caching and
    chunked prefill have changed across releases, so passing the negative form
    explicitly is what makes the recorded config the one that ran -- and the
    speculative-config JSON shape. `test_serving_commands.py` passes against this
    with argv unchanged; if it had needed editing, the move was not mechanical.
    """

    config: ServeConfig

    def command(self) -> list[str]:
        config = self.config
        profile = config.profile
        if profile.kv_cache_scale != "default":
            raise ValueError(
                "checkpoint KV scales require the Phase 3 quantized checkpoint; "
                "vLLM 0.29 has no independent --kv-cache-scale flag"
            )
        command = [
            "vllm",
            "serve",
            config.model_path,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--served-model-name",
            config.served_model_name,
            "--max-model-len",
            str(config.max_model_len),
            "--dtype",
            profile.dtype,
            "--kv-cache-dtype",
            profile.kv_cache_dtype,
            "--reasoning-parser",
            config.reasoning_parser,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            config.tool_call_parser,
            "--max-num-seqs",
            str(profile.max_num_seqs),
            "--max-num-batched-tokens",
            str(profile.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(profile.gpu_memory_utilization),
        ]
        command.append(
            "--enable-prefix-caching"
            if config.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )
        command.append(
            "--enable-chunked-prefill"
            if config.enable_chunked_prefill
            else "--no-enable-chunked-prefill"
        )
        if config.model_revision:
            command.extend(["--revision", config.model_revision])
        if profile.max_num_queued_reqs is not None:
            command.extend(
                [
                    "--max-num-queued-reqs",
                    str(profile.max_num_queued_reqs),
                    "--max-num-queued-tokens",
                    str(profile.max_num_queued_tokens),
                ]
            )
        if profile.quantization:
            command.extend(["--quantization", profile.quantization])
        if config.speculative_num_tokens is not None:
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(
                        {
                            "method": "mtp",
                            "num_speculative_tokens": config.speculative_num_tokens,
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
        return command
