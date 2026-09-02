"""Per-task vLLM sizing, derived from the resolved `ProfileConfig`.

These profiles do not declare sizing of their own. `max_model_len`,
`gpu_memory_utilization` and the batch width already exist as
`ProfileConfig.max_seq_length` (`config_models.py:54`), `vllm_kv_fraction`
(`:61`) and `generation_concurrency` (`:59`), and every stage config embeds a
`ProfileConfig`. Declaring them again would put them outside the overlay chain --
`config.py:245` resolves a profile YAML only into the `profile` subtree -- so
`--profile l4` would silently stop sizing evaluation while `runner.py:42` kept
recording `max_seq_length` into the manifest's **invariant** set, certifying
comparability between runs that truncated differently.

So: read the resolved profile, add only genuinely new knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from smolqwen.config_models import EvalConfig, GrpoConfig, ServeConfig
from smolqwen.inference.turn_engine import TurnEngineConfig


@dataclass(frozen=True)
class EvalProfile:
    """What an offline `vllm.LLM` needs to score a benchmark.

    `max_model_len` is `ProfileConfig.max_seq_length` and must stay that way:
    `runner.py:42` records that field into the manifest invariant as
    `max_context_tokens`. If the two ever need to diverge, that line changes in
    the same commit -- otherwise a run truncating at one length certifies
    comparability with a run that truncated at another.
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
    seed: int | None

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
            seed=decoding.seed,
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
    }
    defaults.update(overrides)
    return TurnEngineConfig(**defaults)


@dataclass(frozen=True)
class ServeProfile:
    """The `vllm serve` argv, moved here so one package owns every vLLM surface.

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
            config.dtype,
            "--reasoning-parser",
            config.reasoning_parser,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            config.tool_call_parser,
            "--max-num-seqs",
            str(config.max_num_seqs),
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
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
        if config.quantization:
            command.extend(["--quantization", config.quantization])
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
