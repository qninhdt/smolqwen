"""Two-layer configuration: semantics in `configs/base/`, sizing in `configs/profiles/`.

The merge order is ``budgets.json -> base -> profile -> --override``, deepest wins.
``budgets.json`` is a real layer in that chain rather than a suggestion: Phase 2
measures the trajectory distribution and writes the caps, and every later stage
reads them instead of hardcoding a number.

Keeping "which GPU" out of the semantic layer is what makes the L4-vs-A100
comparison honest -- the same experiment, different sizing. `ProfileConfig` is a
closed model, so a semantic field placed in a profile fails at load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STAGES = ("data", "sft", "grpo", "eval", "serve")
Stage = Literal["data", "sft", "grpo", "eval", "serve"]
PROFILES = ("l4", "a100")

# budgets.json key -> the profile field it seeds. Phase 2 owns the file; the
# consuming phase may lower its own field but never raise it above the budget.
BUDGET_SEEDED_FIELDS = {
    "max_seq_length": "max_seq_length",
    "max_new_tokens_per_step": "max_new_tokens_per_step",
    "max_env_steps": "max_env_steps",
}


class ConfigError(Exception):
    """Raised when configuration cannot be resolved or violates the schema."""


class StrictModel(BaseModel):
    """Base for every config model: unknown keys are an error, not a silent drop."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileConfig(StrictModel):
    """GPU sizing only. Every field is owned by exactly one phase.

    A field here must never change what the experiment means -- only how much of
    it fits on the card at once. `extra="forbid"` is the enforcement: a semantic
    key in a profile YAML fails at load rather than thirty minutes into a run.
    """

    # Phase 3 (OOM sweep)
    micro_batch: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=8192, ge=256)
    max_tokens_per_microbatch: int = Field(default=32768, ge=256)

    # Phase 6 (pool sweep). `generation_concurrency` is the vLLM batch width
    # inside one rollout call; it is NOT `num_generations`, the GRPO group size.
    generation_concurrency: int = Field(default=8, ge=1)
    active_pool_multiplier: int = Field(default=1, ge=1)
    vllm_kv_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    max_new_tokens_per_step: int = Field(default=1024, ge=16)

    # Phase 7 (widening sweep)
    num_generations: int = Field(default=4, ge=2)
    max_env_steps: int = Field(default=16, ge=1)

    # Phase 4 (timeline profile)
    env_worker_count: int = Field(default=4, ge=1)
    env_episodes_per_worker: int = Field(default=8, ge=1)

    @property
    def generation_batch_size(self) -> int:
        """TRL prompt-pool size for one synchronous `rollout_func` call."""
        return self.generation_concurrency * self.active_pool_multiplier


class TrackingConfig(StrictModel):
    """W&B and Hub plumbing. Absent tokens degrade to offline, never to a crash."""

    wandb_project: str = "smolqwen"
    wandb_entity: str | None = None
    run_name: str | None = None
    hub_repo_id: str | None = None
    local_artifact_dir: str = "artifacts"


class DatasetPin(StrictModel):
    """A dataset identified by repo, file and revision sha.

    The revision is not optional. The two metadata files whose contents get
    ``exec()``ed are the only inputs where a silent content change is invisible:
    191 classes still compile, the suffix split is unchanged, the ``K`` counts are
    unchanged. `sha256` is verified at registry build in Phase 4.
    """

    repo_id: str
    filename: str
    revision: str
    repo_type: Literal["dataset", "model"] = "dataset"
    sha256: str | None = None
    local_path: str | None = None


class DataConfig(StrictModel):
    """Phase 2: trajectory profiling and SFT conversion."""

    # The model whose chat template and tokenizer produce the rendered samples.
    # The pipeline is text-only (see `tokenizer.assert_text_only_processing_class`):
    # Qwen3.5-2B is multimodal and a processor would flip TRL onto VLM code paths.
    model_id: str = "Qwen/Qwen3.5-2B"
    sft_trajectories: DatasetPin
    rl_scenarios: DatasetPin
    env_metadata: DatasetPin
    output_dir: str = "artifacts/data"
    max_seq_length: int = Field(default=32768, ge=256)
    split_seed: int = 1234
    val_fraction: float = Field(default=0.02, gt=0.0, lt=0.5)
    # Which shape a tool result takes on the way into the chat template. Phase 2
    # step 1 fixes this against real excerpts, and Phase 6's rollout must append
    # the same shape -- the two render one newline apart on every observation.
    tool_result_shape: Literal["tool_role", "tool_response_user"] = "tool_role"
    max_trajectories: int | None = None
    profile: ProfileConfig = ProfileConfig()
    tracking: TrackingConfig = TrackingConfig()


class LoraConfig(StrictModel):
    r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str | Sequence[str] = "all-linear"
    # bf16 adapters, not PEFT's fp32 default: with all-linear targets fp32 forces
    # an upcast/downcast plus an fp32 GEMM at every linear. The fp32 default
    # exists for low-bit QLoRA bases, which this is not.
    adapter_dtype: Literal["bfloat16", "float32"] = "bfloat16"


class OptimizationFlags(StrictModel):
    """Each toggle carries a measured before/after in the Phase 3 ledger or is off."""

    bf16: bool = True
    gradient_checkpointing: bool = True
    # Load-bearing, not a nice-to-have: a dense logits tensor over the 248,320
    # vocab is the dominant activation and the first thing to OOM on 24 GB.
    liger_fused_linear_cross_entropy: bool = True
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    regional_torch_compile: bool = False
    # GDN layers run Triton kernels that upstream marks `torch.compiler.disable`.
    # Compiling through them raises inductor errors, so the mixer body stays eager.
    compile_exclude_patterns: Sequence[str] = ("linear_attn", "mixer", "conv1d")


class TrainingConfig(StrictModel):
    learning_rate: float = Field(default=1e-4, gt=0.0)
    num_train_epochs: float = Field(default=2.0, gt=0.0)
    max_steps: int = -1
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_scheduler_type: str = "cosine"
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    seed: int = 1234


class SftConfig(StrictModel):
    """Phase 3: reasoning SFT."""

    model_id: str = "Qwen/Qwen3.5-2B"
    model_revision: str | None = None
    dataset_dir: str = "artifacts/data/sft"
    output_dir: str = "artifacts/models/qwen3.5-2b-sft"
    merged_dir: str = "artifacts/models/qwen3.5-2b-sft-merged"
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    optimization: OptimizationFlags = OptimizationFlags()
    profile: ProfileConfig = ProfileConfig()
    tracking: TrackingConfig = TrackingConfig()


class EnvRuntimeConfig(StrictModel):
    """Phase 4: the executable environment layer."""

    # Which artifact is authoritative for `191_env_metadata.json`: the vendored
    # EnvScaler copy or the Hub download (a different file of a different size).
    env_metadata_source: Literal["vendored", "hub"] = "vendored"
    vendored_env_metadata: str = (
        "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/"
        "191_env_metadata.json"
    )
    vendored_rl_scenarios: str = (
        "third_party/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env/data/"
        "envscaler_rl_scenario_metadata.json"
    )
    # These are separate from the Phase 2 source DatasetPins because Phase 4
    # consumes the vendored copies directly. They make a modified executable
    # body fail before a worker executes it.
    vendored_env_metadata_sha256: str = (
        "d2c0010f16ff77d6d55868ee386353b1d0aadace58beed1eed678e8f7c84c33d"
    )
    vendored_rl_scenarios_sha256: str = (
        "5977bda0b941a9111b290cbf5ffd6d70678a36ddc499b8f153826fd22999337e"
    )
    step_timeout_s: float = Field(default=10.0, gt=0.0)
    # K ranges 2 -> 445 with median 14, so the verifier budget cannot assume a
    # handful of checks.
    verify_timeout_s: float = Field(default=60.0, gt=0.0)
    create_timeout_s: float = Field(default=30.0, gt=0.0)
    env_split_manifest: str = "artifacts/data/env_split.json"


class CurriculumConfig(StrictModel):
    """Phase 7: prioritise scenarios where `0 < P(success) < 1`."""

    enabled: bool = True
    profile_rollouts: int = Field(default=4, ge=2)
    profile_scenario_sample: int = Field(default=128, ge=1)
    band_weight: float = Field(default=1.0, gt=0.0)
    always_zero_weight: float = Field(default=0.1, ge=0.0)
    always_one_weight: float = Field(default=0.1, ge=0.0)
    difficulty_profile_path: str = "artifacts/rl/difficulty_profile.json"
    heldout_env_count: int = Field(default=10, ge=1)
    heldout_scenarios_per_env: int = Field(default=8, ge=1)
    trajectory_samples_per_log: int = Field(default=8, ge=1)
    zero_variance_stop_margin: float = Field(default=0.25, ge=0.0, le=1.0)
    zero_variance_stop_multiplier: float = Field(default=1.5, ge=1.0)
    zero_variance_stop_after_steps: int = Field(default=10, ge=1)


class GrpoConfig(StrictModel):
    """Phases 6 and 7: async rollout plus GRPO."""

    model_id: str = "artifacts/models/qwen3.5-2b-sft-merged"
    model_revision: str | None = None
    output_dir: str = "artifacts/models/qwen3.5-2b-sft-grpo"
    merged_dir: str = "artifacts/models/qwen3.5-2b-sft-grpo-merged"
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    optimization: OptimizationFlags = OptimizationFlags()
    env: EnvRuntimeConfig = EnvRuntimeConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
    beta: float = Field(default=0.0, ge=0.0)
    loss_type: str = "dapo"
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # The model advertises 262,144 positions. Inheriting that under colocated
    # vLLM is an instant OOM, so the KV budget is always explicit.
    vllm_max_model_len: int = Field(default=16384, ge=512)
    # vLLM v1 enables APC by default for supported models, while TRL 1.12 does
    # not expose the engine argument. Make disabling it an invalid experiment
    # config and assert the live engine state after trainer construction.
    vllm_enable_prefix_caching: Literal[True] = True
    vllm_enable_sleep_mode: bool = False
    # The only visible symptom of logprob misalignment under importance-sampling
    # correction. Above this, stop the run rather than train on garbage ratios.
    logp_difference_stop_threshold: float = Field(default=2.0, gt=0.0)
    # Phase 6 scheduler semantics. An episode wall-clock budget: a ping-pong
    # episode that never terminates must not outlive the step cap's slower
    # bound. The drift classifier's fork threshold is upstream's default,
    # surfaced here so the drift tally has a knob to be tuned against.
    episode_timeout_s: float = Field(default=600.0, gt=0.0)
    fork_threshold_tokens: int = Field(default=1024, ge=1)
    rollout_path: Literal["async", "factory_oracle"] = "async"
    profile: ProfileConfig = ProfileConfig()
    tracking: TrackingConfig = TrackingConfig()


class DecodingConfig(StrictModel):
    """Part of the eval manifest's invariant set -- identical for every checkpoint."""

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = -1
    max_new_tokens: int = Field(default=2048, ge=1)
    seed: int = 1234


class EvalConfig(StrictModel):
    """Phase 5: the benchmark adapter layer and the headline table."""

    adapters: Sequence[str] = ()
    # Adapter-owned settings stay opaque to the core config model. A new
    # benchmark validates its own entry in its adapter module, so adding one
    # does not require another field here or another branch in the runner.
    adapter_options: Mapping[str, Mapping[str, object]] = Field(default_factory=dict)
    http_model: str = "smolqwen"
    http_timeout_s: float = Field(default=60.0, gt=0.0)
    max_steps_per_task: int = Field(default=20, ge=1)
    decoding: DecodingConfig = DecodingConfig()
    output_dir: str = "artifacts/evaluation"
    profile: ProfileConfig = ProfileConfig()
    tracking: TrackingConfig = TrackingConfig()


class ServeConfig(StrictModel):
    """Phase 8: the OpenAI-compatible endpoint and its measurement."""

    model_path: str = "artifacts/models/qwen3.5-2b-sft-grpo-merged"
    model_revision: str | None = None
    served_model_name: str = "smolqwen"
    dtype: str = "bfloat16"
    max_model_len: int = Field(default=32768, ge=512)
    # vLLM binds to loopback; the key-checking proxy is the only exposed service,
    # because `--api-key` challenges GUARDED_PREFIX paths only.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    proxy_port: int = Field(default=8080, ge=1, le=65535)
    reasoning_parser: str = "qwen3"
    tool_call_parser: str = "hermes"
    quantization: str | None = None
    speculative_num_tokens: int | None = None
    enable_prefix_caching: bool = True
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    benchmark_num_prompts: int = Field(default=100, ge=1)
    benchmark_input_len: int = Field(default=1024, ge=1)
    benchmark_output_len: int = Field(default=256, ge=1)
    benchmark_percentiles: Sequence[int] = (50, 95, 99)
    readiness_timeout_s: float = Field(default=600.0, gt=0.0)
    readiness_poll_interval_s: float = Field(default=2.0, gt=0.0)
    sweep_num_runs: int = Field(default=3, ge=1)
    output_dir: str = "artifacts/serving"
    profile: ProfileConfig = ProfileConfig()
    tracking: TrackingConfig = TrackingConfig()


STAGE_MODELS: Mapping[str, type[StrictModel]] = {
    "data": DataConfig,
    "sft": SftConfig,
    "grpo": GrpoConfig,
    "eval": EvalConfig,
    "serve": ServeConfig,
}
