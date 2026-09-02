"""SFT's in-training eval: a sleeping vLLM engine over the checkpoint TRL wrote.

`SFTTrainer` has no `use_vllm` -- only `GRPOTrainer` does -- so there is no colocated
engine to borrow. That difference is about who calls the API, not about capability:
sleep mode is an engine feature reachable from the offline `LLM` entrypoint.

And SFT does not need weight sync at all. GRPO syncs because generation is part of
the algorithm and must run under the current policy. SFT evaluation scores a *saved
checkpoint*, which is the pattern open post-training pipelines follow -- and the
checkpoint is already on disk: `sft.py`'s `CheckpointPushCallback` runs at `on_save`,
where TRL has just written `checkpoint-N`.

    on_save (TRL wrote checkpoint-N; the store copied and pushed it)
       -> wake_up()                     # base weights back in VRAM
       -> register checkpoint-N as a LoRA adapter
       -> score the dev subset ; log
       -> sleep()                       # weights offloaded, KV freed
       -> training continues, optimizer state never touched

Base weights never change during SFT -- only the adapter does -- so there is nothing
to reload on the training side, and the doc-style `unload -> reload optimizer` step
is unnecessary.

**Two ordering facts this module exists to enforce.**

`gpu_memory_utilization` sizes the KV pool against **total** GPU memory, not against
what is free. Built after a resident trainer, the engine either OOMs or reserves
against a figure it cannot honour on a 24 GB L4. So the engine is built and slept
*before* the trainer exists, and the trainer sizes itself against what remains.

And each boundary registers its checkpoint under a **new** adapter name. vLLM caches
LoRA weights by integer id, and `OfflineEngine.load_adapter` derives that id from how
many adapters it has registered -- so reusing one name would hand vLLM the same id
with a different path and serve step 100's weights for step 200's score. A curve
built from that would be flat and every number in it plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig, SftConfig
from smolqwen.console import logger
from smolqwen.inference.engine import OfflineEngine, OfflineEngineBackend
from smolqwen.inference.profiles import EvalProfile

LOG = logger(__name__)


class CheckpointEvalError(RuntimeError):
    """Raised when the checkpoint an eval boundary must score is not there."""


def eval_config_for(config: SftConfig) -> EvalConfig:
    """The eval stage's config, resized by the SFT run's own profile.

    Resolved from the same YAML `smolqwen evaluate` reads, so the adapter's held-out
    selection and decoding are identical -- that is what makes one number mean one
    thing in both places.

    Its `profile` subtree is then replaced with this run's. `resolve("eval")` takes no
    `--profile`, so it would otherwise carry `ProfileConfig` defaults: an engine sized
    at the default KV fraction while the trainer was sized by `--profile l4` is two
    sets of numbers for one card.
    """
    from smolqwen.config import resolve

    resolved = resolve("eval")
    if not isinstance(resolved, EvalConfig):
        raise CheckpointEvalError(
            f"expected EvalConfig from the eval stage, got {type(resolved).__name__}"
        )
    return resolved.model_copy(update={"profile": config.profile})


@dataclass
class CheckpointEngine:
    """A sleeping offline engine, awake only inside an eval boundary.

    Holds the *base* model. SFT trains an adapter, so the base is loaded once and the
    per-boundary work is registering a checkpoint directory as a LoRA adapter.
    """

    engine: OfflineEngine
    output_dir: str
    step: int = 0
    registered: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(cls, config: SftConfig, *, engine: OfflineEngine | None = None) -> CheckpointEngine:
        """Construct and sleep the engine. Call before the trainer exists."""
        built = engine or OfflineEngine(
            config.model_id,
            EvalProfile.from_config(eval_config_for(config)),
            revision=config.model_revision,
            enable_lora=True,
            enable_sleep_mode=True,
        )
        built.build()
        built.sleep()
        LOG.info(
            "bench-eval engine built and asleep before the trainer; "
            "the trainer now sizes itself against what remains"
        )
        return cls(engine=built, output_dir=config.output_dir)

    def prepare(self, step: int) -> None:
        """Record which boundary is being scored. No I/O, so it cannot fail here."""
        self.step = step

    def backend(self) -> OfflineEngineBackend:
        """Wake, register `checkpoint-<step>`, and return the engine's backend.

        Called inside the runner's own try block, which is deliberate: a missing
        checkpoint raises here and is recorded as `bench_failed` with the path named,
        rather than either ending the training run or skipping silently.

        **Step 0 is the base model, with no adapter.** The plan asked for a step-0
        anchor and for a present-checkpoint assertion in the same breath, and at step
        0 there is no `checkpoint-0` -- TRL has not saved anything. GRPO's anchor works
        because its colocated engine holds live weights with an untrained adapter. The
        SFT equivalent is the base model, which is also exactly the `base` arm of the
        final Base | SFT | SFT+RL table, so the anchor is a number that already means
        something rather than an artifact of when the callback fired.
        """
        self.engine.wake_up()
        if self.step == 0 and not self.checkpoint_dir(0).is_dir():
            return OfflineEngineBackend(self.engine, adapter=None)
        checkpoint = self.checkpoint_dir(self.step)
        if not checkpoint.is_dir():
            raise CheckpointEvalError(
                f"no checkpoint to score at step {self.step}: {checkpoint} does not exist. "
                "Eval boundaries are save boundaries; scoring the previous checkpoint "
                "would attribute one step's weights to another step's number."
            )
        # A fresh name per boundary, so vLLM gets a fresh integer id. See the module
        # docstring: a reused id with a new path serves the older weights.
        name = f"checkpoint-{self.step}"
        if self.registered.get(self.step) != str(checkpoint):
            self.engine.load_adapter(name, str(checkpoint))
            self.registered[self.step] = str(checkpoint)
        return OfflineEngineBackend(self.engine, adapter=name)

    def release(self, step: int | None = None) -> None:
        """Offload weights and drop the KV cache, so training steps pay nothing."""
        self.engine.sleep()

    def checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / f"checkpoint-{step}"

    def shutdown(self) -> None:
        self.engine.shutdown()


def build_bench_eval_callback(
    config: SftConfig,
    trainer: Any,
    tokenizer: Any,
    *,
    engine: CheckpointEngine,
) -> Any:
    """The in-training dev-eval callback for SFT, over a checkpoint-based engine.

    Structurally identical to GRPO's, and deliberately so: the same runner, the same
    turn engine, the same aggregation. Only where the weights come from differs --
    a directory on disk instead of a colocated engine's live parameters -- which is
    why there is no `sync_weights` here and why the weight version is the checkpoint's
    own step rather than a sync counter.
    """
    from smolqwen.training.bench_eval import BenchEvalCallback, BenchEvalRunner

    runner = BenchEvalRunner(
        eval_config=eval_config_for(config),
        bench_config=config.bench_eval,
        engine_source=engine.backend,
        tokenizer_source=lambda: tokenizer,
        metric_prefix="sft",
        sink=(lambda payload: trainer.log(dict(payload))) if hasattr(trainer, "log") else None,
        weight_version=lambda: f"checkpoint-{engine.step}",
        artifact_dir=config.output_dir,
    )
    return BenchEvalCallback(runner, before_each=engine.prepare, after_each=engine.release)
