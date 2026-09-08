"""Optimization toggles, one function per switch, each returning what it decided.

Every toggle answers two questions in one place: *is it actually available here*
and *what did it do*. The second half is why each returns a `Toggle` carrying a
human-readable detail string — those strings become the W&B run config and the
Phase 3 ledger, so "every enabled optimization has a measured justification"
is a property of the code rather than of someone's notes.

Three of these are architecture-specific and none is optional bookkeeping:

- **Liger fused linear cross-entropy** is load-bearing. Vocabulary is 248,320
  with tied embeddings, so a dense logits tensor is the dominant activation and
  the first thing to OOM on a 24 GB card — ahead of the weights. If it is
  unavailable the response is to lower `max_seq_length`, because the logits
  tensor scales with sequence length; `liger_unavailable_guidance` says so at the
  point of failure rather than in a doc.
- **Attention implementation** cannot be taken on faith: `flash_attention_2`
  needs both the `flash_attn` wheel and a CUDA device, and the honest failure is
  a downgrade with a recorded reason, not an exception thirty minutes in.
- **Regional `torch.compile`** must leave the GDN mixer body eager. Those layers
  run `flash-linear-attention` Triton kernels that upstream marks
  `torch.compiler.disable`; compiling through them raises inductor errors. So the
  compile walk is opt-in per module with an explicit exclusion list, and it
  returns the names it compiled so the exclusion is auditable.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Module classes worth compiling. Everything else -- and anything matching the
# exclusion patterns -- stays eager. Matching on the class name keeps this free of
# a transformers import, so the selection is unit-testable on CPU.
COMPILE_TARGET_CLASS_SUFFIXES: tuple[str, ...] = ("MLP", "RMSNorm", "Attention")

# Class names that must never be compiled even if a suffix above matches. The
# gated-delta-net block is the mixer body itself.
COMPILE_FORBIDDEN_CLASSES: tuple[str, ...] = ("Qwen3_5GatedDeltaNet",)

# FlashAttention-2's kernels are Ampere-and-newer. Below this, the installed wheel
# is irrelevant: the only other attention implementation this project runs is sdpa.
FLASH_ATTENTION_2_MIN_CAPABILITY: tuple[int, int] = (8, 0)

# bf16 tensor cores also arrive with Ampere. Torch emulates bf16 on Turing rather
# than refusing it, so the capability number -- not `torch.cuda.is_bf16_supported()`
# -- is what decides whether a run gets bf16 or fp16.
BF16_MIN_CAPABILITY: tuple[int, int] = (8, 0)

# Qwen3.5's released checkpoint wraps the text model beside a visual tower. The
# project runs text-only, so `all-linear` must cover the language branch without
# creating adapters for modules vLLM's text-only LoRA mapper cannot serve.
TEXT_ONLY_PEFT_EXCLUDE_MODULES = r".*\.visual(?:\..*)?$"


@dataclass(frozen=True)
class Toggle:
    """One optimization decision: what it is, whether it is on, and why.

    `detail` is the ledger row. A toggle that is off carries the reason it is off,
    so a run that quietly lost flash-attention is visible in the run config
    instead of only in the throughput number.
    """

    name: str
    enabled: bool
    detail: str

    def as_config_entry(self) -> tuple[str, str]:
        state = "on" if self.enabled else "off"
        return (f"optimization/{self.name}", f"{state}: {self.detail}")


def liger_kernel_available() -> bool:
    """True when `liger_kernel` can be imported. No side effects, no CUDA needed."""
    try:
        import liger_kernel.transformers  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return False
    return bool(torch.cuda.is_available())


def cuda_capability() -> tuple[int, int] | None:
    """This host's compute capability, or None when there is no CUDA device."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def flash_attn_available() -> bool:
    """Whether the `flash_attn` wheel is importable.

    Probed with `find_spec` rather than an import: the wheel is a `colab` extra
    that is absent in CPU-only environments, and an unresolvable import would be
    a type-check error in every one of them.
    """
    return importlib.util.find_spec("flash_attn") is not None


def resolve_liger(requested: bool, *, available: bool | None = None) -> Toggle:
    """Decide whether the fused linear cross-entropy head is in play."""
    if not requested:
        return Toggle(
            "liger_fused_linear_cross_entropy",
            False,
            "disabled by config; full logits over 248,320 vocab are materialized",
        )
    present = liger_kernel_available() if available is None else available
    if not present:
        return Toggle(
            "liger_fused_linear_cross_entropy",
            False,
            "requested but liger_kernel is not importable; " + liger_unavailable_guidance(),
        )
    return Toggle(
        "liger_fused_linear_cross_entropy",
        True,
        "fused linear CE: no dense [batch, seq, 248320] logits tensor is materialized",
    )


def liger_unavailable_guidance() -> str:
    """What to do instead, stated where the failure is observed.

    Shortening the sequence is the right lever because the logits tensor scales
    with sequence length; cutting batch size below 1 is not an option.
    """
    return (
        "lower profile.max_seq_length before touching micro_batch -- "
        "the logits tensor scales with sequence length, not with batch size"
    )


def resolve_selective_logits(
    requested: bool, *, liger_enabled: bool, padding_free: bool, supervised_fraction: float | None
) -> Toggle:
    """Decide whether the head projects only the supervised positions.

    Liger's fused head is chunked but not selective: it sizes chunks from the row
    count and runs the GEMM over every position, using the label mask only to zero
    the loss afterwards. So a full-trajectory shard pays the projection on
    observation and prompt tokens that contribute nothing.

    `logits_to_keep` as an index tensor is the upstream contract for skipping them
    (`modeling_qwen3_5.py:1643`), and Liger's `lce_forward` forwards it unchanged.
    Two hard preconditions, both about correctness rather than speed:

    - **The fused head must be in play.** Without it the dense `[1, T, 248320]`
      logits tensor is materialized anyway, and slicing rows off it saves nothing
      that matters at this vocabulary.
    - **The batch must be padding-free.** `hidden_states[:, index, :]` selects the
      *same* positions in every row, which is only meaningful when there is one
      row. Under padded execution each row has its own supervised positions, so a
      shared index would train the wrong tokens -- silently.
    """
    if not requested:
        return Toggle(
            "selective_logit_loss",
            False,
            "disabled by config; the head projects every position, supervised or not",
        )
    if not liger_enabled:
        return Toggle(
            "selective_logit_loss",
            False,
            "requested but the fused head is off; dense logits are materialized "
            "regardless, so selecting rows saves nothing",
        )
    if not padding_free:
        return Toggle(
            "selective_logit_loss",
            False,
            "requested but this run uses padded batches; one shared position index "
            "cannot describe per-row supervision without training wrong tokens",
        )
    measured = (
        f"; shard supervises {supervised_fraction:.1%} of positions"
        if supervised_fraction is not None
        else ""
    )
    return Toggle(
        "selective_logit_loss",
        True,
        f"head projects supervised positions only, via logits_to_keep{measured}",
    )


def resolve_attn_implementation(
    requested: str,
    *,
    has_flash_attn: bool | None = None,
    has_cuda: bool | None = None,
    capability: tuple[int, int] | None = None,
) -> Toggle:
    """Pick an attention implementation, downgrading with a recorded reason.

    The `Toggle.name` is the implementation actually selected, so the caller reads
    the decision off one object rather than re-deriving it.

    Only two implementations are ever selected for `flash_attention_2`: the
    official FlashAttention-2 wheel on Ampere-plus, and `sdpa` everywhere else.
    FA2's kernels are Ampere-and-newer, so on Turing the wheel being installed is
    not enough -- the capability check is the deciding one.
    """
    flash = flash_attn_available() if has_flash_attn is None else has_flash_attn
    cuda = cuda_available() if has_cuda is None else has_cuda

    if requested != "flash_attention_2":
        return Toggle(requested, True, f"requested explicitly: {requested}")
    if not cuda:
        return Toggle("sdpa", False, "flash_attention_2 requested but no CUDA device; using sdpa")
    if capability is None:
        capability = cuda_capability()
    if capability is not None and capability < FLASH_ATTENTION_2_MIN_CAPABILITY:
        return Toggle(
            "sdpa",
            False,
            "flash_attention_2 requested but its kernels need compute capability "
            f"{FLASH_ATTENTION_2_MIN_CAPABILITY[0]}.{FLASH_ATTENTION_2_MIN_CAPABILITY[1]}; "
            f"sm{capability[0]}{capability[1]} uses sdpa",
        )
    if not flash:
        return Toggle(
            "sdpa",
            False,
            "flash_attention_2 requested but the flash_attn wheel is not installed; using sdpa",
        )
    return Toggle(
        "flash_attention_2",
        True,
        "flash_attention_2 on the 6 full-attention layers; GDN layers use their own kernels",
    )


def resolve_precision(bf16_requested: bool, *, capability: tuple[int, int] | None = None) -> Toggle:
    """Resolve the effective compute dtype for a training run.

    BF16 tensor cores arrive with Ampere. On Turing torch still *accepts* bf16 and
    emulates it, which is the wrong kind of quiet: the run is slower than FP16 and
    the numerics are not the BF16 ones an Ampere run produced. So a pre-Ampere card
    resolves to FP16 with the downgrade recorded in the ledger.

    Returned as a `Toggle` so the reason travels with the decision; the effective
    dtype name is `Toggle.name`.
    """
    if not bf16_requested:
        return Toggle("float32", True, "bf16 disabled by config; running float32")
    if capability is None:
        capability = cuda_capability()
    if capability is not None and capability < BF16_MIN_CAPABILITY:
        return Toggle(
            "float16",
            False,
            f"bf16 requested but sm{capability[0]}{capability[1]} has no bf16 tensor cores; "
            "running float16 under GradScaler. Numerics differ from a bf16 run -- "
            "do not compare the two as one experiment",
        )
    return Toggle("bfloat16", True, "bf16 on a card with bf16 tensor cores")


def _is_excluded(name: str, class_name: str, exclude_patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    if class_name in COMPILE_FORBIDDEN_CLASSES:
        return True
    return any(pattern.lower() in lowered for pattern in exclude_patterns)


def select_compile_targets(
    named_classes: Sequence[tuple[str, str]], exclude_patterns: Sequence[str]
) -> list[str]:
    """Choose which submodules to compile, given `(module_name, class_name)` pairs.

    Separated from the walk over a live model so the exclusion logic is testable
    without instantiating anything: pass the names and classes, get back the list
    that would be compiled.
    """
    selected: list[str] = []
    for name, class_name in named_classes:
        if not name:
            continue
        if not class_name.endswith(COMPILE_TARGET_CLASS_SUFFIXES):
            continue
        if _is_excluded(name, class_name, exclude_patterns):
            continue
        selected.append(name)
    return selected


def apply_regional_compile(
    model: Any, *, exclude_patterns: Sequence[str], enabled: bool = True
) -> Toggle:
    """Compile the selected submodules in place, leaving the mixer body eager.

    Compiles per module rather than wrapping the whole model, because a single
    `torch.compile(model)` traces straight through the GDN mixer and raises.
    """
    if not enabled:
        return Toggle("regional_torch_compile", False, "disabled by config; model runs eager")

    named_classes = [(name, type(module).__name__) for name, module in model.named_modules()]
    targets = select_compile_targets(named_classes, exclude_patterns)
    if not targets:
        return Toggle(
            "regional_torch_compile",
            False,
            f"no compilable submodule survived the exclusion list {list(exclude_patterns)}",
        )

    lookup = dict(model.named_modules())
    for name in targets:
        lookup[name].compile()
    excluded = sorted(
        {
            type(module).__name__
            for name, module in lookup.items()
            if name
            and name not in set(targets)
            and type(module).__name__ in COMPILE_FORBIDDEN_CLASSES
        }
    )
    return Toggle(
        "regional_torch_compile",
        True,
        f"compiled {len(targets)} submodules; left eager: {excluded or 'none'} "
        f"plus anything matching {list(exclude_patterns)}",
    )


def cast_adapters(model: Any, dtype_name: str) -> Toggle:
    """Cast trainable (LoRA) parameters to the safe effective dtype.

    PEFT upcasts adapters to fp32 by default, which exists for low-bit QLoRA
    bases. With all-linear targets on a bf16 base it instead forces an
    upcast/downcast plus an fp32 GEMM at every linear. FP16 is different:
    Accelerate's GradScaler rejects FP16 gradients, so FP16 mixed-precision
    training must retain trainable adapters in FP32.
    """
    import torch

    if dtype_name in {"float16", "float32"}:
        detail = (
            "adapters left in fp32; GradScaler rejects FP16 gradients"
            if dtype_name == "float16"
            else "adapters left in fp32 (PEFT default); every linear pays an fp32 GEMM"
        )
        return Toggle(
            "adapter_dtype",
            False,
            detail,
        )

    dtype = getattr(torch, dtype_name)
    count = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != dtype:
            param.data = param.data.to(dtype)
            count += 1
    return Toggle(
        "adapter_dtype",
        True,
        f"cast {count} trainable tensors to {dtype_name}, matching the base dtype",
    )


def ledger(toggles: Sequence[Toggle]) -> dict[str, str]:
    """Flatten toggles into the mapping that goes into the W&B run config."""
    return dict(toggle.as_config_entry() for toggle in toggles)


def format_ledger(toggles: Sequence[Toggle]) -> str:
    """Compact terminal summary; detailed reasons remain in the run config."""
    return "optimizations: " + ", ".join(
        f"{toggle.name}={'on' if toggle.enabled else 'off'}" for toggle in toggles
    )
