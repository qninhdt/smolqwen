"""The selective head must change what is computed, never what is learned.

Liger's fused head is chunked but not selective: it sizes chunks from the row
count and runs the 248,320-wide GEMM over every position, using the label mask
only to zero the loss afterwards. A full-trajectory shard therefore pays the
projection on prompt and observation tokens that contribute nothing -- and after
the non-reasoning switch that is the large majority of positions.

`logits_to_keep` as an index tensor is upstream's own contract for skipping them
(`modeling_qwen3_5.py:1643`; Liger's `lce_forward` forwards it unchanged). What
makes it safe rather than merely faster is that dropping a row whose target is
`-100` removes a term that was already zero. That claim is what these tests pin,
against the *real* patched forward rather than a reimplementation:

- The loss and every parameter gradient must match the dense path. Loss matches
  exactly (0.0e+00 over four seeds), which is the strong statement: the term set
  is identical. Gradients agree only to fp32 rounding, and the bound is not
  uniform -- selection changes how many fp32 chunks liger accumulates, and
  Qwen3.5's Gated DeltaNet layers amplify that ~14-43x through their recurrent
  state. Measured on the production layer mix: head 1e-6, non-GDN 5e-5, GDN 1e-4;
  with an all-attention model the same probe gives 1e-6 everywhere, which is what
  identifies GDN rather than the head as the amplifier.

  The falsifying control is in the test itself: an index covering *every*
  position, supervised rows first, is mathematically identical to dense and keeps
  the chunk layout -- it agrees to 6e-8. Selection differing by more than
  rounding would have to exceed that control by orders of magnitude, not by the
  amplification an equivalent reordering already shows.
- The refusals matter as much as the equivalence. Padded batches carry per-row
  supervision that one shared position index cannot describe, so selecting there
  would train the wrong tokens silently -- `resolve_selective_logits` must refuse,
  and the reason must be in the ledger.
"""

from __future__ import annotations

from typing import Any

import pytest

from smolqwen.training.collate import CollateError, supervised_positions
from smolqwen.training.optim import resolve_selective_logits

VOCAB = 256
LENGTH = 96
SUPERVISED_FRACTION = 0.235  # the measured non-reasoning share


def test_supervised_positions_selects_the_shifted_prediction_frame() -> None:
    """Position `i` predicts `labels[i + 1]`, so the rows to project sit one left."""
    import torch

    labels = torch.tensor([[-100, -100, 7, 8, -100, 9]])
    index, shift_labels = supervised_positions(labels)

    # Targets 7, 8, 9 live at label positions 2, 3, 5 -> rows 1, 2, 4.
    assert index.tolist() == [1, 2, 4]
    assert shift_labels.tolist() == [[7, 8, 9]]
    # The final position is never a prediction target: nothing follows it.
    assert int(index.max()) < labels.shape[-1] - 1


def test_supervised_positions_refuses_a_padded_batch() -> None:
    """Two rows have two different supervision patterns; one index cannot say so."""
    import torch

    with pytest.raises(CollateError, match="one flattened row"):
        supervised_positions(torch.tensor([[-100, 5], [6, -100]]))


def _labels(torch: Any, *, seed: int = 0) -> Any:
    """One flattened row whose supervised share matches a non-reasoning shard."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, VOCAB, (1, LENGTH), generator=generator)
    keep = torch.rand(1, LENGTH, generator=generator) < SUPERVISED_FRACTION
    return torch.where(keep, labels, torch.full_like(labels, -100))


def _model(torch: Any, dtype: Any) -> Any:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    from tests.helpers import tiny_qwen35_model

    torch.manual_seed(0)
    # Liger's kernels are Triton, so this comparison only exists on a device.
    model = tiny_qwen35_model(vocab_size=VOCAB).to(dtype).cuda().train()
    # The production path patches the fused head; a dense head would make the
    # comparison meaningless because it materializes every logit either way.
    _apply_liger_kernel_to_instance(model=model)
    return model


def _loss_and_grads(model: Any, input_ids: Any, labels: Any, *, selective: bool) -> Any:
    for parameter in model.parameters():
        parameter.grad = None
    if selective:
        index, shift_labels = supervised_positions(labels)
        output = model(input_ids=input_ids, shift_labels=shift_labels, logits_to_keep=index)
    else:
        output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.loss.detach().item(), grads


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("dtype_name", "tolerance"),
    [
        # fp32 bounds the GDN-amplified rounding measured over four seeds (worst
        # 1.4e-4); the reorder-only control below is what makes this a bound on
        # rounding rather than a fitted number.
        ("float32", 1e-3),
        # bf16 is only a gross-error guard -- its own reordering noise is ~1e-2.
        ("bfloat16", 5e-2),
    ],
)
def test_selective_head_learns_what_the_dense_head_learns(
    dtype_name: str, tolerance: float
) -> None:
    import torch

    dtype = getattr(torch, dtype_name)
    model = _model(torch, dtype)
    labels = _labels(torch).cuda()
    input_ids = torch.where(labels == -100, torch.zeros_like(labels), labels)
    index, _ = supervised_positions(labels)
    assert 0 < int(index.numel()) < LENGTH, "the fixture must actually mask something"

    dense_loss, dense_grads = _loss_and_grads(model, input_ids, labels, selective=False)
    selective_loss, selective_grads = _loss_and_grads(model, input_ids, labels, selective=True)

    # The strong claim: the term set is identical, so the loss is too.
    assert abs(dense_loss - selective_loss) <= 1e-6 * max(abs(dense_loss), 1.0), (
        f"loss diverged: dense {dense_loss!r} vs selective {selective_loss!r}"
    )
    assert selective_grads.keys() == dense_grads.keys()
    for name, dense_grad in dense_grads.items():
        scale = dense_grad.abs().max().clamp(min=1e-30)
        relative = ((dense_grad - selective_grads[name]).abs().max() / scale).item()
        assert relative <= tolerance, f"{name}: gradient relative diff {relative:.2e}"


@pytest.mark.gpu
@pytest.mark.slow
def test_a_mathematically_identical_reorder_agrees_far_more_tightly() -> None:
    """The control that makes the fp32 tolerance above a bound, not a fitted number.

    An index covering every position that can be a target -- supervised rows first,
    then the masked ones -- computes exactly the dense term set (a `-100` row
    contributes zero) while keeping liger's chunk layout. It agrees to ~1e-7. So
    the 1e-4 the selective path shows is the cost of *changing the chunk layout*,
    amplified through GDN's recurrent state, and a real semantic error would have
    to be far larger than either.
    """
    import torch

    model = _model(torch, torch.float32)
    labels = _labels(torch).cuda()
    input_ids = torch.where(labels == -100, torch.zeros_like(labels), labels)
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    supervised = shifted[0].ne(-100).nonzero(as_tuple=True)[0]
    masked = shifted[0].eq(-100).nonzero(as_tuple=True)[0]
    reordered = torch.cat([supervised, masked])

    _, dense_grads = _loss_and_grads(model, input_ids, labels, selective=False)
    for parameter in model.parameters():
        parameter.grad = None
    model(
        input_ids=input_ids,
        shift_labels=shifted[:, reordered],
        logits_to_keep=reordered,
    ).loss.backward()

    worst = max(
        (
            (dense_grads[name] - parameter.grad).abs().max()
            / dense_grads[name].abs().max().clamp(min=1e-30)
        ).item()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    assert worst <= 1e-6, f"an identical-math reorder diverged by {worst:.2e}"


@pytest.mark.gpu
@pytest.mark.slow
def test_the_selective_head_projects_fewer_rows_than_it_skips() -> None:
    """The point of the toggle, as a count rather than a claim.

    Asserted through the model's own logits shape: with `logits_to_keep` the head
    returns one row per supervised position, which is exactly the GEMM it ran.
    """
    import torch

    model = _model(torch, torch.float32)
    labels = _labels(torch).cuda()
    input_ids = torch.where(labels == -100, torch.zeros_like(labels), labels)
    index, shift_labels = supervised_positions(labels)

    # `skip_logits=False` forces the head to return logits so the row count is
    # observable; training uses the fused path, which computes the same rows.
    dense = model(input_ids=input_ids, labels=labels, skip_logits=False)
    selective = model(
        input_ids=input_ids,
        shift_labels=shift_labels,
        logits_to_keep=index,
        skip_logits=False,
    )
    assert dense.logits.shape[1] == LENGTH
    assert selective.logits.shape[1] == int(index.numel())
    assert selective.logits.shape[1] < dense.logits.shape[1]


def test_the_toggle_refuses_every_configuration_that_would_be_wrong() -> None:
    """Each refusal carries its reason: the ledger is how a silent downgrade shows."""
    enabled = resolve_selective_logits(
        True, liger_enabled=True, padding_free=True, supervised_fraction=0.235
    )
    assert enabled.enabled
    assert "23.5%" in enabled.detail

    off = resolve_selective_logits(
        False, liger_enabled=True, padding_free=True, supervised_fraction=0.235
    )
    assert not off.enabled
    assert "disabled by config" in off.detail

    # Without the fused head the dense logits tensor exists regardless, so
    # selecting rows off it saves nothing at this vocabulary.
    dense_head = resolve_selective_logits(
        True, liger_enabled=False, padding_free=True, supervised_fraction=0.235
    )
    assert not dense_head.enabled
    assert "fused head is off" in dense_head.detail

    # The refusal that prevents training the wrong tokens rather than merely
    # losing a speedup.
    padded = resolve_selective_logits(
        True, liger_enabled=True, padding_free=False, supervised_fraction=0.235
    )
    assert not padded.enabled
    assert "padded batches" in padded.detail
