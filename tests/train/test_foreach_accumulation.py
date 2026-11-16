"""``train.step.foreach_accumulation``: bitwise autograd accumulation.

Several backwards accumulated with one multi-tensor add each give BITWISE the
gradients of autograd's own per-parameter accumulation, keep the ``.grad`` tensor
objects, and leave parameters without a gradient alone (CPU; CUDA when available).
"""

from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn

from foldforge.train.step import foreach_accumulation, trainable


class Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 32))
        self.head_a = nn.Linear(32, 5)
        self.head_b = nn.Linear(32, 3)
        self.late = nn.Linear(32, 2)  # first gets a gradient in a later backward
        self.unused = nn.Linear(2, 2)  # never gets one
        self.frozen = nn.Linear(32, 32)
        self.frozen.requires_grad_(requires_grad=False)


def _backwards(model, x, k) -> Iterator[Callable[[], None]]:
    """Yield three backwards per micro step, like ``step.micro_step_device``.

    Two heads on a leaf copy of the trunk output, then the trunk.
    """
    h = model.frozen(model.trunk(x))
    leaf = h.detach().requires_grad_(requires_grad=True)
    losses = [
        model.head_a(leaf).pow(2).sum() * 1e-3,
        (model.head_b(leaf).sin().sum() + (model.late(leaf).sum() if k > 0 else 0))
        * 0.37,
    ]
    for loss in losses:
        yield lambda loss=loss: loss.backward()
    yield lambda: torch.autograd.backward([h], [leaf.grad])


def _run(device, use_foreach, zero_in_place) -> Heads:
    torch.manual_seed(0)
    model = Heads().to(device)
    params = trainable(model)
    for k in range(4):
        x = torch.randn(64, 6, device=device) * (k + 1)
        if zero_in_place and k == 2:
            torch._foreach_zero_([p.grad for p in params if p.grad is not None])
        for run in _backwards(model, x, k):
            if use_foreach:
                with foreach_accumulation(params):
                    run()
            else:
                run()
    return model


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
    ],
)
@pytest.mark.parametrize("zero_in_place", [False, True])
def test_same_bits_as_autograd_accumulation(device, zero_in_place):
    ref = _run(device, use_foreach=False, zero_in_place=zero_in_place)
    ours = _run(device, use_foreach=True, zero_in_place=zero_in_place)
    for (name, p), q in zip(ref.named_parameters(), ours.parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None, name
            continue
        assert q.grad is not None, name
        assert torch.equal(p.grad.view(torch.int32), q.grad.view(torch.int32)), name
    assert ours.unused.weight.grad is None
    assert ours.frozen.weight.grad is None


def test_keeps_the_grad_tensor_objects():
    model = Heads()
    params = trainable(model)
    for p in params:
        p.grad = torch.zeros_like(p)
    before = [p.grad for p in params]
    with foreach_accumulation(params):
        model.head_a(model.trunk(torch.randn(4, 6))).sum().backward()
    assert all(p.grad is g for p, g in zip(params, before, strict=True))
    assert model.head_a.weight.grad.abs().sum() > 0
    assert model.unused.weight.grad.abs().sum() == 0


def test_error_restores_the_old_gradients():
    model = Heads()
    params = trainable(model)
    for p in params:
        p.grad = torch.ones_like(p)
    before = [p.grad for p in params]

    def _backward_then_fail() -> None:
        model.head_a(model.trunk(torch.randn(4, 6))).sum().backward()
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError), foreach_accumulation(params):
        _backward_then_fail()
    assert all(
        p.grad is g and bool((g == 1).all())
        for p, g in zip(params, before, strict=True)
    )


def test_switch_off(monkeypatch):
    monkeypatch.setenv("FOLDFORGE_FOREACH_GRAD_ACCUM", "0")
    model = Heads()
    params = trainable(model)
    for p in params:
        p.grad = torch.zeros_like(p)
    with foreach_accumulation(params):
        assert all(
            p.grad is not None for p in params
        )  # autograd accumulates into them itself
