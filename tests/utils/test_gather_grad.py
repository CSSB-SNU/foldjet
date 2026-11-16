"""``foldforge.utils.gather_grad`` against torch's own ``index`` backward, BITWISE (CUDA): every dtype / slice size / leading-dim layout the
atom layouts use, pile-ups of thousands of duplicates on one target, exact zeros of both signs, subnormals, inf / NaN, overflow; and
``atom_layout.convert`` (forward + backward) with the kernel on and off."""

import pytest
import torch

from foldforge.utils import atom_layout, gather_grad

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and gather_grad.HAVE_TRITON), reason="CUDA + triton required")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _torch_grad(shape, dtype, idx, dim, grad):
    x = torch.zeros(shape, dtype=dtype, device="cuda", requires_grad=True)
    (gx,) = torch.autograd.grad(x[(slice(None),) * dim + (idx,)], x, grad)
    return gx


def _indices(n_in, idx_shape, gen, pile=0.0, target=0, negative=False):
    n = 1
    for s in idx_shape:
        n *= s
    idx = torch.randint(0, n_in, (n,), generator=gen)
    if pile:
        idx[torch.rand(n, generator=gen) < pile] = target  # padding slots: one target, thousands of times
    if negative:
        flip = torch.rand(n, generator=gen) < 0.3
        idx[flip] -= n_in  # ATen wraps negative indices
    return idx.view(idx_shape).cuda()


def _gradient(shape, dtype, gen, zeros=0.5):
    g = torch.randn(shape, generator=gen) * torch.exp(torch.randn(shape, generator=gen) * 3)
    z = torch.rand(shape, generator=gen)
    g[z < zeros] = 0.0
    g[(z < zeros) & (torch.rand(shape, generator=gen) < 0.5)] = -0.0  # both signs of zero
    g = g.to(dtype)
    flat = g.view(-1)
    k = flat.numel()
    pick = torch.randint(0, k, (8,), generator=gen)
    tiny = torch.finfo(dtype).smallest_normal
    flat[pick[0]] = tiny / 4  # subnormals
    flat[pick[1]] = -tiny / 2
    flat[pick[2]] = torch.finfo(dtype).max  # overflow when added up
    flat[pick[3]] = torch.finfo(dtype).max
    return g.cuda()


CASES = [
    # (leading, n_in, idx_shape, slice shape, pile)
    ((), 384, (128, 32), (128,), 0.4),  # queries -> tokens (generic kernel), pile on the padding token
    ((), 9216, (128, 32), (128,), 0.4),  # token atoms -> queries
    ((), 4096, (128, 128), (128,), 0.05),  # queries -> keys
    ((), 4096, (384, 24), (384,), 0.7),  # queries -> token atoms, S = 384
    ((), 147456, (64, 32, 128), (16,), 0.5),  # token pair -> atom pair (small-stride kernel), 19k-deep pile
    ((48,), 9216, (128, 32), (128,), 0.4),  # leading sample dim
    ((48,), 4096, (384, 24), (3,), 0.7),  # coordinates, S = 3
    ((2, 3), 500, (700,), (2,), 0.3),
    ((), 300, (900,), (32,), 0.3),  # last small-stride size
    ((), 300, (900,), (33,), 0.3),  # first generic size
    ((), 300, (900,), (4, 5), 0.3),  # two feature dims
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_index_grad_bitwise(dtype, case):
    lead, n_in, idx_shape, feat, pile = CASES[case]
    gen = torch.Generator().manual_seed(case * 7 + int(dtype.itemsize))
    dim = len(lead)
    shape = (*lead, n_in, *feat)
    for trial in range(2):
        idx = _indices(n_in, idx_shape, gen, pile=pile, target=n_in - 1 if trial else 0, negative=trial == 1)
        grad = _gradient((*lead, *idx_shape, *feat), dtype, gen)
        if trial == 1:
            g = grad.view(-1)
            g[torch.randint(0, g.numel(), (3,), generator=gen)] = float("nan")
            g[torch.randint(0, g.numel(), (3,), generator=gen)] = float("inf")
            g[torch.randint(0, g.numel(), (3,), generator=gen)] = float("-inf")
        ref = _torch_grad(shape, dtype, idx, dim, grad)
        ours = gather_grad.index_grad(grad, idx, torch.Size(shape), dim)
        assert ours.shape == ref.shape and ours.dtype == ref.dtype
        assert torch.equal(_bits(ours), _bits(ref)), f"case {case} trial {trial}: {(_bits(ours) != _bits(ref)).sum().item()} elements differ"


def test_index_grad_dense_duplicates_bitwise():
    """Every tile holds many nonzero duplicates of a target (the one-by-one path), plus all-zero and one-nonzero tiles."""
    gen = torch.Generator().manual_seed(3)
    for s, dtype in [(128, torch.bfloat16), (16, torch.bfloat16), (384, torch.float16)]:
        n_in = 7
        idx = torch.randint(0, n_in, (5000,), generator=gen).cuda()
        grad = _gradient((5000, s), dtype, gen, zeros=0.0)
        grad[idx == 3] = 0.0  # a target whose whole pile is zero
        sparse = (idx == 5).nonzero().view(-1)
        grad[sparse[(torch.rand(len(sparse), generator=gen) < 0.97).cuda()]] = 0.0  # mostly zero: one-nonzero tiles
        ref = _torch_grad((n_in, s), dtype, idx, 0, grad)
        ours = gather_grad.index_grad(grad, idx, torch.Size((n_in, s)), 0)
        assert torch.equal(_bits(ours), _bits(ref))


def test_convert_forward_backward_same_bits(monkeypatch):
    """``atom_layout.convert`` with the kernel and with torch's backward: identical output and input gradient (mask applied, padding pile-up)."""
    gen = torch.Generator().manual_seed(11)
    idx = _indices(384, (128, 32), gen, pile=0.4, target=383)
    mask = (torch.rand((128, 32), generator=gen) < 0.6).cuda()
    info = atom_layout.GatherInfo(gather_idxs=idx, gather_mask=mask, input_shape=(384,))
    base = torch.randn((48, 384, 128), generator=gen).to(torch.bfloat16).cuda()
    grad = _gradient((48, 128, 32, 128), torch.bfloat16, gen)
    outs, grads = [], []
    for flag in ("1", "0"):
        monkeypatch.setenv("FOLDFORGE_GATHER_GRAD", flag)
        x = base.clone().requires_grad_(True)
        y = atom_layout.convert(info, x, layout_axes=(1,))
        (gx,) = torch.autograd.grad(y, x, grad)
        outs.append(y.detach())
        grads.append(gx)
    assert torch.equal(_bits(outs[0]), _bits(outs[1]))
    assert torch.equal(_bits(grads[0]), _bits(grads[1]))


def test_applies_only_where_torch_is_reproduced(monkeypatch):
    x = torch.zeros((10, 4), device="cuda", requires_grad=True)
    idx = torch.zeros(3, dtype=torch.long, device="cuda")
    assert gather_grad.applies(x, idx, 0)
    assert not gather_grad.applies(x.detach(), idx, 0)  # no autograd graph
    assert not gather_grad.applies(torch.zeros(10, device="cuda", requires_grad=True), idx, 0)  # S == 1: torch's warp reduction
    assert not gather_grad.applies(x, idx.int(), 0)
    with torch.no_grad():
        assert not gather_grad.applies(x, idx, 0)
    monkeypatch.setenv("FOLDFORGE_GATHER_GRAD", "0")
    assert not gather_grad.applies(x, idx, 0)
