"""Backward of the atom-layout gather (:func:`foldforge.utils.atom_layout.convert`) without torch's serial pile-ups, BITWISE equal to torch.

``arr[..., idx, ...]`` backpropagates through ``aten::_index_put_impl_(zeros, idx, grad, accumulate=True)``: a stable radix sort of ``idx``,
then ONE warp per distinct target walks every duplicate of that target in a serial loop (``indexing_backward_kernel*``, ATen
``native/cuda/Indexing.cu``). The atom layouts have pile-ups: padded query / key / atom-pair slots all point to one target (1751..19261
duplicates of one index on real Protenix v1 batches), so one warp walks thousands of rows while the GPU idles (~12 ms of a ~560 ms micro step).

What torch computes for a target ``t`` (per leading index ``z`` and feature ``f``; ``j_1 < j_2 < ...`` the positions with ``idx[j] == t`` in
the order of the stable sort) depends on the slice size ``S`` (product of the feature dims), as ATen dispatches it:

* ``S > 32`` (``indexing_backward_kernel``): ``w = +0``; ``w = round(float(w) + float(g[j_k]))`` for k = 1, 2, ... -- the running value is
  read from and written back to the ``dtype`` gradient after EVERY duplicate (``round``: round-to-nearest-even to ``dtype``; identity for fp32).
* ``2 <= S <= 32`` (``indexing_backward_kernel_small_stride``): ``s = 0.0f``; ``s += float(g[j_k])`` for k = 1, 2, ...; ``out = round(0.0f + s)``.
* ``S == 1`` (``indexing_backward_kernel_stride_1``, a warp-shuffle reduction): not reproduced; :func:`applies` keeps torch for it.

The kernel below runs the SAME sequences (one program per target x feature block x leading index, the duplicates in the same sorted order)
but SKIPS the contributions that are exactly ``±0`` -- a pile-up of padding gradients is mostly zeros. That leaves every bit unchanged:

* ``x + (±0) = x`` for every nonzero ``x`` (``NaN`` and ``inf`` count as nonzero and are added like any value, so they propagate as in torch),
  and ``round`` of a ``dtype`` value is the value itself.
* The running value is never ``-0``: it starts at ``+0``; ``+0 + (±0) = +0``; an exact cancellation rounds to ``+0``; and a nonzero sum of
  two ``dtype`` (or fp32) values cannot round to zero, both being multiples of the smallest subnormal of their type (no flush-to-zero: neither
  ATen nor this kernel is built with fast math). So a zero contribution never changes it.

A target with more than ``SUB`` duplicates (a pile-up) is screened ``BIG`` duplicates at a time, else ``SUB`` at a time; a tile in which every
feature column holds at most ONE nonzero contribution is applied in one vector step (the column sum of one value and zeros is that value,
exactly); any other ``SUB`` tile runs its duplicates one by one in sorted order.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # the kernels below are only defined when the runtime import works
    import triton
    import triton.language as tl

try:  # pragma: no cover - the module imports on a CPU-only box without triton
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

#: dtypes the kernel reproduces (ATen's opmath for all of them is fp32)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
#: ATen's dispatch threshold between the small-stride and the generic kernel (CUDA warp size)
_WARP = 32
#: duplicates per tile of the general path
_SUB = 16

if HAVE_TRITON:

    @triton.jit
    def _apply_tile(acc, c, nz, ROUND_EACH: tl.constexpr, out_ty: tl.constexpr):
        """Add a tile whose columns hold at most one nonzero contribution each: the column sum is that value, exactly."""
        cnt = tl.sum(nz.to(tl.int32), axis=0)
        v = tl.sum(tl.where(nz, c, 0.0), axis=0)
        if ROUND_EACH:
            return tl.where(cnt == 1, (acc + v).to(out_ty).to(tl.float32), acc)
        return tl.where(cnt == 1, acc + v, acc)

    @triton.jit
    def _gather_grad_kernel(
        grad_ptr,  # [outer, n_idx, S] contiguous: gradient of the gathered tensor
        order_ptr,  # [n_idx] int64: positions j sorted by target, stable (ascending j within a target)
        offs_ptr,  # [n_in + 1] int64: CSR offsets of each target into ``order``
        out_ptr,  # [outer, n_in, S]: gradient of the source
        n_idx,
        n_in,
        S,
        ROUND_EACH: tl.constexpr,  # True: the S > 32 sequence (round to the dtype after every add); False: fp32 sum, one rounding at the end
        BIG: tl.constexpr,  # screening tile of a pile-up: applied at once when no column holds two nonzeros
        SUB: tl.constexpr,  # tile of the general path
        BLOCK_F: tl.constexpr,
    ):
        t = tl.program_id(0)
        z = tl.program_id(2).to(tl.int64)
        f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
        fmask = f < S
        start = tl.load(offs_ptr + t)
        end = tl.load(offs_ptr + t + 1)
        gbase = grad_ptr + z * n_idx * S
        out_ty: tl.constexpr = out_ptr.dtype.element_ty
        rows_b = tl.arange(0, BIG)
        rows_s = tl.arange(0, SUB)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for kb in range(start, end, BIG):
            hi = tl.minimum(kb + BIG, end)
            done = 0
            if hi - kb > SUB:  # a pile-up: screen BIG duplicates at once
                kr = kb + rows_b
                rm = kr < hi
                js = tl.load(order_ptr + kr, mask=rm, other=0)
                m2 = rm[:, None] & fmask[None, :]
                c = tl.load(
                    gbase + js[:, None] * S + f[None, :], mask=m2, other=0.0
                ).to(tl.float32)
                nz = (
                    c != 0.0
                ) & m2  # NaN compares unequal to 0: a nonzero contribution
                if tl.max(tl.sum(nz.to(tl.int32), axis=0), axis=0) <= 1:
                    acc = _apply_tile(acc, c, nz, ROUND_EACH, out_ty)
                    done = 1
            if done == 0:
                for ks in range(kb, hi, SUB):
                    kr = ks + rows_s
                    rm = kr < hi
                    js = tl.load(order_ptr + kr, mask=rm, other=0)
                    m2 = rm[:, None] & fmask[None, :]
                    c = tl.load(
                        gbase + js[:, None] * S + f[None, :], mask=m2, other=0.0
                    ).to(tl.float32)
                    nz = (c != 0.0) & m2
                    if tl.max(tl.sum(nz.to(tl.int32), axis=0), axis=0) <= 1:
                        acc = _apply_tile(acc, c, nz, ROUND_EACH, out_ty)
                    else:  # the duplicates one by one, in sorted order
                        for r in tl.static_range(SUB):
                            ok = ks + r < hi
                            j = tl.load(order_ptr + ks + r, mask=ok, other=0)
                            cr = tl.load(
                                gbase + j * S + f, mask=fmask & ok, other=0.0
                            ).to(tl.float32)
                            nzr = (cr != 0.0) & fmask & ok
                            if ROUND_EACH:
                                acc = tl.where(
                                    nzr, (acc + cr).to(out_ty).to(tl.float32), acc
                                )
                            else:
                                acc = tl.where(nzr, acc + cr, acc)
        if not ROUND_EACH:
            acc = (
                0.0 + acc
            )  # ATen: float(+0 gradient) + sum (a no-op: the sum is never -0)
        tl.store(
            out_ptr + z * n_in * S + t.to(tl.int64) * S + f, acc.to(out_ty), mask=fmask
        )


def enabled() -> bool:
    """``FOLDFORGE_GATHER_GRAD=0`` turns the kernel off (torch's own backward everywhere)."""
    return HAVE_TRITON and os.environ.get("FOLDFORGE_GATHER_GRAD", "1") != "0"


def applies(arr: torch.Tensor, idx: torch.Tensor, dim: int) -> bool:
    """Whether :class:`IndexGather` serves ``arr[(:,)*dim, idx]``: a CUDA gather that records an autograd graph, a dtype and slice size whose
    torch backward is reproduced (``S >= 2``), int64 indices on the same device, outside torch.compile.
    """
    if not (
        torch.is_grad_enabled() and arr.requires_grad and arr.is_cuda and enabled()
    ):
        return False
    if arr.dtype not in DTYPES or idx.dtype != torch.int64 or idx.device != arr.device:
        return False
    if (
        math.prod(arr.shape[dim + 1 :]) < 2
        or idx.numel() == 0
        or arr.shape[dim] == 0
        or math.prod(arr.shape[:dim]) > 65535
    ):
        return False
    return not torch.compiler.is_compiling()


def index_grad(
    grad: torch.Tensor, idx: torch.Tensor, in_shape: torch.Size, dim: int
) -> torch.Tensor:
    """Gradient of ``x[(:,)*dim, idx]`` w.r.t. ``x`` (shape ``in_shape``) for the output gradient ``grad``: bitwise torch's ``index`` backward."""
    outer = math.prod(in_shape[:dim])
    n_in = in_shape[dim]
    s = math.prod(in_shape[dim + 1 :])
    flat = idx.reshape(-1)
    n_idx = flat.numel()
    # ATen wraps negative indices before it sorts
    flat = torch.where(flat < 0, flat + n_in, flat)
    keys, order = torch.sort(flat, stable=True)
    offs = torch.searchsorted(
        keys, torch.arange(n_in + 1, device=idx.device, dtype=keys.dtype)
    )
    g = grad.contiguous()
    out = torch.empty((outer, n_in, s), dtype=grad.dtype, device=grad.device)
    block_f = min(128, triton.next_power_of_2(s))
    _gather_grad_kernel[(n_in, triton.cdiv(s, block_f), outer)](
        g,
        order,
        offs,
        out,
        n_idx,
        n_in,
        s,
        ROUND_EACH=s > _WARP and grad.dtype != torch.float32,
        BIG=max(2 * _SUB, min(1024, 2048 // block_f)),
        SUB=_SUB,
        BLOCK_F=block_f,
        # num_warps is a launch option of the triton JIT launcher (not in its stub signature)
        num_warps=1,  # pyright: ignore[reportCallIssue]  # many small programs (one per target): measured fastest on the real Protenix v1 layouts (H100)
    )
    return out.view(in_shape)


class IndexGather(torch.autograd.Function):
    """``arr[(:,)*dim, idx]`` (the same ``aten::index`` forward) whose backward is :func:`index_grad` instead of torch's sort + serial
    ``index_put_``. Bitwise equal in forward and backward.
    """

    @staticmethod
    def forward(ctx, arr: torch.Tensor, idx: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.in_shape = arr.shape
        ctx.dim = dim
        return arr[(slice(None),) * dim + (idx,)]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]  # one output -> one grad, as autograd calls it
        (idx,) = ctx.saved_tensors
        return index_grad(grad, idx, ctx.in_shape, ctx.dim), None, None
