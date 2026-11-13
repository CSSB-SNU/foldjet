"""The optimizer step (global grad-norm clip + Adam + EMA) as a few multi-tensor launches; no per-tensor python loop, no host sync.

Same recipe as :mod:`foldforge.train.optim` (Adam(0.9, 0.95), weight_decay 0, eps 1e-8, the step-decay schedule ``optim.lr_factor``, EMA lerp over
``model.named_parameters()``), selected by ``--fused-opt``; the old path stays. What changes:

* clip: ``torch._foreach_norm`` over the gradients, ``stack`` + one norm, ``torch._foreach_mul_`` by a device scalar. The sum is therefore
  ordered like (per-tensor norms -> norm of the stack), the same as the loop it replaces, so the clip scale is bit-identical when the
  per-tensor norm kernels agree (tests/train/test_fused_opt.py quantifies it).
* Adam: ``torch.optim.Adam`` with ``capturable=True`` (CUDA) and a CUDA ``lr`` tensor per group: the step counter and the lr live on the
  device, a CUDA graph changes the lr by ``fill_`` (:meth:`FusedOptimizer.set_lr`) instead of re-capturing. ``impl='fused'`` (default: torch's fused Adam, capturable, 358 launches), ``'foreach'` (capturable foreach Adam: 9030 launches, measured SLOWER than the eager loop) or ``'triton'`` (whole step in 3 Triton launches, see the end of the file).
* EMA: ``torch._foreach_lerp_`` over cached tensor lists.

Parameters are the fp32 masters (frozen ones, ``requires_grad`` False, are not in the optimizer but in the EMA, as before). A parameter whose
``.grad`` is None is skipped by the clip and by Adam. Order in a data-parallel step is unchanged: ``dist.allreduce_gradients`` first, then
:meth:`FusedOptimizer.step`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from foldforge.train import optim

if TYPE_CHECKING:
    from torch import nn

    from foldforge.train.config import TrainConfig


@torch.no_grad()
def foreach_clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Global L2 clip with multi-tensor launches; returns the pre-clip norm (a device tensor). Same formula as ``optim.clip_grad_norm``."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    # fp32 masters on one device (the model's); _foreach_norm accumulates in fp32 whatever the gradient dtype
    total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, scale)
    return total


class FusedEMA(optim.EMA):
    """:class:`optim.EMA` whose ``update`` is one ``_foreach_lerp_`` over cached lists (same shadow dict, checkpoint format and swap)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__(model, decay)
        names = [n for n, _ in model.named_parameters()]
        self._shadow_list = [self.shadow[n] for n in names]
        self._param_list = [p for _, p in model.named_parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module | None = None) -> None:
        torch._foreach_lerp_(self._shadow_list, [p.detach() for p in self._param_list], 1.0 - self.decay)


class FusedOptimizer:
    """``optimizer`` (torch Adam, capturable on CUDA), ``ema`` and the clip as one :meth:`step`."""

    def __init__(self, model: nn.Module, cfg: TrainConfig, impl: str = "fused") -> None:
        if impl not in ("foreach", "fused", "triton"):
            raise ValueError(f"impl must be 'foreach', 'fused' or 'triton', got {impl!r}")
        self.cfg = cfg
        self.impl = impl
        trunk, heads = optim.split_parameters(model)
        cuda = any(p.is_cuda for _, p in trunk + heads) and impl != "triton"  # triton: Adam state is ours, lr tensor lives in TritonStep
        self.capturable = cuda
        self._opt_names = [n for n, _ in heads + trunk]
        device = next(p.device for _, p in trunk + heads) if cuda else None

        def lr(value: float):
            return torch.tensor(value, dtype=torch.float32, device=device) if cuda else value

        groups = [
            {"params": [p for _, p in heads], "lr": lr(cfg.max_lr), "scale": 1.0, "name": "heads"},
            {"params": [p for _, p in trunk], "lr": lr(cfg.max_lr * cfg.trunk_lr_scale), "scale": cfg.trunk_lr_scale, "name": "trunk"},
        ]
        kw = {"fused": True} if impl == "fused" else {"foreach": impl == "foreach"}
        self.optimizer = torch.optim.Adam(groups, betas=cfg.betas, capturable=cuda, **kw)
        self.params = [p for g in self.optimizer.param_groups for p in g["params"]]
        self.ema = FusedEMA(model, cfg.ema_decay) if cfg.use_ema else None
        self.model = model
        self.triton = None
        if impl == "triton":
            in_opt = {id(p) for p in self.params}
            frozen = [n for n, p in model.named_parameters() if id(p) not in in_opt]
            allp = dict(model.named_parameters())
            self.ema_frozen_shadow = [self.ema.shadow[n] for n in frozen] if self.ema is not None else []
            self.ema_frozen_params = [allp[n].detach() for n in frozen]
            self.triton = TritonStep(self)

    def set_lr(self, step: int, cfg: TrainConfig | None = None) -> float:
        """Same schedule as ``optim.set_lr`` (warmup, step decay, ``min_lr`` floor); the lr tensors are filled in place (graph friendly)."""
        cfg = cfg or self.cfg
        factor = optim.lr_factor(step, cfg)
        base = max(cfg.min_lr, cfg.max_lr * factor) if step >= cfg.warmup_steps else cfg.max_lr * factor
        for group in self.optimizer.param_groups:
            value = base * group["scale"]
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value
        if self.triton is not None:
            self.triton.lrs.copy_(torch.tensor([g["lr"] for g in self.optimizer.param_groups], dtype=torch.float32))
        return base

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, max_norm: float | None = None) -> torch.Tensor:
        """Clip -> Adam -> EMA (after the gradient all-reduce). Returns the pre-clip global norm as a device tensor."""
        if self.triton is not None:
            return self.triton.step(self.cfg.grad_clip_max_norm if max_norm is None else max_norm)
        norm = foreach_clip_grad_norm(self.params, self.cfg.grad_clip_max_norm if max_norm is None else max_norm)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return norm


def build_fused(model: nn.Module, cfg: TrainConfig, impl: str = "fused") -> FusedOptimizer:
    return FusedOptimizer(model, cfg, impl)


# ----------------------------------------------------------------------------------------------------------------------------------
# impl='triton': the whole step in ~15 launches with NO per-call host work per tensor except one pointer-table upload.
# Portable Triton (no TMA): every tensor is cut into BLOCK-element chunks; a table of device pointers lets one launch walk all tensors.
#   K1  per-chunk sum of squared gradients -> K2 one program: global norm, clip coefficient -> K3 per chunk: g * coef -> Adam (torch's
#   FusedAdam math: lerp m, v = b2 v + (1 - b2) g g, p -= lr / bc1 * m / (sqrt(v) / sqrt(bc2) + eps)) and the EMA lerp of the new p.
# The step counters live on the device (one fp32 vector, ``state[p]['step']`` is a view of it); the lrs are a device vector (``set_lr``
# fills it), so a CUDA graph changes the lr by ``fill_``. The clipped gradient is NOT written back (p.grad keeps the raw value).
# ----------------------------------------------------------------------------------------------------------------------------------
try:  # pragma: no cover - the module imports on a CPU-only box without triton
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

_BLOCK = 4096

if HAVE_TRITON:

    @triton.jit
    def _sumsq_kernel(gptr, numel, chunk_t, chunk_off, part, BLOCK: tl.constexpr):
        c = tl.program_id(0)
        t = tl.load(chunk_t + c)
        off = tl.load(chunk_off + c).to(tl.int64)
        n = tl.load(numel + t)
        g = tl.load(gptr + t).to(tl.pointer_type(tl.float32))
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(g + idx, mask=idx < n, other=0.0)
        tl.store(part + c, tl.sum(x * x, axis=0))

    @triton.jit
    def _norm_kernel(part, nchunks, scal, max_norm, BLOCK: tl.constexpr):
        acc = tl.zeros([BLOCK], tl.float32)
        for i in range(0, nchunks, BLOCK):
            o = i + tl.arange(0, BLOCK)
            acc += tl.load(part + o, mask=o < nchunks, other=0.0)
        total = tl.sqrt_rn(tl.sum(acc, axis=0))
        tl.store(scal, total)
        tl.store(scal + 1, tl.minimum(tl.div_rn(max_norm, total + 1e-6), 1.0))

    @triton.jit
    def _update_kernel(
        pptr, gptr, mptr, vptr, sptr, numel, chunk_t, chunk_off, step_size, bc2s, scal, b1, b2, eps, ema_w, BLOCK: tl.constexpr, HAS_EMA: tl.constexpr
    ):
        c = tl.program_id(0)
        t = tl.load(chunk_t + c)
        off = tl.load(chunk_off + c).to(tl.int64)
        n = tl.load(numel + t)
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n
        P = tl.load(pptr + t).to(tl.pointer_type(tl.float32))
        G = tl.load(gptr + t).to(tl.pointer_type(tl.float32))
        M = tl.load(mptr + t).to(tl.pointer_type(tl.float32))
        V = tl.load(vptr + t).to(tl.pointer_type(tl.float32))
        coef = tl.load(scal + 1)
        ss = tl.load(step_size + t)
        bcs = tl.load(bc2s + t)
        g = tl.load(G + idx, mask=mask, other=0.0) * coef
        m = tl.load(M + idx, mask=mask, other=0.0)
        v = tl.load(V + idx, mask=mask, other=0.0)
        p = tl.load(P + idx, mask=mask, other=0.0)
        m = m + (g - m) * (1.0 - b1)
        v = v * b2 + (1.0 - b2) * g * g
        denom = tl.div_rn(tl.sqrt_rn(v), bcs) + eps
        p = p - ss * tl.div_rn(m, denom)
        tl.store(M + idx, m, mask=mask)
        tl.store(V + idx, v, mask=mask)
        tl.store(P + idx, p, mask=mask)
        if HAS_EMA:
            S = tl.load(sptr + t).to(tl.pointer_type(tl.float32))
            s = tl.load(S + idx, mask=mask, other=0.0)
            tl.store(S + idx, s + (p - s) * ema_w, mask=mask)


class TritonStep:
    """Clip + Adam + EMA for the optimizer parameters in three Triton launches (see the block comment above)."""

    def __init__(self, fo: FusedOptimizer) -> None:
        if not HAVE_TRITON:
            raise RuntimeError("impl='triton' needs triton")
        cfg = fo.cfg
        if not (1.0 - cfg.betas[0] < 0.5):
            raise ValueError("triton impl: lerp weight 1 - beta1 must be < 0.5 (torch's lerp formula branch)")
        self.fo = fo
        self.params = fo.params
        dev = self.params[0].device
        self.dev = dev
        T = len(self.params)
        gid = []
        for gi, g in enumerate(fo.optimizer.param_groups):
            gid += [gi] * len(g["params"])
        self.gid = torch.tensor(gid, dtype=torch.int64, device=dev)
        self.lrs = torch.zeros(len(fo.optimizer.param_groups), dtype=torch.float32, device=dev)
        self.steps = torch.zeros(T, dtype=torch.float32, device=dev)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        for i, p in enumerate(self.params):  # torch Adam's state layout (checkpoint.save -> optimizer.state_dict())
            fo.optimizer.state[p] = {"step": self.steps[i], "exp_avg": self.m[i], "exp_avg_sq": self.v[i]}
        self._ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64).to(dev)
        self._sig = None
        self.ema = fo.ema
        self.shadow = [fo.ema.shadow[n] for n in fo._opt_names] if fo.ema is not None else None
        self.hp = (float(cfg.betas[0]), float(cfg.betas[1]), 1e-8, 1.0 - float(cfg.ema_decay))
        self.part = None
        self.scal = torch.zeros(2, dtype=torch.float32, device=dev)

    def _tables(self, sig: tuple[bool, ...]) -> None:
        dev = self.dev
        act = [i for i, a in enumerate(sig) if a]
        self.act_list = act
        self.act = torch.tensor(act, dtype=torch.int64, device=dev)
        ps = [self.params[i] for i in act]
        for p in ps:
            assert p.dtype == torch.float32 and p.is_contiguous() and p.grad.is_contiguous() and p.grad.dtype == torch.float32, "triton impl: fp32 contiguous"
        numel = torch.tensor([p.numel() for p in ps], dtype=torch.int64, device=dev)
        counts = (numel + _BLOCK - 1) // _BLOCK
        self.nchunks = int(counts.sum())
        ids = torch.arange(len(act), device=dev)
        self.chunk_t = torch.repeat_interleave(ids, counts).to(torch.int32)
        starts = torch.cumsum(counts, 0) - counts
        self.chunk_off = ((torch.arange(self.nchunks, device=dev) - starts[self.chunk_t.long()]) * _BLOCK).to(torch.int32)
        self.numel = numel
        self.pptr = self._ptr(ps)
        self.mptr = self._ptr([self.m[i] for i in act])
        self.vptr = self._ptr([self.v[i] for i in act])
        self.sptr = self._ptr([self.shadow[i] for i in act]) if self.shadow is not None else self.pptr
        self.mask_f = torch.tensor([float(a) for a in sig], dtype=torch.float32, device=dev)
        self.part = torch.empty(self.nchunks, dtype=torch.float32, device=dev)
        # EMA of the optimizer parameters that got no gradient (and of the frozen ones): foreach lerp, shadow follows the unchanged parameter
        rest = [i for i, a in enumerate(sig) if not a]
        self.rest = (
            [self.shadow[i] for i in rest] + list(self.fo.ema_frozen_shadow),
            [self.params[i].detach() for i in rest] + list(self.fo.ema_frozen_params),
        ) if self.shadow is not None else None
        self._sig = sig

    @torch.no_grad()
    def step(self, max_norm: float) -> torch.Tensor:
        sig = tuple(p.grad is not None for p in self.params)
        if not any(sig):
            return torch.zeros(())
        if sig != self._sig:
            self._tables(sig)
        gptr = self._ptr([self.params[i].grad for i in self.act_list])  # new grad tensors after zero_grad(set_to_none): one small H2D copy per step
        b1, b2, eps, ema_w = self.hp
        _sumsq_kernel[(self.nchunks,)](gptr, self.numel, self.chunk_t, self.chunk_off, self.part, BLOCK=_BLOCK)
        _norm_kernel[(1,)](self.part, self.nchunks, self.scal, max_norm, BLOCK=1024)
        self.steps += self.mask_f  # the counters of the parameters that got a gradient (torch Adam skips the others)
        s = self.steps.double()
        bc1 = 1.0 - b1**s
        bc2s = torch.sqrt(1.0 - b2**s)
        step_size = (self.lrs[self.gid].double() / bc1).float()
        _update_kernel[(self.nchunks,)](
            self.pptr, gptr, self.mptr, self.vptr, self.sptr, self.numel, self.chunk_t, self.chunk_off,
            step_size[self.act], bc2s.float()[self.act], self.scal, b1, b2, eps, ema_w, BLOCK=_BLOCK, HAS_EMA=self.shadow is not None,
        )
        if self.rest is not None and self.rest[0]:
            torch._foreach_lerp_(self.rest[0], self.rest[1], ema_w)
        return self.scal[0]

