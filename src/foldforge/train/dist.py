"""Data parallelism for the custom training step WITHOUT ``DistributedDataParallel``.

The step (:func:`foldforge.train.step.micro_step`) runs a forward, several partial backwards and accumulates into ``param.grad``; DDP's
hooks do not fit that. Every rank therefore holds the whole model, runs its own micro steps, and after the accumulation loop the gradients
are summed over the ranks in a few large flattened buckets. The loss scale of a micro step is ``1 / (grad_accum_steps * world_size)``, so the
SUM over ranks equals the gradient a single process gets from ``grad_accum_steps * world_size`` micro steps with ``scale = 1 / that``.
Clip, optimizer and EMA then run replicated on every rank on identical gradients and identical parameters.

Micro steps are numbered globally: optimizer step ``it``, micro ``m`` on rank ``r`` of ``W`` ranks with ``A`` accumulation steps is
``(it * A + m) * W + r``. Rank ``r`` therefore takes the micro steps ``r, r + W, ...`` of a single-process run with ``A * W`` accumulation
steps (see :func:`foldforge.train.bench.micro_plan`), which makes the data-parallel run testable against the single-process one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as torch_dist

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class WorldInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def world_info(environ: Mapping[str, str] | None = None) -> WorldInfo:
    """``torchrun``'s environment (RANK / WORLD_SIZE / LOCAL_RANK); a plain process is rank 0 of 1."""
    env = os.environ if environ is None else environ
    return WorldInfo(
        rank=int(env.get("RANK", 0)),
        world_size=int(env.get("WORLD_SIZE", 1)),
        local_rank=int(env.get("LOCAL_RANK", 0)),
    )


def accum_for_effective_batch(effective_batch: int, world_size: int) -> int:
    """Micro steps per rank so that ``micro batch 1 x grad_accum x world_size == effective_batch``."""
    if effective_batch < 1 or world_size < 1:
        message = f"effective batch {effective_batch} / world size {world_size} must be positive"
        raise ValueError(message)
    if effective_batch % world_size:
        message = f"--effective-batch {effective_batch} is not divisible by the world size {world_size}"
        raise ValueError(message)
    return effective_batch // world_size


def init_process_group(info: WorldInfo, backend: str = "nccl") -> None:
    """Initialise ``torch.distributed`` under torchrun (a no-op for one process) and pin the rank's CUDA device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank % torch.cuda.device_count())
    if info.world_size > 1 and not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=backend)


def barrier(info: WorldInfo) -> None:
    if info.world_size > 1:
        torch_dist.barrier()


@torch.no_grad()
def broadcast_parameters(params: Iterable[torch.Tensor], info: WorldInfo, src: int = 0) -> None:
    """Make the parameters (and buffers passed in) identical on every rank: copy rank ``src``'s values, once, in a few flattened buckets."""
    if info.world_size == 1:
        return
    for bucket in _buckets([p.data for p in params], 2**26):
        flat = torch.cat([t.reshape(-1) for t in bucket])
        torch_dist.broadcast(flat, src=src)
        _scatter(flat, bucket)


def _buckets(tensors: list[torch.Tensor], max_numel: int) -> list[list[torch.Tensor]]:
    """Group consecutive tensors of one dtype / device into buckets of at most ``max_numel`` elements (a bigger tensor is its own bucket)."""
    buckets: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    size = 0
    for t in tensors:
        if current and (t.dtype != current[0].dtype or t.device != current[0].device or size + t.numel() > max_numel):
            buckets.append(current)
            current, size = [], 0
        current.append(t)
        size += t.numel()
    if current:
        buckets.append(current)
    return buckets


def _scatter(flat: torch.Tensor, tensors: list[torch.Tensor]) -> None:
    offset = 0
    for t in tensors:
        n = t.numel()
        t.copy_(flat[offset : offset + n].view_as(t))
        offset += n


@torch.no_grad()
def allreduce_gradients(params: Iterable[torch.Tensor], info: WorldInfo, bucket_numel: int = 2**26, group=None) -> int:
    """SUM the ``.grad`` of every parameter over the ranks, flattened into buckets of ``bucket_numel`` elements (default 64 Mi = 256 MiB fp32).
    A parameter without a gradient on this rank contributes zeros (every rank must call this with the same parameter list: a parameter some rank
    never reached still takes part). Returns the number of all-reduce calls."""
    if info.world_size == 1:
        return 0
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    calls = 0
    for bucket in _buckets(grads, bucket_numel):
        flat = torch.cat([g.reshape(-1) for g in bucket])
        torch_dist.all_reduce(flat, op=torch_dist.ReduceOp.SUM, group=group)
        _scatter(flat, bucket)
        calls += 1
    return calls


@torch.no_grad()
def mean_over_ranks(values: Mapping[str, float], info: WorldInfo, device: torch.device) -> dict[str, float]:
    """Average a dict of floats over the ranks (the printed loss parts); one all-reduce."""
    if info.world_size == 1:
        return dict(values)
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    torch_dist.all_reduce(t, op=torch_dist.ReduceOp.SUM)
    t /= info.world_size
    return dict(zip(keys, t.tolist(), strict=True))


@torch.no_grad()
def param_checksum(params: Iterable[torch.Tensor]) -> float:
    """sum(|p|) in float64: equal on every rank when the parameters are identical."""
    params = list(params)
    total = torch.zeros((), dtype=torch.float64, device=_first_device(params))
    for p in params:
        total += p.detach().double().abs().sum()
    return float(total)


def _first_device(params: Iterable[torch.Tensor]) -> torch.device:
    for p in params:
        return p.device
    return torch.device("cpu")


def ranks_agree(value: float, info: WorldInfo, device: torch.device) -> bool:
    """True when every rank holds exactly ``value`` (bitwise, as float64)."""
    if info.world_size == 1:
        return True
    t = torch.tensor([value], dtype=torch.float64, device=device)
    lo, hi = t.clone(), t.clone()
    torch_dist.all_reduce(lo, op=torch_dist.ReduceOp.MIN)
    torch_dist.all_reduce(hi, op=torch_dist.ReduceOp.MAX)
    return bool(lo.item() == hi.item())
