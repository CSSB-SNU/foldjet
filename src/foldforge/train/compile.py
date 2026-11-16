"""``torch.compile`` of the training step's repeated blocks.

``TrainConfig.compile``, ``scripts/train_protenix1.py --compile``; default off: the
eager step stays the default and the fallback.

Granularity: the ``forward`` of one block (the engine ``PairformerBlock`` / FoldForge
``EvoformerBlock`` of the pair stack, ``TokenDiTProtenix`` / ``AtomDiTProtenix`` of the
diffusion head), NOT the whole recycle or step: the micro step has host control flow
(random recycle count, chunk loops, host reads of the loss parts) and a whole-step
trace would specialise on all of it. The blocks are called 100+ times per micro step
on identical shapes, so one compiled graph per (class, grad mode, input signature)
serves all instances: the nn.Module is a call argument, its parameters graph inputs.

Rules (the MiniWorld training precedent, ``model.compile(dynamic=False)`` + bucketed
shapes, and the engine measurer's ``compile+graph`` mode):

* ``dynamic=False``; shapes are bucketed by the data path (``--bench-bucket`` / the
  recipe's crop), so a class has a bounded set of variants: no-grad recycles, grad /
  checkpoint recompute, 48-sample diffusion training, 1-sample rollout.
  ``recompile_limit`` is raised to hold them; past it dynamo runs the block eagerly
  (bounded cost, never an unbounded compile loop).
* ``triton.cudagraphs`` off: the step graph (``--step-graph``) is OUR capture,
  inductor's cudagraph trees would nest in it.
* ``fallback_random``: inductor draws the dropout scales with eager's RNG ops, so a
  compiled step consumes the CUDA generator exactly as the eager step (same seed ->
  same random numbers, and CUDA-graph capture sees the registered generator).
* After the warm-up nothing may compile: :func:`strict` (``error_on_recompile``) is set
  for the timed region / the kept graph capture.
"""

from __future__ import annotations

import contextlib
import types
from typing import TYPE_CHECKING, Any

import torch
import torch._dynamo
from torch._dynamo.utils import counters

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch import nn

#: Region name -> module classes whose forward is compiled.
REGIONS: dict[str, tuple[str, ...]] = {
    "blocks": ("PairformerBlock", "EvoformerBlock"),
    "diffusion": ("TokenDiTProtenix", "AtomDiTProtenix"),
}
REGIONS["all"] = REGIONS["blocks"] + REGIONS["diffusion"]

OPTIONS: dict[str, str | int | bool | Callable[..., Any]] = {
    "triton.cudagraphs": False,
    "fallback_random": True,
}
RECOMPILE_LIMIT = 32


def _call(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module._ff_eager_forward(*args, **kwargs)  # noqa: SLF001


def enable_compile(model: nn.Module, region: str = "all") -> int:
    """Compile the ``forward`` of every module of ``REGIONS[region]``.

    The forward is replaced by a call through ONE ``torch.compile(dynamic=False)``
    function. Call it BEFORE
    :func:`foldforge.train.prepare.enable_activation_checkpointing` (the checkpoint
    wrapper then recomputes the compiled forward). Returns the module count.
    """
    names = REGIONS[region]
    config = torch._dynamo.config  # noqa: SLF001
    for key in ("recompile_limit", "cache_size_limit"):
        if hasattr(config, key):
            setattr(config, key, RECOMPILE_LIMIT)
    for key in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(config, key):
            setattr(config, key, 8 * RECOMPILE_LIMIT)
    compiled = torch.compile(_call, dynamic=False, options=OPTIONS)
    count = 0
    for module in model.modules():
        if type(module).__name__ not in names or hasattr(module, "_ff_eager_forward"):
            continue
        module._ff_eager_forward = module.forward  # noqa: SLF001  # pyright: ignore[reportArgumentType]  # a bound method stored on the module
        module.forward = types.MethodType(
            lambda self, *a, **k: compiled(self, *a, **k),  # noqa: PLW0108 - bind a plain function, not the compiled wrapper
            module,
        )
        count += 1
    return count


@contextlib.contextmanager
def strict() -> Iterator[None]:
    """Allow no compilation inside (``error_on_recompile``).

    The warm-up has compiled every variant of the step.
    """
    with torch._dynamo.config.patch(error_on_recompile=True):  # noqa: SLF001
        yield


def report() -> dict[str, object]:
    """Dynamo counters since the process start.

    Compiled graphs, frames, graph breaks (reason -> count) and recompiles.
    """
    stats, frames = counters["stats"], counters["frames"]
    return {
        "unique_graphs": int(stats["unique_graphs"]),
        "calls_captured": int(stats["calls_captured"]),
        "frames_ok": int(frames["ok"]),
        "frames_total": int(frames["total"]),
        "graph_breaks": {str(k): int(v) for k, v in counters["graph_break"].items()},
        "recompiles": int(sum(counters["recompiles"].values()))
        if "recompiles" in counters
        else 0,
    }
