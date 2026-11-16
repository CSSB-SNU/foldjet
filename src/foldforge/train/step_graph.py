"""The WHOLE training micro step (trunk recycles + diffusion forward / backward + distogram + trunk backward + the 20-step mini-rollout + confidence
forward / backward, gradients accumulated into the persistent ``.grad`` tensors) as ONE CUDA graph per ``(passes, shape signature)``
(``TrainConfig.step_graph``, ``scripts/train_protenix1.py --step-graph``; default off: the eager path of ``train/step.py`` stays valid).

Eagerly the micro step is ~52k kernel launches for ~0.6 s of GPU work (launch bound, ~70 % idle). A graph replays them with one launch.

Design (deviations from the plan 'T+G split recycle graphs' are listed in the commit / report):

* ONE graph per micro step instead of a T (trunk) graph and a G (rest) graph: nothing between them needs the host (the loss parts stay on the
  device, ``step.DeviceParts``), and the mini-rollout is recorded INSIDE the graph as its plain op sequence (``rollout_graph.rollout_body``;
  a CUDA graph cannot launch another graph during capture, and its inputs, the detached trunk outputs, are produced in the same graph). The
  rollout graph of WP-B (``--rollout-order graph``) is therefore not used while a step graph serves a micro step (it still serves the eager
  fallback). One graph per recycle count (1..4): the number of ``no_grad`` recycles is part of the recorded op sequence.
* The graph is keyed on ``(passes, scale, shape signature of the example, cfg, model, precision state)``. The signature is the layout of every
  feature tensor (the host ``*:input_shape`` metadata only by shape: its value is a validity check, the real token count differs per example): with ``--bench-bucket`` every example has the same signature, so there
  are at most ``recycles_max`` graphs. Without bucketing every distinct shape signature is another graph: at most ``max_graphs`` are made, the
  rest of the keys run the eager step (counted in ``fallbacks``; a warning names ``--bench-bucket``).
* Inputs: the first example of a key is cloned into STATIC buffers; every micro step ``copy_``s the new example into them (device tensors only; the
  host metadata is part of the key). The loss parts are static outputs; a replay hands out CLONES (``DeviceParts``).
* Gradients: the graph ACCUMULATES into ``param.grad`` in place, so those tensors must exist before the capture and must never be freed:
  the capture saves the caller's gradients, runs the warm-up, zero-fills / restores them IN PLACE, and the training loop zeroes them with
  ``zero_()`` (``zero_grads``) instead of ``set_to_none``.
* Weights: nothing is declared static for the engine (no ``static_weights`` / ``static_inputs``): pack and table builds are recorded in the capture
  and run at every replay from the current weights (WP-B lesson). The warm-up bumps ``_version`` of every parameter so the capture records them
  itself; an optimizer that does not bump ``_version`` (fused Adam) must call :func:`rollout_graph.bump_versions` for the EAGER caches.
* Random numbers (augmentation, noise, dropout, MSA draw, rollout) come from the default CUDA generator, which a capture registers with the graph:
  a replay draws NEW numbers from the generator's current seed / offset (``torch.manual_seed`` before a replay seeds it). The stream differs from the
  eager one (other op order), the distribution is the same.
* Activation checkpointing is not special-cased: ``torch.utils.checkpoint`` (non-reentrant) is recorded like any op when it is on.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch

from foldforge.modules.engine import capture as engine_capture
from foldforge.train import compile as train_compile
from foldforge.train import rollout_graph

if TYPE_CHECKING:
    from foldforge.train.config import TrainConfig
    from foldforge.train.data import TrainBatch
    from foldforge.train.step import DeviceParts

LOG = logging.getLogger("foldforge.train")

#: Eager micro steps on the static example before the capture (kernel autotune, engine caches, lazily built modules, gradient buffers).
WARMUP_STEPS = 2
_FLAG = "_foldforge_step_graph_mode"  # model.__dict__ entry: "capture" (inline rollout, no recursion) | "bypass" (eager, no recursion)


# ------------------------------------------------------------------------------------------------------------------------------- mode flags
def mode(model) -> str | None:
    return getattr(model, "__dict__", {}).get(_FLAG)


@contextlib.contextmanager
def _mode(model, value: str):
    saved = model.__dict__.get(_FLAG)
    model.__dict__[_FLAG] = value
    try:
        yield
    finally:
        if saved is None:
            model.__dict__.pop(_FLAG, None)
        else:
            model.__dict__[_FLAG] = saved


def inline_rollout(model) -> bool:
    """True while a micro step is warmed up / captured as a step graph: ``step.mini_rollout`` then runs the rollout as plain ops."""
    return mode(model) == "capture"


# ------------------------------------------------------------------------------------------------------------------------------- examples
def _is_host(key: str, tensor: torch.Tensor) -> bool:
    return not tensor.is_cuda or key.endswith(":input_shape")


def example_signature(example: TrainBatch) -> tuple:
    """The graph-relevant layout of an example: per feature (key, shape, dtype, stride) for device tensors, (key, shape, dtype) for host tensors (the
    ``*:input_shape`` metadata: Python ints to the model; its VALUE is a validity check only, models/execution.py:99), plus the presence / layout of the optional fields.
    """
    feats = []
    for key in sorted(example.features):
        t = example.features[key]
        if _is_host(key, t):
            feats.append(
                (key, "host", tuple(t.shape), str(t.dtype))
            )  # the VALUE is not keyed: `:input_shape` is only a validity check of the gather (models/execution.py:99), the real token count differs per example
        else:
            feats.append((key, tuple(t.shape), str(t.dtype), tuple(t.stride())))

    def lay(t):
        return None if t is None else (tuple(t.shape), str(t.dtype), tuple(t.stride()))

    return (
        tuple(feats),
        lay(example.atom_positions),
        lay(example.atom_mask),
        lay(example.bond_pairs),
        lay(example.present_index),
        lay(example.present_valid),
    )


def _device_leaves(example: TrainBatch) -> list[torch.Tensor]:
    """The device tensors of an example in a fixed order (what a replay copies into the static buffers)."""
    leaves = [t for k, t in sorted(example.features.items()) if not _is_host(k, t)]
    leaves += [
        t
        for t in (
            example.atom_positions,
            example.atom_mask,
            example.bond_pairs,
            example.present_index,
            example.present_valid,
        )
        if t is not None
    ]
    return leaves


def static_example(example: TrainBatch) -> TrainBatch:
    """A TrainBatch of private clones of the device tensors (host metadata shared: it is part of the key)."""
    features = {
        k: (t if _is_host(k, t) else t.detach().clone())
        for k, t in example.features.items()
    }

    def c(t):
        return None if t is None else t.detach().clone()

    return dataclasses.replace(
        example,
        features=features,
        atom_positions=c(example.atom_positions),
        atom_mask=c(example.atom_mask),
        bond_pairs=c(example.bond_pairs),
        present_index=c(example.present_index),
        present_valid=c(example.present_valid),
    )


def load_example(static: TrainBatch, example: TrainBatch) -> None:
    """``copy_`` the device tensors of ``example`` into ``static`` (same signature: checked by the key)."""
    dst, src = _device_leaves(static), _device_leaves(example)
    if len(dst) != len(src):
        message = f"{len(src)} device tensors for {len(dst)} static buffers"
        raise RuntimeError(message)
    for d, s in zip(dst, src, strict=True):
        d.copy_(s)


# ------------------------------------------------------------------------------------------------------------------------------- loss parts
def clone_parts(parts: DeviceParts) -> DeviceParts:
    """A ``DeviceParts`` with the same structure over CLONES of the static output tensors (device ops, no sync)."""
    from foldforge.train.step import DeviceParts

    out = DeviceParts(
        parts.passes, (parts.w_diffusion, parts.w_bond, parts.w_distogram)
    )
    out.chunks = [
        (share, d.clone(), b.clone(), c.clone()) for share, d, b, c in parts.chunks
    ]
    out.distogram = None if parts.distogram is None else parts.distogram.clone()
    out.confidence = None if parts.confidence is None else parts.confidence.clone()
    out.conf_parts = {k: v.clone() for k, v in parts.conf_parts.items()}
    out.plain = {k: v.clone() for k, v in parts.plain.items()}
    return out


def zero_grads(params: list[torch.nn.Parameter]) -> None:
    """Zero the gradients IN PLACE (a step graph accumulates into these very tensors; ``set_to_none`` would orphan them). ``None`` stays ``None``."""
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch._foreach_zero_(grads)


# ------------------------------------------------------------------------------------------------------------------------------- the cache
class _Entry:
    def __init__(self, graph, static, parts, pool_bytes: int, held: Any) -> None:
        self.graph, self.static, self.parts, self.held = graph, static, parts, held
        self.pool_bytes = pool_bytes
        self.static_bytes = engine_capture.tensor_bytes(_device_leaves(static))
        self.held_bytes = engine_capture.tensor_bytes(held)


class StepGraph:
    """Per-key cache of captured micro steps. Counters are attributes; :meth:`report` collects them."""

    def __init__(
        self,
        *,
        max_graphs: int = 8,
        warmup: int = WARMUP_STEPS,
        sync_debug: str | None = None,
    ) -> None:
        self.entries: dict[tuple, _Entry] = {}
        self.failed: dict[tuple, str] = {}
        self.max_graphs = max_graphs
        self.warmup = warmup
        self.sync_debug = sync_debug  # "error" | "warn" while capturing (audit)
        self.pool = None
        self.captures = self.replays = self.fallbacks = 0
        self.capture_seconds = 0.0
        self.capture_log: list[dict[str, Any]] = []
        self.warned = False
        self.disabled_reason: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "fallbacks": self.fallbacks,
            "capture_seconds": round(self.capture_seconds, 3),
            "entries": len(self.entries),
            "pool_gib": round(
                sum(e.pool_bytes for e in self.entries.values()) / 2**30, 3
            ),
            "static_gib": round(
                sum(e.static_bytes for e in self.entries.values()) / 2**30, 3
            ),
            "held_gib": round(
                engine_capture.tensor_bytes([e.held for e in self.entries.values()])
                / 2**30,
                3,
            ),  # distinct storages: the engine caches are shared by the graphs
            "failed": {str(k[0:2]): v for k, v in self.failed.items()},
            "disabled": self.disabled_reason,
            "captures_detail": self.capture_log,
        }

    def clear(self) -> None:
        self.entries.clear()
        self.pool = None

    @staticmethod
    def refuses(model, example: TrainBatch, cfg: TrainConfig) -> str | None:
        """None when this micro step can be served by a graph, else why not."""
        if example.atom_positions.device.type != "cuda":
            return "not a CUDA device"
        if cfg.only:
            return "--only measurement modes run eager"
        if cfg.confidence and cfg.rollout not in ("after", "graph"):
            return f"rollout={cfg.rollout!r}: only 'after' / 'graph' placements are recorded (the rollout runs inline in the graph)"
        spec = getattr(model, "spec", None)
        if spec is not None and getattr(spec, "structural_tokens", False):
            return "structural tokens (the diffusion layout reads the raw features)"
        return None

    def key(
        self, model, example: TrainBatch, cfg: TrainConfig, passes: int, scale: float
    ) -> tuple:
        return (
            passes,
            scale,
            example_signature(example),
            cfg,
            id(model),
            torch.get_float32_matmul_precision(),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )

    # -- the call -------------------------------------------------------------------------------------------------------------------------
    def __call__(
        self,
        model,
        example: TrainBatch,
        cfg: TrainConfig,
        passes: int,
        scale: float = 1.0,
    ) -> DeviceParts | None:
        """Replay (capturing first when the key is new) one micro step; ``None`` when this call cannot be served (the caller runs it eagerly)."""
        reason = self.disabled_reason or self.refuses(model, example, cfg)
        if reason is not None:
            self.fallbacks += 1
            return None
        key = self.key(model, example, cfg, passes, scale)
        if key in self.failed:
            self.fallbacks += 1
            return None
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                if not self.warned:
                    LOG.warning(
                        "step graph: %d graphs captured, a new key (passes=%d, another example shape) runs eager; pad every example to one shape (--bench-bucket)",
                        len(self.entries),
                        passes,
                    )
                    self.warned = True
                self.fallbacks += 1
                return None
            if os.environ.get(
                "FF_STEP_GRAPH_DEBUG"
            ):  # diagnosis of a split signature: what differs from the first captured example
                sig = example_signature(example)
                first = getattr(self, "_debug_sig", None)
                if first is None:
                    self._debug_sig = sig
                else:
                    diff = [
                        (a, b) for a, b in zip(first[0], sig[0], strict=False) if a != b
                    ] + [
                        (a, b)
                        for a, b in zip(first[1:], sig[1:], strict=False)
                        if a != b
                    ]
                    LOG.warning(
                        "step graph key %d (%s): signature differs from the first in %d fields: %s",
                        len(self.entries),
                        example.name,
                        len(diff),
                        str(diff)[:1500],
                    )
                    if not diff:
                        LOG.warning(
                            "step graph: signature equal, key differs in (passes, scale, cfg, model, precision): %s",
                            (passes, scale),
                        )
            try:
                entry = self._capture(model, example, cfg, passes, scale)
            except torch.OutOfMemoryError:
                raise
            except Exception as error:  # noqa: BLE001 - a capture that cannot be made: this key runs eagerly from now on
                LOG.warning(
                    "step graph capture failed (%s: %s): this key runs the eager step",
                    type(error).__name__,
                    str(error)[:400],
                )
                self.failed[key] = f"{type(error).__name__}: {str(error)[:200]}"
                self.fallbacks += 1
                torch.cuda.synchronize()
                return None
            self.entries[key] = entry
        load_example(entry.static, example)
        entry.graph.replay()
        self.replays += 1
        return clone_parts(entry.parts)

    # -- capture --------------------------------------------------------------------------------------------------------------------------
    def _capture(
        self, model, example: TrainBatch, cfg: TrainConfig, passes: int, scale: float
    ) -> _Entry:
        """Warm up eagerly on a static clone of ``example`` (grads saved / restored in place), then record ``micro_step_device`` on it."""
        from foldforge.train.step import micro_step_device

        t0 = time.time()
        marks = engine_capture.cache_marks()
        params = [p for p in model.parameters() if p.requires_grad]
        static = static_example(example)
        device = torch.cuda.current_device()
        rng_state = torch.cuda.get_rng_state(device)
        # The warm-up runs for the FIRST graph only (kernel autotune, engine caches, lazily built modules, the gradient buffers). A later key (another
        # recycle count) captures directly: the graphs share ONE private pool, and an eager warm-up next to the pool's reserved memory does not fit
        # for the memory-hungry backends (pytorch: OOM on 80 GB).
        if not self.entries:
            # the caller's accumulated gradients (a capture in the middle of a gradient accumulation): saved, restored in place afterwards
            saved = [
                None if p.grad is None else p.grad.detach().clone() for p in params
            ]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with _mode(model, "capture"), torch.cuda.stream(side):
                # compile: the deepest recycle count, so every variant (no-grad recycles, grad, diffusion, rollout) is compiled before ANY capture
                warm_passes = max(passes, cfg.recycles_max) if cfg.compile else passes
                for _ in range(self.warmup):
                    micro_step_device(
                        model, static, cfg, warm_passes, scale=scale
                    ).floats()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            # every gradient buffer exists now (the warm-up made the missing ones): the capture ACCUMULATES into them
            for p, g in zip(params, saved, strict=True):
                if p.grad is None:
                    continue
                if g is None:
                    p.grad.zero_()
                else:
                    p.grad.copy_(g)
            del saved
            torch.cuda.set_rng_state(rng_state, device)
        if (
            cfg.compile and not self.entries
        ):  # a throwaway capture first (as the engine measurer): code that branches on "is the stream capturing" compiles its capture variant here; the kept capture then compiles nothing
            with (
                _mode(model, "capture"),
                torch.cuda.graph(
                    torch.cuda.CUDAGraph(), capture_error_mode="thread_local"
                ),
            ):
                micro_step_device(model, static, cfg, passes, scale=scale)
            torch.cuda.synchronize()
            model.diffusion_head.release_fold_state()
            torch.cuda.empty_cache()
        rollout_graph.bump_versions(
            model
        )  # the engine packs of the warm-up must not be served: the capture records the pack kernels itself
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        grads_before = [
            p.grad for p in params
        ]  # identity: the capture must accumulate into these very tensors
        torch.cuda.empty_cache()  # what torch.cuda.graph does on entry: the reserved delta below is then the graph's memory
        reserved = torch.cuda.memory_reserved()
        previous_debug = torch.cuda.get_sync_debug_mode()
        try:
            with (
                _mode(model, "capture"),
                torch.cuda.graph(
                    graph, pool=self.pool, capture_error_mode="thread_local"
                ),
                train_compile.strict() if cfg.compile else contextlib.nullcontext(),
            ):
                if (
                    self.sync_debug
                ):  # set inside the context: capture_begin itself synchronizes
                    torch.cuda.set_sync_debug_mode(self.sync_debug)
                parts = micro_step_device(model, static, cfg, passes, scale=scale)
        finally:
            torch.cuda.set_sync_debug_mode(previous_debug)
        torch.cuda.synchronize()
        orphaned = [
            p for p, g in zip(params, grads_before, strict=True) if p.grad is not g
        ]
        if orphaned:  # a gradient created INSIDE the capture would live in the pool and be overwritten, not accumulated, by every replay
            for p, g in zip(params, grads_before, strict=True):
                p.grad = g
            message = f"{len(orphaned)} parameter(s) got a new .grad tensor during the capture (none existed before): the step cannot accumulate in a graph"
            raise RuntimeError(message)
        pool_bytes = max(0, torch.cuda.memory_reserved() - reserved)
        model.diffusion_head.release_fold_state()  # references to graph-pool tensors (the kept pair, the fold conditioning)
        entry = _Entry(graph, static, parts, pool_bytes, engine_capture.hold_new(marks))
        seconds = time.time() - t0
        self.captures += 1
        self.capture_seconds += seconds
        self.capture_log.append(
            {
                "passes": passes,
                "seconds": round(seconds, 2),
                "pool_gib": round(pool_bytes / 2**30, 3),
            }
        )
        return entry


def micro_step_device(
    model,
    example: TrainBatch,
    cfg: TrainConfig,
    passes: int,
    *,
    scale: float = 1.0,
    timer=None,
) -> DeviceParts | None:
    """``step.micro_step_device`` through the model's :class:`StepGraph` (made on first use); ``None`` = not served, run it eagerly."""
    graph = model.__dict__.get("_foldforge_step_graph")
    if graph is None:
        graph = model.__dict__["_foldforge_step_graph"] = StepGraph()
    return graph(model, example, cfg, passes, scale)


@contextlib.contextmanager
def bypass(model):
    """Run the eager step without re-entering the graph path."""
    with _mode(model, "bypass"):
        yield
