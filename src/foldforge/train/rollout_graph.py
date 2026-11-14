"""The training mini-rollout (20 sampler steps, 1 sample, no grad) as ONE CUDA graph (``TrainConfig.rollout = "graph"``).

Eagerly the rollout is ~8,650 kernel launches (433 per sampler step) for 48 ms of GPU work: launch bound, 89 % idle. Here the whole
trajectory -- the noise schedule, the fold conditioning (pair conditioning, atom state), the conditioning table of the 20 noise levels, and the
20 steps (re-pose, re-noise, denoise, align, Euler update) -- is recorded once per input SHAPE and replayed per micro step.

What makes this safe for TRAINING (the weights change at every optimizer step) is what the inference sampler graph does NOT do
(``models.sampler_graph``, engine ``static_weights()`` / ``static_inputs()`` / ``refresh_static_inputs``): nothing here is declared static.

* Weight packs and the per-fold tables (token-DiT pair bias, atom conditioning, conditioning folds) are built INSIDE the capture, so their
  kernels are recorded and every replay rebuilds them from the CURRENT weights and the CURRENT inputs (the engine scopes its pack caches to a
  capture: ``miniworld_engine.kernels._capture``). The cost is paid once per 20 steps, not per step.
* The graph reads static buffers (a copy of the batch features and of the DETACHED trunk outputs); each micro step ``copy_``s the new fold into
  them. There is no refresh API involved.
* The random numbers (the initial noise, the random rigid motion, the re-noising) are drawn inside the graph from the default CUDA generator,
  which a capture registers with the graph: a replay draws NEW numbers (it advances the generator like the eager rollout does). Measured
  (--bench, 8-target pool): the philox increments of the graph equal the eager ones, so the losses and the parameters after 5 optimizer steps are
  BITWISE those of the eager rollout; this is a property of torch's generator, not a contract: the claim that holds is the same distribution.
* The churn branch is a ``torch.where`` on device scalars (``sampler_graph.sample_inline``), so the 20 host reads of ``sigma_next`` of the
  eager sampler are gone too.
* An optimizer that writes parameters WITHOUT bumping ``_version`` (torch's fused Adam) must be followed by :func:`bump_versions`: the graph does
  not care (it repacks at every replay), the engine's EAGER caches would serve stale packs. The training optimizer (foreach Adam) bumps.

Memory: one private pool per graph (a replay's intermediates), the static inputs, and the engine cache entries the capture made (held while the
graph lives). ``max_graphs`` and ``budget_fraction`` bound them like ``SamplerGraphCache`` (least recently used graphs are dropped first; a
capture that finds too little free memory drops them before it runs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from typing import Any

import torch

from foldforge.models import sampler_graph
from foldforge.models.execution import _flatten, _map
from foldforge.modules.engine import capture as engine_capture

LOG = logging.getLogger("foldforge.train")

#: Eager rollouts run on the static buffers before the capture (kernels autotune, engine caches fill, lazily built modules exist); their random
#: numbers are rolled back.
WARMUP_ROLLOUTS = 2



@contextlib.contextmanager
def _no_autocast_cache():
    """Autocast without its weight-cast cache. The cache keeps the bf16 cast of every fp32 parameter that requires grad until the OUTERMOST autocast
    context exits (it ignores no_grad): a capture inside the micro step's autocast would be served the casts the eager warm-up made (no cast kernel
    recorded: a replay reads the weights of the capture, and, once the context exits and the cache frees them, freed memory: NaN / illegal address).
    With the cache off the casts run (and are recorded) at every call."""
    if not torch.is_autocast_enabled("cuda"):
        yield
        return
    with torch.autocast("cuda", dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False):
        yield


class InlineScope:
    """The sampler scope that makes ``AlphaFold3._sample_diffusion`` run :func:`sampler_graph.sample_inline` (the loop as plain ops, no capture of
    its own): the rollout graph captures around the whole call."""

    inline = True
    cache = None


_INLINE = InlineScope()


def bump_versions(model: torch.nn.Module) -> None:
    """Bump ``_version`` of every parameter and buffer (no kernel). After an optimizer that writes without bumping (``Adam(fused=True)``)."""
    torch.autograd.graph.increment_version(list(model.parameters()) + list(model.buffers()))


def rollout_body(model, batch, embeddings: dict[str, torch.Tensor], steps: int) -> torch.Tensor:
    """The rollout as one straight-line op sequence (no host read): ``[res, atoms, 3]``. Eager or under a capture, the same arithmetic.

    ``model.num_samples`` / ``diffusion_steps`` / ``foldforge_sampler_scope`` are set for the call; the head runs in eval mode (no dropout) as in
    ``step.mini_rollout``."""
    head = model.diffusion_head
    saved = (model.num_samples, model.diffusion_steps, getattr(model, "foldforge_sampler_scope", None))
    head.eval()
    model.num_samples, model.diffusion_steps, model.foldforge_sampler_scope = 1, steps, _INLINE
    try:
        with torch.no_grad():
            return model._sample_diffusion(batch, embeddings)["atom_positions"][0]  # noqa: SLF001
    finally:
        model.num_samples, model.diffusion_steps, model.foldforge_sampler_scope = saved
        head._fold = None  # noqa: SLF001 - the rollout's fold-scoped conditioning
        head.train()


def rollout_batch(batch_data):
    """The batch the rollout reads (without the MSA and the templates: the diffusion head never reads them, and they are the big tensors)."""
    # `seq_length` (the real token count, a CPU int tensor) is read by no module, yet `_flatten` puts its VALUE in the key: it would make every
    # crop length its own graph
    tokens = dataclasses.replace(batch_data.token_features, seq_length=None)
    return dataclasses.replace(batch_data, msa=None, templates=None, token_features=tokens)


def graph_tree(batch_data, detached: dict[str, torch.Tensor]) -> dict[str, Any]:
    """What the rollout reads: the batch (:func:`rollout_batch`) and the detached trunk outputs."""
    return {"batch": rollout_batch(batch_data), "emb": dict(detached)}


def graph_key(model, structure: Any, steps: int) -> tuple:
    """The per-shape key: layout of every input tensor (``structure``: shapes / dtypes / strides), the step count, the model, and the precision
    state a capture bakes in."""
    return (
        structure,
        steps,
        id(model),
        torch.get_float32_matmul_precision(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )


class _Entry:
    """One captured rollout with the static buffers it reads (the cache's value; the interface ``SamplerGraphCache`` needs: ``retained``, ``release``)."""

    def __init__(self, graph, tree, leaves, out, pool_bytes: int, held: Any) -> None:
        self.graph, self.tree, self.leaves, self.out, self.held = graph, tree, leaves, out, held
        self.pool_bytes = pool_bytes
        self.static_bytes = engine_capture.tensor_bytes(leaves) + out.numel() * out.element_size()
        self.held_bytes = engine_capture.tensor_bytes(held)
        #: bytes this entry keeps on the GPU: the static inputs, the output, the graph's pool and the engine cache entries of the capture
        self.retained = self.static_bytes + pool_bytes + self.held_bytes

    def load(self, leaves: list[torch.Tensor]) -> None:
        """Write this micro step's tensors into the static buffers."""
        if len(leaves) != len(self.leaves):
            raise RuntimeError(f"{len(leaves)} tensors for {len(self.leaves)} static buffers")
        for dst, src in zip(self.leaves, leaves, strict=True):
            keep = sampler_graph._unique_part(dst)  # noqa: SLF001
            if keep is None:
                dst.copy_(src)
            else:  # an expanded (stride 0) leaf: its storage is the unique part
                dst[keep].copy_(src[keep])

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.out.clone()

    def release(self) -> None:
        self.graph = self.tree = self.leaves = self.out = self.held = None


class RolloutGraph:
    """Per-shape cache of captured rollouts. ``__call__`` has the signature of ``step.mini_rollout`` and returns ``[res, atoms, 3]``, or ``None`` when
    this call cannot be served (CPU, structural tokens, a capture that failed): the caller then runs the eager rollout.

    Counters (``captures``, ``replays``, ``cache_hits`` / ``cache_misses`` / ``cache_evictions``, ``uncached``, ``fallbacks``, ``capture_seconds``)
    are attributes; ``report()`` collects them."""

    def __init__(self, *, max_graphs: int = 4, budget_fraction: float = 0.1, warmup: int = WARMUP_ROLLOUTS) -> None:
        self.cache = sampler_graph.SamplerGraphCache(self, max_entries=max_graphs, budget_fraction=budget_fraction)
        self.warmup = warmup
        self.failed: set = set()
        self.captures = self.replays = self.cache_hits = self.cache_misses = self.cache_evictions = self.uncached = self.fallbacks = 0
        self.capture_seconds = 0.0
        self.disabled_reason: str | None = None

    # -- bookkeeping (no device access) ------------------------------------------------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
            "uncached": self.uncached,
            "fallbacks": self.fallbacks,
            "capture_seconds": round(self.capture_seconds, 3),
            "entries": len(self.cache.entries),
            "retained_gib": round(self.cache.retained_bytes / 2**30, 3),
            "pool_gib": round(sum(e.pool_bytes for e in self.cache.entries.values()) / 2**30, 3),
            "disabled": self.disabled_reason,
        }

    def clear(self) -> None:
        self.cache.clear()

    @staticmethod
    def serves(model, device: torch.device) -> str | None:
        """None when this model / device can run the graph, else why not."""
        if device.type != "cuda":
            return "not a CUDA device"
        spec = getattr(model, "spec", None)
        if spec is not None and getattr(spec, "structural_tokens", False):
            return "structural tokens (the diffusion layout reads the raw features)"
        return None

    # -- the call ---------------------------------------------------------------------------------------------------------------------------------
    def __call__(self, model, features, batch_data, detached: dict[str, torch.Tensor], steps: int) -> torch.Tensor | None:  # noqa: ARG002
        if self.disabled_reason is not None:
            return None
        device = next(t for t in detached.values() if torch.is_tensor(t)).device
        if self.serves(model, device) is not None:
            self.fallbacks += 1
            return None
        tree = graph_tree(batch_data, detached)
        try:
            structure, leaves = _flatten(tree)
        except (TypeError, ValueError):
            self.fallbacks += 1
            return None
        key = graph_key(model, structure, steps)
        if key in self.failed:
            self.fallbacks += 1
            return None
        entry = self.cache.lookup(key)
        kept = True
        if entry is None:
            try:
                entry = self.cache.build(lambda: self._capture(model, tree, steps))
            except torch.OutOfMemoryError:
                raise
            except Exception as error:  # noqa: BLE001 - a capture that cannot be made: this shape runs eagerly from now on
                LOG.warning("rollout graph capture failed (%s: %s): this shape runs the eager rollout", type(error).__name__, str(error)[:300])
                self.failed.add(key)
                self.fallbacks += 1
                torch.cuda.synchronize()
                return None
            kept = self.cache.insert(key, entry, self.cache.budget_bytes)
        else:
            entry.load(leaves)
        result = entry.replay()
        self.replays += 1
        if not kept:
            torch.cuda.synchronize()
            entry.release()
        return result

    # -- capture ----------------------------------------------------------------------------------------------------------------------------------
    def _capture(self, model, tree: dict[str, Any], steps: int) -> _Entry:
        """Warm up on a static clone of ``tree`` (eager), then record ``rollout_body`` on it. No static_weights / static_inputs scope."""
        t0 = time.time()
        marks = engine_capture.cache_marks()
        static = _map(tree, sampler_graph._static_clone)  # noqa: SLF001
        _, leaves = _flatten(static)
        batch, embeddings = static["batch"], static["emb"]
        device = torch.cuda.current_device()
        head = model.diffusion_head
        rng_state = torch.cuda.get_rng_state(device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with _no_autocast_cache(), torch.cuda.stream(side):
            for _ in range(self.warmup):
                rollout_body(model, batch, embeddings, steps)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng_state, device)  # the warm-up's numbers are not the caller's
        # entries the eager warm-up made must not be served to the capture: the capture has to record the packing kernels itself
        torch.autograd.graph.increment_version(list(head.parameters()))
        graph = torch.cuda.CUDAGraph()
        reserved = torch.cuda.memory_reserved()
        with _no_autocast_cache(), torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = rollout_body(model, batch, embeddings, steps)
        torch.cuda.synchronize()
        pool_bytes = max(0, torch.cuda.memory_reserved() - reserved)
        head.release_fold_state()  # references to graph-pool tensors (the kept pair, the fold conditioning)
        entry = _Entry(graph, static, leaves, out, pool_bytes, engine_capture.hold_new(marks))
        self.captures += 1
        self.capture_seconds += time.time() - t0
        return entry


def mini_rollout(model, features, batch_data, detached: dict[str, torch.Tensor], steps: int, eager) -> torch.Tensor:
    """``cfg.rollout = "graph"``: the rollout of ``train/step.py`` through the model's :class:`RolloutGraph` (made on first use), or ``eager(...)`` when the
    graph does not serve this call (CPU, structural tokens, a failed capture)."""
    graph = model.__dict__.get("_foldforge_rollout_graph")
    if graph is None:
        graph = model.__dict__["_foldforge_rollout_graph"] = RolloutGraph()
    result = graph(model, features, batch_data, detached, steps)
    return eager(model, features, batch_data, detached, steps) if result is None else result
