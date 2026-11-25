"""The whole EDM sampler step as one CUDA graph.

A step is: re-pose the iterate (random rigid motion), re-noise it, denoise, align,
take the Euler step. Run eagerly it is ~100 small launches around the denoiser and
one host synchronisation (the churn branch reads ``sigma_next`` on the host), so the
GPU idles between steps. Here the branch is a ``where`` on device scalars, the
alignment needs no SVD (``utils.rigid_align``), the random numbers come from
generators registered with the graph, and the denoiser runs inside the same capture.
Nothing in a step reads the device on the host, so one replay is one step.

The graph is captured once per SHAPE, not once per fold (``SamplerGraphCache``, engine
request R26). The first fold of a shape is captured on a static clone of its per-fold
tensors (the batch, the fold's conditioning, the atom state); every later fold of that
shape is written into those buffers with ``copy_``, the engine rebuilds the hoisted
tables made from them in place (``StaticTables.refresh``), and the same graph replays.
Without a cache (training, an engine without the refresh API, a shape larger than the
cache budget) the graph is built for ONE fold: it reads the fold's conditioning tensors
directly (they stand for the life of the sampler call), so a sampler call builds its
own and drops it.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import torch

from foldjet.models.execution import _flatten, _map, own_cublas_workspace
from foldjet.modules.engine import capture as engine_capture

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable

    from foldjet.diffusion.edm.sampling import EulerSampler

#: Eager steps run before the capture (kernels autotune, caches fill); their random
#:numbers are rolled back.
WARMUP_STEPS = 2


def _expand(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.reshape(*value.shape, *(1,) * (target.ndim - value.ndim))


def _hatted(config: EulerSampler.Config, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute hatted noise levels for the complete schedule.

    ``sigma_hat`` of every step of the schedule (the arithmetic of ``step``, over
    the whole vector).
    """
    sigma, sigma_next = sigmas[:-1], sigmas[1:]
    one = torch.ones((), dtype=sigmas.dtype, device=sigmas.device)
    if config.churn_gamma is None:
        on = sigma_next > config.gamma_min
        factor = torch.where(on, one * (1 + config.gamma_0), one)
    else:
        on = (sigma >= config.churn_sigma_min) & (sigma <= config.churn_sigma_max)
        factor = torch.where(on, one * (1 + config.churn_gamma), one)
    return sigma * factor


class _Glue:
    """The engine's fused rigid-motion glue modules of one sampler call."""

    def __init__(self, augment: Any, align: Any, mask: torch.Tensor) -> None:
        self.augment, self.align, self.mask = augment, align, mask
        self._served: dict[tuple, bool] = {}

    def serves(self, flat: torch.Tensor) -> bool:
        """Whether both modules accept coordinates like ``flat``.

        Checked once per layout. Placeholder tensors (``torch.empty``) stand for the
        random draws, so the check consumes no random numbers.
        """
        key = (tuple(flat.shape), flat.dtype, flat.device)
        if key not in self._served:
            batch = flat.shape[0]
            empty = functools.partial(torch.empty, dtype=flat.dtype, device=flat.device)
            scalar = empty(())
            self._served[key] = (
                self.augment.unsupported(
                    flat,
                    self.mask,
                    empty((batch, 2, 3)),
                    empty((batch, 3)),
                    empty(flat.shape),
                    scalar,
                    scalar,
                )
                is None
                and self.align.unsupported(flat, flat, self.mask, empty((2,)), scalar)
                is None
            )
        return self._served[key]


def _rigid_glue(
    sampler: EulerSampler, rigid_mask: torch.Tensor | None, coordinate_dims: int
) -> _Glue | None:
    """Build the engine's fused sampler glue modules, or None when it does not serve.

    ``AugmentNoise`` and ``AlignEuler`` (``miniworld_engine.modules``) are built ONCE
    per sampler call, here, never inside a graph capture. They serve the dense
    layout's step: fp32 coordinates, one random rigid motion per sample before the
    noise, one Kabsch alignment after the denoiser. Their ``unsupported`` says per
    call whether the backend (``cuda`` on an H100, ``triton`` on other CUDA cards)
    accepts it; a batched sigma or a non-CUDA mask keeps the torch ops.
    """
    config = sampler.config
    if rigid_mask is None or config.batched_sigma or coordinate_dims != 2:  # noqa: PLR2004 - dense token and atom axes
        return None
    from foldjet.modules.engine.sampler_ops import (  # noqa: PLC0415 - engine glue loads only when a sampler call needs it
        augment_align,
    )

    window = config.churn_gamma is not None
    churn = {
        "window": window,
        "gamma": config.churn_gamma if window else config.gamma_0,
        "gamma_min": config.gamma_min,
        "sigma_min": config.churn_sigma_min,
        "sigma_max": config.churn_sigma_max,
        "noise_scale": config.noise_scale,
        "variance_floor": config.variance_floor,
        "clamp": bool(config.clamp_variance or config.variance_floor),
    }
    modules = augment_align(
        rigid_mask.device, churn=churn, step_scale=config.step_scale
    )
    if modules is None:
        return None
    mask = rigid_mask.reshape(-1).to(torch.float32).contiguous()
    return _Glue(*modules, mask)


def build_step(
    sampler: EulerSampler,
    denoise: Callable,
    *,
    generator: torch.Generator | None,
    augment: Callable | None,
    align: Callable | None,
    coordinate_dims: int,
    rigid_mask: torch.Tensor | None = None,
) -> Callable:
    """Build a sampler step whose branches use device scalars.

    ``step(coords, sigma, sigma_next)``: ``EulerSampler.step`` with every branch on
    device scalars.

    The arithmetic is that of ``EulerSampler.prepare`` / ``update`` operation for
    operation: the churn factor ``1 + gamma`` is the same float32 constant the eager
    path multiplies by.

    ``rigid_mask`` (the atom mask) declares that ``augment`` and ``align`` are the
    dense layout's random rigid motion and masked Kabsch alignment; the engine then
    runs both, with the churn arithmetic and the Euler update, as six kernels
    (random numbers drawn in the same order from the same generator).
    """
    config = sampler.config
    glue = (
        _rigid_glue(sampler, rigid_mask, coordinate_dims)
        if augment is not None and align is not None
        else None
    )

    def fused_step(
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None,
    ) -> torch.Tensor:
        # called only when `glue` is not None (see `step`)
        fused = cast("_Glue", glue)
        mask = fused.mask
        batch = coords.shape[0]
        flat = coords.reshape(batch, -1, 3)
        frame = torch.randn((batch, 2, 3), device=coords.device, dtype=coords.dtype)
        shift = torch.randn((batch, 3), device=coords.device, dtype=coords.dtype)
        noise = torch.randn(
            flat.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )
        noisy, scalars = fused.augment(
            flat, mask, frame, shift, noise, sigma, sigma_next
        )
        noisy = noisy.view(coords.shape)
        denoised = (
            denoise(noisy, scalars[0])
            if row is None
            else denoise(noisy, scalars[0], row)
        )
        flat_out = fused.align(
            noisy.view(batch, -1, 3),
            denoised.reshape(batch, -1, 3).to(torch.float32),
            mask,
            scalars,
            sigma_next,
        )
        return flat_out.view(coords.shape)

    def step(
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            glue is not None
            and coords.ndim == coordinate_dims + 2
            and glue.serves(coords.reshape(coords.shape[0], -1, 3))
        ):
            return fused_step(coords, sigma, sigma_next, row)
        if augment is not None:
            coords = augment(coords)
        one = torch.ones((), dtype=sigma.dtype, device=sigma.device)
        if config.churn_gamma is None:
            on = sigma_next > config.gamma_min
            factor = torch.where(on, one * (1 + config.gamma_0), one)
        else:
            on = (sigma >= config.churn_sigma_min) & (sigma <= config.churn_sigma_max)
            factor = torch.where(on, one * (1 + config.churn_gamma), one)
        sigma_hat = sigma * factor
        variance = sigma_hat**2 - sigma**2
        if config.clamp_variance or config.variance_floor:
            variance = variance.clamp(min=config.variance_floor)
        magnitude = config.noise_scale * torch.sqrt(variance)
        noisy = coords + magnitude * torch.randn(
            coords.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )
        if config.batched_sigma:
            batch = coords.shape[: -(coordinate_dims + 1)]
            sigma_hat = sigma_hat.reshape((1,) * len(batch)).expand(batch)
        denoised = (
            denoise(noisy, sigma_hat) if row is None else denoise(noisy, sigma_hat, row)
        )
        if align is not None:
            noisy = align(noisy, denoised)
        direction = (noisy - denoised) / _expand(sigma_hat, noisy)
        dt = _expand(sigma_next - sigma_hat, noisy)
        return noisy + config.step_scale * dt * direction

    # the fused glue modules (or None): the per-shape cache reads the mask it made
    step.glue = glue  # pyright: ignore[reportFunctionMemberAccess]  # attribute on the step function, read with getattr
    return step


class _CapturedStep:
    def __init__(
        self,
        step: Callable,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None,
        scopes: Callable[[], contextlib.AbstractContextManager],
    ) -> None:
        self.scopes = scopes
        self.row = None if row is None else row.detach().clone()
        self.coords = coords.detach().clone()
        # (sigma, sigma_next) of the step in one buffer: one copy per replay;
        # the step's result is carried into ``coords`` inside the graph
        self.pair = torch.stack([sigma, sigma_next]).detach().clone()
        self.sigma, self.sigma_next = self.pair[0], self.pair[1]
        #: the engine's handle of the hoisted tables this capture reads (``None`` on an
        #: engine without the refresh API)
        self.tables = None
        #: reserved bytes the capture itself took (its private pool)
        self.pool_bytes = 0
        device = torch.cuda.current_device()
        with scopes() as tables:
            self.tables = tables
            # The warmup consumes random numbers the real run must still
            # see: roll the streams back.
            default_state = torch.cuda.get_rng_state(device)
            generator_state = None if generator is None else generator.get_state()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(WARMUP_STEPS):
                    step(self.coords, self.sigma, self.sigma_next, self.row)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            torch.cuda.set_rng_state(default_state, device)
            if generator is not None:
                generator.set_state(generator_state)  # pyright: ignore[reportArgumentType]  # read above whenever generator is not None
            self.graph = torch.cuda.CUDAGraph()
            if generator is not None:
                self.graph.register_generator_state(generator)
            reserved = torch.cuda.memory_reserved()
            # a cuBLAS workspace of its own: a cached graph outlives the graphs captured
            # before it (see ``own_cublas_workspace``)
            with own_cublas_workspace(), torch.cuda.graph(self.graph):
                self.out = step(self.coords, self.sigma, self.sigma_next, self.row)
                self.coords.copy_(self.out)
            torch.cuda.synchronize()
            self.pool_bytes = max(0, torch.cuda.memory_reserved() - reserved)
        _bump(scopes, "captures")

    def __call__(self, pair: torch.Tensor, row: torch.Tensor | None = None) -> None:
        """Advance the iterate using the current sigma pair.

        One step: ``coords`` (the iterate this object carries) advance by the step
        at ``pair`` = (sigma, sigma_next).
        """
        if row is not None:
            self.row.copy_(row)  # pyright: ignore[reportOptionalMemberAccess]  # a step captured with a row is replayed with one
        self.pair.copy_(pair)
        self.graph.replay()
        _bump(self.scopes, "replays")

    @property
    def static_bytes(self) -> int:
        """Bytes of this step's own buffers (iterate, sigma pair, conditioning row)."""
        return sum(
            t.numel() * t.element_size()
            for t in (self.coords, self.pair, self.row)
            if t is not None
        )


def _bump(counters: object, name: str, amount: int = 1) -> None:
    """Add to a counter of the sampler scope.

    A scope without it (e.g. a test's) is left alone.
    """
    if hasattr(counters, name):
        setattr(counters, name, getattr(counters, name) + amount)


# ---- one graph per shape, reused over folds ------------------------------------------
# (FoldJet side of engine request R26)


@dataclasses.dataclass
class Fold:
    """What a sampler call hands over so its graph can serve later folds of a shape.

    ``tree()``: every per-fold object the step reads, as a tree of tensors (dict /
    list / tuple / dataclass of them, plus plain values); evaluated after
    ``conditioning`` made the fold's tables. ``pieces(tree)``: ``(denoise, augment,
    align, rigid_mask)`` reading ``tree`` (the static clone of the first fold of a
    shape). ``bind(tree)``: a context in which the model's own per-fold state is
    ``tree`` (the capture runs inside it). ``casts()``: ``(source, copy)`` pairs of
    tensors made OUTSIDE the graph from a leaf of the tree (a dtype cast of the pair
    the token DiT stack keeps): ``copy`` is rewritten from ``source`` before each fold.
    ``model``: the root of the engine modules (the engine's refresh is limited to the
    tables its modules made).
    """

    model: torch.nn.Module
    tree: Callable[[], Any]
    pieces: Callable[
        [Any], tuple[Callable, Callable | None, Callable | None, torch.Tensor | None]
    ]
    bind: Callable[[Any], contextlib.AbstractContextManager]
    casts: Callable[[], list[tuple[torch.Tensor, torch.Tensor]]] = list


def _unique_part(t: torch.Tensor) -> tuple[slice, ...] | None:
    """Return the index of the part of an expanded tensor that owns its storage.

    That part has size 1 along every stride-0 dim; None for an ordinary tensor.
    """
    if not any(
        stride == 0 and size > 1
        for size, stride in zip(t.shape, t.stride(), strict=True)
    ):
        return None
    return tuple(
        slice(0, 1) if stride == 0 and size > 1 else slice(None)
        for size, stride in zip(t.shape, t.stride(), strict=True)
    )


def _static_clone(t: torch.Tensor) -> torch.Tensor:
    """Clone t into a tensor of fixed identity, its expanded (stride 0) dims kept.

    The engine reads a shared mask / conditioning by its stride 0 (one fold's mask
    served to the NT folds of a grouped call) and takes a slower general path for a
    dense copy.
    """
    t = t.detach()
    keep = _unique_part(t)
    return t.clone() if keep is None else t[keep].clone().expand(t.shape)


class _Entry:
    """One captured sampler step with the static buffers it reads.

    The per-shape cache's value.
    """

    def __init__(
        self,
        captured: _CapturedStep,
        tree: Any,
        leaves: list[torch.Tensor],
        casts: list[tuple[torch.Tensor, torch.Tensor]],
        mask_copies: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # declared without None: `release` clears them on an entry never read again
        self.captured: _CapturedStep = captured
        self.tree = tree  # keeps the static tensors the graph reads alive
        self.keepalive: Any = None
        #: entries of the engine's caches at the capture (see
        #: ``engine.capture.hold_caches``)
        self.engine_held: Any = None
        self.leaves: list[torch.Tensor] = leaves
        self.casts: list[tuple[torch.Tensor, torch.Tensor]] = casts
        self.mask_copies: list[tuple[torch.Tensor, torch.Tensor]] = mask_copies
        self.tables = captured.tables
        seen: set[int] = set()
        static = 0
        for leaf in leaves:
            pointer = leaf.untyped_storage().data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                static += leaf.untyped_storage().nbytes()
        #: bytes this entry keeps on the GPU: the static fold, the iterate buffers and
        #: the graph's pool
        self.retained = static + captured.static_bytes + captured.pool_bytes

    def load(
        self,
        leaves: list[torch.Tensor],
        model: torch.nn.Module,  # noqa: ARG002 - part of the call signature
    ) -> int:
        """Write another fold of the same shape into the static buffers.

        Also refreshes the engine's tables made from them; returns the tables
        refreshed.
        """
        if len(leaves) != len(self.leaves):
            msg = f"{len(leaves)} tensors for {len(self.leaves)} static buffers"
            raise _LayoutMismatch(msg)
        for dst, src in zip(self.leaves, leaves, strict=True):
            keep = _unique_part(dst)
            if keep is None:
                dst.copy_(src)
            else:
                # an expanded (stride 0) leaf: its storage is the unique part,
                # written through the size-1 view
                dst[keep].copy_(src[keep])
        for source, copy in self.casts:
            copy.copy_(source.unsqueeze(0))
        for dst, src in self.mask_copies:
            dst.copy_(src.reshape(-1))
        # the handle selects this graph's tables (a ``root=`` filter selects by the
        # module that made them, which is not always under the model)
        return 0 if self.tables is None else int(self.tables.refresh())

    def release(self) -> None:
        """Drop the engine's refresh records and the buffers.

        The graph and its pool go with the last reference.
        """
        if self.tables is not None:
            self.tables.release()
            self.tables = None
        # a released entry is never read again
        self.captured = None  # pyright: ignore[reportAttributeAccessIssue]  # cleared on release
        self.tree = self.leaves = self.casts = self.mask_copies = None  # pyright: ignore[reportAttributeAccessIssue]  # cleared on release
        self.keepalive = self.engine_held = None


class _LayoutMismatch(RuntimeError):  # noqa: N818 - private name kept as it is
    """A fold does not fit the static buffers of the graph chosen for it.

    Never raised for a key that matched: the caller recaptures.
    """


def _release_entries(entries: dict) -> None:
    for entry in list(entries.values()):
        entry.release()
    entries.clear()


#: Every live cache, so that a process-wide event (the engine's weight caches are
#: cleared) can drop all of them.
_LIVE_CACHES: weakref.WeakSet = weakref.WeakSet()


def release_all_caches() -> None:
    """Drop every cached sampler graph of this process.

    Called before the engine's weight caches are cleared: a graph reads those entries.
    """
    for cache in list(_LIVE_CACHES):
        cache.clear()


def weights_token(model: torch.nn.Module) -> tuple[int, int]:
    """``(hash, count)`` of the identity and version of every parameter and buffer.

    The identity is the address, the version the in-place one: the token changes when
    a weight is replaced or updated in place. Host metadata only.
    """
    items = []
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        try:
            version = tensor._version  # noqa: SLF001 - the in-place version counter
        except RuntimeError:  # inference tensors carry none
            version = -1
        items.append((tensor.data_ptr(), version))
    return hash(tuple(items)), len(items)


class SamplerGraphCache:
    """Captured sampler steps keyed by shape, least recently used first.

    A graph captured on one fold serves every later fold of the same shape: the fold's
    tensors are written into the graph's static buffers and the engine rebuilds the
    hoisted tables made from them (``StaticTables.refresh``), instead of capturing
    again (two eager warm-up steps and the capture, 44-92 ms). Memory: at most
    ``max_entries`` graphs and ``budget_fraction`` of the device's memory in all (each
    graph keeps a private pool and its static fold); a capture that finds less than
    1.5x the previous capture's reserved memory free first drops the least recently
    used graphs; an entry larger than the whole budget is used for its fold and
    dropped. The weights are part of the contract: when any parameter or buffer
    changed (address or in-place version) every graph is dropped.

    ``counters`` is the object whose attributes count (``cache_hits``,
    ``cache_misses``, ``cache_evictions``, ...; missing ones are skipped).
    """

    def __init__(
        self,
        counters: object | None = None,
        *,
        max_entries: int = 4,
        budget_fraction: float = 0.3,
        max_fold_bytes: int = 2 << 30,
    ) -> None:
        self.counters = counters
        self.max_entries = max_entries
        self.budget_fraction = budget_fraction
        #: a fold whose tensors are larger than this is not cached: its static copy
        #: would sit next to the fold (the memory a big fold needs), and the capture it
        #: saves is a few percent of such a forward
        self.max_fold_bytes = max_fold_bytes
        self.entries: OrderedDict[Any, _Entry] = OrderedDict()
        self.weights: tuple[int, int] | None = None
        #: bytes the previous capture reserved (0 before the first)
        self.last_capture_bytes = 0
        _LIVE_CACHES.add(self)
        weakref.finalize(self, _release_entries, self.entries)

    # -- bookkeeping (no device access) ---------------------------------------------
    def bump(self, name: str, amount: int = 1) -> None:
        """Add to a counter of ``counters``."""
        _bump(self.counters, name, amount)

    @property
    def retained_bytes(self) -> int:
        """Bytes all entries keep on the GPU."""
        return sum(entry.retained for entry in self.entries.values())

    def lookup(self, key: Any) -> _Entry | None:
        """Return the entry of ``key`` (now the most recently used).

        Counts the hit or the miss.
        """
        entry = self.entries.get(key)
        if entry is None:
            self.bump("cache_misses")
            return None
        self.entries.move_to_end(key)
        self.bump("cache_hits")
        return entry

    def drop(self, key: Any) -> None:
        """Remove and release the entry of ``key``, if any."""
        entry = self.entries.pop(key, None)
        if entry is not None:
            entry.release()

    def evict_oldest(self) -> bool:
        """Drop the least recently used entry; False when there is none."""
        if not self.entries:
            return False
        _, entry = self.entries.popitem(last=False)
        entry.release()
        self.bump("cache_evictions")
        return True

    def insert(self, key: Any, entry: _Entry, budget_bytes: int) -> bool:
        """Keep ``entry`` unless it alone exceeds the budget.

        Then drop the least recently used ones until the entries fit the count and the
        bytes. Returns whether the entry was kept (the caller releases it after its
        fold when not).
        """
        if entry.retained > budget_bytes:
            self.bump("uncached")
            return False
        self.entries[key] = entry
        while len(self.entries) > self.max_entries or (
            self.retained_bytes > budget_bytes and len(self.entries) > 1
        ):
            self.evict_oldest()
        return True

    def clear(self) -> None:
        """Release every entry and forget the weights token."""
        for entry in list(self.entries.values()):
            entry.release()
        self.entries.clear()
        self.weights = None
        self.last_capture_bytes = 0

    def check_weights(self, model: torch.nn.Module) -> None:
        """Drop every graph when the model's weights are not the ones captured with."""
        token = weights_token(model)
        if token != self.weights:
            if self.entries:
                self.clear()
            self.weights = token

    # -- device (memory) ------------------------------------------------------------
    @property
    def budget_bytes(self) -> int:
        """``budget_fraction`` of the current device's total memory, in bytes."""
        return int(
            self.budget_fraction
            * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        )

    def make_room(self) -> None:
        """Before a capture: drop the least recently used graphs when memory is short.

        Short means less than 1.5x the previous capture's reserved memory free.
        """
        if not self.entries or not self.last_capture_bytes:
            return
        need = int(1.5 * self.last_capture_bytes)
        if torch.cuda.mem_get_info()[0] >= need:
            return
        torch.cuda.synchronize()
        while self.entries and torch.cuda.mem_get_info()[0] < need:
            self.evict_oldest()
            torch.cuda.empty_cache()

    def build(self, make: Callable[[], _Entry]) -> _Entry:
        """Capture a new entry: room first.

        An out-of-memory capture drops every cached graph and tries once more.
        """
        self.make_room()
        reserved = torch.cuda.memory_reserved()
        try:
            entry = make()
        except torch.OutOfMemoryError:
            if not self.entries:
                raise
            torch.cuda.synchronize()
            while self.evict_oldest():
                pass
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved()
            entry = make()
        self.last_capture_bytes = max(0, torch.cuda.memory_reserved() - reserved)
        return entry


def _sampler_signature(sampler: EulerSampler) -> tuple:
    config = sampler.config
    return (
        tuple(
            (field.name, getattr(config, field.name))
            for field in dataclasses.fields(config)
        )
        if dataclasses.is_dataclass(config)
        else (repr(config),)
    )


def _fold_key(
    fold: Fold,
    leaves: list[torch.Tensor],
    structure: Any,
    sampler: EulerSampler,
    coords: torch.Tensor,
    row: torch.Tensor | None,
    generator: torch.Generator | None,
    glue_on: bool,  # noqa: FBT001 - private helper, called positionally
) -> tuple:
    """Build the per-shape key.

    The key holds the structure and layout of the fold's tensors, the engine's own
    signature of them, the iterate and conditioning row, the sampler's configuration,
    and the precision state a capture bakes in.
    """
    return (
        structure,
        engine_capture.signature(fold.model, *leaves, coords, row),
        _sampler_signature(sampler),
        id(generator),
        glue_on,
        torch.get_float32_matmul_precision(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )


def _cache_of(scopes: object, fold: Fold | None) -> SamplerGraphCache | None:
    """Return the scope's cache when this call may use it.

    It may when a fold was handed over, in inference, on an engine with the refresh
    API.
    """
    cache = getattr(scopes, "cache", None)
    if fold is None or cache is None or torch.is_grad_enabled():
        if cache is not None and fold is not None:
            cache.bump("uncached")
        return None
    return cache


def _capture_entry(
    cache: SamplerGraphCache,  # noqa: ARG001 - part of the call signature
    fold: Fold,
    sampler: EulerSampler,
    tree: Any,
    coords: torch.Tensor,
    sigmas: torch.Tensor,
    row: torch.Tensor | None,
    *,
    generator: torch.Generator | None,
    coordinate_dims: int,
    scopes: Callable[[], contextlib.AbstractContextManager],
    use_glue: bool,
) -> _Entry:
    """Capture the step on a STATIC clone of the fold.

    The caller's tensors stay the caller's; later folds are written into the clone.
    """
    marks = engine_capture.cache_marks()
    static = _map(tree, _static_clone)
    _, leaves = _flatten(static)
    with fold.bind(static):
        denoise, augment, align, rigid_mask = fold.pieces(static)
        step = build_step(
            sampler,
            denoise,
            generator=generator,
            augment=augment,
            align=align,
            coordinate_dims=coordinate_dims,
            rigid_mask=rigid_mask if use_glue else None,
        )
        captured = _CapturedStep(
            step, coords, sigmas[0], sigmas[1], row, generator=generator, scopes=scopes
        )
    glue = getattr(step, "glue", None)
    mask_copies = []
    if (
        glue is not None
        and rigid_mask is not None
        and glue.mask.data_ptr() != rigid_mask.data_ptr()
    ):
        # the float flat mask the glue made from the atom mask: a copy outside the
        # graph
        mask_copies.append((glue.mask, rigid_mask))
    # only copies of a leaf of this fold's tree (a stale entry of some other call is
    # not ours to rewrite)
    casts = [cast for cast in fold.casts() if any(cast[0] is leaf for leaf in leaves)]
    entry = _Entry(captured, static, leaves, casts, mask_copies)
    # the engine glue modules (they own device buffers the graph reads); NOT the
    # closures over the fold (they hold the caller's embeddings: the previous fold's
    # tensors must not stay alive in the cache)
    entry.keepalive = glue

    entry.engine_held = engine_capture.hold_new(marks)
    entry.retained += engine_capture.tensor_bytes(entry.engine_held)
    return entry


def _sample_cached(
    cache: SamplerGraphCache,
    fold: Fold,
    sampler: EulerSampler,
    coords: torch.Tensor,
    table: torch.Tensor | None,
    pairs: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: torch.Generator | None,
    coordinate_dims: int,
    scopes: Callable[[], contextlib.AbstractContextManager],
    use_glue: bool,
) -> torch.Tensor | None:
    """Run the steps through the per-shape cache.

    ``None`` when this fold cannot use it (the caller then runs the per-fold capture).
    """
    tree = fold.tree()
    try:
        structure, leaves = _flatten(tree)
    except (TypeError, ValueError):
        cache.bump("uncached")
        return None
    if engine_capture.tensor_bytes(leaves) > cache.max_fold_bytes:
        cache.bump("uncached")
        return None
    row0 = None if table is None else table[0]
    key = _fold_key(fold, leaves, structure, sampler, coords, row0, generator, use_glue)
    cache.check_weights(fold.model)
    entry = cache.lookup(key)
    if entry is not None:
        try:
            entry.load(leaves, fold.model)
        except (engine_capture.stale_error(), _LayoutMismatch):
            # a weight or an input layout changed under the graph: capture this
            # fold again, once
            cache.drop(key)
            cache.bump("stale_recaptures")
            entry = None
        else:
            cache.bump("refreshes", 1 if entry.tables is not None else 0)
            entry.captured.coords.copy_(coords)
    kept = True
    if entry is None:
        entry = cache.build(
            lambda: _capture_entry(
                cache,
                fold,
                sampler,
                tree,
                coords,
                sigmas,
                row0,
                generator=generator,
                coordinate_dims=coordinate_dims,
                scopes=scopes,
                use_glue=use_glue,
            )
        )
        # no engine handle: nothing could refresh it for another fold
        if entry.tables is None:
            kept = False
            cache.bump("uncached")
        else:
            kept = cache.insert(key, entry, cache.budget_bytes)
    captured = entry.captured
    for index in range(len(pairs)):
        captured(pairs[index], None if table is None else table[index])
    result = captured.coords.clone()
    if not kept:
        torch.cuda.synchronize()
        entry.release()
    return result


def sample(
    sampler: EulerSampler,
    denoise: Callable,
    shape: tuple[int, ...],
    sigmas: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None,
    augment: Callable | None,
    align: Callable | None,
    coordinate_dims: int,
    scopes: Callable[[], contextlib.AbstractContextManager],
    dtype: torch.dtype = torch.float32,
    rigid_mask: torch.Tensor | None = None,
    conditioning: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fold: Fold | None = None,
) -> torch.Tensor:
    """Sample a trajectory with the complete step captured as a graph.

    ``EulerSampler.sample`` (one chunk, all samples drawn first) with the step in
    one graph.

    ``conditioning(sigma_hat)`` makes everything the denoiser needs that depends on
    the noise level and not on the iterate, for every step at once (``[steps,
    ...]``); the step then reads its row as ``denoise(coords, sigma_hat, row)``
    instead of making it per step.

    ``fold`` (a :class:`Fold`) and ``scopes.cache`` (a :class:`SamplerGraphCache`)
    make the graph a per-shape one: captured on the first fold of a shape, served to
    every later fold by writing the fold into the graph's static buffers. Without them
    the graph is built for this fold and dropped (``denoise`` / ``augment`` /
    ``align`` / ``rigid_mask`` then read the fold's own tensors).
    """
    # the training rollout graph (train/rollout_graph.py): the loop runs in the
    # caller's capture
    if getattr(scopes, "inline", False):
        return sample_inline(
            sampler,
            denoise,
            shape,
            sigmas,
            device=device,
            generator=generator,
            augment=augment,
            align=align,
            coordinate_dims=coordinate_dims,
            dtype=dtype,
            conditioning=conditioning,
        )
    coords = sigmas[0] * torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    table = (
        None if conditioning is None else conditioning(_hatted(sampler.config, sigmas))
    )
    pairs = torch.stack([sigmas[:-1], sigmas[1:]], 1)
    cache = _cache_of(scopes, fold)
    if cache is not None:
        use_glue = rigid_mask is not None
        result = _sample_cached(
            cache,
            fold,  # pyright: ignore[reportArgumentType]  # _cache_of returns a cache only with a fold
            sampler,
            coords,
            table,
            pairs,
            sigmas,
            generator=generator,
            coordinate_dims=coordinate_dims,
            scopes=scopes,
            use_glue=use_glue,
        )
        if result is not None:
            return result
    step = build_step(
        sampler,
        denoise,
        generator=generator,
        augment=augment,
        align=align,
        coordinate_dims=coordinate_dims,
        rigid_mask=rigid_mask,
    )
    captured = None
    for index in range(len(pairs)):
        row = None if table is None else table[index]
        if captured is None:
            captured = _CapturedStep(
                step,
                coords,
                sigmas[0],
                sigmas[1],
                row,
                generator=generator,
                scopes=scopes,
            )
        captured(pairs[index], row)
    return captured.coords.clone()  # pyright: ignore[reportOptionalMemberAccess]  # a schedule has at least one step


def sample_inline(
    sampler: EulerSampler,
    denoise: Callable,
    shape: tuple[int, ...],
    sigmas: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None,
    augment: Callable | None,
    align: Callable | None,
    coordinate_dims: int,
    dtype: torch.dtype = torch.float32,
    conditioning: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run the whole trajectory as one straight-line sequence of ops.

    No host read (the churn branch is a ``where`` on the device scalars of
    :func:`build_step`), no capture of its own and no cache. A caller that captures
    (the training mini-rollout, ``train/rollout_graph.py``) or runs it eagerly gets
    the same arithmetic. ``conditioning`` is evaluated here, so a capture around this
    call records it too: every replay makes the conditioning table from the weights of
    that moment.
    """
    coords = sigmas[0] * torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    table = (
        None if conditioning is None else conditioning(_hatted(sampler.config, sigmas))
    )
    step = build_step(
        sampler,
        denoise,
        generator=generator,
        augment=augment,
        align=align,
        coordinate_dims=coordinate_dims,
        rigid_mask=None,
    )
    for index in range(len(sigmas) - 1):
        coords = step(
            coords,
            sigmas[index],
            sigmas[index + 1],
            None if table is None else table[index],
        )
    return coords
