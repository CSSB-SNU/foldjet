"""The ONE place FoldJet calls the engine's capture API (``miniworld_engine.capture``: ``static_weights`` / ``static_inputs`` /
``clear_weight_caches`` / ``refresh_static_inputs``, public since engine 2.5.6; R1). Everything else imports from here."""

from __future__ import annotations

import contextlib
from typing import Any


def static_weights() -> contextlib.AbstractContextManager[None]:
    """The weights stay fixed between graph replays (inference): the engine may serve its eager weight re-layouts to a capture."""
    from miniworld_engine import capture

    return capture.static_weights()


def static_inputs() -> contextlib.AbstractContextManager[Any]:
    """The pair and conditioning inputs of the captured callable keep their contents for the life of the graph.

    The context yields the engine's ``StaticTables`` handle (the hoisted tables the captures inside the block read) on an engine that has the
    refresh API (:func:`refresh_available`), else whatever the engine's context yields (``None``)."""
    from miniworld_engine import capture

    return capture.static_inputs()


def refresh_available() -> bool:
    """Whether the installed engine can serve a NEW item from a graph captured on another (R26: ``StaticTables.refresh`` and ``signature``)."""
    try:
        from miniworld_engine import capture
    except ImportError:
        return False
    return all(
        hasattr(capture, name)
        for name in (
            "refresh_static_inputs",
            "StaticTables",
            "signature",
            "StaleStaticInputs",
        )
    )


def stale_error() -> type[BaseException]:
    """The engine's ``StaleStaticInputs`` (a class nothing raises on an engine without the refresh API)."""
    if refresh_available():
        from miniworld_engine.capture import StaleStaticInputs

        return StaleStaticInputs
    return _NeverRaised


class _NeverRaised(Exception):  # noqa: N818 - a placeholder that is never raised
    """Stands for ``StaleStaticInputs`` on an engine that has none."""


def signature(root: Any, *tensors: Any) -> tuple:
    """The engine's per-shape cache key of a graph over ``tensors`` (identity of ``root`` and each tensor's layout); requires :func:`refresh_available`."""
    from miniworld_engine.capture import signature as engine_signature

    return engine_signature(root, *tensors)


def clear_weight_caches() -> None:
    """Drop the engine's eager weight re-layout caches before weights are replaced.

    The H100 kernels keep weight forms (``w.t().contiguous()``, TF32-rounded triples, packed blocks) keyed by tensor identity and version; a
    new weight allocated at the address of a freed one must not be served the old copy. A process that builds a second model, or a test
    that builds many, clears them first. The sampler graphs cached per shape (``models.sampler_graph``) read those entries: they are dropped first."""
    from foldjet.models import sampler_graph

    sampler_graph.release_all_caches()
    from miniworld_engine import capture

    capture.clear_weight_caches()  # every registered store (packs, hoisted tables, forms, TF32 operands) since engine 2.5.6


def hold_caches() -> list:
    """Strong references to every entry now in the engine's weight / per-item caches (``miniworld_engine.capture.weight_caches``).

    A captured graph reads some of those entries (a conditioning weight chunk of the token DiT, capture-scoped packs) WITHOUT the engine
    recording them for refresh; the engine evicts them on later work (another shape's capture, an LRU of 4). A graph that outlives the call that
    captured it (the per-shape sampler graph) keeps this list, so what its kernels read is not freed -- and, once the allocator releases
    the cached block, unmapped -- under it. Empty when the engine has no ``weight_caches``."""
    try:
        from miniworld_engine.capture import weight_caches
    except ImportError:
        return []
    return [list(store.items()) for store in weight_caches().values()]


def tensor_bytes(held: object, seen: set[int] | None = None) -> int:
    """Bytes of the distinct CUDA storages reachable from ``held`` (lists / tuples / dicts of tensors)."""
    import torch

    seen = set() if seen is None else seen
    if isinstance(held, torch.Tensor):
        if not held.is_cuda:
            return 0
        storage = held.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return storage.nbytes()
    if isinstance(held, dict):
        return sum(tensor_bytes(v, seen) for v in held.values())
    if isinstance(held, (list, tuple)):
        return sum(tensor_bytes(v, seen) for v in held)
    return 0


def cache_marks() -> dict[str, dict]:
    """``{cache: {key: id(entry)}}`` of the engine's caches now (for :func:`hold_new`)."""
    try:
        from miniworld_engine.capture import weight_caches
    except ImportError:
        return {}
    return {
        name: {key: id(value) for key, value in list(store.items())}
        for name, store in weight_caches().items()
    }


def hold_new(marks: dict[str, dict]) -> list:
    """Strong references to the engine cache entries made (or replaced) since ``marks`` was taken: what a capture and its warm-up built.

    See :func:`hold_caches`: a graph reads some of them without the engine recording it, and the engine evicts them on later work."""
    try:
        from miniworld_engine.capture import weight_caches
    except ImportError:
        return []
    held = []
    for name, store in weight_caches().items():
        old = marks.get(name, {})
        held.append(
            [
                (key, value)
                for key, value in list(store.items())
                if old.get(key) != id(value)
            ]
        )
    return held
