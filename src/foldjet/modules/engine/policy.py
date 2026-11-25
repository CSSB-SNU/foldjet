"""FoldJet's side of the engine's backend contract: per-module backend choice, the guard that asks ``unsupported`` before a call, and
the fallback / error policy.

The engine never falls back (a call outside a backend's range raises ``BackendUnsupported``) and has no global switch. This module does
what the engine leaves to its caller:

1. :func:`apply_plan` walks a model, gives every engine op module the backend the plan chose for ITS class
   (:func:`~foldjet.modules.engine.backends.choose_backend`) and installs the guard. Composite blocks (``PairformerBlock``,
   ``Pairformer``) stay on ``pytorch`` themselves and their children are set one by one, so an op that has no kernel is refused alone
   instead of taking the whole block down.
2. The guard asks ``module.unsupported(*args)`` before the call. A reason under ``engine.unsupported: fallback`` runs the next backend
   of the engine ladder that takes the call (``cuda`` -> ``triton`` -> ``pytorch``, :func:`fallback_backend`) and logs the reason
   ONCE; under ``error`` it raises ``BackendUnsupportedError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from foldjet.modules.engine.backends import (
    BackendPlan,
    BackendUnsupportedError,
    op_of,
)
from foldjet.modules.engine.serving import observe_call

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger("foldjet.backend")

_LOGGED: set[tuple[str, str, str]] = set()
_GUARDED = "_foldjet_guarded"
_COMPOSITES = ("PairformerBlock", "Pairformer")


def forget_logged_refusals() -> None:
    """Let the next refused call log again (tests; a long-lived server that wants a fresh line per request)."""
    _LOGGED.clear()


def refuse(module: nn.Module, reason: str, served_by: str = "pytorch") -> bool:
    """A call the module's backend refuses: True when the caller must run ``served_by`` (the next backend of the ladder that takes the
    call), after ONE log line per (class, backend, reason); raises under ``engine.unsupported: error`` (the module's
    ``foldjet_unsupported``)."""
    cls, backend = type(module).__name__, cast("str", module.backend)
    if getattr(module, "foldjet_unsupported", "fallback") == "error":
        message = f"{cls} backend {backend!r} refuses a call (engine.unsupported: error): {reason}"
        raise BackendUnsupportedError(message)
    key = (cls, backend, reason)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.warning(
            "%s backend %s refuses a call: %s; running its %s backend for it",
            cls,
            backend,
            reason,
            served_by,
        )
    return True


def fallback_backend(module: nn.Module, backend: str, *args: Any, **kwargs: Any) -> str:
    """The backend a call refused by ``backend`` runs on: the engine's own ladder, ``cuda`` -> ``triton`` -> ``pytorch``.

    ``triton`` when the class has it and its own range takes the call (``_supports_triton``; without one it claims every call), else the
    reference ``pytorch``. E.g. MSAPairWeightedAveraging past 2048 tokens: the H100 kernels hold a row of N logits (N <= 2048 in
    inference), the Triton kernels have no such limit."""
    if backend == "cuda" and "triton" in getattr(module, "backends", ()):
        check = getattr(module, "_supports_triton", None)
        if check is None or check(*args, **kwargs) is None:
            return "triton"
    return "pytorch"


def guard(module: nn.Module, unsupported: str = "fallback") -> nn.Module:
    """Wrap ``module.forward`` so that a call the selected backend refuses follows ``unsupported`` (idempotent; the policy is read per
    call from ``module.foldjet_unsupported``)."""
    module.foldjet_unsupported = unsupported  # pyright: ignore[reportArgumentType]  # plain attribute
    if module.__dict__.get(_GUARDED):
        return module
    original = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        backend = cast("str", module.backend)
        if backend == "pytorch":
            observe_call(module, backend, None)
            return original(*args, **kwargs)
        reason = module.unsupported(*args, **kwargs)  # pyright: ignore[reportCallIssue]  # engine op method
        if reason is None:
            observe_call(module, backend, None)
            return getattr(module, f"_forward_{backend}")(*args, **kwargs)
        served_by = fallback_backend(module, backend, *args, **kwargs)
        refuse(module, reason, served_by)
        observe_call(module, backend, reason, served_by)
        module.backend = served_by  # pyright: ignore[reportArgumentType]  # plain attribute
        try:
            return getattr(module, f"_forward_{served_by}")(*args, **kwargs)
        finally:
            module.backend = backend  # pyright: ignore[reportArgumentType]  # plain attribute

    module.forward = forward
    object.__setattr__(module, _GUARDED, True)
    return module


def apply_plan(
    model: nn.Module, plan: BackendPlan, device: object | None = None
) -> dict[str, dict[str, str | None]]:
    """Give every engine op module of ``model`` its backend and guard; returns ``{module name: {"backend", "requested", "reason"}}``.

    Does not descend into an op module (its parts are not ops of their own: a ``Transition``'s ``LayerNorm`` is a parameter holder). The
    sampler glue (``AugmentNoise`` ...) is an op like the others."""
    report: dict[str, dict[str, str | None]] = {}

    def visit(name: str, module: nn.Module) -> None:
        if type(module).__name__ in _COMPOSITES and hasattr(module, "set_backend"):
            # also puts every child on pytorch; the loop below gives each its own backend
            module.set_backend("pytorch")  # pyright: ignore[reportCallIssue]  # engine op method
            module.foldjet_unsupported = plan.unsupported  # pyright: ignore[reportArgumentType]  # plain attribute
        elif op_of(module) is not None:
            backend, reason = plan.for_class(type(module), device)
            if reason is not None and plan.unsupported == "error":
                raise BackendUnsupportedError(
                    f"{name or '<root>'} ({type(module).__name__}): {reason}"
                )
            module.set_backend(backend)  # pyright: ignore[reportCallIssue]  # engine op method
            guard(module, plan.unsupported)
            op = cast("str", op_of(module))  # an op module (checked above)
            report[name] = {
                "op": op,
                "backend": backend,
                "requested": plan.requested(op),
                "reason": reason,
            }
            return
        for child_name, child in module.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", model)
    refused = {n: r for n, r in report.items() if r["reason"] is not None}
    if refused:
        lines = "; ".join(
            f"{n} ({r['requested']}: {r['reason']})"
            for n, r in sorted(refused.items())[:8]
        )
        logger.warning(
            "backend %s: %d module(s) run the engine pytorch backend: %s%s",
            plan.backend,
            len(refused),
            lines,
            " ..." if len(refused) > 8 else "",
        )
    return report
