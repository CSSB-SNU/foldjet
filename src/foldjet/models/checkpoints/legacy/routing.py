"""The dense released-layout graph's bindings to the shared reference equations.

Converter graph only. The dense modules (:mod:`foldjet.models.checkpoints.legacy`)
hold the released parameter layout; these functions bind that layout to the reference
equations in :mod:`.reference`. Nothing here is imported at inference time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from foldjet.models.checkpoints.legacy.reference.transition import (
    TransitionConfig,
    transition,
)
from foldjet.models.checkpoints.legacy.reference.triangle import (
    TriangleConfig,
    triangle_update,
)

if TYPE_CHECKING:
    from typing import Any


def transition_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply the complete transition including its residual."""
    return x + module(x)


def _inverse_length(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """1 / the real sequence length as a 0-d device tensor.

    Counted without a host read (graph-capture safe).
    """
    if mask is None:
        return torch.full((), 1.0 / x.shape[-2], device=x.device)
    count = mask.reshape(-1, mask.shape[-1]).any(0).sum().clamp_min(1)
    return count.float().reciprocal()


def triangle_residual(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply the complete triangle multiplication including its residual.

    RoseTTAFold3: the contraction divided by the real length.
    """
    scale = (
        _inverse_length(x, mask) if getattr(module, "divide_by_length", False) else None
    )
    return x + reference_triangle(module, x, mask, contraction_scale=scale)


# Frozen configurations must be built outside forward: Pydantic validation inside a
# compiled call breaks its graph.
_DEFAULT_TRANSITION_CONFIG = TransitionConfig()
_TRIANGLE_CONFIGS = {
    outgoing: TriangleConfig(outgoing=outgoing) for outgoing in (False, True)
}


def reference_transition(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bind the dense transition projection names to the shared equation."""
    # nn.Module attributes are dynamic (``__getattr__``): the submodules typed as Any
    layers: Any = module
    return transition(
        x,
        normalize=layers.input_layer_norm,
        project_a=lambda value: (
            value.to(layers.transition1.weight.dtype) @ layers.transition1.weight.T
        ),
        project_b=None,
        project_out=layers.transition2,
        config=getattr(module, "transition_config", _DEFAULT_TRANSITION_CONFIG),
        training=module.training,
    )


def reference_triangle(
    module: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    contraction_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map the packed checkpoint weights to the shared contraction (a raw delta)."""
    # nn.Module attributes are dynamic (``__getattr__``): the submodules typed as Any
    layers: Any = module
    return triangle_update(
        x,
        mask,
        normalize=layers.left_norm_input,
        project_a=layers.projection,
        gate_a=layers.gate,
        center=layers.center_norm,
        project_out=layers.output_projection,
        gate_out=layers.gating_linear,
        config=_TRIANGLE_CONFIGS[module.equation == "cik,cjk->cij"],
        contraction_scale=contraction_scale,
    )
