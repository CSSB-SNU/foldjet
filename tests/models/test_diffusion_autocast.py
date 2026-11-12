"""Which autocast the diffusion head runs under: ``af3_default`` (``reference_precision``) disables the bf16 autocast for it only."""

from __future__ import annotations

import torch
from torch import nn

from foldforge.modules.engine.configure import activation_autocast, diffusion_autocast


class _Fake(nn.Module):
    def __init__(self, precision: str, reference: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.foldforge_precision = precision
        self.foldforge_activation_dtype = torch.bfloat16
        self.reference_precision = reference


def _enabled_in(manager) -> bool:
    with manager:
        return torch.is_autocast_enabled("cpu")


def test_bf16_head_stays_under_the_model_autocast() -> None:
    model = _Fake("master", reference=False)
    with activation_autocast(model):
        assert _enabled_in(diffusion_autocast(model))
    assert _enabled_in(diffusion_autocast(model))


def test_af3_default_head_runs_outside_the_autocast_the_trunk_runs_in() -> None:
    model = _Fake("master", reference=True)
    with activation_autocast(model):
        assert torch.is_autocast_enabled("cpu")  # the trunk / entry points
        assert not _enabled_in(diffusion_autocast(model))  # the head
        assert torch.is_autocast_enabled("cpu")  # restored for the code after it


def test_fp32_precision_has_no_autocast_anywhere() -> None:
    model = _Fake("fp32", reference=True)
    assert not _enabled_in(diffusion_autocast(model))
    assert not _enabled_in(activation_autocast(model))
