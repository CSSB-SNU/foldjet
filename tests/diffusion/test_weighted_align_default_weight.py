"""weighted_align without a weight aligns with unit weights.

It raised AttributeError before (`weight.shape` read before the None branch).
"""

import torch

from foldjet.diffusion.geometry.alignment import weighted_align


def test_weighted_align_none_weight_is_unit_weight():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 17, 3, generator=g)
    y = torch.randn(2, 17, 3, generator=g)
    assert torch.equal(weighted_align(x, y), weighted_align(x, y, torch.ones(2, 17)))
