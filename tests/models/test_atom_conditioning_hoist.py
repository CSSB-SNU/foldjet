# ruff: noqa: E501, FBT001, FBT003
"""The atom stack's conditioning projections made once per fold (``project_cond`` / ``hoisted``) equal the ones made at every call.

ESMFold2's atom stack is FoldForge-own; its AdaLN scales / shifts and output gates read only the atoms' single conditioning, which is the same
at every sampler step, so ``hoist_conditioning`` makes them once and the step takes them back. The hoisted call must compute the same values.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from foldforge.modules.dense.diffusion_transformer import (
    AdaLNZero,
    AdaptiveLayerNorm,
    CrossAttention,
    DiffusionCrossAttTransformer,
    conditioned_transition,
)
from foldforge.modules.dense.spec import spec_for


def randomise(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))


@pytest.mark.parametrize("rms", [False, True])
def test_adaptive_layer_norm_projected_equals_unprojected(rms: bool) -> None:
    torch.manual_seed(0)
    norm = AdaptiveLayerNorm(32, 24, True, identity_scale=rms, rms_norm=rms, activate_cond=rms)
    randomise(norm, 1)
    x, cond = torch.randn(3, 5, 32), torch.randn(3, 5, 24)
    assert torch.equal(norm(x, cond), norm(x, cond, norm.project_cond(cond)))


@pytest.mark.parametrize("rms", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_output_gate_hoisted_equals_unhoisted(rms: bool, project: bool) -> None:
    torch.manual_seed(0)
    gate = AdaLNZero(32, 32, 24, True, project=project, raw_gate=rms)
    randomise(gate, 2)
    x, cond = torch.randn(3, 5, 32), torch.randn(3, 5, 24)
    assert torch.equal(gate(x, cond), gate(x, cond, hoisted_gate=gate.project_cond(cond)))


@pytest.mark.parametrize("up_gate", [False, True])
def test_transition_hoisted_equals_unhoisted(up_gate: bool) -> None:
    torch.manual_seed(0)
    spec = spec_for("esmfold2", "fast")
    spec = replace(spec, transition_up_gate=up_gate)
    transition = conditioned_transition(spec)(
        c_x=32, c_single_cond=24, use_single_cond=True, identity_scale=True, rms_conditioning=True
    )
    randomise(transition, 3)
    x, cond = torch.randn(3, 5, 32), torch.randn(3, 5, 24)
    assert torch.equal(transition(x, cond), transition(x, cond, transition.project_cond(cond)))


def test_cross_attention_hoisted_equals_unhoisted() -> None:
    torch.manual_seed(0)
    attention = CrossAttention(
        key_dim=32, value_dim=32, c_single_cond=32, num_head=4, identity_scale=True, rms_conditioning=True,
        gating_query=True, project_output=True, zero_bias=False, key_masked=True,
    )
    randomise(attention, 4)
    subsets, queries, keys = 2, 4, 6
    x_q, x_k = torch.randn(subsets, queries, 32), torch.randn(subsets, keys, 32)
    cond_q, cond_k = torch.randn(subsets, queries, 32), torch.randn(subsets, keys, 32)
    arguments = {
        "x_q": x_q,
        "x_k": x_k,
        "mask_q": torch.ones(subsets, queries),
        "mask_k": torch.ones(subsets, keys),
        "pair_logits": torch.randn(subsets, 4, queries, keys),
        "single_cond_q": cond_q,
        "single_cond_k": cond_k,
    }
    assert torch.equal(attention(**arguments), attention(**arguments, hoisted=attention.project_cond(cond_q, cond_k)))


def test_stack_exposes_one_entry_per_block() -> None:
    spec = spec_for("esmfold2", "fast")
    stack = DiffusionCrossAttTransformer(c_query=128, c_single_cond=128, c_pair_cond=16, num_blocks=3, num_head=4, spec=spec)
    cond = torch.randn(2, 4, 128)
    hoisted = stack.project_cond(cond, torch.randn(2, 6, 128))
    assert len(hoisted) == 3
    assert all(len(block) == 2 for block in hoisted)
