# ruff: noqa: E501
"""The two FoldForge-side sources of run-to-run differences (2026-11-13): the chirality gradient's float-atomic sum and the post-atom-cond
LayerNorm that inductor compiled with a timing-picked block size inside the sampler step."""

from __future__ import annotations

import pytest
import torch

from foldforge.modules.dense import atom_cross_attention as aca
from foldforge.modules.dense.chirality import _segment_sum, dihedral_error_gradient


def _overlapping(generator: torch.Generator, atoms: int = 40, centres: int = 90):
    index = torch.randint(
        0, atoms - 5, (centres * 4,), generator=generator
    )  # atoms >= atoms - 5 stay empty
    values = (
        torch.randn(3, centres * 4, 3, generator=generator)
        * torch.logspace(-3, 3, centres * 4)[None, :, None]
    )
    return values, index, atoms


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segment_sum_is_the_float64_sum_whatever_the_order(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    values, index, atoms = _overlapping(generator)
    reference = (
        torch.zeros(3, atoms, 3, dtype=torch.float64)
        .index_add_(1, index, values.double())
        .float()
    )
    got = _segment_sum(values, index, atoms)
    assert got.dtype == torch.float32
    torch.testing.assert_close(
        got, reference, rtol=0, atol=0
    )  # the rounded float64 sum, bit for bit
    for _ in range(3):
        permutation = torch.randperm(index.numel(), generator=generator)
        assert torch.equal(
            _segment_sum(values[:, permutation], index[permutation], atoms), got
        )
    assert torch.equal(got[:, -5:], torch.zeros(3, 5, 3))  # atoms in no centre


def test_segment_sum_zeroes_non_finite_values_of_their_atom_only() -> None:
    values = torch.ones(1, 6, 3)
    values[0, 1, 0] = float("nan")
    values[0, 4, 1] = float("inf")
    index = torch.tensor([0, 0, 1, 2, 2, 3])
    got = _segment_sum(values, index, 5)
    assert torch.isfinite(got).all()
    assert (
        got[0, 0, 0] == 1 and got[0, 1, 0] == 1 and got[0, 3, 0] == 1
    )  # one NaN row costs only its own term
    assert got[0, 4].abs().sum() == 0


def test_dihedral_error_gradient_is_repeatable_and_centre_order_free() -> None:
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(2, 30, 3, generator=generator)
    centres = torch.stack(
        [torch.randperm(30, generator=generator)[:4] for _ in range(60)]
    )  # overlapping centres
    targets = torch.randn(60, generator=generator)
    first = dihedral_error_gradient(xyz, centres, targets)
    assert torch.equal(first, dihedral_error_gradient(xyz, centres, targets))
    permutation = torch.randperm(60, generator=generator)
    shuffled = dihedral_error_gradient(xyz, centres[permutation], targets[permutation])
    torch.testing.assert_close(
        shuffled, first, rtol=1e-5, atol=1e-6
    )  # the centres' own float32 terms are the same set; only the sum order moved
    assert dihedral_error_gradient(xyz, centres[:0], targets[:0]).abs().sum() == 0


class _Stack(
    torch.nn.Module
):  # not a DiffusionCrossAttTransformer: stands for an engine stack
    pass


def _state(channels: int = 8) -> aca.AtomFoldState:
    q, k = torch.randn(2, 4, channels), torch.randn(2, 6, channels)
    return aca.AtomFoldState(
        query_base=q,
        queries_mask=torch.ones(2, 4),
        queries_single_cond=q,
        keys_mask=torch.ones(2, 6),
        keys_single_cond=k,
        pair_cond=torch.zeros(2, 4, 6, 1),
    )


def _modules(norm: torch.nn.Module | None):
    encoder = torch.nn.Module()
    encoder.atom_transformer_encoder = _Stack()
    decoder = torch.nn.Module()
    decoder.atom_transformer_decoder = _Stack()
    decoder.post_atom_cond_layer_norm = norm
    return encoder, decoder


def test_engine_stack_hoists_only_the_post_norm() -> None:
    norm = torch.nn.LayerNorm(8)
    state = _state()
    encoder, decoder = _modules(norm)
    with torch.no_grad():
        out = aca.hoist_conditioning(state, encoder, decoder)
        expected = (norm(state.queries_single_cond), norm(state.keys_single_cond))
    projections, (queries, keys, decoder_projections) = out.hoisted
    assert (
        projections is None and decoder_projections is None
    )  # the engine stack projects its own conditioning
    assert torch.equal(queries, expected[0]) and torch.equal(keys, expected[1])


def test_engine_stack_without_post_norm_or_with_gradients_is_left_alone() -> None:
    state = _state()
    with torch.no_grad():
        assert aca.hoist_conditioning(state, *_modules(None)) is state
    encoder, decoder = _modules(torch.nn.LayerNorm(8))
    with torch.enable_grad():
        assert aca.hoist_conditioning(state, encoder, decoder) is state
