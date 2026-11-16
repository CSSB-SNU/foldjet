"""ESMFold2's atom stack on the engine's ``AtomDiT("esmfold2_pair")`` block.

It equals FoldForge's own ``DiffusionCrossAttTransformer``.

The released-layout weights convert to the block layout
(``released.esmfold2_atom_blocks_from_released``), the stack passes the per-atom
rotary tables, and a native file written while the stack was FoldForge's own loads
into the engine layout. Skipped on an engine without the class (2.5.1): there
``model_keys("esmfold2").atom_dit`` is ``None`` and the family keeps its own stack.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from foldforge.models.checkpoints import miniworld
from foldforge.models.checkpoints.legacy import released
from foldforge.modules.dense import atom_cross_attention as aca
from foldforge.modules.dense.diffusion_transformer import DiffusionCrossAttTransformer
from foldforge.modules.dense.spec import ESMFOLD2
from foldforge.modules.engine.modules import engine_has_atom_dit, model_keys
from foldforge.modules.engine.transformers import AtomDiTStack, atom_dit_stack
from foldforge.utils import atom_layout

pytestmark = pytest.mark.skipif(
    not engine_has_atom_dit("esmfold2_pair"),
    reason="engine without AtomDiT('esmfold2_pair') (older than R6)",
)

C, HEADS, BLOCKS, KEYS, QUERIES = 128, 4, 3, 192, 32
# 7 query windows, the last partly padding: the real atoms come first
ATOMS, REAL = 200, 187


def _randomize(module: nn.Module, std: float = 0.12) -> nn.Module:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, std)
    return module


def _layout(seed: int = 0) -> dict[str, Any]:
    """The tensors a fold's atom layout hands the stack.

    Queries [W, 32, C], the 192-key windows slid AF3's way, the pair, the rank window
    mask and the rotary tables of queries and keys, from one reference conformer.
    """
    from miniworld_engine.modules.functional import slide_starts

    generator = torch.Generator().manual_seed(seed)
    windows = (ATOMS + QUERIES - 1) // QUERIES
    padded = windows * QUERIES
    real = torch.arange(padded) < REAL
    starts = slide_starts(torch.tensor([REAL]), windows, KEYS)[0]  # [W]
    # [W, 192] into the flat padded atom list
    idx = starts[:, None] + torch.arange(KEYS)
    in_range = idx < padded
    idx = idx.clamp(max=padded - 1)
    keys_mask = (in_range & real[idx]).to(torch.float32)
    gather = atom_layout.GatherInfo(
        gather_idxs=idx, gather_mask=in_range, input_shape=(windows, QUERIES)
    )
    queries_mask = real.reshape(windows, QUERIES).to(torch.float32)
    act = torch.randn(windows, QUERIES, C, generator=generator)
    cond = torch.randn(windows, QUERIES, C, generator=generator)
    pair = torch.randn(windows, QUERIES, KEYS, 16, generator=generator)
    position = torch.randn(windows, QUERIES, 3, generator=generator) * 4.0
    uid = (torch.arange(padded) // 9).reshape(windows, QUERIES)
    keys_position = atom_layout.convert(gather, position, layout_axes=(-3, -2))
    keys_uid = atom_layout.convert(gather, uid, layout_axes=(-2, -1))
    head_dim = C // HEADS
    rope_q = aca.build_atom_rope(position, uid, head_dim, ESMFOLD2.atom_rope)
    rope_k = aca.build_atom_rope(keys_position, keys_uid, head_dim, ESMFOLD2.atom_rope)
    keys_cond = atom_layout.convert(gather, cond, layout_axes=(-3, -2))
    pair_mask = aca.sliding_window_mask(
        queries_mask, keys_mask, gather, ESMFOLD2.atom_window_half
    )
    return {
        "act": act,
        "cond": cond,
        "keys_cond": keys_cond,
        "pair": pair,
        "gather": gather,
        "queries_mask": queries_mask,
        "keys_mask": keys_mask,
        "pair_mask": pair_mask,
        "rope_q": rope_q,
        "rope_k": rope_k,
    }


def _dense_and_stack() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(0)
    dense = _randomize(DiffusionCrossAttTransformer(spec=ESMFOLD2)).eval()
    stack = atom_dit_stack("esmfold2", n_blocks=BLOCKS, precision="fp32")
    blocks = released.esmfold2_atom_blocks_from_released(
        dense.state_dict(), n_head=HEADS
    )
    for block, tensors in zip(stack.blocks, blocks, strict=True):
        block.load_state_dict(tensors, strict=True)
    return dense, stack.eval()


def test_the_family_selects_the_engine_block():
    assert model_keys("esmfold2").atom_dit == "esmfold2_pair"
    assert model_keys("esmfold2-fast").atom_dit == "esmfold2_pair"
    stack = atom_dit_stack("esmfold2", n_blocks=BLOCKS, precision="fp32")
    assert isinstance(stack, AtomDiTStack)
    assert stack.rotary
    # the atom cross-attention builds its rotary tables for this width (the whole model
    # failed on its absence, which no stack-only test saw)
    assert isinstance(stack.key_dim_per_head, int)
    assert stack.key_dim_per_head > 0
    assert type(stack.blocks[0]).__name__ == "AtomDiTESMFold2Pair"


def test_the_converted_blocks_have_exactly_the_blocks_parameters():
    dense, stack = _dense_and_stack()
    # strict loads above already prove the key sets; every dense tensor was read by the
    # conversion
    tracked = released.Tracked(dense.state_dict())
    released.esmfold2_atom_blocks_from_released(tracked, n_head=HEADS)
    assert set(tracked.used) == set(dense.state_dict())
    shared = dense.state_dict()["pair_logits_projection.weight"]
    assert torch.equal(
        stack.blocks[2].attention.to_bias.weight, shared[2 * HEADS : 3 * HEADS]
    )


def test_the_engine_stack_equals_the_dense_stack():
    dense, stack = _dense_and_stack()
    x = _layout()
    with torch.no_grad():
        expected = dense(
            x["act"],
            x["queries_mask"],
            x["gather"],
            x["keys_mask"],
            x["cond"],
            x["keys_cond"],
            x["pair"],
            pair_mask=x["pair_mask"],
            rope_q=x["rope_q"],
            rope_k=x["rope_k"],
        )
        actual = stack(
            x["act"],
            x["queries_mask"],
            x["gather"],
            x["keys_mask"],
            x["cond"],
            x["keys_cond"],
            x["pair"],
            pair_mask=x["pair_mask"],
            rope_q=x["rope_q"],
            rope_k=x["rope_k"],
        )
    real = x["queries_mask"].bool()
    scale = expected[real].abs().max()
    error = (actual[real] - expected[real]).abs().max()
    assert error / scale < 5e-4, f"max relative difference {float(error / scale):.2e}"


def test_the_stack_demands_the_rope_tables():
    _, stack = _dense_and_stack()
    x = _layout()
    with pytest.raises(ValueError, match="rotates q and k"):
        stack(
            x["act"],
            x["queries_mask"],
            x["gather"],
            x["keys_mask"],
            x["cond"],
            x["keys_cond"],
            x["pair"],
            pair_mask=x["pair_mask"],
        )


def test_a_native_file_of_the_dense_layout_loads_into_the_engine_layout():
    dense, stack = _dense_and_stack()

    class Graph(nn.Module):
        def __init__(self, atoms: nn.Module) -> None:
            super().__init__()
            self.atom_transformer_encoder = atoms
            self.other = nn.Linear(4, 4)

    graph = Graph(stack)
    file_state = {
        f"atom_transformer_encoder.{k}": v.clone()
        for k, v in dense.state_dict().items()
    }
    file_state["other.weight"], file_state["other.bias"] = (
        torch.ones(4, 4),
        torch.zeros(4),
    )
    miniworld._adopt_dense_atom_stacks(graph, file_state)
    assert not any(
        ".cross_attention." in k or ".transition_block." in k for k in file_state
    )
    graph.load_state_dict(file_state, strict=True)
    # a file already in the block layout is left alone
    again = dict(graph.state_dict())
    before = {k: v.clone() for k, v in again.items()}
    miniworld._adopt_dense_atom_stacks(graph, again)
    assert all(torch.equal(again[k], before[k]) for k in before)
