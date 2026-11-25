"""CPU regressions of the release converter, from the 2026-11-24 weight install.

* RoseTTAFold3: ``spec.msa_shared_block`` (2026-11-12) made the MSA stack ONE module
  under every ``msa_stack.<i>``; ``_converters`` walked ``named_modules()``, which lists
  a shared module once, so ``msa_stack.1..`` were left to the pass-through and the
  conversion failed (``outer_product_mean.output_w`` is not a parameter of the engine
  graph). Every index is converted now, each must equal the first, and the Haiku
  importer refuses stack slices that differ for one shared parameter.
* Chai-1: a tiny synthetic checkpoint converts to exactly the engine graph's names /
  shapes (the relative encoding at 143 columns), and the chain / entity unfold of the
  pair-init mapping reproduces the release's two-stage map (``W_p @ (P @ x + b)``) on
  pairs across chains and entities.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foldjet.models.architectures.af3 import AlphaFold3
from foldjet.models.checkpoints import miniworld
from foldjet.models.checkpoints.convert import _unfold_chai1_chain_features
from foldjet.models.checkpoints.legacy import dense_model
from foldjet.models.checkpoints.legacy.haiku import Param, assign
from foldjet.modules.dense.spec import spec_for
from foldjet.modules.engine import modules

KWARGS = {"num_recycles": 1, "num_samples": 1, "diffusion_steps": 1}


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 16))
    yield
    torch.set_num_threads(before)


def tiny(family: str):
    """The release's MSA depth (RoseTTAFold3: 4 shared blocks), everything else 1."""
    spec = spec_for(family, "fast")
    return replace(
        spec,
        trunk_layers=1,
        confidence_layers=1,
        diffusion_blocks=4,
        template_layers=1 if spec.template_layers else 0,
    )


def synthetic_release(family: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """(dense released-layout model with random weights, engine graph of the same spec).

    The random weights satisfy what a real release satisfies and the converter demands:
    K / V AdaLN equal to the query AdaLN, RoseTTAFold3's query bias zero.
    """
    modules.OPM_COUNT_OFFSET = 1e-3
    try:
        dense = dense_model(tiny(family), **KWARGS)
        engine = AlphaFold3(spec=tiny(family), **KWARGS)
    finally:
        modules.OPM_COUNT_OFFSET = None
    generator = torch.Generator().manual_seed(7)
    parameters = dict(dense.named_parameters())
    with torch.no_grad():
        for p in parameters.values():
            p.copy_(0.08 * torch.randn(p.shape, generator=generator))
        for name, p in parameters.items():
            if "k_adaptive_layernorm" in name:
                p.copy_(
                    parameters[
                        name.replace("k_adaptive_layernorm", "q_adaptive_layernorm")
                    ]
                )
            if (
                family == "rosettafold3"
                and name.endswith(".q_projection.bias")
                and ("self_attention" in name or "cross_attention" in name)
            ):
                p.zero_()
    fourier = dense.diffusion_head.fourier_embeddings
    fourier.register_buffer("weight", torch.randn(256, generator=generator))
    fourier.register_buffer("bias", torch.randn(256, generator=generator))
    return dense, engine


@pytest.mark.parametrize("family", ["rosettafold3", "chai1"])
def test_tiny_release_converts_to_the_engine_names_and_shapes(family, tmp_path):
    dense, engine = synthetic_release(family)
    state = miniworld.convert_state(dense.state_dict(), engine)
    target = engine.state_dict()
    assert set(state) == set(target) | set(miniworld.LOOSE)
    for key, value in target.items():
        assert tuple(state[key].shape) == tuple(value.shape), key
        assert state[key].dtype == torch.float32, key
    n_msa = tiny(family).msa_layers
    assert {
        int(k.split(".")[2]) for k in state if k.startswith("evoformer.msa_stack.")
    } == set(range(n_msa))
    if family == "chai1":
        assert tuple(state["evoformer.position_activations.weight"].shape)[1] == 143
    # the file round trip the loader reads (strict)
    path = tmp_path / f"{family}.miniworld.pt"
    torch.save(
        {
            "format": miniworld.FORMAT,
            "version": miniworld.VERSION,
            "family": family,
            "source": "synthetic",
            "state_dict": state,
        },
        path,
    )
    modules.OPM_COUNT_OFFSET = 1e-3
    try:
        fresh = AlphaFold3(spec=tiny(family), **KWARGS)
    finally:
        modules.OPM_COUNT_OFFSET = None
    miniworld.load_into(fresh, path, family=family)
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, state[key]), key


def test_rf3_shared_msa_block_converts_every_index_equal():
    assert tiny("rosettafold3").msa_shared_block
    assert tiny("rosettafold3").msa_layers > 1
    dense, engine = synthetic_release("rosettafold3")
    prefixes = [p for p, _, _ in miniworld._converters(engine)]
    n_msa = tiny("rosettafold3").msa_layers
    assert [p for p in prefixes if p.startswith("evoformer.msa_stack.")] == [
        f"evoformer.msa_stack.{i}" for i in range(n_msa)
    ]
    state = miniworld.convert_state(dense.state_dict(), engine)
    first = {
        k[len("evoformer.msa_stack.0.") :]: v
        for k, v in state.items()
        if k.startswith("evoformer.msa_stack.0.")
    }
    assert any(k.startswith("outer_product_mean.") for k in first)
    for i in range(1, n_msa):
        for key, value in first.items():
            assert torch.equal(state[f"evoformer.msa_stack.{i}.{key}"], value), key


def test_rf3_shared_msa_block_refuses_released_blocks_that_differ():
    dense, engine = synthetic_release("rosettafold3")
    state = dict(dense.state_dict())
    key = next(
        k
        for k in state
        if k.startswith("evoformer.msa_stack.1.outer_product_mean.")
        and k.endswith("weight")
    )
    state[key] = state[key] + 1e-3
    with pytest.raises(miniworld.ConversionError, match="shared"):
        miniworld.convert_state(state, engine)


def test_stacked_slices_into_one_shared_parameter_must_be_equal():
    shared = torch.nn.Parameter(torch.zeros(3, 2))
    source = torch.randn(2, 3, 2)
    param = Param(param=[shared, shared], stacked=True)
    with pytest.raises(ValueError, match="share one parameter"):
        assign({"k": param}, {"k": source})
    assign({"k": param}, {"k": torch.stack([source[0], source[0]])})
    assert torch.equal(shared.data, source[0])


# ------------------------------------------------------------ Chai-1 pair-init unfold
#: The release's TOKEN_PAIR stream column ranges (alphabetical concat).
_PAIR_COLS = {
    "DockingConstraintGenerator": (0, 6),
    "RelativeChain": (6, 12),
    "RelativeEntity": (12, 15),
    "RelativeSequenceSeparation": (15, 82),
    "RelativeTokenSeparation": (82, 149),
    "TokenDistanceRestraint": (149, 156),
    "TokenPairPocketRestraint": (156, 163),
}


def _constant_columns() -> list[int]:
    """The columns a fold sets constant (no docking, no restraint sentinel masks)."""
    return [
        _PAIR_COLS["DockingConstraintGenerator"][0] + 5,
        _PAIR_COLS["TokenDistanceRestraint"][1] - 1,
        _PAIR_COLS["TokenPairPocketRestraint"][1] - 1,
    ]


def _reference_map_pair_init(sds: dict, params: dict) -> None:
    """The reference converter's folding (chain and entity at single-chain values)."""
    fe, te = sds["feature_embedding"], sds["token_embedder"]
    wp = te["token_pair_proj_in_trunk.weight"]
    half = te["token_single_to_token_pair_outer_sum_proj.weight"].shape[0] // 2
    p = fe["input_projs.TOKEN_PAIR.0.weight"][:half]
    b = fe["input_projs.TOKEN_PAIR.0.bias"][:half]
    rel = np.concatenate(
        [
            p[:, slice(*_PAIR_COLS["RelativeSequenceSeparation"])],
            p[:, slice(*_PAIR_COLS["RelativeTokenSeparation"])],
        ],
        axis=1,
    )
    const = np.zeros(p.shape[1])
    const[_constant_columns()] = 1.0
    const[_PAIR_COLS["RelativeChain"][0] + 2] = 1.0
    const[_PAIR_COLS["RelativeEntity"][0] + 1] = 1.0
    params["diffuser/evoformer/~_relative_encoding/position_activations"] = {
        "weights": (wp @ rel).T,
        "bias": wp @ (p @ const + b),
    }


def test_chai1_unfold_reproduces_the_release_pair_map_across_chains():
    rng = np.random.default_rng(0)
    c_z, width = 8, 163
    sds = {
        "feature_embedding": {
            "input_projs.TOKEN_PAIR.0.weight": rng.standard_normal((2 * c_z, width)),
            "input_projs.TOKEN_PAIR.0.bias": rng.standard_normal(2 * c_z),
        },
        "token_embedder": {
            "token_pair_proj_in_trunk.weight": rng.standard_normal((c_z, c_z)),
            "token_single_to_token_pair_outer_sum_proj.weight": rng.standard_normal(
                (2 * c_z, 5)
            ),
        },
    }
    chai1 = SimpleNamespace(
        PAIR_COLS=_PAIR_COLS, map_pair_init=_reference_map_pair_init
    )
    _unfold_chai1_chain_features(chai1)
    params: dict = {}
    chai1.map_pair_init(sds, params)
    scope = params["diffuser/evoformer/~_relative_encoding/position_activations"]
    weights, bias = scope["weights"], scope["bias"]
    assert weights.shape == (67 + 67 + 6 + 3, c_z)

    wp = sds["token_embedder"]["token_pair_proj_in_trunk.weight"]
    p = sds["feature_embedding"]["input_projs.TOKEN_PAIR.0.weight"][:c_z]
    b = sds["feature_embedding"]["input_projs.TOKEN_PAIR.0.bias"][:c_z]
    for seq, tok, chain, entity in [
        (33, 32, 2, 1),
        (66, 66, 5, 0),
        (66, 66, 3, 1),
        (10, 66, 5, 2),
    ]:
        full = np.zeros(width)
        full[_constant_columns()] = 1.0
        full[_PAIR_COLS["RelativeSequenceSeparation"][0] + seq] = 1.0
        full[_PAIR_COLS["RelativeTokenSeparation"][0] + tok] = 1.0
        full[_PAIR_COLS["RelativeChain"][0] + chain] = 1.0
        full[_PAIR_COLS["RelativeEntity"][0] + entity] = 1.0
        release = wp @ (p @ full + b)
        # featurization.chai_relative_encoding's column order
        x = np.zeros(143)
        x[seq] = x[67 + tok] = x[134 + chain] = x[140 + entity] = 1.0
        np.testing.assert_allclose(x @ weights + bias, release, rtol=1e-12, atol=1e-12)
