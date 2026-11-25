"""``train/freeze.py``: which parameters training does not update.

Also the gradient classification of a step, and RoseTTAFold3's shared MSA block.

The name tables use the real parameter names of the graph on stand-in modules (CPU,
tiny tensors); the RoseTTAFold3 checks build the real graph.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from foldjet.models.architectures.af3 import AlphaFold3
from foldjet.modules.dense.spec import SPECS, spec_for
from foldjet.train import freeze, optim, prepare
from foldjet.train.config import TrainConfig


class Stand(nn.Module):
    """A module tree with the given dotted parameter names and the family's spec.

    Each parameter is a 2x2 tensor of ``fill``.
    """

    def __init__(self, family: str, names: dict[str, float]) -> None:
        super().__init__()
        self.spec = SPECS[family]
        for name, fill in names.items():
            *path, leaf = name.split(".")
            node = self
            for part in path:
                if not hasattr(node, part):
                    setattr(node, part, nn.Module())
                node = getattr(node, part)
            setattr(node, leaf, nn.Parameter(torch.full((2, 2), fill)))


def frozen(family: str, names: dict[str, float]) -> set[str]:
    return set(freeze.frozen_parameters(Stand(family, names)))


ATOM_PAIR = {
    f"{where}.{leaf}.weight": 0.5
    for where in ("evoformer_conditioning", "diffusion_head.atom_cross_att_encoder")
    for leaf in (
        "single_to_pair_cond_row",
        "single_to_pair_cond_col",
        "embed_pair_offsets",
        "embed_pair_distances",
        "single_to_pair_cond_row_1",
        "single_to_pair_cond_col_1",
        "embed_pair_offsets_1",
        "embed_pair_distances_1",
        "embed_ref_charge",
    )
}


@pytest.mark.parametrize(
    "family",
    ["protenix1", "protenix2", "opendde", "boltz2", "rosettafold3", "alphafold3"],
)
def test_the_atom_pair_slots_without_a_twin_are_frozen_and_the_twins_train(family):
    got = frozen(family, ATOM_PAIR)
    assert len(got) == 8  # 4 slots x (input embedder, diffusion head)
    assert all(not name.rsplit(".", 2)[-2].endswith(("_1", "charge")) for name in got)
    assert "evoformer_conditioning.embed_pair_distances_1.weight" not in got
    assert "diffusion_head.atom_cross_att_encoder.embed_ref_charge.weight" not in got


def test_the_last_msa_block_updates_nothing_the_loss_reads_in_the_opm_first_order():
    names = {
        f"evoformer.msa_stack.{i}.{part}": 0.5
        for i in range(4)
        for part in (
            "msa_attention1.to_value.weight",
            "msa_transition.squeeze.weight",
            "outer_product_mean.to_left.weight",
            "pair_transition.squeeze.weight",
        )
    }
    got = frozen("protenix1", names)
    assert got == {
        "evoformer.msa_stack.3.msa_attention1.to_value.weight",
        "evoformer.msa_stack.3.msa_transition.squeeze.weight",
    }, (
        "only the MSA-side parameters of the LAST block; its pair side (OPM, triangle, "
        "transition) trains"
    )
    assert not frozen("boltz2", names), (
        "Boltz order (MSA update before the OPM): the last block's MSA update reaches "
        "the pair"
    )


def test_the_experimentally_resolved_head_is_frozen_because_the_recipe_has_no_loss_for_it():  # noqa: E501 - test node id kept (pass tokens)
    names = {
        "confidence_head.experimentally_resolved_ln.weight": 1.0,
        "confidence_head.experimentally_resolved_ln.bias": 0.0,
        "confidence_head.experimentally_resolved_logits.weight": 0.5,
        "confidence_head.plddt_logits.weight": 0.5,
    }
    assert frozen("protenix1", names) == {
        n for n in names if "experimentally_resolved" in n
    }


def test_boltz2_contact_fourier_is_frozen_and_no_other_family_loses_it():
    names = {
        "evoformer.contact_conditioning.contact_fourier.weight": 0.5,
        "evoformer.contact_conditioning.contact_fourier.bias": 0.5,
        "confidence_head.reembedding.contact_conditioning.contact_fourier.weight": 0.5,
        "evoformer.contact_conditioning.contact_encoder.weight": 0.5,
    }
    assert frozen("boltz2", names) == {n for n in names if "contact_fourier" in n}
    assert not frozen("rosettafold3", names)


def test_esmfold2_slots_the_original_lacks_are_frozen_only_while_they_are_zero():
    zero = {
        "evoformer_conditioning.pair_mlp_1.weight": 0.0,
        "diffusion_head.atom_cross_att_encoder.embed_trunk_pair_cond.weight": 0.0,
        "diffusion_head.atom_cross_att_decoder.atom_transformer_decoder."
        "pair_logits_projection.weight": 0.0,
        "confidence_head.confidence_pairformer.2.single_transition.squeeze.weight": 0.0,
        "confidence_head.reembedding.contact_conditioning.contact_fourier.weight": 0.0,
        "evoformer.single_activations.weight": 0.0,
    }
    assert frozen("esmfold2", zero) == set(zero)
    assert frozen("esmfold2-fast", zero) == set(zero)
    assert not frozen("esmfold2", dict.fromkeys(zero, 0.25)), (
        "a slot a later checkpoint fills is a real parameter: not frozen"
    )
    assert not frozen("protenix1", zero), (
        "other families keep these (they are real there)"
    )


def test_apply_sets_requires_grad_and_the_optimizer_skips_the_frozen_parameters():
    model = Stand(
        "boltz2",
        {
            **ATOM_PAIR,
            "diffusion_head.x.weight": 0.5,
            "evoformer.contact_conditioning.contact_fourier.weight": 0.5,
        },
    )
    model.confidence_head = nn.Module()
    model.confidence_head.save_logits = False
    model.distogram_head = nn.Module()
    info = prepare.enable_training(model, TrainConfig(confidence=False))
    assert info["frozen"] == {"atom_pair_unused": 8, "boltz2_contact_fourier": 1}
    names = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert len(names) == 9
    trunk, heads = optim.split_parameters(model)
    in_optimizer = {n for n, _ in trunk} | {n for n, _ in heads}
    assert not in_optimizer & names
    assert in_optimizer == {n for n, _ in model.named_parameters()} - names


def test_classify_gradient_separates_the_defects_from_the_by_design_zeros():
    def param(value: float, grad: float | None) -> nn.Parameter:
        p = nn.Parameter(torch.full((2, 2), value))
        p.grad = None if grad is None else torch.full((2, 2), grad)
        return p

    scale = 0.1
    assert freeze.classify_gradient("a.weight", param(0.5, None), scale) == "none"
    assert (
        freeze.classify_gradient("a.weight", param(0.5, float("nan")), scale)
        == "nonfinite"
    )
    assert freeze.classify_gradient("a.weight", param(0.5, 1e-9), scale) == "ok"
    assert freeze.classify_gradient("a.weight", param(0.5, 0.0), scale) == "zero", (
        "a zero gradient on a live weight is a defect"
    )
    assert freeze.classify_gradient("a.weight", param(5e-37, 0.0), scale) == "underflow"
    assert (
        freeze.classify_gradient("a.weight", param(7e-12, 0.0), scale) == "underflow"
    ), "relative: 7e-12 is far below the model's weights (the confidence pairformer)"
    assert (
        freeze.classify_gradient(
            "evoformer.bond_embedding.weight", param(6.8, 0.0), scale
        )
        == "data_dependent"
    )
    assert (
        freeze.classify_gradient(
            "evoformer_conditioning.embed_ref_charge.weight", param(1.5, 0.0), scale
        )
        == "data_dependent"
    )
    assert freeze.classify_gradient("a.weight", param(1e-4, 0.0), scale) == "zero", (
        "1e-4 is not small relative to a 0.1 model scale"
    )


# --- RoseTTAFold3: one MSA module run four times ---


def small_rf3(*, shared: bool):
    spec = replace(
        spec_for("rosettafold3", "fast"),
        trunk_layers=1,
        msa_layers=4,
        confidence_layers=1,
        diffusion_blocks=4,
        template_layers=1,
        msa_shared_block=shared,
    )
    return AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=spec)


def randomise(model: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=generator)
            p.copy_(
                0.08 * noise
                if p.ndim > 1
                else (1.0 + 0.1 * noise if name.endswith("weight") else 0.05 * noise)
            )


def test_rosettafold3_declares_the_shared_msa_block_and_no_other_family_does():
    assert SPECS["rosettafold3"].msa_shared_block
    assert [f for f, s in SPECS.items() if s.msa_shared_block] == ["rosettafold3"]


def test_the_shared_msa_block_is_one_parameter_set_with_the_unchanged_checkpoint_keys():
    shared, copies = small_rf3(shared=True), small_rf3(shared=False)
    stack = shared.evoformer.msa_stack
    assert len(stack) == 4
    assert all(block is stack[0] for block in stack)
    assert len(set(map(id, copies.evoformer.msa_stack))) == 4
    assert set(shared.state_dict()) == set(copies.state_dict()), (
        "same checkpoint keys: msa_stack.0 .. msa_stack.3"
    )
    names = [
        n for n, _ in shared.named_parameters() if n.startswith("evoformer.msa_stack.")
    ]
    assert names, "the optimizer sees the block once"
    assert all(n.startswith("evoformer.msa_stack.0.") for n in names), (
        "the optimizer sees the block once"
    )
    assert sum(p.numel() for p in shared.parameters()) < sum(
        p.numel() for p in copies.parameters()
    )


def test_the_shared_block_gives_the_same_msa_stack_output_as_four_equal_copies():
    copies = small_rf3(shared=False)
    randomise(copies, 3)
    state = copies.state_dict()
    for key in list(state):  # the release's layout: block 0's weights under every index
        if key.startswith("evoformer.msa_stack.") and not key.startswith(
            "evoformer.msa_stack.0."
        ):
            state[key] = state[f"evoformer.msa_stack.0.{key.split('.', 3)[3]}"].clone()
    copies.load_state_dict(state, strict=True)
    shared = small_rf3(shared=True)
    shared.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(4)
    size, rows = 6, 3
    c_msa, c_pair = shared.evoformer.msa_channel, shared.evoformer.pair_channel
    msa0 = torch.randn(rows, size, c_msa, generator=generator)
    pair0 = torch.randn(size, size, c_pair, generator=generator)
    msa_mask, pair_mask = torch.ones(rows, size), torch.ones(size, size)
    outputs = []
    for model in (copies, shared):
        model.eval()
        msa, pair = msa0.clone(), pair0.clone()
        with torch.no_grad():
            for block in model.evoformer.msa_stack:
                msa, pair = block(
                    msa=msa, pair=pair, msa_mask=msa_mask, pair_mask=pair_mask
                )
        outputs.append((msa, pair))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


def test_training_a_shared_block_gets_the_gradient_of_all_its_uses_in_one_parameter():
    shared = small_rf3(shared=True)
    randomise(shared, 5)
    shared.train()
    generator = torch.Generator().manual_seed(6)
    size, rows = 6, 3
    c_msa, c_pair = shared.evoformer.msa_channel, shared.evoformer.pair_channel
    msa = torch.randn(rows, size, c_msa, generator=generator)
    pair = torch.randn(size, size, c_pair, generator=generator, requires_grad=True)
    msa_mask, pair_mask = torch.ones(rows, size), torch.ones(size, size)
    for block in shared.evoformer.msa_stack:
        msa, pair = block(msa=msa, pair=pair, msa_mask=msa_mask, pair_mask=pair_mask)
    pair.square().mean().backward()
    weight = shared.evoformer.msa_stack[0].msa_transition.expand_a.weight
    assert weight.grad is not None
    assert all(
        n.startswith("evoformer.msa_stack.0.")
        for n, p in shared.named_parameters()
        if p is weight
    )
