"""``--recipe protenix`` (``train/recipes.py``): resolved values, the schedule / optimizer / clip it implies, the experimentally-resolved loss, the zero-weight
freeze of the PAE head. CPU, tiny tensors. Protenix's own functions are compared in ``test_protenix_parity.py``."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from foldforge.train import freeze, optim, recipes
from foldforge.train.config import LossWeights, TrainConfig
from foldforge.train.loss import confidence as conf


class Stand(nn.Module):
    """A module tree with the given dotted parameter names (each a 2x2 tensor of ``fill``) and the protenix1 spec."""

    def __init__(self, family: str, names: dict[str, float]) -> None:
        super().__init__()
        from foldforge.modules.dense.spec import SPECS

        self.spec = SPECS[family]
        for name, fill in names.items():
            *path, leaf = name.split(".")
            node = self
            for part in path:
                if not hasattr(node, part):
                    setattr(node, part, nn.Module())
                node = getattr(node, part)
            setattr(node, leaf, nn.Parameter(torch.full((2, 2), fill)))


def test_miniworld_is_the_default_and_leaves_the_config_alone():
    cfg = TrainConfig()
    assert recipes.apply("miniworld", cfg) is cfg
    assert (
        cfg.grad_clip_max_norm,
        cfg.weight_decay,
        cfg.decay_from_warmup,
        cfg.recipe,
    ) == (1.0, 0.0, True, "miniworld")
    assert cfg.loss.resolved_loss == 0.0 and not cfg.targets_protenix
    with pytest.raises(ValueError):
        recipes.apply("nope", cfg)


def test_the_protenix_recipe_sets_every_documented_value():
    cfg = recipes.apply("protenix", TrainConfig())
    assert cfg.recipe == "protenix"
    # optimizer (configs_base.py:80,83-87; finetune_demo.sh)
    assert (
        cfg.grad_clip_max_norm == 10.0
        and cfg.betas == (0.9, 0.95)
        and cfg.weight_decay == 1e-8
    )
    assert (
        cfg.max_lr,
        cfg.warmup_steps,
        cfg.decay_steps,
        cfg.decay_factor,
        cfg.min_lr,
    ) == (1e-3, 2000, 50_000, 0.95, 0.0)
    assert cfg.decay_from_warmup is False and cfg.ema_decay == 0.999 and cfg.use_ema
    assert (cfg.recycles_min, cfg.recycles_max, cfg.rollout_steps, cfg.num_augment) == (
        1,
        4,
        20,
        48,
    )
    # loss (configs_base.py:401-407, docs table "fine-tuning stage 1"): bond = alpha_diffusion x alpha_bond, confidence = 1e-4 x 1.0, PAE = 1e-4 x alpha_pae 1.0 (stage 3 value, switched on)
    w = cfg.loss
    assert (w.diffusion_loss, w.alpha_dna, w.alpha_rna, w.alpha_ligand) == (
        4.0,
        5.0,
        5.0,
        10.0,
    )
    assert (
        w.bond_loss == 4.0 * 1.0
        and w.smooth_lddt_loss == 0.0
        and w.distogram_loss == 0.03
    )
    assert (w.plddt_loss, w.pde_loss, w.resolved_loss, w.pae_loss) == (
        1e-4,
        1e-4,
        1e-4,
        1e-4,
    )
    assert cfg.targets_protenix
    # the fields the recipe does not name stay MiniWorld's
    assert (cfg.noise, cfg.grad_accum_steps, cfg.rollout, cfg.confidence) == (
        TrainConfig().noise,
        32,
        "after",
        True,
    )
    # every documented reference names a source
    assert all(
        value and source for value, source in recipes.PROTENIX_REFERENCES.values()
    )


def test_the_af3_schedule_decays_from_step_zero_and_miniworld_from_the_end_of_the_warmup():
    protenix = recipes.apply("protenix", TrainConfig())
    lr = lambda step: protenix.max_lr * optim.lr_factor(step, protenix)
    assert (
        lr(0) == 0.0
        and lr(1000) == pytest.approx(5e-4)
        and lr(2000) == pytest.approx(1e-3)
    )
    assert (
        lr(49_999) == pytest.approx(1e-3)
        and lr(50_000) == pytest.approx(0.95e-3)
        and lr(100_000) == pytest.approx(0.95**2 * 1e-3)
    )
    mini = TrainConfig(
        warmup_steps=2000
    )  # MiniWorld counts from the end of the warm-up
    assert optim.lr_factor(50_000, mini) == 1.0 and optim.lr_factor(
        52_000, mini
    ) == pytest.approx(0.95)
    # no floor in the protenix recipe: set_lr follows the formula all the way down
    model = nn.Linear(2, 2)
    model.requires_grad_(True)
    opt = optim.build_optimizer(model, protenix)
    assert optim.set_lr(opt, 4_000_000, protenix) == pytest.approx(1e-3 * 0.95**80)


def test_the_adam_is_torch_adam_with_coupled_weight_decay_and_clip_ten():
    torch.manual_seed(0)
    cfg = replace(recipes.apply("protenix", TrainConfig()), warmup_steps=1)
    a, b = nn.Linear(3, 3), nn.Linear(3, 3)
    b.load_state_dict(a.state_dict())
    opt = optim.build_optimizer(a, cfg)
    assert {g["weight_decay"] for g in opt.param_groups} == {1e-8} and type(
        opt
    ) is torch.optim.Adam
    ref = torch.optim.Adam(
        b.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-8
    )
    optim.set_lr(opt, 1, cfg)  # lr(1) = max_lr
    for _ in range(3):
        grads = [
            torch.randn_like(p) * 30.0 for p in a.parameters()
        ]  # norm well above 10: the clip acts
        for p, q, g in zip(a.parameters(), b.parameters(), grads, strict=True):
            p.grad, q.grad = g.clone(), g.clone()
        norm = optim.clip_grad_norm(list(a.parameters()), cfg.grad_clip_max_norm)
        ref_norm = torch.nn.utils.clip_grad_norm_(b.parameters(), 10.0)
        assert (
            float(norm) == pytest.approx(float(ref_norm), rel=1e-6)
            and float(norm) > 10.0
        )
        opt.step()
        ref.step()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(p, q, rtol=1e-6, atol=1e-9)
    # below the threshold the gradient is untouched
    p = nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), 1.0)  # norm 2 < 10
    optim.clip_grad_norm([p], 10.0)
    assert torch.equal(p.grad, torch.ones(4))


def test_the_resolved_loss_is_the_mean_two_class_cross_entropy_over_the_real_atoms():
    ln3 = math.log(3.0)
    logits = torch.zeros(2, 3, 2)
    logits[0, 0] = torch.tensor([0.0, 0.0])  # class 1 -> ln 2
    logits[0, 1] = torch.tensor([0.0, ln3])  # class 1 -> -ln(3/4)
    logits[1, 0] = torch.tensor([0.0, ln3])  # class 0 -> -ln(1/4)
    logits[1, 1] = torch.tensor([2.0, 0.0])  # class 0 -> ln(1 + e^-2)
    logits[:, 2] = torch.tensor([50.0, -50.0])  # padding slots: huge, masked out
    resolved = torch.tensor(
        [[1, 1, 1], [0, 0, 1]]
    )  # the padding slots are class 1 against logits that say class 0: a loss of 100 if they counted
    real = torch.tensor([[True, True, False], [True, True, False]])
    expected = (
        math.log(2.0) - math.log(0.75) - math.log(0.25) + math.log(1.0 + math.exp(-2.0))
    ) / 4.0
    assert float(conf.resolved_loss_value(logits, resolved, real)) == pytest.approx(
        expected, rel=1e-6
    )
    # an unresolved REAL atom (class 0) counts; a padding slot does not
    all_real = torch.ones(2, 3, dtype=torch.bool)
    assert float(conf.resolved_loss_value(logits, resolved, all_real)) > expected


def test_the_targets_and_the_loss_carry_the_resolved_term_and_skip_a_zero_weight_pae():
    res, atoms = 3, 2
    head = nn.Module()
    head.num_plddt_bins = 50
    head.register_buffer("distance_breaks", torch.linspace(0.0, 31.0, 63))
    head.register_buffer("pae_breaks", torch.linspace(0.0, 31.0, 63))
    from foldforge.utils.atom_layout import GatherInfo

    gather = GatherInfo(
        gather_idxs=torch.arange(res) * atoms,
        gather_mask=torch.ones(res, dtype=torch.bool),
        input_shape=(res, atoms),
    )
    gt = torch.randn(res, atoms, 3) * 4
    resolved = torch.tensor([[True, True], [True, False], [False, False]])
    real = torch.tensor([[True, True], [True, True], [True, False]])
    targets = conf.confidence_targets(
        gt + 0.1,
        gt,
        resolved,
        gather,
        torch.ones(res),
        torch.zeros(res, dtype=torch.long),
        torch.zeros(res, dtype=torch.bool),
        torch.zeros(res, dtype=torch.bool),
        head,
        real_mask=real,
        with_pae=False,
    )
    assert targets["pae"] is None
    cls, mask = targets["resolved"]
    assert cls.tolist() == resolved.long().tolist() and torch.equal(mask, real)
    outputs = {
        "plddt_logits": torch.zeros(res, atoms, 50),
        "pde_logits": torch.zeros(res, res, 64),
        "pae_logits": torch.zeros(res, res, 64),
        "resolved_logits": torch.zeros(res, atoms, 2),
    }
    total, parts = conf.confidence_loss(
        outputs,
        targets,
        plddt_loss=1e-4,
        pde_loss=1e-4,
        pae_loss=0.0,
        resolved_loss=1e-4,
    )
    assert float(parts["pae"]) == 0.0 and float(parts["resolved"]) == pytest.approx(
        math.log(2.0)
    )
    assert float(total) == pytest.approx(
        1e-4 * (math.log(50) + math.log(64) + math.log(2.0)), rel=1e-6
    )
    # a MiniWorld-style call (no resolved term) has no "resolved" part at all
    _, parts = conf.confidence_loss(
        outputs, targets, plddt_loss=1.0, pde_loss=1.0, pae_loss=0.0
    )
    assert set(parts) == {"plddt", "pde", "pae"}


CONFIDENCE_HEAD = {
    "confidence_head.experimentally_resolved_ln.weight": 1.0,
    "confidence_head.experimentally_resolved_logits.weight": 0.5,
    "confidence_head.pae_logits_ln.weight": 1.0,
    "confidence_head.pae_logits.weight": 0.5,
    "confidence_head.pae_inter_logits.weight": 0.5,
    "confidence_head.plddt_logits.weight": 0.5,
    "confidence_head.left_half_distance_logits.weight": 0.5,
}


def test_the_frozen_set_follows_the_loss_weights_resolved_and_pae():
    model = Stand("protenix1", CONFIDENCE_HEAD)
    miniworld = {
        n: r.key for n, r in freeze.frozen_parameters(model, TrainConfig().loss).items()
    }
    assert (
        set(miniworld.values()) == {"experimentally_resolved"} and len(miniworld) == 2
    )
    assert freeze.frozen_parameters(model) == freeze.frozen_parameters(
        model, LossWeights()
    ), "no weights = the MiniWorld defaults"
    protenix = {
        n: r.key
        for n, r in freeze.frozen_parameters(
            model, recipes.apply("protenix", TrainConfig()).loss
        ).items()
    }
    assert protenix == {}, (
        "the protenix recipe trains the PAE head (pae_loss 1e-4) and the resolved head: nothing frozen"
    )
    # a recipe with the PAE weight 0 freezes the PAE head
    no_pae = replace(recipes.apply("protenix", TrainConfig()).loss, pae_loss=0.0)
    assert {r.key for r in freeze.frozen_parameters(model, no_pae).values()} == {
        "pae_zero_weight"
    }
    assert set(freeze.frozen_parameters(model, no_pae)) == {
        "confidence_head.pae_logits_ln.weight",
        "confidence_head.pae_logits.weight",
        "confidence_head.pae_inter_logits.weight",
    }
    # a recipe that trains both heads freezes neither
    both = replace(LossWeights(), resolved_loss=1.0, pae_loss=1.0)
    assert freeze.frozen_parameters(model, both) == {}
    freeze.apply(model, recipes.apply("protenix", TrainConfig()).loss)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert (
        "confidence_head.experimentally_resolved_logits.weight" in trainable
        and "confidence_head.pae_logits.weight" in trainable
    )
