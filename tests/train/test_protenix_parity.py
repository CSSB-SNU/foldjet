"""FoldJet's Protenix-recipe pieces against Protenix's OWN functions (CPU).

Runs on a tiny structure. Skipped unless ``FOLDFORGE_PROTENIX_PY`` (the Protenix
environment's python, ``~/foldforge-base/envs/protenix/.venv/bin/python``) and
``FOLDFORGE_PROTENIX_SRC`` (``~/foldforge-base/protenix``) are set:
``protenix_reference.py`` runs in a subprocess of that environment and returns
Protenix's values (resolved loss, pLDDT / PDE target bins, af3 lr scheduler,
configs_base settings); the FoldJet side runs in this process. Run it in a Slurm
CPU job.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from foldjet.data.constants import atom_types, residue_names
from foldjet.train import optim, recipes
from foldjet.train.config import TrainConfig
from foldjet.train.loss import confidence as conf
from foldjet.utils.atom_layout import GatherInfo

PY = os.environ.get("FOLDFORGE_PROTENIX_PY")
SRC = os.environ.get("FOLDFORGE_PROTENIX_SRC")
pytestmark = pytest.mark.skipif(
    not (PY and SRC),
    reason=(
        "needs FOLDFORGE_PROTENIX_PY and FOLDFORGE_PROTENIX_SRC (Protenix environment)"
    ),
)

CODES = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP
ATOMS = atom_types.DENSE_ATOM_NUM
LIGAND = len(CODES) + 1  # an aatype outside the polymer alphabet
#: (aatype name or None = ligand, atom names present in the crop, atom names that
#: are NOT resolved, pseudo-beta atom, plddt neighbour atom)
TOKENS = [
    (residue_names.ALA, ["N", "CA", "C", "O", "CB"], [], "CB", "CA"),
    (residue_names.ALA, ["N", "CA", "C", "O", "CB"], [], "CB", "CA"),
    (residue_names.ALA, ["N", "CA", "C", "O", "CB"], ["CB", "O"], "CB", "CA"),
    (residue_names.GLY, ["N", "CA", "C", "O"], [], "CA", "CA"),
    (
        residue_names.A,
        ["C1'", "C2'", "C3'", "C4'", "O4'", "C4", "N9", "N1"],
        ["N1"],
        "C4",
        "C1'",
    ),
    (
        residue_names.ALA,
        ["N", "CA", "C", "O", "CB"],
        ["N", "CA", "C", "O", "CB"],
        "CB",
        "CA",
    ),  # a token that is not resolved at all
    (None, ["lig"], [], "lig", "lig"),
    (None, ["lig"], [], "lig", "lig"),
]
RES = len(TOKENS)


def _system(seed: int = 0) -> dict[str, Any]:
    """Dense tensors plus the flat per-atom view Protenix takes.

    The flat view holds every atom of the crop: resolved or not.
    """
    g = torch.Generator().manual_seed(seed)
    real = torch.zeros(RES, ATOMS, dtype=torch.bool)
    resolved = torch.zeros(RES, ATOMS, dtype=torch.bool)
    beta_slot, m_slot = [], []
    aatype = torch.zeros(RES, dtype=torch.long)
    is_nuc, is_lig = (
        torch.zeros(RES, dtype=torch.bool),
        torch.zeros(RES, dtype=torch.bool),
    )
    for t, (name, present, unresolved, beta, nbr) in enumerate(TOKENS):
        if name is None:
            aatype[t], is_lig[t] = LIGAND, True
            order = {"lig": 0}
        else:
            aatype[t] = CODES.index(name)
            order = atom_types.DENSE_ATOM_ORDER[name]
            is_nuc[t] = name == residue_names.A
        for atom in present:
            real[t, order[atom]] = True
            resolved[t, order[atom]] = atom not in unresolved
        beta_slot.append(order[beta])
        m_slot.append(order[nbr])
    gt = torch.randn(RES, ATOMS, 3, generator=g) * 6.0
    pred = gt + torch.randn(RES, ATOMS, 3, generator=g) * 1.5
    # unresolved atoms carry no coordinates
    gt = torch.where(resolved[..., None], gt, torch.zeros_like(gt))
    logits = torch.randn(RES, ATOMS, 2, generator=g) * 2.0
    flat = real.reshape(-1).nonzero(as_tuple=True)[0]
    token = flat // ATOMS
    slot = flat % ATOMS
    is_m = (slot == torch.tensor(m_slot)[token]) & ~is_lig[token]
    is_beta = slot == torch.tensor(beta_slot)[token]
    flat_in = {
        "pred": pred.reshape(-1, 3)[flat].numpy(),
        "true": gt.reshape(-1, 3)[flat].numpy(),
        "coordinate_mask": resolved.reshape(-1)[flat].numpy().astype(np.int64),
        "is_nuc": is_nuc[token].numpy().astype(np.int64),
        "is_polymer": (~is_lig[token]).numpy().astype(np.int64),
        "plddt_m_rep": is_m.numpy().astype(np.int64),
        "distogram_rep": is_beta.numpy().astype(np.int64),
        "resolved_logits": logits.reshape(-1, 2)[flat].numpy(),
    }
    gather = GatherInfo(
        gather_idxs=torch.arange(RES) * ATOMS + torch.tensor(beta_slot),
        gather_mask=torch.ones(RES, dtype=torch.bool),
        input_shape=(RES, ATOMS),
    )
    return {
        "real": real,
        "resolved": resolved,
        "pred": pred,
        "gt": gt,
        "aatype": aatype,
        "is_nuc": is_nuc,
        "is_lig": is_lig,
        "logits": logits,
        "gather": gather,
        "flat_index": flat,
        "flat_in": flat_in,
    }


class _Head(nn.Module):
    num_plddt_bins = 50

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("distance_breaks", torch.linspace(0.0, 31.0, 63))
        self.register_buffer("pae_breaks", torch.linspace(0.0, 31.0, 63))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    work = tmp_path_factory.mktemp("protenix_ref")
    system = _system()
    np.savez(work / "in.npz", **system["flat_in"])
    script = Path(__file__).with_name("protenix_reference.py")
    env = {**os.environ, "PROTENIX_SRC": SRC}
    done = subprocess.run(  # noqa: S603 - trusted: the configured Protenix python
        [PY, str(script), str(work / "in.npz"), str(work / "out.npz")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return system, dict(np.load(work / "out.npz"))


def test_the_experimentally_resolved_loss_equals_protenixs(reference):
    system, ref = reference
    ours = conf.resolved_loss_value(
        system["logits"], system["resolved"].long(), system["real"]
    )
    assert float(ours) == pytest.approx(float(ref["resolved"]), rel=1e-5)
    assert system["real"].sum() > system["resolved"].sum(), (
        "the structure has real but unresolved atoms: they are classes in the loss"
    )


def test_the_plddt_targets_equal_protenixs_atom_by_atom(reference):
    system, ref = reference
    targets = conf.confidence_targets(
        system["pred"],
        system["gt"],
        system["resolved"],
        system["gather"],
        torch.ones(RES),
        system["aatype"],
        system["is_nuc"],
        system["is_lig"],
        _Head(),
        real_mask=system["real"],
        protenix_targets=True,
    )
    bins, valid = targets["plddt"]
    flat = system["flat_index"]
    resolved_flat = system["resolved"].reshape(-1)[flat]
    ours = bins.reshape(-1)[flat][resolved_flat].numpy()
    assert np.array_equal(
        valid.reshape(-1)[flat][resolved_flat].numpy(), np.ones(len(ours), bool)
    ), "Protenix keeps every resolved atom, with or without neighbours"
    assert len(ours) == len(ref["plddt_bins"])
    # identical bins; a ``<`` vs float noise tie at a bin edge may move one atom by
    # one bin
    differ = ours != ref["plddt_bins"]
    assert differ.sum() <= max(1, len(ours) // 50), (ours, ref["plddt_bins"])
    assert np.abs(ours - ref["plddt_bins"]).max() <= 1
    # the default (MiniWorld) pLDDT neighbour set is NOT Protenix's: the two must
    # differ on this system
    mini = (
        conf.confidence_targets(
            system["pred"],
            system["gt"],
            system["resolved"],
            system["gather"],
            torch.ones(RES),
            system["aatype"],
            system["is_nuc"],
            system["is_lig"],
            _Head(),
        )["plddt"][0]
        .reshape(-1)[flat][resolved_flat]
        .numpy()
    )
    assert (mini != ref["plddt_bins"]).any()


def test_the_pde_targets_equal_protenixs(reference):
    system, ref = reference
    targets = conf.confidence_targets(
        system["pred"],
        system["gt"],
        system["resolved"],
        system["gather"],
        torch.ones(RES),
        system["aatype"],
        system["is_nuc"],
        system["is_lig"],
        _Head(),
        protenix_targets=True,
    )
    bins, mask = targets["pde"]
    beta = system["gather"].gather_idxs
    resolved_beta = system["resolved"].reshape(-1)[beta]
    assert np.array_equal(mask.numpy(), ref["pde_mask"].astype(bool)), (
        "same token pairs enter the loss"
    )
    assert np.array_equal(
        bins.numpy()[ref["pde_mask"].astype(bool)],
        ref["pde_bins"][ref["pde_mask"].astype(bool)],
    )
    assert int(resolved_beta.sum()) < RES, (
        "a token whose representative atom is unresolved drops out"
    )
    # the MiniWorld bucketisation is one bin higher everywhere (bin 0 = exact zero)
    mini = conf.confidence_targets(
        system["pred"],
        system["gt"],
        system["resolved"],
        system["gather"],
        torch.ones(RES),
        system["aatype"],
        system["is_nuc"],
        system["is_lig"],
        _Head(),
    )["pde"][0]
    sel = ref["pde_mask"].astype(bool)
    assert (mini.numpy()[sel] - ref["pde_bins"][sel] >= 0).all()
    assert (mini.numpy()[sel] != ref["pde_bins"][sel]).any()


def test_the_af3_learning_rate_equals_protenixs_scheduler_at_every_checked_step(
    reference,
):
    _, ref = reference
    cfg = recipes.apply("protenix", TrainConfig())
    ours = np.array(
        [cfg.max_lr * optim.lr_factor(int(s), cfg) for s in ref["lr_steps"]]
    )
    assert np.allclose(ours, ref["lr_values"], rtol=1e-12, atol=0.0), (
        ours,
        ref["lr_values"],
    )


def test_the_recipe_values_equal_the_defaults_and_stage_one_overrides_of_configs_base(
    reference,
):
    _, ref = reference
    cfg = recipes.apply("protenix", TrainConfig())
    alpha_conf, alpha_pae, alpha_except, alpha_diff, alpha_disto, alpha_bond, smooth = (
        ref["weights"]
    )
    # configs_base.py defaults are the INITIAL-training values (alpha_bond 0,
    # smooth_lddt 1); the fine-tuning stage 1 of the docs table is bond 1 / smooth 0
    assert (alpha_conf, alpha_pae, alpha_except, alpha_diff, alpha_disto) == (
        1e-4,
        0.0,
        1.0,
        4.0,
        0.03,
    )
    assert (alpha_bond, smooth) == (0.0, 1.0)
    w = cfg.loss
    assert w.diffusion_loss == alpha_diff
    assert w.distogram_loss == alpha_disto
    assert w.plddt_loss == w.pde_loss == w.resolved_loss == alpha_conf * alpha_except
    # PAE on: alpha_pae 1.0 (configs_base.py:402 "1 in finetuning stage 3"; the
    # default there is 0.0)
    assert w.pae_loss == alpha_conf * 1.0
    # stage 1: alpha_bond 1.0, smooth_lddt 0
    assert w.bond_loss == alpha_diff * 1.0
    assert w.smooth_lddt_loss == 0.0
    clip, beta1, beta2, wd, adamw = ref["optim"]
    assert (cfg.grad_clip_max_norm, cfg.betas, cfg.weight_decay) == (
        clip,
        (beta1, beta2),
        wd,
    )
    assert adamw == 0.0
    mse_weight, w_dna, w_rna, w_lig = ref["mse"]
    assert math.isclose(mse_weight, 1 / 3)
    assert (w.alpha_dna, w.alpha_rna, w.alpha_ligand) == (w_dna, w_rna, w_lig)
    p_mean, p_std, sigma_data, model_sigma_data = ref["noise"]
    assert (cfg.noise.P_mean, cfg.noise.P_std, cfg.noise.sigma_data) == (
        p_mean,
        p_std,
        sigma_data,
    )
    assert model_sigma_data == cfg.noise.sigma_data
