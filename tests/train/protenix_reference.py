"""Reference values from Protenix's own code, for ``test_protenix_parity.py``.

NOT a pytest module: run it with the Protenix environment's python.

    export PROTENIX_SRC=~/foldforge-base/protenix
    ~/foldforge-base/envs/protenix/.venv/bin/python protenix_reference.py IN.npz OUT.npz

``IN.npz`` holds flat per-atom arrays of one tiny structure (every atom of the crop,
resolved or not); the outputs are what ``protenix/model/loss.py`` makes of them: the
experimentally-resolved loss, the pLDDT target bins (``calculate_atom_bespoke_lddt`` +
``PLDDTLoss.bins_from_lddt``), the PDE target bins (``PDELoss.calculate_label``), the
af3 learning-rate scheduler values, and the loss / optimizer settings of
``configs/configs_base.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

src = os.environ["PROTENIX_SRC"]
sys.path.insert(0, src)
os.chdir(src)

from protenix.model.loss import (
    ExperimentallyResolvedLoss,
    PDELoss,
    PLDDTLoss,
    calculate_atom_bespoke_lddt,
)
from protenix.utils.lr_scheduler import AlphaFold3LRScheduler


def main(path_in: str, path_out: str) -> None:
    d = {k: torch.from_numpy(v) for k, v in np.load(path_in).items()}
    coord_mask = d["coordinate_mask"].bool()
    pred, true = d["pred"].float(), d["true"].float()
    out: dict[str, np.ndarray] = {}

    # experimentally resolved (loss.py:1010-1060): logits [N_sample, N_atom, 2]
    out["resolved"] = ExperimentallyResolvedLoss()(
        logits=d["resolved_logits"].float()[None],
        coordinate_mask=d["coordinate_mask"].float(),
    ).numpy()

    # pLDDT target, as ProtenixLoss.calculate_losses (loss.py:1753-1770): only the
    # atoms with coordinates
    lddt, weight = calculate_atom_bespoke_lddt(
        pred_coordinate=pred[None][..., coord_mask, :],
        true_coordinate=true[..., coord_mask, :],
        is_nucleotide=d["is_nuc"][coord_mask].bool(),
        is_polymer=d["is_polymer"][coord_mask].long(),
        rep_atom_mask=d["plddt_m_rep"][coord_mask].bool(),
        is_nucleotide_threshold=30.0,
        is_not_nucleotide_threshold=15.0,
    )
    bins = PLDDTLoss().bins_from_lddt(lddt, weight)  # [1, N, 50] one-hot
    out["plddt_bins"] = bins.argmax(-1)[0].numpy()
    out["plddt_weight"] = weight[0, :, 0].numpy()
    out["plddt_lddt"] = (lddt / (weight + 1e-6))[0, :, 0].numpy()

    # PDE target (loss.py:667-725)
    pde_bins, pde_mask = PDELoss().calculate_label(
        pred_coordinate=pred[None],
        true_coordinate=true,
        coordinate_mask=d["coordinate_mask"].float(),
        rep_atom_mask=d["distogram_rep"].bool(),
    )
    out["pde_bins"] = pde_bins.argmax(-1)[0].numpy()
    out["pde_mask"] = pde_mask.numpy()

    # learning rate (lr_scheduler.py:85-91)
    sched = AlphaFold3LRScheduler(
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3),
        warmup_steps=2000,
        lr=1e-3,
        decay_every_n_steps=50000,
        decay_factor=0.95,
    )
    steps = [
        0,
        1,
        1000,
        1999,
        2000,
        2001,
        49999,
        50000,
        50001,
        100000,
        149999,
        150000,
        300000,
    ]
    out["lr_steps"] = np.array(steps)
    out["lr_values"] = np.array(
        [
            sched._get_step_lr(s)  # noqa: SLF001 - Protenix's own lr is the reference
            for s in steps
        ],
        dtype=np.float64,
    )

    # settings of configs_base.py
    sys.path.insert(0, src)
    from configs.configs_base import configs

    w = configs["loss"]["weight"]
    out["weights"] = np.array(
        [
            w["alpha_confidence"],
            w["alpha_pae"],
            w["alpha_except_pae"],
            w["alpha_diffusion"],
            w["alpha_distogram"],
            w["alpha_bond"],
            w["smooth_lddt"],
        ],
        dtype=np.float64,
    )
    adam = configs["adam"]
    out["optim"] = np.array(
        [
            configs["grad_clip_norm"],
            adam["beta1"],
            adam["beta2"],
            adam["weight_decay"],
            float(adam["use_adamw"]),
        ],
        dtype=np.float64,
    )
    out["mse"] = np.array(
        [
            configs["loss"]["diffusion"]["mse"][k]
            for k in ("weight_mse", "weight_dna", "weight_rna", "weight_ligand")
        ],
        dtype=np.float64,
    )
    out["noise"] = np.array(
        [
            configs["train_noise_sampler"]["p_mean"],
            configs["train_noise_sampler"]["p_std"],
            configs["train_noise_sampler"]["sigma_data"],
            configs["sigma_data"],
        ],
        dtype=np.float64,
    )
    np.savez(path_out, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
