"""Named training recipes: ``miniworld`` and ``protenix``.

``miniworld`` is the default, :class:`foldjet.train.config.TrainConfig` as it is;
``protenix`` is Protenix v1's own fine-tuning recipe, ``--recipe protenix``.

``protenix`` changes nothing but the fields listed in :data:`PROTENIX_REFERENCES`; each
value below is traceable to a Protenix file and line (repository ``protenix`` =
``~/foldforge-base/protenix``). An explicit command-line flag of
``scripts/train_protenix1.py`` still overrides a recipe value.

The recipe is the FINE-TUNING STAGE 1 loss of
``docs/training_inference_instructions.md`` (table "Training Cost": alpha_bond 1.0,
smooth_lddt 0, alpha_pae 0, alpha_confidence 1e-4, alpha_diffusion 4.0, alpha_distogram
0.03) with the optimizer / schedule / EMA / cycle / sample settings of
``finetune_demo.sh``. NOT part of the recipe, because the
data is kept as it is (MiniWorld PDB data, MiniWorld sampler): the crop (384 here, 640
in stage 1), the sample weights, the chain / atom permutation, the resolution filter of
the confidence losses (``loss.resolution`` 0.1-4.0 A: the data carries no resolution).
"""

from __future__ import annotations

from dataclasses import replace

from foldjet.train.config import LossWeights, TrainConfig

RECIPES = ("miniworld", "protenix")

#: parameter -> (Protenix value, source). The values are the ones :func:`protenix` sets
#: (checked by tests/train/test_recipes.py).
PROTENIX_REFERENCES: dict[str, tuple[str, str]] = {
    "grad_clip_max_norm": (
        "10",
        "configs/configs_base.py:80 `grad_clip_norm`: 10; runner/train.py:570-574 "
        "(`update`; called at 614-616) "
        "`torch.nn.utils.clip_grad_norm_(model.parameters(), 10)`",
    ),
    "betas": (
        "(0.9, 0.95)",
        "configs/configs_base.py:83-84 `adam.beta1` / `adam.beta2`; "
        "protenix/utils/training.py:105-109 `torch.optim.Adam(..., betas=...)`",
    ),
    "weight_decay": (
        "1e-8, torch.optim.Adam (coupled L2, NOT AdamW), ONE parameter group over "
        "every parameter",
        "configs/configs_base.py:85 `adam.weight_decay` 1e-8, :87 `use_adamw` False; "
        "protenix/utils/training.py:73-117 `get_optimizer` "
        "(`param_names` is [''] in finetune_demo.sh -> "
        "`param_groups = [{'params': model.parameters(), 'lr': configs.lr}]`, "
        "no per-parameter exclusion)",
    ),
    "max_lr": (
        "1e-3",
        "finetune_demo.sh `--lr 0.001` (configs_base.py:74 default 0.0018)",
    ),
    "warmup_steps": (
        "2000",
        "finetune_demo.sh `--warmup_steps 2000` (configs_base.py:76 default 10)",
    ),
    "decay": (
        "x0.95 every 50000 steps, count = step // 50000 (from step 0), lr(0) = 0",
        "configs/configs_base.py:74-79,90-95 (`lr_scheduler` 'af3', "
        "`decay_every_n_steps` 50000, `decay_factor` 0.95); "
        "protenix/utils/lr_scheduler.py:85-91 "
        "`AlphaFold3LRScheduler._get_step_lr`: `step <= warmup: step / warmup * lr` "
        "else `lr * 0.95 ** (step // decay_every_n_steps)`",
    ),
    "min_lr": (
        "none (no floor)",
        "protenix/utils/lr_scheduler.py:85-91 has no floor; `min_lr_ratio` belongs to "
        "the cosine scheduler of `finetune_optim_configs` only",
    ),
    "ema_decay": (
        "0.999, updated after every optimizer step, over every parameter",
        "finetune_demo.sh `--ema_decay 0.999`; runner/ema.py:57-71, "
        "runner/train.py:694-695",
    ),
    "recycles": (
        "N_cycle uniform in 1..4",
        "finetune_demo.sh `--model.N_cycle 4`; protenix/model/protenix.py:887-888 "
        "`RandomState(step).randint(1, N_cycle + 1)`",
    ),
    "rollout_steps": (
        "20",
        "finetune_demo.sh `--sample_diffusion.N_step 20`; "
        "configs/configs_base.py:183 `N_step_mini_rollout` 20, "
        ":184 `N_sample_mini_rollout` 1",
    ),
    "num_augment": (
        "48 diffusion samples",
        "finetune_demo.sh `--diffusion_batch_size 48` (configs_base.py:122); "
        "docs table: 48 initial, 32 fine-tuning stages 1-2 (at crop 640)",
    ),
    "diffusion_loss": (
        "4.0",
        "configs/configs_base.py:404 `alpha_diffusion`; "
        "protenix/model/loss.py:1469 `mse_loss` weight",
    ),
    "bond_loss": (
        "4.0 = alpha_diffusion 4.0 x alpha_bond 1.0",
        "configs/configs_base.py:406 (`alpha_bond` 0.0 'or 1 in finetuning stages'), "
        "docs table (stage 1: 1.0); protenix/model/loss.py:1470 `bond_loss` = "
        "alpha_diffusion * alpha_bond. "
        "FoldJet's `LossWeights.bond_loss` multiplies the bond term directly "
        "(step.py), so it is the product",
    ),
    "smooth_lddt_loss": (
        "0.0",
        "configs/configs_base.py:407 `smooth_lddt` 1.0 'or 0 in finetuning stages'; "
        "docs table: stage 1 0; loss.py:1471 weight alpha_diffusion * smooth_lddt",
    ),
    "distogram_loss": ("0.03", "configs/configs_base.py:405 `alpha_distogram`"),
    "plddt_loss / pde_loss / resolved_loss": (
        "1e-4",
        "configs/configs_base.py:401,403 `alpha_confidence` 1e-4 x `alpha_except_pae` "
        "1.0; loss.py:1464-1466",
    ),
    "pae_loss": (
        "1e-4",
        "configs/configs_base.py:402 `alpha_pae` 0.0 default, "
        "'1 in finetuning stage 3': we switch it on (alpha_pae 1.0); "
        "loss.py:1467 = alpha_confidence * alpha_pae = 1e-4 x 1.0",
    ),
    "targets_protenix": (
        "pLDDT neighbours / thresholds and PDE, PAE bin edges as Protenix "
        "(PDE edges shifted one step: loss.py:693-725, PAE :916-938)",
        "protenix/model/loss.py:1212-1280 `calculate_atom_bespoke_lddt` "
        "(neighbours m = plddt_m_rep_atom_mask & polymer, `<` thresholds at "
        "1253-1258; `PLDDTLoss` 1283-1440, an atom with weight 0 keeps bin 0 at "
        "1324-1349)",
    ),
}


def protenix(cfg: TrainConfig) -> TrainConfig:
    """Return ``cfg`` with the Protenix v1 fine-tuning recipe.

    See :data:`PROTENIX_REFERENCES`.
    """
    loss = replace(
        cfg.loss,
        diffusion_loss=4.0,
        # protenix configs_base.py:433-436 `weight_mse` 1/3, `weight_dna` /
        # `weight_rna` / `weight_ligand`
        alpha_dna=5.0,
        alpha_rna=5.0,
        alpha_ligand=10.0,
        smooth_lddt_loss=0.0,
        bond_loss=4.0,
        distogram_loss=0.03,
        plddt_loss=1e-4,
        pde_loss=1e-4,
        resolved_loss=1e-4,
        # alpha_confidence 1e-4 x alpha_pae 1.0 (configs_base.py:402 "1 in finetuning
        # stage 3"); the user switched the PAE loss ON
        pae_loss=1e-4,
    )
    return replace(
        cfg,
        loss=loss,
        recipe="protenix",
        grad_clip_max_norm=10.0,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        max_lr=1e-3,
        min_lr=0.0,
        warmup_steps=2000,
        decay_steps=50_000,
        decay_factor=0.95,
        decay_from_warmup=False,
        ema_decay=0.999,
        use_ema=True,
        recycles_min=1,
        recycles_max=4,
        rollout_steps=20,
        num_augment=48,
        targets_protenix=True,
    )


def apply(name: str, cfg: TrainConfig) -> TrainConfig:
    """Return ``cfg`` under the recipe ``name`` (``miniworld``: unchanged)."""
    if name == "miniworld":
        return cfg
    if name == "protenix":
        return protenix(cfg)
    msg = f"unknown recipe {name!r}; one of {RECIPES}"
    raise ValueError(msg)


__all__ = ["PROTENIX_REFERENCES", "RECIPES", "LossWeights", "apply", "protenix"]
