"""Build MiniWorld's own ``BioMolData.BioMolConfig`` from a data-config YAML + the PDB-only overlay (no hydra needed)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _mwshim  # noqa: F401  (must precede the miniworld imports)
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = HERE / "data_20261014_local_ft.yaml"  # MiniWorld v2.4.0 data (was H100_default.yaml: the older BioMol data)
DEFAULT_OVERLAY = HERE / "overlay_pdb_only.yaml"


def load_data_dict(data_config: str | Path = DEFAULT_DATA_CONFIG, overlay: str | Path | None = DEFAULT_OVERLAY) -> dict[str, Any]:
    cfg = OmegaConf.load(str(data_config))
    if overlay:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(overlay)))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def build_biomol_config(data: dict[str, Any]):
    from miniworld.configs.data import CropConfig, MSAConfig, SamplerConfig, TemplateConfig, TokenizerConfig
    from miniworld.data.dataloader.dataloader import BioMolData

    return BioMolData.BioMolConfig(
        crop_config=CropConfig(**data.get("crop", {})),
        msa_config=MSAConfig(**data.get("msa", {})),
        template_config=TemplateConfig(**data.get("template", {})),
        DB_config=data["train_db"],
        tokenizer_config=TokenizerConfig(**data.get("tokenizer", {})),
        sampler_config=SamplerConfig(**data.get("sampler", {})),
    )
