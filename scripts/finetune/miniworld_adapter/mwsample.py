"""MiniWorld sample access: the dataset (catalog + weights + WeightedSampler, MiniWorld's own classes) and a per-item preprocess that also hands over the
intermediate objects MiniWorld's ``make_batch`` receives (the cropped CIFMol, token maps, MSA, templates), without copying or changing any MiniWorld code:
``preprocess.make_batch`` is wrapped for the duration of one call and its keyword arguments are kept."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import mwconfig  # noqa: F401  (installs the import shims first)

logger = logging.getLogger("ft_adapter")


@dataclass
class RawSample:
    item_id: str
    record: Any
    batch: Any  # miniworld Batch (batch size 1)
    cifmol: Any  # cropped CIFMolAttached (terminal OXT/OP3 already removed)
    atom_to_token: np.ndarray  # [n_atoms] token index of each atom of ``cifmol``
    token_to_residue: np.ndarray  # [n_tokens] residue index (into cifmol.residues) of each token
    msa: Any  # miniworld MSAFeatures at residue level (sample_msa result)
    templates: Any  # miniworld ProteinTemplate at residue level
    tries: int = 1
    seconds: float = 0.0


class _Capture:
    def __init__(self, module) -> None:
        self.module = module
        self.kwargs: dict[str, Any] | None = None
        self.original = module.make_batch

    def __enter__(self):
        def wrapper(*args, **kwargs):
            self.kwargs = dict(kwargs)
            return self.original(*args, **kwargs)

        self.module.make_batch = wrapper
        return self

    def __exit__(self, *exc) -> None:
        self.module.make_batch = self.original


def build_dataset(data: dict, light: bool = False):
    """``BioMolData`` for the data dict. ``light=True`` skips the 26M-item weight balancing (a worker needs items + preprocessor only)."""
    from miniworld.data.dataloader.dataloader import BioMolData

    cfg = mwconfig.build_biomol_config(data)
    if not light:
        return BioMolData(cfg)

    class Light(BioMolData):
        def _load_items(self) -> None:  # catalog mmap only; no source balancing
            from miniworld.data.dataloader.dataloader import _load_catalog_arrow

            cache = self.config.DB_config.catalog_cache_path
            items, _w, _s, _is_raw, _fp = _load_catalog_arrow(cache, self.config.sampler_config)
            self.items, self.weights = items, []

    return Light(cfg)


def process_item(dataset, index: int, rng: np.random.Generator, max_tries: int = 20, crop_indices=None, spares=None) -> RawSample:
    """``BioMolData.__getitem__`` with a seeded ``rng`` (the stock one draws an unseeded generator per item) and the same retry rules
    (WrongCroppingError / missing LMDB record / unparsable coordinate: draw another item). MiniWorld draws that replacement UNIFORMLY over the whole
    catalog (a quirk that is harmless with the default source mix but pulls distillation items into a PDB-only run), so the replacement here is the
    next entry of ``spares`` (indices from the same weighted sampler) when given."""
    from miniworld.data.dataloader import preprocess as pre
    from miniworld.data.dataloader.preprocess import WrongCroppingError

    t0 = time.perf_counter()
    idx = int(index)
    for n_try in range(1, max_tries + 1):
        record = dataset.items[idx]
        try:
            with _Capture(pre) as cap:
                batch = dataset.preprocessor.process(record, rng=rng, crop_indices=crop_indices)
        except WrongCroppingError:
            pass
        except KeyError as exc:
            if "not found in LMDB database" not in str(exc):
                raise
        except ValueError as exc:
            if "could not convert string to float" not in str(exc):
                raise
        else:
            k = cap.kwargs or {}
            return RawSample(
                item_id=record.item_id,
                record=record,
                batch=batch,
                cifmol=k["cifmol"],
                atom_to_token=np.asarray(k["atom_to_token_idx_map"]),
                token_to_residue=np.asarray(k["token_to_residue_idx_map"]),
                msa=k["msa"],
                templates=k["templates"],
                tries=n_try,
                seconds=time.perf_counter() - t0,
            )
        if spares is not None:
            if not spares:
                break
            idx = int(spares.pop(0))
        else:
            idx = int(rng.integers(0, len(dataset)))
    msg = f"no usable item after {max_tries} draws (last {record.item_id})"
    raise RuntimeError(msg)
