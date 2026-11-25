"""The batch protocol and the plug-in point for a data loader.

FoldJet does not import MiniWorld. A training data source is ``module:callable``;
the callable receives ``(cfg, device)`` and returns an iterator of :class:`TrainBatch`
(one example = batch size 1, as MiniWorld's ``num_batch: 1``). MiniWorld (or any
loader) provides the adapter: it turns its crop / MSA / template sampling and
ground-truth atoms into the dense layout of FoldJet's featurisation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import torch

    from foldjet.train.config import TrainConfig


@dataclass
class TrainBatch:
    """One training example on the GPU.

    In FoldJet's dense layout ``[res, atoms_per_token]``.
    """

    #: The model's feature dict, exactly what ``AlphaFold3.run_trunk`` takes
    #: (FoldJet featurisation; MSA pool, templates, ref conformers).
    features: dict[str, torch.Tensor]
    #: Ground-truth atom positions ``[res, atoms, 3]`` in the layout of
    #: ``features['pred_dense_atom_mask']``.
    atom_positions: torch.Tensor
    #: Resolved-atom mask ``[res, atoms]`` (bool); ``None`` =
    #: ``features['pred_dense_atom_mask']``.
    atom_mask: torch.Tensor | None = None
    #: Covalent bonds for the bond loss ``[P, 4]`` int
    #: ``(token_i, atom_i, token_j, atom_j)``, ``-1`` padded; ``None`` = no bond term.
    bond_pairs: torch.Tensor | None = None
    name: str = ""
    #: OPTIONAL sample-fixed index of the present ground-truth atoms for the
    #: confidence targets: flat ``[res * atoms]`` indices of the atoms of
    #: ``atom_mask`` (``features['pred_dense_atom_mask']`` when ``None``), in
    #: ascending order, padded to a fixed length (``FIXED_ATOMS``) with 0, plus the
    #: validity ``[K]`` (bool). ``None`` = computed on the fly with ``nonzero`` (a host
    #: sync, a data-dependent shape). Build both with :func:`fixed_atom_index` where
    #: the data are still on the host (the loader, ``TrainBatch`` construction).
    present_index: torch.Tensor | None = None
    present_valid: torch.Tensor | None = None


#: the fixed length of the present-atom index (MiniWorld ``crop.max_atoms``: 4096); an
#: example with more atoms pads up to the next multiple of 512.
FIXED_ATOMS = 4096


def fixed_atom_index(
    atom_mask: torch.Tensor, length: int = FIXED_ATOMS
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(index [K] int64, valid [K] bool)`` of the true entries of ``atom_mask``.

    ``atom_mask`` ``[res, atoms]``; the indices are flat, ascending, padded with 0.
    Call it on host (CPU) data: it uses ``nonzero``. ``K = length``, or the count
    rounded up to a multiple of 512 when the mask has more atoms than ``length``.
    """
    import torch  # noqa: PLC0415 - torch is a typing-only import at module level

    flat = atom_mask.reshape(-1).bool()
    index = flat.nonzero(as_tuple=True)[0]
    count = int(index.numel())
    if count > length:
        length = -(-count // 512) * 512
    out = torch.zeros(length, dtype=torch.long, device=flat.device)
    out[:count] = index
    valid = torch.zeros(length, dtype=torch.bool, device=flat.device)
    valid[:count] = True
    return out, valid


def features_to_device(
    features: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Move ``features`` to ``device``, except the atom-layout ``input_shape`` metadata.

    The ``GatherInfo.input_shape`` entries (``<prefix>:input_shape``) stay on the
    HOST: ``GatherInfo.from_dict`` reads them as Python ints in every micro step, and
    a device copy would be a device->host sync each time.
    """
    return {
        k: (v.cpu() if k.endswith(":input_shape") else v.to(device))
        for k, v in features.items()
    }


def with_fixed_atom_index(example: TrainBatch, length: int = FIXED_ATOMS) -> TrainBatch:
    """Return ``example`` with ``present_index`` / ``present_valid`` filled.

    Computed from the mask the step uses; a no-op when already present.
    """
    from dataclasses import replace  # noqa: PLC0415 - local import kept as is

    if example.present_index is not None:
        return example
    mask = (
        example.atom_mask
        if example.atom_mask is not None
        else example.features["pred_dense_atom_mask"]
    )
    index, valid = fixed_atom_index(mask.cpu(), length)
    device = example.atom_positions.device
    return replace(
        example, present_index=index.to(device), present_valid=valid.to(device)
    )


BatchSource = "Callable[[TrainConfig, torch.device], Iterator[TrainBatch]]"


def load_source(
    spec: str,
) -> Callable[[TrainConfig, torch.device], Iterator[TrainBatch]]:
    """Resolve ``module:callable``.

    The callable may be a generator function or return any iterator.
    """
    module, _, name = spec.partition(":")
    if not module or not name:
        message = f"--batch-source must be 'module:callable', got {spec!r}"
        raise ValueError(message)
    return getattr(importlib.import_module(module), name)
