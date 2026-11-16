"""The steric potential's early exits (no atoms, one chain) return no candidates.

They raised TypeError before (two arguments to a one-argument helper).
"""

import torch

from foldforge.guidance.potentials import VinaStericPotential


def _feats(asym: list[int]) -> dict[str, torch.Tensor]:
    n = len(asym)
    return {
        "asym_id": torch.tensor(asym, dtype=torch.long),
        "atom_to_token_idx": torch.arange(n, dtype=torch.long),
        "ref_element": torch.zeros(n, 128),
    }


def test_single_chain_and_empty_return_empty_candidates():
    pot = VinaStericPotential.__new__(VinaStericPotential)
    for asym in ([0, 0, 0, 0], []):
        pot._cache_key, pot._cache_val = None, None
        idx, r_eq = pot._get_collision_candidates(_feats(asym))
        assert idx.shape == (2, 0)
        assert idx.dtype == torch.long
        assert r_eq.shape == (0,)
