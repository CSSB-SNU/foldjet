"""AlphaFold3's sample ranking score, computed by FoldJet itself.

    ranking_score = (0.8 ipTM + 0.2 pTM, or pTM for a single chain) + 0.5 fraction_disordered - 100 has_clash

``fraction_disordered``, ``has_clash``, ``windowed_solvent_accessible_area`` and ``get_ranking_score`` are AlphaFold3's own functions
(``alphafold3/model/confidences.py``, DeepMind, Apache License 2.0), copied here unchanged in logic: AlphaFold3 keeps them in its model
package, which FoldJet does not install. They run on AlphaFold3's ``Structure`` class and AlphaFold3's ``mkdssp`` binding
(``alphafold3.cpp.mkdssp``: DSSP v4.4.7 on libcifpp v7.0.3, the pins of AlphaFold3's CMakeLists), both shipped by ``foldjet-af3-data``;
so the disorder fraction is the same DSSP run on the same mmCIF as in AlphaFold3. Every FoldJet sample mmCIF is written by that
``Structure`` class (``models/io/dense_atoms.py``), and ``structure.from_mmcif`` reads it back unchanged.

Only the ranking changes: no coordinate, pLDDT, PAE or pTM / ipTM value is touched.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import spatial

_LOG = logging.getLogger(__name__)

# From Sander & Rost 1994 https://doi.org/10.1002/prot.340200303 (AlphaFold3 MAX_ACCESSIBLE_SURFACE_AREA)
MAX_ACCESSIBLE_SURFACE_AREA = {
    "ALA": 106.0,
    "ARG": 248.0,
    "ASN": 157.0,
    "ASP": 163.0,
    "CYS": 135.0,
    "GLN": 198.0,
    "GLU": 194.0,
    "GLY": 84.0,
    "HIS": 184.0,
    "ILE": 169.0,
    "LEU": 164.0,
    "LYS": 205.0,
    "MET": 188.0,
    "PHE": 197.0,
    "PRO": 136.0,
    "SER": 130.0,
    "THR": 142.0,
    "TRP": 227.0,
    "TYR": 222.0,
    "VAL": 142.0,
}

# Weights for ranking confidence (AlphaFold3).
_IPTM_WEIGHT = 0.8
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALIZATION_WEIGHT = 100.0


def windowed_solvent_accessible_area(cif: str, window: int = 25) -> np.ndarray:
    """AlphaFold-RSA (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767): per-residue relative ASA from DSSP, windowed."""
    from alphafold3.constants import residue_names
    from alphafold3.cpp import mkdssp

    result = mkdssp.get_dssp(cif, calculate_surface_accessibility=True)
    parse_row = False
    rasa = []
    for row in result.splitlines():
        if parse_row:
            aa = row[13:14]
            if aa == "!":
                continue
            aa3 = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(aa, "ALA")
            max_acc = MAX_ACCESSIBLE_SURFACE_AREA[aa3]
            acc = int(row[34:38])
            norm_acc = acc / max_acc
            norm_acc = min(norm_acc, 1.0)
            rasa.append(norm_acc)
        if row.startswith("  #  RESIDUE"):
            parse_row = True

    half_w = (window - 1) // 2
    pad_rasa = np.pad(rasa, (half_w, half_w), "reflect")
    return np.convolve(pad_rasa, np.ones(window), "valid") / window


def fraction_disordered(struc, rasa_disorder_cutoff: float = 0.581) -> float:
    """Fraction of protein residues that are disordered (windowed rASA > cutoff); identical-sequence chains reuse one DSSP run."""
    struc = struc.filter_to_entity_type(protein=True)
    rasa = []
    seq_rasa = {}
    for chain_id, chain_seq in struc.chain_single_letter_sequence().items():
        if chain_seq in seq_rasa:
            # We assume that identical sequences have approximately similar rasa values to speed up the computation.
            rasa.extend(seq_rasa[chain_seq])
            continue
        chain_struc = struc.filter(chain_id=chain_id)
        # Rename the chain to 'A' as MKDSSP supports only single letter chain IDs.
        chain_struc = chain_struc.rename_chain_ids(new_id_by_old_id={chain_id: "A"})
        try:
            rasa_per_residue = windowed_solvent_accessible_area(chain_struc.to_mmcif())
            seq_rasa[chain_seq] = rasa_per_residue
            rasa.extend(rasa_per_residue)
        except (ValueError, RuntimeError) as e:
            _LOG.warning("%s: rasa calculation failed: %s", struc.name, e)

    if not rasa:
        return 0.0
    return float(np.mean(np.array(rasa) > rasa_disorder_cutoff))


def has_clash(
    struc,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
    """True if a polymer chain has > 100 atoms, or > 50 % of its atoms, within 1.1 A of another polymer atom (non-adjacent residue or other chain)."""
    struc = struc.filter_to_entity_type(protein=True, rna=True, dna=True)
    if not struc.chains:
        return False
    coords = struc.coords
    coord_kdtree = spatial.cKDTree(coords)
    clashes_per_atom = coord_kdtree.query_ball_point(coords, p=2.0, r=cutoff_radius)
    per_atom_has_clash = np.zeros(len(coords), dtype=np.int32)
    for atom_idx, clashing_indices in enumerate(clashes_per_atom):
        for clashing_idx in clashing_indices:
            if np.abs(struc.res_id[atom_idx] - struc.res_id[clashing_idx]) > 1 or (
                struc.chain_id[atom_idx] != struc.chain_id[clashing_idx]
            ):
                per_atom_has_clash[atom_idx] = True
                break
    for chain_id in struc.chains:
        mask = struc.chain_id == chain_id
        num_atoms = np.sum(mask)
        if num_atoms == 0:
            continue
        num_clashes = np.sum(per_atom_has_clash * mask)
        frac_clashes = num_clashes / num_atoms
        if (
            num_clashes > min_clashes_for_overlap
            or frac_clashes > min_fraction_for_overlap
        ):
            return True
    return False


def get_ranking_score(
    ptm: float, iptm: float, fraction_disordered_: float, has_clash_: bool
) -> float:
    """AlphaFold3's ranking score; ipTM is NaN for a single-chain structure, then pTM is used."""
    if np.isnan(iptm):
        ptm_iptm_average = ptm
    else:
        ptm_iptm_average = _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm
    return (
        ptm_iptm_average
        + _FRACTION_DISORDERED_WEIGHT * fraction_disordered_
        - _CLASH_PENALIZATION_WEIGHT * has_clash_
    )


@dataclass(frozen=True)
class RankingTerms:
    """One sample's ranking score and the terms it is made of."""

    ptm: float
    iptm: float | None
    fraction_disordered: float
    has_clash: bool
    ranking_score: float

    def as_dict(self) -> dict:
        """JSON-ready (ipTM None for a single chain)."""
        return asdict(self)


def ranking_terms(cif: str, ptm: float, iptm: float | None) -> RankingTerms:
    """AlphaFold3's ranking terms of one sample mmCIF written by FoldJet (before any chain renaming).

    ``iptm`` None or NaN, or a structure with one chain, ranks by pTM (AlphaFold3: ipTM is NaN for a single chain).
    """
    from alphafold3 import structure

    struc = structure.from_mmcif(cif)
    single = (
        iptm is None
        or (isinstance(iptm, float) and math.isnan(iptm))
        or len(struc.chains) == 1
    )
    fd = fraction_disordered(struc)
    clash = bool(has_clash(struc))
    ip = float("nan") if single else float(iptm)
    score = float(get_ranking_score(float(ptm), ip, fd, clash))
    return RankingTerms(float(ptm), None if single else float(iptm), fd, clash, score)
