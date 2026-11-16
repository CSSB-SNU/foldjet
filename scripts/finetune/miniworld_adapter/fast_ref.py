"""Bit-identical, cached replacement for ``alphafold3.model.features.get_reference`` (the reference-conformer routine).

What the original does per call (one call per residue / ligand instance of an example, twice per example: once with the per-sample RandomState, once with
AF3's fixed frames seed 12312837): parse the CCD cif into an RDKit mol, draw ``seed = random_state.randint(1, 1 << 31)``, embed an ETKDGv3 conformer with that
seed, then ``random_augmentation`` (draws more numbers from ``random_state``) and build the per-atom feature dicts.

Everything except the ``random_state`` stream is a pure function of the CCD code:
  * the parsed mol, the atom name / element / charge / atom_name_chars tables and the bond lists are cached per CCD code (per process);
  * the embedded conformer is a pure function of (CCD code, seed); in the deterministic frames pass the seed sequence is fixed (the augmentation draws a
    fixed number of normals whatever the coordinates), so (CCD code, seed) repeats from example to example: those conformers are read from a prebuilt
    on-disk cache (``FT_CONF_CACHE``) and misses are computed and kept in memory (bounded). The per-sample pass uses fresh random seeds (never repeated):
    it is computed (RDKit) every time and never stored.
``random_state`` is advanced exactly as in the original (same calls in the same order), so every downstream draw is unchanged.
"""
from __future__ import annotations

import contextlib
import glob
import os
import pickle
from typing import Any

import numpy as np

_ENTRIES: dict[str, Any] = {}
_CONF: dict[tuple, Any] = {}
_CONF_MAX = int(os.environ.get("FT_CONF_MEM_MAX", "300000"))
_state = {"cacheable": False, "orig": None, "hits": 0, "misses": 0, "stored": 0, "calls": 0, "preloaded": 0}
_MISSING = object()


def stats() -> dict[str, int]:
    return dict(_state, conf_entries=len(_CONF), mol_entries=len(_ENTRIES))


@contextlib.contextmanager
def cacheable(flag: bool = True):
    """Conformers drawn inside this context (the deterministic frames pass) are read from / kept in the conformer cache."""
    prev = _state["cacheable"]
    _state["cacheable"] = flag
    try:
        yield
    finally:
        _state["cacheable"] = prev


def load_cache(path: str | None) -> int:
    """Load every ``*.pkl`` shard (dict {(ccd, seed, max_iter): float32 [n_atoms, 3] | None}) of ``path`` (a directory or one file)."""
    if not path:
        return 0
    files = sorted(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True)) if os.path.isdir(path) else [path]
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        _CONF.update(d)
        n += len(d)
    _state["preloaded"] += n
    return n


def _entry(res_name, ccd, features, ru, ref_max_modified_date):
    e = _ENTRIES.get(res_name, _MISSING)
    if e is not _MISSING:
        return e
    ccd_cif = ccd.get(res_name)
    if not ccd_cif:
        _ENTRIES[res_name] = None  # SMILES path: not cached, the original handles it
        return None
    mol = None
    try:
        mol = ru.mol_from_ccd_cif(ccd_cif, remove_hydrogens=False)
    except ru.MolFromMmcifError:
        features.logging.warning("Failed to construct mol from ccd_cif for: %s", res_name)
    e = {"ccd_cif": ccd_cif, "mol": mol, "from_atom": ccd_cif.get("_chem_comp_bond.atom_id_1", None), "dest_atom": ccd_cif.get("_chem_comp_bond.atom_id_2", None),
         "mol_tab": None, "ccd_tab": None, "ccd_pos": {}}
    _ENTRIES[res_name] = e
    return e


def _table(atom_names, elements, charges, features):
    """Static per-atom feature tables, exactly the per-atom dict construction of the original (first occurrence of a repeated atom name wins)."""
    stat = {}
    idxs = {}
    for atom_name in atom_names:
        idx = atom_names.index(atom_name)
        charge = 0 if charges[idx] == "?" else int(charges[idx])
        chars = np.array([ord(c) - 32 for c in atom_name], dtype=int)
        chars = features._pad_to(chars, (4,))
        stat[atom_name] = {"mask": 1, "element": elements[idx], "charge": charge, "atom_name_chars": chars}
        idxs[atom_name] = idx
    return stat, idxs


def _build(pos, table):
    stat, idxs = table
    out = {}
    for atom_name, s in stat.items():
        d = {"positions": pos[idxs[atom_name]]}
        d.update(s)
        out[atom_name] = d
    return out


def embed(mol, res_name, seed, max_iterations):
    """The conformer of the original ``get_reference`` before augmentation: float32 [n_atoms, 3] in the mol atom order, or None if RDKit fails."""
    from alphafold3.data.tools import rdkit_utils as ru

    conformer = ru.get_random_conformer(mol=mol, random_seed=seed, max_iterations=max_iterations, logging_name=res_name)
    if not conformer:
        return None
    return np.array([[c.x, c.y, c.z] for c in (conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))], dtype=np.float32)


def frames_seeds(n: int) -> list[int]:
    """The first ``n`` RDKit seeds of the deterministic frames pass (RandomState(12312837); every instance draws one seed then the augmentation draws)."""
    from alphafold3.model import features

    r = np.random.RandomState(12312837)
    out = []
    for _ in range(n):
        out.append(int(r.randint(1, 1 << 31)))
        features.random_augmentation(np.zeros((3, 3)), r)
    return out


def install(features) -> None:
    """Replace ``features.get_reference`` (idempotent)."""
    if _state["orig"] is not None:
        return
    from alphafold3.data.tools import rdkit_utils as ru

    orig = features.get_reference
    _state["orig"] = orig

    def get_reference(res_name, chemical_components_data, ccd, random_state, ref_max_modified_date, conformer_max_iterations):
        _state["calls"] += 1
        e = _entry(res_name, ccd, features, ru, ref_max_modified_date)
        if e is None:
            return orig(res_name, chemical_components_data, ccd, random_state, ref_max_modified_date, conformer_max_iterations)
        mol = e["mol"]
        conformer_pos = None
        have = False
        if mol is not None:
            seed = int(random_state.randint(1, 1 << 31))
            key = (res_name, seed, conformer_max_iterations)
            if _state["cacheable"]:
                got = _CONF.get(key, _MISSING)
                if got is not _MISSING:
                    _state["hits"] += 1
                    conformer_pos, have = got, True
            if not have:
                if _state["cacheable"]:
                    _state["misses"] += 1
                if _REC is not None:  # recording pass: collect the embeds, return a placeholder (CCD path); the result of this pass is discarded
                    _REC.append((res_name, seed, conformer_max_iterations, mol))
                    conformer_pos = None
                elif (res_name, seed, conformer_max_iterations) in _PRE:
                    conformer_pos = _PRE[(res_name, seed, conformer_max_iterations)]
                else:
                    conformer_pos = embed(mol, res_name, seed, conformer_max_iterations)
                if _REC is None and _state["cacheable"] and len(_CONF) < _CONF_MAX:
                    _CONF[key] = conformer_pos
                    _state["stored"] += 1
        if conformer_pos is not None:
            if e["mol_tab"] is None:
                atoms = list(mol.GetAtoms())
                e["mol_tab"] = _table([a.GetProp("atom_name") for a in atoms], [a.GetAtomicNum() for a in atoms], [a.GetFormalCharge() for a in atoms], features)
            table = e["mol_tab"]
            pos = conformer_pos
        else:
            ccd_cif = e["ccd_cif"]
            if e["ccd_tab"] is None:
                atom_names = ccd_cif["_chem_comp_atom.atom_id"]
                charges = ccd_cif["_chem_comp_atom.charge"]
                type_symbols = ccd_cif["_chem_comp_atom.type_symbol"]
                elements = [features.periodic_table.ATOMIC_NUMBER.get(t.capitalize(), 0) for t in type_symbols]
                e["ccd_tab"] = _table(atom_names, elements, charges, features)
            table = e["ccd_tab"]
            dkey = ref_max_modified_date
            if dkey not in e["ccd_pos"]:
                e["ccd_pos"][dkey] = features._get_reference_positions_from_ccd_cif(ccd_cif=ccd_cif, ref_max_modified_date=ref_max_modified_date, logging_name=res_name)
            pos = e["ccd_pos"][dkey]
        pos = features.random_augmentation(pos, random_state)
        return _build(pos, table), e["from_atom"], e["dest_atom"]

    features.get_reference = get_reference


# ---------------------------------------------------------------------------------------------------------------- parallel embedding of slow examples
# The embedding is a pure function of (mol, seed, max_iterations). A rare example (e.g. a photosystem with ~50 chlorophyll instances: RDKit ETKDG FAILS on CLA after
# 9-18 s each, 100+ s per example) is dominated by a few slow embeds. ``record`` runs the per-sample RefStructure pass once with placeholder conformers and collects
# the (code, seed) of every embed (the random_state draws do not depend on the coordinates); ``precompute`` embeds them (sequentially for the first
# FT_EMBED_SERIAL_S seconds, the rest on FT_EMBED_THREADS threads: RDKit releases the GIL) into ``_PRE``; the real pass then reads ``_PRE``. Same values, same order of draws.
_REC: list | None = None
_PRE: dict = {}


def record_start() -> None:
    global _REC
    _REC = []


def record_stop() -> list:
    global _REC
    rec, _REC = _REC or [], None
    return rec


def precompute(rec: list) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    serial_s = float(os.environ.get("FT_EMBED_SERIAL_S", "0"))
    threads = int(os.environ.get("FT_EMBED_THREADS", "8"))
    t0 = time.perf_counter()
    todo = []
    for code, seed, mi, mol in rec:
        if time.perf_counter() - t0 < serial_s:
            _PRE[(code, seed, mi)] = embed(mol, code, seed, mi)
        else:
            todo.append((code, seed, mi, mol))
    if todo:
        _state["parallel_embeds"] = _state.get("parallel_embeds", 0) + len(todo)
        with ThreadPoolExecutor(threads) as ex:
            for (code, seed, mi, _), res in zip(todo, ex.map(lambda a: embed(a[3], a[0], a[1], a[2]), todo)):
                _PRE[(code, seed, mi)] = res


def has_big_component(res_names, ccd, features, min_heavy: int | None = None) -> bool:
    """True when one of the CCD codes of the example is large (>= FT_EMBED_PAR_MIN_HEAVY heavy atoms, default 40): only those can make an embed take seconds
    (CLA, 65 heavy atoms: every RDKit ETKDG attempt fails after 9-18 s); the recording passes are skipped for ordinary examples."""
    from alphafold3.data.tools import rdkit_utils as ru

    limit = int(os.environ.get("FT_EMBED_PAR_MIN_HEAVY", "40")) if min_heavy is None else min_heavy
    for code in {str(c) for c in res_names}:
        e = _entry(code, ccd, features, ru, None)
        if e is not None and e["mol"] is not None and e["mol"].GetNumHeavyAtoms() >= limit:
            return True
    return False
