"""Pair type (MiniWorld sampler bucket) of every catalog row, from the train_item TSV the catalog was built from (cluster prefixes, MiniWorld's
own ``_pdb_edge_type``). The Arrow catalog keeps the rows in TSV order (only rows of sources in the resources index are kept), so row i of the
catalog is row i of the filtered TSV; the item ids are compared as a check."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import mwconfig  # noqa: F401

TYPES = ("protein_protein", "protein_ligand", "protein_dna", "protein_rna", "antibody_protein", "dna_dna", "rna_rna", "dna_rna",
         "antibody_antibody", "antibody_ligand", "na_ligand", "etc_interface", "sole")
NON_PDB = 100  # code of a non-pdb row: NON_PDB + index of the source in ``sources``


def build(train_item_path: str | Path, sources: dict[str, int], out: Path) -> np.ndarray:
    from miniworld.data.dataloader.sources import _pdb_edge_type

    codes: list[int] = []
    ids: dict[int, str] = {}
    with open(train_item_path) as handle:
        handle.readline()
        for raw in handle:
            row = raw.rstrip("\n").split("\t")
            if len(row) < 9 or row[0] not in sources:
                continue
            source, c1, c2, pdb_id, asm, mdl, alt, ch1, ch2 = row[:9]
            if source == "pdb":
                edge = c1 if c2 == "None" else f"{c1}_{c2}"
                codes.append(TYPES.index(_pdb_edge_type(edge)))
            else:
                codes.append(NON_PDB + sources[source])
            n = len(codes) - 1
            if n % 1_000_003 == 0 or n < 3:
                chains = (ch1,) if c2 == "None" else (ch1, ch2)
                rid = pdb_id.lower() if source == "pdb" else pdb_id
                ids[n] = f"{source}:{rid}:{asm}:{mdl}:{alt}:{':'.join(chains)}"
    arr = np.asarray(codes, dtype=np.uint8)
    np.save(out, arr)
    np.save(out.with_suffix(".ids.npy"), np.array(sorted(ids.items()), dtype=object), allow_pickle=True)
    return arr
