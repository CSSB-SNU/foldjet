"""CPU-side output work of fb_multi_worker.py (CIF chain renaming + writing), kept in a module of its own for the writer processes."""

import io
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx


def rename_chains(cif: str, new_names: list[str]) -> str:
    """The k-th chain of the prediction (in order of appearance) gets FoldBench's k-th chain id in every place the CIF carries one."""
    f = pdbx.CIFFile.read(io.StringIO(cif))
    block = f.block
    atom = block["atom_site"]
    old = []
    for c in atom["label_asym_id"].as_array():
        if c not in old:
            old.append(c)
    if len(old) != len(new_names):
        raise ValueError(
            f"the prediction has {len(old)} chains, the input {len(new_names)}"
        )
    mapping = dict(zip(old, new_names, strict=True))
    for col in ("label_asym_id", "auth_asym_id"):
        atom[col] = pdbx.CIFColumn(np.array([mapping[c] for c in atom[col].as_array()]))
    # Every other chain-id column of every category (struct_asym.id, *_scheme asym_id / pdb_strand_id, entity_poly.pdbx_strand_id,
    # struct_conn ptnr*_asym_id, ...): a column left with the old ids (pdbx_poly_seq_scheme was) makes the file unreadable for
    # AlphaFold3's own parser (alphafold3.structure.from_mmcif: "Lookup for the following atom from the _atom_site table failed").
    targets = [("struct_asym", "id")] + [
        (cat_name, col)
        for cat_name in block
        if cat_name != "atom_site"
        for col in block[cat_name]
        if col.endswith(("asym_id", "strand_id"))
    ]
    for cat, col in targets:
        if cat in block and col in block[cat]:
            vals = block[cat][col].as_array()
            block[cat][col] = pdbx.CIFColumn(
                np.array(
                    [
                        ",".join(mapping.get(x, x) for x in str(v).split(","))
                        for v in vals
                    ]
                )
            )
    buf = io.StringIO()
    f.write(buf)
    return buf.getvalue()


def write_seed(
    d: str, t: str, s: int, cifs: list[str], scores: list[float], new_names: list[str]
) -> list[tuple[int, int, float]]:
    """Rename + write every sample CIF of one trunk_seed; returns its (seed, sample, ranking_score) rows."""
    rows = []
    for k, cif in enumerate(cifs):
        Path(d, f"{t}_seed{s}_sample{k}_pred.cif").write_text(
            rename_chains(cif, new_names)
        )
        rows.append((s, k, scores[k]))
    return rows
