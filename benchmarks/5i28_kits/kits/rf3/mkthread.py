"""RF3 template input for the 5I28 benchmark: a threaded backbone PDB per n (query sequence; 5syd_A N/CA/C/O coordinates on the
query positions its hit row aligns; occupancy 0 elsewhere). RF3's only template track is the input structure's CA distogram
(template_selection), one coordinate set per chain, so the one template is 5syd_A -- FoldForge's first of its 4.
Same construction as ~/foldforge-scratch/anthropic-kits/mkreal_rf3.py; sources here are the benchmark's: cif_raw/sy/5syd.cif.gz
(gunzipped to inputs/src/5syd.cif) and the 20-hit LMDB a3m (inputs/src/5i28_templates_hits20.a3m).
usage: mkthread.py OUTROOT N [N ...]   -> OUTROOT/x<N>/tpl_x<N>.pdb + tpl_info.json
"""

import json
import sys
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "inputs" / "src"
AA3 = dict(
    A="ALA",
    R="ARG",
    N="ASN",
    D="ASP",
    C="CYS",
    Q="GLN",
    E="GLU",
    G="GLY",
    H="HIS",
    I="ILE",
    L="LEU",
    K="LYS",
    M="MET",
    F="PHE",
    P="PRO",
    S="SER",
    T="THR",
    W="TRP",
    Y="TYR",
    V="VAL",
)
TID = "5syd_A"

lines = (SRC / "5i28_templates_hits20.a3m").read_text().split("\n")
rows = {
    lines[i][1:].split("/")[0]: lines[i + 1]
    for i in range(0, len(lines) - 1, 2)
    if lines[i].startswith(">")
}
row = rows[TID]
q = next(
    l
    for l in (HERE / "inputs" / "msa" / "5i28.a3m").read_text().split("\n")
    if l and not l.startswith(">")
)
assert len(q) == len(row) == 128, (len(q), len(row))
aa = pdbx.get_structure(
    pdbx.CIFFile.read(SRC / "5syd.cif"), model=1, use_author_fields=False
)
aa = aa[
    (aa.chain_id == "A") & np.isin(aa.atom_name, ["N", "CA", "C", "O"]) & ~aa.hetero
]
mapping, t = (
    {},
    0,
)  # query position (0-based) -> template label_seq_id (the hit row is query-aligned, residues 1..88)
for qi, ch in enumerate(row):
    if ch != "-":
        t += 1
        mapping[qi] = t
atoms, nal, nmis = [], 0, 0
for qi in range(128):
    res = aa[aa.res_id == mapping[qi]] if qi in mapping else aa[:0]
    ok = len(res) == 4 and set(res.atom_name) == {"N", "CA", "C", "O"}
    if ok:
        nal += 1
        nmis += res.res_name[0] != AA3[q[qi]]
    for k, an in enumerate(["N", "CA", "C", "O"]):
        xyz = (
            res.coord[list(res.atom_name).index(an)]
            if ok
            else np.array([3.8 * qi, 0.4 * k, 0.0])
        )
        atoms.append((qi + 1, AA3[q[qi]], an, xyz, 1.0 if ok else 0.0))
resid = sorted(qi + 1 for qi in mapping)
lo, hi = resid[0], resid[-1]
assert hi - lo + 1 == len(resid)  # contiguous: one template_selection range per chain
print(
    f"template {TID}: aligned query positions {len(mapping)} ({lo}-{hi}), complete backbone {nal}, residue-type mismatches {nmis}"
)
for n in map(int, sys.argv[2:]):
    out = Path(sys.argv[1]) / f"x{n}"
    out.mkdir(parents=True, exist_ok=True)
    arrs = []
    for c in range(n):
        a = struc.AtomArray(len(atoms))
        a.chain_id[:] = chr(65 + c)
        a.res_id = np.array([x[0] for x in atoms])
        a.res_name = np.array([x[1] for x in atoms])
        a.atom_name = np.array([x[2] for x in atoms])
        a.element = np.array([x[2][0] for x in atoms])
        a.coord = np.array([x[3] for x in atoms], dtype=np.float32) + np.array(
            [0, 80.0 * c, 0], dtype=np.float32
        )
        a.set_annotation("occupancy", np.array([x[4] for x in atoms], dtype=float))
        a.set_annotation("b_factor", np.zeros(len(atoms)))
        arrs.append(a)
    f = pdb.PDBFile()
    pdb.set_structure(f, struc.concatenate(arrs))
    f.write(out / f"tpl_x{n}.pdb")
    json.dump(
        dict(
            template=TID,
            aligned=len(mapping),
            complete_backbone=nal,
            mismatches=int(nmis),
            res_range=[lo, hi],
        ),
        open(out / "tpl_info.json", "w"),
    )
    print(f"wrote {out}/tpl_x{n}.pdb")
