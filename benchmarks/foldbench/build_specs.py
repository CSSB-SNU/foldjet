"""FoldBench `inputs/<target>/data.yaml` (MiniWorld spec: a3m_db list, a3m_key, chain_names, template_offset) -> FoldForge spec
(msa_db / msa per chain letter), plus `<target>.chains.json` = FoldForge chain letter -> FoldBench chain id (for renaming the output CIF).

  build_specs.py TARGETS_FILE OUT_DIR [--foldbench DIR] [--resources DIR] [--ccd-db PATH] [--rna-msa PATH]
                 [--n-trunk 1] [--n-diffusion 5] [--glycans none|entity]

  --foldbench  the FoldBench release (inputs/*/data.yaml + chain FASTA files, meta/chain_seqid_map.tsv, ground_truth/*.cif);
               default $FOLDBENCH_ROOT
  --resources  the BioMolDB `lmdb` directory holding the FoldBench cut package `foldbench/msa/a3m.lmdb`, `foldbench/msa/a3m_rna.lmdb`,
               `foldbench/template/template.lmdb`; default $FOLDBENCH_RESOURCES (the 2026-11-17 run: BioMolDB_20261014/lmdb)
  --ccd-db     FoldForge's CCD LMDB (its own format, manifest.json; holds all 956 codes FoldBench needs); default foldforge.paths.ccd_db()
               ($FOLDFORGE_CCD_DB, else ~/.cache/foldforge/ccd/preprocessed_CCD.lmdb)
  --rna-msa    another RNA MSA database for a sensitivity run (the release's 58 RNA records are 1-3 rows deep for most keys); default the release's

Every spec carries n_trunk_samples 1 (fb_multi_worker.py runs its own `--seeds` trunk_seeds), n_diffusion_samples / diffusion_batch_size 5 and
template_dedup false (FoldBench gives every chain its own template record, duplicates included). The specs hold absolute paths of the FASTA files
and databases: build them once per machine. Dropped on purpose and logged:
  - a chain's MSA when its key is not in the database for the chain's entity type (MiniWorld falls back to a query-only MSA the same way),
    and every DNA chain (AF3-family models take no DNA MSA);
  - a chain's template when the chain is not a protein (released checkpoints take protein templates only), when its key is not in the template
    database, or when `template_offset` is non-zero (FoldForge's spec has no offset field: the template would be aligned to the wrong residues)."""

import argparse
import csv
import json
import os
from pathlib import Path

import lmdb
import yaml
from biotite.structure.io import pdbx

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("targets")
ap.add_argument("out")
ap.add_argument("--foldbench", default=os.environ.get("FOLDBENCH_ROOT"))
ap.add_argument("--resources", default=os.environ.get("FOLDBENCH_RESOURCES"))
ap.add_argument("--ccd-db", default=None)
ap.add_argument("--rna-msa", default=None)
ap.add_argument("--n-trunk", type=int, default=1)
ap.add_argument("--n-diffusion", type=int, default=5)
ap.add_argument(
    "--glycans",
    default="none",
    choices=("none", "entity"),
    help="none = the FoldBench protocol; entity = comparison only",
)
args = ap.parse_args()
if not args.foldbench or not args.resources:
    ap.error(
        "--foldbench / --resources (or FOLDBENCH_ROOT / FOLDBENCH_RESOURCES) are required"
    )
if args.ccd_db is None:
    from foldforge import paths

    args.ccd_db = str(paths.ccd_db())
FB = Path(args.foldbench)
RES = Path(args.resources)
MSA = {
    "polypeptide(L)": RES / "foldbench/msa/a3m.lmdb",
    "polyribonucleotide": Path(args.rna_msa)
    if args.rna_msa
    else RES / "foldbench/msa/a3m_rna.lmdb",
}
TEMPLATE_DB = RES / "foldbench/template/template.lmdb"
CCD_DB = Path(args.ccd_db)


def keyset(path):
    env = lmdb.open(
        str(path), subdir=path.is_dir(), readonly=True, lock=False, readahead=False
    )
    with env.begin() as txn:
        ks = {k.decode() for k in txn.cursor().iternext(values=False)}
    env.close()
    return ks


targets = Path(args.targets).read_text().split()
out = Path(args.out)
nt, nd, GLYCAN_MODE = args.n_trunk, args.n_diffusion, args.glycans
out.mkdir(parents=True, exist_ok=True)
msa_keys = {kind: keyset(p) for kind, p in MSA.items()}
tpl_keys = keyset(TEMPLATE_DB)

etype, trim_offset = {}, {}
for row in csv.DictReader(open(FB / "meta" / "chain_seqid_map.tsv"), delimiter="\t"):
    etype[(row["target"], row["chain_id"])] = row["entity_type"]
    trim_offset[(row["target"], row["chain_id"])] = int(row["trim_offset"] or 0)


def glycan_bonds(t, names, etype, mode):
    """The ``bonded_atom_pairs`` of a target's glycans (branched chains), by ``mode``.

    ``none`` (the FoldBench protocol): no bonds. FoldBench's own Protenix preprocessing (algorithms/Protenix/preprocess.py) carries the
    sequence, the modifications and the ligand CCD codes of its AF3 input and reads no ``bondedAtomPairs``: a glycan reaches every method as
    a multi-CCD ligand with nothing joining its components, so giving this pipeline more would be giving it information the baselines lack.

    ``entity`` (for comparison, never the headline numbers): the links INSIDE each glycan from the entity's ``_pdbx_entity_branch_link`` of the
    ground-truth CIF (residue numbers = ``_pdbx_entity_branch_list.num``, the FASTA order) -- the chemical definition of the glycan, no
    coordinates. The attachment of a glycan to a protein is NOT added: it is not in these CIFs as a record and reading it off the coordinates
    would hand the model structural information from the answer.

    Returns ([[end, end], ...], note); an end is [chain index, residue number, atom name]."""
    kind = {i: etype.get((t, names[i])) for i in names}
    branched = [i for i in sorted(names, key=int) if kind[i] == "branched"]
    if not branched:
        return [], ""
    if mode == "none":
        return [], "glycans as unbonded multi-CCD ligands (FoldBench protocol)"
    try:
        block = pdbx.CIFFile.read(FB / "ground_truth" / f"{t}.cif").block

        def col(cat, c):
            return [str(x) for x in block[cat][c].as_array()]

        entity_of = dict(
            zip(col("struct_asym", "id"), col("struct_asym", "entity_id"), strict=True)
        )
        link = {
            c: col("pdbx_entity_branch_link", c)
            for c in (
                "entity_id",
                "entity_branch_list_num_1",
                "atom_id_1",
                "entity_branch_list_num_2",
                "atom_id_2",
            )
        }
        pairs = []
        for i in branched:
            for k, ent in enumerate(link["entity_id"]):
                if ent == entity_of.get(names[i]):
                    pairs.append(
                        [
                            [
                                int(i),
                                int(link["entity_branch_list_num_1"][k]),
                                link["atom_id_1"][k],
                            ],
                            [
                                int(i),
                                int(link["entity_branch_list_num_2"][k]),
                                link["atom_id_2"][k],
                            ],
                        ]
                    )
        return (
            pairs,
            f"{len(pairs)} glycan-internal links from the entity table (comparison mode)",
        )
    except Exception as exc:
        return [], f"glycan links unavailable ({type(exc).__name__}: {str(exc)[:80]})"


report = []
for t in targets:
    d = yaml.safe_load(open(FB / "inputs" / t / "data.yaml"))
    letters, names = d["chain_letters"], d["chain_names"]
    msa, msa_db, no_msa = {}, {}, []
    for idx, key in (d.get("a3m_key") or {}).items():
        kind = etype.get((t, names[idx]))
        if kind not in MSA:  # DNA (and anything that is not protein / RNA) takes no MSA
            no_msa.append(f"{names[idx]}:{kind}")
        elif (
            key not in msa_keys[kind]
        ):  # a key the database does not hold: query-only for this chain
            no_msa.append(f"{names[idx]}:key {key} not in the {kind} MSA database")
        else:
            msa[letters[idx]] = key
            msa_db[letters[idx]] = str(MSA[kind])
    offsets = d.get("template_offset") or {}
    template, dropped = {}, []
    for idx, key in (d.get("template") or {}).items():
        if etype.get((t, names[idx])) != "polypeptide(L)":
            dropped.append(f"{names[idx]}:{etype.get((t, names[idx]))}")
        elif key not in tpl_keys:
            dropped.append(f"{names[idx]}:key {key} not in the template database")
        else:
            template[idx] = key
    template_offset = {
        idx: int(offsets[idx]) for idx in template if int(offsets.get(idx, 0)) != 0
    }
    bonded, bond_note = glycan_bonds(t, names, etype, GLYCAN_MODE)
    has_glycan = any(etype.get((t, names[i])) == "branched" for i in names)
    spec = {
        "name": t,
        "chain_letters": letters,
        "fasta": {
            k: str(FB / "inputs" / t / Path(v).name) for k, v in d["fasta"].items()
        },  # the chain FASTA files of the release, wherever it lives
        "ccd_db": str(CCD_DB),
        "msa": msa,
        "msa_db": msa_db,
        "template_db": str(TEMPLATE_DB) if template else None,
        "template": template,
        "template_offset": template_offset,
        "bonded_atom_pairs": bonded,
        "allow_unbonded_branched": has_glycan and not bonded,
        "n_trunk_samples": nt,
        "n_diffusion_samples": nd,
        "diffusion_batch_size": nd,
        "save_trajectory": False,
        "template_dedup": False,
    }
    if not template:
        spec.pop("template_db")
        spec.pop("template")
        spec.pop("template_offset")
    if not bonded:
        spec.pop("bonded_atom_pairs")
    if not spec["allow_unbonded_branched"]:
        spec.pop("allow_unbonded_branched")
    (out / f"{t}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (out / f"{t}.chains.json").write_text(
        json.dumps([names[i] for i in sorted(names, key=int)])
    )  # FoldBench chain id by chain INDEX (copies of one sequence share a letter)
    report.append(
        (t, len(letters), len(msa), len(template), dropped, no_msa, bond_note)
    )
n_msa_drop = sum(1 for r in report for x in r[5] if "not in the" in x)
n_tpl_drop = sum(1 for r in report for x in r[4] if "not in the template" in x)
n_glycan = sum(
    1 for r in report if r[6] and "unavailable" not in r[6] and "unbonded" not in r[6]
)
n_glycan_unbonded = sum(1 for r in report if "unbonded" in r[6])
n_glycan_bad = sum(1 for r in report if "unavailable" in r[6])
for t, nch, nmsa, ntpl, dropped, no_msa, bond_note in [r for r in report if r[6]][
    :6
] + report[:3]:
    print(
        f"CONVERT {t}: {nch} chains, {nmsa} msa, {ntpl} templates"
        + (f", DROPPED template of {dropped}" if dropped else "")
        + (f", no MSA for {no_msa}" if no_msa else "")
        + (f", {bond_note}" if bond_note else "")
    )
print(
    f"CONVERT_SUMMARY {len(report)} targets -> {out} | MSA keys absent from the database: {n_msa_drop} chains | template keys absent: {n_tpl_drop} chains | glycan targets: {n_glycan} with internal links, {n_glycan_unbonded} as unbonded multi-CCD ligands, {n_glycan_bad} links unavailable"
)
