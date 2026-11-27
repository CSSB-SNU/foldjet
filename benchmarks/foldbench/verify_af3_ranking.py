"""Check foldjet.eval.af3_ranking against AlphaFold3's own functions on saved FoldBench predictions.

Reference CSV (AlphaFold3's confidences.fraction_disordered / has_clash, run in the af3_jax kit's stock environment on gemmi-cleaned copies):
columns target,seed,sample,ptm_iptm,fraction_disordered,has_clash,ranking_score_af3,error.

For every reference row, the saved sample CIF (chains renamed to FoldBench ids by fb_io.rename_chains) is
  (a) given back FoldJet's own chain ids (the k-th chain in order of appearance -> AlphaFold3's k-th id A, B, ..., in every *asym_id /
      *strand_id column), i.e. the CIF FoldJet's Structure wrote, and ranked by foldjet.eval.af3_ranking.ranking_terms (production path);
  (b) renamed again with the fixed fb_io.rename_chains and read by alphafold3.structure.from_mmcif (the written file must parse).
Agreement: fraction_disordered within 1e-9, has_clash equal, top-1 (seed, sample) per target equal.

    python verify_af3_ranking.py REF_CSV PRED_DIR OUT_CSV [WORKERS]
"""

from __future__ import annotations

import csv
import io
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))


def to_foldjet_ids(cif: str) -> tuple[str, list[str]]:
    """Undo the FoldBench chain renaming of a saved sample: (CIF with FoldJet's chain ids, the FoldBench names in order)."""
    from alphafold3.structure import mmcif as af3_mmcif

    f = pdbx.CIFFile.read(io.StringIO(cif))
    block = f.block
    new = []
    for c in block["atom_site"]["label_asym_id"].as_array():
        if c not in new:
            new.append(str(c))
    back = {n: af3_mmcif.int_id_to_str_id(k + 1) for k, n in enumerate(new)}
    # The old writer renamed atom_site but only some of the per-entity tables (e.g. pdbx_poly_seq_scheme kept FoldJet's ids, so a value can
    # name a different chain there). Those tables are rebuilt from atom_site instead: within each entity, the k-th distinct id met in the
    # table is the entity's k-th chain in atom_site order.
    a = block["atom_site"]
    by_entity: dict[str, list[str]] = defaultdict(list)
    for c, e in zip(a["label_asym_id"].as_array(), a["label_entity_id"].as_array(), strict=True):
        if str(c) not in by_entity[str(e)]:
            by_entity[str(e)].append(str(c))
    per_entity = {
        "struct_asym": ("entity_id", ["id"]),
        "pdbx_poly_seq_scheme": ("entity_id", ["asym_id", "pdb_strand_id"]),
        "pdbx_nonpoly_scheme": ("entity_id", ["asym_id", "pdb_strand_id"]),
        "pdbx_branch_scheme": ("entity_id", ["asym_id", "pdb_asym_id"]),
    }
    for cat_name, (ecol, cols) in per_entity.items():
        if cat_name not in block:
            continue
        cat = block[cat_name]
        ents = [str(e) for e in cat[ecol].as_array()]
        for col in cols:
            if col not in cat:
                continue
            seen: dict[str, list[str]] = defaultdict(list)
            out = []
            for v, e in zip(cat[col].as_array(), ents, strict=True):
                if str(v) not in seen[e]:
                    seen[e].append(str(v))
                chains = by_entity.get(e, [])
                k = seen[e].index(str(v))
                out.append(back[chains[k]] if k < len(chains) else str(v))
            cat[col] = pdbx.CIFColumn(np.array(out))
    if "entity_poly" in block:
        ep = block["entity_poly"]
        if "pdbx_strand_id" in ep:
            ep["pdbx_strand_id"] = pdbx.CIFColumn(
                np.array([",".join(back[c] for c in by_entity.get(str(e), [])) for e in ep["entity_id"].as_array()])
            )
    for cat_name in list(block.keys()):
        if cat_name in per_entity or cat_name == "entity_poly":
            continue
        cat = block[cat_name]
        for col in list(cat.keys()):
            if col.endswith(("asym_id", "strand_id")):
                vals = cat[col].as_array()
                cat[col] = pdbx.CIFColumn(
                    np.array(
                        [
                            ",".join(back.get(x, x) for x in str(v).split(","))
                            for v in vals
                        ]
                    )
                )
    buf = io.StringIO()
    f.write(buf)
    return buf.getvalue(), new


def one(job):
    target, seed, sample, path, ref_fd, ref_clash = job
    import fb_io
    from alphafold3 import structure

    from foldjet.eval import af3_ranking as R

    out = {
        "target": target,
        "seed": seed,
        "sample": sample,
        "ref_fd": ref_fd,
        "ref_clash": ref_clash,
    }
    try:
        orig, names = to_foldjet_ids(Path(path).read_text())
        st = structure.from_mmcif(orig)
        out["fd"] = R.fraction_disordered(st)
        out["clash"] = int(R.has_clash(st))
        out["err"] = ""
    except Exception as e:
        out.update(fd=float("nan"), clash=-1, err=f"{type(e).__name__}: {str(e)[:150]}")
    try:  # the fixed renaming: the written file must be readable by AlphaFold3's parser and give the same terms
        st2 = structure.from_mmcif(fb_io.rename_chains(orig, names))
        out["fd_renamed"] = R.fraction_disordered(st2)
        out["clash_renamed"] = int(R.has_clash(st2))
        out["err_renamed"] = ""
    except Exception as e:
        out.update(
            fd_renamed=float("nan"),
            clash_renamed=-1,
            err_renamed=f"{type(e).__name__}: {str(e)[:150]}",
        )
    return out


def main() -> int:
    ref_csv, pred, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    ref = list(csv.DictReader(open(ref_csv)))
    jobs = []
    for r in ref:
        p = (
            pred
            / r["target"]
            / f"{r['target']}_seed{r['seed']}_sample{r['sample']}_pred.cif"
        )
        jobs.append(
            (
                r["target"],
                int(r["seed"]),
                int(r["sample"]),
                str(p),
                float(r["fraction_disordered"]),
                int(r["has_clash"]),
            )
        )
    print(f"{len(jobs)} samples from {ref_csv}", flush=True)
    with mp.Pool(workers) as pool:
        res = list(pool.imap_unordered(one, jobs, chunksize=8))
    cols = [
        "target",
        "seed",
        "sample",
        "ref_fd",
        "fd",
        "fd_renamed",
        "ref_clash",
        "clash",
        "clash_renamed",
        "err",
        "err_renamed",
    ]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for o in sorted(res, key=lambda o: (o["target"], o["seed"], o["sample"])):
            w.writerow({k: o.get(k) for k in cols})
    refnan = sum(1 for o in res if o["ref_fd"] != o["ref_fd"])
    ok = [o for o in res if not o["err"] and o["ref_fd"] == o["ref_fd"]]
    fd_eq = sum(1 for o in ok if abs(o["fd"] - o["ref_fd"]) <= 1e-9)
    cl_eq = sum(1 for o in ok if o["clash"] == o["ref_clash"])
    ren_ok = [o for o in res if not o["err_renamed"] and not o["err"]]
    ren_eq = sum(
        1
        for o in ren_ok
        if abs(o["fd_renamed"] - o["fd"]) <= 1e-9 and o["clash_renamed"] == o["clash"]
    )
    # top-1 per target: reference score vs this score (same base ptm_iptm from the reference row)
    base = {
        (r["target"], int(r["seed"]), int(r["sample"])): float(r["ptm_iptm"])
        for r in ref
    }
    refscore = {
        (r["target"], int(r["seed"]), int(r["sample"])): float(r["ranking_score_af3"])
        for r in ref
    }
    mine = {
        (o["target"], o["seed"], o["sample"]): base[
            (o["target"], o["seed"], o["sample"])
        ]
        + 0.5 * o["fd"]
        - 100.0 * o["clash"]
        for o in ok
    }
    by_t = defaultdict(list)
    for k in mine:
        by_t[k[0]].append(k)
    top_same = sum(
        1
        for t, ks in by_t.items()
        if max(ks, key=lambda k: refscore[k]) == max(ks, key=lambda k: mine[k])
    )
    print(
        f"computed {len(ok)} / {len(res)} (errors {sum(1 for o in res if o['err'])}; reference NaN {refnan})"
    )
    print(
        f"fraction_disordered equal (|d| <= 1e-9): {fd_eq} / {len(ok)};  max |d| = {max((abs(o['fd'] - o['ref_fd']) for o in ok), default=0):.3g}"
    )
    print(
        f"has_clash equal: {cl_eq} / {len(ok)};  clash=True here {sum(o['clash'] for o in ok)}, in the reference {sum(o['ref_clash'] for o in ok)}"
    )
    print(f"top-1 equal: {top_same} / {len(by_t)} targets")
    print(
        f"renamed file parses: {len(ren_ok)} / {len(res)}; same terms as the FoldJet ids: {ren_eq} / {len(ren_ok)}"
    )
    for o in [o for o in res if o["err"]][:5] + [o for o in res if o["err_renamed"]][
        :5
    ]:
        print(
            "ERR", o["target"], o["seed"], o["sample"], o["err"], "|", o["err_renamed"]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
