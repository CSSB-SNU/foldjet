"""F5 -> CSV: the one-to-one sample comparison (pair.py output) and the scores against the experimental structures (FoldBench evaluator).

    python collect.py --card h100 --prefix OUTDIR/F5_h100 [--pairs PAIRS.csv] [--eval-root EVAL_ROOT] [--engine-version ...] [--commit ...]

Writes (each row carries card, engine_version, foldforge_commit):
  <prefix>_pairs_summary.csv   per (arm, model): n sample pairs, median / p90 RMSD (Å), median |ΔpLDDT|, median |ΔPAE| (Å); arm "baseline" =
                               PyTorch sample k vs PyTorch sample l (k < l), the sampling spread the kernel arms are judged against
  <prefix>_gt.csv              per (model, task, interface or target): mean over the 5 samples of the score against the experimental structure
                               per arm (monomers lDDT, interfaces DockQ, protein-DNA / RNA DockQ v2, protein-ligand lDDT-PLI), PyTorch's s.d.
                               over its 5 samples, and arm - torch
  <prefix>_gt_summary.csv      per arm (and per kind): n, median |arm - torch|, mean signed difference, fraction within PyTorch's 1 s.d., r
EVAL_ROOT/<arm>__<model>/eval/<arm>__<model>/raw/*.csv are the evaluator's per-sample tables (eval.sbatch). Only (target, interface) keys with
all 5 samples scored count. CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KI = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]


def gt_scores(eval_root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in eval_root.iterdir() if p.is_dir() and "__" in p.name):
        arm, model = d.name.split("__", 1)
        raw = d / "eval" / d.name / "raw"
        for f in sorted(raw.glob("*.csv")):
            task = f.name.rsplit("_", 1)[0]
            if task in ("interface_protein_dna", "interface_protein_rna") and f.name.endswith("_ost.csv"):
                continue  # protein-nucleic-acid interfaces are scored by DockQ v2 (the *_dockqv2.csv tables)
            t = pd.read_csv(f).rename(columns={"native_chain_id_1": "interface_chain_id_1", "native_chain_id_2": "interface_chain_id_2"})
            if task.startswith("monomer"):
                key, col, kind = ["pdb_id"], "lddt", "monomer"
            elif task == "interface_protein_ligand":
                key, col, kind = [*KI, "ligand_id"], "lddt-pli", "ligand"
            else:
                key, col, kind = KI, "dockq_score", "interface"
            if col not in t.columns:
                continue
            m = t.groupby(key)[col].agg(["mean", "std", "size"]).reset_index()
            m = m[m["size"] == 5]
            for r in m.itertuples(index=False):
                rows.append({"arm": arm, "model": model, "task": task, "kind": kind, "key": "|".join(map(str, r[: len(key)])),
                             "v": r[-3], "sd": r[-2]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--eval-root", type=Path)
    ap.add_argument("--engine-version", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--reference", default="torch")
    args = ap.parse_args()
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    stamp = {"id": "F5", "card": args.card, "engine_version": args.engine_version, "foldforge_commit": args.commit}
    if args.pairs:
        df = pd.read_csv(args.pairs)
        s = df.groupby(["arm", "model"]).agg(
            n=("rmsd", "size"), rmsd_median=("rmsd", "median"), rmsd_p90=("rmsd", lambda x: x.quantile(0.9)),
            plddt_abs_median=("plddt_abs", "median"), pae_abs_median=("pae_abs", "median"),
        ).reset_index()
        allm = df.groupby("arm").agg(n=("rmsd", "size"), rmsd_median=("rmsd", "median"), rmsd_p90=("rmsd", lambda x: x.quantile(0.9)),
                                     plddt_abs_median=("plddt_abs", "median"), pae_abs_median=("pae_abs", "median")).reset_index()
        allm["model"] = "ALL"
        s = pd.concat([s, allm], ignore_index=True)
        for k, v in reversed(list(stamp.items())):
            s.insert(0, k, v)
        s.round(4).to_csv(f"{args.prefix}_pairs_summary.csv", index=False)
        print(allm.round(4).to_string(index=False))
    if args.eval_root:
        g = gt_scores(args.eval_root)
        wide = g.pivot_table(index=["model", "task", "kind", "key"], columns="arm", values="v").reset_index()
        ref_sd = g[g.arm == args.reference].set_index(["model", "task", "kind", "key"]).sd.rename(f"{args.reference}_sd")
        wide = wide.join(ref_sd, on=["model", "task", "kind", "key"])
        arms = [a for a in sorted(g.arm.unique()) if a != args.reference]
        for a in arms:
            wide[f"{a}_minus_{args.reference}"] = wide[a] - wide[args.reference]
        for k, v in reversed(list(stamp.items())):
            wide.insert(0, k, v)
        wide.round(4).to_csv(f"{args.prefix}_gt.csv", index=False)
        summ = []
        for a in arms:
            h = wide.dropna(subset=[args.reference, a])
            for kind, x in [("all", h), *h.groupby("kind")]:
                dl = x[a] - x[args.reference]
                summ.append({**stamp, "arm": a, "kind": kind, "n": len(x), "median_abs_diff": dl.abs().median(), "mean_signed_diff": dl.mean(),
                             "within_ref_sd": float((dl.abs() <= x[f"{args.reference}_sd"]).mean()) if len(x) else np.nan,
                             "r": float(np.corrcoef(x[args.reference], x[a])[0, 1]) if len(x) > 2 else np.nan})
        ss = pd.DataFrame(summ).round(4)
        ss.to_csv(f"{args.prefix}_gt_summary.csv", index=False)
        print(ss.to_string(index=False))


if __name__ == "__main__":
    main()
