"""FoldBench task scores from the evaluator's raw per-sample tables: top-1 (ranking score) and oracle (best of the 25 samples), with the selection
rules of upstream `task_score_summary.py` (BEAM-Labs FoldBench), re-implemented because upstream's own `--metric_type best` aborts on the monomer
GDT-TS column (it reads `gdt_ts`, which is in neither of its metric lists). Same rules, line by line:

* rows are restricted to the targets of the task CSV (`find_overlap_sample`: (pdb_id, chain_id_1, chain_id_2) for interfaces, pdb_id for monomers);
* rows whose metric is NaN are dropped BEFORE the selection (`df[df[metric].notna()]`), and the rate is over the remaining selected rows;
* top-1 = per group the row with the largest `ranking_score` (`idxmax`: the first row in file order on a tie); oracle = per group the row with the best
  metric (`idxmax` for the metrics in `metric_max`, `idxmin` for `metric_min`);
* interfaces: group (pdb_id, interface_chain_id_1, interface_chain_id_2); success = dockq_score >= 0.23 (protein-DNA / protein-RNA: from the
  `*_dockqv2.csv` table, the others from `*_ost.csv`); protein-ligand: rows with rmsd and lddt-pli, the oracle takes the smallest rmsd, success = rmsd < 2.0
  and lddt-pli > 0.8; monomers: group pdb_id, value = mean lddt (reported x 100).

`python score_tasks.py RAW_DIR TARGETS_DIR [--csv OUT.csv]` prints / writes one row per task: task, metric, n, top1, oracle (percent).
`python score_tasks.py --check-upstream SUMMARY_rank.csv NAME RAW_DIR TARGETS_DIR` compares the top-1 column with upstream's own summary (a test of this file).
CPU only; no GPU, no network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

INTERFACES = ["interface_antibody_antigen", "interface_protein_protein", "interface_protein_peptide", "interface_protein_ligand", "interface_protein_dna", "interface_protein_rna"]
MONOMERS = ["monomer_protein", "monomer_rna", "monomer_dna"]
TASKS = INTERFACES + MONOMERS
METRIC_MAX = ("dockq_score", "lddt-lp", "lddt-pli", "gdt-ts", "tm-score", "lddt")
KEY_I = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]
#: task -> reported metric (the column names of upstream's summary and of the paper tables)
METRIC = {**{t: "dockq_score_success_rate" for t in INTERFACES}, "interface_protein_ligand": "rmsd_lddt-pli_success_rate", **{t: "lddt" for t in MONOMERS}}


def _rename(d: pd.DataFrame) -> pd.DataFrame:
    return d.rename(columns={"native_chain_id_1": "interface_chain_id_1", "native_chain_id_2": "interface_chain_id_2"})


def _overlap(d: pd.DataFrame, ref: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    keep = set(map(tuple, ref[key].values))
    return d[[tuple(r) in keep for r in d[key].values]]


def key_of(task: str) -> list[str]:
    return KEY_I if task in INTERFACES else ["pdb_id"]


def table(task: str, raw: Path, targets: Path) -> pd.DataFrame:
    """The per-sample rows of `task` (seed, sample, ranking_score and the metrics), restricted to the targets of the task CSV."""
    f = raw / (f"{task}_dockqv2.csv" if task in ("interface_protein_dna", "interface_protein_rna") else f"{task}_ost.csv")
    d = _rename(pd.read_csv(f))
    ref = _rename(pd.read_csv(targets / f"{task}.csv"))
    return _overlap(d, ref, key_of(task)).reset_index(drop=True)


def select(d: pd.DataFrame, task: str, how: str) -> pd.DataFrame:
    """One row per group: how = 'rank' (largest ranking_score) or 'best' (best metric), after dropping rows with a NaN metric (upstream)."""
    key = key_of(task)
    if task == "interface_protein_ligand":
        d = d[d["rmsd"].notna() & d["lddt-pli"].notna()]
        col = "rmsd"  # upstream selects by rmsd ('use rmsd to select best by default')
        pick_max = False
    elif task in MONOMERS:
        d = d[d["lddt"].notna()]
        col, pick_max = "lddt", True
    else:
        col, pick_max = "dockq_score", True
        d = d[d[col].notna()]
    if how == "rank":
        return d.loc[d.groupby(key)["ranking_score"].idxmax()]
    return d.loc[d.groupby(key)[col].idxmax() if pick_max else d.groupby(key)[col].idxmin()]


def success(rows: pd.DataFrame, task: str) -> pd.Series:
    """Boolean per selected row (interfaces), the per-target value (monomer lddt)."""
    if task == "interface_protein_ligand":
        return (rows["rmsd"] < 2.0) & (rows["lddt-pli"] > 0.8)
    if task in MONOMERS:
        return rows["lddt"]
    return rows["dockq_score"] >= 0.23


def value(rows: pd.DataFrame, task: str) -> float:
    s = success(rows, task)
    return 100.0 * float(s.mean())


def per_target(raw: Path, targets: Path, tasks: list[str] | None = None) -> pd.DataFrame:
    """Long table of the selected rows: task, group key, how (top1 | oracle), seed, sample, ranking_score and the success / lddt value."""
    out = []
    for task in tasks or TASKS:
        if not (raw / (f"{task}_dockqv2.csv" if task in ("interface_protein_dna", "interface_protein_rna") else f"{task}_ost.csv")).exists():
            continue
        d = table(task, raw, targets)
        for how, name in (("rank", "top1"), ("best", "oracle")):
            r = select(d, task, how)
            sel = r[[*key_of(task), "seed", "sample", "ranking_score"]].copy()
            sel["task"], sel["how"], sel["value"] = task, name, success(r, task).astype(float).values
            out.append(sel)
    return pd.concat(out, ignore_index=True)


def scores(raw: Path, targets: Path, tasks: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for task in tasks or TASKS:
        f = raw / (f"{task}_dockqv2.csv" if task in ("interface_protein_dna", "interface_protein_rna") else f"{task}_ost.csv")
        if not f.exists():
            continue
        d = table(task, raw, targets)
        top, best = select(d, task, "rank"), select(d, task, "best")
        rows.append({"task": task, "metric": METRIC[task], "n": len(top), "top1": round(value(top, task), 2), "oracle": round(value(best, task), 2)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path, nargs="?")
    ap.add_argument("targets", type=Path, nargs="?")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--check-upstream", nargs=2, metavar=("SUMMARY_RANK_CSV", "NAME"))
    a = ap.parse_args()
    t = scores(a.raw, a.targets)
    print(t.to_string(index=False))
    if a.csv:
        t.to_csv(a.csv, index=False)
    if a.check_upstream:
        up = pd.read_csv(a.check_upstream[0])
        col = a.check_upstream[1]
        bad = 0
        for _, r in t.iterrows():
            m = up[(up.target == r.task) & (up.metric == r.metric)]
            if m.empty:
                continue
            theirs = float(m[col].iloc[0]) * (100 if r.metric == "lddt" else 1)
            ok = abs(theirs - r.top1) < 0.02 if r.metric != "lddt" else abs(theirs - r.top1) < 1.0  # upstream rounds lddt to 2 decimals (x100 here)
            bad += not ok
            print(f"check {r.task:28s} ours {r.top1:7.2f} upstream {theirs:7.2f} {'OK' if ok else 'DIFFERENT'}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
