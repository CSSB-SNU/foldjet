"""F2: per-task accuracy of several runs (arms) -> the CSVs the paper scripts read.

    python collect.py --arm af3kit=EVAL/af3kit --arm af3_bf16=EVAL/af3_bf16 --arm af3_fp8=EVAL/af3_fp8 [--stamp af3_bf16=F1RUN/stamp.json ...] --out results/ [--prefix f2]

EVAL/<arm> = the --eval-dir of evaluate.py (holds scores_<name>.csv). Writes
  <prefix>_task_scores.csv   long: arm, task, metric, n, top1, oracle, plus the run stamp (FoldForge commit, engine, GPU, node, Slurm job id) when --stamp is given
  <prefix>_top1_wide.csv     rows = task, columns = arm: the shape of the paper's tables/foldbench_full_af3_accuracy.csv (',arm1,arm2,...')
  <prefix>_oracle_wide.csv   the same for the oracle (best of 25)
Values are percent (interfaces: DockQ >= 0.23 / ligand rmsd < 2 and lDDT-PLI > 0.8 success rate; monomers: mean lDDT x 100), rules of score_tasks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ORDER = ["interface_antibody_antigen", "interface_protein_protein", "interface_protein_peptide", "interface_protein_ligand", "interface_protein_dna",
         "interface_protein_rna", "monomer_protein", "monomer_rna", "monomer_dna"]


def load(arm: str, eval_dir: Path) -> pd.DataFrame:
    files = sorted(eval_dir.glob("scores_*.csv"))
    if not files:
        raise SystemExit(f"{arm}: no scores_*.csv under {eval_dir} (run evaluate.py first)")
    d = pd.read_csv(files[0])
    d.insert(0, "arm", arm)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=EVAL_DIR")
    ap.add_argument("--stamp", action="append", default=[], metavar="NAME=stamp.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="f2")
    a = ap.parse_args()
    stamps = {}
    for s in a.stamp:
        name, path = s.split("=", 1)
        st = json.loads(Path(path).read_text())
        stamps[name] = {"foldforge_commit": st.get("foldforge_commit"), "engine_version": st.get("engine_version"), "gpu": "; ".join(g.split(",")[1].strip() for g in st.get("gpus", [])[:1]),
                        "node": st.get("node"), "slurm_job_id": st.get("slurm_job_id"), "launch": st.get("launch"), "precision": st.get("precision")}
    frames = [load(*x.split("=", 1)) for x in a.arm]
    long = pd.concat(frames, ignore_index=True)
    for k in ("foldforge_commit", "engine_version", "gpu", "node", "slurm_job_id", "launch", "precision"):
        long[k] = long.arm.map(lambda n, k=k: stamps.get(n, {}).get(k))
    long["task"] = pd.Categorical(long.task, ORDER, ordered=True)
    long = long.sort_values(["arm", "task"]).reset_index(drop=True)
    a.out.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.out / f"{a.prefix}_task_scores.csv", index=False)
    arms = [x.split("=", 1)[0] for x in a.arm]
    for col in ("top1", "oracle"):
        wide = long.pivot(index="task", columns="arm", values=col).reindex(ORDER)[arms]
        wide.index.name = None
        wide.round(2).to_csv(a.out / f"{a.prefix}_{col}_wide.csv")
    print(long[["arm", "task", "n", "top1", "oracle"]].to_string(index=False))


if __name__ == "__main__":
    main()
