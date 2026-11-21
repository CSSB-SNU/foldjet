"""F2 task scores of a whole-FoldBench run in which a subset of targets was re-predicted (engine 3.0.6, the 32 targets > 2048 AF3 tokens whose MSA
pair-weighted averaging fell back to Triton on engine 3.0.1): the base run's per-sample rows with the subset's targets replaced by the re-run's rows,
then the same top-1 / oracle rules (score_tasks.py). Also prints the subset alone (base vs re-run) so the change is visible.

  python splice.py --base-raw B/raw --base-targets B/targets --patch-raw P/raw --patch-targets P/targets --subset gt2048.txt --label af3_bf16 [--csv OUT]
CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import score_tasks as st


def spliced(task: str, a) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = set(Path(a.subset).read_text().split())
    base = st.table(task, a.base_raw, a.base_targets)
    f = a.patch_raw / (f"{task}_dockqv2.csv" if task in ("interface_protein_dna", "interface_protein_rna") else f"{task}_ost.csv")
    if not f.exists():  # no re-run target belongs to this task
        return base, base.iloc[0:0], base.iloc[0:0]
    patch = st.table(task, a.patch_raw, a.patch_targets)
    patch = patch[patch.pdb_id.isin(sub)]
    merged = pd.concat([base[~base.pdb_id.isin(set(patch.pdb_id))], patch], ignore_index=True)
    return merged, base[base.pdb_id.isin(set(patch.pdb_id))], patch


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("base-raw", "base-targets", "patch-raw", "patch-targets", "subset"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = []
    for task in st.TASKS:
        f = a.base_raw / (f"{task}_dockqv2.csv" if task in ("interface_protein_dna", "interface_protein_rna") else f"{task}_ost.csv")
        if not f.exists():
            continue
        merged, b_sub, p_sub = spliced(task, a)
        r = {"label": a.label, "task": task, "metric": st.METRIC[task]}
        full_base = st.table(task, a.base_raw, a.base_targets)
        for name, d in (("base", full_base), ("spliced", merged), ("subset_base", b_sub), ("subset_patch", p_sub)):
            if d.empty:
                r[f"n_{name}"], r[f"top1_{name}"], r[f"oracle_{name}"] = 0, float("nan"), float("nan")
                continue
            top, best = st.select(d, task, "rank"), st.select(d, task, "best")
            r[f"n_{name}"], r[f"top1_{name}"], r[f"oracle_{name}"] = len(top), round(st.value(top, task), 2), round(st.value(best, task), 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if a.csv:
        t.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
