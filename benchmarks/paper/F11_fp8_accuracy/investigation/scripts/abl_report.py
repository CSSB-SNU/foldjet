"""Route ablation report: per config, antibody-antigen and protein-protein, seed 42 x 5 samples vs the bf16 config of the same ablation set.
python abl_report.py ABL_DIR"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1 import kabsch_rmsd, read_res, score_tasks, xyz  # noqa: E402

A = Path(sys.argv[1])
TASKS = {"interface_antibody_antigen": "ab-ag", "interface_protein_protein": "prot-prot"}
K = 5


def raw_of(cfg: str) -> Path | None:
    for p in (A / cfg / "eval" / f"abl_{cfg}", A / cfg / "eval"):
        if (p / "raw").is_dir():
            return p
    return None


def rmsd_rows(args):
    cfg, t = args
    out = []
    for k in range(K):
        fb = A / "bf16" / "pred" / t / f"{t}_seed42_sample{k}_pred.cif"
        fc = A / cfg / "pred" / t / f"{t}_seed42_sample{k}_pred.cif"
        if not (fb.exists() and fc.exists()):
            continue
        b, c = xyz(read_res(fb)), xyz(read_res(fc))
        if len(b) == len(c) and len(b) >= 3:
            out.append({"cfg": cfg, "target": t, "sample": k, "d_cfg_bf16": kabsch_rmsd(c, b)})
    return out


def main() -> None:
    cfgs = sorted(p.name for p in A.iterdir() if (p / "pred").is_dir())
    targets = (A / "targets.txt").read_text().split()
    with Pool(16) as p:
        rr = p.map(rmsd_rows, [(c, t) for c in cfgs if c != "bf16" for t in targets], chunksize=8)
    rm = pd.DataFrame([r for x in rr for r in x])
    rm.to_csv(A / "abl_rmsd.csv", index=False)
    ref = raw_of("bf16")
    rows = []
    for task, short in TASKS.items():
        key = score_tasks.KEY_I
        tb = score_tasks.table(task, ref / "raw", ref / "targets")[[*key, "seed", "sample", "dockq_score", "ranking_score"]].dropna(subset=["dockq_score"])
        for cfg in cfgs:
            ev = raw_of(cfg)
            if ev is None:
                continue
            tc = score_tasks.table(task, ev / "raw", ev / "targets")[[*key, "seed", "sample", "dockq_score", "ranking_score"]].dropna(subset=["dockq_score"])
            m = tb.merge(tc, on=[*key, "seed", "sample"], suffixes=("_b", "_c"))
            if m.empty:
                continue
            g = m.groupby(key)

            def top1(col_dq, col_rank):
                return float(g.apply(lambda x: x[col_dq].values[x[col_rank].values.argmax()] >= 0.23).mean() * 100)

            def oracle(col):
                return float(g[col].max().ge(0.23).mean() * 100)

            tset = set(m.pdb_id)
            dr = rm[(rm.cfg == cfg) & rm.target.isin(tset)].d_cfg_bf16 if cfg != "bf16" else pd.Series([0.0])
            flips_down = int(((m.dockq_score_b >= .23) & (m.dockq_score_c < .23)).sum())
            flips_up = int(((m.dockq_score_b < .23) & (m.dockq_score_c >= .23)).sum())
            rows.append({"task": short, "cfg": cfg, "groups": g.ngroups, "median_rmsd_to_bf16": float(dr.median()) if len(dr) else np.nan,
                         "mean_dDockQ": float((m.dockq_score_c - m.dockq_score_b).mean()), "mean_DockQ": float(m.dockq_score_c.mean()),
                         "top1_%": top1("dockq_score_c", "ranking_score_c"), "oracle_%": oracle("dockq_score_c"),
                         "sample_success_%": float((m.dockq_score_c >= .23).mean() * 100), "flips_down": flips_down, "flips_up": flips_up})
    r = pd.DataFrame(rows)
    r.to_csv(A / "abl_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
