"""Per target (whole FoldBench, engine 3.0.1 pair): bf16 confidence (mean ranking score = 0.8 ipTM + 0.2 pTM) vs FP8 perturbation d_prec, per task."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1 import V301, FB, score_tasks
out = Path(sys.argv[1])
w = pd.read_csv(out / "wholefb_dprec_dseed.csv").dropna(subset=["median_d_prec"])
def rank(t):
    f = V301[0] / t / "ranking.csv"
    return pd.read_csv(f).ranking_score.mean() if f.exists() else np.nan
w["bf16_rank"] = [rank(t) for t in w.target]
rows = []
for task in score_tasks.TASKS:
    ids = set(pd.read_csv(FB / "targets" / f"{task}.csv").pdb_id)
    x = w[w.target.isin(ids)]
    lo = x[x.bf16_rank < x.bf16_rank.median()]; hi = x[x.bf16_rank >= x.bf16_rank.median()]
    rows.append({"task": task, "targets": len(x), "median_bf16_rank": x.bf16_rank.median(), "median_d_prec": x.median_d_prec.median(),
                 "rho_rank_dprec": x[["bf16_rank", "median_d_prec"]].corr(method="spearman").iloc[0, 1],
                 "d_prec_low_conf_half": lo.median_d_prec.median(), "d_prec_high_conf_half": hi.median_d_prec.median()})
r = pd.DataFrame(rows); r.to_csv(out / "conf_vs_dprec.csv", index=False); print(r.round(3).to_string())
# pooled: does d_prec depend on task once confidence is matched? bin by confidence
w["task"] = ""
tm = {}
for task in score_tasks.TASKS:
    for t in pd.read_csv(FB / "targets" / f"{task}.csv").pdb_id.unique(): tm.setdefault(t, task)
w["task"] = w.target.map(tm)
w["conf_bin"] = pd.cut(w.bf16_rank, [0, .5, .7, .8, .9, 1.0])
print((w.groupby(["conf_bin", "task"], observed=True).median_d_prec.median().unstack().round(2)).to_string())
