import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abl_report import raw_of, score_tasks
A = Path(sys.argv[1]); task = "interface_antibody_antigen"; key = score_tasks.KEY_I
ref = raw_of("bf16"); tb = score_tasks.table(task, ref / "raw", ref / "targets")[[*key, "sample", "dockq_score"]].dropna()
rng = np.random.default_rng(0); out = []
for cfg in sorted(p.name for p in A.iterdir() if (p / "eval").is_dir() and p.name != "bf16"):
    ev = raw_of(cfg); tc = score_tasks.table(task, ev / "raw", ev / "targets")[[*key, "sample", "dockq_score"]].dropna()
    m = tb.merge(tc, on=[*key, "sample"], suffixes=("_b", "_c")); m["d"] = m.dockq_score_c - m.dockq_score_b
    m["down"] = (m.dockq_score_b >= .23) & (m.dockq_score_c < .23); m["up"] = (m.dockq_score_b < .23) & (m.dockq_score_c >= .23)
    g = m.groupby(key).agg(d=("d", "mean"), net=("up", "sum")); g["net"] = m.groupby(key).up.sum() - m.groupby(key).down.sum()
    v = g.d.to_numpy(); nf = g.net.to_numpy(); n = len(v)
    bs = [v[rng.integers(0, n, n)].mean() for _ in range(5000)]; bn = [nf[rng.integers(0, n, n)].sum() for _ in range(5000)]
    out.append({"cfg": cfg, "mean_dDockQ": v.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "net_flips(up-down)": int(nf.sum()), "net_lo": np.percentile(bn, 2.5), "net_hi": np.percentile(bn, 97.5)})
r = pd.DataFrame(out).sort_values("mean_dDockQ"); r.to_csv(A / "abl_ci_abag.csv", index=False); print(r.round(4).to_string(index=False))
