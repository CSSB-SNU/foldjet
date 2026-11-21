"""Aggregate step-1 antibody-antigen detail (engine 3.0.6 pair): where the FP8 deviation sits, and what the DockQ drop correlates with.
python abag_report.py OUT_DIR"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1 import EVAL306, V306, score_tasks  # noqa: E402

out = Path(sys.argv[1])
d = pd.read_csv(out / "abag_per_sample.csv")
att = pd.read_csv(out / "abag_target_attrs.csv")
task = "interface_antibody_antigen"
tb = {arm: score_tasks.table(task, ev / "raw", ev / "targets") for arm, ev in zip(("b", "f"), EVAL306)}
key = score_tasks.KEY_I
m = tb["b"].merge(tb["f"], on=[*key, "seed", "sample"], suffixes=("_b", "_f"))
m = m[m.dockq_score_b.notna() & m.dockq_score_f.notna()]
m["d_dq"] = m.dockq_score_f - m.dockq_score_b
m["d_rank"] = m.ranking_score_f - m.ranking_score_b
m["target"] = m.pdb_id
s = d.merge(m.groupby(["target", "seed", "sample"]).agg(d_dq=("d_dq", "mean"), dq_b=("dockq_score_b", "mean"), d_rank=("d_rank", "mean")).reset_index(),
            on=["target", "seed", "sample"], how="inner")
s["fab"] = np.where(s.n_ab_chains >= 2, "Fab (>=2 antibody chains)", "single-chain (nanobody / scFv)")
s["d_plddt_ab_if"] = s.plddt_ab_if_f - s.plddt_ab_if_b
s["d_plddt_ag"] = s.plddt_ag_f - s.plddt_ag_b

lines = ["== where the FP8 deviation sits (antibody-antigen, 3.0.6, per sample, after superposing FP8 on bf16 by the ANTIGEN), median / mean A"]
for c in ["dev_ab_if_on_ag", "dev_ab_nonif_on_ag", "dev_ag_if", "dev_ag_nonif", "ab_pose_shift", "rmsd_ab_internal", "dev_ab_if_internal", "dev_ab_nonif_internal"]:
    lines.append(f"  {c:24s} median {s[c].median():6.2f}  mean {s[c].mean():6.2f}")
lines.append("== by antibody format")
for g, x in s.groupby("fab"):
    lines.append(f"  {g:32s} n_samples {len(x):5d} targets {x.target.nunique():3d}  pose_shift med {x.ab_pose_shift.median():5.2f}  ab_internal med {x.rmsd_ab_internal.median():5.2f}"
                 f"  mean dDockQ {x.d_dq.mean():+.4f}  bf16 DockQ {x.dq_b.mean():.3f}")
lines.append("== Spearman correlation of per-sample dDockQ (FP8 - bf16) with")
for c in ["ab_pose_shift", "rmsd_ab_internal", "dev_ab_if_internal", "dev_ag_if", "d_plddt_ab_if", "d_plddt_ag", "d_rank", "dq_b", "n_res"]:
    lines.append(f"  {c:22s} rho {s[['d_dq', c]].corr(method='spearman').iloc[0, 1]:+.3f}")
# per target: mean dDockQ vs attributes
tg = s.groupby("target").agg(d_dq=("d_dq", "mean"), dq_b=("dq_b", "mean"), pose=("ab_pose_shift", "median"), n_res=("n_res", "first"),
                             n_ab=("n_ab_chains", "first")).reset_index().merge(att, on="target", how="left")
lines.append("== per target (113): Spearman of mean dDockQ with")
for c in ["pose", "n_res", "n_ab", "ab_template_frac", "ag_template_frac", "dq_b"]:
    lines.append(f"  {c:18s} rho {tg[['d_dq', c]].corr(method='spearman').iloc[0, 1]:+.3f}")
lines.append("== dDockQ by bf16 DockQ bin (per sample): is the loss on near-threshold / docked samples?")
s["bin"] = pd.cut(s.dq_b, [-0.01, 0.1, 0.23, 0.49, 0.8, 1.01])
for b, x in s.groupby("bin", observed=True):
    lines.append(f"  bf16 DockQ {str(b):14s} n {len(x):5d}  mean dDockQ {x.d_dq.mean():+.4f}  pose shift med {x.ab_pose_shift.median():5.2f}  success flips +{int(((x.dq_b < .23) & (x.dq_b + x.d_dq >= .23)).sum())} / -{int(((x.dq_b >= .23) & (x.dq_b + x.d_dq < .23)).sum())}")
txt = "\n".join(lines)
(out / "abag_report.txt").write_text(txt + "\n")
s.to_csv(out / "abag_per_sample_joined.csv", index=False)
tg.to_csv(out / "abag_per_target.csv", index=False)
print(txt)
