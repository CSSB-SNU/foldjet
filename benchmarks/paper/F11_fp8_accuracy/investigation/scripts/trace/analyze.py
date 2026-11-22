"""Aggregate F11 trace outputs (out/<t>/{trace,sched}.jsonl, meta.json) + down.csv -> report tables (stdout) and CSVs.
python analyze.py OUT_DIR"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(sys.argv[1])
GROUP = {"8f5n-assembly1": "loss", "8a44-assembly1": "loss", "8be2-assembly1": "loss", "8w84-assembly1": "loss",
         "7xld-assembly1": "noloss", "8aom-assembly1": "noloss", "8c3l-assembly1": "noloss", "8e0e-assembly1": "noloss",
         "7spq-assembly1": "pp", "8a51-assembly1": "pp", "8upb-assembly1": "pp"}
REG = ["inter", "ab", "ag", "all"]
SREG = ["ab", "ag", "all"]
ORDER = ["tmo", "tmi", "tas", "tae", "tr"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


def load(name):
    rows = []
    for t in sorted(GROUP):
        f = O / t / name
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            r["target"] = t
            rows.append(r)
    return pd.DataFrame(rows)


def blk(pt):
    if pt.startswith("b"):
        b, m = pt[1:].split(".")
        return int(b), m
    return (-1, pt)


tr = load("trace.jsonl")
sc = load("sched.jsonl")
meta = {t: json.loads((O / t / "meta.json").read_text()) for t in GROUP if (O / t / "meta.json").exists()}
tr["grp"] = tr.target.map(GROUP)
tr[["block", "mod"]] = tr.pt.apply(lambda p: pd.Series(blk(p)))
pair = tr[tr.kind == "pair"].copy()
for k in REG:
    pair[f"rel_{k}"] = pair[f"err_{k}"] / pair[f"ref_{k}"]
single = tr[tr.kind == "single"].copy()
for k in SREG:
    single[f"rel_{k}"] = single[f"err_{k}"] / single[f"ref_{k}"]
inj = tr[tr.kind == "inj"].copy()

print("== meta")
for t, m in meta.items():
    print(f"  {t:16s} {GROUP[t]:6s} n_tok {m['n_tok']} real {m['n_real']} ab {m['n_ab']} ag {m['n_ag']} S {m['S']} passes {m['passes']} "
          f"extra {len(m['extra'])} wall {m['wall_s']:.0f}s")

# 1. trajectory: block outputs (tr) per pass, inter-chain relative error
bo = pair[pair["mod"] == "tr"]
bo = bo[bo.block >= 0]
print("\n== 1. inter-chain relative error of the block output (pair after transition_pair), % ; rows pass, cols block")
for t in GROUP:
    d = bo[bo.target == t]
    if d.empty:
        continue
    piv = d.pivot_table(index="pass", columns="block", values="rel_inter") * 100
    cols = [c for c in [0, 1, 2, 4, 8, 16, 24, 32, 40, 47] if c in piv.columns]
    print(f"-- {t} ({GROUP[t]})")
    print(piv[cols].round(2).to_string())
bo.to_csv(O / "traj_blockout.csv", index=False)

print("\n== 1b. relative error % at pass-start (block-0 input) and last block, all regions, per group (median over targets)")
inp = pair[pair.pt == "in"]
last = bo[bo.block == bo.block.max()]
for nm, d in (("block0-in", inp), ("block47-out", last)):
    g = d.groupby(["grp", "pass"])[[f"rel_{k}" for k in REG]].median() * 100
    print(f"-- {nm}")
    print(g.round(2).unstack(0).to_string())

print("\n== 1c. single relative error % (last block single_transition output) per pass, median per group")
ls = single[single.pt == f"b{int(bo.block.max())}.st"]
print((ls.groupby(["grp", "pass"])[[f"rel_{k}" for k in SREG]].median() * 100).round(2).unstack(0).to_string())

# 2. onset
print("\n== 2. onset: first (pass, block) where the inter-chain block-output relative error exceeds X %")
on = []
for t in GROUP:
    d = bo[bo.target == t].sort_values(["pass", "block"])
    if d.empty:
        continue
    r = {"target": t, "grp": GROUP[t]}
    for x in (1, 2, 5, 10, 20):
        hit = d[d.rel_inter * 100 > x]
        r[f">{x}%"] = f"p{int(hit.iloc[0]['pass'])}b{int(hit.iloc[0].block)}" if len(hit) else "-"
    r["p0b0"] = round(float(d.iloc[0].rel_inter * 100), 2)
    r["final"] = round(float(d.iloc[-1].rel_inter * 100), 2)
    on.append(r)
print(pd.DataFrame(on).to_string(index=False))

# 3. injection vs amplification: per module growth of the error norm, per region
print("\n== 3. per module: median ratio |e_out| / |e_in| (error norm after / before the module) per region; FP8 run")
seq = pair.sort_values(["target", "pass"]).copy()
gr = []
for t, d in pair.groupby("target"):
    d = d.copy()
    d["ord"] = d.apply(lambda r: (r["pass"], -2 if r.pt.startswith("x.") else (-1 if r.pt == "in" else r.block),
                                  0 if r.pt.startswith("x.") else (ORDER.index(r["mod"]) if r["mod"] in ORDER else 0)), axis=1)
    d = d.sort_values("ord")
    for k in REG:
        d[f"g_{k}"] = d[f"err_{k}"] / d[f"err_{k}"].shift(1)
    gr.append(d)
gr = pd.concat(gr)
gr["mtype"] = gr["mod"].where(gr.block >= 0, gr.pt.str.replace(r"x\..*", "msa_tr", regex=True))
gtab = gr[gr["pass"] >= 0].groupby(["grp", "mtype"])[[f"g_{k}" for k in REG]].median()
print(gtab.round(3).to_string())
print("-- same, pass 0 only (from the first injection on)")
print(gr[gr["pass"] == 0].groupby(["grp", "mtype"])[[f"g_{k}" for k in REG]].median().round(3).to_string())

print("\n== 3b. FP8 Transition calls: injection |f8(x)-fb(x)| vs incoming error |e_in| and propagated part, inter / all, median over blocks")
inj["grp"] = inj.target.map(GROUP)
inj["isx"] = inj.pt.str.startswith("x.")
for k in ("inter", "ab", "ag", "all"):
    inj[f"inj_over_x_{k}"] = inj[f"inj_{k}"] / inj[f"x_{k}"]
    inj[f"inj_over_u_{k}"] = inj[f"inj_{k}"] / inj[f"u_{k}"]
    inj[f"u_over_x_{k}"] = inj[f"u_{k}"] / inj[f"x_{k}"]
    if f"ein_{k}" in inj:
        inj[f"inj_over_ein_{k}"] = inj[f"inj_{k}"] / inj[f"ein_{k}"]
        inj[f"prop_over_ein_{k}"] = inj[f"prop_{k}"] / inj[f"ein_{k}"]
print("check (kernel on slice == kernel in model) max |diff|:", float(inj.check_maxdiff.max()))
cols = ["inj_over_x_inter", "inj_over_x_all", "inj_over_u_inter", "inj_over_u_all", "u_over_x_inter", "u_over_x_all",
        "inj_over_ein_inter", "inj_over_ein_all"]
print(inj[~inj.isx].groupby(["grp", "pass"])[cols].median().round(4).unstack(0).iloc[:, :].to_string())

print("\n== 3c. share of the error growth by FP8 Transition injection (sum over calls of |inj|^2) vs total error energy at the end, per target")
rows = []
for t, d in inj.groupby("target"):
    fin = bo[(bo.target == t)].sort_values(["pass", "block"]).iloc[-1]
    rows.append({"target": t, "grp": GROUP[t], "sum_inj2_inter": float((d.inj_inter ** 2).sum()), "final_err2_inter": float(fin.err_inter ** 2),
                 "ratio_final/sum_inj (amplification)": float(fin.err_inter ** 2 / (d.inj_inter ** 2).sum()),
                 "ratio_all": float(fin.err_all ** 2 / (d.inj_all ** 2).sum())})
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n== 4. magnitudes (bf16 ref, block output, median over blocks and passes): pair rms per region; Transition update / x; h vs e4m3 scale")
mag = bo.groupby("grp")[[f"ref_{k}" for k in REG]].median()
print(mag.round(1).to_string())
hc = []
for k in REG:
    hc += [f"h_rms_{k}", f"h_sub_{k}", f"h_zero_{k}", f"h_qrel_{k}", f"h_qrel_row_{k}", f"xn_qrel_{k}", f"h_amax_scaled_{k}"]
print(inj[~inj.isx].groupby("grp")[hc].median().T.round(4).to_string())
print("sh per group (median):", inj.groupby("grp").sh.median().to_dict(), " sx:", inj.groupby("grp").sx.median().to_dict())

# 5. schedules
print("\n== 5. schedules: last-pass trunk output (block-47 pair) relative error %, per region; + single")
sc["grp"] = sc.target.map(GROUP)
sp = sc[(sc.kind == "pair") & (sc.pt == f"b{int(bo.block.max())}.tr")].copy()
lastp = sp["pass"].max()
sp = sp[sp["pass"] == lastp]
for k in REG:
    sp[f"rel_{k}"] = sp[f"err_{k}"] / sp[f"ref_{k}"] * 100
ss = sc[(sc.kind == "single") & (sc.pt == f"b{int(bo.block.max())}.st") & (sc["pass"] == lastp)].copy()
for k in SREG:
    ss[f"rel_{k}"] = ss[f"err_{k}"] / ss[f"ref_{k}"] * 100
tab = sp.pivot_table(index="sched", columns="grp", values="rel_inter", aggfunc="median")
tab2 = ss.pivot_table(index="sched", columns="grp", values="rel_all", aggfunc="median")
print("-- pair inter-chain rel err % (median per group)")
print(tab.round(2).to_string())
print("-- pair all rel err %")
print(sp.pivot_table(index="sched", columns="grp", values="rel_all", aggfunc="median").round(2).to_string())
print("-- single rel err %")
print(tab2.round(2).to_string())
print("-- per target, pair inter-chain rel err %")
print(sp.pivot_table(index="sched", columns="target", values="rel_inter").round(2).to_string())
sp.to_csv(O / "sched_last.csv", index=False)
print("-- schedule trajectories (inter %, block-47 out per pass, median over all targets)")
s47 = sc[(sc.kind == "pair") & (sc.pt == f"b{int(bo.block.max())}.tr")].copy()
s47["rel"] = s47.err_inter / s47.ref_inter * 100
print(s47.pivot_table(index="sched", columns="pass", values="rel", aggfunc="median").round(2).to_string())

# 6. downstream
print("\n== 6. downstream (abl seed 42 x 5, only_transition vs bf16) vs final trunk error")
dn = pd.read_csv(O / "down.csv")
dn = dn[dn.cfg == "only_transition"][["target", "d_dq", "dq_b", "down", "up", "pose_med"]]
fin = bo.sort_values(["pass", "block"]).groupby("target").tail(1)[["target", "rel_inter", "rel_ab", "rel_ag", "rel_all"]]
fs = ls.sort_values("pass").groupby("target").tail(1)[["target", "rel_ab", "rel_ag"]].rename(columns={"rel_ab": "s_rel_ab", "rel_ag": "s_rel_ag"})
m = fin.merge(fs, on="target").merge(dn, on="target", how="left")
m["grp"] = m.target.map(GROUP)
for c in ["rel_inter", "rel_ab", "rel_ag", "rel_all", "s_rel_ab", "s_rel_ag"]:
    m[c] = m[c] * 100
print(m.sort_values("grp").round(3).to_string(index=False))
print("Spearman pose_med vs rel_inter:", round(m[["pose_med", "rel_inter"]].corr("spearman").iloc[0, 1], 3),
      " vs s_rel_ab:", round(m[["pose_med", "s_rel_ab"]].corr("spearman").iloc[0, 1], 3))
m.to_csv(O / "downstream_join.csv", index=False)
