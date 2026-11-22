"""Downstream per target (abl set, seed 42 x 5): dDockQ (cfg - bf16) and rigid pose shift of chain group 1 (antibody / pp chain 1) after
superposing on group 2 (antigen / pp chain 2), for configs given. python down.py OUT_CSV cfg1 [cfg2 ...]"""
from __future__ import annotations
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "foldforge-scratch/paper/F11_investigation/scripts"))
from step1 import FB, kabsch_fit, read_res, score_tasks, xyz  # noqa: E402

A = Path.home() / "foldforge-scratch/paper/F11_investigation/abl"
TASKS = {"interface_antibody_antigen": "ab-ag", "interface_protein_protein": "prot-prot"}


def raw_of(cfg):
    for p in (A / cfg / "eval" / f"abl_{cfg}", A / cfg / "eval"):
        if (p / "raw").is_dir():
            return p


def groups(t):
    tg = pd.read_csv(FB / "targets/interface_antibody_antigen.csv"); g = tg[tg.pdb_id == t]
    if len(g):
        return "ab-ag", sorted(set(g.antibody_chain)), sorted(set(g.antigen_chain))
    pp = pd.read_csv(FB / "targets/interface_protein_protein.csv"); r = pp[pp.pdb_id == t].iloc[0]
    return "prot-prot", [r.interface_chain_id_1], [r.interface_chain_id_2]


def pose(args):
    cfg, t = args
    task, g1, g2 = groups(t)
    out = []
    for k in range(5):
        fb = A / "bf16/pred" / t / f"{t}_seed42_sample{k}_pred.cif"; fc = A / cfg / "pred" / t / f"{t}_seed42_sample{k}_pred.cif"
        if not (fb.exists() and fc.exists()):
            continue
        b, f = read_res(fb), read_res(fc)
        if len(b) != len(f):
            continue
        m1, m2 = b.chain.isin(g1).to_numpy(), b.chain.isin(g2).to_numpy()
        if m1.sum() < 3 or m2.sum() < 3:
            continue
        B, F = xyz(b), xyz(f)
        r, pc, qc = kabsch_fit(F[m2], B[m2]); Fa = (F - pc) @ r + qc
        out.append({"cfg": cfg, "target": t, "task": task, "sample": k, "pose_shift": float(np.linalg.norm(Fa[m1].mean(0) - B[m1].mean(0))),
                    "dev_g1": float(np.linalg.norm(Fa[m1] - B[m1], axis=1).mean())})
    return out


def main():
    outp, cfgs = Path(sys.argv[1]), sys.argv[2:]
    targets = (A / "targets.txt").read_text().split()
    with Pool(16) as p:
        rr = p.map(pose, [(c, t) for c in cfgs for t in targets], chunksize=4)
    ps = pd.DataFrame([r for x in rr for r in x])
    ref = raw_of("bf16"); rows = []
    for task, short in TASKS.items():
        key = score_tasks.KEY_I
        tb = score_tasks.table(task, ref / "raw", ref / "targets")[[*key, "seed", "sample", "dockq_score"]].dropna(subset=["dockq_score"])
        for cfg in cfgs:
            tc = score_tasks.table(task, raw_of(cfg) / "raw", raw_of(cfg) / "targets")[[*key, "seed", "sample", "dockq_score"]].dropna(subset=["dockq_score"])
            m = tb.merge(tc, on=[*key, "seed", "sample"], suffixes=("_b", "_c"))
            m["d_dq"] = m.dockq_score_c - m.dockq_score_b
            m["flip_down"] = (m.dockq_score_b >= .23) & (m.dockq_score_c < .23); m["flip_up"] = (m.dockq_score_b < .23) & (m.dockq_score_c >= .23)
            g = m.groupby("pdb_id").agg(d_dq=("d_dq", "mean"), dq_b=("dockq_score_b", "mean"), down=("flip_down", "sum"), up=("flip_up", "sum")).reset_index()
            g["cfg"], g["task"] = cfg, short
            rows.append(g.rename(columns={"pdb_id": "target"}))
    dq = pd.concat(rows)
    pm = ps.groupby(["cfg", "target"]).agg(pose_med=("pose_shift", "median"), pose_mean=("pose_shift", "mean"), dev_g1=("dev_g1", "median")).reset_index()
    res = dq.merge(pm, on=["cfg", "target"], how="left")
    res.to_csv(outp, index=False)
    ps.to_csv(outp.with_name(outp.stem + "_samples.csv"), index=False)
    for cfg in cfgs:
        for task in ("ab-ag", "prot-prot"):
            r = res[(res.cfg == cfg) & (res.task == task)]
            print(cfg, task, len(r), "mean d_dq %.4f" % r.d_dq.mean(), "pose med %.2f" % r.pose_med.median())
    print(res[res.cfg == cfgs[0]].sort_values("d_dq").to_string())


main()
