"""F11 investigation, step 1 (CPU): is the FP8 perturbation larger on antibody-antigen, or the same perturbation with a more sensitive metric?

  python step1.py wholefb   --out DIR [--workers 32]   # engine 3.0.1 pair, whole FoldBench: per target d_prec / d_seed
  python step1.py tasks     --out DIR                  # per task: metric deltas, per-sample |d metric| prec vs seed, joined with d_prec
  python step1.py abag      --out DIR [--workers 32]   # engine 3.0.6 pair, antibody-antigen: where the FP8 deviation sits, correlates
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home()
F11 = H / "foldforge-scratch/paper-306/benchmarks/paper/F11_fp8_accuracy"
sys.path.insert(0, str(F11))
sys.path.insert(0, str(F11.parent / "F2_foldbench_accuracy"))
import score_tasks  # noqa: E402
from analyze import kabsch_rmsd  # noqa: E402

SEEDS = [42, 66, 101, 2024, 8888]
K = 5
FB = H / "data/foldbench"
SPECS = H / "foldforge-scratch/foldbench/specs_all"
V301 = (H / "foldforge-scratch/foldbench/ff_full/af3_301/pred", H / "foldforge-scratch/foldbench/ff_full/af3_301_fp8/pred")
V306 = (H / "foldforge-scratch/paper/F11/h100/af3_306_bf16_abagrna/pred", H / "foldforge-scratch/paper/F11/h100/af3_306_fp8_abagrna/pred")
EVAL301 = (H / "foldforge-scratch/paper/F2/h100/af3_301/eval_all/af3_301", H / "foldforge-scratch/paper/F2/h100/af3_301_fp8/eval_all/af3_301_fp8")
EVAL306 = (H / "foldforge-scratch/paper/F11/h100/eval/af3_306_bf16_abagrna/af3_306_bf16_abagrna",
           H / "foldforge-scratch/paper/F11/h100/eval/af3_306_fp8_abagrna/af3_306_fp8_abagrna")


def cif(pred: Path, t: str, s: int, k: int) -> Path:
    return pred / t / f"{t}_seed{s}_sample{k}_pred.cif"


def read_res(path: Path) -> pd.DataFrame:
    """One row per polymer residue: chain, residue index, rep atom xyz (CA / C1'), B-factor (pLDDT)."""
    cols, rows = [], []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split(".")[1].strip())
            continue
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            d = dict(zip(cols, f))
            if d.get("group_PDB") != "ATOM" or d.get("label_atom_id") not in ("CA", "C1'"):
                continue
            rows.append((d.get("auth_asym_id", d.get("label_asym_id")), int(d.get("label_seq_id", "0") if d.get("label_seq_id", ".") != "." else 0),
                         float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"]), float(d.get("B_iso_or_equiv", "nan"))))
    return pd.DataFrame(rows, columns=["chain", "resi", "x", "y", "z", "b"])


def xyz(d: pd.DataFrame) -> np.ndarray:
    return d[["x", "y", "z"]].to_numpy()


def kabsch_fit(p: np.ndarray, q: np.ndarray):
    """Rotation / translation that maps p onto q (least squares)."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    dd = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, dd]) @ vt
    return r, pc, qc


# ----------------------------------------------------------------------------------------------------------------- whole FoldBench
def _wfb(t: str) -> dict:
    b0, f0 = V301
    try:
        b = {(s, k): xyz(read_res(cif(b0, t, s, k))) for s in SEEDS for k in range(K)}
        f = {(s, k): xyz(read_res(cif(f0, t, s, k))) for s in SEEDS for k in range(K)}
    except FileNotFoundError as e:
        return {"target": t, "error": str(e)[:120]}
    n = len(b[(42, 0)])
    if n < 3 or any(len(v) != n for v in [*b.values(), *f.values()]):
        return {"target": t, "error": "rep atom count mismatch"}
    dp = [kabsch_rmsd(f[(s, k)], b[(s, k)]) for s in SEEDS for k in range(K)]
    ds = [kabsch_rmsd(b[(s, k)], b[(s2, k)]) for k in range(K) for s, s2 in itertools.combinations(SEEDS, 2)]
    return {"target": t, "n_rep": n, "median_d_prec": float(np.median(dp)), "median_d_seed": float(np.median(ds)),
            "mean_d_prec": float(np.mean(dp)), "mean_d_seed": float(np.mean(ds))}


def stage_wholefb(out: Path, workers: int) -> None:
    targets = (H / "foldforge-scratch/foldbench/all_targets.txt").read_text().split()
    with Pool(workers) as p:
        rows = p.map(_wfb, targets, chunksize=4)
    pd.DataFrame(rows).to_csv(out / "wholefb_dprec_dseed.csv", index=False)
    print("wholefb rows", len(rows), "errors", sum("error" in r and isinstance(r.get("error"), str) for r in rows))


# ----------------------------------------------------------------------------------------------------------------- tasks
def stage_tasks(out: Path) -> None:
    dps = pd.read_csv(out / "wholefb_dprec_dseed.csv").set_index("target")
    rows, per_group = [], []
    for task in score_tasks.TASKS:
        key = score_tasks.key_of(task)
        met = score_tasks.METRIC[task]
        col = {"lddt": "lddt", "dockq_score": "dockq_score"}.get(met, met)
        tb = {}
        for arm, ev in zip(("bf16", "fp8"), EVAL301):
            d = score_tasks.table(task, ev / "raw", ev / "targets")
            tb[arm] = d
        main = "lddt-pli" if task == "interface_protein_ligand" else ("dockq_score" if "dockq_score" in tb["bf16"].columns else "lddt")
        m = tb["bf16"].merge(tb["fp8"], on=[*key, "seed", "sample"], suffixes=("_b", "_f"))
        m = m[m[f"{main}_b"].notna() & m[f"{main}_f"].notna()]
        if m.empty:
            continue
        m["d_metric"] = m[f"{main}_f"] - m[f"{main}_b"]
        # seed spread of the same metric: same group, same sample index, other seed (bf16)
        b = tb["bf16"][[*key, "seed", "sample", main]].dropna()
        sd = []
        for _, g in b.groupby([*key, "sample"]):
            v = g[main].to_numpy()
            sd += [abs(x - y) for x, y in itertools.combinations(v, 2)]
        gm = m.groupby(key).agg(dm_mean=("d_metric", "mean"), b_mean=(f"{main}_b", "mean"), f_mean=(f"{main}_f", "mean")).reset_index()
        gm["target"] = gm.pdb_id
        gm = gm.join(dps[["median_d_prec", "median_d_seed", "n_rep"]], on="target")
        gm["task"] = task
        per_group.append(gm)
        rows.append({"task": task, "metric": main, "groups": len(gm), "targets": gm.target.nunique(),
                     "median_d_prec_A": float(gm.median_d_prec.median()), "median_d_seed_A": float(gm.median_d_seed.median()),
                     "ratio_prec_seed": float((gm.median_d_prec / gm.median_d_seed).median()),
                     "mean_d_metric_fp8_minus_bf16": float(m.d_metric.mean()),
                     "median_abs_d_metric_prec": float(m.d_metric.abs().median()),
                     "median_abs_d_metric_seed": float(np.median(sd)) if sd else np.nan,
                     "metric_sensitivity_ratio": float(m.d_metric.abs().median() / np.median(sd)) if sd and np.median(sd) > 0 else np.nan,
                     "abs_d_metric_per_A_prec": float((m.d_metric.abs().median()) / gm.median_d_prec.median()),
                     "bf16_mean_metric": float(m[f"{main}_b"].mean())})
    pd.DataFrame(rows).to_csv(out / "tasks_sensitivity.csv", index=False)
    pd.concat(per_group).to_csv(out / "tasks_per_group.csv", index=False)
    print(pd.DataFrame(rows).to_string())


# ----------------------------------------------------------------------------------------------------------------- antibody-antigen detail
def _abag(args) -> list[dict]:
    t, groups = args  # groups: list of (antibody_chain, antigen_chain)
    b0, f0 = V306
    ab_chains = sorted({a for a, _ in groups})
    ag_chains = sorted({g for _, g in groups})
    out = []
    try:
        bres = {(s, k): read_res(cif(b0, t, s, k)) for s in SEEDS for k in range(K)}
        fres = {(s, k): read_res(cif(f0, t, s, k)) for s in SEEDS for k in range(K)}
    except FileNotFoundError:
        return out
    for (s, k), b in bres.items():
        f = fres[(s, k)]
        if len(b) != len(f):
            continue
        ab = b.chain.isin(ab_chains).to_numpy()
        ag = b.chain.isin(ag_chains).to_numpy()
        if ab.sum() < 3 or ag.sum() < 3:
            continue
        B, F = xyz(b), xyz(f)
        # interface residues in bf16: antibody residues within 10 A of any antigen residue and vice versa
        dmat = np.linalg.norm(B[ab][:, None] - B[ag][None], axis=-1)
        ab_if = np.zeros(len(b), bool)
        ag_if = np.zeros(len(b), bool)
        ab_if[np.where(ab)[0][(dmat < 10).any(1)]] = True
        ag_if[np.where(ag)[0][(dmat < 10).any(0)]] = True
        # superpose FP8 onto bf16 on the antigen
        r, pc, qc = kabsch_fit(F[ag], B[ag])
        Fa = (F - pc) @ r + qc
        dev = np.linalg.norm(Fa - B, axis=1)
        # antibody alone (internal deformation)
        r2, pc2, qc2 = kabsch_fit(F[ab], B[ab])
        dev_ab_int = np.linalg.norm((F[ab] - pc2) @ r2 + qc2 - B[ab], axis=1)
        abi = np.where(ab)[0]
        int_if = ab_if[abi]
        rows = {"target": t, "seed": s, "sample": k, "n_ab_chains": len(ab_chains), "n_ag_chains": len(ag_chains),
                "n_res": len(b), "n_ab": int(ab.sum()), "n_ag": int(ag.sum()),
                "dev_ab_if_on_ag": float(dev[ab_if].mean()) if ab_if.any() else np.nan,
                "dev_ab_nonif_on_ag": float(dev[ab & ~ab_if].mean()),
                "dev_ag_if": float(dev[ag_if].mean()) if ag_if.any() else np.nan,
                "dev_ag_nonif": float(dev[ag & ~ag_if].mean()) if (ag & ~ag_if).any() else np.nan,
                "rmsd_ab_internal": float(np.sqrt(np.mean(dev_ab_int ** 2))),
                "dev_ab_if_internal": float(dev_ab_int[int_if].mean()) if int_if.any() else np.nan,
                "dev_ab_nonif_internal": float(dev_ab_int[~int_if].mean()) if (~int_if).any() else np.nan,
                "ab_pose_shift": float(np.linalg.norm(Fa[ab].mean(0) - B[ab].mean(0))),
                "plddt_ab_b": float(b.b[ab].mean()), "plddt_ab_f": float(f.b[ab].mean()),
                "plddt_ab_if_b": float(b.b[ab_if].mean()) if ab_if.any() else np.nan,
                "plddt_ab_if_f": float(f.b[ab_if].mean()) if ab_if.any() else np.nan,
                "plddt_ag_b": float(b.b[ag].mean()), "plddt_ag_f": float(f.b[ag].mean())}
        out.append(rows)
    return out


def stage_abag(out: Path, workers: int) -> None:
    tg = pd.read_csv(FB / "targets/interface_antibody_antigen.csv")
    targets = sorted(set(tg.pdb_id) & set((F11 / "targets_abag_rna.txt").read_text().split()))
    jobs = [(t, list(zip(g.antibody_chain, g.antigen_chain))) for t, g in tg[tg.pdb_id.isin(targets)].groupby("pdb_id")]
    with Pool(workers) as p:
        res = p.map(_abag, jobs, chunksize=1)
    d = pd.DataFrame([r for rr in res for r in rr])
    d.to_csv(out / "abag_per_sample.csv", index=False)
    # target-level attributes: templates per chain, token counts
    attrs = []
    for t, g in tg[tg.pdb_id.isin(targets)].groupby("pdb_id"):
        sp = SPECS / f"{t}.yaml"
        txt = sp.read_text() if sp.exists() else ""
        import yaml  # noqa: PLC0415

        y = yaml.safe_load(txt) if txt else {}
        letters = {str(k): v for k, v in (y.get("chain_letters") or {}).items()}
        tmpl = {str(k) for k in (y.get("template") or {})}
        chains_json = json.loads((SPECS / f"{t}.chains.json").read_text()) if (SPECS / f"{t}.chains.json").exists() else []
        idx = {c: str(i) for i, c in enumerate(chains_json)}
        ab = sorted(set(g.antibody_chain)); ag = sorted(set(g.antigen_chain))
        attrs.append({"target": t, "ab_chains": ",".join(ab), "ag_chains": ",".join(ag),
                      "ab_template_frac": float(np.mean([idx.get(c) in tmpl for c in ab])) if ab else np.nan,
                      "ag_template_frac": float(np.mean([idx.get(c) in tmpl for c in ag])) if ag else np.nan,
                      "n_spec_chains": len(letters)})
    pd.DataFrame(attrs).to_csv(out / "abag_target_attrs.csv", index=False)
    print("abag samples", len(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["wholefb", "tasks", "abag"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"wholefb": lambda: stage_wholefb(a.out, a.workers), "tasks": lambda: stage_tasks(a.out), "abag": lambda: stage_abag(a.out, a.workers)}[a.stage]()
