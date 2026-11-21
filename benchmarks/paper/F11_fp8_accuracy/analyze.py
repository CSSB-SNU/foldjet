"""F11: why is AF3 with FP8 less accurate than bf16 on antibody-antigen and protein-RNA? Rounding of the structures, or a different order of near-tied samples?

Runs compared (FoldForge a9e1549 + engine 3.0.1, whole-FoldBench runs, fb_multi, 5 trunk_seeds x 5 samples, same seeds):
  bf16   ff_full/af3_301        (Slurm 58554)       FP8 ff_full/af3_301_fp8     (58867)       FP8 re-run ff_full/af3_301_fp8chk (58996, the 162 targets only)
The re-run of the SAME FP8 configuration on the same targets measures how much two FP8 runs differ from each other by themselves (scheduling, GPU
non-determinism, MPS grouping): the noise floor against which an FP8-vs-bf16 difference has to be read.

Stages (CPU only; run through Slurm, see README.md):
  rmsd     per sample (target, seed, sample): superposed C-alpha / C1' RMSD of FP8 vs bf16, FP8 re-run vs FP8, FP8 re-run vs bf16, and of two different bf16
           samples of the same seed (the scale of "a different draw"); ranking scores of all three runs.      -> results/f11_sample_rmsd.csv
  analyze  reads that table and the per-sample FoldBench evaluation (F2's evaluate.py without --top1-only) of the three runs:
           * top-1 and oracle per task (rules of upstream task_score_summary.py, score_tasks.py)                 -> results/f11_task_scores.csv
           * mean DockQ over all 25 samples per target (the quality of the samples themselves)                    -> results/f11_sample_quality.csv
           * per target: top-1 sample of each run, gap to the runner-up, flip flags, outcome flags                -> results/f11_targets.csv
           * counterfactuals: FP8 structures picked with bf16's ranking scores, bf16 structures picked with FP8's -> results/f11_swaps.csv
           * bootstrap intervals (over targets) and the verdict                                                  -> results/f11_verdict.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "F2_foldbench_accuracy"))
import score_tasks  # noqa: E402

FF_FULL = Path.home() / "foldforge-scratch/foldbench/ff_full"
RUNS = {"bf16": "af3_301", "fp8": "af3_301_fp8", "fp8b": "af3_301_fp8chk"}  # label -> run directory name under ff_full
SEEDS = [42, 66, 101, 2024, 8888]
TASKS = ["interface_antibody_antigen", "interface_protein_rna"]  # the two tasks of the question; the other tasks of the 162 targets are in f11_task_scores.csv
FB = Path.home() / "data/foldbench"


# ----------------------------------------------------------------------------- stage 1: per-sample RMSD
def read_rep(cif: Path) -> np.ndarray:
    """Coordinates of the polymer representative atoms (C-alpha of amino acids, C1' of nucleotides), in file order."""
    cols: list[str] = []
    xyz = []
    ia = ix = iy = iz = ig = None
    for line in cif.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split(".")[1].strip())
            continue
        if line.startswith(("ATOM", "HETATM")):
            if ia is None:
                ia, ix, iy, iz, ig = (cols.index(c) for c in ("label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB"))
            f = line.split()
            if f[ig] == "ATOM" and f[ia].strip('"') in ("CA", "C1'"):
                xyz.append((float(f[ix]), float(f[iy]), float(f[iz])))
    return np.array(xyz)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((p @ r - q) ** 2, axis=1))))


def ranking(run: str, t: str) -> dict[tuple[int, int], float]:
    f = FF_FULL / run / "pred" / t / "ranking.csv"
    return {(int(s), int(k)): float(v) for s, k, v in (ln.split(",") for ln in f.read_text().splitlines()[1:])}


def one_target(t: str) -> list[dict]:
    rows = []
    rk = {lab: ranking(run, t) for lab, run in RUNS.items()}
    xyz: dict[tuple[str, int, int], np.ndarray] = {}
    for lab, run in RUNS.items():
        for s in SEEDS:
            for k in range(5):
                xyz[(lab, s, k)] = read_rep(FF_FULL / run / "pred" / t / f"{t}_seed{s}_sample{k}_pred.cif")
    for s in SEEDS:
        for k in range(5):
            a, b, c = xyz[("bf16", s, k)], xyz[("fp8", s, k)], xyz[("fp8b", s, k)]
            o = xyz[("bf16", s, (k + 1) % 5)]
            ok = len(a) == len(b) == len(c) == len(o) >= 3
            rows.append({
                "target": t, "seed": s, "sample": k, "n_rep_atoms": len(a),
                "rmsd_fp8_bf16": kabsch_rmsd(b, a) if ok else np.nan,
                "rmsd_fp8b_fp8": kabsch_rmsd(c, b) if ok else np.nan,
                "rmsd_fp8b_bf16": kabsch_rmsd(c, a) if ok else np.nan,
                "rmsd_bf16_nextsample": kabsch_rmsd(o, a) if ok else np.nan,
                "score_bf16": rk["bf16"][(s, k)], "score_fp8": rk["fp8"][(s, k)], "score_fp8b": rk["fp8b"][(s, k)],
            })
    return rows


def stage_rmsd(out: Path, workers: int) -> None:
    targets = (HERE / "targets_abag_rna.txt").read_text().split()
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(one_target, targets) for r in part]
    df = pd.DataFrame(rows).sort_values(["target", "seed", "sample"])
    df.to_csv(out / "f11_sample_rmsd.csv", index=False)
    print(f"{len(df)} samples of {len(targets)} targets -> {out / 'f11_sample_rmsd.csv'}")


# ----------------------------------------------------------------------------- stage 2: accuracy, flips, verdict
def task_members() -> dict[str, str]:
    """assembly -> task (antibody-antigen / protein-RNA / other) from FoldBench's task CSVs."""
    m = {}
    for task in score_tasks.INTERFACES + score_tasks.MONOMERS:
        for pdb in pd.read_csv(FB / "targets" / f"{task}.csv").pdb_id.unique():
            m.setdefault(pdb, task)
    return m


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage_analyze(out: Path, evals: Path) -> None:
    s = pd.read_csv(out / "f11_sample_rmsd.csv")
    raws = {lab: evals / run / run / "raw" for lab, run in RUNS.items()}
    tdirs = {lab: evals / run / run / "targets" for lab, run in RUNS.items()}
    # --- task scores (top-1, oracle) per run, per task
    ts = []
    for lab in RUNS:
        t = score_tasks.scores(raws[lab], tdirs[lab])
        t.insert(0, "run", lab)
        ts.append(t)
    ts = pd.concat(ts, ignore_index=True)
    ts.to_csv(out / "f11_task_scores.csv", index=False)

    # --- per-sample evaluation tables -> mean DockQ over the 25 samples; swaps
    members = task_members()
    per = {lab: {task: score_tasks.table(task, raws[lab], tdirs[lab]) for task in TASKS} for lab in RUNS}
    qual = []
    for task in TASKS:
        col = "dockq_score"
        frames = {lab: per[lab][task][["pdb_id", "interface_chain_id_1", "interface_chain_id_2", "seed", "sample", "ranking_score", col]].rename(columns={col: f"dq_{lab}", "ranking_score": f"rs_{lab}"}) for lab in RUNS}
        key = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2", "seed", "sample"]
        m = frames["bf16"].merge(frames["fp8"], on=key).merge(frames["fp8b"], on=key)
        m["task"] = task
        qual.append(m)
    q = pd.concat(qual, ignore_index=True)
    q.to_csv(out / "f11_samples_dockq.csv", index=False)
    g = ["task", "pdb_id", "interface_chain_id_1", "interface_chain_id_2"]
    sq = q.groupby(g).agg(**{f"mean_dq_{lab}": (f"dq_{lab}", "mean") for lab in RUNS}, **{f"max_dq_{lab}": (f"dq_{lab}", "max") for lab in RUNS}, n=("seed", "size")).reset_index()
    sq.to_csv(out / "f11_sample_quality.csv", index=False)

    # --- per interface group: top-1 of each run, success, runner-up gap, flips, swaps
    rows = []
    for (task, pdb, c1, c2), d in q.groupby(g):
        d = d.reset_index(drop=True)
        r = {"task": task, "target": pdb, "chain1": c1, "chain2": c2, "n_samples": len(d)}
        top = {}
        for lab in RUNS:
            order = d[f"rs_{lab}"].values
            i = int(np.argmax(order))  # first on a tie (upstream idxmax)
            srt = np.sort(order)[::-1]
            top[lab] = i
            r[f"top1_{lab}"] = f"{int(d.seed[i])}/{int(d['sample'][i])}"
            r[f"score_top1_{lab}"] = float(srt[0])
            r[f"gap12_{lab}"] = float(srt[0] - srt[1]) if len(srt) > 1 else np.nan
            r[f"dq_top1_{lab}"] = float(d[f"dq_{lab}"][i])
            r[f"ok_top1_{lab}"] = bool(d[f"dq_{lab}"][i] >= 0.23)
            r[f"ok_oracle_{lab}"] = bool(d[f"dq_{lab}"].max() >= 0.23)
        for a, b in (("fp8", "bf16"), ("fp8b", "fp8"), ("fp8b", "bf16")):
            r[f"flip_{a}_{b}"] = top[a] != top[b]
            r[f"outcome_changed_{a}_{b}"] = r[f"ok_top1_{a}"] != r[f"ok_top1_{b}"]
        # swaps: the structures of A picked with the ranking scores of B (same (seed, sample) index in both runs)
        for sa, sb in (("fp8", "bf16"), ("bf16", "fp8"), ("fp8b", "bf16"), ("bf16", "fp8b"), ("fp8b", "fp8")):
            i = top[sb]
            r[f"swap_struct_{sa}_rank_{sb}"] = bool(d[f"dq_{sa}"][i] >= 0.23)
        rows.append(r)
    tg = pd.DataFrame(rows)
    tg.to_csv(out / "f11_targets.csv", index=False)

    sw = []
    for task, d in tg.groupby("task"):
        row = {"task": task, "n": len(d)}
        for lab in RUNS:
            row[f"top1_{lab}"] = 100 * d[f"ok_top1_{lab}"].mean()
            row[f"oracle_{lab}"] = 100 * d[f"ok_oracle_{lab}"].mean()
        for c in [c for c in d.columns if c.startswith("swap_struct_")]:
            row[c] = 100 * d[c].mean()
        for c in [c for c in d.columns if c.startswith(("flip_", "outcome_changed_"))]:
            row[c] = 100 * d[c].mean()
        sw.append(row)
    pd.DataFrame(sw).to_csv(out / "f11_swaps.csv", index=False)

    # --- statistics and verdict
    lines = ["F11 verdict (generated by analyze.py; numbers in f11_*.csv)", ""]
    s = s.merge(pd.Series(members, name="task_of").rename_axis("target").reset_index(), on="target", how="left")
    lines.append("Per-sample distances (C-alpha / C1' RMSD after superposition, Angstrom; all samples of the 162 targets):")
    for col, label in (("rmsd_fp8_bf16", "FP8 vs bf16, same seed and sample"), ("rmsd_fp8b_fp8", "FP8 re-run vs FP8 (noise floor)"), ("rmsd_fp8b_bf16", "FP8 re-run vs bf16"),
                       ("rmsd_bf16_nextsample", "bf16 sample k vs k+1 of the same seed (a different draw)")):
        for tk in TASKS:
            x = s.loc[s.task_of == tk, col].dropna()
            lines.append(f"  {label:58s} {tk:28s} median {x.median():6.2f}  mean {x.mean():6.2f}  p90 {x.quantile(.9):6.2f}  n {len(x)}")
    lines.append("")
    lines.append("Ranking score of the same (seed, sample) in the two runs:")
    for a, b, label in (("fp8", "bf16", "FP8 - bf16"), ("fp8b", "fp8", "FP8 re-run - FP8")):
        dlt = (s[f"score_{a}"] - s[f"score_{b}"]).values
        lines.append(f"  {label}: mean {dlt.mean():+.4f}  sd {dlt.std():.4f}  median |d| {np.median(np.abs(dlt)):.4f}  (score range of a target ~ {s.groupby('target').score_bf16.apply(np.ptp).median():.3f} median)")
    lines.append("")
    lines.append("Top-1 / oracle success (DockQ >= 0.23) per task, % of interface groups; bootstrap 95% interval over groups of the paired difference:")
    verdict_rows = {}
    for task, d in tg.groupby("task"):
        lines.append(f"  {task} (n = {len(d)})")
        for lab in RUNS:
            lines.append(f"    {lab:5s} top-1 {100 * d[f'ok_top1_{lab}'].mean():6.2f}   oracle {100 * d[f'ok_oracle_{lab}'].mean():6.2f}   mean DockQ of all samples {sq.loc[sq.task == task, f'mean_dq_{lab}'].mean():.4f}")
        for a, b in (("fp8", "bf16"), ("fp8b", "bf16"), ("fp8", "fp8b")):
            for what in ("top1", "oracle"):
                x = d[f"ok_{what}_{a}"].astype(float).values - d[f"ok_{what}_{b}"].astype(float).values
                m, lo, hi = boot_ci(x)
                lines.append(f"    {what:6s} {a:4s} - {b:4s}: {100 * m:+6.2f} pt  [{100 * lo:+6.2f}, {100 * hi:+6.2f}]")
                verdict_rows[(task, what, a, b)] = (m, lo, hi)
            x = (sq.loc[sq.task == task, f"mean_dq_{a}"] - sq.loc[sq.task == task, f"mean_dq_{b}"]).values
            m, lo, hi = boot_ci(x)
            lines.append(f"    mean DockQ of the 25 samples {a:4s} - {b:4s}: {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            verdict_rows[(task, "meandq", a, b)] = (m, lo, hi)
        lines.append(f"    targets whose top-1 sample differs: FP8 vs bf16 {100 * d.flip_fp8_bf16.mean():.1f}%, FP8 re-run vs FP8 {100 * d.flip_fp8b_fp8.mean():.1f}%, FP8 re-run vs bf16 {100 * d.flip_fp8b_bf16.mean():.1f}%")
        lines.append(f"    targets whose top-1 success changes: FP8 vs bf16 {100 * d.outcome_changed_fp8_bf16.mean():.1f}%, FP8 re-run vs FP8 {100 * d.outcome_changed_fp8b_fp8.mean():.1f}%")
        near = d["gap12_bf16"] < 0.01
        lines.append(f"    near-tied top-1 in bf16 (gap to runner-up < 0.01): {100 * near.mean():.1f}% of groups; flips there {100 * d.loc[near, 'flip_fp8_bf16'].mean() if near.any() else float('nan'):.1f}%, elsewhere {100 * d.loc[~near, 'flip_fp8_bf16'].mean() if (~near).any() else float('nan'):.1f}%")
        lines.append(f"    swaps: FP8 structures ranked by bf16 {100 * d.swap_struct_fp8_rank_bf16.mean():.2f}%, bf16 structures ranked by FP8 {100 * d.swap_struct_bf16_rank_fp8.mean():.2f}%  (own-ranking: FP8 {100 * d.ok_top1_fp8.mean():.2f}%, bf16 {100 * d.ok_top1_bf16.mean():.2f}%)")
        lines.append("")
    (out / "f11_verdict.txt").write_text("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}-{k[3]}": v for k, v in verdict_rows.items()}, (out / "f11_intervals.json").open("w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rmsd", "analyze"])
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--evals", type=Path, default=Path.home() / "foldforge-scratch/paper/F11/h100/eval", help="eval dirs <evals>/<run>/<run>/raw of evaluate.py (no --top1-only)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stage_rmsd(a.out, a.workers) if a.stage == "rmsd" else stage_analyze(a.out, a.evals)
