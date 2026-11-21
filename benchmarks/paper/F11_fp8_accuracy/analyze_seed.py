"""F11 (redesigned 2026-11-21): is the FP8 perturbation of the SAME seed small compared with the natural variation BETWEEN seeds?

Primary pair: engine 3.0.6, bf16 (`af3_306_bf16_abagrna`) vs FP8 with check() (`af3_306_fp8_abagrna`); replicate pair: engine 3.0.1, bf16 (`af3_301`) vs
FP8 (`af3_301_fp8`). 162 antibody-antigen + protein-RNA targets, 5 trunk seeds x 5 samples, same seeds and sample indices in both arms. CPU only.

A. structure (per target; C-alpha / C1' RMSD after superposition):
   d_prec(s, k)   = RMSD(FP8 s k, bf16 s k)                 same seed, same sample index
   d_seed(s,s',k) = RMSD(bf16 s k, bf16 s' k), s != s'        same sample index, other seed
   d_sample       = RMSD(bf16 s k, bf16 s k'), k != k'        same seed, other sample (context)
   per target: medians, ratio median d_prec / median d_seed; per (target, s, k): is the FP8 twin closer to bf16 (s, k) than the nearest bf16 sample of
   any other seed (min over s' != s and all k')?
B. accuracy, each seed = a replicate of 5 samples (interface groups of the task, DockQ from the FoldBench evaluator, upstream rules: NaN rows dropped):
   per seed s and arm: success of the seed's own top-1 (largest ranking score of its 5 samples), the seed's oracle (best of its 5), mean DockQ of its 5.
   between-seed spread (bf16): SD and range over the 5 seeds, |m(bf16 s) - m(bf16 s')| over the 10 seed pairs;
   precision effect: m(fp8 s) - m(bf16 s) per seed; bootstrap over interface groups for both.
   verdict per task and metric: the FP8 effect is "inside" the between-seed spread when |mean precision effect| <= mean |between-seed difference| and its
   95% interval includes 0; "outside" (a precision loss) when the interval excludes 0 and |effect| > the mean between-seed difference; else "borderline".

  python analyze_seed.py structure --out DIR [--workers 16]
  python analyze_seed.py accuracy  --out DIR
"""

from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "F2_foldbench_accuracy"))
import score_tasks  # noqa: E402
from analyze import kabsch_rmsd, read_rep  # noqa: E402

FF_FULL = Path.home() / "foldforge-scratch/foldbench/ff_full"
P306 = Path.home() / "foldforge-scratch/paper/F11/h100"
EVALS = P306 / "eval"
SEEDS = [42, 66, 101, 2024, 8888]
K = 5
TASKS = ["interface_antibody_antigen", "interface_protein_rna"]
FB = Path.home() / "data/foldbench"
# pair -> (engine version, bf16 run dir, fp8 run dir); eval name = run dir name
PAIRS = {
    "v306": ("3.0.6", P306 / "af3_306_bf16_abagrna", P306 / "af3_306_fp8_abagrna"),
    "v301": ("3.0.1", FF_FULL / "af3_301", FF_FULL / "af3_301_fp8"),
}


def task_of() -> dict[str, str]:
    m = {}
    for task in TASKS:
        for pdb in pd.read_csv(FB / "targets" / f"{task}.csv").pdb_id.unique():
            m.setdefault(pdb, task)
    return m


# ----------------------------------------------------------------------------- A. structure
def _one(args: tuple[str, str]) -> tuple[list[dict], dict]:
    pair, t = args
    _, bdir, fdir = PAIRS[pair]
    b = {(s, k): read_rep(bdir / "pred" / t / f"{t}_seed{s}_sample{k}_pred.cif") for s in SEEDS for k in range(K)}
    f = {(s, k): read_rep(fdir / "pred" / t / f"{t}_seed{s}_sample{k}_pred.cif") for s in SEEDS for k in range(K)}
    keys = list(b)
    n = len(b[keys[0]])
    if n < 3 or any(len(v) != n for v in [*b.values(), *f.values()]):
        return [], {"pair": pair, "target": t, "error": "atom count mismatch"}
    D = {}  # bf16 all pairs
    for i, j in itertools.combinations(keys, 2):
        D[(i, j)] = D[(j, i)] = kabsch_rmsd(b[i], b[j])
    rows = []
    d_seed, d_sample = [], []
    for k in range(K):
        for s, s2 in itertools.combinations(SEEDS, 2):
            d_seed.append(D[((s, k), (s2, k))])
    for s in SEEDS:
        for k, k2 in itertools.combinations(range(K), 2):
            d_sample.append(D[((s, k), (s, k2))])
    for s in SEEDS:
        for k in range(K):
            dp = kabsch_rmsd(f[(s, k)], b[(s, k)])
            nearest_other_seed = min(D[((s, k), (s2, k2))] for s2 in SEEDS if s2 != s for k2 in range(K))
            same_k_other_seed = [D[((s, k), (s2, k))] for s2 in SEEDS if s2 != s]
            rows.append({"pair": pair, "target": t, "seed": s, "sample": k, "d_prec": dp, "nearest_other_seed": nearest_other_seed,
                         "median_same_k_other_seed": float(np.median(same_k_other_seed)), "twin_closer_than_any_other_seed": dp < nearest_other_seed})
    summ = {"pair": pair, "target": t, "n_rep_atoms": n, "median_d_prec": float(np.median([r["d_prec"] for r in rows])),
            "median_d_seed": float(np.median(d_seed)), "median_d_sample": float(np.median(d_sample)),
            "frac_twin_closer": float(np.mean([r["twin_closer_than_any_other_seed"] for r in rows]))}
    summ["ratio_prec_over_seed"] = summ["median_d_prec"] / summ["median_d_seed"] if summ["median_d_seed"] > 0 else np.nan
    return rows, summ


def stage_structure(out: Path, workers: int) -> None:
    targets = (HERE / "targets_abag_rna.txt").read_text().split()
    jobs = [(p, t) for p in PAIRS for t in targets]
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=2)
    tk = task_of()
    samples = pd.DataFrame([r for rows, _ in res for r in rows])
    per_t = pd.DataFrame([s for _, s in res])
    for d in (samples, per_t):
        d.insert(1, "engine_version", d.pair.map(lambda p: PAIRS[p][0]))
        d["task"] = d.target.map(tk)
    samples.to_csv(out / "f11seed_structure_samples.csv", index=False)
    per_t.to_csv(out / "f11seed_structure_per_target.csv", index=False)
    rows = []
    for (pair, task), d in per_t.dropna(subset=["median_d_prec"]).groupby(["pair", "task"]):
        s = samples[(samples.pair == pair) & (samples.task == task)]
        q = d.ratio_prec_over_seed
        rows.append({"pair": pair, "engine_version": PAIRS[pair][0], "task": task, "targets": len(d),
                     "median_d_prec": d.median_d_prec.median(), "median_d_seed": d.median_d_seed.median(), "median_d_sample": d.median_d_sample.median(),
                     "ratio_median": q.median(), "ratio_q25": q.quantile(.25), "ratio_q75": q.quantile(.75),
                     "frac_targets_d_prec_lt_d_seed": float((d.median_d_prec < d.median_d_seed).mean()),
                     "frac_samples_twin_closer_than_any_other_seed": float(s.twin_closer_than_any_other_seed.mean())})
    pd.DataFrame(rows).to_csv(out / "f11seed_structure_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------- B. accuracy per seed
def _per_seed_metrics(d: pd.DataFrame) -> pd.DataFrame:
    """d: per-sample rows of one task and arm (group key, seed, sample, ranking_score, dockq_score). -> per (group, seed): top1 ok, oracle ok, mean dockq."""
    d = d[d.dockq_score.notna()]
    key = score_tasks.KEY_I
    out = []
    for _, x in d.groupby([*key, "seed"]):
        i = int(x.ranking_score.values.argmax())  # first on a tie (upstream idxmax)
        out.append({**{c: x[c].iloc[0] for c in key}, "seed": int(x.seed.iloc[0]), "top1": float(x.dockq_score.values[i] >= 0.23),
                    "oracle": float(x.dockq_score.max() >= 0.23), "mean_dq": float(x.dockq_score.mean())})
    return pd.DataFrame(out)


def _boot(f, groups: list, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    est = f(groups)
    bs = [f([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def stage_accuracy(out: Path) -> None:
    key = score_tasks.KEY_I
    per_seed_rows, summ, lines = [], [], ["F11 (seed design) verdict, generated by analyze_seed.py", ""]
    for pair, (ver, bdir, fdir) in PAIRS.items():
        for task in TASKS:
            arms = {}
            for arm, rdir in (("bf16", bdir), ("fp8", fdir)):
                name = rdir.name
                raw = EVALS / name / name
                if not (raw / "raw").is_dir():
                    raw = Path.home() / "foldforge-scratch/paper/F2/h100" / name / "eval_all" / name  # the whole-FoldBench evaluations of 3.0.1
                arms[arm] = _per_seed_metrics(score_tasks.table(task, raw / "raw", raw / "targets")).set_index([*key, "seed"])
            m = arms["bf16"].join(arms["fp8"], lsuffix="_bf16", rsuffix="_fp8", how="inner").reset_index()
            m["pair"], m["engine_version"], m["task"] = pair, ver, task
            per_seed_rows.append(m)
            gkeys = sorted({tuple(r) for r in m[key].values})
            gi = {g: i for i, g in enumerate(gkeys)}
            si = {s: j for j, s in enumerate(SEEDS)}
            groups = list(range(len(gkeys)))
            lines.append(f"{task}  (engine {ver}, {len(groups)} interface groups)")
            for met in ("top1", "oracle", "mean_dq"):
                scale = 100.0 if met != "mean_dq" else 1.0
                A = {arm: np.full((len(gkeys), len(SEEDS)), np.nan) for arm in ("bf16", "fp8")}  # group x seed
                for r in m.itertuples(index=False):
                    g, j = gi[tuple(getattr(r, c) for c in key)], si[int(r.seed)]
                    for arm in A:
                        A[arm][g, j] = getattr(r, f"{met}_{arm}")

                def per_seed(idx, arm, A=A, scale=scale):  # task metric of every seed over the groups idx
                    return scale * np.nanmean(A[arm][idx], axis=0)

                def prec(idx):
                    return float(np.nanmean(per_seed(idx, "fp8") - per_seed(idx, "bf16")))

                def seed_diff(idx):
                    v = per_seed(idx, "bf16")
                    return float(np.nanmean([abs(a - b) for a, b in itertools.combinations(v, 2)]))

                all_idx = np.array(groups)
                bf, fp = list(per_seed(all_idx, "bf16")), list(per_seed(all_idx, "fp8"))
                pe, pe_lo, pe_hi = _boot(lambda gs: prec(np.array(gs)), groups)
                sd, sd_lo, sd_hi = _boot(lambda gs: seed_diff(np.array(gs)), groups, seed=1)
                pair_diffs = [abs(a - b) for a, b in itertools.combinations(bf, 2)]
                if pe_lo <= 0 <= pe_hi and abs(pe) <= sd:
                    verdict = "inside the between-seed spread (FP8 ~ choosing another seed)"
                elif not (pe_lo <= 0 <= pe_hi) and abs(pe) > sd:
                    verdict = "outside the between-seed spread (a precision loss)"
                else:
                    verdict = "borderline"
                summ.append({"pair": pair, "engine_version": ver, "task": task, "metric": met, "groups": len(groups),
                             **{f"bf16_seed{s}": v for s, v in zip(SEEDS, bf)}, **{f"fp8_seed{s}": v for s, v in zip(SEEDS, fp)},
                             "bf16_mean_over_seeds": float(np.mean(bf)), "bf16_sd_over_seeds": float(np.std(bf, ddof=1)),
                             "bf16_range_over_seeds": float(np.max(bf) - np.min(bf)),
                             "bf16_seed_pair_absdiff_median": float(np.median(pair_diffs)), "bf16_seed_pair_absdiff_max": float(np.max(pair_diffs)),
                             "between_seed_absdiff_mean": sd, "between_seed_absdiff_ci_lo": sd_lo, "between_seed_absdiff_ci_hi": sd_hi,
                             "fp8_minus_bf16_per_seed": ";".join(f"{f - b:+.3f}" for f, b in zip(fp, bf)),
                             "precision_effect_mean": pe, "precision_effect_ci_lo": pe_lo, "precision_effect_ci_hi": pe_hi, "verdict": verdict})
                u = "pt" if met != "mean_dq" else ""
                lines.append(f"  {met:8s} bf16 per seed {' '.join(f'{v:6.2f}' if met != 'mean_dq' else f'{v:.3f}' for v in bf)}  (SD {np.std(bf, ddof=1):.3f}, range {np.max(bf) - np.min(bf):.3f})")
                lines.append(f"  {'':8s} fp8  per seed {' '.join(f'{v:6.2f}' if met != 'mean_dq' else f'{v:.3f}' for v in fp)}")
                lines.append(f"  {'':8s} precision effect (fp8 - bf16, same seed) {pe:+.3f}{u} [{pe_lo:+.3f}, {pe_hi:+.3f}]   between-seed |difference| (bf16) {sd:.3f}{u} [{sd_lo:.3f}, {sd_hi:.3f}]  -> {verdict}")
            lines.append("")
    pd.concat(per_seed_rows, ignore_index=True).to_csv(out / "f11seed_accuracy_per_group_seed.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "f11seed_accuracy_summary.csv", index=False)
    s = pd.read_csv(out / "f11seed_structure_summary.csv") if (out / "f11seed_structure_summary.csv").exists() else None
    if s is not None:
        lines.append("Structure (per task; RMSD in Angstrom; ratio = median d_prec / median d_seed per target):")
        for _, r in s.iterrows():
            lines.append(f"  engine {r.engine_version} {r.task:28s} d_prec {r.median_d_prec:5.2f}  d_seed {r.median_d_seed:5.2f}  d_sample {r.median_d_sample:5.2f}  "
                         f"ratio {r.ratio_median:.2f} [IQR {r.ratio_q25:.2f}-{r.ratio_q75:.2f}]  targets d_prec < d_seed {100 * r.frac_targets_d_prec_lt_d_seed:.0f}%  "
                         f"samples whose FP8 twin is closer than any other-seed sample {100 * r.frac_samples_twin_closer_than_any_other_seed:.0f}%")
    (out / "f11seed_verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["structure", "accuracy"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stage_structure(a.out, a.workers) if a.stage == "structure" else stage_accuracy(a.out)
