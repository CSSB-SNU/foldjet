"""F5 (a): one-to-one comparison of each kernel arm with the PyTorch arm, sample k against sample k (same input, same seed): superposed
Cα / C1' RMSD, mean |ΔpLDDT| per atom and mean |ΔPAE| per token pair (and their signed means). No top-1 selection, no ground truth.

    python pair.py RUNS_ROOT OUT_CSV [--arms engine,engine_fp8] [--reference torch] [--models ...] [--workers 8]

RUNS_ROOT: the F4 output dir (rows/<arm>__<model>.jsonl, pred/<arm>/<model>/<target>/*.npz + *.cif). Baseline (the sampling spread): the same
measures between DIFFERENT samples k < l of the reference arm, i.e. how much two draws of the same model differ anyway (arm = "baseline").
Writes OUT_CSV (one row per arm x model x target x sample) and <stem>_summary_by_{model,task}.csv. CPU only (Slurm --gres=none).
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["af3", "boltz2", "chai1", "esmfold2", "intellifold2", "opendde", "openfold3", "openfold3-preview2", "protenix1", "protenix2", "rosettafold3"]
COLS = ["arm", "model", "task", "target", "tokens", "sample", "rmsd", "plddt_abs", "plddt_signed", "pae_abs", "pae_signed"]


def ok_rows(runs: Path, arm: str, model: str) -> dict[str, dict]:
    p = runs / "rows" / f"{arm}__{model}.jsonl"
    if not p.exists():
        return {}
    return {r["target"]: r for r in map(json.loads, p.read_text().splitlines()) if r.get("status") == "ok"}


def rep_atoms(cif: Path) -> tuple[np.ndarray, int]:
    """Indices (atom_site order) of the polymer representative atoms (CA of amino acids, C1' of nucleotides) and the atom count."""
    idx, i = [], 0
    for line in cif.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if line.startswith("ATOM") and line.split()[3].strip('"') in ("CA", "C1'"):
                idx.append(i)
            i += 1
    return np.array(idx, dtype=int), i


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((p @ r - q) ** 2, axis=1))))


def measures(a: dict, b: dict, k: int, l: int, idx: np.ndarray, n: int) -> tuple:
    rmsd = kabsch_rmsd(a["xyz"][k][idx], b["xyz"][l][idx]) if len(idx) >= 3 else np.nan
    dp = a["plddt"][k] - b["plddt"][l]
    dpae = a["pae"][k][:n, :n] - b["pae"][l][:n, :n]
    return rmsd, float(np.abs(dp).mean()), float(dp.mean()), float(np.abs(dpae).mean()), float(dpae.mean())


def load(runs: Path, arm: str, model: str, target: str) -> dict:
    z = np.load(next((runs / "pred" / arm / model / target).glob("*.npz")))
    return {"xyz": z["coordinates"], "plddt": z["plddt"], "pae": z["full_pae"]}


def one_model(model: str, runs: Path, arms: list[str], reference: str) -> list[tuple]:
    rows: list[tuple] = []
    ref_ok = ok_rows(runs, reference, model)
    arm_ok = {arm: ok_rows(runs, arm, model) for arm in arms}
    for target, info in sorted(ref_ok.items()):
        try:
            cif = runs / "pred" / reference / model / target / f"{target}-seed0-0.cif"
            idx, n_atoms = rep_atoms(cif)
            t = load(runs, reference, model, target)
            if t["xyz"].shape[1] != n_atoms:
                print(f"skip {model} {target}: {t['xyz'].shape[1]} coordinates vs {n_atoms} CIF atoms", flush=True)
                continue
            tokens = int(info.get("tokens", t["pae"].shape[1]))
            n = min(tokens, t["pae"].shape[1])
            task = info.get("task", "")
            for k, l in combinations(range(t["xyz"].shape[0]), 2):
                rows.append(("baseline", model, task, target, tokens, f"{k}-{l}", *measures(t, t, k, l, idx, n)))
            for arm in arms:
                if target not in arm_ok[arm]:
                    continue
                a = load(runs, arm, model, target)
                if a["xyz"].shape != t["xyz"].shape:
                    print(f"skip {arm} {model} {target}: shape {a['xyz'].shape} vs {t['xyz'].shape}", flush=True)
                    continue
                for k in range(t["xyz"].shape[0]):
                    rows.append((arm, model, task, target, tokens, str(k), *measures(a, t, k, k, idx, n)))
        except Exception as error:  # noqa: BLE001 - one bad target does not stop the model
            print(f"failed {model} {target}: {type(error).__name__}: {str(error)[:150]}", flush=True)
    print(f"{model}: {len(rows)} rows", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arms", default="engine,engine_fp8")
    ap.add_argument("--reference", default="torch")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    fn = partial(one_model, runs=args.runs, arms=arms, reference=args.reference)
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(fn, args.models.split(",")) for r in part]
    df = pd.DataFrame(rows, columns=COLS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    for by in ("model", "task"):
        s = df.groupby(["arm", by]).agg(
            n=("rmsd", "size"), rmsd_median=("rmsd", "median"), rmsd_p90=("rmsd", lambda x: x.quantile(0.9)),
            plddt_abs_median=("plddt_abs", "median"), plddt_signed_mean=("plddt_signed", "mean"),
            pae_abs_median=("pae_abs", "median"), pae_signed_mean=("pae_signed", "mean"),
        ).reset_index()
        s.to_csv(args.out.with_name(f"{args.out.stem}_summary_by_{by}.csv"), index=False)
    print(df.groupby("arm")[["rmsd", "plddt_abs", "pae_abs"]].median().to_string())
    print("done", len(df), "rows ->", args.out)


if __name__ == "__main__":
    main()
