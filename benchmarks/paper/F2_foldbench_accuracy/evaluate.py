"""Score FoldJet (or kit) FoldBench predictions with the upstream FoldBench evaluator (BEAM-Labs FoldBench, OpenStructure lDDT / DockQ v2) and
write the task scores, top-1 and oracle. CPU only: run it as a Slurm job with --gres=none (never on the login node); see eval.sbatch.

    python evaluate.py --pred-dir RUN/pred_s0 --pred-dir RUN/pred_s1 --eval-dir OUT --name af3_bf16 [--targets-file LIST] [--top1-only]

Inputs: prediction directories `<pred>/<target>/<target>_seed{s}_sample{k}_pred.cif` + `ranking.csv` (seed,sample,ranking_score) as fb_multi.py writes
them (several --pred-dir are merged: the shard layout of a one-process-per-shard run). The evaluator reads the CIFs in place (absolute paths in prediction_reference.csv).

What it adds to upstream's driver (`evaluate.py` + `task_score_summary.py`, run unchanged with the paths below):
  * `prediction_reference.csv` from the directory layout, with the ranking score of ranking.csv (upstream expects one row per (pdb_id, seed, sample));
  * task CSVs restricted to the predicted targets (upstream left-joins onto all 1,522 and crashes on NaN paths);
  * `--top1-only`: only the top-ranked sample of each target is scored (the first on a ranking-score tie, as upstream's idxmax). Oracle needs all samples;
  * the scores of F2 (`scores_<name>.csv`: top-1 and oracle per task, rules of task_score_summary.py, see score_tasks.py), and upstream's own `rank`
    summary as a cross-check (`summary_<name>_rank.csv`).

Environment: FB_ROOT (default ~/data/foldbench: targets/, ground_truth/, upstream/ or upstream_cy/ with bin/ost and images/ost.sif; upstream_cy has the
Cython DockQ v2 operations built for Linux, ~10x faster), FB_UPSTREAM (default upstream_cy if present, else upstream), MW_OST_WORKERS / MW_DOCKQ_WORKERS
(worker counts of the evaluator copy). The interpreter (--python) needs numpy, pandas, biopython, parallelbar, networkx: the FoldJet env with parallelbar on
PYTHONPATH (see eval.sbatch).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FB = Path(os.environ.get("FB_ROOT", str(Path.home() / "data/foldbench")))
UPSTREAM = Path(os.environ.get("FB_UPSTREAM", str(FB / "upstream_cy" if (FB / "upstream_cy").exists() else FB / "upstream")))
ALL_TARGETS = [
    "interface_protein_ligand",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_peptide",
    "interface_protein_dna",
    "interface_protein_rna",
    "monomer_protein",
    "monomer_dna",
    "monomer_rna",
]
RE_SEEDED = re.compile(r"^(?P<tid>.+)_seed(?P<seed>\d+)_sample(?P<sample>\d+)_pred$")


def ranking_of(target_dir: Path) -> dict[tuple[int, int], float]:
    f = target_dir / "ranking.csv"
    if not f.exists():
        return {}
    return {(int(s), int(k)): float(v) for s, k, v in (line.split(",") for line in f.read_text().splitlines()[1:])}


def build_reference(pred_dirs: list[Path], out_csv: Path, wanted: set[str] | None, top1_only: bool, require: int) -> tuple[set[str], int]:
    """prediction_reference.csv (pdb_id, seed, sample, ranking_score, prediction_path); paths are absolute."""
    rows, targets = [], set()
    for pred in pred_dirs:
        for tdir in sorted(d for d in pred.iterdir() if d.is_dir()):
            t = tdir.name
            if wanted is not None and t not in wanted:
                continue
            cifs = sorted(tdir.glob("*_pred.cif"))
            if len(cifs) < max(require, 1):
                continue  # half-written / failed target: never scored on fewer samples than the others
            rk = ranking_of(tdir)
            part = []
            for cif in cifs:
                m = RE_SEEDED.match(cif.stem)
                if m is None:
                    continue
                seed, sample = int(m["seed"]), int(m["sample"])
                part.append((t, seed, sample, rk.get((seed, sample), 1.0), cif))
            if top1_only and part:
                best = part[0]
                for r in part:  # first row in file order wins a tie (upstream idxmax)
                    if r[3] > best[3]:
                        best = r
                part = [best]
            targets.add(t)
            rows.extend(part)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as fh:
        fh.write("pdb_id,seed,sample,ranking_score,prediction_path\n")
        for t, seed, sample, score, cif in rows:
            fh.write(f"{t},{seed},{sample},{score},{cif.resolve()}\n")
    return targets, len(rows)


def write_filtered_targets(pdb_ids: set[str], tasks: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for t in tasks:
        src = FB / "targets" / f"{t}.csv"
        if not src.exists():
            continue
        with src.open() as fh:
            rows = list(csv.DictReader(fh))
        sub = [r for r in rows if r["pdb_id"] in pdb_ids]
        if not sub:
            continue
        with (out_dir / f"{t}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(sub)
        kept.append(t)
        print(f"  {t}: {len(sub)}/{len(rows)} rows over {len({r['pdb_id'] for r in sub})} predicted targets", flush=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, action="append", required=True)
    ap.add_argument("--eval-dir", type=Path, required=True, help="results land in <eval-dir>/<name>/")
    ap.add_argument("--name", required=True)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS, help="FoldBench task sets to score")
    ap.add_argument("--targets-file", type=Path, help="only these assemblies (one pdb_id per line)")
    ap.add_argument("--top1-only", action="store_true", help="score only the top-ranked sample per target (no oracle)")
    ap.add_argument("--require-samples", type=int, default=0, help="skip targets with fewer predictions than this (a run still writing)")
    ap.add_argument("--skip-ost", action="store_true", help="reuse the OST / DockQ tables in <eval-dir>/<name>/raw, only aggregate")
    a = ap.parse_args()
    eval_root = a.eval_dir.resolve()
    sub = eval_root / a.name
    wanted = set(a.targets_file.read_text().split()) if a.targets_file else None
    ref = sub / "prediction_reference.csv"
    ids, n = build_reference([p.resolve() for p in a.pred_dir], ref, wanted, a.top1_only, a.require_samples)
    print(f"prediction_reference.csv: {len(ids)} targets, {n} predictions -> {ref}", flush=True)
    if n == 0:
        sys.exit("no predictions found")
    print("filtered task CSVs:", flush=True)
    kept = write_filtered_targets(ids, a.targets, sub / "targets")
    if not kept:
        sys.exit("none of the predicted targets appear in the requested task CSVs")
    env = dict(os.environ, PATH=f"{UPSTREAM / 'bin'}:{os.environ['PATH']}")
    if not a.skip_ost:
        cmd = [str(a.python), "evaluate.py", "--targets_dir", str(sub / "targets"), "--evaluation_dir", str(eval_root), "--algorithm_name", a.name,
               "--ground_truth_dir", str((FB / "ground_truth").resolve()), "--targets", *a.targets]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=UPSTREAM, env=env, check=True)
    for raw in sorted((sub / "raw").glob("*.csv")):  # a pass without rows leaves a headerless csv that upstream's summary cannot read
        if "," not in raw.read_text(errors="ignore").partition("\n")[0]:
            print(f"  dropping empty {raw.name}")
            raw.unlink()
    have = [t for t in kept if (sub / "raw" / f"{t}_ost.csv").exists()]
    if not have:
        sys.exit("no raw OST results were produced")
    out = eval_root / f"summary_{a.name}_rank.csv"
    cmd = [str(a.python), "task_score_summary.py", "--evaluation_dir", str(eval_root), "--target_dir", str(sub / "targets"), "--output_path", str(out),
           "--algorithm_names", a.name, "--targets", *have, "--metric_type", "rank"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=UPSTREAM, env=env, check=True)  # upstream's own top-1 summary: the cross-check of score_tasks.py
    sys.path.insert(0, str(HERE))
    import score_tasks  # noqa: PLC0415

    t = score_tasks.scores(sub / "raw", sub / "targets")
    if a.top1_only:
        t["oracle"] = float("nan")  # one sample per target was scored: no oracle
    t.to_csv(eval_root / f"scores_{a.name}.csv", index=False)
    print(t.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
