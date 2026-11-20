#!/usr/bin/env python3
"""F6 collect: run logs of run_f6.sh -> CSVs the paper scripts read. Standard library only.

  collect.py --out-dir ~/foldforge-scratch/paper/F6_training_speed/<card> [--csv-dir DIR]

Writes (default into --out-dir, copy the final CSVs into the repository / paper data):
  f6_runs_<card>.csv      one row per process (an attempt at one augmentation chunk): status ok | oom | error, step walls, peak memory, versions
  f6_summary_<card>.csv   per model x arm: the chunk used (first value of the ladder that ran), seconds per training example = median of optimizer
                          steps 1-5 / micro steps per step (step 0 = graph capture / compile, excluded), median / min / max over repeats
  steps_<card>.txt        the shape of paper data/train_speed/steps.txt:  ts[48]-<model>-<arm>_<job>.log <step0> <step1> ... <step5>  (the
                          repeat with the median seconds per example; "ts48-" = all 48 augmentations in one pass, "ts-" = chunked)
A failed (OOM) attempt is a result: its row stays in f6_runs with status oom and is not plotted.
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

LOG = re.compile(r"^(?P<model>\w+?)_(?P<arm>[A-Za-z0-9]+)_a(?P<chunk>\d+)_r(?P<rep>\d+)\.log$")


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    m = LOG.match(path.name)
    row = {"model": m["model"], "arm": m["arm"], "aug_chunk": int(m["chunk"]), "repeat": int(m["rep"]), "log": path.name}
    meta = re.search(r"^RUNMETA (.*)$", text, re.M)
    if meta:
        row.update(dict(kv.split("=", 1) for kv in meta.group(1).split() if "=" in kv and kv.split("=", 1)[0] in ("host", "slurm_job", "backend", "compile", "steps", "accum")))
    walls = [float(w) for w in re.findall(r"^BENCHSTEP it=\d+ wall_s=([\d.]+)", text, re.M)]
    bench = next((json.loads(line[6:]) for line in text.splitlines() if line.startswith("BENCH ")), None)
    oom = re.search(r"^OOM (.*)$", text, re.M)
    rc = re.search(r"^rc=(\d+)", text, re.M)
    if oom:
        row["status"], row["error"] = "oom", oom.group(1)[:120]
    elif bench is None:
        row["status"], row["error"] = "error", (f"rc={rc.group(1)}" if rc else "unfinished")
    else:
        row["status"], row["error"] = "ok", ""
    accum = int(row.get("accum", 8) or 8)
    row["step_walls_s"] = " ".join(f"{w:.3f}" for w in walls)
    if len(walls) >= 2:
        timed = walls[1:6]  # step 0 excluded
        row["median_step_s"] = round(statistics.median(timed), 4)
        row["per_example_s"] = round(statistics.median(timed) / accum, 4)
        row["first_step_s"] = walls[0]
    if bench:
        for k_out, k_in in (("peak_alloc_gib", "peak_allocated_gib"), ("peak_reserved_gib", "peak_reserved_gib"), ("nonfinite_steps", "nonfinite_steps"),
                            ("foldforge_commit", "commit"), ("engine_version", "engine_version"), ("engine_file", "engine_file"), ("gpu", "gpu")):
            row[k_out] = bench.get(k_in)
        served = {c: v.get("served", {}) for c, v in (bench.get("engine_table") or {}).items()}
        row["engine_served"] = json.dumps(served, sort_keys=True) if row.get("backend") == "miniworld" else ""
        row["triton_calls"] = sum(s.get("triton", 0) for s in served.values()) if row.get("backend") == "miniworld" else ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir)
    csv_dir = Path(a.csv_dir) if a.csv_dir else out
    card = out.name
    rows = [parse(p) for p in sorted(out.glob("*_a*_r*.log")) if LOG.match(p.name)]
    cols = ["model", "arm", "backend", "compile", "aug_chunk", "repeat", "status", "error", "first_step_s", "median_step_s", "per_example_s", "step_walls_s",
            "peak_alloc_gib", "peak_reserved_gib", "nonfinite_steps", "triton_calls", "engine_served", "steps", "accum", "foldforge_commit", "engine_version",
            "engine_file", "gpu", "host", "slurm_job", "log"]
    csv_dir.mkdir(parents=True, exist_ok=True)
    with (csv_dir / f"f6_runs_{card}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary, steps_lines = [], []
    for model in sorted({r["model"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["model"] == model}):
            mine = [r for r in rows if r["model"] == model and r["arm"] == arm]
            ok = [r for r in mine if r["status"] == "ok" and "per_example_s" in r]
            tried = sorted({r["aug_chunk"] for r in mine})
            if not ok:
                summary.append({"model": model, "arm": arm, "status": "no run fitted", "chunks_tried": " ".join(map(str, tried))})
                continue
            chunk = min(ok, key=lambda r: tried.index(r["aug_chunk"]))["aug_chunk"]  # same value for the repeats
            use = [r for r in ok if r["aug_chunk"] == chunk]
            vals = [r["per_example_s"] for r in use]
            mid = sorted(use, key=lambda r: r["per_example_s"])[len(use) // 2]
            summary.append({"model": model, "arm": arm, "status": "ok", "aug_chunk": chunk, "aug_passes": 1 if chunk == 0 else 48 // chunk,
                            "chunks_tried": " ".join(map(str, tried)), "repeats": len(use), "per_example_s_median": round(statistics.median(vals), 4),
                            "per_example_s_min": min(vals), "per_example_s_max": max(vals), "peak_alloc_gib": max(r.get("peak_alloc_gib") or 0 for r in use),
                            "first_step_s": mid["first_step_s"], "engine_version": mid.get("engine_version"), "foldforge_commit": mid.get("foldforge_commit"),
                            "gpu": mid.get("gpu"), "triton_calls": mid.get("triton_calls")})
            prefix = "ts48" if chunk == 0 else "ts"
            steps_lines.append(f"{prefix}-{model}-{arm}_{mid.get('slurm_job', 'local')}.log {mid['step_walls_s']} ")
    scols = ["model", "arm", "status", "aug_chunk", "aug_passes", "chunks_tried", "repeats", "per_example_s_median", "per_example_s_min", "per_example_s_max",
             "peak_alloc_gib", "first_step_s", "engine_version", "foldforge_commit", "gpu", "triton_calls"]
    with (csv_dir / f"f6_summary_{card}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    (csv_dir / f"steps_{card}.txt").write_text("\n".join(steps_lines) + "\n")
    print(f"{len(rows)} runs, {len(summary)} model x arm rows -> {csv_dir}")


if __name__ == "__main__":
    main()
