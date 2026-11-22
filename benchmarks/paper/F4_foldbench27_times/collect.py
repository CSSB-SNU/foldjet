"""F4: per-target model times of one card -> CSV in the shape of the paper's table (`tables/supp_foldbench_times.csv`).

    python collect.py --card h100 [--root ~/foldforge-scratch/paper/F4]

Reads  <root>/<card>/rows/<arm>__<model>.jsonl (run_foldbench27.py).
Writes F4_times_<card>.csv   model,target,task,tokens,anthropic,engine,torch,engine_fp8,...  seconds = the warm model time; an arm that failed or did not run
                             is an empty cell (the failure is in F4_failures_<card>.csv: status, error, oom flag), never a number.
                             Extra columns: ratios torch/engine, torch/engine_fp8, and the run information (commit, engine, job, host).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = ["anthropic", "engine", "torch", "engine_fp8"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090", "b200"])
    parser.add_argument("--root", type=Path, default=Path.home() / "foldforge-scratch/paper/F4")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str], dict] = {}
    failures = []
    for jl in sorted((args.root / args.card / "rows").glob("*__*.jsonl")):
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            if r["status"] in ("started",):
                continue
            key = (r["model"], r["target"])
            c = cells.setdefault(key, {"model": r["model"], "target": r["target"], "task": r["task"], "tokens": r["tokens"]})
            arm = r["arm"]
            if r["status"] == "ok" and r.get("model_seconds_warm_median") is not None:
                c[arm] = r["model_seconds_warm_median"]
                c[f"{arm}_peak_gib"] = r.get("peak_alloc_gib", "")
                for k in ("foldforge_commit", "engine_version_installed", "engine_version_loaded", "slurm_job_id", "host", "gpu"):
                    c.setdefault(f"{arm}_{k}", r.get(k, ""))
                c.pop(f"{arm}_failed", None)
            else:
                c.pop(arm, None)
                failures.append({"model": r["model"], "target": r["target"], "arm": arm, "status": r["status"], "oom": r.get("oom", ""), "error": r.get("error", r.get("note", ""))})
    rows = sorted(cells.values(), key=lambda c: (c["model"], c["tokens"], c["target"]))
    for c in rows:
        for arm in ("engine", "engine_fp8"):
            c[f"torch_over_{arm}"] = round(c["torch"] / c[arm], 3) if c.get("torch") and c.get(arm) else ""
    head = ["model", "target", "task", "tokens", *ARMS]
    cols = head + sorted({k for c in rows for k in c} - set(head))
    with (args.out / f"F4_times_{args.card}.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, cols)
        w.writeheader()
        w.writerows(rows)
    with (args.out / f"F4_failures_{args.card}.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, ["model", "target", "arm", "status", "oom", "error"])
        w.writeheader()
        w.writerows(failures)
    print(f"{len(rows)} (model, target) rows, {len(failures)} failed cells -> {args.out}")


if __name__ == "__main__":
    main()
