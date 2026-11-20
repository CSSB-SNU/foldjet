"""F9: the limit.json / sizes.jsonl of one card -> CSVs.

    python collect.py --card h100 [--root ~/foldforge-scratch/paper/F9]

Reads  <root>/<card>/<arm>/<predictor>/{limit.json,sizes.jsonl}      (arm: engine, engine_fp8, torch, kit)
Writes F9_<card>.csv          one row per predictor x arm: card, predictor, arm, largest_ok_contiguous (copies), tokens_ok_contiguous, largest_ok_any, first_not_ok,
                              first_not_ok_status, non_monotonic, peak_alloc_gib_at_edge, process_peak_alloc_gib_at_edge, stop_reason, tried, run information
       F9_<card>_sizes.csv    one row per probe (status, memory, seconds, error)
A failed / out-of-memory probe is a row with its status, never a number in a plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090"])
    ap.add_argument("--root", type=Path, default=Path.home() / "foldforge-scratch/paper/F9")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, sizes = [], []
    for lim in sorted((a.root / a.card).glob("*/*/limit.json")):
        arm, pred = lim.parent.parent.name, lim.parent.name
        s = json.loads(lim.read_text())
        probes = [json.loads(x) for x in (lim.parent / "sizes.jsonl").read_text().splitlines() if x.strip()]
        ident = next((p for p in probes if p.get("foldforge_commit")), {})
        rows.append({"card": a.card, **s, "tried": ";".join(map(str, s.get("tried", []))),
                     **{k: ident.get(k, "") for k in ("gpu", "gpu_total_gib", "host", "slurm_job_id", "foldforge_commit", "engine_version_installed", "engine_loaded_from", "exec")}})
        for p in sorted(probes, key=lambda p: p["n"]):
            sizes.append({"card": a.card, "predictor": pred, "arm": arm, **{k: p.get(k, "") for k in (
                "n", "tokens", "status", "model_peak_alloc_gib", "model_peak_reserved_gib", "process_peak_alloc_gib", "process_peak_reserved_gib", "model_s", "wall_s",
                "probe_wall_s", "retried_expandable", "mode", "fp8_saturated", "slurm_job_id", "error")}})
    for name, data in ((f"F9_{a.card}.csv", rows), (f"F9_{a.card}_sizes.csv", sizes)):
        cols = list(dict.fromkeys(k for r in data for k in r))
        with (a.out / name).open("w", newline="") as h:
            w = csv.DictWriter(h, cols)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {a.out / name} ({len(data)} rows)")


if __name__ == "__main__":
    main()
