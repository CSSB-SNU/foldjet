#!/usr/bin/env python3
"""F10 collect: results.jsonl + summaries of bisect.py -> CSVs. Standard library only.

  collect.py --out-dir ~/foldforge-scratch/paper/F10_training_limit/<card> [--csv-dir DIR]
"""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir)
    csv_dir = Path(a.csv_dir) if a.csv_dir else out
    card = out.name
    csv_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()] if (out / "results.jsonl").exists() else []
    cols = ["card", "part", "arm", "tokens", "atoms", "blocks", "ckpt", "chunk", "status", "error", "peak_alloc_gib", "peak_reserved_gib", "median_step_s", "first_step_s",
            "wall_s", "commit", "engine_version", "engine_file", "gpu", "host", "slurm_job", "log"]
    with (csv_dir / f"f10_trials_{card}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    limits = []
    for s in sorted(out.glob("summary_*.json")):
        j = json.loads(s.read_text())
        at = j.get("at_max") or {}
        part = j["part"]
        limits.append({
            "card": j["card"], "part": part, "arm": j["arm"],
            "what": "longest crop (tokens)" if part == "a" else "most trunk Pairformer blocks",
            "tokens": j["max"] if part == "a" else int(part[1:]),
            "atoms": (-(-j["max"] * 32 // 3 // 32) * 32) if part == "a" else (-(-int(part[1:]) * 32 // 3 // 32) * 32),
            "blocks": 48 if part == "a" else j["max"], "checkpointing": "on" if part == "a" else "off",
            "max": j["max"], "first_failing": j["first_failing"], "note": j["note"], "chunk_of_search": j["chunk_of_search"],
            "chunk_at_max": at.get("chunk"), "peak_alloc_gib_at_max": at.get("peak_alloc_gib"), "peak_reserved_gib_at_max": at.get("peak_reserved_gib"),
            "median_step_s_at_max": at.get("median_step_s"),
        })
    lc = list(limits[0]) if limits else ["card", "part", "arm", "max"]
    with (csv_dir / f"f10_limits_{card}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lc)
        w.writeheader()
        w.writerows(limits)
    print(f"{len(rows)} trials, {len(limits)} limits -> {csv_dir}")


if __name__ == "__main__":
    main()
