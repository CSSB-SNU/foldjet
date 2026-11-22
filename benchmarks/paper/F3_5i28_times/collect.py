"""F3: turn the run directories of one card into CSVs the paper scripts read (pure python + stdlib; runs anywhere, also as a CPU job).

    python collect.py --card h100 [--root ~/foldforge-scratch/paper/F3] [--out DIR]

Reads   <root>/<card>/{engine,engine_fp8}_rep<r>/<model>/cells.jsonl      (run_ff.py)
        <root>/<card>/kits_rep<r>/done_final/{kit_*,sep_*}                (the kit harness's done markers: `KIT ... fwd=[...]`)
Writes  F3_foldforge_<card>.csv   run,model,n,status,s,...   first five columns = `data/records_5i28/5i28_foldforge_301_bf16_fp8.csv` (run = bf16 | fp8)
        F3_kits_<card>.csv        kit,n,tokens,value_s,mode,status,n_timed,forwards_s,templates,source,job,node,reps,...   first columns =
                                  `5i28_kits_baseline_20261118.csv`
The value of a cell = the median over the repetitions of the per-repetition steady value (forward 1 = cold and forward 2 dropped, median of the rest);
a cell counts as `ok` only when every repetition that ran is ok (otherwise `oom` / `failed`, recorded, never drawn). `reps` = the number of repetitions
that produced a value; the per-repetition values are in `s_reps` / `value_reps`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SIZES = (1, 2, 4, 8, 12, 16)
RUN_NAME = {"engine": "bf16", "engine_fp8": "fp8"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def steady(forwards: list[float], drop: int = 2) -> float | None:
    rest = forwards[drop:]
    return statistics.median(rest) if rest else None


def foldforge_rows(card_dir: Path) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    for jl in sorted(card_dir.glob("*_rep*/*/cells.jsonl")):
        for row in read_jsonl(jl):
            cells.setdefault((row["arm"], row["model"], row["n"]), []).append(row)
    rows = []
    for (arm, model, n), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        last = {}
        for r in reps:  # the last row of a repetition wins (a rerun of a failed cell)
            last[r["rep"]] = r
        values = [r["steady_s"] for r in last.values() if r["status"] == "ok" and r.get("steady_s") is not None]
        bad = [r["status"] for r in last.values() if r["status"] != "ok"]
        rows.append({
            "run": RUN_NAME[arm], "model": model, "n": n, "status": ("ok" if values and not bad else (bad[0] if bad else "missing")),
            "s": f"{statistics.median(values):.4f}" if values and not bad else "",
            "card": next(iter(last.values())).get("card", ""), "tokens": 128 * n, "reps": len(values),
            "s_reps": ";".join(f"{v:.4f}" for v in values),
            "peak_alloc_gib": f"{max((r.get('peak_alloc_gib') or 0) for r in last.values()):.2f}",
            "fp8_saturated": ("" if arm == "engine" else str(any(bool(r.get("fp8_saturated")) for r in last.values()))),
            "jobs": ";".join(sorted({str(r.get("slurm_job_id")) for r in last.values()})),
            "foldforge_commit": ";".join(sorted({r.get("foldforge_commit", "")[:9] for r in last.values()})),
            "engine": ";".join(sorted({str(r.get("engine_loaded") or r.get("engine")) for r in last.values()})),
            "all_cuda": ";".join(sorted({str(r.get("all_cuda")) for r in last.values()})),
        })
    return rows


def fwd_list(text: str) -> list[float]:
    m = re.search(r"fwd=\[([^\]]*)\]", text)
    return [float(x) for x in re.split(r"[ ,]+", m.group(1).strip()) if x] if m else []


def kit_rows(card_dir: Path) -> list[dict]:
    per: dict[tuple, list[dict]] = {}
    for done in sorted(card_dir.glob("kits_rep*/done_final")):
        rep = int(re.search(r"kits_rep(\d+)", str(done)).group(1))
        for marker in sorted(done.iterdir()):
            m = re.match(r"(?:kit|sep)_(.+)_x(\d+)$", marker.name)
            if not m:
                continue
            kit, n = m.group(1), int(m.group(2))
            text = marker.read_text().strip()
            f = fwd_list(text)
            v = steady(f)
            mode = (re.search(r"mode=(\w+)", text) or [None, "fast"])[1]
            if "fast=oom" in text:
                mode = "big"
            status = "ok" if v is not None else ("oom" if re.search(r"oom|RESOURCE", text) else "failed")
            per.setdefault((kit, n), []).append({
                "rep": rep, "v": v, "f": f, "mode": mode, "status": status, "job": (re.search(r"job=(\d+)", text) or [None, ""])[1],
                "node": (re.search(r"host=(\w+)", text) or [None, ""])[1], "tpl": (re.search(r"templates_real=\S+", text) or [""])[0]})
    rows = []
    for (kit, n), reps in sorted(per.items()):
        values = [r["v"] for r in reps if r["v"] is not None]
        bad = [r["status"] for r in reps if r["v"] is None]
        rows.append({
            "kit": kit, "n": n, "tokens": 128 * n, "value_s": f"{statistics.median(values):.3f}" if values and not bad else "",
            "mode": ";".join(sorted({r["mode"] for r in reps})), "status": "ok" if values and not bad else (bad[0] if bad else "missing"),
            "n_timed": ";".join(str(max(0, len(r["f"]) - 2)) for r in reps), "forwards_s": " | ".join(";".join(f"{x:.3f}" for x in r["f"]) for r in reps),
            "templates": ";".join(sorted({r["tpl"] for r in reps})), "source": f"F3 {card_dir.name}", "job": ";".join(r["job"] for r in reps),
            "node": ";".join(sorted({r["node"] for r in reps})), "reps": len(values), "value_reps": ";".join(f"{v:.3f}" for v in values)})
    return rows


def write(path: Path, rows: list[dict], head: list[str]) -> None:
    cols = head + [k for k in (rows[0] if rows else {}) if k not in head]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090", "b200"])
    parser.add_argument("--root", type=Path, default=Path.home() / "foldforge-scratch/paper/F3")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    card_dir = args.root / args.card
    write(args.out / f"F3_foldforge_{args.card}.csv", foldforge_rows(card_dir), ["run", "model", "n", "status", "s"])
    write(args.out / f"F3_kits_{args.card}.csv", kit_rows(card_dir),
          ["kit", "n", "tokens", "value_s", "mode", "status", "n_timed", "forwards_s", "templates", "source", "job", "node"])


if __name__ == "__main__":
    main()
