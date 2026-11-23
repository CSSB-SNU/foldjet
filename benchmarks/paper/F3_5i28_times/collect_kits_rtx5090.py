"""F3 Anthropic-kit arm on the RTX 5090 (cssb-rtx): CSV `data/F3_kits_rtx5090.csv` from the kit queue records (pure python + stdlib; CPU only).

    python collect_kits_rtx5090.py --f3 ~/mwe/agents/f3_kit/reps [--existing data/existing/kits_rtx5090_20261118.csv] [--out data/F3_kits_rtx5090.csv]

Repetition 1 = the 2026-11-18 serial campaign (`data/existing/kits_rtx5090_20261118.csv`: one row per kit x n with the steady value and the reason of a
non-number). Repetitions 2 and 3 = `<f3>/rep<r>/done/<item>` markers of `run_rep.sh` (one job each, same harness, same cell list: only the cells that
produced a number in repetition 1 are repeated; a cell that failed deterministically in repetition 1 - RAM cap / refused on sm_120 / GPU OOM in every
mode - is not re-run and keeps its recorded reason).
Same protocol as the H100 / A100 kit rows: forward 1 (cold) and 2 dropped, steady = median of the rest; the cell value = the median over the repetitions of the
per-repetition steady values; a cell with any failed repetition is not a number. Columns: the first twelve are those of `F3_kits_<card>.csv` (collect.py),
then `reps`, `value_reps`, `kit_active` (the kit's own exit gate: partial / not active on sm_120) and `reason`.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

HEAD = ["kit", "n", "tokens", "value_s", "mode", "status", "n_timed", "forwards_s", "templates", "source", "job", "node", "reps", "value_reps", "kit_active", "reason"]
SIZES = (1, 2, 4, 8, 12, 16)
# kits without a row in repetition 1: the 2026-11-23 probe at n = 1 (~/mwe/agents/f3_kit/reps/probe_*.out, kit logs next to it). A probe that cannot run at
# n = 1 cannot run at a larger n (same weights in host RAM / same refused kernels), so every size gets the n = 1 reason.
PROBE = {
    "esmfold2_fast": ("ram_cap_16GB", "fast", "probe n=1 2026-11-23: killed by systemd-oomd (16 GB RAM cap of mwe.slice) while loading the 24.4 GiB checkpoint shards, "
                      "before the first forward; same as esmfold2 in repetition 1"),
    "intellifold2": ("refused_on_sm120", "fast", "probe n=1 2026-11-23: kit exit 3, 0/35 predictions: FPF_TRIATT / FPF_TRIMUL = fallback:no_tiles_cc120 (the kit core's "
                     "Pallas serve layer has no tile table for compute capability 12.0; a mode is all of its levers), same refusal as af3_jax"),
    "intellifold2_notxla": ("refused_on_sm120", "fast", "probe n=1 2026-11-23 (KITTAG=notxla, MODEL_OPT_LEVERS_OFF=TRIATT_XLA): NOTXLA_REASON"),
}
KIT_OF_ITEM = {"openfold3c4": "openfold3", "rf3": "rosettafold3", "rf3c4": "rosettafold3", "af3_torchbig": "af3_torch"}


def steady(fwd: list[float]) -> float | None:
    rest = fwd[2:]
    return statistics.median(rest) if rest else None


def parse_fwd(text: str) -> list[float]:
    # the marker is `mode=.. rc=.. earlier=[<failed attempts, each with its own KIT line>] KIT ... fwd=[...]`: the final attempt is the LAST fwd=[...]
    found = re.findall(r"fwd=\[([^\]]*)\]", text)
    return [float(x) for x in re.split(r"[ ,;]+", found[-1].strip()) if x] if found else []


def rep1_rows(path: Path) -> dict[tuple, dict]:
    """The per-cell record of repetition 1 (the final row of a cell: a `big` retry row replaces a fast row that ran out of memory)."""
    cells: dict[tuple, dict] = {}
    for r in csv.DictReader(path.open()):
        key = (r["kit"], int(r["n"]))
        fwd = [float(x) for x in r["forwards_s"].split(";") if x]
        row = {"fwd": fwd, "mode": r["mode"], "status": r["status"], "item": r["item"], "notes": r["notes"], "tpl": r["templates"],
               "attempts": r["attempts"], "num": steady(fwd) is not None, "full": len(fwd) >= 7}
        old = cells.get(key)
        # a retry row (`big` mode, 4-query `c4`) with a number wins over a failed or shorter row of the same cell
        if old is None or (row["num"] and (not old["num"] or (row["full"] and not old["full"]))):
            cells[key] = row
    return cells


def rep_marker_rows(rep_done: Path) -> dict[tuple, dict]:
    cells: dict[tuple, dict] = {}
    for marker in sorted(rep_done.glob("*")):
        m = re.match(r"(?:kit|sep)_(.+)_x(\d+)$", marker.name)
        if not m:
            continue
        item, n = m.group(1), int(m.group(2))
        kit = KIT_OF_ITEM.get(item, item)
        text = marker.read_text().strip()
        fwd = parse_fwd(text)
        mode = (re.match(r"mode=(\w+)", text) or [None, "fast"])[1]
        full = len(fwd) >= (4 if item.endswith("c4") else 7)
        prev = cells.get((kit, n))
        if prev and prev["full"] and not full:
            continue  # a failed re-run must not displace a complete cell
        if prev and prev["ok"] and not prev["full"] and not steady(fwd):
            continue
        cells[(kit, n)] = {"fwd": fwd, "mode": mode, "ok": steady(fwd) is not None, "full": full, "text": text, "item": item}
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f3", type=Path, required=True)
    ap.add_argument("--existing", type=Path, default=Path(__file__).resolve().parent / "data/existing/kits_rtx5090_20261118.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data/F3_kits_rtx5090.csv")
    args = ap.parse_args()
    r1 = rep1_rows(args.existing)
    later = {r: rep_marker_rows(args.f3 / f"rep{r}" / "done") for r in (2, 3)}
    rows = []
    for (kit, n), c in sorted(r1.items()):
        v1 = steady(c["fwd"]) if c["num"] else None
        per = {1: (v1, c["fwd"], c["mode"])}
        for r, cells in later.items():
            x = cells.get((kit, n))
            if x is not None:
                per[r] = (steady(x["fwd"]) if x["ok"] else None, x["fwd"], x["mode"])
        values = [v for v, _, _ in per.values() if v is not None]
        failed_reps = [r for r, (v, _, _) in per.items() if v is None]
        ok = bool(values) and not failed_reps
        active = "not active (kit exit 3)" if "kit_not_active" in c["status"] else ("partial (kit exit 3)" if "kit_partial" in c["status"] else "")
        rows.append({
            "kit": kit, "n": n, "tokens": 128 * n, "value_s": f"{statistics.median(values):.3f}" if ok else "",
            "mode": ";".join(sorted({m for _, _, m in per.values()})),
            "status": "ok" if ok else (c["status"] if not values else "failed_in_repetition"),
            "n_timed": ";".join(str(max(0, len(f) - 2)) for _, f, _ in per.values()),
            "forwards_s": " | ".join(";".join(f"{x:.3f}" for x in f) for _, f, _ in per.values()),
            "templates": c["tpl"], "source": "F3 rtx5090 kits (rep1 = 2026-11-18 serial campaign; rep2/3 = run_rep.sh)",
            "job": "local (no Slurm)", "node": "cssb-rtx", "reps": len(values), "value_reps": ";".join(f"{v:.3f}" for v in values),
            "kit_active": active, "reason": "" if ok else (c["notes"] or c["attempts"]).replace("\n", " ")})
    have = {k for k, _ in r1}
    for kit, (status, mode, reason) in PROBE.items():
        if kit in have:
            continue
        for n in SIZES:
            rows.append({"kit": kit, "n": n, "tokens": 128 * n, "value_s": "", "mode": mode, "status": status, "n_timed": "0", "forwards_s": "",
                         "templates": "", "source": "F3 rtx5090 kits (probe 2026-11-23; n = 1)", "job": "local (no Slurm)", "node": "cssb-rtx",
                         "reps": 0, "value_reps": "", "kit_active": "not active (kit exit 3)" if status == "refused_on_sm120" else "", "reason": reason})
    rows.sort(key=lambda r: (r["kit"], r["n"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as h:
        w = csv.DictWriter(h, HEAD)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
