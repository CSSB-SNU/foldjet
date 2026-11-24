"""results/<tag>/{timing-rank*.json,summary.json,smi.csv} -> results/summary.csv (+ printed table).

Per run: optimizer-step wall time median over steps >= SKIP+1 (warm; steps 1..SKIP dropped), per-micro-step time, samples/s
(world x grad_accum x 1 sample / step time), all-reduce time per step (comm_ms: CUDA events around the bucketed gradient all-reduce,
which includes waiting for the slowest rank), per-rank median replay (compute) time, peak memory, loss / sync checks.
"""
import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/psk6950/mwe/agents/mw_ddp/results")
SKIP = 2
rows = []
for d in sorted(root.iterdir()):
    s = d / "summary.json"
    if not s.exists():
        continue
    summ = json.loads(s.read_text())
    ranks = {}
    for f in sorted(glob.glob(str(d / "timing-rank*.json"))):
        r = int(f.split("rank")[-1].split(".")[0])
        ranks[r] = [x for x in json.loads(Path(f).read_text()) if x["step"] > SKIP]
    if 0 not in ranks or not ranks[0]:
        continue
    world = summ["world"]
    ga = len(ranks[0][0]["recycles"])
    wall = st.median(x["wall_ms"] for x in ranks[0])
    comm = st.median(x["comm_ms"] for x in ranks[0])
    opt = st.median(x["optimizer_ema_ms"] for x in ranks[0])
    replay = {r: st.median(x["replay_gpu_ms"] for x in v) / ga for r, v in ranks.items()}
    copy = st.median(x["copy_gpu_ms"] for x in ranks[0]) / ga
    peak = max(g["peak_reserved_gib"] for g in summ["ranks"])
    peak_alloc = max(g["peak_alloc_gib"] for g in summ["ranks"])
    smi = [l.split(",") for l in (d / "smi.csv").read_text().splitlines() if l.strip()] if (d / "smi.csv").exists() else []
    smi_mem = max((int(x[1].split()[0]) for x in smi), default=0) / 1024
    rows.append(dict(
        tag=d.name, world=world, grad_accum=ga, steps=len(ranks[0]), step_ms=round(wall, 1), micro_ms=round(wall / ga, 1),
        samples_per_s=round(world * ga / (wall / 1000), 3), comm_ms=round(comm, 1), comm_pct=round(100 * comm / wall, 1),
        opt_ema_ms=round(opt, 1), copy_ms_per_micro=round(copy, 1),
        replay_ms_per_micro_by_rank=" ".join(f"{r}:{v:.1f}" for r, v in sorted(replay.items())),
        peak_alloc_gib=round(peak_alloc, 2), peak_reserved_gib=round(peak, 2), smi_peak_gib=round(smi_mem, 1),
        loss_first=" ".join(f"{g['loss_first']:.4f}" for g in summ["ranks"]), loss_last=" ".join(f"{g['loss_last']:.4f}" for g in summ["ranks"]),
        finite=all(g["finite"] for g in summ["ranks"]), max_param_diff=summ["max_param_diff"],
        triton=sum(sum(g["triton_launches"].values()) for g in summ["ranks"]),
    ))
# scaling efficiency vs the 1-GPU run (GPU1) with the same grad_accum
base = {r["grad_accum"]: r["samples_per_s"] for r in rows if r["tag"].startswith("n1g1_")}
for r in rows:
    b = base.get(r["grad_accum"])
    r["efficiency_pct"] = round(100 * r["samples_per_s"] / (r["world"] * b), 1) if b else ""
keys = list(rows[0].keys()) if rows else []
with open(root / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(" | ".join(str(r[k]) for k in keys))
