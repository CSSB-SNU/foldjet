"""F1: raw run records -> the CSVs / JSON the paper scripts read (GPU-hours of the Anthropic kit vs FoldForge bf16 / FP8 on the whole FoldBench).

    python collect.py --kit KITRUN --ff af3_bf16_shard=FFRUN --ff af3_fp8_shard=FFRUN [--ff-multi af3_bf16_multi=FFRUN ...] --out results/

KITRUN  = run_kit.sbatch output dir ($PAPER_OUT/F1/h100/<name>): s<k>/{run_info.txt, out/, out_big/, forward_fast.log, forward_big.log}, job_info.txt
FFRUN   = run_ff.sbatch output dir: launch.txt, stamp.json, and either pred_s<k>/ (LAUNCH=shard: job_info.txt per shard) or pred/ (LAUNCH=multi)
Which launch mode a FoldForge run used is read from its launch.txt and written to every row: the paper must say which one the number is.

Writes
  f1_shard_wall.csv    one row per (arm, shard): launch, targets, done, failed, wall_s, wall_h (kit: fast pass, big pass, total) + run stamp
  f1_summary.csv       one row per arm: launch, gpu_h (kit and shard mode: sum of the shard wall times; multi mode: GPUs x wall), slower-shard wall, done, failed,
                       speed-up over the kit (gpu_h ratio and slower-shard ratio)
  f1_per_target.csv    arm, target, shard, status (ok | failed | oom | missing), mode (fast | big | ff), tokens, gpu_s (kit: sum of forward_s over the seeds;
                       FoldForge: run.json gpu_s), peak_alloc_gib, fp8 saturation checks (FoldForge FP8)
  f1_record.json       the same numbers in the layout of the paper's data/foldbench_full_af3.json ("kit_wall_h", "kit_fallback", "ff": per arm)
Failures and OOM are results: listed in f1_per_target.csv (status), never dropped, and drawn nowhere by the paper scripts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def stamp_of(run: Path) -> dict:
    f = run / "stamp.json"
    if not f.exists():
        return {}
    st = json.loads(f.read_text())
    return {"foldforge_commit": st.get("foldforge_commit"), "engine_version": st.get("engine_version"), "gpu": ((st.get("gpus") or [""])[0].split(",") + ["", ""])[1].strip(),
            "node": st.get("node"), "slurm_job_id": st.get("slurm_job_id")}


def kv(path: Path) -> dict[str, str]:
    """`key value...` and `key=value` tokens of a job_info.txt / run_info.txt (the last occurrence wins)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(start|end|fast_end|big_end)\s+([\d.]+)(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
            for t in m.group(3).split():
                if "=" in t:
                    k, v = t.split("=", 1)
                    out[f"{m.group(1)}_{k}"] = v
            continue
        for t in line.split():
            if "=" in t:
                k, v = t.split("=", 1)
                out[k] = v
    return out


def forward_lines(*logs: Path) -> dict[str, dict]:
    """Kit forward.log lines `[forward] <t> seed=<s> ok tokens=.. forward_s=.. peak_gb=..` / `FAILED OutOfMemoryError(...)`, per target."""
    t: dict[str, dict] = {}
    for log in logs:
        if not log.exists():
            continue
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"\[forward\] (\S+) seed=(\d+) (\w+)", line)
            if not m:
                continue
            d = t.setdefault(m.group(1), {"gpu_s": 0.0, "n_ok": 0, "failed": None, "tokens": None, "peak_gb": 0.0})
            if m.group(3) == "ok":
                d["gpu_s"] += float(re.search(r"forward_s=([\d.]+)", line).group(1))
                d["n_ok"] += 1
                d["tokens"] = int(re.search(r"tokens=(\d+)", line).group(1))
                d["peak_gb"] = max(d["peak_gb"], float(re.search(r"peak_gb=([\d.]+)", line).group(1)))
            else:
                d["failed"] = "oom" if "OutOfMemory" in line else "error"
    return t


def collect_kit(run: Path) -> tuple[list[dict], list[dict]]:
    walls, targets = [], []
    st = stamp_of(run)
    job = run / "job_info.txt"
    job_id = re.search(r"JOB (\d+)", job.read_text()).group(1) if job.exists() and re.search(r"JOB (\d+)", job.read_text()) else None
    for sd in sorted(run.glob("s[0-9]*")):
        k = sd.name[1:]
        info = kv(sd / "run_info.txt")
        shard_targets = [p.stem for p in (sd / "inputs").glob("*.json")] if (sd / "inputs").exists() else []
        fwd = forward_lines(sd / "forward_fast.log", sd / "forward_big.log")
        done_fast = {p.parent.name for p in (sd / "out").glob("*/*_model.cif")} if (sd / "out").exists() else set()
        done_big = {p.parent.name for p in (sd / "out_big").glob("*/*_model.cif")} if (sd / "out_big").exists() else set()
        fast_s = float(info["end"]) - float(info["start"]) if "end" in info else float("nan")
        big_s = float(info["big_end"]) - float(info["fast_end"]) if "big_end" in info and "fast_end" in info else 0.0
        walls.append({"arm": "kit", "launch": "one kit process per shard", "shard": k, "targets": len(shard_targets), "done": len(done_fast | done_big), "failed": len(set(shard_targets) - done_fast - done_big),
                      "wall_s": round(fast_s + big_s, 1), "wall_h": round((fast_s + big_s) / 3600, 3), "fast_wall_h": round(fast_s / 3600, 3), "big_wall_h": round(big_s / 3600, 3),
                      "n_big": len(done_big), "node": info.get("host"), "slurm_job_id": job_id or info.get("job"), **{k_: v for k_, v in st.items() if k_ not in ("node", "slurm_job_id")}})
        for t in shard_targets:
            f = fwd.get(t, {})
            status = "ok" if t in done_fast | done_big else ("oom" if f.get("failed") == "oom" else "failed")
            targets.append({"arm": "kit", "target": t, "shard": k, "status": status, "mode": "big" if t in done_big else "fast", "tokens": f.get("tokens"),
                            "gpu_s": round(f["gpu_s"], 2) if f.get("n_ok") else None, "peak_alloc_gib": f.get("peak_gb") or None, "fp8_saturated": None, "fp8_redo": None})
    return walls, targets


def collect_ff(arm: str, run: Path) -> tuple[list[dict], list[dict]]:
    launch_txt = (run / "launch.txt").read_text() if (run / "launch.txt").exists() else ""
    launch = re.search(r"launch=(\w+)", launch_txt).group(1) if "launch=" in launch_txt else "unknown"
    precision = re.search(r"precision=(\w+)", launch_txt).group(1) if "precision=" in launch_txt else "?"
    st = stamp_of(run)
    walls, targets = [], []
    n_gpus = len([g for g in json.loads((run / "stamp.json").read_text()).get("gpus", [])]) if (run / "stamp.json").exists() else 2
    dirs = sorted(run.glob("pred_s[0-9]*")) if launch == "shard" else [run / "pred"]
    for d in dirs:
        shard = d.name[6:] if launch == "shard" else "all"
        info = kv((d if launch == "shard" else run) / "job_info.txt")
        wall = float(info["end"]) - float(info["start"]) if "end" in info and "start" in info else float("nan")
        runs = [json.loads(p.read_text()) | {"_t": p.parent.name} for p in d.glob("*/run.json")]
        failed = [p.parent.parent.name for p in d.glob("*/failed/marker")]
        listed = (run / f"targets_s{shard}.txt").read_text().split() if launch == "shard" and (run / f"targets_s{shard}.txt").exists() else None
        n_t = len(listed) if listed else len(runs) + len(failed)
        walls.append({"arm": arm, "launch": launch, "shard": shard, "targets": n_t, "done": len(runs), "failed": len(failed), "wall_s": wall, "wall_h": round(wall / 3600, 3),
                      "fast_wall_h": None, "big_wall_h": None, "n_big": None, "gpus": 1 if launch == "shard" else n_gpus, **st})
        for r in runs:
            fp8 = r.get("fp8") or {}
            targets.append({"arm": arm, "target": r["_t"], "shard": shard, "status": "ok", "mode": "ff", "tokens": None, "gpu_s": r.get("gpu_s"), "peak_alloc_gib": r.get("peak_alloc_gib"),
                            "fp8_saturated": fp8.get("saturated"), "fp8_redo": fp8.get("redo"), "fp8_unresolved": fp8.get("unresolved"), "graphs": (r.get("multi") or {}).get("graphs")})
        for t in failed:
            targets.append({"arm": arm, "target": t, "shard": shard, "status": "failed", "mode": "ff", "tokens": None, "gpu_s": None, "peak_alloc_gib": None, "fp8_saturated": None, "fp8_redo": None})
        for t in set(listed or []) - {r["_t"] for r in runs} - set(failed):
            targets.append({"arm": arm, "target": t, "shard": shard, "status": "missing", "mode": "ff", "tokens": None, "gpu_s": None, "peak_alloc_gib": None, "fp8_saturated": None, "fp8_redo": None})
    for w in walls:
        w["precision"] = precision
    return walls, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, help="kit run dir")
    ap.add_argument("--ff", action="append", default=[], metavar="ARM=DIR", help="FoldForge run dirs (shard or multi, read from launch.txt)")
    ap.add_argument("--sizes", type=Path, default=Path(__file__).resolve().parents[2] / "foldbench/foldbench_af3_sizes.tsv", help="AF3 token counts (FoldForge featurisation)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    walls, targets = [], []
    if a.kit:
        w, t = collect_kit(a.kit)
        walls += w
        targets += t
    for x in a.ff:
        arm, d = x.split("=", 1)
        w, t = collect_ff(arm, Path(d))
        walls += w
        targets += t
    W, T = pd.DataFrame(walls), pd.DataFrame(targets)
    if a.sizes.exists():
        sizes = pd.read_csv(a.sizes, sep="\t").rename(columns={"target": "target", "af3_tokens": "tokens_af3"})
        T = T.merge(sizes[["target", "tokens_af3"]], on="target", how="left")
        T["tokens"] = T["tokens"].astype(float).fillna(T["tokens_af3"])
        T = T.drop(columns="tokens_af3")
    # summary: GPU-hours = sum of shard wall times (kit; FoldForge shard mode) | GPUs x wall (FoldForge multi mode)
    rows = []
    for arm, g in W.groupby("arm", sort=False):
        launch = g.launch.iloc[0]
        gpu_s = g.wall_s.sum() if launch != "multi" else float(g.wall_s.iloc[0]) * int(g.gpus.iloc[0])
        rows.append({"arm": arm, "launch": launch, "gpu_h": round(gpu_s / 3600, 3), "slower_shard_wall_h": round(g.wall_s.max() / 3600, 3), "targets": int(g.targets.sum()), "done": int(g.done.sum()), "failed": int(g.failed.sum()),
                     **{k: g[k].iloc[0] for k in ("foldforge_commit", "engine_version", "gpu", "node", "slurm_job_id") if k in g}})
    S = pd.DataFrame(rows)
    if "kit" in set(S.arm):
        kit = S[S.arm == "kit"].iloc[0]
        S["speedup_gpu_h"] = (kit.gpu_h / S.gpu_h).round(3)
        S["speedup_slower_shard"] = (kit.slower_shard_wall_h / S.slower_shard_wall_h).round(3)
    a.out.mkdir(parents=True, exist_ok=True)
    W.to_csv(a.out / "f1_shard_wall.csv", index=False)
    S.to_csv(a.out / "f1_summary.csv", index=False)
    T.sort_values(["arm", "target"]).to_csv(a.out / "f1_per_target.csv", index=False)
    rec: dict = {"what": "AlphaFold 3 (DeepMind weights) on the whole FoldBench (1,522 targets, seeds 42,66,101,2024,8888, 5 samples, 200 steps, 11 trunk passes); GPU-hours = sum of the shard wall times (kit and FoldForge shard mode) or GPUs x wall (FoldForge multi mode); see f1_summary.csv",
                 "summary": S.to_dict("records"), "shard_wall": W.to_dict("records")}
    kw = W[W.arm == "kit"]
    if len(kw):
        rec["kit_wall_h"] = kw.wall_h.tolist()
        rec["kit_fallback"] = {"fast_h": kw.fast_wall_h.tolist(), "big_h": kw.big_wall_h.tolist(), "total_h": kw.wall_h.tolist(), "n_big": int(kw.n_big.sum()), "done": int(kw.done.sum())}
        rec["kit_gpu_h"] = float(S[S.arm == "kit"].gpu_h.iloc[0])
    (a.out / "f1_record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(S.to_string(index=False))
    print(W.to_string(index=False))


if __name__ == "__main__":
    main()
