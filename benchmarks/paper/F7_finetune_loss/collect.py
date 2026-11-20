#!/usr/bin/env python3
"""F7 collect: Slurm logs of the two fine-tuning runs -> the CSV / text / JSON the paper scripts read. Standard library only.

  collect.py --logs DIR  --out DIR  [--engine-glob 'slurm_f7-engine_*.log'] [--pytorch-glob 'slurm_f7-pytorch_*.log']

Writes into --out:
  engine_steps.txt, pytorch_steps.txt   `job=<id> STEP n lr=... loss=... t=...s`, the shape of data/finetune/*_steps.txt (fig2_training_accuracy.py)
  finetune_steps.csv                    step, then per run: loss terms, dt (seconds of that step), hours (cumulative)
  finetune.json                         steps, median s/step, hours, common steps, per-term correlation and mean difference
A step repeated after a resume keeps its last record. The fine-tune is the PyTorch run to the SAME step count as the engine run: the JSON says
whether it is (`same_step_count`).
"""
import argparse
import json
import re
import statistics
from pathlib import Path

TERMS = ["loss", "diffusion", "bond", "distogram", "plddt", "pde", "pae", "resolved", "gnorm"]


def read_run(logs: Path, pattern: str) -> tuple[list[str], dict[int, dict]]:
    lines, rows = [], {}
    last: dict[str, float] = {}
    for log in sorted(logs.glob(pattern), key=lambda p: (len(p.name), p.name)):
        job = re.search(r"_(\d+)\.log$", log.name)
        jid = job.group(1) if job else log.stem
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("STEP"):
                continue
            step = int(line.split()[1])
            t = float(re.search(r"t=([\d.]+)s", line).group(1))
            kv = {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.e+]+)", line) if k != "t"}
            lines.append(f"job={jid} {line}")
            rows[step] = {"step": step, "dt": t - last.get(jid, 0.0), **kv}
            last[jid] = t
    return lines, rows


def corr(x, y):
    return statistics.correlation(x, y) if len(x) > 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-glob", default="slurm_f7-engine_*.log")
    ap.add_argument("--pytorch-glob", default="slurm_f7-pytorch_*.log")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, pat in (("engine", a.engine_glob), ("pytorch", a.pytorch_glob)):
        lines, rows = read_run(Path(a.logs), pat)
        (out / f"{name}_steps.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        cum = 0.0
        for s in sorted(rows):
            cum += rows[s]["dt"]
            rows[s]["hours"] = cum / 3600
        runs[name] = rows
    common = sorted(set(runs["engine"]) & set(runs["pytorch"]))
    cols = ["step"]
    for name in runs:
        cols += [f"{name}_{c}" for c in (*TERMS, "dt", "hours")]
    with (out / "finetune_steps.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for s in sorted(set(runs["engine"]) | set(runs["pytorch"])):
            cells = [str(s)]
            for name in runs:
                r = runs[name].get(s, {})
                cells += [f"{r[c]:.6g}" if c in r else "" for c in (*TERMS, "dt", "hours")]
            f.write(",".join(cells) + "\n")
    stats: dict = {"steps": {k: (max(v) + 1 if v else 0) for k, v in runs.items()}, "common_steps": len(common), "terms": {}}
    stats["same_step_count"] = stats["steps"]["engine"] == stats["steps"]["pytorch"] and stats["steps"]["engine"] > 0
    stats["s_per_step_median"] = {k: round(statistics.median(r["dt"] for r in v.values()), 1) if v else None for k, v in runs.items()}
    stats["hours"] = {k: round(v[max(v)]["hours"], 1) if v else None for k, v in runs.items()}
    if len(common) > 2:
        for t in ("loss", "diffusion", "bond", "distogram", "plddt", "pde", "pae", "resolved"):
            e = [runs["engine"][s][t] for s in common]
            p = [runs["pytorch"][s][t] for s in common]
            de = [b - c for b, c in zip(e[1:], e[:-1], strict=True)]
            dp = [b - c for b, c in zip(p[1:], p[:-1], strict=True)]
            me, mp = statistics.fmean(e), statistics.fmean(p)
            stats["terms"][t] = {"r": round(corr(e, p), 3), "r_diff": round(corr(de, dp), 3), "mean_engine": round(me, 4), "mean_pytorch": round(mp, 4),
                                 "mean_rel_diff_pct": round(100 * (me - mp) / mp, 3) if mp else None}
    (out / "finetune.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("steps", "common_steps", "same_step_count", "s_per_step_median", "hours")}))


if __name__ == "__main__":
    main()
