"""F8: fine-tuned Protenix v1 (10 epochs) vs the released Protenix v1 on the whole FoldBench, top-1 and oracle.

    python collect.py --released EVAL_DIR_RELEASED --ft10 EVAL_DIR_FT10 [--stamp-released F8RUN/stamp.json --stamp-ft10 F8RUN/stamp.json] --out results/

EVAL_DIR_* = the --eval-dir of F2's evaluate.py (scores_*.csv: rules of upstream task_score_summary.py). Writes
  f8_task_scores.csv   long (arm, task, metric, n, top1, oracle + run stamp)                      [same columns as F2's f2_task_scores.csv]
  f8_top1.csv          ',released,ft10,diff'  (the shape of the paper's tables/finetune_foldbench.csv; values in percent)
  f8_oracle.csv        the same for the oracle (best of 25)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F2 = HERE.parent / "F2_foldbench_accuracy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", type=Path, required=True)
    ap.add_argument("--ft10", type=Path, required=True)
    ap.add_argument("--stamp-released", type=Path)
    ap.add_argument("--stamp-ft10", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    cmd = [sys.executable, str(F2 / "collect.py"), "--arm", f"released={a.released}", "--arm", f"ft10={a.ft10}", "--out", str(a.out), "--prefix", "f8"]
    for name, p in (("released", a.stamp_released), ("ft10", a.stamp_ft10)):
        if p:
            cmd += ["--stamp", f"{name}={p}"]
    subprocess.run(cmd, check=True)
    for col, name in (("top1", "f8_top1.csv"), ("oracle", "f8_oracle.csv")):
        w = pd.read_csv(a.out / f"f8_{col}_wide.csv", index_col=0)
        w["diff"] = (w["ft10"] - w["released"]).round(2)
        w.to_csv(a.out / name)
        (a.out / f"f8_{col}_wide.csv").unlink()
    print((a.out / "f8_top1.csv").read_text())


if __name__ == "__main__":
    main()
