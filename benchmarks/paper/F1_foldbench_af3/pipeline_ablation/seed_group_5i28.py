"""Seed grouping (FoldBench runner --group 5) vs target size, measured on 5I28 (F1 ablation addendum, 2026-11-27).

What the FoldBench worker does per target (fb_multi_worker.py): five trunk_seeds, each `run_trunk` on its own; then the diffusion of
the five retained trunks either one by one (`sample_diffusion`, 5 samples each = 5 sampler runs of batch 5) or grouped
(`sample_diffusion_group`, ONE sampler run of batch 25). Grouping only changes the diffusion part.

Here, per size (5I28, n identical chains of 128 tokens, real MSA + 4 templates, AF3 bf16, 200 steps, compile + CUDA graphs + bucketing,
trunk scope, the worker's settings), each round:
  trunks:   run_trunk for the five seeds (kept),
  single:   sample_diffusion of each trunk (5 x batch 5),
  grouped:  sample_diffusion_group of the five trunks (1 x batch 25),
the order single/grouped alternating between rounds. 7 rounds, the first 2 dropped (cold / graph capture), median of the other 5.
One JSON line per size in OUT (resumable: sizes already there are skipped).

    python seed_group_5i28.py OUT.jsonl N [N ...]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "5i28_ff"))
import run_5i28_ff as H  # noqa: E402

import foldjet as ff  # noqa: E402

SEEDS = [42, 66, 101, 2024, 8888]
ROUNDS, DROP = 7, 2
CONFIG = {
    "precision": "bf16",
    "backend": "miniworld",
    "diffusion": {"steps": 200},
    "execution": {"compile": True, "cuda_graph": True, "bucketing": True, "scope": "trunk", "max_graphs": 32},
}


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> int:
    out = Path(sys.argv[1])
    sizes = [int(x) for x in sys.argv[2:]]
    done = {json.loads(line)["n"] for line in out.read_text().splitlines()} if out.exists() else set()
    spec_dir = out.parent / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    with ff.load_model("af3", config=CONFIG, device="cuda:0") as model:
        for n in sizes:
            if n in done:
                continue
            spec = spec_dir / f"x{n}.yaml"
            spec.write_text(yaml.safe_dump(H.make_spec(n, 4), sort_keys=False))
            inputs = ff.load_inputs(spec)
            rows = []
            try:
                for r in range(ROUNDS):
                    trunks, t_trunk = [], []
                    for s in SEEDS:
                        tr, dt = timed(lambda s=s: model.run_trunk(inputs, seed=s))
                        trunks.append(tr)
                        t_trunk.append(dt)
                    res = {}
                    order = ("single", "grouped") if r % 2 == 0 else ("grouped", "single")
                    for mode in order:
                        if mode == "single":
                            _, res[mode] = timed(
                                lambda: [model.sample_diffusion(tr, samples=5, steps=200, seed=s) for tr, s in zip(trunks, SEEDS, strict=True)]
                            )
                        else:
                            _, res[mode] = timed(lambda: model.sample_diffusion_group(trunks, samples=5, steps=200, seed=SEEDS[0]))
                    for tr in trunks:
                        tr.close()
                    rows.append({"trunk5_s": sum(t_trunk), **{f"{k}_s": v for k, v in res.items()}})
                    print(f"n={n} round {r}: trunk x5 {sum(t_trunk):.2f} s  single {res['single']:.2f} s  grouped {res['grouped']:.2f} s", flush=True)
                keep = rows[DROP:]
                med = {k: statistics.median(x[k] for x in keep) for k in keep[0]}
                cell = {
                    "n": n,
                    "tokens": 128 * n,
                    "status": "ok",
                    **{f"{k}_median": round(v, 3) for k, v in med.items()},
                    "diffusion_saving_pct": round(100 * (1 - med["grouped_s"] / med["single_s"]), 1),
                    "target_saving_pct": round(
                        100 * (med["single_s"] - med["grouped_s"]) / (med["trunk5_s"] + med["single_s"]), 1
                    ),
                    "rounds": rows,
                    "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                    "gpu": torch.cuda.get_device_name(0),
                }
            except torch.cuda.OutOfMemoryError as e:
                cell = {"n": n, "tokens": 128 * n, "status": "oom", "error": str(e).splitlines()[0][:200], "rounds": rows}
            print("CELL " + json.dumps({k: v for k, v in cell.items() if k != "rounds"}), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(cell) + "\n")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
