"""F3, FoldForge side: seconds per forward on 5I28, n = 1, 2, 4, 8, 12, 16 chains (128 .. 2,048 tokens), one predictor, one arm, one job.

    python run_ff.py --model af3 --arm engine|engine_fp8 --out DIR [--sizes 1 2 4 ...] [--repeats 6] [--rep 1]

Protocol: the 5I28 benchmark (docs/guides/BENCHMARK-5I28.md; `benchmarks/5i28_ff/run_5i28_ff.py` is the harness, imported unchanged: real MSA +
4 distinct real templates, 4 trunk passes, 200 steps, 5 samples, bf16, compile + CUDA graphs, `scope: trunk`). Per size ONE predictor call with
`benchmark_repeats` = REPEATS warm repeats after the cold forward; the cell value `steady_s` is the median of the warm forwards after the first
warm one is dropped (the kit side drops forward 1 = cold and forward 2, then takes the median of the rest; with REPEATS 6 that is 5 timed
forwards on both sides). Forward = one complete model forward (trunk, sampler, confidence heads), CUDA-synchronized, no featurization /
loading / writing (`model_seconds_warm` of the report).

Arms: engine = the engine's bf16 kernels; engine_fp8 = the opt-in FP8 path (H100 only): MINIWORLD_FP8=safe, MINIWORLD_FP8_MODEL=<model>, lazy
guards; `fp8.check()` after every cell records whether a route saturated (`fp8_saturated`; None for the bf16 arm).
One row per cell in <out>/cells.jsonl; run information in <out>/meta.json. Resumable (finished and out-of-memory cells are final).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import ffpaper  # noqa: E402  (before foldforge: sets the FP8 environment)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", required=True, choices=ffpaper.PREDICTORS_5I28)
parser.add_argument("--arm", required=True, choices=["engine", "engine_fp8"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
parser.add_argument("--repeats", type=int, default=6, help="warm repeats after the cold forward (the 2026-11-18 steady runs: 6)")
parser.add_argument("--rep", type=int, default=1, help="repetition id (the three repeats run as three different jobs)")
args = parser.parse_args()
ffpaper.apply_arm_env(args.arm, args.model)

import torch  # noqa: E402
import yaml  # noqa: E402

REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "benchmarks/5i28_ff"))
import run_5i28_ff as H  # noqa: E402

import foldforge as ff  # noqa: E402


def main() -> int:
    recycles, template_n, extra = H.MODELS[args.model]
    config = H.make_config(recycles, extra, args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    cells = args.out / "cells.jsonl"
    (args.out / "specs").mkdir(exist_ok=True)
    meta = ffpaper.run_meta(experiment="F3", model=args.model, arm=args.arm, rep=args.rep, repeats=args.repeats, config=config)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    done = {r["n"] for r in ffpaper.read_jsonl(cells) if r["model"] == args.model and r["arm"] == args.arm and r["rep"] == args.rep
            and r["status"] in ("ok", "oom")}
    todo = [n for n in sorted(args.sizes) if n not in done]
    print(f"RUN F3 model={args.model} arm={args.arm} rep={args.rep} sizes={todo} repeats={args.repeats} gpu={meta['gpu']} host={meta['host']} "
          f"job={meta['slurm_job_id']} engine={meta['engine_version_installed']} from {meta['engine_loaded_from']} ff={meta['foldforge_commit'][:9]}", flush=True)
    if not todo:
        return 0
    predictor = ff.load_model(args.model, config=config)
    if args.arm == "engine_fp8":
        ffpaper.fp8_check()  # clear a flag left from loading
    for n in todo:
        spec = args.out / "specs" / f"x{n}.yaml"
        spec.write_text(yaml.safe_dump(H.make_spec(n, template_n), sort_keys=False))
        torch.cuda.reset_peak_memory_stats()
        cell = {"experiment": "F3", "model": args.model, "arm": args.arm, "rep": args.rep, "n": n, "tokens": H.TOKENS_PER_CHAIN * n,
                "card": meta["card"], "gpu": meta["gpu"], "host": meta["host"], "slurm_job_id": meta["slurm_job_id"],
                "foldforge_commit": meta["foldforge_commit"], "engine": meta["engine_version_installed"], "engine_from": meta["engine_loaded_from"]}
        t0 = time.time()
        try:
            result = predictor(ff.load_inputs(spec))
            report = result.report
            warm = list(report["model_seconds_warm"])
            cell |= {
                "status": "ok", "cold_s": report["model_seconds_cold"], "warm_s": warm, "warm_median_s": report["model_seconds_warm_median"],
                "steady_s": ffpaper.steady_median([report["model_seconds_cold"], *warm], drop=2),
                "peak_alloc_gib": (report.get("model_peak_allocated_bytes") or 0) / 2**30,
                "peak_reserved_gib": (report.get("model_peak_reserved_bytes") or 0) / 2**30,
                "cuda_graph_captures": report.get("cuda_graph_captures"), "compile": report.get("compile"),
            }
            result = None
        except Exception as error:  # noqa: BLE001 - a failed cell is data
            oom = ffpaper.oom_of(error)
            cell |= {"status": "oom" if oom else "failed", "error": str(oom or error).splitlines()[0][:300],
                     "max_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            if not oom:
                cell["traceback"] = traceback.format_exc()[-800:]
            traceback.clear_frames(error.__traceback__)
            error = oom = None
        if args.arm == "engine_fp8":
            cell["fp8_saturated"] = ffpaper.fp8_check()  # True: a static-scale route saturated since the last check (the cell is then suspect)
        cell["wall_s"] = round(time.time() - t0, 1)
        ffpaper.append_jsonl(cells, cell)
        print(f"CELL {args.model} {args.arm} n={n} {cell['status']} steady={cell.get('steady_s')} warm={cell.get('warm_s')} cold={cell.get('cold_s')} "
              f"peak={cell.get('peak_alloc_gib')} fp8_saturated={cell.get('fp8_saturated')} wall={cell['wall_s']}s", flush=True)
        before, after = H.release_size_memory()
        print(f"FREE after n={n}: reserved {before / 2**30:.1f} -> {after / 2**30:.1f} GiB", flush=True)
    if args.arm == "engine_fp8":
        (args.out / "fp8_status.json").write_text(json.dumps(ffpaper.fp8_status(), default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
