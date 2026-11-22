"""F9, one probe: ONE forward of n identical 5I28 chains (n x 128 tokens) for one predictor and one arm, in a fresh process.

    python run_size.py --model af3 --arm engine|engine_fp8|torch --n 24 --out DIR [--exec benchmark|eager]

Input = the F3 protocol (real MSA + 4 distinct real templates, 4 trunk passes, 200 steps, 5 samples, bf16; `benchmarks/5i28_ff/run_5i28_ff.py` helpers).
Exactly one forward (`benchmark_repeats` 0: no warm repeat). A size counts as completed (`status: ok`) when the forward returns and the coordinates and
pLDDT are finite; out of memory -> `oom` (with the allocator's numbers); non-finite output -> `nonfinite`; anything else -> `failed` with the reason.
`--exec benchmark` (default for engine arms) = compile + CUDA graphs + bucketing as the 5I28 benchmark; `--exec eager` (default for the torch arm) = no
compile, no CUDA graph: PyTorch's operators as such (as F4). Writes DIR/n<N>.json (one cell) and prints one `CELL {...}` line.
Peak memory = the model forward's own peak (`model_peak_*_bytes` of the report) and the process peak (`torch.cuda.max_memory_*`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import ffpaper  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", required=True, choices=ffpaper.PREDICTORS_5I28)
parser.add_argument("--arm", required=True, choices=["engine", "engine_fp8", "torch"])
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--exec", dest="exec_mode", choices=["benchmark", "eager"], default=None)
args = parser.parse_args()
ffpaper.apply_arm_env(args.arm, args.model)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "benchmarks/5i28_ff"))
import run_5i28_ff as H  # noqa: E402

import foldforge as ff  # noqa: E402


def main() -> int:
    exec_mode = args.exec_mode or ("eager" if args.arm == "torch" else "benchmark")
    recycles, template_n, extra = H.MODELS[args.model]
    config = H.make_config(recycles, extra, 0)
    config["backend"] = ffpaper.BACKEND[args.arm]
    if exec_mode == "eager":
        config["execution"].update(compile=False, cuda_graph=False)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = ffpaper.run_meta(experiment="F9", model=args.model, arm=args.arm, n=args.n, exec=exec_mode, config=config)
    cell = {"experiment": "F9", "model": args.model, "arm": args.arm, "n": args.n, "tokens": H.TOKENS_PER_CHAIN * args.n, "exec": exec_mode,
            **{k: meta[k] for k in ("card", "gpu", "gpu_total_gib", "host", "slurm_job_id", "foldforge_commit", "engine_version_installed",
                                    "engine_version_loaded", "engine_loaded_from", "alloc_conf")}}
    spec = args.out / f"x{args.n}.yaml"
    spec.write_text(yaml.safe_dump(H.make_spec(args.n, template_n), sort_keys=False))
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        predictor = ff.load_model(args.model, config=config)
        if args.arm != "torch":
            ffpaper.backend_plan()
        if args.arm == "engine_fp8":
            ffpaper.fp8_check()
        torch.cuda.reset_peak_memory_stats()
        result = predictor(ff.load_inputs(spec))
        report = result.report
        coords = result.coords.float().numpy()
        plddt = result.plddt.float().numpy() if result.plddt is not None else None
        finite = bool(np.isfinite(coords).all() and (plddt is None or np.isfinite(plddt).all()))
        gib = 2**30
        cell |= {
            "status": "ok" if finite else "nonfinite", "finite": finite,
            "model_s": report.get("model_seconds") or report.get("model_seconds_cold"),
            "model_peak_alloc_gib": (report.get("model_peak_allocated_bytes") or 0) / gib,
            "model_peak_reserved_gib": (report.get("model_peak_reserved_bytes") or 0) / gib,
            "plddt_mean": None if plddt is None else float(plddt.mean()),
        }
        if args.arm != "torch":
            tab = ffpaper.backend_table(predictor)
            cell["all_cuda"] = tab["all_cuda"] if tab else None
        if args.arm == "engine_fp8":
            cell["fp8_saturated"] = ffpaper.fp8_check()
    except Exception as error:  # noqa: BLE001 - a failed probe is a result
        oom = ffpaper.oom_of(error)
        text = str(oom or error)
        cell |= {"status": "oom" if oom else "failed", "error": text.splitlines()[0][:300] if text else repr(error)[:300],
                 "fragmented": bool(oom and "reserved by PyTorch but unallocated" in text)}
        if "exceeds largest inference bucket" in text:
            cell["status"] = "bucket"
        if not oom:
            cell["traceback"] = traceback.format_exc()[-900:]
    cell["process_peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    cell["process_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    # A non-OOM CUDA error (illegal memory access, reported asynchronously) with the allocator at >= 98 % of the card is the out-of-memory edge, not a failure:
    # B200 chai1 n=40 (reserved 176.9 of 178.35 GiB); the same cell under CUDA_LAUNCH_BLOCKING gives a clean CUDA out of memory. The original error stays in `oom_classified_from`.
    total = cell.get("gpu_total_gib")
    if cell.get("status") == "failed" and "CUDA error" in str(cell.get("error", "")) and total and cell["process_peak_reserved_gib"] >= 0.98 * total:
        cell["oom_classified_from"] = cell["error"]
        cell["status"] = "oom"
    cell["wall_s"] = round(time.time() - t0, 1)
    (args.out / f"n{args.n}.json").write_text(json.dumps(cell, indent=1))
    print("CELL " + json.dumps(cell), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
