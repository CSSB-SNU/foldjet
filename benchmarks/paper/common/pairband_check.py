"""Check of the pair-band decision (FoldJet `DiffusionHead._pair_band_rows`): banded and unbanded pair conditioning give the same structure bit for bit, and what the
decision does at a size.

    python pairband_check.py --model af3 --n 4 [--n 12 ...] [--fraction 0.4]

Per n, in one process (same weights, seeds, input): forward 1 with the banded path forced (PAIR_BAND_THRESHOLD_BYTES = 0), forward 2 with the unbanded path forced
(threshold = infinity), forward 3 with the DEFAULT decision; coordinates and pLDDT compared with `array_equal`. Prints the estimated unbanded peak, the limit the decision used
(min of the fixed cap and `PAIR_BAND_FREE_FRACTION` x allocator-spare memory) and which path the default took. A size at which the unbanded path itself runs out of memory
gives `unbanded_oom` (then only banded vs default is compared). One line `PAIRBAND {...}` per n. Run through Slurm / the GPU lock only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ffpaper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="af3")
ap.add_argument("--n", type=int, nargs="+", required=True)
ap.add_argument("--fraction", type=float, default=None)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
ffpaper.apply_arm_env("engine", args.model)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(HERE.parents[2] / "benchmarks/5i28_ff"))
import run_5i28_ff as H  # noqa: E402

import foldjet as ff  # noqa: E402
import foldjet.modules.dense.diffusion_head as dh  # noqa: E402

if args.fraction is not None:
    dh.PAIR_BAND_FREE_FRACTION = args.fraction
decisions: list[dict] = []
_orig = dh.DiffusionHead._pair_band_rows


def _spy(self, pair):
    rows = _orig(self, pair)
    decisions.append({"peak_gib": round(self._unbanded_peak_bytes(pair) / 2**30, 2), "limit_gib": round(self._band_limit_bytes(pair) / 2**30, 2),
                      "banded": rows is not None, "rows_per_band": rows})
    return rows


dh.DiffusionHead._pair_band_rows = _spy
recycles, template_n, extra = H.MODELS[args.model]
predictor = ff.load_model(args.model, config=H.make_config(recycles, extra, 0))
default_threshold = dh.PAIR_BAND_THRESHOLD_BYTES
rows = []
for n in args.n:
    spec = Path(f"/tmp/pairband_{args.model}_x{n}.yaml")
    spec.write_text(yaml.safe_dump(H.make_spec(n, template_n), sort_keys=False))
    outs, info = {}, {"model": args.model, "n": n, "tokens": 128 * n}
    for label, thr in (("banded", 0), ("unbanded", 1 << 62), ("default", default_threshold)):
        dh.PAIR_BAND_THRESHOLD_BYTES = thr
        decisions.clear()
        torch.cuda.reset_peak_memory_stats()
        try:
            r = predictor(ff.load_inputs(spec))
            outs[label] = (r.coords.float().numpy().copy(), r.plddt.float().numpy().copy())
            info[f"{label}_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            r = None
        except Exception as error:  # noqa: BLE001
            info[f"{label}_error"] = ("oom: " if ffpaper.oom_of(error) else "") + str(error).splitlines()[0][:160]
        info[f"{label}_decisions"] = list(decisions[:2])
        torch.cuda.empty_cache()
    dh.PAIR_BAND_THRESHOLD_BYTES = default_threshold
    def same(a, b):
        return bool(a in outs and b in outs and np.array_equal(outs[a][0], outs[b][0]) and np.array_equal(outs[a][1], outs[b][1]))
    info["banded_equals_unbanded"] = same("banded", "unbanded") if "unbanded" in outs else None
    info["banded_equals_default"] = same("banded", "default") if "default" in outs else None
    if "banded" in outs:
        info["finite"] = bool(np.isfinite(outs["banded"][0]).all())
    rows.append(info)
    print("PAIRBAND " + json.dumps(info), flush=True)
if args.out:
    args.out.write_text(json.dumps({"meta": ffpaper.run_meta(experiment="pairband_check"), "rows": rows}, indent=1))
