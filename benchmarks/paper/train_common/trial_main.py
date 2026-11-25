"""One F10 trial = this process: scripts/train_protenix1.py (unchanged, run through runpy) with the trunk Pairformer set to PF_BLOCKS blocks.

PF_BLOCKS=N (optional): after the model is loaded and before training is set up, the trunk Pairformer stack is cut to its first N blocks or
extended to N blocks by copying its blocks in turn (deepcopy: the same shapes and kernels, the weights of the copies are those of the original
blocks; this is a memory / speed probe, not a trained model). The repository is not changed. The same hook is used for every backend.
Prints PF_BLOCKS (kept / full), and at exit PF_PEAK (CUDA peak allocated / reserved) and, on an out-of-memory error, an `OOM` line (as the
script's own __main__ does, which runpy does not run). Other arguments go to the training script.
"""

import copy
import os
import runpy
import sys
import traceback

import torch

import foldjet as ff

N = int(os.environ["PF_BLOCKS"]) if os.environ.get("PF_BLOCKS") else None
_REF: dict = {}
_orig = ff.load_model


def _load(*a, **k):
    pred = _orig(*a, **k)
    ev = pred._network.evoformer
    _REF["pred"] = pred
    if N is not None:
        base = list(ev.trunk_pairformer)
        full = len(base)
        blocks = base[:N] if N <= full else base + [copy.deepcopy(base[i % full]) for i in range(full, N)]
        ev.trunk_pairformer = torch.nn.ModuleList(blocks)
        ev.pairformer_num_layer = N
        print(f"PF_BLOCKS kept={len(ev.trunk_pairformer)} of {full}", flush=True)
    return pred


ff.load_model = _load
script = os.path.join(os.environ["FF_REPO"], "scripts/train_protenix1.py")
sys.argv = [script, *sys.argv[1:]]
_rc = 0
try:
    runpy.run_path(script, run_name="train_protenix1")["main"]()
except torch.OutOfMemoryError as error:
    print("OOM", str(error)[:240].replace("\n", " "), flush=True)
    _rc = 3
except Exception as error:  # noqa: BLE001 - recorded as an error result, never retried
    msg = str(error)
    # an out-of-memory error inside a CUDA graph capture / compile worker surfaces as a RuntimeError
    if "out of memory" in msg.lower() or "cudaErrorMemoryAllocation" in msg:
        print("OOM", msg[:240].replace("\n", " "), flush=True)
        _rc = 3
    else:
        print("ERROR", type(error).__name__, msg[:400].replace("\n", " "), flush=True)
        print("".join(traceback.format_exc().splitlines(True)[-40:]), flush=True)
        _rc = 4
if torch.cuda.is_available():
    print(
        f"PF_PEAK blocks={N} alloc_gib={torch.cuda.max_memory_allocated() / 2**30:.2f} "
        f"reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f}",
        flush=True,
    )
sys.stdout.flush()
os._exit(_rc)
