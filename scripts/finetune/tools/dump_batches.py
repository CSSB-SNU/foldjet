"""Dump the worker outputs (SampleDataset.__getitem__, i.e. exactly what a loader worker yields) for stream segments. usage: dump_batches.py ADAPTER_DIR OUTDIR NPROC N RANK:START [RANK:START ...]"""

import multiprocessing as mp
import os
import sys
import time

AD, OUT, NP, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SEGS = [tuple(map(int, a.split(":"))) for a in sys.argv[5:]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ds = {}


def work(args):
    rank, chunk = args
    sys.path.insert(0, AD)
    import ft_adapter
    import torch

    if rank not in _ds:
        s = ft_adapter.Settings()
        s.workers = 0
        _ds[rank] = ft_adapter.SampleDataset(s, 0, rank)
    done = 0
    for key in chunk:
        out = _ds[rank][key]
        info = {
            k: v
            for k, v in out["info"].items()
            if k in ("tokens", "atoms", "msa_rows", "templates", "bonds", "attempts")
        }
        torch.save(
            {
                "name": out["name"],
                "features": out["features"],
                "atom_positions": out["atom_positions"],
                "atom_mask": out["atom_mask"],
                "bond_pairs": out["bond_pairs"],
                "present_index": out["present_index"],
                "present_valid": out["present_valid"],
                "info": info,
            },
            os.path.join(OUT, "r%d_p%d.pt" % (rank, key[0])),
        )
        done += 1
    return rank, done, os.getpid(), time.process_time()


if __name__ == "__main__":
    import positions

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for rank, start in SEGS:
        keys = positions.get_keys(AD, rank, 2, start, N)
        jobs += [(rank, keys[i::NP]) for i in range(NP)]
    t = time.time()
    with mp.get_context("spawn").Pool(NP) as p:
        res = p.map(work, jobs, chunksize=1)
    print(
        "dumped",
        sum(r[1] for r in res),
        "examples in %.0fs" % (time.time() - t),
        "AD",
        AD,
        flush=True,
    )
