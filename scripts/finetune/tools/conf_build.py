"""Build conformer-cache shards: for every CCD code of CODES (json list) and the first K seeds of the deterministic frames pass, the RDKit conformer of
fast_ref.embed. Resumable (a finished shard is skipped), one log line per code, stops starting new shards after MAXSEC.
usage: conf_build.py ADAPTER_DIR CODES.json K SHARD_SIZE NPROC OUTDIR [MAXSEC]
"""

import json
import multiprocessing as mp
import os
import pickle
import sys
import time

AD, CODES, K, SHARD, NP, OUT = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    sys.argv[6],
)
MAXSEC = float(sys.argv[7]) if len(sys.argv) > 7 else 480
T0 = time.time()
_ctx = {}


def build_shard(job):
    si, codes = job
    sys.path.insert(0, AD)
    import fast_ref
    from alphafold3.data.tools import rdkit_utils as ru
    from alphafold3.model import features

    from foldforge import paths
    from foldforge.data.ccd import CCDDatabase

    if "ccd" not in _ctx:
        db = CCDDatabase(paths.ccd_db())
        _ctx["db"] = db
        _ctx["ccd"] = db.af3_ccd(None)
        _ctx["seeds"] = fast_ref.frames_seeds(K)
    ccd = _ctx["ccd"]
    seeds = _ctx["seeds"]
    out = {}
    with _ctx["db"].activate():
        for code in codes:
            t = time.time()
            e = fast_ref._entry(code, ccd, features, ru, None)
            if e is None or e["mol"] is None:
                print("shard", si, code, "no mol (skipped)", flush=True)
                continue
            for s in seeds:
                out[(code, s, None)] = fast_ref.embed(e["mol"], code, s, None)
            print(
                "shard %d code %s atoms %d %.1fs"
                % (si, code, e["mol"].GetNumAtoms(), time.time() - t),
                flush=True,
            )
    path = os.path.join(OUT, "conf_%05d_K%d.pkl" % (si, K))
    pickle.dump(out, open(path + ".tmp", "wb"), protocol=5)
    os.replace(path + ".tmp", path)
    return si, len(out)


if __name__ == "__main__":
    codes = json.load(open(CODES))
    os.makedirs(OUT, exist_ok=True)
    jobs = [
        (i, codes[i * SHARD : (i + 1) * SHARD])
        for i in range((len(codes) + SHARD - 1) // SHARD)
    ]
    jobs = [
        j
        for j in jobs
        if not os.path.exists(os.path.join(OUT, "conf_%05d_K%d.pkl" % (j[0], K)))
    ]
    print("shards to build", len(jobs), flush=True)

    def gen():
        for j in jobs:
            if time.time() - T0 > MAXSEC:
                print("time guard: stop issuing shards", flush=True)
                return
            yield j

    with mp.get_context("spawn").Pool(NP) as p:
        for si, n in p.imap_unordered(build_shard, gen(), chunksize=1):
            print("SHARD DONE", si, n, "entries %.0fs" % (time.time() - T0), flush=True)
