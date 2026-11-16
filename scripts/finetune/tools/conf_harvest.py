"""Count the CCD codes (residue names) of the cropped structures of N stream positions (MiniWorld load only). usage: conf_harvest.py ADAPTER_DIR RANK START N NPROC OUT.json"""
import sys, os, json, collections, multiprocessing as mp, time
AD, RANK, START, N, NP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ds = None
def work(chunk):
    global _ds
    sys.path.insert(0, AD)
    import numpy as np, torch
    torch.set_num_threads(1)
    import ft_adapter, mwconfig, mwsample
    if _ds is None:
        s = ft_adapter.Settings()
        _ds = mwsample.build_dataset(mwconfig.load_data_dict(s.data_config, s.overlay), light=True)
    cnt = collections.Counter(); ninst = []
    for pos, idx, spares in chunk:
        rng = np.random.default_rng(ft_adapter.sample_seed(0, RANK, pos))
        try:
            raw = mwsample.process_item(_ds, idx, rng, spares=list(spares))
        except Exception as e:
            continue
        names = np.char.strip(np.asarray(raw.cifmol.residues.chem_comp_id.value).astype(str))
        cnt.update(names.tolist()); ninst.append(len(names))
    return dict(cnt), ninst
if __name__ == "__main__":
    import positions
    keys = positions.get_keys(AD, RANK, 2, START, N)
    chunks = [keys[i:i + 4] for i in range(0, len(keys), 4)]
    t = time.time(); tot = collections.Counter(); ninst = []
    with mp.get_context("spawn").Pool(NP) as p:
        for i, (c, n) in enumerate(p.imap_unordered(work, chunks)):
            tot.update(c); ninst += n
            print("chunk", i, "done %.0fs" % (time.time() - t), "examples", len(ninst), flush=True)
            if i % 10 == 0: json.dump({"counts": dict(tot), "n_examples": len(ninst), "max_instances": max(ninst)}, open(OUT, "w"))
            if time.time() - t > 470: print("time guard", flush=True); p.terminate(); break
    json.dump({"counts": dict(tot), "n_examples": len(ninst), "max_instances": max(ninst)}, open(OUT, "w"))
    print("codes", len(tot), "examples", len(ninst), "max residue instances", max(ninst))
    for k, v in tot.most_common(60): print(k, v)
