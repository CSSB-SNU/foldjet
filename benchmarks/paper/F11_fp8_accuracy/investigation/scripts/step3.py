"""F11 investigation step 3: where the FP8 error of a trunk route grows. Eager AF3 trunk (seed 42, 11 recycles = 12 passes), forward hooks on every trunk
Pairformer block; at the LAST pass, the block outputs (pair, single) are saved (mode ref, bf16 process) or compared with the saved bf16 ones (mode cmp,
FP8-route process: the FP8 environment is set by the caller). Per block: relative error of pair / single, split by region (antibody x antigen interface
rows, antibody intra, antigen intra) and the per-channel dynamic range of the bf16 pair at the interface vs elsewhere.

  python step3.py ref --target T --out DIR     (no MINIWORLD_FP8 in the env)
  python step3.py cmp --target T --out DIR     (MINIWORLD_FP8=... set; writes DIR/T/cmp_<tag>.json)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import foldjet as ff

SPECS = Path.home() / "foldforge-scratch/foldbench/specs_all"
FB = Path.home() / "data/foldbench"
EVERY = 4  # blocks kept: 0, 4, 8, ..., 44 and the last one


def regions(t: str, n_tok: int) -> dict[str, np.ndarray]:
    """Token masks from the spec's chain order and FASTA lengths (protein chains: one token per residue)."""
    import yaml  # noqa: PLC0415

    y = yaml.safe_load((SPECS / f"{t}.yaml").read_text())
    names = json.loads((SPECS / f"{t}.chains.json").read_text())
    letters = y["chain_letters"]
    lens = {}
    for i, let in letters.items():
        seq = "".join(ln.strip() for ln in Path(y["fasta"][let]).read_text().splitlines() if not ln.startswith(">"))
        lens[names[int(i)]] = len(seq)
    tg = pd.read_csv(FB / "targets/interface_antibody_antigen.csv")
    g = tg[tg.pdb_id == t]
    if len(g):
        ab, ag = set(g.antibody_chain), set(g.antigen_chain)
    else:  # protein-protein control: first two chains
        pp = pd.read_csv(FB / "targets/interface_protein_protein.csv")
        r = pp[pp.pdb_id == t].iloc[0]
        ab, ag = {r.interface_chain_id_1}, {r.interface_chain_id_2}
    m_ab, m_ag = np.zeros(n_tok, bool), np.zeros(n_tok, bool)
    pos = 0
    for i in sorted(letters, key=int):
        c = names[int(i)]
        L = lens[c]
        if c in ab:
            m_ab[pos:pos + L] = True
        if c in ag:
            m_ag[pos:pos + L] = True
        pos += L
    return {"ab": m_ab, "ag": m_ag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "cmp"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default=os.environ.get("CFG", "fp8"))
    a = ap.parse_args()
    t = a.target
    d = a.out / t
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"precision": "bf16", "trunk": {"recycles": 11}, "execution": {"compile": False, "cuda_graph": False}}
    inputs = ff.load_inputs(SPECS / f"{t}.yaml")
    with ff.load_model("af3", config=cfg, device="cuda:0") as model:
        net = model._network
        blocks = [(n, m) for n, m in net.named_modules() if type(m).__name__ == "PairformerBlock" and "trunk" in n and n.split(".")[-1].isdigit()]
        blocks.sort(key=lambda x: int(x[0].split(".")[-1]))
        nb = len(blocks)
        keep = sorted(set(range(0, nb, EVERY)) | {nb - 1})
        calls = {i: 0 for i in range(nb)}
        res: dict = {}
        state = {"masks": None}

        def make_hook(i):
            def hook(_m, _inp, out):
                calls[i] += 1
                if i not in keep:
                    return
                pair, single = (out if isinstance(out, tuple) else (out, None))
                res[i] = (pair.detach().float().cpu().squeeze(0), None if single is None else single.detach().float().cpu().squeeze(0))
            return hook

        hs = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(blocks)]
        with torch.no_grad(), model.run_trunk(inputs, seed=42):
            pass
        for h in hs:
            h.remove()
    passes = calls[0]
    n_tok = res[keep[0]][0].shape[0]
    masks = regions(t, n_tok)
    ab, ag = torch.tensor(masks["ab"]), torch.tensor(masks["ag"])
    if a.mode == "ref":
        torch.save({i: (p.half(), None if s is None else s.half()) for i, (p, s) in res.items()}, d / "ref.pt")
        json.dump({"target": t, "blocks": nb, "passes": passes, "kept": keep, "n_tok": n_tok, "n_ab": int(ab.sum()), "n_ag": int(ag.sum())},
                  open(d / "ref.json", "w"))
        print("REF", t, nb, passes, n_tok)
        return
    ref = torch.load(d / "ref.pt")
    rows = []
    for i in keep:
        pr, sr = (x.float() if x is not None else None for x in ref[i])
        pc, sc = res[i]
        e = pc - pr

        def rel(mask_r, mask_c):
            sub = (slice(None),)
            er = e[mask_r][:, mask_c]
            rr = pr[mask_r][:, mask_c]
            return float(er.norm() / rr.norm().clamp_min(1e-12))

        n = pr.shape[0]
        allm = torch.ones(n, dtype=torch.bool)
        rr_if = pr[ab][:, ag]
        row = {"block": i, "pair_rel_all": float(e.norm() / pr.norm()), "pair_rel_ab_ag": rel(ab, ag), "pair_rel_ab_ab": rel(ab, ab),
               "pair_rel_ag_ag": rel(ag, ag), "pair_absmax_if": float(rr_if.abs().max()), "pair_absmax_all": float(pr.abs().max()),
               "pair_rms_if": float(rr_if.pow(2).mean().sqrt()), "pair_rms_all": float(pr.pow(2).mean().sqrt())}
        if sr is not None and sc is not None:
            es = sc - sr
            row["single_rel_all"] = float(es.norm() / sr.norm())
            row["single_rel_ab"] = float(es[ab].norm() / sr[ab].norm())
            row["single_rel_ag"] = float(es[ag].norm() / sr[ag].norm())
        rows.append(row)
    json.dump({"target": t, "tag": a.tag, "rows": rows, "fp8_status": _fp8_status()}, open(d / f"cmp_{a.tag}.json", "w"), indent=1)
    for r in rows:
        print("CMP", t, a.tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))


def _fp8_status():
    try:
        import miniworld_engine.fp8 as f  # noqa: PLC0415

        return {"status": f.status(), "routes": f.route_stats(), "saturated": f.check()} if f.preset() else None
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


if __name__ == "__main__":
    main()
