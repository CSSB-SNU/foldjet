"""F11 trace: where the Transition-FP8 error of the AF3 trunk starts to grow (antibody-antigen vs protein-protein).

One process per target, eager AF3 trunk, seed 42, 11 recycles. The FP8 environment is Transition-only (set by the caller, as step3 cmp);
the bf16 reference is the same process with ``fp8_sm90a.POLICY = None`` (the 2.5.12 bf16 route, bit-identical to no FP8).

Runs:
  ref    POLICY None: subsampled region slices of the pair after every pair module of every block and pass (+ block-0 input, + the MSA-module
         Transitions), the full single after single_attention / single_transition. Kept in host memory.
  fp8    POLICY calib (calibrated on its first call, as a fresh FP8 process): the same points -> error per region vs ref; at every FP8
         Transition: local injection (kernel vs bf16 route on the SAME input slice), propagated part, h statistics against the e4m3 scale.
  sched  further trunk runs with the calibrated scales kept: FP8 only in some passes / blocks, region protection, torch emulations
         (per-tensor = the kernel's scheme, per-row scales), TriMul-only FP8 (a route known to be harmless): error of every pass's trunk output.

Regions (token subsample, <= 64 tokens each): inter = ab x ag + ag x ab, ab = ab x ab, ag = ag x ag, all = 64 tokens evenly over all real tokens.
Protein-protein controls: ab := interface chain 1, ag := interface chain 2.

  python trace.py --target T --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import foldforge as ff

sys.path.insert(0, str(Path.home() / "foldforge-scratch/paper/F11_investigation/scripts"))
from step3 import SPECS, _fp8_status, regions  # noqa: E402

from miniworld_engine.kernels.transition.cuda import fp8_sm90a as tr  # noqa: E402

NSUB = 64
E4M3 = 448.0
SUB = 2.0 ** -6  # smallest normal e4m3
PAIR_MODS = ("tri_multi_outgoing", "tri_multi_incoming", "tri_atten_starting", "tri_atten_ending", "transition_pair")
SHORT = {"tri_multi_outgoing": "tmo", "tri_multi_incoming": "tmi", "tri_atten_starting": "tas", "tri_atten_ending": "tae", "transition_pair": "tr",
         "single_attention": "sa", "single_transition": "st"}


def even(idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k:
        return idx
    return idx[np.round(np.linspace(0, len(idx) - 1, k)).astype(int)]


def find_pair_single(args, kwargs):
    vals = [v for v in list(args) + list(kwargs.values()) if torch.is_tensor(v) and v.is_floating_point()]
    pair = next(v for v in vals if v.dim() == 4 and v.shape[1] == v.shape[2])
    single = next((v for v in vals if v.dim() == 3 and v.shape[1] == pair.shape[1]), None)
    return pair, single


def q8(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-E4M3, E4M3).to(torch.float8_e4m3fn).float()


def pow2floor(v):
    return torch.exp2(torch.floor(torch.log2(v)))


def emulate(x: torch.Tensor, m, mode: str) -> torch.Tensor:
    """y = x + transition(x) with FP8 emulated in torch. mode tensor: the kernel's scheme (xn: LayerNorm-bound per-tensor scale, Wa|Wb and Ws
    per-tensor, h: the calibrated per-tensor sh); row: xn and h with a power-of-two scale per row (per pair entry), weights per output channel."""
    wa, wb, ws = m.expand_a.weight.float(), m.expand_b.weight.float(), m.squeeze.weight.float()
    g, b = m.ln_in.weight.float(), m.ln_in.bias.float()
    cache = tr._CACHES[m.expand_a.weight.data_ptr()]
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    out = torch.empty_like(x2)
    if mode == "tensor":
        w1 = torch.cat([wa, wb])
        s1 = pow2floor(E4M3 / w1.abs().amax())
        waq, wbq = q8(wa * s1) / s1, q8(wb * s1) / s1
        s2 = pow2floor(E4M3 / ws.abs().amax())
        wsq = q8(ws * s2) / s2
        sx = cache["sx"].float()
        sh = cache["sh"].float()
    else:
        s1a = pow2floor(E4M3 / wa.abs().amax(1, keepdim=True))
        s1b = pow2floor(E4M3 / wb.abs().amax(1, keepdim=True))
        waq, wbq = q8(wa * s1a) / s1a, q8(wb * s1b) / s1b
        s2 = pow2floor(E4M3 / ws.abs().amax(1, keepdim=True))
        wsq = q8(ws * s2) / s2
    for i in range(0, x2.shape[0], 65536):
        xr = x2[i:i + 65536].float()
        xn = F.layer_norm(xr, (d,), g, b, m.ln_in.eps)
        if mode == "tensor":
            xq = q8(xn * sx) / sx
        else:
            sxr = pow2floor(E4M3 / xn.abs().amax(1, keepdim=True).clamp_min(1e-30))
            xq = q8(xn * sxr) / sxr
        h = F.silu(xq @ waq.t()) * (xq @ wbq.t())
        if mode == "tensor":
            hq = q8(h * sh) / sh
        else:
            shr = pow2floor(E4M3 / (h.abs().amax(1, keepdim=True) * 2).clamp_min(1e-30))
            hq = q8(h * shr) / shr
        out[i:i + 65536] = (xr + hq @ wsq.t()).to(x.dtype)
    return out.reshape(x.shape)


def _rec32_add(ln, lin, base: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
    """base + lin(ln(rec)) with the LayerNorm, the projection and the add in fp32; rounded once to base's dtype."""
    with torch.autocast("cuda", enabled=False):
        x = rec.float()
        w = ln.weight.float() if getattr(ln, "weight", None) is not None else None
        b = ln.bias.float() if getattr(ln, "bias", None) is not None else None
        y = F.layer_norm(x, tuple(ln.normalized_shape), w, b, ln.eps)
        term = F.linear(y, lin.weight.float(), lin.bias.float() if lin.bias is not None else None)
        return (base.float() + term).to(base.dtype)


def install_rec32() -> None:
    """--rec32: the two recycle adds of Evoformer._single_pair_trunk (pair and single) in fp32; everything else unchanged."""
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    from foldforge.models.architectures import af3  # noqa: PLC0415

    src = textwrap.dedent(inspect.getsource(af3.Evoformer._single_pair_trunk))
    old_pair = """pair_activations = pair_activations + self._recycle_pair(
        recycled.to(pair_activations.dtype), recycle_dropout
    )"""
    old_single = """single_activations = single_activations + self.prev_single_embedding(
        self.prev_single_embedding_layer_norm(
            recycled_single.to(single_activations.dtype)
        )
    )"""
    assert src.count(old_pair) == 1 and src.count(old_single) == 1, "af3._single_pair_trunk changed: rec32 patch does not apply"
    src = src.replace(old_pair, "assert not recycle_dropout\n    pair_activations = _rec32_add(self.prev_embedding_layer_norm, self.prev_embedding, pair_activations, recycled)")
    src = src.replace(old_single, "single_activations = _rec32_add(self.prev_single_embedding_layer_norm, self.prev_single_embedding, single_activations, recycled_single)")
    g = dict(vars(af3))
    g["_rec32_add"] = _rec32_add
    exec(compile(src, "<rec32 _single_pair_trunk>", "exec"), g)  # noqa: S102
    af3.Evoformer._single_pair_trunk = g["_single_pair_trunk"]
    print("REC32 installed: pair + single recycle adds in fp32", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sched", default="all,pf_only,msa_only,pass0,lastpass,notlast,blk0_23,blk24_47,protect_inter,emu_tensor,emu_row,trimul_only")
    ap.add_argument("--rec32", action="store_true", help="recycle adds in fp32 for EVERY run of the process (the bf16 reference too)")
    a = ap.parse_args()
    if a.rec32:
        install_rec32()
    t = a.target
    d = a.out / t
    d.mkdir(parents=True, exist_ok=True)
    CALIB = tr.POLICY
    assert CALIB == "calib", f"Transition FP8 not on in the env: {CALIB!r}"
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as tmf  # noqa: PLC0415

    assert tmf.POLICY is None, "TriMul FP8 must be off (Transition-only env)"
    cfg = {"precision": "bf16", "trunk": {"recycles": 11}, "execution": {"compile": False, "cuda_graph": False}}
    inputs = ff.load_inputs(SPECS / f"{t}.yaml")
    T0 = time.time()
    with ff.load_model("af3", config=cfg, device="cuda:0") as model:
        net = model._network
        blocks = [(n, m) for n, m in net.named_modules() if type(m).__name__ == "PairformerBlock" and "trunk" in n and n.split(".")[-1].isdigit()]
        blocks.sort(key=lambda x: int(x[0].split(".")[-1]))
        nb = len(blocks)
        bnames = {n for n, _ in blocks}
        extra = [(n, m) for n, m in net.named_modules() if type(m).__name__ == "Transition" and "msa_stack" in n
                 and not any(n.startswith(b + ".") for b in bnames)
                 and getattr(m, "expand_a", None) is not None and m.expand_a.weight.shape[1] == 128]
        print("EXTRA_TRANSITIONS", [n for n, _ in extra], flush=True)
        S = {"run": "ref", "pass": -1, "n": None, "idx": None, "store": {}, "rows": [], "prev_e": None, "prev_es": None, "sched": None,
             "keep_points": None, "prot": None}

        def setup(n_tok):
            masks = regions(t, n_tok)
            ab, ag = np.where(masks["ab"])[0], np.where(masks["ag"])[0]
            n_real = int(max(ab.max(), ag.max()) + 1)
            sa, sg, sr = even(ab, NSUB), even(ag, NSUB), even(np.arange(n_real), NSUB)
            idx = np.unique(np.concatenate([sa, sg, sr]))
            pos = {v: i for i, v in enumerate(idx)}
            A = torch.zeros(len(idx), dtype=torch.bool)
            G = torch.zeros(len(idx), dtype=torch.bool)
            R = torch.zeros(len(idx), dtype=torch.bool)
            A[[pos[v] for v in sa]] = True
            G[[pos[v] for v in sg]] = True
            R[[pos[v] for v in sr]] = True
            dev = "cuda:0"
            reg = {"inter": (A[:, None] & G[None]) | (G[:, None] & A[None]), "ab": A[:, None] & A[None], "ag": G[:, None] & G[None],
                   "all": R[:, None] & R[None]}
            S.update(n=n_tok, idx=torch.tensor(idx, device=dev), reg={k: v.to(dev) for k, v in reg.items()},
                     sreg={"ab": torch.tensor(masks["ab"], device=dev), "ag": torch.tensor(masks["ag"], device=dev),
                           "all": torch.arange(n_tok, device=dev) < n_real},
                     full_ab=torch.tensor(masks["ab"], device=dev), full_ag=torch.tensor(masks["ag"], device=dev))
            S["meta"] = {"target": t, "n_tok": n_tok, "n_real": n_real, "n_ab": int(masks["ab"].sum()), "n_ag": int(masks["ag"].sum()),
                         "S": len(idx), "n_sub": {"ab": len(sa), "ag": len(sg), "all": len(sr)}}

        def sl(p):
            i = S["idx"]
            return p[0].index_select(0, i).index_select(1, i)

        def rstats(v, reg):  # sum of squares and count per region (fp32 [S,S,C] or [n,C])
            out = {}
            for k, mk in reg.items():
                sel = v[mk]
                out[k] = (float(sel.float().pow(2).sum()), int(sel.numel()))
            return out

        def rec_pair(key, out_pair):
            if S["keep_points"] is not None and key[1] not in S["keep_points"]:
                return
            s = sl(out_pair)
            if S["run"] == "ref":
                S["store"][key] = s.cpu()
                return
            ref = S["store"][key].to(s.device).float()
            e = s.float() - ref
            row = {"run": S["run"], "pass": key[0], "pt": key[1], "kind": "pair"}
            sr, se = rstats(ref, S["reg"]), rstats(e, S["reg"])
            if S["prev_e"] is not None and S["keep_points"] is None:
                sd = rstats(e - S["prev_e"], S["reg"])
                row.update({f"dE_{k}": math.sqrt(sd[k][0] / sd[k][1]) for k in sd})
            for k in sr:
                row[f"ref_{k}"] = math.sqrt(sr[k][0] / sr[k][1])
                row[f"err_{k}"] = math.sqrt(se[k][0] / se[k][1])
            S["prev_e"] = e
            S["rows"].append(row)

        def rec_single(key, single):
            if S["keep_points"] is not None and key[1] not in S["keep_points"]:
                return
            s = single[0]
            skey = (key[0], "single:" + key[1])
            if S["run"] == "ref":
                S["store"][skey] = s.cpu()
                return
            ref = S["store"][skey].to(s.device).float()
            e = s.float() - ref
            row = {"run": S["run"], "pass": key[0], "pt": key[1], "kind": "single"}
            sr, se = rstats(ref, S["sreg"]), rstats(e, S["sreg"])
            if S["prev_es"] is not None and S["keep_points"] is None:
                sd = rstats(e - S["prev_es"], S["sreg"])
                row.update({f"dE_{k}": math.sqrt(sd[k][0] / sd[k][1]) for k in sd})
            for k in sr:
                row[f"ref_{k}"] = math.sqrt(sr[k][0] / sr[k][1])
                row[f"err_{k}"] = math.sqrt(se[k][0] / se[k][1])
            S["prev_es"] = e
            S["rows"].append(row)

        def policy_for(bi: int | None) -> str | None:  # bi None: an MSA-module Transition
            sc, p, last = S["sched"], S["pass"], S["npass"] - 1 if "npass" in S else None
            if S["run"] == "ref":
                return None
            if sc in (None, "all", "protect_inter"):
                return CALIB
            if sc in ("emu_tensor", "emu_row", "trimul_only"):
                return None
            if sc == "pf_only":
                return CALIB if bi is not None else None
            if sc == "msa_only":
                return CALIB if bi is None else None
            if sc == "pass0":
                return CALIB if p == 0 else None
            if sc == "lastpass":
                return CALIB if p == last else None
            if sc == "notlast":
                return CALIB if p != last else None
            if sc == "blk0_23":
                return CALIB if (bi is not None and bi < 24) else None
            if sc == "blk24_47":
                return CALIB if (bi is not None and bi >= 24) else None
            raise ValueError(sc)

        def injection(m, x, out, key):
            """Kernel vs bf16 route on the same input slice (FP8 run): local injection; h statistics vs the calibrated scale."""
            xs = sl(x).contiguous()
            Sx = xs.shape[0]
            rows = Sx * Sx
            need = max(16384, -(-rows // 128) * 128)
            x2 = xs.reshape(rows, -1)
            if need > rows:
                x2 = x2.repeat(-(-need // rows), 1)[:need]
            x4 = x2.reshape(1, 1, need, -1).contiguous()
            args = (m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight, m.ln_in.eps)
            assert tr.serves(x4, m.expand_a.weight, m.squeeze.weight)
            y8 = tr.inference(x4, *args).reshape(need, -1)[:rows].reshape(xs.shape).float()
            yb = tr._default(x4, *args).reshape(need, -1)[:rows].reshape(xs.shape).float()
            outs = sl(out).float()
            xf = xs.float()
            inj, u = y8 - yb, yb - xf
            row = {"run": S["run"], "pass": key[0], "pt": key[1], "kind": "inj", "check_maxdiff": float((outs - y8).abs().max())}
            ref_in = S["store"].get(S["in_key"])
            ref_out = S["store"].get(key)
            for k, mk in S["reg"].items():
                nx = float(xf[mk].pow(2).mean().sqrt())
                row[f"x_{k}"] = nx
                row[f"u_{k}"] = float(u[mk].pow(2).mean().sqrt())
                row[f"inj_{k}"] = float(inj[mk].pow(2).mean().sqrt())
            if ref_in is not None and ref_out is not None:
                e_in = xf - ref_in.to(xf.device).float()
                e_out = outs - ref_out.to(xf.device).float()
                prop = e_out - e_in - inj
                for k, mk in S["reg"].items():
                    row[f"ein_{k}"] = float(e_in[mk].pow(2).mean().sqrt())
                    row[f"eout_{k}"] = float(e_out[mk].pow(2).mean().sqrt())
                    row[f"prop_{k}"] = float(prop[mk].pow(2).mean().sqrt())
            # h against the calibrated e4m3 scale
            cache = tr._CACHES[m.expand_a.weight.data_ptr()]
            sh, sx = float(cache["sh"]), float(cache["sx"])
            xn = F.layer_norm(xf, (xf.shape[-1],), m.ln_in.weight.float(), m.ln_in.bias.float(), m.ln_in.eps)
            h = F.silu(xn @ m.expand_a.weight.float().t()) * (xn @ m.expand_b.weight.float().t())
            hs = h * sh
            hq = q8(hs) / sh
            xq = q8(xn * sx) / sx
            row.update({"sh": sh, "sx": sx, "h_amax_scaled_all": float(hs.abs().max())})
            for k, mk in S["reg"].items():
                hk, hsk = h[mk], hs[mk].abs()
                row[f"h_rms_{k}"] = float(hk.pow(2).mean().sqrt())
                row[f"h_sub_{k}"] = float((hsk < SUB).float().mean())  # fraction of h elements in the e4m3 subnormal range
                row[f"h_zero_{k}"] = float((hsk < 2.0 ** -10).float().mean())  # rounds to 0
                row[f"h_qrel_{k}"] = float((hq[mk] - hk).norm() / hk.norm().clamp_min(1e-30))  # relative h quantization error
                # the same error with a per-row (per entry) scale: what per-row scaling would buy
                shr = pow2floor(E4M3 / (hk.abs().amax(-1, keepdim=True) * 2).clamp_min(1e-30))
                row[f"h_qrel_row_{k}"] = float((q8(hk * shr) / shr - hk).norm() / hk.norm().clamp_min(1e-30))
                row[f"xn_qrel_{k}"] = float((xq[mk] - xn[mk]).norm() / xn[mk].norm())
                row[f"h_amax_scaled_{k}"] = float(hsk.max())
            S["rows"].append(row)

        # ---- hooks
        def blk_pre(bi):
            def hook(_m, args, kwargs):
                if bi == 0:
                    S["pass"] += 1
                    pair, single = find_pair_single(args, kwargs)
                    if S["n"] is None:
                        setup(pair.shape[1])
                    rec_pair((S["pass"], "in"), pair)
                    rec_single((S["pass"], "in"), single)
                S["bi"] = bi
            return hook

        def sub_hook(bi, name):
            short = SHORT[name]

            def pre(m, args):
                if name == "transition_pair":
                    tr.POLICY = policy_for(bi)
                    S["in_key"] = (S["pass"], f"b{bi}.tae")

            def post(m, args, out):
                key = (S["pass"], f"b{bi}.{short}")
                if name in PAIR_MODS:
                    if name == "transition_pair":
                        if S["sched"] == "protect_inter":
                            out = protect(m, args[0], out)
                        elif S["sched"] in ("emu_tensor", "emu_row"):
                            out = emulate(args[0], m, S["sched"][4:])
                    if S["run"] == "fp8" and name == "transition_pair":
                        injection(m, args[0], out, key)
                    rec_pair(key, out)
                    return out
                rec_single(key, out)
                return None
            return pre, post

        def protect(m, x, out):
            yb = tr._default(x, m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight, m.ln_in.eps)
            A, G = S["full_ab"], S["full_ag"]
            inter = ((A[:, None] & G[None]) | (G[:, None] & A[None]))[None, :, :, None]
            return torch.where(inter, yb, out)

        def extra_hook(name):
            def pre(m, args):
                tr.POLICY = policy_for(None)
                if S["n"] is None and args[0].dim() == 4 and args[0].shape[1] == args[0].shape[2]:
                    setup(args[0].shape[1])
                if S["n"] is not None and args[0].dim() == 4 and args[0].shape[1] == S["n"]:
                    S["in_key"] = (S["pass"] + 1, f"x.{name}.in")
                    rec_pair(S["in_key"], args[0])

            def post(m, args, out):
                if S["n"] is None or out.dim() != 4 or out.shape[1] != S["n"]:
                    return None
                key = (S["pass"] + 1, f"x.{name}")  # runs before block 0 of the next pass
                if S["sched"] == "protect_inter":
                    out = protect(m, args[0], out)
                elif S["sched"] in ("emu_tensor", "emu_row"):
                    out = emulate(args[0], m, S["sched"][4:])
                if S["run"] == "fp8":
                    injection(m, args[0], out, key)
                rec_pair(key, out)
                return out
            return pre, post

        hs = []
        for bi, (_, b) in enumerate(blocks):
            hs.append(b.register_forward_pre_hook(blk_pre(bi), with_kwargs=True))
            for name in SHORT:
                sm = getattr(b, name, None)
                if sm is None:
                    continue
                pre, post = sub_hook(bi, name)
                hs.append(sm.register_forward_pre_hook(pre))
                hs.append(sm.register_forward_hook(post))
        for name, m in extra:
            pre, post = extra_hook(name)
            hs.append(m.register_forward_pre_hook(pre))
            hs.append(m.register_forward_hook(post))

        def trunk(run, sched=None, keep=None):
            S.update(run=run, sched=sched, keep_points=keep, prev_e=None, prev_es=None, rows=[])
            S["pass"] = -1
            t0 = time.time()
            if sched == "trimul_only":
                tmf.POLICY = "calib"
            with torch.no_grad(), model.run_trunk(inputs, seed=42):
                pass
            if sched == "trimul_only":
                print("TRIMUL_GUARD", dict(tmf.GUARD_STATS) if hasattr(tmf, "GUARD_STATS") else None, flush=True)
            tmf.POLICY = None
            tr.POLICY = CALIB
            torch.cuda.synchronize()
            print(f"RUN {t} {run} {sched} passes={S['pass'] + 1} {time.time() - t0:.1f}s", flush=True)
            return S["rows"]

        # the MSA-module Transitions run before block 0 of a pass: the first pass's ones are keyed pass 0 because S["pass"] is -1 then; the
        # pre-hook of block 0 cannot set up the subsample before them in the first pass, so in pass 0 they are skipped (S["n"] None).
        trunk("ref")
        S["npass"] = S["pass"] + 1
        rows = trunk("fp8")
        with open(d / "trace.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        status = _fp8_status()
        last = S["npass"] - 1
        keep = {"in"} | {f"b{nb - 1}.tr", f"b{nb - 1}.st"} | {f"b{i}.tr" for i in range(0, nb, 8)}
        with open(d / "sched.jsonl", "w") as fh:
            for sc in a.sched.split(","):
                for r in trunk("sched", sc, keep):
                    r["sched"] = sc
                    fh.write(json.dumps(r) + "\n")
        S["meta"].update(blocks=nb, passes=S["npass"], extra=[n for n, _ in extra], fp8_status=status, wall_s=time.time() - T0,
                         sh={n: float(tr._CACHES[m.transition_pair.expand_a.weight.data_ptr()]["sh"]) for n, m in blocks
                             if m.transition_pair.expand_a.weight.data_ptr() in tr._CACHES})
        json.dump(S["meta"], open(d / "meta.json", "w"), indent=1, default=str)
        for h in hs:
            h.remove()
    print("DONE", t, f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
