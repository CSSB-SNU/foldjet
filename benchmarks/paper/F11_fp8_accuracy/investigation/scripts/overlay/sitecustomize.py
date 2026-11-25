# F11 investigation: TriAttn FP8 variants for the route ablation (fb_multi workers start with this dir first on PYTHONPATH via ENGINE_OVERLAY).
# F11INV_TRIATTN = pvqk (as the af3 whitelist: fused + unfused PV, e4m3 q/k) | pv (PV on both kinds, q/k off) | none (TriAttn on the fp16 core)
import os

_v = os.environ.get("F11INV_TRIATTN")
if _v:
    import foldjet_engine.fp8 as _fp8

    _choice = {"pvqk": (frozenset({"fused", "unfused"}), True), "pv": (frozenset({"fused", "unfused"}), False), "none": (frozenset(), False)}[_v]
    _fp8.triattn_for = lambda _name, _c=_choice: _c

# Transition FP8 variants (F11 step 3): only when one of these is set the kernel module is imported here and patched.
#   F11INV_TR_HIDDEN=512|256   route only Transitions with that hidden width through FP8 (trunk pair transition n=4: 512; diffusion pair conditioning n=2: 256)
#   F11INV_TR_DYNAMIC=1        recalibrate the h scale on every eager call (dynamic scale; needs eager, no CUDA graphs)
#   F11INV_TR_LOG=FILE         append one line per FP8 Transition call: x shape, hidden width, calibrated h scale
if any(os.environ.get(k) for k in ("F11INV_TR_HIDDEN", "F11INV_TR_DYNAMIC", "F11INV_TR_LOG")):
    import torch as _torch
    from foldjet_engine.kernels.transition.cuda import fp8_sm90a as _tr

    _hid = os.environ.get("F11INV_TR_HIDDEN")
    if _hid:
        _serves0 = _tr.serves
        _tr.serves = lambda x, wa, ws, _s=_serves0, _h=int(_hid): wa.shape[0] == _h and _s(x, wa, ws)
    _inf0 = _tr.inference
    _dyn = os.environ.get("F11INV_TR_DYNAMIC") == "1"
    _log = os.environ.get("F11INV_TR_LOG")

    def _inference(x, gamma, beta, wa, wb, ws, eps, _f=_inf0):
        if _dyn and not _torch.cuda.is_current_stream_capturing():
            c = _tr._CACHES.get(wa.data_ptr())
            if c is not None:
                c["calibrated"] = False
        y = _f(x, gamma, beta, wa, wb, ws, eps)
        if _log and not _torch.cuda.is_current_stream_capturing():
            c = _tr._CACHES.get(wa.data_ptr())
            with open(_log, "a") as fh:
                fh.write(f"{tuple(x.shape)} hidden={wa.shape[0]} d={wa.shape[1]} sh={float(c['sh']) if c else 'na'}\n")
        return y

    _tr.inference = _inference
