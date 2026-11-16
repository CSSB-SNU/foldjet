"""Forward timer for the RF3 kit, plus input evidence printed OUTSIDE the timed region.

Same timer as ~/foldforge-scratch/anthropic-kits/fwdtimer_real.py:
torch.cuda.synchronize() before and after RF3WithConfidence.forward, one
`FWDTIME <i> <s>` line per call. Input evidence:
- `REALINPUT call=<i> ...`: the model's input at that forward -- per-recycle MSA stack
  shape, templated tokens (diagonal of has_distogram_condition) and template pairs;
- `MSAFULL rows=<r> tokens=<t>`: the full MSA depth handed to atomworks'
  FeaturizeMSALikeAF3 (before RF3 subsamples n_msa rows per recycle). The kit's prefetch
  helper is forked from this process, so it prints this line too (item 0 is featurised
  twice by design).
usage: fwdtimer.py <model class path> <entry module:function | script.py> [args...]
"""

import importlib
import runpy
import sys
import time

import torch


def _patch_msa_probe() -> None:
    try:
        from atomworks.ml.transforms.msa import msa as _m

        orig = _m.FeaturizeMSALikeAF3.forward

        def forward(self: object, data: dict) -> object:
            try:
                r, t = data["encoded"]["msa"].shape[:2]
                print(
                    f"MSAFULL rows={int(r)} tokens={int(t)} n_msa={self.n_msa} "
                    f"n_recycles={self.n_recycles}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print("MSAFULL probe failed", repr(e), flush=True)
            return orig(self, data)

        _m.FeaturizeMSALikeAF3.forward = forward
    except Exception as e:  # noqa: BLE001
        print("MSAFULL probe not installed", repr(e), flush=True)


def main():
    cls_path, entry, *args = sys.argv[1:]
    mod, name = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod), name)
    _patch_msa_probe()
    orig = cls.forward
    count = [0]

    def forward(self: object, *a: object, **k: object) -> object:
        try:
            f = (a[0] if a else k["input"])["f"]
            h = f["has_distogram_condition"]
            print(
                f"REALINPUT call={count[0]} msa_stack={tuple(f['msa_stack'].shape)} "
                f"n_token={h.shape[0]} "
                f"templated_tokens={int(h.diagonal().sum())} "
                f"template_pairs={int(h.sum())} "
                "distogram_condition_onehot_nonmask_pairs="
                f"{int((f['distogram_condition'][..., :-1].sum(-1) > 0).sum())}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print("REALINPUT probe failed", repr(e), flush=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig(self, *a, **k)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"FWDTIME {count[0]} {dt:.3f}", flush=True)
        count[0] += 1
        return out

    cls.forward = forward
    if entry.endswith(".py"):
        sys.argv = [entry, *args]
        runpy.run_path(entry, run_name="__main__")
    else:
        m, f = entry.split(":")
        sys.argv = [f, *args]
        sys.exit(getattr(importlib.import_module(m), f)())


if __name__ == "__main__":
    main()
