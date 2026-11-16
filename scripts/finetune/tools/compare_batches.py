"""torch.equal of every tensor of every dumped example. usage: compare_batches.py DIR_A DIR_B"""

import os
import sys

import torch

A, B = sys.argv[1], sys.argv[2]
fa, fb = sorted(os.listdir(A)), sorted(os.listdir(B))
assert fa == fb, "file sets differ"
n_ex = n_t = bad = 0
for f in fa:
    a, b = torch.load(os.path.join(A, f)), torch.load(os.path.join(B, f))
    n_ex += 1
    if a["name"] != b["name"]:
        print("NAME DIFF", f, a["name"], b["name"])
        bad += 1
    if a["info"] != b["info"]:
        print("INFO DIFF", f, a["info"], b["info"])
        bad += 1
    ta = {
        **{"f:" + k: v for k, v in a["features"].items()},
        **{
            k: a[k]
            for k in (
                "atom_positions",
                "atom_mask",
                "bond_pairs",
                "present_index",
                "present_valid",
            )
        },
    }
    tb = {
        **{"f:" + k: v for k, v in b["features"].items()},
        **{
            k: b[k]
            for k in (
                "atom_positions",
                "atom_mask",
                "bond_pairs",
                "present_index",
                "present_valid",
            )
        },
    }
    if ta.keys() != tb.keys():
        print("KEY DIFF", f)
        bad += 1
        continue
    for k in ta:  # noqa: PLC0206 - indexes ta and tb in parallel
        n_t += 1
        if (
            ta[k].dtype != tb[k].dtype
            or ta[k].shape != tb[k].shape
            or not torch.equal(ta[k], tb[k])
        ):
            print("TENSOR DIFF", f, k, ta[k].dtype, tb[k].dtype)
            bad += 1
print(
    "COMPARE examples=%d tensors=%d mismatches=%d  tensors/example=%.1f"
    % (n_ex, n_t, bad, n_t / max(n_ex, 1))
)
