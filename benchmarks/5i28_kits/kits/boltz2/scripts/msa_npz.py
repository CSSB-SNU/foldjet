"""Rows of the MSA Boltz-2 itself processed for the run (<out>/boltz_results_*/processed/msa/*.npz, written by boltz's preprocessing from the
yaml's msa: path with max_msa_seqs 8192; no --subsample_msa passed, upstream default off). usage: msa_npz.py <out_dir>
"""

import glob
import sys

import numpy as np

fs = sorted(glob.glob(sys.argv[1] + "/**/processed/msa/*.npz", recursive=True))
for f in fs:
    z = np.load(f, allow_pickle=True)
    print(
        f"MSA_ROWS file={f.split('/out/')[-1]} sequences={z['sequences'].shape[0]}",
        flush=True,
    )
if not fs:
    print("MSA_ROWS none (no processed/msa/*.npz)", flush=True)
