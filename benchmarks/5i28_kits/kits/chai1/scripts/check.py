"""CPU check (no GPU, no ESM): chai's own feature-context build for 5I28 x<N> with the bench inputs -- the MSA directory and the m8 hits
with the offline mmCIF shim -- and what it loaded: chai_lab's 'Loaded k templates: [...]' line per chain (INFO, via the shim), the merged
TemplateContext and the MSA depth. usage (inside the container, chai venv130, PYTHONPATH=shim, REAL_CIF_DIR set): check.py N [N ...]
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
from chai_lab.chai1 import make_all_atom_feature_context

IN = Path(os.environ["BENCH_DIR"] + "/chai1/inputs")
TMP = Path(os.environ["BENCH_DIR"] + "/chai1/tmp")
for n in map(int, sys.argv[1:]):
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        ctx = make_all_atom_feature_context(
            fasta_file=IN / f"5i28_x{n}.fasta",
            output_dir=Path(td) / "out",
            use_esm_embeddings=False,
            use_msa_server=False,
            msa_directory=IN / "msa",
            constraint_path=None,
            use_templates_server=False,
            templates_path=IN / f"templates_n{n}.m8",
            esm_device=torch.device("cpu"),
        )
        tc = ctx.template_context
        print(
            f"[check] n={n} tokens={ctx.structure_context.num_tokens} {tc}", flush=True
        )
        print(
            f"[check] n={n} num_templates={tc.num_templates} nonnull_templates={tc.num_nonnull_templates} msa_depth={ctx.msa_context.depth} "
            f"restype_tokens_per_slot={tc.template_mask.sum(-1).tolist()} cb_tokens_per_slot={tc.template_pseudo_beta_mask.sum(-1).tolist()}",
            flush=True,
        )
