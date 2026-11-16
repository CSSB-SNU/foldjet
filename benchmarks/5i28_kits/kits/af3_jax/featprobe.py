# CPU featurisation probe with the kit's OWN alphafold3 package (the interpreter the kit
# featurises with) on the exact input JSON the kit runs. Gives the MSA and template
# evidence the kit's run log does not print.
# usage: <kit alphafold3 python> -I featprobe.py <input.json> <resolve_msa_overlaps 0|1>
#        <kit tag>
#   af3_jax  : af3_jax/venv/bin/python, resolve_msa_overlaps=1
#              (run_alphafold_fast.py flag default, used by the kit run)
#   af3_torch: af3_torch/alphafold3_venv/bin/python, resolve_msa_overlaps=0
#              (af3_torch_opt/featurise.py featurise_one)
# Prints one line: FEATPROBE kit=... msa_rows=<rows of the featurised msa with any
# unmasked position> tmpl_real_min=<real slots, min over chains>
# tmpl_distinct_min=<distinct real slots by coordinates, min over chains> ...
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation

src, resolve, kit = sys.argv[1], bool(int(sys.argv[2])), sys.argv[3]
j = json.loads(pathlib.Path(src).read_text())
# rows/templates do not depend on the seed; one example is enough
j["modelSeeds"] = j["modelSeeds"][:1]
tmp = (
    pathlib.Path(os.environ.get("TMPDIR", "/tmp"))  # noqa: S108 - per-pid scratch file
    / f"featprobe_{kit}_{os.getpid()}.json"
)
tmp.write_text(json.dumps(j))
fis = list(folding_input.load_fold_inputs_from_path(tmp))
assert len(fis) == 1
fi = fis[0]
t = time.time()
ex = featurisation.featurise_input(
    fold_input=fi,
    buckets=None,
    ccd=chemical_components.Ccd(user_ccd=fi.user_ccd),
    verbose=False,
    resolve_msa_overlaps=resolve,
)[0]
feat_s = time.time() - t
tmp.unlink()
msa_mask = np.asarray(ex["msa_mask"])
rows = int((msa_mask.sum(1) > 0).sum())
first1024 = int((msa_mask[:1024].sum(1) > 0).sum())
pos = np.asarray(ex["template_atom_positions"])  # [slots, tokens, 24, 3]
am = np.asarray(ex["template_atom_mask"])  # [slots, tokens, 24]
asym = np.asarray(ex["asym_id"])
chains = sorted(set(asym.tolist()))
real, distinct = [], []
for c in chains:
    sel = asym == c
    r = [s for s in range(am.shape[0]) if am[s, sel].sum() > 0]
    h = {
        hashlib.sha256(
            np.round(pos[s, sel] * am[s, sel][..., None], 2).tobytes()
        ).hexdigest()
        for s in r
    }
    real.append(len(r))
    distinct.append(len(h))
print(
    f"FEATPROBE kit={kit} resolve_msa_overlaps={int(resolve)} chains={len(chains)} "
    f"tokens={int(np.asarray(ex['seq_length']))} "
    f"msa_shape={tuple(msa_mask.shape)} msa_rows={rows} "
    f"msa_rows_in_first1024={first1024} tmpl_slots={am.shape[0]} "
    f"tmpl_real_min={min(real)} tmpl_distinct_min={min(distinct)} "
    f"atom_mask_sum={[int(am[s].sum()) for s in range(am.shape[0])]} "
    f"feat_s={feat_s:.1f}",
    flush=True,
)
