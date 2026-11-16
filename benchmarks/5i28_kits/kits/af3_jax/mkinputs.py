# Build the 5I28-benchmark AF3-JSON inputs for BOTH AF3 kits (af3_jax, af3_torch),
# n = 1 2 4 8 12 16 identical chains.
# Source = FoldForge's own input resolution (foldforge.data.inputs.build.load on the
# 1-chain spec): the real MSA (a3m_16k.lmdb key P0113359, 6789 a3m rows incl. the query
# (AF3 featurises them to 6790 MSA rows), paired == unpaired exactly as FoldForge's
# adapter input) + the 4 distinct real templates that
# foldforge.data.inputs.lmdb.load_templates keeps (one per hit sequence): 5SYD_A,
# 6KOL_A, 5YT7_A, 6TFD_A, each as Template.af3() (inline mmCIF + queryIndices +
# templateIndices).
# Homomer = ONE protein entry with ids A,B,... (AF3 JSON form, as the earlier
# anthropic-kits/real/mkreal.py did).
# Run with FoldForge's interpreter on a compute node (never on the login node):
#   srun -p h100 -A cssb --qos=normal_h100 -n 1 -c 4 --mem=16G --time=00:30:00 \
#     ~/FoldForge/.pixi/envs/default/bin/python -I mkinputs.py
import copy
import hashlib
import json
import os
import string
from pathlib import Path

from foldforge.data.inputs.build import load, write_adapter_input

B = Path(os.environ["BENCH_DIR"] + "")
SPEC = Path(os.environ["INPUT_SCRATCH"] + "/ff_x1.yaml")
WANT = ["5SYD_A", "6KOL_A", "5YT7_A", "6TFD_A"]
NS = (1, 2, 4, 8, 12, 16)
MSA_ROWS = 6789  # a3m rows incl. the query
TOKENS_PER_CHAIN = 128
N_TEMPLATES = 4
# forwards per model process (= modelSeeds count): first is cold, the rest warm. Same
# counts as the earlier real-MSA kit rows, except af3_torch n>=8 gets 3 (was 2-3) so
# every n has >= 2 warm forwards.
SEEDS = {
    "af3_jax": {1: 6, 2: 6, 4: 6, 8: 5, 12: 4, 16: 4},
    "af3_torch": {1: 4, 2: 4, 4: 4, 8: 3, 12: 3, 16: 3},
}

tgt = load(SPEC)
assert len(tgt.chains) == 1, tgt.chains
c = tgt.chains[0]
ids = [t.id for t in c.templates]
assert sorted(ids) == sorted(WANT), f"FoldForge picked {ids}, want {WANT}"
assert c.alignment is not None
assert len(c.alignment.tokens) == MSA_ROWS, len(c.alignment.tokens)
B.joinpath("af3_jax/inputs").mkdir(parents=True, exist_ok=True)
# FoldForge's own adapter JSON, kept for reference
write_adapter_input(tgt, B / "af3_jax/inputs/ff_adapter_x1.json")
base = tgt.af3(seed=0)
p = base["sequences"][0]["protein"]
assert len(p["sequence"]) == TOKENS_PER_CHAIN
assert len(p["templates"]) == N_TEMPLATES
assert p["unpairedMsa"].count(">") == MSA_ROWS
assert p["pairedMsa"] == p["unpairedMsa"]
dig = [hashlib.sha256(t["mmcif"].encode()).hexdigest()[:12] for t in p["templates"]]
assert len(set(dig)) == N_TEMPLATES, dig
print(
    "TEMPLATES",
    ids,
    "mmcif_sha",
    dig,
    "atoms",
    [t["mmcif"].count("\nATOM") for t in p["templates"]],
)
print("MSA rows", p["unpairedMsa"].count(">"), "species aliases", len(tgt.msa_species))
for n in NS:
    prot = copy.deepcopy(p)
    prot["id"] = list(string.ascii_uppercase[:n]) if n > 1 else "A"
    for kit, seeds in SEEDS.items():
        j = {
            "name": f"5i28_x{n}",
            "modelSeeds": list(range(seeds[n])),
            "dialect": "alphafold3",
            "version": 2,
            "sequences": [{"protein": prot}],
        }
        out = B / kit / "inputs" / f"x{n}" / "5i28.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(j))
        print("WROTE", out, "n", n, "seeds", seeds[n], "bytes", out.stat().st_size)
