"""CPU check (no GPU): Boltz-2's own input parsing of the bench YAMLs.

boltz.data.parse.yaml.parse_yaml (boltz2=True), which reads each template mmCIF, takes
its label chain A and aligns it to every query chain
(get_template_records_from_matching) -- and the MSA through boltz's parse_a3m
(max_msa_seqs 8192, upstream's default). usage (inside the container, boltz venv130):
check.py N [N ...]
"""

import os
import sys
from pathlib import Path

from boltz.data.mol import load_canonicals
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.yaml import parse_yaml

IN = Path(os.environ["BENCH_DIR"] + "/boltz2/inputs")
CACHE = Path(os.environ["KITS_DIR"] + "/boltz2/weights")
mol_dir = CACHE / "mols"
ccd = load_canonicals(mol_dir)
msa = parse_a3m(Path(os.environ["A3M_5I28"]), taxonomy=None, max_seqs=8192)
print(
    f"[check] boltz parse_a3m: sequences={len(msa.sequences)} (max_seqs 8192)",
    flush=True,
)
for n in map(int, sys.argv[1:]):
    # positional boltz2 flag: boltz's own parse_yaml signature
    t = parse_yaml(IN / f"5i28_x{n}.yaml", ccd, mol_dir, True)  # noqa: FBT003
    recs = t.record.templates or []
    names = sorted({r.name for r in recs})
    print(
        f"[check] n={n} chains={len(t.record.chains)} template_records={len(recs)} "
        f"distinct_names={len(names)} {names} "
        f"template_structures={sorted((t.templates or {}).keys())}",
        flush=True,
    )
    for r in recs:
        if r.query_chain == "A":
            print(
                f"[check]   n={n} {r.name}: query "
                f"{r.query_chain}[{r.query_st},{r.query_en}) "
                f"<- template chain {r.template_chain}"
                f"[{r.template_st},{r.template_en})",
                flush=True,
            )
