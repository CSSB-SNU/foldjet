"""FoldBench target -> AlphaFold 3 JSON (dialect alphafold3, version 2) exactly as FoldForge writes it for its own af3 model
(foldforge/data/inputs/build.py Target.af3: species-tagged unpairedMsa = pairedMsa, AF3 templates with mmCIF), modelSeeds set to SEEDS.

  build_inputs.py SPEC_DIR OUT_DIR SEEDS TARGET...     (FoldForge env, CPU Slurm job through ../run_py.sh)
"""
import json
import sys
from pathlib import Path

import foldforge as ff

spec_dir, out, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
out.mkdir(parents=True, exist_ok=True)
for t in sys.argv[4:]:
    inp = ff.load_inputs(spec_dir / f"{t}.yaml")
    data = inp.target.af3(seeds[0])
    data["name"] = t
    data["modelSeeds"] = seeds
    (out / f"{t}.json").write_text(json.dumps(data))
    rows = []
    for s in data["sequences"]:
        k = next(iter(s))
        b = s[k]
        n_msa = (b.get("unpairedMsa") or "").count(">")
        rows.append(f"{k}:{len(b.get('sequence', '') or '')}aa,msa={n_msa},tpl={len(b.get('templates') or [])}")
    print("AF3_INPUT", t, " ".join(rows), flush=True)
