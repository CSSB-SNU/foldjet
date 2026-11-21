"""F5: lay out one (arm, model) of the F4 runs for the FoldBench evaluator (CPU only, Slurm --gres=none).

    python prepare_eval.py RUNS_ROOT ARM MODEL EVAL_ROOT

RUNS_ROOT is the F4 output dir (rows/<arm>__<model>.jsonl, pred/<arm>/<model>/<target>/). For every target with an `ok` row it writes
EVAL_ROOT/<arm>__<model>/pred/<target>/<target>_seed0_sample<k>_pred.cif (the 5 samples, chains renamed to FoldBench's label_asym_id through
chain_names.json: the prediction lists its chains in the input's chain order) and ranking.json {"<k>": mean pLDDT} (the ranking the
evaluator's `rank` column uses; F5 reports the mean over the 5 samples, which does not depend on it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

CHAIN_COLUMNS = {
    "atom_site": ("label_asym_id", "auth_asym_id"),
    "struct_asym": ("id",),
    "entity_poly": ("pdbx_strand_id",),
    "pdbx_poly_seq_scheme": ("asym_id", "pdb_strand_id"),
    "pdbx_nonpoly_scheme": ("asym_id", "pdb_strand_id"),
}


def rename_chains(src: Path, dst: Path, names: dict[str, str]) -> None:
    cif = pdbx.CIFFile.read(src)
    block = cif.block
    order = list(dict.fromkeys(block["atom_site"]["label_asym_id"].as_array(str).tolist()))
    index_order = [names[k] for k in sorted(names, key=int)]
    if len(order) != len(index_order):
        raise ValueError(f"{len(order)} chains in the CIF, {len(index_order)} in chain_names")
    mapping = dict(zip(order, index_order, strict=True))
    for category, columns in CHAIN_COLUMNS.items():
        if category not in block:
            continue
        for column in columns:
            if column not in block[category]:
                continue
            values = block[category][column].as_array(str).tolist()
            new = [",".join(mapping.get(v, v) for v in str(value).split(",")) for value in values]
            block[category][column] = pdbx.CIFColumn(pdbx.CIFData(new))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cif.write(dst)


def main() -> None:
    runs, arm, model, eval_root = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    rows = [json.loads(line) for line in (runs / "rows" / f"{arm}__{model}.jsonl").read_text().splitlines()]
    ok = {r["target"] for r in rows if r.get("status") == "ok"}
    pred = eval_root / f"{arm}__{model}" / "pred"
    done = failed = 0
    for target in sorted(ok):
        src = runs / "pred" / arm / model / target
        try:
            names = json.loads((src / "chain_names.json").read_text())
            plddt = np.load(next(src.glob("*.npz")))["plddt"]  # [samples, atoms], 0..1
            for k in range(plddt.shape[0]):
                rename_chains(src / f"{target}-seed0-{k}.cif", pred / target / f"{target}_seed0_sample{k}_pred.cif", names)
            (pred / target / "ranking.json").write_text(json.dumps({str(k): float(plddt[k].mean()) for k in range(plddt.shape[0])}))
            done += 1
        except Exception as error:  # noqa: BLE001 - a target that cannot be laid out is reported, the rest go on
            failed += 1
            print(f"prepare failed {target}: {type(error).__name__}: {str(error)[:200]}", flush=True)
    print(f"prepared {done} targets, {failed} failed -> {pred}", flush=True)


if __name__ == "__main__":
    main()
