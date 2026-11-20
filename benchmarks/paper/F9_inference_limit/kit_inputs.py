"""F9, kit arm: the kit inputs and commands for sizes beyond the n = 1 .. 16 the 5I28 kit harness prepared (2026-11-18 steady harness, `$KIT_BENCH`).

`ensure(kit, n, bench)` writes the kit's own input form of n identical 5I28 chains if it is missing (new files in the harness only: nothing existing is edited,
no kit code is touched) and returns None, or a reason string when this kit's input builder is not automated. `command(kit, n, bench)` = (shell command, log
path) for ONE n, ONE forward where the kit allows (AF3 kits: modelSeeds [0]; Protenix family: the harness script's own 7 seeds; ESMFold2: FWD_CALLS=1).
Same inputs as the benchmark: the real MSA and the 4 distinct real templates are the ones inside the n = 16 input; only the chain count changes.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
from pathlib import Path


def chain_ids(n: int) -> list[str]:
    """A..Z, then AA, AB, ... (AF3 / Boltz / Chai chain ids are strings)."""
    letters = string.ascii_uppercase
    out = list(letters[: min(n, 26)])
    i = 0
    while len(out) < n:
        out.append(letters[i // 26] + letters[i % 26])
        i += 1
    return out


def ensure(kit: str, n: int, bench: Path) -> str | None:
    b = bench
    if kit in ("af3_jax", "af3_torch"):
        dst = b / kit / "inputs" / f"x{n}" / "5i28.json"
        if dst.exists():
            return None
        doc = json.loads((b / kit / "inputs/x16/5i28.json").read_text())
        doc["name"] = f"5i28_x{n}"
        doc["modelSeeds"] = [0]
        doc["sequences"][0]["protein"]["id"] = chain_ids(n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(doc))
        return None
    if kit in ("protenix_v1", "protenix_v2", "opendde"):
        dst = b / "in" / f"5i28_x{n}.json"
        if dst.exists():
            return None
        doc = json.loads((b / "in/5i28_x16.json").read_text())
        doc[0]["name"] = f"5i28_x{n}"
        doc[0]["sequences"][0]["proteinChain"]["count"] = n
        dst.write_text(json.dumps(doc))
        return None
    if kit == "openfold3":
        dst = b / "openfold3" / "inputs" / f"x{n}.json"
        if dst.exists():
            return None
        subprocess.run(["bash", str(b / "openfold3/make_inputs.sh"), str(n), "1", str(dst)], check=True)  # noqa: S603, S607
        return None
    if kit == "boltz2":
        dst = b / "boltz2" / "inputs" / f"5i28_x{n}.yaml"
        if dst.exists():
            return None
        text = (b / "boltz2/inputs/5i28_x16.yaml").read_text()
        head, _, rest = text.partition("sequences:\n")
        blocks = re.split(r"(?m)^(?=  - protein:)", rest)
        blocks = [x for x in blocks if x.strip()]
        tail = ""
        if blocks and not blocks[-1].lstrip().startswith("- protein:"):
            tail = blocks.pop()
        first = blocks[0]
        out = [re.sub(r"(?m)^(      id: ).*$", lambda m, cid=cid: m.group(1) + cid, first) for cid in chain_ids(n)]
        dst.write_text(head + "sequences:\n" + "".join(out) + tail)
        return None
    if kit == "chai1":
        fa, m8 = b / "chai1/inputs" / f"5i28_x{n}.fasta", b / "chai1/inputs" / f"templates_n{n}.m8"
        if fa.exists() and m8.exists():
            return None
        seq = [x for x in (b / "chai1/inputs/5i28_x1.fasta").read_text().splitlines() if x and not x.startswith(">")][0]
        ids = chain_ids(n)
        fa.write_text("".join(f">protein|name={c}\n{seq}\n" for c in ids))
        lines = [x for x in (b / "chai1/inputs/templates_n1.m8").read_text().splitlines() if x.strip()]
        m8.write_text("".join(re.sub(r"^\S+", c, line) + "\n" for c in ids for line in lines))
        return None
    if kit in ("esmfold2", "esmfold2_fast"):
        return None  # esm_fwd_multi.py builds the n-chain input itself (SIZES)
    return f"no automated input builder for kit {kit} beyond n = 16 (rf3: threaded template PDB per n; intellifold2: AF3-JAX overlay inputs)"


def command(kit: str, n: int, bench: Path) -> tuple[str, str]:
    b = bench
    if kit in ("af3_jax", "af3_torch"):
        return f"bash {b}/{kit}/run.sh {n}", f"{b}/{kit}/logs/x{n}.log"
    if kit in ("protenix_v1", "protenix_v2", "opendde"):
        return f"bash {b}/protenix_family_run.sh {kit} {n}", f"{b}/{kit}/logs/x{n}.log"
    if kit == "openfold3":
        return f"COPIES=1 bash {b}/openfold3/run.sh {n}", f"{b}/openfold3/logs/x{n}.log"
    if kit == "chai1":
        return f"FWDS=1 bash {b}/chai1/run.sh {n}", f"{b}/chai1/logs/x{n}.log"
    if kit == "boltz2":
        return f"DONE=/tmp/f9_done_boltz2_{n} TAG=f9_n{n} SIZES={n}:1 bash {b}/boltz2/run_multi.sh", f"{b}/boltz2/logs/f9_n{n}.log"
    if kit == "esmfold2":
        return f"DONE=/tmp/f9_done_esm_{n} KNAME=esmfold2 TAG=f9_n{n} SIZES={n} FWD_CALLS=1 bash {b}/esmfold2/run_multi.sh", f"{b}/esmfold2/logs/f9_n{n}.log"
    if kit == "esmfold2_fast":
        return (f"DONE=/tmp/f9_done_esmf_{n} VARIANT=fast ESM_REPO=biohub/ESMFold2-Fast KNAME=esmfold2_fast TAG=f9_n{n} SIZES={n} FWD_CALLS=1 "
                f"bash {b}/esmfold2_fast/run_multi.sh"), f"{b}/esmfold2_fast/logs/f9_n{n}.log"
    raise ValueError(kit)
