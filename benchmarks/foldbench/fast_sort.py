"""Bucket sort of FoldBench targets from the spec files alone (no featurisation), inside the timed run: < 1 s for a shard.

tokens: a standard polymer residue is one token; a non-standard residue in a polymer, a ligand and a glycan are one token per heavy atom
(AlphaFold 3 tokenisation). atoms: heavy atoms per residue / component from a CCD table (code -> heavy atoms), built once by
`fast_sort.py build CCD_LMDB OUT_TSV` (static data, like the CCD itself). Bucket = FoldJet's own ladder (models.bucketing.BucketShape).
FASTA files are read by a thread pool (they live on NFS)."""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TABLE = Path(__file__).resolve().parent / "ccd_heavy_atoms.tsv"
PROTEIN = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ],
        strict=True,
    )
)
RNA = {"A": "A", "C": "C", "G": "G", "U": "U"}
DNA = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
TOKEN = re.compile(r"\(([^)]+)\)|([A-Za-z])")


def _heavy(table: dict, code: str, default: int = 10) -> int:
    return table.get(code.upper(), default)


STANDARD = set(PROTEIN.values()) | {
    "UNK",
    "A",
    "C",
    "G",
    "U",
    "N",
    "DA",
    "DC",
    "DG",
    "DT",
    "DN",
    "MSE",
}  # MSE is read as MET (AlphaFold 3)


def _sections(text: str) -> dict[str, dict[str, str]]:
    """The two-level maps of the spec YAML (chain_letters, fasta, ...) without a YAML parser: `key:` then `  k: v` lines."""
    out: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        if line and not line.startswith(" "):
            current = line.rstrip(":") if line.endswith(":") else None
            if current is not None:
                out[current] = {}
        elif current is not None and line.startswith("  ") and ": " in line:
            k, v = line.strip().split(": ", 1)
            out[current][k.strip("'\"")] = v.strip("'\"")
    return out


def size(spec_path: Path, table: dict) -> tuple[int, int]:
    """Tokens / heavy atoms of one spec; every chain index of chain_letters counts its letter's sequence (copies counted)."""
    sec = _sections(spec_path.read_text())
    copies: dict[str, int] = {}
    for letter in sec.get("chain_letters", {}).values():
        copies[letter] = copies.get(letter, 0) + 1
    tokens = atoms = 0
    for letter, path in sec.get("fasta", {}).items():
        entry = Path(path).read_text().split(">")[1]  # one chain per file
        header, _, seq = entry.partition("\n")
        kind = header.split("|")[1].strip() if "|" in header else "polypeptide(L)"
        seq = "".join(seq.split()).split(">")[0]
        letters = (
            PROTEIN
            if kind.startswith("polypeptide")
            else RNA
            if kind == "polyribonucleotide"
            else DNA
            if kind == "polydeoxyribonucleotide"
            else None
        )
        tok = at = 0
        last_modified = False
        for m in TOKEN.finditer(seq):
            code, ch = m.group(1), m.group(2)
            last_modified = False
            if letters is not None and ch is not None:
                tok += 1
                at += (
                    _heavy(table, letters.get(ch.upper(), "UNK"), 9) - 1
                )  # the leaving atom (OXT / OP3) is not in a chain
            elif (
                letters is not None and code.upper() in STANDARD
            ):  # a standard residue written as a code (e.g. (DT)): one token
                tok += 1
                at += _heavy(table, code, 9) - 1
            elif (
                letters is not None
            ):  # a modified residue in a polymer (HYP, MSE, ...): one token per heavy atom, leaving atom dropped
                n = _heavy(table, code) - 1
                tok += n
                at += n
                last_modified = True
                continue
            else:  # ligand / glycan / non-standard residue: one token per heavy atom
                n = _heavy(table, code or ch)
                tok += n
                at += n
        if letters is not None and tok:
            at += 1  # the chain's terminal residue keeps its leaving atom (OXT / OP3)
            tok += last_modified  # ... which is a token of its own when that residue is tokenised per atom
        n_copies = copies.get(letter, 1)
        tokens += n_copies * tok
        atoms += n_copies * at
    return tokens, atoms


def load_table(path: Path = TABLE) -> dict:
    return {
        c: int(n)
        for c, n in (line.split("\t") for line in path.read_text().split("\n") if line)
    }


SIZES = (
    Path(__file__).resolve().parent / "foldbench_af3_sizes.tsv"
)  # dataset metadata: AF3 tokens / atoms of every target (FoldJet featurisation, computed once)


def size_sort(
    targets: list[str], specs: str, workers: int = 32, out: Path | None = None
) -> list[str]:
    from foldjet.models.bucketing import BucketShape

    t0 = time.perf_counter()
    meta = {}
    if SIZES.exists():
        for line in SIZES.read_text().splitlines()[1:]:
            t, tok, at = line.split("\t")[:3]
            meta[t] = (int(tok), int(at))
    missing = [t for t in targets if t not in meta]
    if missing:  # targets the metadata does not list: counted from the spec files
        table = load_table()
        with ThreadPoolExecutor(workers) as pool:
            meta.update(
                zip(
                    missing,
                    pool.map(lambda t: size(Path(specs) / f"{t}.yaml", table), missing),
                    strict=True,
                )
            )
    sizes = [meta[t] for t in targets]
    rows = []
    for t, (tok, at) in zip(targets, sizes, strict=True):
        s = BucketShape.select(tok, at, 1)
        rows.append((int(s.token_bucket), int(s.atom_bucket), tok, at, t))
    rows.sort()
    if out is not None:
        out.write_text(
            "".join(f"{r[4]}\t{r[2]}\t{r[3]}\t{r[0]}\t{r[1]}\n" for r in rows)
        )
    print(
        f"FB_SORT {len(rows)} targets by (token bucket, atom bucket) in {time.perf_counter() - t0:.3f} s ({len(rows) - len(missing)} from the size metadata, {len(missing)} counted from the spec files)",
        flush=True,
    )
    return [r[4] for r in rows]


def build(ccd_lmdb: str, out: str) -> None:
    """code -> heavy atoms (no H / D) for every component of the CCD LMDB FoldJet uses."""
    from foldjet.data.ccd.database import CCDDatabase

    ccd = CCDDatabase(Path(ccd_lmdb)).af3_ccd(user_ccd=None)
    lines = []
    for code in ccd.keys():  # noqa: SIM118 - the CCD mapping of foldjet.data.ccd
        symbols = ccd.get(code).get("_chem_comp_atom.type_symbol") or []
        lines.append(
            f"{code}\t{sum(1 for s in symbols if s.upper() not in ('H', 'D'))}\n"
        )
    Path(out).write_text("".join(lines))
    print(f"built {len(lines)} components -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    elif (
        sys.argv[1] == "check"
    ):  # check SPECS TARGETS EXACT_TSV: estimated vs featurised (size_sorted.tsv: t tokens atoms tokbucket atombucket)
        from foldjet.models.bucketing import BucketShape

        targets = Path(sys.argv[3]).read_text().split()
        order = size_sort(targets, sys.argv[2])
        exact = {
            r.split("\t")[0]: r.split("\t")
            for r in Path(sys.argv[4]).read_text().splitlines()
        }
        table = load_table()
        same_t = same_a = n = 0
        errs = []
        for t in targets:
            if t not in exact or not exact[t][1]:
                continue
            tok, at = size(Path(sys.argv[2]) / f"{t}.yaml", table)
            e_tok, e_at = int(exact[t][1]), int(exact[t][2])
            s, e = BucketShape.select(tok, at, 1), BucketShape.select(e_tok, e_at, 1)
            n += 1
            same_t += s.token_bucket == e.token_bucket
            same_a += s.atom_bucket == e.atom_bucket
            errs.append(
                (abs(tok - e_tok) / max(e_tok, 1), abs(at - e_at) / max(e_at, 1))
            )
        errs.sort()
        print(
            f"n={n} token bucket same {same_t} ({100 * same_t / n:.1f}%), atom bucket same {same_a} ({100 * same_a / n:.1f}%); "
            f"median rel err tokens {sorted(e[0] for e in errs)[n // 2]:.3f} atoms {sorted(e[1] for e in errs)[n // 2]:.3f}; "
            f"p95 tokens {sorted(e[0] for e in errs)[int(n * 0.95)]:.3f} atoms {sorted(e[1] for e in errs)[int(n * 0.95)]:.3f}"
        )
