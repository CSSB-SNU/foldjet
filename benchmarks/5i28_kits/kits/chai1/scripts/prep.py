"""CPU prep for the 5I28 benchmark (benchmarks/5i28.md), run INSIDE the cu130 container with the Chai-1 kit's venv130 (scripts/prep_inner.sh).

Writes under bench5i28/chai1/inputs:
  msa/<sha256(query)>.aligned.pqt   the real MSA (5i28_unpaired.a3m, all rows) as chai's aligned parquet (UNIREF90 tag, no pairing key; as the
                                     kit's earlier real runs, anthropic-kits/chai1/real/prep.py)
  templates_n<N>.m8                 4 hits per chain, in this order (e-values 1e-30..4e-30; chai sorts by e-value): 5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A
                                     subject ranges = the span kalign aligns, so chai's TemplateHit consistency check passes (as real/prep2.py)
  5i28_x<N>.fasta                   N identical chains, names A, B, ...
Also prints, per template, the mmCIF chains (auth name -> label subchain, resolved polymer length) and whether the query-aligned row of the
20-hit a3m (kitv2real/in/5i28_templates_hits20.a3m, = FoldForge's template) is the polymer of auth chain A (chai) and of label chain A (Boltz).
"""

import os
import re
from pathlib import Path

import gemmi
import torch
from chai_lab.data.parsing.msas.a3m import tokenize_sequences_to_arrays
from chai_lab.data.parsing.msas.aligned_pqt import (
    a3m_to_aligned_dataframe,
    expected_basename,
)
from chai_lab.data.parsing.msas.data_source import MSADataSource
from chai_lab.data.parsing.templates.template_hit import TemplateHit
from chai_lab.tools.kalign import kalign_query_to_reference

B = Path(os.environ["BENCH_DIR"] + "/chai1")
IN = B / "inputs"
A3M = Path(os.environ["A3M_5I28"])
HITS = Path(os.environ["INPUT_SCRATCH"] + "/in/5i28_templates_hits20.a3m")
TPL = [("5SYD", "A"), ("6KOL", "A"), ("5YT7", "A"), ("6TFD", "A")]
NS = (1, 2, 4, 8, 12, 16)

lines = A3M.read_text().split("\n")
query = lines[1].strip()
assert len(query) == 128, len(query)
(IN / "msa").mkdir(parents=True, exist_ok=True)
fr = a3m_to_aligned_dataframe(A3M, MSADataSource.UNIREF90, insert_pairing_key=False)
fr.to_parquet(IN / "msa" / expected_basename(query))
print(
    f"[prep] MSA {A3M}: {A3M.read_text().count('>')} a3m records -> {len(fr)} parquet rows -> {IN / 'msa' / expected_basename(query)}",
    flush=True,
)

# the 20-hit a3m: header '>5yt7_A/1-114 ...' -> query-aligned row
hits, name = {}, None
for ln in HITS.read_text().splitlines():
    if ln.startswith(">"):
        name = ln[1:].split("/")[0]
    elif name:
        hits[name] = ln.strip()
        name = None


def resolved_seq(ch):
    return ch.get_polymer().make_one_letter_sequence().replace("-", "")


rng = {}
for pid, ch in TPL:
    cif = IN / "cifs" / f"{pid}.cif.gz"
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    row = hits[f"{pid.lower()}_{ch}"]
    tseq = "".join(c for c in row if c.isalpha() and c.isupper())
    print(f"[prep] {pid}: {cif} chains:", flush=True)
    label_a = None
    for c in st[0]:
        subs = [s.subchain_id() for s in c.subchains()]
        poly = c.get_polymer()
        psub = poly.subchain_id() if len(poly) else "-"
        if psub == ch:
            label_a = c.name
        print(
            f"[prep]   auth={c.name} polymer_label={psub} subchains={subs} resolved_polymer_len={len(resolved_seq(c))}",
            flush=True,
        )
    auth_seq = resolved_seq(st[0][ch])
    print(
        f"[prep]   hits20 row {pid.lower()}_{ch}: {len(tseq)} aligned residues; in auth chain {ch} resolved polymer: {tseq in auth_seq}; "
        f"label chain {ch} = auth chain {label_a}",
        flush=True,
    )
    s0, e0 = 0, len(auth_seq)
    for it in range(8):
        al = kalign_query_to_reference(ref=query, query=auth_seq[s0:e0])
        ms, md = tokenize_sequences_to_arrays([al.query_a3m_line])
        try:
            h = TemplateHit(
                query_pdb_id="A",
                query_sequence=query,
                index=0,
                pdb_id=pid,
                chain_id=ch,
                hit_start=s0,
                hit_end=e0,
                hit_tokens=torch.from_numpy(ms[0]).int(),
                deletion_matrix=torch.from_numpy(md[0]),
                query_seq_realigned=al.query_a3m_line,
            )
            assert h.query_start_end == al.reference_span
            print(
                f"[prep]   kalign: OK subject [{s0},{e0}) of {len(auth_seq)} resolved residues; query span {h.query_start_end}; "
                f"aligned residues {len(h.indices_hit)}",
                flush=True,
            )
            rng[(pid, ch)] = (s0, e0)
            break
        except ValueError as e:
            m = re.search(r"indices: (\d+) (\d+) !=", str(e))
            if not m:
                raise
            a, b = map(int, m.groups())
            print(
                f"[prep]   kalign: iter {it} range [{s0},{e0}) -> aligned hit idx {a}..{b}",
                flush=True,
            )
            s0, e0 = s0 + a, s0 + b + 1
    else:
        raise SystemExit(f"{pid}_{ch}: no consistent subject range")

for n in NS:
    ids = [chr(ord("A") + i) for i in range(n)]
    rows = []
    for c in ids:
        for k, (pid, ch) in enumerate(TPL):
            s0, e0 = rng[(pid, ch)]
            rows.append(
                "\t".join(
                    [
                        c,
                        f"{pid}_{ch}",
                        "90.0",
                        str(e0 - s0),
                        "0",
                        "0",
                        "1",
                        "128",
                        str(s0 + 1),
                        str(e0),
                        f"{k + 1}e-30",
                        "200.0",
                        "-",
                    ]
                )
            )
    (IN / f"templates_n{n}.m8").write_text("\n".join(rows) + "\n")
    (IN / f"5i28_x{n}.fasta").write_text(
        "".join(f">protein|name={c}\n{query}\n" for c in ids)
    )
print("[prep] templates_n1.m8:\n" + (IN / "templates_n1.m8").read_text(), flush=True)
print("[prep] done", flush=True)
