# 5I28 benchmark inputs

Every input of the 5I28 benchmark ([BENCHMARK-5I28.md](../../docs/guides/BENCHMARK-5I28.md)), so a run needs nothing outside the
repository: the real MSA and the 4 distinct real templates, for FoldJet and for the kits. The chain sequence is
[`tests/release_references/inputs/5i28/A.fasta`](../../tests/release_references/inputs/5i28/A.fasta).

| file | used by | content |
|---|---|---|
| `a3m_16k.lmdb/data.mdb` | FoldJet `msa_db` (`FOLDJET_5I28_MSA_DB`) | key `P0113359` of `a3m_16k.lmdb`, BioMolDB_20260224: the real MSA |
| `template.lmdb/data.mdb` | FoldJet `template_db` (`FOLDJET_5I28_TEMPLATE_DB`) | key `P0113359` of `template.lmdb`, BioMolDB_20260224: 20 records, 6 distinct sequences; with `template_n: 4` and the default `template_dedup` FoldJet loads `5SYD_A`, `6KOL_A`, `5YT7_A`, `6TFD_A` |
| `5i28_unpaired.a3m` | kits (`A3M_5I28`) | the same MSA as an a3m (`Alignment.load(...).a3m(species_aliases(...))`) |
| `5i28_templates_hits20.a3m` | Protenix-family / OpenFold3 kits | the 20 template hits as an a3m; the kits keep one hit per sequence, which gives the same four |
| `mmcif/{5syd,6kol,5yt7,6tfd}.cif.gz` | kits that take template coordinates as mmCIF | the four template entries (PDB, CC0) |

The two LMDBs hold only the key `P0113359`, cut from the full BioMolDB_20260224 stores. They open read-only without a lock file
(`lock=False`), so the directories hold only `data.mdb`.

Checked 2026-11-26 (CPU job, FoldJet `rename-foldjet`): the MSA read from `a3m_16k.lmdb` here is byte-identical to the a3m exported
from the full BioMolDB_20260224 on 2026-10-24 (`5i28_unpaired.a3m`); `load_templates(..., 4)` gives `5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A`;
`foldjet.data.inputs.build.load` on the n = 1 and n = 4 specs gives 4 templates and an MSA of 6789 a3m entries per chain.

## SHA-256

```
dd57224797569c3a034fab8be3ed915688fd37f18601c306be879f591184a089  a3m_16k.lmdb/data.mdb
4b8077ec1f566d86954cdee506e6f2ba737d87f11a6065522f5a1ca13047eb57  template.lmdb/data.mdb
1bb919d70282ba603c5a4637a77e707c8766c934597397485f77138aea3cea0a  5i28_unpaired.a3m
b1ea8b0e88c5d08e0461b9bb3c19c7814474e61b5b7e4789432b3e6f54abc9f1  5i28_templates_hits20.a3m
0aff7ce969cda8584484b148fa35050e2447befb7ed9134c3fbd1b497fc23070  mmcif/5syd.cif.gz
73828a6334a076691462f5bf4596cec7398884dd388945f10c4a8df47b2b82ac  mmcif/6kol.cif.gz
b47077733ef4961cfa0ed9f0d6a0478a9d540f14b2d135778bb20c8b897aa51b  mmcif/5yt7.cif.gz
ec8a177c062f06849ca95d668c7b29fa32739f33da1ed02b574c8fd6fc933cf6  mmcif/6tfd.cif.gz
```
