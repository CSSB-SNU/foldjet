# Chai-1 kit — the 5I28 benchmark

Spec: `docs/guides/BENCHMARK-5I28.md`. Kit: `$KITS_DIR/chai1` (read only: tree, venv130, weights, kalign).

## Run

```
CUDA_VISIBLE_DEVICES=<gpu> ./run.sh N        # N = 1 2 4 8 12 16, on a GPU node, host shell
# e.g. srun -p h100 -A cssb --qos=normal_h100 -n 1 -c 8 --mem=96G --gres=gpu:h100:1 --time=02:00:00 ./run.sh 4
```

`run.sh` enters the kit's pinned container (`scripts/cu130run.sh`, a copy of `anthropic-kits/cu130run.sh` with singularity cache/tmp here),
sources `scripts/env_in.sh` (venv130, kalign on PATH, every cache/JIT root/TMPDIR here, `PYTHONPATH=shim`, `REAL_CIF_DIR=inputs/cifs`) and runs

```
run.sh pred --config h100 --mode fast --seed 0 --input out/xN/in/5i28_xN.fasta --out_dir out/xN/out --msa-directory inputs/msa \
  --template-hits-path out/xN/in/templates.m8 --num-trunk-recycles 3 --num-diffn-timesteps 200 --num-diffn-samples 5 --num-trunk-samples F
```

F forwards in one process (seeds 0..F-1): F = 4 (N=1), 3 (N<=8), 2 (N>=12); `FWDS=k` overrides, `MODE=exact` changes the kit mode.
Log: `logs/xN.log`. Last line printed:

```
KIT chai1 n=<N> exit=<rc> fwd=[<per forward, cold first>] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
```

## Inputs (`scripts/prep.py`, built 2026-10-29 on CPU, `check/cpu_check.log`)

- `inputs/5i28_xN.fasta`: N chains `>protein|name=A..`.
- `inputs/msa/<sha256>.aligned.pqt`: `5i28_unpaired.a3m` (6789 a3m records incl. the query) -> chai's parquet (UNIREF90 tag, no pairing key),
  as the kit's earlier real runs.
- `inputs/templates_nN.m8`: per chain 4 hits, e-values 1e-30..4e-30 (chai sorts by e-value, keeps up to 4, no sequence dedupe):
  5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A. Subject ranges (1-based, in the resolved auth-chain sequence) are the span kalign aligns, so chai's
  TemplateHit check passes: 5SYD 1-88, 6KOL 2-124, 5YT7 20-133, 6TFD 63-213 (6TFD chain A is a 423-residue fusion chain).
- `inputs/cifs/<ID>.cif.gz` = `~/data/cif_raw/<..>/<id>.cif.gz`; the shim copies them where chai would download them from RCSB.

## Timing: the forward-only span

The kit's `FORWARD ... forward_s` line wraps all of upstream's `run_folding_on_context`: it also collates the features before the model and
ranks the 5 samples and writes the CIF files after it (audit `docs/records/anthropic_kits_fwd_definition_audit_20261029.md`). So `fwd` here
is the shim's span (`shim/sitecustomize.py`), per fold:

```
[bench5i28] FWDSPAN fold=k out=.. fwd_s=<s> pre_s=<collate etc.> post_s=<scores, D2H, ranking, CIF writing> call_s=<whole call> conf_calls=5 loads=6
```

START = `torch.cuda.synchronize()` + CUDA event at the fold's first `load_exported` call (upstream calls it right after `Collate`, before the
feature embedder); END = CUDA event after the 5th confidence-head forward (the loaded part wrapped in a pass-through proxy; no syncs between
samples); read after the call returns and synchronizes. Covered: feature/bond/token embedders, 4 trunk passes, 200 sampler steps x 5 samples,
confidence head x 5. Not covered: Collate featurization, confidence logits -> pAE/pLDDT scores and the copies to CPU, ranking, CIF/npz/msa-plot
writing. `load_exported` is rebound only during each fold call and restored after; the kit's own FORWARD line stays in the log
(`census` line) for comparison. `warm_median` = median of all forwards after the first. The span logic was checked on CPU against a mock fold
(fwd/pre/post as expected); it has not run on the real kit on a GPU yet, so check the first log: `conf_calls=5`, `fwd_s` < kit `forward_s`.

## templates_real and msa_rows

- `templates_real`: chai_lab's own `Loaded k templates: ['5SYD|A', '6KOL|A', '5YT7|A', '6TFD|A']` (INFO, one line per chain): the number of
  the 4 expected names listed, min over lines (0 if fewer lines than chains), capped by the shim's `CONTEXT ... nonnull_templates=` (the merged
  TemplateContext the fold receives).
- `msa_rows`: chai_lab's `MSA found for sequence: ..., msa.depth=<d>` (distinct values), else the shim's `CONTEXT ... msa_depth=`.

CPU check (`scripts/cpu_check_inner.sh`, chai's own `make_all_atom_feature_context` without ESM, n = 1, 2): `Loaded 4 templates: ['5SYD|A',
'6KOL|A', '5YT7|A', '6TFD|A']` per chain, `TemplateContext(num_templates=4, num_nonnull_templates=4)`, msa depth 6789. Template residues with
coordinates per slot (n=1): 88 / 120 / 114 / 127 (FoldJet's rows: 88 / 120 / 114 / 101 -- chai's kalign maps 6TFD onto all 127 positions).

## Notes

- Exit code: the kit's earlier real-input fast/exact runs exited 1 although every fold finished, because its `templ_empty` lever is refused
  when templates are present (`PAIRTRACK GATE REFUSED templ_empty ... (all_fallback:templates_present)`). Expect the same; the census lines at
  the end of the log show the kit verdict.
- Caches: `cache/` and `jit/` were seeded (copied) from the kit's `cache130` / `jit_venv130`, the state of its earlier runs.
