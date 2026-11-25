# Boltz-2 kit — the 5I28 benchmark

Spec: `docs/guides/BENCHMARK-5I28.md`. Kit: `$KITS_DIR/boltz2` (read only: tree, venv130, weights).

## Run

```
CUDA_VISIBLE_DEVICES=<gpu> ./run.sh N        # N = 1 2 4 8 12 16, on a GPU node, host shell
# e.g. srun -p h100 -A cssb --qos=normal_h100 -n 1 -c 8 --mem=160G --gres=gpu:h100:1 --time=02:00:00 ./run.sh 4
```

`run.sh` enters the kit's pinned container (`scripts/cu130run.sh`, a copy of `anthropic-kits/cu130run.sh` with singularity cache/tmp here),
sources `scripts/env_in.sh` (venv130; every cache, JIT root and TMPDIR under this directory), and runs

```
run.sh pred --config h100 --mode fast --input out/xN/in --out_dir out/xN/out --recycling_steps 3 --sampling_steps 200 \
  --diffusion_samples 5 --seed 0 --output_format mmcif --override
```

`out/xN/in` holds F copies of `inputs/5i28_xN.yaml` (a0..): F forwards in one process, F = 4 (N=1), 3 (N<=8), 2 (N>=12); `FWDS=k` overrides,
`MODE=exact|off` changes the kit mode. Log: `logs/xN.log` (kit stdout+stderr, then `out/xN/out/pred_worker.log` appended). Last line printed:

```
KIT boltz2 n=<N> exit=<rc> fwd=[<per forward, cold first>] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
```

## Inputs

- `inputs/5i28_xN.yaml`: N chains A.., each `msa: $FF_SCRATCH/prec_vs_kit/real/5i28_unpaired.a3m` (the real MSA, 6789 a3m records
  incl. the query); `templates:` 5syd_A, 6kol_A, 5yt7_A, 6tfd_A (`inputs/tpl/<id>_A.cif` = `~/data/cif_raw/<..>/<id>.cif.gz` unzipped),
  `chain_id: [all chains]`, `template_id: [A, ...]`. Template name = file stem, so 4 distinct names.
- Boltz reads the template's label chain A (gemmi subchain); for all four files label A = auth A (`check/` and `../chai1/check/cpu_check.log`).

## What the summary fields are

- `fwd`: the kit's `PHASE item=aK ... fwd_s=` per input = `Boltz2.forward`, CUDA-synchronized (the span FoldJet times; audit
  `docs/records/anthropic_kits_fwd_definition_audit_20261029.md`). `warm_median` = median of all forwards after the first.
- `templates_real`: min over inputs of min(names, live_slots) from the kit's model-entry census `[boltz2-opt] TEMPLATES record=aK declared=..
  names=.. slots=.. live_slots=.. real=j/T`, capped by how many of the 4 expected names appear in the featurizer census
  (`pred_worker.log`: `TEMPLATE record=.. template=<name> chain=.. aligned_residues_resolved=<n>/..` with n > 0).
- `msa_rows`: rows of the MSA Boltz itself processed for the run (`out/xN/out/boltz_results_*/processed/msa/*.npz`, printed as `MSA_ROWS` by
  `scripts/msa_npz.py`); the kit log has no MSA-depth line. No `--subsample_msa` is passed (upstream default off), max_msa_seqs 8192.

## CPU check (done 2026-10-29, `check/cpu_check.log`)

`scripts/cpu_check_inner.sh` (srun, no GPU, inside the container): boltz's own `parse_yaml` on n = 1, 2, 16 gives 4 distinct template names
per chain (4N records) and `parse_a3m` gives 6789 sequences. Boltz aligns a template to the query with an UNGAPPED local alignment
(`schema.get_local_alignments`: blastp scoring, gap open/extend -1000), so per query chain the templates cover:
5syd_A query [42,128) (86 res), 6kol_A [48,97) (49), 5yt7_A [0,114) (114), 6tfd_A [106,117) (11). That is stock Boltz-2's template form, not
FoldJet's alignment (88 / 120 / 114 / 101 residues); the kit still counts each as a live slot when it carries coordinates.

## Notes

- Caches: `cache/` and `jit/` were seeded (copied) from the kit's `cache130` / `jit_venv130`, the state the kit's earlier real-input runs had.
- The earlier real-input run at n = 12 (fast) exited 3 (`NOT ACTIVE: fused TriMul gate refused ... esm_shapes:batch`) after its forwards; if it
  recurs, the forwards are still in the log, the exit code is the kit's verdict.
