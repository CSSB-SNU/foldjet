# adapter_fast: ft_adapter with a bit-identical reference-conformer cache

`adapter_fast/` is `adapter_v1/` (= the adapter of the first runs, kept untouched) plus ONE change: AF3's reference-conformer routine
`alphafold3.model.features.get_reference` is replaced by `fast_ref.get_reference`, which returns exactly the same values. Nothing else
(sampler, seeds, MiniWorld load, MSA/template handling, padding, tensors, `FT_*` knobs, stream positions, resume via `FT_START_STEP`) is changed.
The sampled item names and every output tensor are bit-identical to `adapter_v1` (see "Proof").

## Why

Profile of one example in the old adapter (measured, 60 stream positions, 1 thread): 4.2 CPU-s per example, of which 86 % is `get_reference`
(604 RDKit ETKDGv3 embeddings per example: one per residue / ligand instance, in TWO passes: the per-sample pass for `ref_pos` and the
deterministic pass (AF3 seed 12312837) whose conformers only decide the ligand frames), incl. 0.3 s re-parsing the CCD cif into an RDKit mol for
every instance and ~0.3 s of per-atom Python overhead (`np.pad` in `_pad_to`).

## What changed (`fast_ref.py`, hooked in `convert.py`)

1. Per CCD code (per process): the parsed RDKit mol, the atom name / element / charge / name-char tables and the bond lists are cached.
2. The embedded conformer is a pure function of (CCD code, RDKit seed). In the deterministic frames pass the seed sequence is the same for every
   example (RandomState(12312837); the augmentation draws a coordinate-independent number of normals), so (CCD code, seed) repeats from example to
   example: those conformers come from an on-disk cache (`conf_cache/`, read-only, loaded once per worker) and misses are computed and kept in
   memory (bounded by `FT_CONF_MEM_MAX`, default 300000 entries).
3. The per-sample pass uses fresh random seeds per example (never repeated): it is always computed with RDKit (never cached, never stored).
4. `random_state` is advanced exactly as in the original (same `randint`, same `random_augmentation`), so every downstream draw is unchanged.
   CCD codes without a CCD entry (SMILES path) and codes whose mol fails to parse fall back to the original behaviour.

## How to enable

`source ~/foldforge-scratch/ft_train/env_fast.sh` instead of `env.sh` (the only difference is `PYTHONPATH=$FT/adapter_fast:...`). The run scripts
source `env.sh` through `common_match2.sh`; change that one line to `env_fast.sh`.

| variable | default | meaning |
|---|---|---|
| `FT_FAST_REF` | 1 | `0` = the original `get_reference` (no cache): same output, old speed |
| `FT_CONF_CACHE` | `$FT/conf_cache` | directory (all `*.pkl` below it) or file of the conformer cache; missing = no preload (misses are computed lazily) |
| `FT_CONF_MEM_MAX` | 300000 | cap on in-memory cache entries (frames pass only) |

All other `FT_*` knobs are as in `README.md` (this directory also keeps the old README).

## The cache and how to rebuild it

`~/foldforge-scratch/ft_train/conf_cache/{std,lig}/conf_*_K384.pkl` (about 61 MB): `{(ccd_code, rdkit_seed, None): float32 [n_atoms, 3] | None}` for
31 standard residue / nucleotide codes (`std/`) and 357 further CCD codes seen in 716 harvested crops (`lig/`), for the first K=384 seeds of the
frames pass (an example has at most 384 residue instances). The pre-augmentation conformer is stored; the augmentation is applied at use.

Rebuild / extend (Slurm CPU jobs, `-c 32`, each shard is resumable, one log line per code; see `ft_train/fast_tools/`):

```
sbatch fast_tools/conf_harvest.sh ADAPTER_DIR RANK START N NPROC OUT.json      # count CCD codes of N stream positions (cold I/O: ~1.5 examples/s)
# codes json = list of CCD codes, e.g. jq over the harvest counts
sbatch fast_tools/conf_build.sh ADAPTER_DIR CODES.json 384 SHARD_SIZE 31 conf_cache/NAME [MAXSEC]
```
A shard that is missing is only a speed loss, never a correctness issue (the key contains the seed).

## Proof (bit-identity vs `adapter_v1`)

`fast_tools/dump_batches.py` + `compare_batches.py`: 240 stream positions (rank 0 positions 0-59, rank 1 positions 1280-1339, rank 0 6400-6459,
rank 1 38400-38459), the exact worker output of `SampleDataset.__getitem__` (65 tensors per example: 60 features + positions / mask / bond pairs /
present index / present valid): 15600 tensors, 0 mismatches under `torch.equal` (+ dtype and shape), sampled item names and info counters equal.
