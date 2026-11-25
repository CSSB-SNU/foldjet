# Protenix v1 fine-tuning on MiniWorld data

`scripts/train_protenix1.py` trains FoldJet's Protenix v1 with `--recipe protenix` (Protenix v1's fine-tuning recipe; PAE loss on at
1e-4, Protenix's stage-3 value). The batches come from MiniWorld's own data loader through the adapter in this directory
(`--batch-source ft_adapter:batches`). FoldJet itself does not depend on MiniWorld; only this adapter does.

- `miniworld_adapter/`: MiniWorld sampler / loader -> FoldJet dense `TrainBatch` (384 tokens, 4096 atoms, MSA pool 8192, 4 templates).
  Per-sample randomness is a pure function of (FT_SEED, rank, stream position), so every backend sees the same stream and a resume
  continues it (`FT_START_STEP`). See `miniworld_adapter/README.md` (knobs `FT_*`) and `README_FAST.md` (the bit-identical conformer cache
  and the parallel RDKit embeds of large CCD components: single chlorophyll-like samples took ~100 s and stalled the in-order DataLoader).
  - Data (default `data_20261014_local_ft.yaml`): MiniWorld v2.4.0's data, `BioMolDB_20261014` on node-local NVMe
    (`/local_data1/psk6950/BioMolDB_20261014`, on node01 and node02), sampler antibody_protein 10 / protein_rna 20, its catalog COPY
    `..._v240_local_ft.arrow` (never point a config at the original `v240_local.arrow`: a fingerprint change makes MiniWorld rebuild and
    replace it). `overlay_pdb_only.yaml` sets the non-PDB sources to weight 0 and the v1.2 data policy (Ab-Ag interface crop, PDB MSA af3 8192).
  - `FT_CONF_CACHE`: the conformer cache directory (a missing cache is only slower, never different).
- `run/`: `env.sh` (python of the FoldJet pixi env, this checkout, MiniWorld source; `FT_EXTRA_SITE` for packages MiniWorld needs that
  the FoldJet env lacks, e.g. pyarrow), `common.sh` (`train RUN_DIR EPOCHS`: 2 ranks, effective batch 256, epoch = 25600 samples = 100 steps,
  `last.miniworld.pt` every epoch + `epoch<E>` every 5, resume from the newer), `train_py.sbatch` (pytorch: compile + step graph) and
  `train_engine.sbatch` (miniworld engine: step graph; compile measured, no gain). Submit a chain with `--dependency=afterany`.
- `tools/`: rebuild / extend the conformer cache (`conf_harvest.py`, `conf_build.py`) and the bit-identity check of an adapter change
  (`dump_batches.py` + `compare_batches.py`, `positions.py`).

Measured (H100, 2 ranks, 2026-11-15): engine ~84 s and pytorch ~272 s per optimizer step (epoch 2.3 h / 7.8 h); loader wait per next()
median 0.2 ms.
