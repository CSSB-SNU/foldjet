# ft_adapter: MiniWorld PDB data -> FoldJet `TrainBatch`

Real-data batch source for `~/foldjet/scripts/train_protenix1.py` (Protenix v1 fine-tuning). Lives outside both repositories; it imports both
(FoldJet for the featurisation, MiniWorld for the sampler / loader). Nothing in `~/foldjet`, `~/MiniWorld`, `~/foldjet-engine` is changed.

## Call

```bash
source ~/foldforge-scratch/ft_train/env.sh          # FoldJet pixi env python (no `pixi run`), PYTHONPATH = adapter + FoldJet/src + MiniWorld/src + pyarrow
cd ~/foldjet
python scripts/train_protenix1.py --batch-source ft_adapter:batches --backend pytorch --effective-batch 256 --ckpt ~/.cache/foldjet/checkpoints/protenix/protenix1.miniworld.pt ...
# 2 ranks: torchrun --nproc-per-node 2 scripts/train_protenix1.py --batch-source ft_adapter:batches ...
```

`--batch-source ft_adapter:batches` -> `batches(cfg, device)` returns an endless iterator of `TrainBatch` at ONE fixed shape (tokens 384, atoms 4096,
MSA pool `FT_MSA_POOL` rows, 4 templates, `bond_pairs` 128 rows, `present_index` 4096). `--bench-bucket` is not needed; passing it with
`--bucket-msa-pool 8192` (the default pool here) is a no-op.

## Knobs (environment variables, all optional)

| variable | default | meaning |
|---|---|---|
| `FT_DATA_CONFIG` | `~/MiniWorld/configs/miniworld/data/H100_default.yaml` | MiniWorld data config (the one whose paths exist on this cluster) |
| `FT_OVERLAY` | `adapter/overlay_pdb_only.yaml` | merged over the data config: PDB only + the v1.2 recipe crop / MSA policy (see below) |
| `FT_SEED` | `cfg.seed` (0) | base seed of the sampler streams and of every per-sample rng |
| `FT_WORKERS` | 14 | CPU worker processes per rank (spawned; 16 cores per rank -> 14 workers + the training process) |
| `FT_PREFETCH` | 4 | batches in flight per worker (14 x 4 = 56 examples ~ 0.7 GB shared memory) |
| `FT_START_STEP` | 0 | optimizer step the run (re)starts at (see resume policy) |
| `FT_MSA_POOL` | 8192 | raw MSA pool rows (MiniWorld recipe: PDB pool 8192 = `bucket_msa_multiple`; FoldJet's own limit is 16384) |
| `FT_TOKENS` / `FT_ATOMS` / `FT_TEMPLATES` / `FT_BOND_ROWS` | 384 / 4096 / 4 / 128 | bucket shape |
| `FT_EPOCH_SIZE` | 131072 | indices drawn per rank per sampler "epoch" (only chunks the stream) |
| `FT_MODE` | `fast` | FoldJet `DenseSpec` mode (the train script loads the model in `fast`) |
| `FT_LOG_EVERY` | 25 | an `FT_DATA ...` line every N samples (the first 8 always): worker seconds, data wait ms, time between `next()` calls |
| `FT_MAX_FAILURES` | 200 | (reserved) |

Rank and world size are read from `torch.distributed` (the train script initialises it before it calls the source), else `RANK` / `WORLD_SIZE`.

## What it does

1. **Dataset and weights are MiniWorld's own**: `BioMolData` (catalog = the Arrow cache named in the data config, 26,120,438 rows, cache hit, no rebuild),
   `WeightedSampler` (weights = catalog raw weights x source weights). The PDB-only override sets `source_weights` to `pdb 1.0, disordered_pdb 0,
   protein_monomer 0, short_protein_monomer 0` (zeros, not removal: MiniWorld gives every *declared* distillation source a default weight 1.0, and
   removing the sources would change the catalog fingerprint and trigger a rebuild of the shared 8.7 GB cache). `sampler:` (pair-type weights) and
   `antibody_seq_id_path` are untouched.
   The overlay also carries the recipe's data policy from `phase2a_diffusion_medium_v120.yaml` (inherited by `..._v200_b200_joint`):
   `crop.ab_ag_interface_only: true`, `crop.prefer_nonprotein_focus: true`, PDB MSA depth policy `af3` with an 8192-row pool (verified by composing the
   recipe with hydra in MiniWorld's env: `logs/hydra_dump.*.log`).
2. **Per rank**: `WeightedSampler(rank, world, seed, epoch)`; a second draw of 16 "spare" indices per position replaces a sample that fails.
3. **Workers** (torch DataLoader, spawn): `Preprocessor.process` with an rng seeded from `(seed, rank, position)` (MiniWorld's stock `__getitem__` uses an
   unseeded rng) -> crop / MSA / templates -> `convert.Converter`.
4. **Conversion** (`convert.py`) builds the dense features with FoldJet's / AF3's own featurisation code, not a re-implementation: the cropped CIFMol
   becomes an AF3 flat `AtomLayout`; AF3 `tokenizer`, `AtomCrossAtt`, `TokenFeatures`, `PredictedStructureInfo`, `RefStructure` (CCD / RDKit conformers),
   `PolymerLigandBondInfo`, `LigandLigandBondInfo`, `PseudoBetaInfo`, `Frames` run on it, then FoldJet `dense_conventions.apply` (protenix1 spec),
   `bond_orders.capture`, `apply_after_bucketing`. MiniWorld contributes the crop, the sampled MSA, the sampled templates and the ground truth.
5. Main process: CPU tensors -> GPU, `TrainBatch(features, atom_positions, atom_mask (resolved), bond_pairs, present_index, present_valid, name)`.

## Seeds, resume, ordering

* Sampler stream of rank r at position p: `WeightedSampler(seed=FT_SEED, epoch=p // FT_EPOCH_SIZE, rank=r)[p % FT_EPOCH_SIZE]` (with replacement, as MiniWorld).
  Per-sample randomness (crop, MSA depth / rows, template choice, conformer seed, reference-pose seed) is a pure function of `(FT_SEED, rank, p)`.
* The train script consumes exactly `grad_accum` samples per optimizer step per rank. **Restart at step S: set `FT_START_STEP=S`**; the stream then continues
  at position `S * grad_accum` of every rank: no sample from before the restart is repeated (a position is never reused) and the continued stream equals
  the uninterrupted one. Samples prefetched but not consumed before the kill are discarded. Without `FT_START_STEP` a restart replays the stream from 0
  (same data as the first run: change `FT_SEED` for a fresh stream instead).
* Different ranks draw different, independent streams (`seed + epoch * world + rank` inside MiniWorld's sampler).
* Sampling is with replacement from 9.4 M weighted rows, so two positions may hit the same row (as in MiniWorld); crops differ.
* A sample that cannot be expressed in the dense layout (rare, ~1 %, counted in `FT_SKIP` lines: tokenisation / alphabet mismatch of unknown or
  modified residues, AF3 token count above 384) or whose records fail to load is replaced by the next spare weighted index. MiniWorld's own retry draws
  uniformly from the whole catalog; here it stays inside the sampler distribution (the stock rule would pull distillation items into a PDB-only run).

## Differences from MiniWorld's own features (deliberate; FoldJet's conventions win because the Protenix weights were trained with them)

* Tokenisation is AF3's: MSE / UNK / unknown nucleotides are one token (MiniWorld's atom tokenizer makes one token per atom for them); MSA and templates are
  taken at residue level and expanded by AF3's tokens (`tok_res`).
* Reference conformers: AF3 `RefStructure` (CCD ideal / RDKit, seeded per sample) + FoldJet conventions, not MiniWorld's `to_reference_features`.
* MSA alphabet translated MiniWorld -> AF3 (`MW_TO_AF3`): gap 31 -> 21, RNA A U G C N 21..25 -> 22 25 23 24 30, DNA A T G C N 26..30 -> 26 29 27 28 30; profile
  columns re-indexed 32 -> 31; deletion counts recovered from MiniWorld's `2 arctan(d/3)/pi` encoding; ligand columns all gap then the Protenix
  convention (`nonprotein_msa_as_query`, `dedupe_self_msa`) as in FoldJet inference.
* Templates: MiniWorld keeps N, CA, C and one CB (CA for glycine) per residue: written to AF3's per-residue-type dense slots (N 0, CA 1, C 2, CB 4); O and side
  chains stay masked. Those are the only atoms the template embedder reads (checked with FoldJet's own `pseudo_beta_fn`).
* Ground truth is centred on the mean of the resolved atoms (the train step rotates about the origin; MiniWorld centres too).
* The loader's crop has no OXT / OP3 (MiniWorld removes them before cropping); FoldJet `fast`-mode inference keeps them. The model is trained without them.

## Files

`ft_adapter.py` (source), `convert.py` (conversion), `mwsample.py` (MiniWorld dataset / per-item capture), `mwconfig.py` + `overlay_pdb_only.yaml` (config),
`_mwshim.py` (import shims), `pdbtypes.py` (pair type of every catalog row). Parent directory: `env.sh`, test and benchmark scripts (`weight_check.py`,
`equiv.py`, `bench_loader.py`, `test_convert.py`, `smoke.sh`) and `logs/`.

## Import shims (no install anywhere)

MiniWorld's data package imports `foldjet_engine.utils.typecheck` (engine 2.4) and, through `miniworld.training/__init__`, the whole old trainer and an
old `foldjet.blocks`. `_mwshim.py` registers an identity `typecheck` module (identity in MiniWorld's engine unless `settings.typecheck` is on) and a
bare `miniworld.training` package (only `miniworld.training.batch` is needed). `pyarrow` (absent from the FoldJet env) is imported from a symlink to the
copy in MiniWorld's own env (`extra_site/pyarrow`, appended last to `sys.path`).
