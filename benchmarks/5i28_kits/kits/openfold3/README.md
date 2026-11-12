# 5I28 benchmark: OpenFold3 Anthropic kit

Spec: `docs/guides/BENCHMARK-5I28.md`. The kit is `$KITS_DIR/openfold3_ob0`, used read-only through
`run.sh pred --config h100 --mode fast`. It runs in the kits' container (`cu130kit.sh`, a copy of the kits' own wrapper with its
singularity cache/tmp moved here).

## Run

```
CUDA_VISIBLE_DEVICES=<gpu> bash run.sh N          # N = 1 2 4 8 12 16; MODE=fast|exact|off (default fast); COPIES (default 4 at n=1, else 3)
```

One process runs COPIES identical queries, so it does COPIES forwards and the first is cold. The full log goes to `logs/x<N>.log`.
Stdout is a single line:
`KIT openfold3 n=<N> exit=<rc> fwd=[...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>`. For a mode other than fast, the kit
name is `openfold3:<mode>`.

- `fwd` is the kit's `PHASE item=.. total_s=` line, in item order. It covers trunk + sampler + confidence and is CUDA-synchronised,
  the same span as `Model forward time` (see the audit). `warm_median` is the median of every forward except the first.
- `templates_real` is the smallest `real_slots` over all of the kit's `TEMPLATES FEATURISED query=.. real_slots=k/T` lines. If an
  item has no such line, it is 0. The log also gets `BENCH` lines with the per-chain `TEMPLATES chain=.. real=k/N` census, any drops,
  and upstream's template log.
- `msa_rows` is R in the kit's feature-shape line `msa[1, 1, R, ...]`. That line is printed in fast and exact mode only; in off mode
  the field is `?`.

## Inputs (`inputs/`, built by `make_inputs.sh`; checks in `check/`)

- **MSA:** `msa/A/colabfold_main.a3m` is the real 5I28 MSA (`$FF_SCRATCH/prec_vs_kit/real/5i28_unpaired.a3m`, LMDB key
  P0113359, 6789 a3m rows including the query). The model sees 6790 rows (OF3 adds one).
- **Templates:** CIF-direct mode (`template_cif_paths` + `template_cif_chain_ids`), 4 distinct entries: 5syd_A, 6kol_A, 5yt7_A,
  6tfd_A, gunzipped from `~/data/cif_raw`. OF3 keys slots `<entry>_<chain>`, so the 4 entries give 4 distinct slots.
- **6tfd is trimmed** (`check/trim_6tfd.py`; the original is `src/6tfd_full.cif`) to chain A, label_seq 26-134, renumbered from 1.
  The full chain cannot become a real slot, for two reasons:
  - OF3's kalign score is 0.085, below `cif_direct_min_score` 0.1.
  - Its alignment pairs 15 residues that have no coordinates (label 1-25), and stock OF3 drops a slot when any aligned template
    residue is unresolved.

  26-134 is the resolved part of FoldForge's LMDB hit span (1-134). The trimmed chain scores 0.113.
- **CPU check** (`check/of3_check.sh`, srun without a GPU; output in `check/of3_check_slot.out`): upstream's TemplatePreprocessor
  gives all 4 ids for x1 and x16. Each one passes the featurisation's real-slot condition:
  - 5yt7_A: score 0.786, 114 aligned pairs
  - 5syd_A: score 0.462, 88 aligned pairs
  - 6kol_A: score 0.214, 100 aligned pairs
  - 6tfd_A: score 0.113, 74 aligned pairs

  OF3 realigns with kalign. For 5yt7 and 5syd that matches FoldForge's LMDB alignment exactly; for 6kol and 6tfd it does not.
- **Sampling:** OF3 defaults: num_recycles 3 + 1 = 4 trunk passes, 200 steps, `--num-diffusion-samples 5`. Query seeds are `[0]`, but
  earlier kit logs print `seed=42`.
- **Caches:** `cache/` is a copy of the kit's warm `cache/openfold3_ob0_gpu130`, with the paths in Triton's group files rewritten.
  TMPDIR, the template cache and the outputs are all under this directory (`tmp/`, `runs/x<N>/`).
