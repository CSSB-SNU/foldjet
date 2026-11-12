# AF3 (Torch) kit — 5I28 benchmark driver

Spec: `docs/guides/BENCHMARK-5I28.md`. Kit install: `$KITS_DIR/af3_torch` (only read).

## Run one size

On a GPU compute node with one GPU in `CUDA_VISIBLE_DEVICES`:

    bash $BENCH_DIR/af3_torch/run.sh N   # N = 1 2 4 8 12 16

The full log goes to `logs/x<N>.log` and the outputs to `out/x<N>/out/`. The last stdout line is
`KIT af3_torch n=<N> exit=<rc> fwd=[...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>`.

## Files

| file | what it does |
|---|---|
| `inputs/x<N>/5i28.json` | Written by `../af3_jax/mkinputs.py` together with the JAX inputs: the same protein entry, with the real MSA and the 4 distinct templates 5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A. The seeds are 4/4/4/3/3/3 for n = 1/2/4/8/12/16. |
| `run.sh N` | On first use it copies the kit's prebuilt `jit/` (inductor and triton, about 1 GB) and its `cache/torch` here, under flock. It then runs `inner.sh` in the kit's `cu130run.sh` container (with `CU130_EXTRA_LD` as in the kit's `job.sh`) and prints the summary line. |
| `inner.sh N` | Runs in the container. `featprobe.py` (CPU) runs first, then the kit's `run.sh pred --config h100 --mode fast --norun_data_pipeline --num_recycles 3 --diffusion_steps 200 --num_diffusion_samples 5`, the same route as the earlier `af3_torch_realmsa` fast rows. `num_recycles 3` gives 4 trunk passes. TMPDIR, XDG and the JIT root stay under this directory. |
| `featprobe.py` | Runs the kit's featurisation env (`alphafold3_venv`, `resolve_msa_overlaps=False` like `af3_torch_opt/featurise.py`) on the same JSON and prints `FEATPROBE ... msa_rows=... tmpl_real_min=... tmpl_distinct_min=...`. The CPU check for all n is in `../af3_jax/logs/cpu_featprobe.log`. |

## Where the numbers come from

- `fwd`: the kit's `[af3-torch-opt] ITEM ... forward_s=X` lines, one per seed, in log order, so the cold forward comes first. Each one is synchronize, then `A.inference()` (trunk + 5-sample sampler + confidence + distogram), then synchronize. That is the same span as FoldForge, per the fwd-definition audit. It equals that item's `PHASE total_s`. `warm_median` is the median of `fwd[1:]`. The kit captures its CUDA graphs again for every item.
- `templates_real`: the minimum of three counts. The first is the smallest `4*a/b` over ITEM `templates=a/b`, which counts slots x chains with b = 4n. The second is `tmpl_distinct_min` from FEATPROBE. The third, `4*evaluated/slots` from the `template_dedupe` LEVER line, applies only when that line shows identical slots. `tables=` on that line is NOT a distinct count: it is the embedder's constant index tables.
- `msa_rows`: FEATPROBE `msa_rows`. The kit's run log has no MSA-row line.
