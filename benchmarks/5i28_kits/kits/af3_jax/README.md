# AF3 (JAX) kit — 5I28 benchmark driver

Spec: `docs/guides/BENCHMARK-5I28.md`. Kit install: `$KITS_DIR/af3_jax` (only read).

## Run one size

On a GPU compute node with one GPU in `CUDA_VISIBLE_DEVICES`:

    bash $BENCH_DIR/af3_jax/run.sh N     # N = 1 2 4 8 12 16

The full log goes to `logs/x<N>.log` and the outputs to `out/x<N>/`. The last stdout line is
`KIT af3_jax n=<N> exit=<rc> fwd=[...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>`.

## Files

| file | what it does |
|---|---|
| `mkinputs.py` | Builds `inputs/x<N>/5i28.json` for both AF3 kits from FoldJet's own resolution (`foldjet.data.inputs.build.load` on `kitv2real/ff_x1.yaml`, `Template.af3()`): the real MSA (P0113359, 6789 a3m rows, paired == unpaired as in FoldJet's adapter) and the 4 distinct templates 5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A. It asserts both. Run it through srun with `~/foldjet/.pixi/envs/default/bin/python` (command in the header). The seeds are 6/6/6/5/4/4 for n = 1/2/4/8/12/16. |
| `run.sh N` | On first use it copies the kit's prebuilt compile cache (`jitroot`, about 60 MB) to `jitroot/`, under flock. It then runs `inner.sh` in the kit's `cu130run.sh` container and prints the summary line. |
| `inner.sh N` | Runs in the container. `featprobe.py` (CPU) runs first, then the kit's `run.sh pred --config h100 --variant p2 --mode fast --num_recycles 3 --num_diffusion_samples 5`, the same route as the earlier `af3_jax_realmsa` fast rows. `num_recycles 3` gives 4 trunk passes, and the kit uses 200 steps. TMPDIR, XDG and the caches stay under this directory. |
| `featprobe.py` | Runs the kit's own `alphafold3` featuriser (`af3_jax/venv`, `resolve_msa_overlaps=True` like the kit run) on the same JSON and prints `FEATPROBE ... msa_rows=... tmpl_real_min=... tmpl_distinct_min=...`. |
| `cpu_featprobe.sh` | The CPU-only check of both kits' featurisers. Its result for every n is in `logs/cpu_featprobe.log`: msa_rows=6790, 4 real, 4 distinct. |

## Where the numbers come from

- `fwd`: the kit lines `Running model inference with seed S took X seconds`, in log order, so the cold forward comes first. Each one is the jitted model call (trunk + 5-sample sampler + confidence) plus the device-to-host copy of its outputs. That span is the one named in `docs/records/anthropic_kits_fwd_definition_audit_20261029.md`. `warm_median` is the median of `fwd[1:]`.
- `templates_real`: the minimum of two counts. One is the smallest `real=K/4` over the kit's `[af3-jax-opt] TEMPLATES chain=X declared=4 ...` lines. The other is `tmpl_distinct_min` from FEATPROBE, which counts slots with distinct coordinates. It is 0 when either is missing.
- `msa_rows`: FEATPROBE `msa_rows`, the featurised MSA rows that have any unmasked position. The kit's run log has no MSA-row line.
