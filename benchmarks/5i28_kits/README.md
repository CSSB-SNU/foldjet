# 5I28 kit-side harness

The scripts that produced [`docs/records/5i28_anthropic_kits_20261111.md`](../../docs/records/5i28_anthropic_kits_20261111.md): the 5I28
protocol ([BENCHMARK-5I28.md](../../docs/guides/BENCHMARK-5I28.md)) run on the Anthropic kits, one H100 per job. Copied from the measurement directory of
2026-11-11 with its absolute paths replaced by environment variables; the copy was **not re-run** after the path replacement (only `bash -n`
on the shell scripts), so expect to fix a path or two on first use. No logs, weights or inputs larger than the Protenix-family JSONs in `in/`.

## Environment variables (export them; the scripts call each other through `bash`)

| variable | meaning |
|---|---|
| `BENCH_DIR` | working directory: per-kit `<kit>/{inputs,logs,out}`, `done_final/`, `done_sep/`, `probe/` are created under it; `<kit>/run.sh` etc. are expected at `$BENCH_DIR/<kit>/` (copy `kits/<kit>/` there) |
| `KITS_DIR` | the kit installs (`<kit>/venv*`, `<kit>/kit/...`, `cu130run.sh`, `cuda13home`, `libstdcxx`, `weights`) |
| `FF_SCRATCH` | scratch holding the cu130 CUDA toolkit pixi env (`cuda13tk`), the CUDA 13 compat libs (`cudacompat13`), `kalign_build`, the Singularity image (`sing/ubuntu2204.sif`) |
| `INPUT_SCRATCH`, `A3M_5I28` | the FoldForge spec / 20-hit template a3m the input builders read; the real 5I28 MSA as an a3m (the files are in the repository: [`benchmarks/5i28_data/`](../5i28_data/README.md); `A3M_5I28` = `benchmarks/5i28_data/5i28_unpaired.a3m`) |
| `PY` | a python with numpy, for the host probe (`queue3.sh`, `kit_multi_job.sh`) |

## Files

- `queue3.sh`: resumable N-GPU work queue over a queue file (`queues/*.txt`, lines `name|command`), one GPU per item, each item pinned to its own
  `PIN` CPUs, host probe / GPU clock log / thread census around every item, done marker per item. `kit_multi_job.sh`: one kit, all n in one process.
- `protenix_family_run.sh KIT N`: Protenix v1 / v2 / OpenDDE, one n, fast mode, seeds 0-3, prints one `KIT ...` line.
  `protenix_family_multi.sh KIT`: the same with all sizes in one process (evidence run only; see the record).
- `kits/<kit>/run.sh N` (and `run_multi.sh`): AF3-JAX, AF3-Torch, Boltz-2, Chai-1, ESMFold2, OpenFold3, RoseTTAFold3. The last stdout line of each is
  `KIT <model> n=<N> exit=<rc> fwd=[<s>,...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>`; `fwd` is every forward in log order (first = cold).
- `kits/*/mk*.py`, `prep.py`, `make_inputs.sh`, `in/`: input builders (real MSA, 4 distinct templates) and the Protenix-family inputs.
- `collect_kit_results.sh`: `BENCH_DIR=... collect_kit_results.sh OUT_PREFIX` turns the done markers into the record's CSV and raw JSON.
- `probe.py`, `node_cmp.sh`, `cu130run.sh`: host probe, node comparison, the cu130 container wrapper.

## Method (see the record)

Protenix v1 / v2 / OpenDDE: one process per n (`queues/nmajor_*.txt`); Boltz-2 and ESMFold2: one process, n ascending; the other kits one process per
cell. One single-GPU Slurm job per kit, `-c 16 --hint=nomultithread`, `PIN=16`. A cell counts only with `real=4/4` templates (RF3 `1/1`).
Run a queue with, for example:

    sbatch -J k_af3 -w node02 -c 16 --hint=nomultithread --gres=gpu:h100:1 --mem=240G \
      --export=ALL,NG=1,QFILE=$PWD/queues/af3_torch.txt,PIN=16,DONE=$BENCH_DIR/done_final queue3.sh

Do not edit a script a running queue executes.
