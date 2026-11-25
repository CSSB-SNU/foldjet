# FoldBench end to end

The walltime of the whole FoldBench set (1,522 targets) through FoldJet AF3 on the engine, from the spec files to the written CIFs, in one
Slurm job on 1..N GPUs: the number to compare with af3kit (the Anthropic AF3 kit) on the same set. Science per target: DeepMind AF3 weights,
bf16, `compile` + CUDA graphs (`scope: trunk`), 5 trunk_seeds (42, 66, 101, 2024, 8888) x 5 samples, 200 steps, `--recycles 11`
(11 trunk passes; the kit's 10 recycles), MSA up to 16,384 rows, templates as FoldBench gives them. Accuracy is scored separately from the CIFs.

Result of record: [docs/records/foldbench_e2e_20261117.md](../../docs/records/foldbench_e2e_20261117.md) — FoldJet on engine 2.5.12
**5.27 / 5.31 h** per shard vs af3kit **6.36 / 6.27 h** (2 H100s, one shard per GPU), 1,522 / 1,522 targets done vs the kit's 1,506
(16 out-of-memory failures on the largest): 1.20x end to end. That run used the earlier one-process-per-shard runner; `fb_multi.py` here
replaces it (bucket sort, one scheduler for all GPUs, no OOM-with-graphs tail, small-target concurrency).

## Files

| file | what |
|---|---|
| `build_specs.py` | FoldBench `inputs/<t>/data.yaml` -> FoldJet spec `<t>.yaml` + `<t>.chains.json` (FoldBench chain ids for the output CIFs) |
| `fb_multi.py` | the parent: list, bucket sort, cost estimate, scheduler, worker processes, MPS, summary (no CUDA) |
| `fb_multi_worker.py` | one worker: loads the model once, prefetches inputs / featurisations of its assigned targets, runs them, writes the outputs |
| `fb_sched.py` | the scheduler and cost model (pure logic; also `fb_multi.py --simulate`) |
| `fast_sort.py` | size + bucket of every target (`BucketShape.select(tokens, atoms, 1)`); `fast_sort.py build CCD_LMDB OUT_TSV` makes the CCD table |
| `fb_data.py`, `fb_io.py` | featurisation of one (target, trunk_seed); CIF chain renaming + writing |
| `foldbench_targets.txt` | the 1,522 targets |
| `foldbench_af3_sizes.tsv` | AF3 tokens / atoms of every target (FoldJet featurisation, computed once) |
| `ccd_heavy_atoms.tsv` | CCD code -> heavy atoms, to count a target missing from the size table from its spec files |
| `foldbench_cost_s.tsv` | measured seconds per target (1,492 targets, the 2026-11-16/17 runs): the cost estimate of the scheduler |
| `run_py.sh` | the env's python directly (`FFENV`, never `pixi run`), this checkout's `src`, caches under `CACHE_ROOT` |
| `run_multi.sbatch` | the full run, one job (example: edit the cluster lines) |

## Build the specs (once per machine)

```bash
# FOLDBENCH_ROOT: the FoldBench release (inputs/, meta/chain_seqid_map.tsv, ground_truth/)
# FOLDBENCH_RESOURCES: BioMolDB_20261014/lmdb (foldbench/msa/a3m.lmdb, foldbench/msa/a3m_rna.lmdb, foldbench/template/template.lmdb)
benchmarks/foldbench/run_py.sh build_specs.py benchmarks/foldbench/foldbench_targets.txt <spec dir> \
    --foldbench $FOLDBENCH_ROOT --resources $FOLDBENCH_RESOURCES      # --ccd-db: default foldjet.paths.ccd_db()
```

CPU only, ~10 s for the 1,522 targets. The specs hold absolute paths (FASTA, databases). Per spec: `n_trunk_samples 1` (the worker runs its own
trunk_seeds), 5 samples, `template_dedup: false`; MSAs / templates the databases do not hold, DNA MSAs, non-protein templates and templates with a
non-zero `template_offset` are dropped and logged (docstring of `build_specs.py`). Glycans: unbonded multi-CCD ligands (the FoldBench protocol;
`--glycans entity` only for comparison).

## Run

```bash
sbatch --export=ALL,SPECS=<spec dir>,OUT=<run dir>,FFENV=<FoldJet pixi env> benchmarks/foldbench/run_multi.sbatch
# directly, on the GPUs of an allocation (CUDA_VISIBLE_DEVICES, or --gpus 0,1):
benchmarks/foldbench/run_py.sh fb_multi.py af3 benchmarks/foldbench/foldbench_targets.txt <spec dir> <run dir> \
    --recycles 11 --group-max-tokens 512 --small-concurrency 3
# the schedule with the estimated costs, CPU only:
benchmarks/foldbench/run_py.sh fb_multi.py af3 benchmarks/foldbench/foldbench_targets.txt <spec dir> <run dir> --gpus 0,1 --simulate
```

One job, any number of GPUs (`--gres=gpu:h100:N`; the example uses 2 with `-c 32 --mem=120G`). A walltime run is one continuous job: do
not split it into time chunks. Resumable: a target with `<run dir>/<t>/done` (or `failed/marker`) is skipped on a resubmit. Arguments that
`fb_multi.py` does not know go to the workers (`--recycles`, `--group-max-tokens`, `--seeds`, `--nd`, `--steps`, `--scope`, `--eager`, ...;
see the top of `fb_multi_worker.py`). `PYTORCH_CUDA_ALLOC_CONF`: FoldJet's default `expandable_segments:True` unless you set it
([CONFIG.md](../../docs/guides/CONFIG.md)). `FF_ALLOW_SIZE_MISMATCH=1` skips the checkpoint blob size check in the workers (a local blob of
another converter version; off by default).

## Scheduler

- **Bucket sort.** Every target gets its exact FoldJet size bucket (token bucket, atom bucket) from `foldbench_af3_sizes.tsv` (targets not
  listed are counted from their spec files with `ccd_heavy_atoms.tsv`, < 1 s). Targets of one bucket share their compiled graphs.
- **Cost.** Measured seconds from `foldbench_cost_s.tsv` (and `--calib-runs <earlier run dirs>`, whose `run.json` `total_s` override it);
  others `a * tokens^b` fitted on the measured targets >= 256 tokens.
- **Assignment** (`fb_sched.Scheduler`): a worker keeps taking its bucket (affinity: its CUDA graphs stay warm); when the bucket is empty it
  claims the unclaimed bucket with the largest remaining cost (`--order total`; `largest` = largest single target): big work first, the
  tail is small targets; with no unclaimed bucket left it shares (steals) the bucket with the largest remaining cost. Each worker holds its
  running target + `--lookahead` (2) assigned ones whose inputs load and trunk_seed featurisations start on assignment.
- **Revoke.** A worker that runs dry while another still holds a queued (not started) target gets that target revoked to it.
- **Failures.** Out of memory with CUDA graphs: the target is retried without graphs and that worker runs the rest of the bucket without
  graphs (per bucket, not for the rest of the run). Out of memory of a trunk_seed group halves the group cap of that residue bucket.
  A target failing `--max-fails` (2) times gets `failed/marker` and a `failures.tsv` line. A dead worker (exit, signal, `--stall` s without
  a message) is killed with its process group, its targets requeued, and it is respawned (`--max-respawns` per GPU).
- **trunk_seed groups.** The 5 trunk_seeds of a target with <= `--group-max-tokens` (512) AF3 tokens are sampled together; larger ones
  one by one (grouping above ~512 tokens measured slower).
- **Small-target concurrency** (`--small-concurrency K`, default 1 = off; the run uses 3). The parent starts one CUDA MPS daemon for the job
  (node-local pipe dir, stopped at the end; `--mps off` = plain time slicing, measured half the gain). A GPU gets up to K workers while it
  runs small targets (token bucket <= `--small-max-tokens`, 512). The extra slots of a GPU start on demand: once its primary worker runs /
  holds only small targets, no unclaimed large bucket is left and >= `--small-min-left-s` (120) estimated seconds of small work per GPU are
  unassigned. Large buckets are claimed before small ones. When only large targets are left to assign, the slots finish what they hold and
  exit, and the primary runs the large target alone. Outputs are byte-identical to K = 1 (measured on 40 targets).

## Outputs (`<run dir>`)

- `<t>/`: `<t>_seed<s>_sample<k>_pred.cif` (25 per target, FoldBench chain ids), `ranking.csv` (`seed,sample,ranking_score`;
  0.8 ipTM + 0.2 pTM, or pTM, or mean pLDDT when the model returns no pTM), `run.json`, `done`.
- `run.json`: `model`, `chains`, `ranking` (the score basis), `timings_s` ([seed, trunk s, trunk + diffusion s] per trunk_seed), `total_s`
  (= `gpu_s` + `load_wait_s`), `gpu_s`, `load_wait_s`, `overlap`, `stages` (per trunk_seed: trunk, diffusion, group size, waits for the loader /
  the featurisation prefetch, in-process featurisation), `peak_alloc_gib`, `peak_reserved_gib`, `multi` (worker id, graphs on/off).
- `multi_events.log`: one line per scheduler event (`list`, `mps`, `slots`, `spawn`, `hello`, `ready`, `assign`, `steal`, `start`,
  `computed`, `done`, `fail`, `retry`, `failed`, `revoke?`, `revoked`, `retire`, `retired`, `dead`, `groupcap`, `graphs`, `bye`, `resolved`,
  `summary`, `gpu`), with the wall time since the start.
- `multi_summary.json`: `wall_s`, `targets`, `done`, `failed`, `finish_spread_s`, `per_gpu` (per worker: incarnations, targets done, model
  load, busy / idle / tail-idle seconds), `order`, `lookahead`, `alloc_conf`, `worker_args`, and with K > 1 `small_concurrency`,
  `small_max_tokens`, `mps`.
- `multi_targets.tsv`: per target worker, tokens, bucket, estimated / measured seconds, start / computed time, graphs, fails.
- `sort_order.tsv`, `failures.tsv`, `job_info.txt` (sbatch), `workers/w<k>_<incarnation>.log`.
