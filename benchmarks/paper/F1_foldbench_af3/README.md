# F1: AlphaFold 3 on the whole FoldBench, GPU-hours: Anthropic kit vs FoldJet bf16 / FP8

**What is measured.** End-to-end wall time of AlphaFold 3 (DeepMind weights) over all 1,522 FoldBench targets on H100s, from the input files to the
written structures: 5 trunk seeds (42, 66, 101, 2024, 8888) x 5 samples per target, 200 diffusion steps, 11 trunk passes (the kit's 10 recycles),
MSA up to 16,384 rows and templates as FoldBench gives them. Three arms:

| arm | what |
|---|---|
| `kit` | Anthropic's `af3_torch` kit, mode `fast`; a target that fails in fast mode (OOM) is run again in the kit's own `big` mode, nothing else is retried |
| `ff_bf16` | FoldJet AF3 on the engine, bf16, compile + CUDA graphs (`scope: trunk`) |
| `ff_fp8` | the same with the engine's FP8 path (`MINIWORLD_FP8=safe`, `MINIWORLD_FP8_MODEL=af3`, lazy guards) |

**GPU-hours** = the sum of the per-shard wall times. The shards are the kit's own split (`shards/s0.txt`, `shards/s1.txt`, 761 targets each,
the split of every earlier kit run); for the kit that is one kit process per GPU. For a FoldJet run in multi mode it is the number of GPUs x the wall time.
Failures and out-of-memory are results: they are listed in `f1_per_target.csv` (`status`), never dropped, and never drawn.

## Launch mode: "one process per shard like the kit" vs `fb_multi` (2 GPUs, MPS)

**Status (2026-11-22): default applied, a shard-mode re-run is not planned.** The paper table uses the numbers that were measured, in multi mode (GPU-h = 2 x wall), and states the mode next to every number; the kit arm is one process per GPU. A shard-mode re-run of the FoldJet arm would cost about 21 GPU-h and is done only if the authors decide so (the 2.5.12 shard-mode numbers below show the same direction).

Both are implemented in `foldjet/run_ff.sbatch`; **the default is the kit-like one** (`LAUNCH=shard`). Every run records the mode it used in `<run>/launch.txt`,
in `stamp.json` and in every row of `f1_shard_wall.csv` / `f1_summary.csv`; **the paper text must say which one a number comes from.**

| `LAUNCH=` | what runs | GPU-h |
|---|---|---|
| `shard` (default) | two independent `fb_multi.py` processes in one 2-GPU job, one per shard and GPU (`CUDA_VISIBLE_DEVICES=k`), the kit's shard lists, one worker per GPU (no MPS, no small-target concurrency; the scheduler still sorts the shard by size bucket and keeps its CUDA graphs warm per bucket) | wall(s0) + wall(s1) |
| `multi` | one `fb_multi.py` over both GPUs (one scheduler for all, small targets 3 at a time under CUDA MPS) | 2 x wall |

The earlier measurements: kit 14.5 GPU-h (fast 12.95 + big 1.51), FoldJet engine 3.0.1 in multi mode 9.81 (bf16) / 7.89 (FP8) GPU-h, i.e. 1.47x / 1.83x; those are multi-mode numbers.
The shard-mode run of the earlier engine (2.5.12, `fb_run_group.py`, one process per shard, no graphs after an OOM) took 5.27 / 5.31 h per shard against the kit's 6.36 / 6.27 h.

## Files

| path | what |
|---|---|
| `shards/s0.txt`, `s1.txt` | the kit's shard split (761 + 761 = the 1,522 targets of `benchmarks/foldbench/foldbench_targets.txt`) |
| `foldjet/run_ff.sbatch` | the FoldJet arm (bf16 / fp8, shard / multi); also the runner of F8 |
| `kit/kit2pred.py` | lays the kit outputs out like FoldJet's (`<t>/<t>_seed{s}_sample{k}_pred.cif` + `ranking.csv`) for the F2 evaluation |
| `kit/build_inputs.py` | FoldBench spec -> AlphaFold 3 JSON (FoldJet's own `af3()` export: MSAs and templates inside), one CPU job |
| `kit/run_kit.sbatch`, `kit/kit_shard.sh` | the kit arm: one kit process per GPU over its shard; fast, then big for the targets without a model |
| `collect.py` | run dirs -> `f1_shard_wall.csv`, `f1_summary.csv`, `f1_per_target.csv`, `f1_record.json` |
| `smoke/targets.txt` | two small targets (8chm, 8chl; both in shard 0) for the smoke runs |
| `results/` | the final CSVs of the campaign (copied from `$PAPER_OUT/F1/h100/results` after `collect.py`) |

Run outputs go to `$PAPER_OUT/F1/h100/<run name>/` (default `~/foldforge-scratch/paper/F1/h100/`), never into the repository. Every run dir has `stamp.json`
(FoldJet commit, engine version and kernels wheels, GPU names and UUIDs, node, Slurm job id, FP8 switches), `launch.txt`, `job_info.txt` (start / end stamps), the
Slurm log, and per target `run.json` (`gpu_s`, `peak_alloc_gib`, per-stage times; FP8: `fp8: {checks, saturated, redo, unresolved}`).

## Conditions that must hold for a campaign

* One engine release and one FoldJet commit for all arms (the FoldJet pin of the experiment commit; `stamp.json` of every run shows both). The kit has no engine.
* H100 (`--qos=cssb_h100`: `normal_h100` jobs are pre-empted by `cssb_h100` jobs), `-c 32 --mem=120G` for the 2-GPU job (Slurm caps: <= 128 GB, <= 32 CPUs per job).
* **A full run is ONE continuous job**, like the baseline: never split into time chunks, never a different job shape. If a submit is refused, stop and ask.
  The run is resumable on a resubmit (done markers), but a resubmit makes the wall time invalid; use it only for recovery and say so.
* Kit: not modified, except the `af3_torch` featuriser's one line (`ref_max_modified_date`, as upstream `run_alphafold.py`; original kept as `featurise.py.orig_kit`) and the
  AF3 weight layout of the copy (`xfold.of3.OF3 = False`). Kit-native modes only: `fast`, `big` only for what fast cannot do. No other retries.
* FP8: the worker calls `fp8.check()` after every trunk-seed group and redoes a saturated group (at most 3 times; the guard contract of `miniworld_engine/fp8.py`); the counts
  (`saturated`, `redo`, `unresolved`) are in each target's `run.json` and in `f1_per_target.csv`. (The 2026-11 whole-FoldBench FP8 runs did not call `check()`: see F11.)
* Warm caches: the default compile / extension caches (`CACHE_ROOT`, `~/.cache/foldforge/cache/foldbench`); do not point a campaign at a fresh cache directory.

## Full-campaign commands (run by the coordinator after the engine release; replace the paths)

```bash
cd ~/foldforge-scratch/paper-foldbench                       # the worktree at the experiment commit
export FOLDFORGE_REPO=$PWD FFENV=<FoldJet env with the released engine> PAPER_OUT=~/foldforge-scratch/paper
mkdir -p $PAPER_OUT/F1/h100/logs
P=benchmarks/paper/F1_foldbench_af3

# 0) inputs (once): FoldJet specs of the 1,522 targets (benchmarks/foldbench/build_specs.py -> $FB_SPECS, see benchmarks/foldbench/README.md)
#    and the kit's AlphaFold 3 JSONs (CPU job, ~minutes):
sbatch -A cssb --qos=cssb_h100 -p h100 --gres=none -c 4 --mem=48G --time=04:00:00 -J paper-F1-kitinputs -o $PAPER_OUT/F1/h100/logs/%x_%j.log --wrap \
  "bash benchmarks/foldbench/run_py.sh $P/kit/build_inputs.py \$FB_SPECS \$FF_SCRATCH/foldbench/af3full/inputs_all 42,66,101,2024,8888 \$(cat benchmarks/foldbench/foldbench_targets.txt | tr '\n' ' ')"

# 1) kit arm (2 H100, one kit process per shard; ~14.5 GPU-h: ~7.2 h wall)
sbatch -J paper-F1-kit -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3kit $P/kit/run_kit.sbatch

# 2) FoldJet bf16 and FP8, kit-like launch (default)  (2 H100 each; expect ~5.3 h wall per shard, i.e. ~10.6 GPU-h, FP8 less)
sbatch -J paper-F1-ff-bf16-shard -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3_bf16_shard,PRECISION=bf16,LAUNCH=shard $P/foldjet/run_ff.sbatch
sbatch -J paper-F1-ff-fp8-shard  -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3_fp8_shard,PRECISION=fp8,LAUNCH=shard  $P/foldjet/run_ff.sbatch
# 2b) the same two with fb_multi (only if the user chooses it; ~4.9 h / 3.9 h wall on 2 GPUs = 9.8 / 7.9 GPU-h with engine 3.0.1)
sbatch -J paper-F1-ff-bf16-multi -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3_bf16_multi,PRECISION=bf16,LAUNCH=multi $P/foldjet/run_ff.sbatch
sbatch -J paper-F1-ff-fp8-multi  -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3_fp8_multi,PRECISION=fp8,LAUNCH=multi  $P/foldjet/run_ff.sbatch

# 3) collect (CPU; FoldJet env python): the CSVs for the paper scripts
sbatch -A cssb --qos=cssb_h100 -p h100 --gres=none -c 4 --mem=16G --time=00:20:00 -J paper-F1-collect -o $PAPER_OUT/F1/h100/logs/%x_%j.log --wrap \
  "$FFENV/bin/python $P/collect.py --kit $PAPER_OUT/F1/h100/af3kit --ff ff_bf16=$PAPER_OUT/F1/h100/af3_bf16_shard --ff ff_fp8=$PAPER_OUT/F1/h100/af3_fp8_shard --out $PAPER_OUT/F1/h100/results"
cp $PAPER_OUT/F1/h100/results/f1_*.csv $P/results/
```

Expected cost (earlier records): kit 6.36 / 6.27 h per shard in fast mode plus 0.64 / 0.87 h for the 16 big-mode targets (14.5 GPU-h); FoldJet shard mode ~5.3 h per
shard (10.6 GPU-h, engine 2.5.12, `docs/records/foldbench_e2e_20261117.md`); multi mode 4.9 h bf16 / 3.9 h FP8 on 2 GPUs (9.8 / 7.9 GPU-h, engine 3.0.1). A campaign with the kit and
both FoldJet arms is ~7 + 5.5 + 4.5 h of 2-GPU time; the arms can run one after the other on the same node pair (they must not share GPUs with other jobs: the wall time is the result;
record `squeue -w <node>` of the neighbours in the notes if the node is shared).

## Output format (`results/`)

* `f1_shard_wall.csv`: `arm, launch, shard, targets, done, failed, wall_s, wall_h, fast_wall_h, big_wall_h, n_big, node, slurm_job_id, foldforge_commit, engine_version, gpu, precision, gpus`
* `f1_summary.csv`: `arm, launch, gpu_h, slower_shard_wall_h, targets, done, failed, ..., speedup_gpu_h, speedup_slower_shard` (vs the kit)
* `f1_per_target.csv`: `arm, target, shard, status, mode, tokens, gpu_s, peak_alloc_gib, fp8_saturated, fp8_redo, ...` (kit `gpu_s` = sum of `forward_s` over the seeds; FoldJet `gpu_s` = `run.json`)
* `f1_record.json`: the same in the layout of the paper's `data/foldbench_full_af3.json`

## Smoke test (done, engine 3.0.6 candidate, see the report of the task)

`sbatch --gres=gpu:h100:1 --time=00:10:00 -c 16 --mem=64G --export=ALL,NAME=smoke_bf16_shard,PRECISION=bf16,LAUNCH=shard,SHARDS=0,TARGETS=$P/smoke/targets.txt,EXTRA="--seeds 42",ENGINE_OVERLAY=<dir of the extracted engine wheels> $P/foldjet/run_ff.sbatch`
(also `LAUNCH=multi`, `PRECISION=fp8`; the kit with `kit/run_kit.sbatch` and `SHARDS=0 SHARD_LIST_S0=$P/smoke/targets.txt KIT_INPUTS=<dir with 1-seed JSONs>`).

## Decision (user, 2026-11-22)

FoldJet's F1 row is the `fb_multi` run (2 GPUs in one job, MPS concurrency for small targets; GPU-hours = 2 x wall time), reported with that protocol
stated. No per-shard re-run.
