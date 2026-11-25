# F7: Protenix v1 fine-tuning loss, engine vs PyTorch (`torch.compile`)

Paper: Fig 2a, b (loss per optimizer step and per hour), Supplementary Fig 11 (every loss term). Card: H100, **2 GPUs per run**.

## What is run

The existing fine-tune of `scripts/finetune/` (see its README): Protenix v1 from `protenix1.miniworld.pt`, `--recipe protenix` (PAE loss on at 1e-4), MiniWorld v2.4.0 PDB data through the
adapter (crop 384 tokens / 4,096 atoms), precision `af3_default` (bf16 trunk, **fp32 diffusion head and fp32 pair-bias logits**), activation checkpointing, 48 augmentations in 2 chunks of 24,
2 ranks, **effective batch 256**, 1 epoch = 25,600 samples = 100 optimizer steps, `last.miniworld.pt` every epoch, `epoch<E>` every 5.
Two runs, **identical recipe, data, seed (`FT_SEED=0`) and sample order** (a pure function of seed, rank and stream position): `BACKEND=miniworld` (engine, step graph) and `BACKEND=pytorch`
(`torch.compile` + step graph). **Both go to the same `TOTAL_EPOCHS`; the PyTorch run is never stopped early** (the earlier PyTorch run stopped at step 432 while the engine run went to 1,424).

`run_ft.sbatch` is one continuous job per run (QOS `cssb_h100` allows 30 days; `--mem=128G` = the per-job cap, `-w node01`: the BioMolDB copy is node-local on node01 / node02).
`submit_chain.sh` is the fallback of the earlier shape (24 h segments, resume from the newer of last / epoch).

## Commands (full campaign; started by the coordinator after the release and the FoldJet pin; 2 + 2 H100)

```
cd <FoldJet checkout at the pinned commit>
BACKEND=miniworld TOTAL_EPOCHS=10 sbatch -J f7-engine  --export=ALL --time=3-00:00:00 benchmarks/paper/F7_finetune_loss/run_ft.sbatch
BACKEND=pytorch   TOTAL_EPOCHS=10 sbatch -J f7-pytorch --export=ALL --time=7-00:00:00 benchmarks/paper/F7_finetune_loss/run_ft.sbatch
# afterwards (any machine with python3, not a cluster login node):
python3 benchmarks/paper/F7_finetune_loss/collect.py --logs ~/foldforge-scratch/paper/F7_finetune_loss/h100 --out ~/foldforge-scratch/paper/F7_finetune_loss/h100/csv
```

`TOTAL_EPOCHS=10` = 1,000 optimizer steps (the 10-epoch model of F8). Expected wall times from the earlier runs (108 s and 369 s per optimizer step, same 2 H100):
engine ~30 h, **PyTorch ~102 h (4.3 days)** for 1,000 steps; at the earlier engine length of 1,424 steps (`TOTAL_EPOCHS` cannot give a fractional epoch: 14 epochs = 1,400 steps)
the PyTorch run takes ~144 h (6 days). Both runs can run at the same time (4 GPUs). Open: the step count (10 or 14 epochs) is the coordinator's decision.

Smoke (2 GPUs, 2 optimizer steps of effective batch 2): `RUN_ROOT=<dir> EFFECTIVE=2 EPOCH_SAMPLES=4 TOTAL_EPOCHS=1 ENGINE_SRC=<engine source> BACKEND=miniworld sbatch --time=00:10:00 -J f7-engine-smoke --export=ALL run_ft.sbatch`.

## Output

`~/foldforge-scratch/paper/F7_finetune_loss/h100/` (`runs/ft_engine`, `runs/ft_pytorch` with the checkpoints; Slurm logs `slurm_f7-engine_<job>.log`, `slurm_f7-pytorch_<job>.log`; `meta_*.txt`
with FoldJet commit, engine source, GPUs, node, job id). `collect.py` writes `engine_steps.txt` / `pytorch_steps.txt` (the shape of paper `data/finetune/*_steps.txt`, read by
`scripts/fig2_training_accuracy.py`), `finetune_steps.csv` (every step, every loss term, seconds and cumulative hours per run) and `finetune.json` (steps, median seconds per step, hours,
common steps, per-term correlation and mean difference, `same_step_count`).

## Decision (user, 2026-11-22): equal wall-clock comparison, no new run

The paper reports F7 as **training for the same wall-clock time** on 2 x H100: in ~43-44 h the engine run reached 1,424 optimizer steps (108.0 s per
step, 42.9 h) and the PyTorch (`torch.compile`) run 433 steps (369.1 s per step, 44.4 h); over their 433 common steps the loss curves agree (mean 0.3841
vs 0.3840, Pearson r 0.979). The full-length PyTorch run above (same step count) is not run. Data: `data/existing/F7_h100_{steps,summary}.csv`.
