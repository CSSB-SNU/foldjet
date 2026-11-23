# F3 Anthropic kit arm on B200 (timing only, random in-process weights)

Status: COMPLETE (2026-11-23 08:12 UTC). Values only from `F3_kits_b200.csv` (collector `collect_b200.py`, 3 repetitions per cell, steady value = drop forward 1 and 2, median of the rest; cell = median of 3).

## Nodes
- **vast B200** (rented vast.ai instance, 1x B200 183 GB, driver 595 / CUDA 13.2): protenix_v1 reps 1-2 (and part of 3) ran there 03:13-03:50 UTC,
  then the user stopped the box. Only rep 1 n=1..8 reached the cluster (`vast_side_note/`). **Not used for any cell** (a cell stays on one node).
- **shared B200 host** DCTN-0831174147-1 (8x B200 sm_100, 178 GiB, driver 580 / CUDA 13.0, Lustre `/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/kits`):
  every cell in the CSV. Column `node` = `<host>:gpu<i>` per repetition.

## Weights (user order 2026-11-23): random, in process, never written to disk
No weight values were downloaded, copied or saved. The checkpoint loaders are intercepted at run time by `randw/randw_boot.py`
(installed as a `zz_randw.pth` line in each kit environment; active only with `RANDW=1`; no kit or upstream file edited):
- **torch checkpoints with a config inside** (RF3 `train_cfg`, Boltz-2 Lightning hyper-parameters, OpenFold3 / OF3-p2 state dicts):
  `ckpt_skeleton.py` fetched ONLY the pickle (`data.pkl`: key names, shapes, dtypes, configs) of the official file by HTTP range requests
  (0.05-0.1 % of the bytes; tensor members never requested) plus small integer storages and tensors named `*version*` (OpenFold3's
  `version_tensor` = [2,0,0], 12 bytes). `torch.load(<checkpoint>)` is answered from the skeleton with every tensor allocated in process
  (1-D: weight 1 / bias 0; >= 2-D: randn / sqrt(fan_in); fixed seeds). The checkpoint path holds a size-matched sparse placeholder.
- **Protenix v1 / v2, OpenDDE**: `torch.load` of the checkpoint returns a sentinel; `load_state_dict` given it keeps the module and sets its
  parameters in process (same rule). Empty placeholder (OpenDDE: size-matched sparse file, its loader re-downloads on a size mismatch).
- **AF3-format kits** (af3_jax, af3_torch: the OF3-p2 port): `alphafold3.model.params.get_model_haiku_params` / `xfold.params.get_alphafold3_params`
  replaced by `randw_hooks.py`: the kit's own converter (`of3_weight_converter.map_of3_to_af3`) run in process on the OF3-p2 skeleton.
- **IntelliFold2 (on af3_jax, full-fat)**: parameter tree from haiku's own init traced abstractly (`jax.eval_shape`) on a 22-residue dummy
  protein with the run's config (IntelliFold overlay applied); Fourier embedding npz drawn in process (`numpy.load` hook).
- **Chai-1** (7 TorchScript archives incl. ESM2-3B): `ts_skeleton.py` fetched every archive member except the parameter storages (code,
  constants, data.pkl); `torch.jit.load` adds the storages with random values in memory.
- **ESMFold2 / -Fast / ESMC-6B** (HF): `hf_skeleton.py` wrote config/tokenizer files and, per safetensors file, only its header followed by a
  sparse hole; `PreTrainedModel.from_pretrained` results get their parameters set in process (`RANDW_HF=1`).
Data files that are not weights were fetched from the official URLs: Protenix/OpenDDE CCD caches (protenix tos), Boltz `ccd.pkl` + `mols.tar`
(HF), Chai `conformers_v1.apkl`, AF3 `build_data` CCD. Template mmCIFs / MSAs / inputs = the harness inputs (copied from the cluster).
Timing caveat: values can matter where a kit branches on them; seen: Protenix v1's first-seed `biascache poison test` fails with tiny weights
(0.02 std) -> the init rule above was chosen so that it passes; no early exit / confidence-dependent recycling is enabled in the protocol flags
(RF3 `early_stopping_plddt_threshold=0`).

## Protocol adaptations (forced)
- Harness = copy of `bench5i28_1118_steady` with paths rewritten (`/home/psk6950/...` -> mirror on the host), no Singularity (route C venvs),
  same kit commands, flags, timers, log parsing and KIT lines; `fb.sh` unchanged; `queue_lines.sh` items unchanged.
- Scheduling (user order): one work queue of items (kit, n, rep) over every free GPU of the host, longest first; reps of one kit may run
  concurrently on different GPUs (recorded per item: `items/<name>.meta`, GPU id + our other running items). Two items with the same
  (kit, n) never run at once (shared per-n log/output paths).
- Shared-host rule: an item starts only on a GPU without any compute process for >= 60 s; a 3-s watcher kills and discards an item the
  moment a foreign PID appears on its GPU (re-queued). All items that ran 04:30-04:43 UTC (colleague's 8-GPU job) are discarded
  (`results/external_contention_0430`).
- No `probe.py` / taskset pinning (whole node shared); GPU clock log every 2 s per item (`items/*.gpu.csv`).
- Toolchain: system CUDA 13.1 nvcc (kits' images: 12.9/13.0); kalign from bioconda (protenix family); openfold3_ob0: DeepSpeed DS4Sci op
  build failed with CUDA 13.1 vs torch cu128 -> DeepSpeed installed without the op (every kit mode keeps the op off).
- `--config`: b200 for openfold3_ob0 and protenix_v2 (shipped `configs/b200.env`); h100 for every other kit (as the harness; no b200.env).
- kalign: bioconda kalign 3.6.0 first; it broke Chai-1's 6TFD template (`discrepant start/end indices`, 3/4 templates). Replaced by kalign
  3.3.5 (the version the kit images pin) at 05:46 UTC before any chai1 / protenix item; every recorded item uses 3.3.5.
- openfold3_ob0: process TMPDIR moved to `/tmp/ffk_of3` (the Lustre path made multiprocessing's AF_UNIX socket path too long).
- CPU pinning: items started before 07:35 UTC ran unpinned on a heavily loaded host (load average up to 157 on 72 cores); from 07:35 every
  item is pinned with taskset to its GPU slot's 9 CPUs (72 / 8) with OMP/MKL threads 9 (queue3.sh's PIN logic; the H100 campaign used
  PIN=16). The rf3 items running at 07:47 were pinned in place by the lead (`taskset -a -c -p`, not restarted). CSV column `job` lists
  `cpu=<rep1>/<rep2>/<rep3>` (unpinned | pinned | pinned@0747). Timed GPU forwards agree across pinned / unpinned reps within noise
  (e.g. boltz2 n16 24.88 / 24.79 / 24.75 s, protenix_v1 n16 33.37 / 33.39 / 33.49 s).
- RoseTTAFold3 (cells n=1, 2, 4, 8 ran entirely with the switch below; n=12, 16 without): its prefetch helper (a nice-10 fork) never answered on this host (`PREFETCH helper gone during the first item: no answer in
  1800 s`, then every fold featurised serially in the parent; lever ended `skipped reason=probe_zero:prefetch`). Featurisation is outside
  the timed forward, so timings are unaffected, but each process lost 30 min. From 07:55 UTC the rf3 items run with the kit's own ablation
  switch `MODEL_OPT_LEVERS_OFF=prefetch` (no kit file changed; the same lever set the earlier rf3 items ended with), recorded in each item's meta.
- Incidents (all discarded, re-run): 04:30-04:43 colleague's 8-GPU job (external contention); 06:41 an in-place overwrite of runitem.sh
  killed 6 running items (re-queued; 2 surviving items recovered from their complete KIT lines); 07:55 three rf3 n8 items restarted with the
  prefetch switch before any forward.

## Kit verdicts on B200 (evidence = the kit's own lines, logs under kits_rep*/items/*.logs and results/smoke)
- **opendde**: exit 5 `[opendde-opt] KERNELS REFUSED route=fast at=arm exit=5 cueq_triatt=fallback:probe_failed: CUDA error: no kernel image
  is available for execution on the device` (its pinned torch 2.7.1+cu126 / cuEquivariance cu12 stack has no sm_100 code). Every cell kit_exit5.
- **openfold3 (openfold3_ob0, configs/b200.env)**: every query `Failed ... pair_transition: the core refused the trunk shape (c_z=128, N=128):
  fold:v2:arch-sm100-not-measured — OPENFOLD3_OB0_OPT_PAIR_STRICT=1`, 0/35 structures, exit 1 (smoke n=1; not queued further).
- **esmfold2 / esmfold2_fast**: `triton.runtime.errors.OutOfResources: out of resource: tensor memory, Required: 528, Hardware limit: 512` in the
  kit's transition kernel (cell inherited from sm_90, `UNCOVERED_CELL ... note=inherited_cc:unmeasured`); not an out-of-memory, so no `big`
  retry under fb.sh (smoke n=1; not queued further).
- **af3_jax**: all forwards run, then `DONE status=FAILED ... reason=kernels_refused:fast: dpa_calls: 4 attention call(s) under
  implementation=None, the route requests triton` -> exit 5; CSV status `ok_kit_exit5` with the steady value.
- **intellifold2 / intellifold2_notxla** (af3_jax kit + IntelliFold overlay): all forwards run, then `DONE status=partial ... levers_short: TTR not
  traced; HOIST_LOGITS not traced; LNP not served; FPF_TRIATT not served: fused=2 fallback=40` -> exit 3; `ok_kit_exit3`. intellifold2 n16:
  fast out of memory, kit `big` mode in a fresh process also out of memory (all 3 reps) -> oom.
- **chai1**: exit 1 with all forwards, `PAIRTRACK GATE REFUSED templ_empty ... all_fallback:templates_present` — the same verdict as on the
  H100 harness (where collect1118.py kept the value); `ok_kit_exit1`.
- **protenix_v1, protenix_v2, boltz2, af3_torch, rf3**: exit 0, ACTIVE lines in the logs.
- Unpinned weights: every kit prints its NOT PINNED / WEIGHTS unknown line and proceeds (random weights by design).

## CSV rule
`collect_b200.py` (as collect1118.py): a repetition counts when it has >= 3 timed forwards (drop forward 1 and 2, median of the rest);
cell = median of the 3 repetition values. status `ok` = exit 0; `ok_kit_exit<rc>` = full forwards but a non-zero kit verdict (named above);
`kit_exit<rc>` = no forwards (kit refused); `oom` = fast and big out of memory; `missing` = not measured.

## Results (s per forward, median of 3 repetitions; [status] when not plain ok; all on the shared B200 host)

| kit | n=1 (128 tok) | n=2 (256 tok) | n=4 (512 tok) | n=8 (1024 tok) | n=12 (1536 tok) | n=16 (2048 tok) |
|---|---|---|---|---|---|---|
| af3_jax | 0.880 [ok_kit_exit5] | 1.310 [ok_kit_exit5] | 2.720 [ok_kit_exit5] | 8.170 [ok_kit_exit5] | 18.690 [ok_kit_exit5] | 36.040 [ok_kit_exit5] |
| af3_torch | 0.898 | 1.384 | 1.720 | 4.897 | 11.430 | 21.743 |
| openfold3 | kit_failed_b200_smoke_n1 | - | - | - | - | - |
| rosettafold3 | 2.182 | 2.970 | 4.101 | 8.384 | 16.518 | 28.759 |
| chai1 | 2.591 [ok_kit_exit1] | 2.646 [ok_kit_exit1] | 6.044 [ok_kit_exit1] | 17.183 [ok_kit_exit1] | 33.451 [ok_kit_exit1] | 56.276 [ok_kit_exit1] |
| esmfold2 | kit_failed_b200_smoke_n1 | - | - | - | - | - |
| esmfold2_fast | kit_failed_b200_smoke_n1 | - | - | - | - | - |
| boltz2 | 0.602 | 0.920 | 2.071 | 6.179 | 13.427 | 24.794 |
| protenix_v1 | 1.388 | 1.608 | 2.434 | 6.525 | 15.000 | 33.385 |
| protenix_v2 | 1.302 | 1.972 | 3.921 | 11.830 | 26.941 | 59.023 |
| opendde | kit_exit5 | kit_exit5 | kit_exit5 | kit_exit5 | kit_exit5 | kit_exit5 |
| intellifold2 | 1.020 [ok_kit_exit3] | 2.150 [ok_kit_exit3] | 6.860 [ok_kit_exit3] | 29.620 [ok_kit_exit3] | 88.210 [ok_kit_exit3] | oom (big(fast OOM)) |
| intellifold2_notxla | 1.020 [ok_kit_exit3] | 1.940 [ok_kit_exit3] | 5.600 [ok_kit_exit3] | 21.860 [ok_kit_exit3] | 53.230 [ok_kit_exit3] | 106.370 [ok_kit_exit3] |

openfold3 / esmfold2 / esmfold2_fast: `kit_failed_b200_smoke_n1` (kit-side failure on sm_100 at n=1, see verdicts; not queued further).
rosettafold3 = the `rf3` kit (rosettafold3 directory). intellifold2 n16 = oom: fast out of memory, then the kit's big mode in a fresh process
also out of memory, all 3 repetitions.

## Files (~/foldforge-scratch/paper/F3/b200/)
- `F3_kits_b200.csv`: columns kit,n,tokens,value_s,mode,status,n_timed,forwards_s,templates,source,job,node,reps,value_reps
  (`job` carries the per-rep CPU state `cpu=r1/r2/r3`; `node` = host:gpu per rep).
- `kits_rep{1,2,3}/done_final/` (KIT marker per item), `kits_rep{r}/items/<item>.{meta,out,gpu.csv,logs/}` (start/end, GPU, concurrent items,
  compute apps at start, pinning, kit logs, nvidia-smi clock log every 2 s).
- `host/dq/` (work queue: items.txt, dispatch.log, finished/), `host/bench/` (adapted harness), `host/randw/` (random-weight loaders,
  skeleton tools), `host/b2/` (install scripts per kit), `host/results/{external_contention_0430,dead_items_0641,reaped,setup_failures*,smoke}`
  (discarded runs, kept for the record), `host/smoke_report.txt`.
- `vast_side_note/`: the vast-box protenix_v1 rep 1 (n=1..8) of 03:13-03:50 UTC; not used in any cell.

## Per-cell detail (reps, CPU state, GPUs)
Per-cell CPU pinning (rep1/rep2/rep3) and nodes:
- af3_jax n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu4; reps 0.880 0.880 0.880
- af3_jax n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu7; reps 1.310 1.310 1.310
- af3_jax n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu3; reps 2.720 2.720 2.720
- af3_jax n=8: unpinned/pinned/pinned; DCTN-0831174147-1:gpu1; reps 8.150 8.180 8.170
- af3_jax n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu6; reps 18.690 18.680 18.700
- af3_jax n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu2 DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 36.040 35.990 36.120
- af3_torch n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu4; reps 0.898 0.906 0.891
- af3_torch n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu0; reps 1.385 1.384 1.378
- af3_torch n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu7; reps 1.720 2.041 1.695
- af3_torch n=8: pinned/pinned/pinned; DCTN-0831174147-1:gpu5; reps 4.895 4.897 4.897
- af3_torch n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu0; reps 11.417 11.444 11.430
- af3_torch n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu4; reps 21.743 21.792 21.693
- rosettafold3 n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu5 DCTN-0831174147-1:gpu7; reps 2.211 2.182 2.178
- rosettafold3 n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu5 DCTN-0831174147-1:gpu6; reps 2.966 3.007 2.970
- rosettafold3 n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu5 DCTN-0831174147-1:gpu6; reps 4.101 4.016 4.172
- rosettafold3 n=8: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu4; reps 8.311 8.526 8.384
- rosettafold3 n=12: unpinned/pinned@0747/pinned@0747; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu2 DCTN-0831174147-1:gpu7; reps 16.518 16.888 16.209
- rosettafold3 n=16: pinned@0747/pinned@0747/unpinned; DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu6; reps 29.503 28.412 28.759
- chai1 n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 2.548 2.676 2.591
- chai1 n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu6 DCTN-0831174147-1:gpu7; reps 2.646 2.778 2.619
- chai1 n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu1; reps 6.044 5.854 6.141
- chai1 n=8: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu4; reps 17.300 16.536 17.183
- chai1 n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu2; reps 33.622 33.451 32.752
- chai1 n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu5; reps 60.351 56.276 56.248
- boltz2 n=1: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 0.602 0.600 0.625
- boltz2 n=2: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 0.911 0.920 0.965
- boltz2 n=4: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 2.071 2.130 2.071
- boltz2 n=8: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 6.151 6.249 6.179
- boltz2 n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 13.420 13.427 13.440
- boltz2 n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu7; reps 24.878 24.794 24.751
- protenix_v1 n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu5; reps 1.388 1.361 1.473
- protenix_v1 n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu4; reps 1.608 1.605 1.689
- protenix_v1 n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu5; reps 2.445 2.434 2.359
- protenix_v1 n=8: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu6; reps 6.546 6.513 6.525
- protenix_v1 n=12: unpinned/pinned/pinned; DCTN-0831174147-1:gpu5; reps 15.000 15.071 14.982
- protenix_v1 n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu7; reps 33.370 33.385 33.493
- protenix_v2 n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu6; reps 1.221 1.302 1.304
- protenix_v2 n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu6; reps 1.972 1.974 1.868
- protenix_v2 n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu4; reps 3.921 3.947 3.892
- protenix_v2 n=8: pinned/pinned/pinned; DCTN-0831174147-1:gpu1; reps 11.830 11.984 11.830
- protenix_v2 n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu6; reps 26.941 26.932 27.069
- protenix_v2 n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu5 DCTN-0831174147-1:gpu7; reps 59.023 59.015 59.066
- opendde n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu0; reps kit_exit5 kit_exit5 kit_exit5
- opendde n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu5; reps kit_exit5 kit_exit5 kit_exit5
- opendde n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu0 DCTN-0831174147-1:gpu5; reps kit_exit5 kit_exit5 kit_exit5
- opendde n=8: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4 DCTN-0831174147-1:gpu6; reps kit_exit5 kit_exit5 kit_exit5
- opendde n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu1; reps kit_exit5 kit_exit5 kit_exit5
- opendde n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu4; reps kit_exit5 kit_exit5 kit_exit5
- intellifold2 n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu5; reps 1.020 1.020 1.020
- intellifold2 n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu1; reps 2.150 2.150 2.150
- intellifold2 n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu3; reps 6.860 6.860 6.860
- intellifold2 n=8: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu2 DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu7; reps 29.830 29.580 29.620
- intellifold2 n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu1 DCTN-0831174147-1:gpu6; reps 88.280 88.210 87.480
- intellifold2 n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu2 DCTN-0831174147-1:gpu4; reps oom oom oom
- intellifold2_notxla n=1: pinned/pinned/pinned; DCTN-0831174147-1:gpu6; reps 1.020 1.020 1.020
- intellifold2_notxla n=2: pinned/pinned/pinned; DCTN-0831174147-1:gpu5; reps 1.950 1.940 1.940
- intellifold2_notxla n=4: pinned/pinned/pinned; DCTN-0831174147-1:gpu5; reps 5.680 5.600 5.600
- intellifold2_notxla n=8: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu3 DCTN-0831174147-1:gpu6; reps 21.980 21.860 21.840
- intellifold2_notxla n=12: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu1; reps 53.330 53.230 53.210
- intellifold2_notxla n=16: unpinned/unpinned/unpinned; DCTN-0831174147-1:gpu3; reps 106.290 106.370 106.380
