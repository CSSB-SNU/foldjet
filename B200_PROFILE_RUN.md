# B200 profile run of the F6 engine arm (prepared 2026-11-23, NOT launched: all 8 GPUs were in use by a colleague)

Purpose: find why the engine arm of F6 is slower on the B200 (AF3 1.150 s, Protenix v1 1.140 s per example) than on the H100 (0.812 / 0.787 s) although
the E1 operators are faster. The H100 profile of the same step (`~/foldforge-scratch/f6prof_out/h100/`) is the reference.

## Rules (user)
- ONE ssh connection with ControlMaster on the command line; no polling loops; one launch, then one read of the result files.
- GPUs 0-3 only, and only when NO compute process is on the GPU (`nvidia-smi -i N --query-compute-apps=pid --format=csv,noheader` is empty). If someone uses it, do not use it.
  `b200_profile.sh` refuses to start on a busy GPU and writes `CONTAMINATED_gpu<N>` when a foreign process appears during the run (discard that result).
- Nothing else runs on the host from this session; the campaign lanes are finished (check `ps -u khidi_hjju | grep b200_lane` is empty first, else wait).

## What is deployed (files from the branch f6prof of ~/foldforge-scratch/f6prof)
`benchmarks/paper/F6_training_speed/{profile_step.py, profile_step.sh, b200_profile.sh, compare_profiles.py}`: copied into the B200 checkout
(`$W/ff/foldforge/benchmarks/paper/F6_training_speed/`; untracked files, the campaign scripts are not touched). `W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200`.
Environment (identical to the F6 engine task of `common/b200_task.sh`): `ffenv` (torch 2.13.0+cu129), engine tree `$W/diag-nan2` (engine main f06154bc7 + B200 clamp
and holes fixes, own sm100a JIT cache `$W/diag-nan2/cache`), `CUDA_HOME=$W/eng-2511/.pixi/envs/default` (nvcc 12.9), `common/cards/b200.env`
(`B200_ENGINE_FIX3=$W/diag-nan2` selects all of that); pool `$W/ff/paper/F6_training_speed/b200/pool/pool_protenix1.pt` (the original H100 stand-in pool).

## Commands (from the H100 cluster login node; the connection `b200` is `ProxyJump cssb` + ControlMaster in ~/.ssh/config, repeated on the command line)
```
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/b200-%C -o ControlPersist=4h b200"
SCP="scp -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/b200-%C"
F=~/foldforge-scratch/f6prof/benchmarks/paper/F6_training_speed
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200
# 1. deploy + choose a free GPU + run graph profile then eager profile in ONE ssh command (background on the host, logs on the host)
$SCP $F/profile_step.py $F/profile_step.sh $F/b200_profile.sh $F/compare_profiles.py b200:$W/ff/foldforge/benchmarks/paper/F6_training_speed/
$SSH 'W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200; D=$W/ff/foldforge/benchmarks/paper/F6_training_speed
 for g in 0 1 2 3; do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && GPU=$g && break; done
 [ -z "${GPU:-}" ] && { echo "no free GPU in 0-3: NOT started"; exit 3; }
 echo using GPU $GPU; mkdir -p $W/ff/paper/F6_profile/b200
 nohup bash -c "GPU=$GPU bash $D/b200_profile.sh > $W/ff/paper/F6_profile/b200/run_graph.log 2>&1; GPU=$GPU EAGER=1 PROBE=0 bash $D/b200_profile.sh > $W/ff/paper/F6_profile/b200/run_eager.log 2>&1" > /dev/null 2>&1 &'
# 2. ~8-12 min later (first run may JIT-build kernels: the cache of diag-nan2 is warm for these exact kernels; r1 of the campaign took 119 s for step 0), ONE read:
$SSH 'cd /NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/paper/F6_profile/b200; ls; ls CONTAMINATED* 2>&1; cat probe_protenix1_miniworld.txt profile_tables_protenix1_miniworld.txt profile_tables_protenix1_miniworld_eager.txt; grep -a "BENCHSTEP\|STEP_GRAPH\|F6PROF micro" log_protenix1_miniworld.txt | cut -c1-300'
# 3. bring the small result files back (not the traces, ~270-800 MB each: kept on the host, delete when done) and compare on the cluster (srun CPU job, never on the login node)
$SCP "b200:$W/ff/paper/F6_profile/b200/{profile_*.json,profile_tables*.txt,ops*.json,probe*.txt,meta*.txt,log_*.txt}" ~/foldforge-scratch/f6prof_out/b200/
srun -p h100 -A cssb --qos=cssb_h100 --gres=none -c 2 --mem=8G --time=00:05:00 python $F/compare_profiles.py ~/foldforge-scratch/f6prof_out/h100 ~/foldforge-scratch/f6prof_out/b200 _protenix1_miniworld_eager
```
(PROBE=1 default of the first command runs the GEMM probe: fp32 IEEE FFMA / TF32 / bf16 / engine ieee_f32.mm at the pair-transition shapes: one minute.)
AF3 model: `MODEL=af3` on a second run if the Protenix v1 profile shows nothing (the two models differ by <1 % on both cards).

## How to read the result (decision table)
| B200 observation | meaning |
|---|---|
| `host launches: cudaGraphLaunch 8`, GPU idle < 5 % and micro wall about 1.15 s | GPU-bound: the slowdown is kernel time; the class / op table says which kernels (hypotheses 1, 3, 5) |
| idle fraction high (> 20 %) or cudaLaunchKernel count in the 10^4 | host-bound / graph not replayed (hypothesis 2) |
| class `ieee_f32` > 15 % of the B200 step (H100: 2.4 %) | hypothesis 1 (FFMA chain) |
| class `triton` > 3 % | hypothesis 3 (LayerNorm w64) |
| the B-A table: a class / op with B > A (e.g. `cublas / cutlass tensor-core gemm` TF32 `xmma`/`nvjet` kernels, `attn_tf32_*`, `token_dit_train_*`, `triattn_*`) | that operator is slower in F6 context than E1 says: E1 operator-case versus F6 shapes (fp32 / TF32 diffusion head, 48 samples) |
| per-op times in the eager run equal to the H100's x 0.5, but micro wall 1.15 s | something outside the kernels (graph node overhead on Blackwell, memcpy, sync) |
| step wall of the profile run at (or near) 9.2 s with 4 lanes absent | hypothesis 4 (concurrent lanes) excluded |
Check also: `engine_file` in the BENCH line must be `.../diag-nan2/src/miniworld_engine` (source tree, hypothesis 5); a wheel run on the B200 does not exist (no sm100 kernels in wheels).
Measuring hypothesis 5 directly (optional second job): the same profile with the 3.0.9 source tree (`ENGINE_ROOT=$W/engine307` without B200_ENGINE_FIX3) is not valid (clamp NaN), so skip.
