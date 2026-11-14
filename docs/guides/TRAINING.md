# Protenix v1 파인튜닝 (학습 스크립트, phase 0/1)

`scripts/train_protenix1.py`와 `src/foldforge/train/`는 FoldForge 모델(protenix1)을 **MiniWorld의 학습 레시피**로 파인튜닝한다.
현재 구현 범위는 phase 0(기울기·엔진 backward 확인)과 phase 1(eager 스텝, 단일 GPU, gradient accumulation)이다. 데이터 로더(phase 2)와
스텝 전체 CUDA 그래프(phase 3)는 아직 없다.

## 레시피 (`foldforge.train.config.TrainConfig`)

출처는 MiniWorld의 `configs/miniworld/phase2a_diffusion_medium_v200_b200_joint.yaml` 계열, `loss/diffusion_v102_ft.yaml`, `train/default.yaml`이다
(Protenix의 `finetune_demo.sh`가 아니다).

| 항목 | 값 |
|---|---|
| 손실 | diffusion 4.0, alpha_dna 5, alpha_rna 5, alpha_ligand 10, bond 1.0, distogram 0.03, smooth LDDT 0.0(미구현) |
| 옵티마이저 | Adam(0.9, 0.95), max_lr 5e-3, min_lr 1e-4, warmup 1000, 5e4 스텝마다 x0.95 |
| clip / EMA | grad clip 1.0, EMA 0.999 |
| 유효 배치 | 배치 1 x grad_accum 32 x GPU 수 |
| 정밀도 | fp32 마스터 가중치 + bf16 autocast, 손실은 fp32 |
| recycle | 1~4 중 균등 추출, 마지막 pass만 기울기 |
| 확산 | 증강 48개, EDM P_mean -1.2, P_std 1.5, sigma_data 16 |

## 경고: 학습률

`max_lr=5e-3`은 MiniWorld에서 **동결된 트렁크 + 새로 초기화한 확산 헤드**에 쓰는 값이다. 사전학습된 Protenix 트렁크를 풀어 이 값으로 돌리면 가중치가
망가질 수 있다. 스모크(고정 예제 한 개, warmup 0)에서 5e-3은 처음 10스텝 평균 손실이 4.5배로 튀었다가(시작 1.02 -> 4.50) 50스텝 안에 회복했다.
1e-4는 안정적으로 감소했다. 실제 파인튜닝에서는 `--trunk-lr-scale`(트렁크 LR 배율)이나 `--max-lr`를 낮춰서 시작하는 것을 권한다.

## 죽은 모듈

Protenix v1에는 모든 파라미터가 1e-37 수준인 모듈이 있다(추론에서는 건너뛴다, [CONFIG.md](CONFIG.md)의 `engine.skip_dead_modules`). 학습에서는 이 모듈도 돌아야
파라미터가 학습된다. 학습 스크립트는 `config={"engine": {"skip_dead_modules": False}}`로 모델을 읽고, `prepare.enable_training`이 남은 `dead` 플래그도 지운다.
알려진 한계: 이 파라미터 494개는 bf16에서 기울기가 0으로 언더플로한다(죽은 가중치 자체가 1e-37 ~ 1e-12이고 pytorch 백엔드에서도 같다). 엔진 문제가 아니다.

## 동결되는 파라미터와 기울기 점검

`src/foldforge/train/freeze.py`의 표 하나(`RULES`)가 학습에서 갱신하지 않는 파라미터를 정한다. `prepare.enable_training`이 `requires_grad`를 끄고, 옵티마이저는
그 파라미터를 받지 않는다. 체크포인트 키는 바뀌지 않는다(저장·읽기 형식 그대로). 규칙마다 이유가 있다.

| 규칙 | 대상 | 이유 |
|---|---|---|
| `atom_pair_unused` | 입력 임베더·확산 헤드의 `single_to_pair_cond_row/col`, `embed_pair_offsets`, `embed_pair_distances` (`_1`이 없는 쪽, 모델당 8개) | `prepare`가 `_per_atom_conditioning`의 atom-pair 출력을 버린다(AF3도 버림). 실제로 쓰는 것은 `_1` 쪽이고 변환기는 원본 가중치 하나를 두 자리에 복사한다. 엔진 backward 문제가 아니다(R16 조사: 백엔드와 무관, 10개 모델 모두 같음) |
| `last_msa_block` | 마지막 MSA 블록의 `msa_attention1`, `msa_transition` (OPM이 먼저인 모델: Protenix, AF3 계열, ESMFold2) | 마지막 블록이 갱신한 MSA는 버려져 손실에 닿지 않는다. Protenix·ESMFold2 체크포인트는 여기가 0으로 채워져 있다. Boltz 순서(MSA 갱신이 OPM보다 먼저)는 해당 없음 |
| `experimentally_resolved` | confidence 헤드의 `experimentally_resolved_ln/logits` | MiniWorld 레시피에는 이 손실이 없다(가중치 0, `train/loss/confidence.py`). 원본 Protenix·Boltz-2는 학습하므로, 손실을 넣게 되면 이 규칙을 지운다 |
| `boltz2_contact_fourier` | Boltz-2의 `contact_fourier` (입력·confidence 재임베딩, 가중치와 편향) | 원본에서 고정(고정 푸리에 특징)인데 FoldForge에서는 학습되고 있었다 |
| `esmfold2_*` | ESMFold2의 단일 트랙(confidence pairformer의 `single_*`, `single_activations`, `prev_single_embedding*`), atom pair 슬롯(`pair_mlp_*`, `*_1`, `embed_trunk_*_cond`, `pair_logits_projection`), confidence의 contact·bond 재임베딩 | 원본에 없어서 변환기가 0으로 채운 자리. 학습하면 원본에 없는 용량이 생긴다. **값이 정확히 0일 때만** 동결한다(나중 체크포인트가 이 자리를 채우면 동결되지 않는다) |

기울기 점검(`GRAD_COVERAGE`, `scripts/train_protenix1.py`)은 한 스텝 뒤 파라미터를 분류한다. 결함으로 세는 것은 `none`(기울기 없음), `zero_live`(살아 있는 가중치의 0 기울기),
`nonfinite`뿐이다. 나머지는 이유가 있는 분류다.

- `frozen`: 위 표의 동결 파라미터.
- `data_dependent`: 예제에 전하 원자·공유 결합이 있어야 기울기가 흐르는 `embed_ref_charge`, `bond_embedding`. 8ayg에는 둘 다 없어서 0이고, **7fvu(전하 원자 26개, 리간드 결합 41개)로 스모크를 돌리면 기울기가 흐른다**
  (`DATA_DEPENDENT ... gradient_flows=True`가 출력된다).
- `underflow`: 0 기울기이고 가중치가 모델의 전형적인 가중치(파라미터별 최대 절대값의 중앙값)의 1e-6배보다 작다. 절대값 1e-30 대신 상대값으로 판정하므로 1e-12 수준으로 감쇠한 confidence pairformer 블록도 잡힌다.

템플릿 쪽 기울기는 실제 템플릿이 있는 타깃으로 확인한다. 7fvu는 FoldBench 템플릿 LMDB의 `7fvu_A` 레코드가 비어 있어(템플릿 0개, 데이터 쪽 사실) 템플릿이 배치에 들어가지 않는다.
템플릿 확인은 7t9r(38잔기, 템플릿 4개)이나 8oys를 쓰고, 스모크 로그의 `TEMPLATES real_templates=... real_atoms=...`로 템플릿이 실제로 들어갔는지 본다.

## RoseTTAFold3의 공유 MSA 블록

원본 RoseTTAFold3는 MSA 모듈 하나를 4번 돌린다. 변환기는 같은 가중치를 `msa_stack.0 ~ 3`에 복사해 두었고, 이전에는 4개의 독립 사본으로 만들어져 학습에서 0~2번 사본이 따로 변하고 3번은
(마지막 블록이라) 학습되지 않았다. `DenseSpec.msa_shared_block`(RoseTTAFold3만 True)이면 블록 하나를 4번 등록한다. `state_dict`에는 키 `msa_stack.0 ~ 3`이 그대로 남고(체크포인트 형식 불변),
`parameters()`에는 한 번만 나와서 옵티마이저와 EMA는 파라미터 한 벌을 본다. 추론 결과는 사본 4개일 때와 같다(`tests/train/test_freeze.py`, RoseTTAFold3 추론 CIF 대조).

## 엔진 2.5.3: 학습 스크립트의 atom 블록 우회 제거

2.5.1에서 atom-DiT `cuda` backward가 죽어(R20) 학습 스크립트가 `atom_blocks`를 `pytorch`로 돌렸다. 2.5.3에서는 고쳐져서 우회를 뺐다(엔진 기본 백엔드). 스모크 5스텝이 우회 없이 통과한다.

## 데이터

FoldForge는 MiniWorld를 import하지 않는다. 데이터 소스는 `--batch-source module:callable`이다. callable은 `(cfg, device)`를 받아 `TrainBatch`의
이터레이터를 돌려준다(`foldforge.train.data`). 어댑터는 MiniWorld 쪽에서 제공한다(BioMolDB 카탈로그와 LMDB 경로는 MiniWorld의
`configs/miniworld/data/H100_default.yaml`).

## 실행

```bash
# 스모크: 고정 예제 하나(FoldBench 타깃, 정답은 모델 자신의 예측으로 대체) 과적합 + 기울기 점검 + 커널 표 + 체크포인트 왕복
python scripts/train_protenix1.py --smoke --target 8ayg-assembly1 --steps 50 --warmup-steps 0 --run-dir RUN
# 학습
python scripts/train_protenix1.py --batch-source my_pkg.adapter:batches --steps 100000 --run-dir RUN [--ckpt X.miniworld.pt]
```

체크포인트는 네이티브 `*.miniworld.pt`(+ 같은 이름의 `.train.pt`: optimizer, EMA, step)로 저장된다. `protenix1.bin.zst` 형식으로의 역변환은 없다.

## Speed measurement of the three backends

`scripts/train_protenix1.py --bench` times REAL optimizer steps (trunk + diffusion + confidence, backward, gradient all-reduce, clip, Adam, EMA); it is not a finetune
(stand-in data, nothing is saved, divergence does not stop it). The same command line with `--backend pytorch | cueq | miniworld` gives the three options:

| `--backend` | what runs |
|---|---|
| `pytorch` | every engine module on its engine `pytorch` reference backend |
| `cueq` | the engine's connection to NVIDIA cuEquivariance for the classes that have it: `TriangleMultiplication`, `TriangleAttention`, `AttentionPairBias`, `TokenDiTProtenix` (forward and backward, served with the gradient enabled); `OuterProductMean`, `MSAPairWeightedAveraging`, `Transition`, `AtomDiTProtenix` and the sampler glue have no `cueq` backend and run `pytorch` (`engine.unsupported: fallback`, listed in `LOAD_REPORT`) |
| `miniworld` (default) | our kernels per module class: `cuda` on an H100 where the class has it, else `triton`, else `pytorch` |

The effective batch is `micro batch 1 x grad_accum x number of GPUs`: 256 = 32 x 8 GPUs. `--effective-batch 256` derives `grad_accum = 256 / world size` (an error if it
does not divide). Data parallelism is NOT DDP (the step runs a forward, several partial backwards and accumulates): every rank runs its own micro steps, the gradients are
summed over the ranks in a few flattened buckets (`train/dist.py`) with the micro-step scale `1 / (grad_accum x world size)` (the same loss scale as one process with that many
micro steps), then clip / Adam / EMA run replicated. Parameters are broadcast from rank 0 once. In `--bench` every micro step is a pure function of
`(--seed, global micro index)` (`train/bench.py::micro_plan`: pool example, 1-4 recycles, torch RNG seed for noise / rotations), rank `r` takes the micro steps `r, r + W, ...`:
the three backends and any GPU count see the same data stream, and a 2-rank run equals the 1-process run with twice the accumulation.

```bash
# one GPU, 30 steps x 256 micro steps (long: see the per-micro-step time below), one pool file shared by all runs
POOL=/path/to/pool.pt        # built by the first run (rank 0), loaded by every later run and rank: identical stand-in labels
for B in pytorch cueq miniworld; do
  python scripts/train_protenix1.py --bench --backend $B --effective-batch 256 --steps 30 --warmup-steps 10 --bench-pool-cache $POOL
done
# 8 GPUs: 8 x 32 = 256
for B in pytorch cueq miniworld; do
  torchrun --nproc-per-node 8 scripts/train_protenix1.py --bench --backend $B --effective-batch 256 --steps 30 --warmup-steps 10 --bench-pool-cache $POOL
done
# smoke of the plumbing: 3 steps of 2 micro steps
python scripts/train_protenix1.py --bench --backend cueq --steps 3 --grad-accum 2 --warmup-steps 1 --bench-pool-cache $POOL
```

Output (rank 0 only):

- `LOAD_REPORT`: the loader's plan and the engine modules that are not on the requested backend, with the reason.
- `ENGINE_AFTER_MICRO1` / `ENGINE_AFTER_MICRO2` / `ENGINE_FINAL`: the engine dispatch per module class from `report_observations` while the gradient is enabled.
  `selected` = the backend each module of the class is set to, `served` = host calls that ran on each backend (a refused call is served by `pytorch`), `refused` = calls the
  selected backend refused, with the reason (for example a backward the cuEquivariance op does not take). Counts are Python calls in the eager training step; nothing here
  is a kernel trace. The comparison is only honest when this table is read with the times: a class listed as `selected=cueq ... refused=...` ran `pytorch` for those calls.
- `POOL targets=... tokens=[...]`: the examples (tokens after FoldForge's featurisation).
- `BENCHSTEP it=.. wall_s=.. loss=.. lr=.. peak_gib=.. micro_steps=.. gnorm=..`: CUDA-synchronized wall time of the whole optimizer step (micro steps, all-reduce, clip, Adam, EMA).
  `micro_steps` is per optimizer step over all ranks (the effective batch); `peak_gib` is rank 0's peak allocated memory of the step. With `--profile-phases` the per-phase
  `PHASES` lines are printed too (they add synchronizations).
- `BENCH {json}`: backend, world size, grad_accum, effective batch, steps, `first_step_s` (includes first-call overheads, reported separately), mean / median / min / max over steps 2..N,
  `total_wall_s`, `micro_steps_per_s` (all ranks), `seconds_per_micro_step_per_gpu`, `tokens_per_s`, `nonfinite_steps` (counted, the run goes on), peak allocated / reserved GiB, the
  engine table, git commit, engine version and `miniworld_engine.__file__`, GPU name, and `ranks_identical` (the parameter checksum equals on every rank).

Data are STAND-INS: FoldForge's featurisation of FoldBench targets (`--bench-pool default` = 8 targets with 0.7 x N .. N tokens (size table of the featurisation, <= 4096 atoms), N = `--bench-tokens`; an example that featurises above N tokens is dropped, default 384 = MiniWorld's recipe crop
`crop.max_tokens: 384`, bucket 384; or `--bench-pool a,b,c`), the ground truth is the model's own pre-training prediction. Step time depends on the shapes, not on the label values,
but the featurisation is FoldForge's inference one (MSA bucket 1024 rows, 4 templates where the target has them, no recipe crop / sampling): the recipe uses `bucket_msa_size: 128`,
so the MSA part of a measured step is heavier than in the real run. Without `--bench-pool-cache` every process builds its own pool and the labels come from the backend being measured.

Memory (80 GB H100, 384 tokens, 48 augmentations): the 48 augmentations run in ONE diffusion forward / backward (`--augment-chunk 0`, the recipe; `--bench` has no chunk default:
a chunk of 8 slowed eager by about 60 %). `--bench` uses `--activation-checkpointing` (the pair-stack blocks are recomputed in the backward): `miniworld` fits an 80 GB card either way
(peak 68 GiB without, 17 GiB with; 5 % slower with). `pytorch` and `cueq` at chunk 0 were NOT re-measured after the chunk default was removed (their earlier numbers were taken at chunk 8
and are invalid). `--no-activation-checkpointing` / `--augment-chunk N` change the flags. Build the pool in a separate first run (`--bench-pool-cache`, one short `--steps 1`): the training numerics of a process that has just run the pool-building
inference differ by ~1 % from a fresh process (pre-existing state of the inference calls; the loaded pool does not have it).
Peak numbers in `BENCHSTEP` are wrong with `--profile-phases` (the phase timer resets the peak counters).

Planning (H100 80GB, default pool 268-384 tokens, 48 augmentations in chunks of 8, activation checkpointing, 1-4 recycles, steps 2..4 of a 4 x 4 micro-step run, 2026-11-13): seconds per micro step per GPU
`pytorch` 6.2, `cueq` 4.0, `miniworld` 2.2. A bench of 30 steps x 256 micro steps is 960 micro steps per GPU on 8 GPUs (about 100 / 65 / 36 minutes) and 7680 on one GPU (13.2 / 8.6 / 4.8 hours);
the optimizer, EMA and the gradient all-reduce add well under a second per step. `--bench` is not resumable: give the job a time limit above that.
`--micro-log` prints one `MICRO rank=.. g=.. example=.. passes=.. loss=..` line per micro step (the global index `g` is the same in a 1-process and an N-rank run).

### Whole micro step as a CUDA graph (`--step-graph`)

`--step-graph` (default off; `train/step_graph.py`) captures the WHOLE micro step (trunk recycles, diffusion forward / backward, distogram, trunk backward, the 20-step mini-rollout,
confidence forward / backward) as one CUDA graph per recycle count and replays it; the eager step stays the default and the fallback. Use it with `--bench-bucket` (one input shape; without
it every distinct shape signature is another graph, at most 8, the rest runs eager) and `--rollout-order graph` or `after` (the rollout is recorded inline in the step graph).
Gradients accumulate in place into persistent `.grad` tensors (the loop zeroes them with `zero_()`, not `set_to_none`); weights are read at every replay (pack / table builds are part of the
capture); the random numbers come from the CUDA generator of the replay (`torch.manual_seed` before a replay seeds it; measured: bitwise the eager numbers for the same seed).
Memory: the graphs share one private pool that stays reserved, on top of the eager peak (the `STEP_GRAPH` line of a `--bench` run reports captures, replays, fallbacks, pool GiB).
Measured on one H100 (stand-in pool, bucketed, activation checkpointing): `miniworld` 2.2 -> 0.95 s per micro step at 4 recycles (GPU idle 64 % -> 3 %), `cueq` and `pytorch` are GPU bound
(1.1x / 1.05x). Do not use `--profile-phases` with it (the phases are inside the graph).

### torch.compile (`--compile`)

`--compile [blocks|diffusion|all]` (default off, bare flag = `all`; `train/compile.py`) wraps the `forward` of the repeated blocks in ONE `torch.compile(dynamic=False)` function
per process: the pair-stack blocks (`PairformerBlock` of trunk / templates / confidence head, `EvoformerBlock`: `blocks`) and the diffusion transformers (`TokenDiTProtenix`,
`AtomDiTProtenix`: `diffusion`). It is meant for the `pytorch` backend (the engine modules are plain ops that Inductor fuses; the kernel backends are opaque autograd nodes).
The granularity is the block, not the recycle or the step: the micro step has host control flow (random recycle count, chunk loop, loss-part host read) that a whole-step trace would
specialise on; the blocks run on identical shapes 100+ times per step, so one graph per (class, grad mode, shape) serves all instances (nn.Module passed as an argument: 91 modules, 13 graphs).
Rules: `dynamic=False` with bucketed shapes (`--bench-bucket` / the recipe crop), `recompile_limit` 32 (past it dynamo runs a block eagerly), `triton.cudagraphs` off (the step graph is our
capture), `fallback_random` (dropout scales come from eager's RNG ops: same seed, same random numbers; CUDA-graph safe). The checkpoint wrapper (`--activation-checkpointing`) recomputes the
compiled forward (compile is applied first). Warm-up: `--bench` / training run one scale-0 micro step at the deepest recycle count before the loop (compiles the no-grad, grad / recompute, 48-sample diffusion and
1-sample rollout variants); `--bench` then runs the timed steps under `error_on_recompile`. With `--step-graph` the first capture warms up at the deepest recycle count, does a throwaway
capture (code that branches on "is the stream capturing"), and every kept capture runs under `error_on_recompile` (a failing capture falls back to eager and is reported in `STEP_GRAPH`).
`COMPILE_WARMUP seconds=..` and a final `COMPILE {...}` line (dynamo `unique_graphs`, `graph_breaks`, `recompiles`) are printed. Persist the compile caches between jobs with `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`
(cold warm-up about 170 s on an H100, 20-30 s with a warm cache). Compiled regions do not update the host dispatch counters (`ENGINE_*` tables show `served=-`).

Measured (2026-11-15, one H100 80 GB, `pytorch` backend, stand-in pool 268-384 tokens bucketed to 384 / 4096 atoms, 48 augmentations, activation checkpointing, random 1-4 recycles, median over timed steps 2..N of 8-micro steps, seconds per micro step):
eager chunk 0 OOM (peak above 75 GiB, OOM at step 1); eager chunk 24 5.45 (peak 46.7 GiB); eager chunk 8 5.84 (32 GiB); eager + `--step-graph` chunk 8 5.11 (chunk 24 and 0 OOM: graph pool on top of the eager peak);
`--compile` chunk 0 2.50 (2.57 over 3 x 32 micro steps; peak 49.7 GiB); `--compile --step-graph` chunk 0 2.12 (2.23 over 3 x 32 micro steps; 4 captures, 0 fallbacks, graph pool 45 GiB on top of a 47 GiB warm-up peak).
Region choice at chunk 0: `blocks` alone OOMs (the 48-sample diffusion activations: peak above 75 GiB) and `diffusion` alone is no faster than eager (about 5.2); `all` is the one that fits and is 2x faster.
Numerics: eager is bitwise reproducible run to run; a compiled step is not (compile vs compile: step-0 gradient relative L2 1.4e-2, cosine 0.9999; eager vs compile 2.9e-2 / 0.9996), per micro step the loss
agrees to 0.2 % at 1 recycle and 1-4 % at 4 recycles. Dynamo: 13 graphs for 91 wrapped modules, 0 graph breaks, 0 recompiles after the warm-up.
`--bench-dump DIR` writes the step-0 gradients and the final parameters for run-to-run comparison; `--msa-rows N` sets the MSA rows drawn per recycle (recipe 128, default 1024).
### L768 stand-in bench

`--bench-tokens 768 --bucket-atoms 8192 --num-augment 32 --augment-chunk 0 --bench-bucket` (MiniWorld `distogram_diffusion_medium_v130_bioai_L768.yaml`: `crop.max_tokens 768`, `max_atoms 8192`; Protenix v1 fine-tuning stages 2-3: crop 768, diffusion batch 32).
`--bucket-atoms` also bounds the default pool (targets above it are skipped) and sets the length of the present-atom index of every example (`bench.FIXED_LENGTH`), so that all examples have ONE step-graph signature.
`FF_BENCH_PASSES_SEQ=1,2,3,4` (bench only) replaces the random recycle count by that cycle over the global micro index (capture-order tests); `FF_STEP_GRAPH_DEBUG=1` logs what differs when a second graph signature appears.
