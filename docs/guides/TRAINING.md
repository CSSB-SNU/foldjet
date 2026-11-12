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
