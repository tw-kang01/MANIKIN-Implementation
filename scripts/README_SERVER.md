# MANIKIN Server Training Guide

원격 GPU 서버에서 MANIKIN 모델을 학습하는 방법을 설명합니다.

## 📋 사전 준비

### 로컬 환경 (Windows)

**필요한 데이터셋 확인:**
```bash
# 3개 데이터셋이 모두 있어야 함
ls data_manikin/BioMotionLab_NTroje/train  # ✓
ls data_manikin/CMU/train                  # ✓
ls data_manikin/MPI_HDM05/train            # ✓
```

**Body 모델 확인:**
```bash
ls AvatarPoser/support_data/body_models/smplh/neutral/model.npz
```

---

## 🚀 Step-by-Step 서버 학습 가이드

### Step 1: 업로드 파일 준비 (로컬)

```bash
# Git Bash에서 실행
cd c:/Users/KTW/Manikin-Sage
bash Manikin/scripts/prepare_server_upload.sh
```

이 스크립트가 생성하는 파일:
- `data_manikin.tar` (약 10-20GB) - 3개 데이터셋 압축
- `body_models.tar` (약 500MB) - SMPL-H 모델

---

### Step 2: 서버로 파일 업로드

#### Option A: SCP 사용

```bash
# 1. 서버 디렉토리 생성
ssh aurora-g8 'mkdir -p /data/ktw3389/Manikin-Sage'

# 2. 데이터셋 업로드 (크기가 크므로 시간이 오래 걸림)
scp data_manikin.tar aurora-g8:/data/ktw3389/Manikin-Sage/

# 3. Body 모델 업로드
scp body_models.tar aurora-g8:/data/ktw3389/Manikin-Sage/

# 4. 코드 업로드 (rsync 권장 - 변경된 파일만 전송)
rsync -avz --progress Manikin/ aurora-g8:/data/ktw3389/Manikin-Sage/Manikin/
rsync -avz --progress AvatarPoser/ aurora-g8:/data/ktw3389/Manikin-Sage/AvatarPoser/
```

#### Option B: Git 사용 (코드만)

```bash
# 서버에서 실행
ssh aurora-g8
cd /data/ktw3389
git clone <your-repo-url> Manikin-Sage

# 데이터셋과 body_models는 여전히 scp로 업로드 필요
```

---

### Step 3: 서버에서 학습 시작

```bash
# 서버 접속
ssh aurora-g8

# 작업 디렉토리 이동
cd /data/ktw3389/Manikin-Sage

# SLURM 작업 제출
sbatch Manikin/scripts/train_slurm.sh
```

**출력 예시:**
```
Submitted batch job 12345
```

---

### Step 4: 학습 모니터링

#### 실시간 로그 확인

```bash
# 로그 파일 위치
tail -f /data/ktw3389/Manikin-Sage/logs/slurm-12345.out
```

#### SLURM 작업 상태 확인

```bash
# 작업 목록 확인
squeue -u ktw3389

# 특정 작업 상세 정보
scontrol show job 12345

# 작업 취소
scancel 12345
```

---

## 📊 학습 중 체크포인트

학습 중 생성되는 파일들:

```
/data/ktw3389/Manikin-Sage/
├── Manikin/outputs/
│   ├── logs/
│   │   └── train_v2_20251209_183730/  # 학습 로그
│   └── models/
│       └── v2_20251209_183730/
│           └── manikin_v2_best.pth    # 최고 성능 모델만 저장
└── logs/
    └── slurm-12345.out                 # SLURM 출력 로그
```

---

## 🔧 Config 파일 설정

### 로컬 vs 서버 Config 비교

**로컬 학습용** (`manikin_config.json`):
- 작은 배치 사이즈 (256)
- 적은 worker (4)
- 짧은 학습 시간

**서버 학습용** (`manikin_config_server.json`):
- 큰 배치 사이즈 (512)
- 많은 worker (8)
- 긴 학습 시간 (100k iterations)

SLURM 스크립트는 자동으로 적절한 config를 선택합니다.

---

## 📝 중요 사항

### 1. 데이터셋 경로 자동 업데이트

SLURM 스크립트가 자동으로:
- 로컬 디스크(`/local_datasets/`)에 데이터셋 압축 해제
- Config 파일의 경로를 로컬 디스크로 변경
- 학습 속도 향상 (NAS보다 로컬 디스크가 빠름)

### 2. GPU 설정

현재 설정: **GPU 1개, 64GB RAM, 8 CPUs**

더 많은 GPU가 필요하면 `train_slurm.sh` 수정:
```bash
#SBATCH --gres=gpu:2  # GPU 2개로 변경
```

### 3. 학습 시간 제한

현재 설정: **3일 (3-0)**

더 긴 시간이 필요하면:
```bash
#SBATCH -t 7-0  # 7일로 변경
```

---

## 🐛 트러블슈팅

### 문제 1: 데이터셋이 로드되지 않음

```bash
# 서버에서 압축 해제 확인
ssh aurora-g8
ls /local_datasets/ktw3389/manikin/data_manikin/BioMotionLab_NTroje/train/
```

**해결**: 압축 파일이 제대로 업로드되었는지 확인

### 문제 2: CUDA out of memory

**해결**: `manikin_config_server.json`에서 배치 사이즈 줄이기
```json
"dataloader_batch_size": 128  // 256 → 128로 감소
```

### 문제 3: Import 에러

**해결**: 서버에서 conda 환경 재설정
```bash
conda activate manikin
pip install human-body-prior pytorch3d opencv-python
```

---

## 📥 학습 완료 후 모델 다운로드

```bash
# 최고 성능 모델 다운로드
scp aurora-g8:/data/ktw3389/Manikin-Sage/Manikin/outputs/models/v2_*/manikin_v2_best.pth \
    ./Manikin/outputs/models/

# 로그 파일 다운로드
scp aurora-g8:/data/ktw3389/Manikin-Sage/logs/slurm-*.out \
    ./Manikin/outputs/logs/
```

---

## 📊 예상 학습 시간

| 데이터셋 조합 | 반복 수 | 예상 시간 (GPU 1개) |
|--------------|---------|-------------------|
| BioMotionLab만 | 10k | ~6시간 |
| BioMotionLab + CMU | 50k | ~1.5일 |
| 전체 (3개) | 100k | ~3일 |

---

## ✅ 체크리스트

학습 시작 전 확인:

- [ ] `data_manikin.tar` 생성 완료
- [ ] `body_models.tar` 생성 완료
- [ ] 서버로 파일 업로드 완료
- [ ] 서버에 conda 환경 `manikin` 설정 완료
- [ ] SLURM 스크립트 실행 권한 확인 (`chmod +x train_slurm.sh`)
- [ ] GPU 할당 확인 (`sinfo`)

학습 중 모니터링:

- [ ] SLURM 로그 확인 (매 100 iteration마다 loss 출력)
- [ ] GPU 사용률 확인 (로그에 nvidia-smi 출력)
- [ ] 디스크 공간 확인 (`df -h /local_datasets`)

학습 완료 후:

- [ ] 최고 성능 모델 다운로드
- [ ] 로그 파일 백업
- [ ] 로컬 디스크 캐시 정리 (필요시)

---

**Last Updated**: 2024-12-09
**MANIKIN Version**: V2 (Hybrid Model)
