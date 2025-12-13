#!/usr/bin/bash
#SBATCH -J manikin-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o /data/ktw3389/repos/manikinsage/logs/slurm-%A.out

# =============================================================================
# MANIKIN (ECCV 2024) Training Script for SLURM
# Neural-Analytic Hybrid Full Body Pose Estimation
# =============================================================================

hostname
nvidia-smi

echo "=========================================="
echo "MANIKIN Training - SLURM Cluster"
echo "=========================================="

# =============================================================================
# Step 1: Setup Directories
# =============================================================================
echo "[Step 1] Setting up directories..."

# NAS 작업 디렉토리 (코드 실행)
WORK_DIR=/data/ktw3389/repos/manikinsage
LOG_DIR=$WORK_DIR/logs
mkdir -p $LOG_DIR

# 로컬 디스크 (빠른 I/O를 위한 데이터셋 캐시)
LOCAL_DIR=/local_datasets/ktw3389
mkdir -p $LOCAL_DIR

echo "Work directory: $WORK_DIR"
echo "Log directory: $LOG_DIR"
echo "Local cache: $LOCAL_DIR"

# =============================================================================
# Step 2: Verify Datasets on Local Disk
# =============================================================================
echo "[Step 2] Verifying datasets on local disk..."

# data_manikin 확인 (BioMotionLab, CMU, MPI_HDM05)
if [ -d "$LOCAL_DIR/data_manikin" ]; then
    echo "✓ Dataset found at $LOCAL_DIR/data_manikin"
else
    echo "ERROR: Dataset not found at $LOCAL_DIR/data_manikin"
    echo ""
    echo "Please extract datasets first:"
    echo "  tar -xf /data/datasets/data_manikin.tar -C $LOCAL_DIR/"
    echo ""
    exit 1
fi

# body_models 확인 (SMPL-H)
if [ -d "$LOCAL_DIR/AvatarPoser" ]; then
    echo "✓ Body models found at $LOCAL_DIR/AvatarPoser"
else
    echo "ERROR: Body models not found at $LOCAL_DIR/AvatarPoser"
    echo ""
    echo "Please extract body models first:"
    echo "  tar -xf /data/datasets/body_models.tar -C $LOCAL_DIR/"
    echo ""
    exit 1
fi

# 데이터셋 상세 확인
echo ""
echo "=== Dataset Details ==="

# BioMotionLab_NTroje
if [ -d "$LOCAL_DIR/data_manikin/BioMotionLab_NTroje/train" ]; then
    TRAIN_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/BioMotionLab_NTroje/train/*.pkl 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/BioMotionLab_NTroje/test/*.pkl 2>/dev/null | wc -l)
    echo "✓ BioMotionLab_NTroje: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
else
    echo "✗ BioMotionLab_NTroje NOT found"
    exit 1
fi

# MPI_HDM05
if [ -d "$LOCAL_DIR/data_manikin/MPI_HDM05/train" ]; then
    TRAIN_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/MPI_HDM05/train/*.pkl 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/MPI_HDM05/test/*.pkl 2>/dev/null | wc -l)
    echo "✓ MPI_HDM05: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
else
    echo "✗ MPI_HDM05 NOT found"
    exit 1
fi

# CMU
if [ -d "$LOCAL_DIR/data_manikin/CMU/train" ]; then
    TRAIN_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/CMU/train/*.pkl 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/CMU/test/*.pkl 2>/dev/null | wc -l)
    echo "✓ CMU: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
else
    echo "✗ CMU NOT found"
    exit 1
fi

# Body models
if [ -f "$LOCAL_DIR/AvatarPoser/support_data/body_models/smplh/neutral/model.npz" ]; then
    echo "✓ SMPL-H neutral model found"
else
    echo "✗ SMPL-H neutral model NOT found"
    echo "  Expected: $LOCAL_DIR/AvatarPoser/support_data/body_models/smplh/neutral/model.npz"
    echo "  Available body_models structure:"
    ls -la $LOCAL_DIR/AvatarPoser/support_data/body_models/ 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

echo "=== End Dataset Check ==="

# =============================================================================
# Step 3: Conda Environment Setup
# =============================================================================
echo "[Step 3] Setting up conda environment..."

source /data/ktw3389/anaconda3/etc/profile.d/conda.sh
conda activate manikin

# 필수 패키지 확인
echo "=== Checking/Installing Required Packages ==="
pip install human-body-prior --quiet
pip install pytorch3d --quiet
pip install opencv-python --quiet
echo "=== End Package Check ==="

# CUDA 확인
echo "=== CUDA Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=== End CUDA Check ==="

# =============================================================================
# Step 4: Update Config for Server Environment
# =============================================================================
echo "[Step 4] Updating config for server environment..."

cd $WORK_DIR

# config 파일 백업
cp Manikin/configs/manikin_config.json Manikin/configs/manikin_config_backup.json

# manikin_config_server.json이 있으면 사용, 없으면 기본 config 사용
if [ -f "Manikin/configs/manikin_config_server.json" ]; then
    CONFIG_FILE="Manikin/configs/manikin_config_server.json"
    echo "Using server config: $CONFIG_FILE"
else
    CONFIG_FILE="Manikin/configs/manikin_config.json"
    echo "Using default config: $CONFIG_FILE"
fi

# Python으로 config 업데이트 (데이터 경로를 로컬 디스크로 변경)
python -c "
import json

# Load config
with open('$CONFIG_FILE', 'r') as f:
    lines = f.readlines()
cleaned = []
for line in lines:
    if '//' in line:
        line = line[:line.index('//')]
    cleaned.append(line)
config = json.loads(''.join(cleaned))

# Update dataset paths to local disk
config['datasets']['train']['dataroot'] = [
    '$LOCAL_DIR/data_manikin/BioMotionLab_NTroje/train',
    '$LOCAL_DIR/data_manikin/MPI_HDM05/train',
    '$LOCAL_DIR/data_manikin/CMU/train'
]
config['datasets']['test']['dataroot'] = [
    '$LOCAL_DIR/data_manikin/BioMotionLab_NTroje/test',
    '$LOCAL_DIR/data_manikin/MPI_HDM05/test',
    '$LOCAL_DIR/data_manikin/CMU/test'
]

# Update body model path
config['body_model_path'] = '$LOCAL_DIR/AvatarPoser/support_data/body_models/smplh/neutral/model.npz'
config['support_dir'] = '$LOCAL_DIR/AvatarPoser/support_data/'

# Update training parameters for server (최신 설정 반영)
config['datasets']['train']['dataloader_num_workers'] = 8
config['datasets']['train']['dataloader_batch_size'] = 256
config['datasets']['test']['dataloader_batch_size'] = 1  # Validation batch size (must be 1 for variable-length sequences)
config['datasets']['test']['test_batch'] = 256  # Internal batching for validation (EgoPoser-style)

# Training config - use values from config file (don't override num_epochs)
# num_epochs is read from manikin_config_server.json
config['train']['checkpoint_save'] = 1000
config['train']['checkpoint_test'] = 1000  # Iteration-based validation
config['train']['checkpoint_print'] = 100

# Save updated config
with open('Manikin/configs/manikin_config_runtime.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Config updated successfully!')
print('\\n=== Training Configuration ===')
print(f'Epochs: {config[\"train\"][\"num_epochs\"]}')
print(f'Batch size (train): {config[\"datasets\"][\"train\"][\"dataloader_batch_size\"]}')
print(f'Batch size (test): {config[\"datasets\"][\"test\"][\"dataloader_batch_size\"]} (set to 1 for variable-length sequences)')
print(f'Workers: {config[\"datasets\"][\"train\"][\"dataloader_num_workers\"]}')
print(f'Checkpoint save: every {config[\"train\"][\"checkpoint_save\"]} iterations')
print(f'Validation: every {config[\"train\"][\"checkpoint_test\"]} iterations')
print(f'\\nDataset paths:')
for path in config['datasets']['train']['dataroot']:
    print(f'  - {path}')
"

echo "Config update complete!"

# =============================================================================
# Step 5: Training Execution
# =============================================================================
echo "[Step 5] Starting MANIKIN training..."
echo "=========================================="
echo "MANIKIN Training Configuration"
echo "=========================================="
echo "Model: MANIKINModelJLM (NN + Torso FK + Analytic IK)"
echo "Script: Manikin/main_train.py"
echo "Config: Manikin/configs/manikin_config_runtime.json"
echo ""
echo "Training Setup:"
echo "  - Epochs: (from config file - manikin_config_server.json)"
echo "  - Batch Size: 256 (train), 1 (validation)"
echo "  - Learning Rate: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5"
echo "  - LR Schedule: [10k, 20k, 30k] iterations"
echo "  - Checkpoints: Save every 1000 iterations"
echo "  - Validation: Every 1000 iterations with MPJPE/MPJVE"
echo "  - Note: Validation batch_size=1 to handle variable-length test sequences"
echo ""
echo "Datasets: BioMotionLab_NTroje, MPI_HDM05, CMU"
echo "=========================================="

# Python path 설정
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# 학습 실행
echo "Starting training at $(date)"
python Manikin/main_train.py --config Manikin/configs/manikin_config_runtime.json

TRAIN_EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully at $(date)"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
fi
echo "=========================================="

# =============================================================================
# Step 6: Results Summary
# =============================================================================
echo "[Step 6] Training results summary..."
echo ""
echo "Output directory: $WORK_DIR/Manikin/outputs/"
echo "Model checkpoints saved to: Manikin/outputs/models/"
echo "Training logs saved to: Manikin/outputs/logs/"
echo ""
echo "Latest checkpoint files:"
if [ -d "$WORK_DIR/Manikin/outputs/models" ]; then
    find $WORK_DIR/Manikin/outputs/models -name "*.pth" -type f -printf "%T@ %p\n" | sort -nr | head -5 | cut -d' ' -f2-
else
    echo "  (No checkpoint files found)"
fi

# =============================================================================
# Step 7: Cleanup (Optional)
# =============================================================================
echo ""
echo "[Step 7] Cleanup options..."
echo "Local cache preserved at: $LOCAL_DIR"
echo ""
echo "To free up space, manually run:"
echo "  rm -rf $LOCAL_DIR/data_manikin"
echo "  rm -rf $LOCAL_DIR/AvatarPoser"

echo ""
echo "=========================================="
echo "MANIKIN Training Script Completed"
echo "Job finished at: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE
