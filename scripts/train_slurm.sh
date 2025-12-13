#!/usr/bin/bash
#SBATCH -J manikin-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

# =============================================================================
# MANIKIN (ECCV 2024) Training Script for SLURM
# Neural-Analytic Hybrid Full Body Pose Estimation
#
# Usage:
#   1. Clone the repository:
#      git clone https://github.com/tw-kang01/MANIKIN-Implementation.git
#      cd MANIKIN-Implementation
#
#   2. Prepare datasets (extract to local disk for fast I/O):
#      tar -xf /data/datasets/data_manikin.tar -C /local_datasets/$USER/
#      tar -xf /data/datasets/body_models.tar -C /local_datasets/$USER/
#
#   3. Submit job:
#      sbatch scripts/train_slurm.sh
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

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Repository root (parent of scripts/)
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Log directory
LOG_DIR=$REPO_DIR/logs
mkdir -p $LOG_DIR

# Local disk for fast I/O (datasets)
LOCAL_DIR=/local_datasets/$USER
mkdir -p $LOCAL_DIR

echo "Repository directory: $REPO_DIR"
echo "Log directory: $LOG_DIR"
echo "Local data cache: $LOCAL_DIR"

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
if [ -d "$LOCAL_DIR/support_data" ]; then
    echo "✓ Body models found at $LOCAL_DIR/support_data"
elif [ -d "$LOCAL_DIR/AvatarPoser/support_data" ]; then
    echo "✓ Body models found at $LOCAL_DIR/AvatarPoser/support_data"
    # Create symlink for compatibility
    ln -sf $LOCAL_DIR/AvatarPoser/support_data $LOCAL_DIR/support_data 2>/dev/null || true
else
    echo "ERROR: Body models not found"
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
fi

# MPI_HDM05
if [ -d "$LOCAL_DIR/data_manikin/MPI_HDM05/train" ]; then
    TRAIN_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/MPI_HDM05/train/*.pkl 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/MPI_HDM05/test/*.pkl 2>/dev/null | wc -l)
    echo "✓ MPI_HDM05: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
fi

# CMU
if [ -d "$LOCAL_DIR/data_manikin/CMU/train" ]; then
    TRAIN_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/CMU/train/*.pkl 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 $LOCAL_DIR/data_manikin/CMU/test/*.pkl 2>/dev/null | wc -l)
    echo "✓ CMU: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
fi

# Body models - check multiple possible locations
SMPLH_PATH=""
if [ -f "$LOCAL_DIR/support_data/body_models/smplh/neutral/model.npz" ]; then
    SMPLH_PATH="$LOCAL_DIR/support_data/body_models/smplh/neutral/model.npz"
elif [ -f "$LOCAL_DIR/AvatarPoser/support_data/body_models/smplh/neutral/model.npz" ]; then
    SMPLH_PATH="$LOCAL_DIR/AvatarPoser/support_data/body_models/smplh/neutral/model.npz"
fi

if [ -n "$SMPLH_PATH" ]; then
    echo "✓ SMPL-H neutral model found: $SMPLH_PATH"
    SUPPORT_DIR=$(dirname $(dirname $(dirname $(dirname $SMPLH_PATH))))
else
    echo "✗ SMPL-H neutral model NOT found"
    exit 1
fi

echo "=== End Dataset Check ==="

# =============================================================================
# Step 3: Conda Environment Setup
# =============================================================================
echo "[Step 3] Setting up conda environment..."

# Try common conda paths
if [ -f "/data/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
    source /data/$USER/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

conda activate manikin

# CUDA 확인
echo "=== CUDA Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=== End CUDA Check ==="

# =============================================================================
# Step 4: Generate Runtime Config
# =============================================================================
echo "[Step 4] Generating runtime config..."

cd $REPO_DIR

# Generate runtime config with correct paths
python -c "
import json
import os

# Load base config
config_path = 'configs/manikin_config_server.json'
if not os.path.exists(config_path):
    config_path = 'configs/manikin_config.json'

with open(config_path, 'r') as f:
    content = f.read()
# Remove // comments
import re
content = re.sub(r'//.*?\n', '\n', content)
content = re.sub(r',(\s*[}\]])', r'\1', content)
config = json.loads(content)

# Update paths
LOCAL_DIR = '$LOCAL_DIR'
SUPPORT_DIR = '$SUPPORT_DIR'

config['datasets']['train']['dataroot'] = [
    f'{LOCAL_DIR}/data_manikin/BioMotionLab_NTroje/train',
    f'{LOCAL_DIR}/data_manikin/MPI_HDM05/train',
    f'{LOCAL_DIR}/data_manikin/CMU/train'
]
config['datasets']['test']['dataroot'] = [
    f'{LOCAL_DIR}/data_manikin/BioMotionLab_NTroje/test',
    f'{LOCAL_DIR}/data_manikin/MPI_HDM05/test',
    f'{LOCAL_DIR}/data_manikin/CMU/test'
]

config['support_dir'] = SUPPORT_DIR + '/'
config['body_model_path'] = f'{SUPPORT_DIR}/body_models/smplh/neutral/model.npz'

# Output path (relative to repo)
config['path']['root'] = 'outputs'

# Save runtime config
with open('configs/manikin_config_runtime.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Runtime config generated: configs/manikin_config_runtime.json')
print(f'  - support_dir: {SUPPORT_DIR}')
print(f'  - data: {LOCAL_DIR}/data_manikin/')
print(f'  - outputs: outputs/')
"

echo "Config generation complete!"

# =============================================================================
# Step 5: Training Execution
# =============================================================================
echo "[Step 5] Starting MANIKIN training..."
echo "=========================================="
echo "MANIKIN Training Configuration"
echo "=========================================="
echo "Model: MANIKINModelJLM (NN + Torso FK + Analytic IK)"
echo "Script: main_train.py"
echo "Config: configs/manikin_config_runtime.json"
echo ""
echo "Training Setup:"
echo "  - Batch Size: 256 (train), 1 (validation)"
echo "  - Checkpoints: Save every 1000 iterations"
echo "  - Validation: Every 1000 iterations"
echo ""
echo "Datasets: BioMotionLab_NTroje, MPI_HDM05, CMU"
echo "=========================================="

# Python path 설정
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# 학습 실행
echo "Starting training at $(date)"
python main_train.py --config configs/manikin_config_runtime.json

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
echo "Output directory: $REPO_DIR/outputs/"
echo "Model checkpoints: outputs/models/"
echo "Training logs: outputs/logs/"
echo ""
echo "Latest checkpoint files:"
if [ -d "$REPO_DIR/outputs/models" ]; then
    find $REPO_DIR/outputs/models -name "*.pth" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-
else
    echo "  (No checkpoint files found)"
fi

echo ""
echo "=========================================="
echo "MANIKIN Training Script Completed"
echo "Job finished at: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE
