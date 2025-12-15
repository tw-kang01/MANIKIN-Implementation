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
#
# Usage:
#   cd /data/ktw3389/repos/manikinsage
#   sbatch Manikin/scripts/train_slurm.sh
# =============================================================================

# =============================================================================
# CRITICAL: Set paths and change to working directory FIRST
# =============================================================================
WORK_DIR=/data/ktw3389/repos/manikinsage
MANIKIN_DIR=${WORK_DIR}/Manikin
LOCAL_DIR=/local_datasets/ktw3389
LOG_DIR=${WORK_DIR}/logs

# Change to working directory immediately
cd ${WORK_DIR} || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

# Create log directory
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "MANIKIN Training - SLURM Cluster"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Manikin directory: ${MANIKIN_DIR}"
echo "Local data cache: ${LOCAL_DIR}"
echo ""

hostname
nvidia-smi

# =============================================================================
# Step 1: Verify Datasets on Local Disk
# =============================================================================
echo ""
echo "[Step 1] Verifying datasets on local disk..."

# data_manikin check
if [ -d "${LOCAL_DIR}/data_manikin" ]; then
    echo "  Dataset found at ${LOCAL_DIR}/data_manikin"
else
    echo "ERROR: Dataset not found at ${LOCAL_DIR}/data_manikin"
    echo ""
    echo "Please extract datasets first:"
    echo "  tar -xf /data/datasets/data_manikin.tar -C ${LOCAL_DIR}/"
    exit 1
fi

# body_models check
SUPPORT_DIR=""
if [ -d "${LOCAL_DIR}/support_data" ]; then
    SUPPORT_DIR="${LOCAL_DIR}/support_data"
    echo "  Body models found at ${SUPPORT_DIR}"
elif [ -d "${LOCAL_DIR}/AvatarPoser/support_data" ]; then
    SUPPORT_DIR="${LOCAL_DIR}/AvatarPoser/support_data"
    echo "  Body models found at ${SUPPORT_DIR}"
else
    echo "ERROR: Body models not found"
    echo ""
    echo "Please extract body models first:"
    echo "  tar -xf /data/datasets/body_models.tar -C ${LOCAL_DIR}/"
    exit 1
fi

# Detailed dataset check
echo ""
echo "=== Dataset Details ==="

for dataset in BioMotionLab_NTroje MPI_HDM05 CMU; do
    if [ -d "${LOCAL_DIR}/data_manikin/${dataset}/train" ]; then
        TRAIN_COUNT=$(ls -1 ${LOCAL_DIR}/data_manikin/${dataset}/train/*.pkl 2>/dev/null | wc -l)
        TEST_COUNT=$(ls -1 ${LOCAL_DIR}/data_manikin/${dataset}/test/*.pkl 2>/dev/null | wc -l)
        echo "  ${dataset}: ${TRAIN_COUNT} train, ${TEST_COUNT} test files"
    fi
done

# SMPL-H model check
SMPLH_PATH="${SUPPORT_DIR}/body_models/smplh/neutral/model.npz"
if [ -f "${SMPLH_PATH}" ]; then
    echo "  SMPL-H neutral model: ${SMPLH_PATH}"
else
    echo "ERROR: SMPL-H neutral model NOT found at ${SMPLH_PATH}"
    exit 1
fi

echo "=== End Dataset Check ==="

# =============================================================================
# Step 2: Conda Environment Setup
# =============================================================================
echo ""
echo "[Step 2] Setting up conda environment..."

# Try common conda paths
if [ -f "/data/${USER}/anaconda3/etc/profile.d/conda.sh" ]; then
    source /data/${USER}/anaconda3/etc/profile.d/conda.sh
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source ${HOME}/anaconda3/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source ${HOME}/miniconda3/etc/profile.d/conda.sh
fi

conda activate manikin

# CUDA check
echo "=== CUDA Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=== End CUDA Check ==="

# =============================================================================
# Step 3: Generate Runtime Config
# =============================================================================
echo ""
echo "[Step 3] Generating runtime config..."

# Check that base config exists
BASE_CONFIG="${MANIKIN_DIR}/configs/manikin_config_server.json"
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "ERROR: Base config not found at ${BASE_CONFIG}"
    ls -la ${MANIKIN_DIR}/configs/
    exit 1
fi
echo "  Base config: ${BASE_CONFIG}"

# Generate runtime config
RUNTIME_CONFIG="${MANIKIN_DIR}/configs/manikin_config_runtime.json"

python << EOF
import json
import re
import os

# Load base config
config_path = "${BASE_CONFIG}"
print(f"Loading config from: {config_path}")

with open(config_path, 'r') as f:
    content = f.read()

# Remove // comments
content = re.sub(r'//.*?\n', '\n', content)
# Remove trailing commas
content = re.sub(r',(\s*[}\]])', r'\1', content)

config = json.loads(content)

# Update paths with actual values
LOCAL_DIR = "${LOCAL_DIR}"
SUPPORT_DIR = "${SUPPORT_DIR}"
MANIKIN_DIR = "${MANIKIN_DIR}"

# Dataset paths
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

# Support directories
config['support_dir'] = f'{SUPPORT_DIR}/'
config['body_model_path'] = f'{SUPPORT_DIR}/body_models/smplh/neutral/model.npz'

# Output path
config['path']['root'] = f'{MANIKIN_DIR}/outputs'

# Save runtime config
runtime_config_path = "${RUNTIME_CONFIG}"
with open(runtime_config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Runtime config generated: {runtime_config_path}")
print(f"  - support_dir: {SUPPORT_DIR}")
print(f"  - data: {LOCAL_DIR}/data_manikin/")
print(f"  - outputs: {MANIKIN_DIR}/outputs/")
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate runtime config"
    exit 1
fi

echo "Config generation complete!"

# =============================================================================
# Step 4: Training Execution
# =============================================================================
echo ""
echo "[Step 4] Starting MANIKIN training..."
echo "=========================================="
echo "MANIKIN Training Configuration"
echo "=========================================="
echo "Model: MANIKINModelJLM (NN + Torso FK + Analytic IK)"
echo "Script: ${MANIKIN_DIR}/main_train.py"
echo "Config: ${RUNTIME_CONFIG}"
echo ""
echo "Training Setup:"
echo "  - Batch Size: 256 (train), 1 (validation)"
echo "  - Checkpoints: Save every 1000 iterations"
echo "  - Validation: Every 1000 iterations"
echo ""
echo "Datasets: BioMotionLab_NTroje, MPI_HDM05, CMU"
echo "=========================================="

# Set Python path
export PYTHONPATH="${MANIKIN_DIR}:${WORK_DIR}:${PYTHONPATH}"

# Verify training script exists
if [ ! -f "${MANIKIN_DIR}/main_train.py" ]; then
    echo "ERROR: Training script not found at ${MANIKIN_DIR}/main_train.py"
    ls -la ${MANIKIN_DIR}/*.py
    exit 1
fi

# Create output directories
mkdir -p ${MANIKIN_DIR}/outputs/models
mkdir -p ${MANIKIN_DIR}/outputs/logs

# Run training
echo "Starting training at $(date)"
echo "Command: python ${MANIKIN_DIR}/main_train.py --config ${RUNTIME_CONFIG}"
echo ""

python ${MANIKIN_DIR}/main_train.py --config ${RUNTIME_CONFIG}

TRAIN_EXIT_CODE=$?
echo ""
echo "=========================================="
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}"
fi
echo "=========================================="

# =============================================================================
# Step 5: Results Summary
# =============================================================================
echo ""
echo "[Step 5] Training results summary..."
echo ""
echo "Output directory: ${MANIKIN_DIR}/outputs/"
echo "Model checkpoints: ${MANIKIN_DIR}/outputs/models/"
echo "Training logs: ${MANIKIN_DIR}/outputs/logs/"
echo ""
echo "Latest checkpoint files:"
if [ -d "${MANIKIN_DIR}/outputs/models" ]; then
    ls -lhtr ${MANIKIN_DIR}/outputs/models/*.pth 2>/dev/null | tail -5 || echo "  (No checkpoint files found)"
else
    echo "  (No checkpoint directory found)"
fi

echo ""
echo "=========================================="
echo "MANIKIN Training Script Completed"
echo "Job finished at: $(date)"
echo "=========================================="

exit ${TRAIN_EXIT_CODE}
