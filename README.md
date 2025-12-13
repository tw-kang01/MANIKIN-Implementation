# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation

Unofficial PyTorch implementation of **MANIKIN** (ECCV 2024).

This implementation combines neural network prediction with biomechanical inverse kinematics for accurate full-body pose estimation from sparse motion sensing (head + hands).

## Overview

MANIKIN improves upon previous sparse motion tracking methods by:
- **50x improvement** in hand position accuracy (5cm → 0.1cm)
- **7-DOF limb model** with biomechanical constraints
- **Swivel angle prediction** for natural elbow/knee positioning
- **Analytic IK solver** for guaranteed biomechanical validity

## Pipeline Architecture

```
Sparse Input (Head + Hands, 54D × 40 frames @ 120Hz)
    ↓
┌─────────────────────────────────────────────────┐
│  Neural Network (Transformer Encoder)           │
│  - 6 layers, 8 heads, 512 hidden dim           │
│  - Outputs: torso(42D), swivel(8D), foot(18D)  │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Torso Forward Kinematics (with SMPL-H betas)  │
│  - Computes shoulder/hip positions             │
│  - Propagates global orientations              │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Analytic IK Solver (Equations 7-14)           │
│  - Arm IK: shoulder → elbow → wrist            │
│  - Leg IK: hip → knee → ankle                  │
│  - Uses swivel angles for elbow/knee rotation  │
└─────────────────────────────────────────────────┘
    ↓
Full Body Pose (22 joints × 6D rotation)
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MANIKIN.git
cd MANIKIN

# Create conda environment
conda create -n manikin python=3.10
conda activate manikin

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Body Model Setup

Download SMPL-H body model from [MANO website](https://mano.is.tue.mpg.de/) and place in:
```
support_data/body_models/smplh/neutral/model.npz
```

## Data Preparation

### Using AMASS Dataset

```bash
# Prepare MANIKIN-format data from AMASS
python data/preprocessing/prepare_manikin_data.py \
    --support_dir /path/to/support_data \
    --root_dir /path/to/amass/data \
    --save_dir data_manikin
```

This generates `.pkl` files with:
- `hmd_position_global_full_gt_list`: Sparse input (head + hands)
- `rotation_local_full_gt_list`: Ground truth 22-joint rotations
- `joint_positions`: Ground truth 3D joint positions
- `arm_swivel_cos_sin`, `leg_swivel_cos_sin`: Swivel angle GT
- `bone_lengths`: Subject-specific bone lengths
- `betas`: SMPL-H body shape parameters

## Training

```bash
# Train with default config
python main_train.py --config configs/manikin_config.json

# Resume from checkpoint
python main_train.py --config configs/manikin_config.json --resume 10000
```

### Training Configuration

Key parameters in `configs/manikin_config.json`:
```json
{
  "datasets": {
    "train": {
      "window_size": 40,
      "dataloader_batch_size": 256
    }
  },
  "netG": {
    "num_layer": 6,
    "embed_dim": 512,
    "nhead": 8
  },
  "train": {
    "num_epochs": 5000,
    "G_optimizer_lr": 1e-4,
    "G_scheduler_milestones": [500, 1000, 2000, 3000, 4000],
    "lambda_weights": {
      "ori": 0.05,
      "rot": 1.0,
      "foot": 1.0,
      "FK_torso": 1.0,
      "swivel": 0.2,
      "m": 0.2
    }
  }
}
```

## Testing

```bash
# Test with trained model
python main_test.py --config configs/manikin_config.json \
    --model outputs/models/10000.pth

# Save prediction results
python main_test.py --config configs/manikin_config.json \
    --model outputs/models/10000.pth \
    --save_results --output_dir outputs/test_results
```

## Loss Function (Equation 15)

```
L_total = λ_ori·L_ori + λ_rot·L_rot + λ_foot·L_foot
        + λ_FK·L_FK_torso + λ_swivel·L_swivel + λ_m·L_m
```

| Term | Weight | Description |
|------|--------|-------------|
| L_ori | 0.05 | Pelvis global orientation |
| L_rot | 1.0 | Full body joint rotations (21 joints) |
| L_foot | 1.0 | Ankle position + rotation |
| L_FK_torso | 1.0 | Shoulder/hip positions via FK |
| L_swivel | 0.2 | Arm/leg swivel angles (cos, sin) |
| L_m | 0.2 | Elbow/knee positions |

## Project Structure

```
Manikin/
├── main_train.py           # Training script
├── main_test.py            # Testing script
├── configs/
│   ├── manikin_config.json      # Local config
│   └── manikin_config_server.json  # Server config
├── data/
│   ├── manikin_dataset.py       # Dataset class
│   └── preprocessing/           # Data preparation scripts
├── extensions/                  # Core MANIKIN modules
│   ├── manikin_model.py         # Integration model (NN + FK + IK)
│   ├── manikin_network.py       # Transformer network
│   ├── analytic_solver.py       # Analytic IK (Eq. 7-14)
│   ├── swivel_gt.py             # Swivel angle computation (Eq. 2-5)
│   ├── torso_fk.py              # Torso forward kinematics
│   └── manikin_core.py          # Quaternion utilities
├── models/
│   ├── model_base.py            # Abstract base class
│   └── model_manikin.py         # Training wrapper
├── utils/
│   ├── manikin_loss_module.py   # 6-term loss function
│   ├── rotation_utils.py        # Rotation conversions
│   └── collate.py               # Custom DataLoader collate
├── scripts/
│   └── train_slurm.sh           # SLURM cluster script
└── outputs/                     # Training outputs
    ├── models/                  # Checkpoints
    └── logs/                    # Training logs
```

## SMPL-H Joint Indices (22 joints)

```
 0: pelvis      1: left_hip     2: right_hip     3: spine1
 4: left_knee   5: right_knee   6: spine2        7: left_ankle
 8: right_ankle 9: spine3      10: left_foot    11: right_foot
12: neck       13: left_collar 14: right_collar 15: head
16: left_shoulder  17: right_shoulder  18: left_elbow   19: right_elbow
20: left_wrist     21: right_wrist
```

## Citation

If you use this code, please cite the original MANIKIN paper:

```bibtex
@inproceedings{jiang2024manikin,
  author    = {Jiang, Jiaxi and Streli, Paul and Luo, Xuejing and Gebhardt, Christoph and Holz, Christian},
  title     = {MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation},
  booktitle = {Computer Vision -- ECCV 2024: 18th European Conference, Milan, Italy, September 29--October 4, 2024, Proceedings, Part II},
  pages     = {128--146},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  year      = {2024},
  doi       = {10.1007/978-3-031-72627-9_8}
}
```

This implementation also builds upon AvatarPoser:

```bibtex
@inproceedings{jiang2022avatarposer,
  author    = {Jiang, Jiaxi and Streli, Paul and Qiu, Huajian and Fender, Andreas and Laber, Larissa and Snape, Patrick and Holz, Christian},
  title     = {AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing},
  booktitle = {Proceedings of European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```

## License

This project is for academic and research purposes only. Please refer to the original paper and SMPL-H license for usage restrictions.

## Acknowledgments

- [MANIKIN](https://siplab.org/projects/MANIKIN) - Original paper and method
- [AvatarPoser](https://github.com/eth-siplab/AvatarPoser) - Foundation codebase
- [SMPL-H](https://mano.is.tue.mpg.de/) - Body model
- [AMASS](https://amass.is.tue.mpg.de/) - Motion capture dataset
