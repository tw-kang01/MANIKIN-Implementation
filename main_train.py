"""
MANIKIN Main Training Script
Based on EgoPoser/AvatarPoser training infrastructure

Usage:
    python Manikin/main_train.py --config Manikin/configs/manikin_config.json
"""

import os
import sys
import re
import json
import math
import argparse
import random
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.manikin_dataset import MANIKINDataset
from models.model_manikin import ModelMANIKIN
from utils.manikin_loss_module import compute_mpjpe, compute_mpjve, compute_jitter


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path):
    """
    Load JSON config file with support for // comments

    Args:
        config_path: path to JSON config file

    Returns:
        dict: parsed config
    """
    with open(config_path, 'r') as f:
        content = f.read()

    # Remove // comments
    content = re.sub(r'//.*?\n', '\n', content)
    # Remove trailing commas before } or ]
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    return json.loads(content)


def setup_paths(opt):
    """Create output directories"""
    root = opt['path'].get('root', 'Manikin/outputs')
    opt['path']['root'] = root
    opt['path']['models'] = os.path.join(root, 'models')
    opt['path']['log'] = os.path.join(root, 'logs')

    os.makedirs(opt['path']['models'], exist_ok=True)
    os.makedirs(opt['path']['log'], exist_ok=True)

    return opt


def setup_logger(name, log_dir):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}_{timestamp}.txt'))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def find_last_checkpoint(models_dir):
    """Find the last checkpoint iteration number"""
    if not os.path.exists(models_dir):
        return 0, None

    checkpoints = [f for f in os.listdir(models_dir) if f.endswith('.pth') and not 'optimizer' in f]
    if not checkpoints:
        return 0, None

    # Extract iteration numbers
    iters = []
    for ckpt in checkpoints:
        try:
            iter_num = int(ckpt.replace('.pth', ''))
            iters.append(iter_num)
        except ValueError:
            continue

    if not iters:
        return 0, None

    max_iter = max(iters)
    return max_iter, os.path.join(models_dir, f'{max_iter}.pth')


# =============================================================================
# Testing Function
# =============================================================================

def test(model, test_loader, _opt, logger, current_step):
    """
    Run evaluation on test set with proper MPJPE/MPJVE metrics

    Args:
        model: ModelMANIKIN instance
        test_loader: DataLoader for test set
        opt: config dict (unused but kept for API compatibility)
        logger: logger instance
        current_step: current training iteration
    """
    model.net.eval()

    # Metric accumulators
    all_mpjpe = []          # Full body MPJPE (mm)
    all_mpjpe_elbow = []    # Elbow MPJPE (mm)
    all_mpjpe_knee = []     # Knee MPJPE (mm)
    all_mpjve = []          # Full body MPJVE (mm/s)
    all_jitter = []         # Jitter (mm/s^2)
    all_rot_error = []      # Rotation L1 error

    for _, test_data in enumerate(tqdm(test_loader, desc='Testing')):
        model.feed_data(test_data, test=True)

        # Get sequence length
        frame_length = model.sparse.shape[1]
        if frame_length <= 10:
            continue

        # Inference
        with torch.no_grad():
            pred = model.test()

        # Ground truth
        gt_poses = model.poses_gt.reshape(1, -1, 22, 6)
        gt_positions = model.joint_positions  # (1, T, 22, 3)

        # Predicted poses
        pred_poses = pred['full_body_6d']  # (1, T, 22, 6)

        # ------------------------------------------
        # Rotation Error (L1 on 6D representation)
        # ------------------------------------------
        rot_err = torch.mean(torch.abs(pred_poses - gt_poses)).item()
        all_rot_error.append(rot_err)

        # ------------------------------------------
        # MPJPE: Mean Per-Joint Position Error
        # ------------------------------------------
        if 'joint_positions' in pred:
            pred_positions = pred['joint_positions']  # (1, T, 22, 3)

            # Full body MPJPE
            mpjpe = compute_mpjpe(
                pred_positions.reshape(-1, 22, 3),
                gt_positions.reshape(-1, 22, 3)
            ) * 1000  # mm
            all_mpjpe.append(mpjpe.item())

            # MPJVE
            if pred_positions.shape[1] > 1:
                mpjve = compute_mpjve(pred_positions, gt_positions, fps=120) * 1000
                all_mpjve.append(mpjve.item())

                # Jitter
                jitter = compute_jitter(pred_positions, fps=120) * 1000
                all_jitter.append(jitter.item())
        else:
            # Use mid-joint positions (elbow, knee)
            pred_elbow = pred['elbow_pos']  # (1, T, 2, 3)
            pred_knee = pred['knee_pos']    # (1, T, 2, 3)

            gt_elbow = torch.stack([
                gt_positions[:, :, 10],  # L_ELBOW
                gt_positions[:, :, 11]   # R_ELBOW
            ], dim=2)
            gt_knee = torch.stack([
                gt_positions[:, :, 4],   # L_KNEE
                gt_positions[:, :, 5]    # R_KNEE
            ], dim=2)

            # Elbow MPJPE
            elbow_mpjpe = compute_mpjpe(
                pred_elbow.reshape(-1, 2, 3),
                gt_elbow.reshape(-1, 2, 3)
            ) * 1000
            all_mpjpe_elbow.append(elbow_mpjpe.item())

            # Knee MPJPE
            knee_mpjpe = compute_mpjpe(
                pred_knee.reshape(-1, 2, 3),
                gt_knee.reshape(-1, 2, 3)
            ) * 1000
            all_mpjpe_knee.append(knee_mpjpe.item())

            # MPJVE for mid-joints
            if pred_elbow.shape[1] > 1:
                elbow_mpjve = compute_mpjve(pred_elbow, gt_elbow, fps=120) * 1000
                knee_mpjve = compute_mpjve(pred_knee, gt_knee, fps=120) * 1000
                all_mpjve.append((elbow_mpjve.item() + knee_mpjve.item()) / 2)

    model.net.train()

    # Compute averages
    results = {}

    # Full body metrics
    if all_mpjpe:
        results['MPJPE'] = sum(all_mpjpe) / len(all_mpjpe)
    # Mid-joint metrics (fallback)
    if all_mpjpe_elbow:
        results['MPJPE_elbow'] = sum(all_mpjpe_elbow) / len(all_mpjpe_elbow)
    if all_mpjpe_knee:
        results['MPJPE_knee'] = sum(all_mpjpe_knee) / len(all_mpjpe_knee)
        results['MPJPE_mid'] = (results.get('MPJPE_elbow', 0) + results['MPJPE_knee']) / 2

    if all_mpjve:
        results['MPJVE'] = sum(all_mpjve) / len(all_mpjve)
    if all_jitter:
        results['Jitter'] = sum(all_jitter) / len(all_jitter)
    if all_rot_error:
        results['RotError'] = sum(all_rot_error) / len(all_rot_error)

    # Log results
    log_msg = f'[Test] Step {current_step} | '
    if 'MPJPE' in results:
        log_msg += f"MPJPE: {results['MPJPE']:.2f}mm | "
    if 'MPJPE_mid' in results:
        log_msg += f"MPJPE_mid: {results['MPJPE_mid']:.2f}mm | "
    if 'MPJVE' in results:
        log_msg += f"MPJVE: {results['MPJVE']:.2f}mm/s | "
    if 'Jitter' in results:
        log_msg += f"Jitter: {results['Jitter']:.2f}mm/s² | "
    if 'RotError' in results:
        log_msg += f"RotErr: {results['RotError']:.4f}"

    logger.info(log_msg)

    return results


# =============================================================================
# Main Training
# =============================================================================

def main(config_path='configs/manikin_config.json'):
    """Main training function"""

    # ----------------------------------------
    # Parse arguments
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to config JSON file')
    parser.add_argument('--resume', type=int, default=None,
                        help='Resume from specific iteration')
    args = parser.parse_args()

    # ----------------------------------------
    # Load config
    # ----------------------------------------
    opt = load_config(args.config)
    opt = setup_paths(opt)

    # ----------------------------------------
    # Find last checkpoint for resuming
    # ----------------------------------------
    init_iter, init_path = find_last_checkpoint(opt['path']['models'])
    if args.resume is not None:
        init_iter = args.resume
        init_path = os.path.join(opt['path']['models'], f'{init_iter}.pth')

    opt['path']['pretrained_netG'] = init_path
    current_step = init_iter

    # ----------------------------------------
    # Setup logger
    # ----------------------------------------
    logger = setup_logger('train', opt['path']['log'])
    logger.info(f'Config: {args.config}')
    logger.info(f'Resume from iteration: {init_iter}')

    # ----------------------------------------
    # Set random seeds
    # ----------------------------------------
    seed = opt.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Random seed: {seed}')

    # ----------------------------------------
    # Create datasets and dataloaders
    # ----------------------------------------
    logger.info('Creating datasets...')

    # Training dataset
    train_opt = opt['datasets']['train']
    train_set = MANIKINDataset(train_opt)
    train_size = int(math.ceil(len(train_set) / train_opt['dataloader_batch_size']))
    logger.info(f'Training samples: {len(train_set)}, iterations per epoch: {train_size}')

    train_loader = DataLoader(
        train_set,
        batch_size=train_opt['dataloader_batch_size'],
        shuffle=train_opt.get('dataloader_shuffle', True),
        num_workers=train_opt.get('dataloader_num_workers', 4),
        drop_last=True,
        pin_memory=True
    )

    # Test dataset
    test_opt = opt['datasets']['test']
    test_set = MANIKINDataset(test_opt)
    logger.info(f'Test samples: {len(test_set)}')

    test_loader = DataLoader(
        test_set,
        batch_size=1,  # Test one sequence at a time
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )

    # ----------------------------------------
    # Create model
    # ----------------------------------------
    logger.info('Creating model...')
    model = ModelMANIKIN(opt)
    model.init_train()
    logger.info(model.info_network())

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    logger.info('Starting training...')

    num_epochs = opt['train'].get('num_epochs', 1000)
    checkpoint_print = opt['train'].get('checkpoint_print', 100)
    checkpoint_save = opt['train'].get('checkpoint_save', 1000)
    checkpoint_test = opt['train'].get('checkpoint_test', 1000)

    for epoch in range(num_epochs):
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # ----------------------------------------
            # 1) Feed data
            # ----------------------------------------
            model.feed_data(train_data)

            # ----------------------------------------
            # 2) Optimize parameters
            # ----------------------------------------
            model.optimize_parameters(current_step)

            # ----------------------------------------
            # 3) Update learning rate
            # ----------------------------------------
            model.update_learning_rate(current_step)

            # ----------------------------------------
            # 4) Logging
            # ----------------------------------------
            if current_step % checkpoint_print == 0:
                logs = model.current_log()
                lr = model.current_learning_rate()

                message = f'<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{lr:.2e}> '
                for k, v in logs.items():
                    if isinstance(v, float):
                        message += f'{k}: {v:.4f} '
                logger.info(message)

            # ----------------------------------------
            # 5) Save checkpoint
            # ----------------------------------------
            if current_step % checkpoint_save == 0:
                logger.info(f'Saving model at iteration {current_step}...')
                model.save(current_step)

            # ----------------------------------------
            # 6) Testing
            # ----------------------------------------
            if current_step % checkpoint_test == 0:
                test(model, test_loader, opt, logger, current_step)

    # ----------------------------------------
    # Final save
    # ----------------------------------------
    logger.info('Saving final model...')
    model.save('latest')
    logger.info('Training complete.')


if __name__ == '__main__':
    main()
