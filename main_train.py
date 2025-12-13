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

def test(model, test_loader, opt, logger, current_step):
    """
    Run evaluation on test set

    Args:
        model: ModelMANIKIN instance
        test_loader: DataLoader for test set
        opt: config dict
        logger: logger instance
        current_step: current training iteration
    """
    model.net.eval()

    rot_error = []
    pos_error = []
    vel_error = []

    for idx, test_data in enumerate(tqdm(test_loader, desc='Testing')):
        model.feed_data(test_data, test=True)

        # Get sequence length
        frame_length = model.sparse.shape[1]
        if frame_length <= 10:
            continue

        # Inference
        pred = model.test()

        # Compute metrics
        pred_poses = pred['full_body_6d']  # (1, T, 22, 6)
        gt_poses = model.poses_gt.reshape(1, -1, 22, 6)

        pred_positions = pred.get('joint_positions', None)
        gt_positions = model.joint_positions  # (1, T, 22, 3)

        # Rotation error (degrees)
        rot_err = torch.mean(torch.abs(pred_poses - gt_poses)).item()
        rot_error.append(rot_err)

        # Position error (MPJPE in meters)
        if pred_positions is not None:
            pos_err = torch.mean(torch.sqrt(
                torch.sum((pred_positions - gt_positions) ** 2, dim=-1)
            )).item()
        else:
            # Use elbow/knee positions if full positions not available
            pred_elbow = pred['elbow_pos']
            pred_knee = pred['knee_pos']
            gt_elbow = torch.stack([
                gt_positions[:, :, 18],  # L_ELBOW
                gt_positions[:, :, 19]   # R_ELBOW
            ], dim=2)
            gt_knee = torch.stack([
                gt_positions[:, :, 4],   # L_KNEE
                gt_positions[:, :, 5]    # R_KNEE
            ], dim=2)

            pos_err = 0.5 * (
                torch.mean(torch.sqrt(torch.sum((pred_elbow - gt_elbow) ** 2, dim=-1))).item() +
                torch.mean(torch.sqrt(torch.sum((pred_knee - gt_knee) ** 2, dim=-1))).item()
            )
        pos_error.append(pos_err)

        # Velocity error (cm/s @ 120fps)
        if gt_positions.shape[1] > 1:
            gt_vel = (gt_positions[:, 1:] - gt_positions[:, :-1]) * 120  # 120fps
            if pred_positions is not None:
                pred_vel = (pred_positions[:, 1:] - pred_positions[:, :-1]) * 120
            else:
                pred_vel = gt_vel  # Skip velocity if no positions
            vel_err = torch.mean(torch.sqrt(
                torch.sum((pred_vel - gt_vel) ** 2, dim=-1)
            )).item()
            vel_error.append(vel_err)

    model.net.train()

    # Compute averages
    avg_rot = sum(rot_error) / len(rot_error) if rot_error else 0
    avg_pos = sum(pos_error) / len(pos_error) if pos_error else 0
    avg_vel = sum(vel_error) / len(vel_error) if vel_error else 0

    # Log results
    logger.info(f'Step {current_step} | '
                f'Rot Error: {avg_rot:.4f} | '
                f'Pos Error (MPJPE): {avg_pos*100:.2f} cm | '
                f'Vel Error: {avg_vel*100:.2f} cm/s')

    return {
        'rot_error': avg_rot,
        'pos_error': avg_pos,
        'vel_error': avg_vel
    }


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
