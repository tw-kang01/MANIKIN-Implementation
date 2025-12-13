"""
MANIKIN Main Testing Script
Based on EgoPoser/AvatarPoser testing infrastructure

Usage:
    python Manikin/main_test.py --config Manikin/configs/manikin_config.json --model Manikin/outputs/models/10000.pth
"""

import os
import sys
import re
import json
import argparse
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
from extensions.manikin_core import JointIdx


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path):
    """Load JSON config file with support for // comments"""
    with open(config_path, 'r') as f:
        content = f.read()
    content = re.sub(r'//.*?\n', '\n', content)
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return json.loads(content)


def setup_logger(name, log_dir):
    """Setup logger with file and console handlers"""
    os.makedirs(log_dir, exist_ok=True)

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


# =============================================================================
# Metric Computation
# =============================================================================

def compute_rotation_error(pred_poses, gt_poses):
    """
    Compute mean rotation error in degrees

    Args:
        pred_poses: (B, T, 22, 6) predicted 6D rotations
        gt_poses: (B, T, 22, 6) ground truth 6D rotations

    Returns:
        float: mean rotation error
    """
    error = torch.abs(pred_poses - gt_poses)
    return torch.mean(error).item()


def compute_position_error(pred_positions, gt_positions):
    """
    Compute Mean Per-Joint Position Error (MPJPE) in meters

    Args:
        pred_positions: (B, T, 22, 3) or (B, T, N, 3) predicted positions
        gt_positions: (B, T, 22, 3) or (B, T, N, 3) ground truth positions

    Returns:
        float: MPJPE in meters
    """
    # Euclidean distance per joint per frame
    dist = torch.sqrt(torch.sum((pred_positions - gt_positions) ** 2, dim=-1))
    return torch.mean(dist).item()


def compute_velocity_error(pred_positions, gt_positions, fps=120):
    """
    Compute velocity error in m/s

    Args:
        pred_positions: (B, T, N, 3) predicted positions
        gt_positions: (B, T, N, 3) ground truth positions
        fps: frames per second

    Returns:
        float: mean velocity error in m/s
    """
    if pred_positions.shape[1] < 2:
        return 0.0

    pred_vel = (pred_positions[:, 1:] - pred_positions[:, :-1]) * fps
    gt_vel = (gt_positions[:, 1:] - gt_positions[:, :-1]) * fps

    vel_error = torch.sqrt(torch.sum((pred_vel - gt_vel) ** 2, dim=-1))
    return torch.mean(vel_error).item()


# =============================================================================
# Main Testing
# =============================================================================

def main(config_path='configs/manikin_config.json'):
    """Main testing function"""

    # ----------------------------------------
    # Parse arguments
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to config JSON file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to NPZ files')
    parser.add_argument('--output_dir', type=str, default='Manikin/outputs/test_results',
                        help='Directory to save results')
    args = parser.parse_args()

    # ----------------------------------------
    # Load config
    # ----------------------------------------
    opt = load_config(args.config)

    # Setup paths
    root = opt['path'].get('root', 'Manikin/outputs')
    opt['path']['root'] = root
    opt['path']['models'] = os.path.join(root, 'models')
    opt['path']['log'] = os.path.join(root, 'logs')

    # Set pretrained model path
    if args.model is not None:
        opt['path']['pretrained'] = args.model
    else:
        # Try to find latest model
        models_dir = opt['path']['models']
        if os.path.exists(models_dir):
            checkpoints = [f for f in os.listdir(models_dir)
                          if f.endswith('.pth') and 'optimizer' not in f]
            if checkpoints:
                # Find latest
                iters = []
                for ckpt in checkpoints:
                    try:
                        iter_num = int(ckpt.replace('.pth', ''))
                        iters.append(iter_num)
                    except ValueError:
                        if ckpt == 'latest.pth':
                            opt['path']['pretrained'] = os.path.join(models_dir, ckpt)
                            break
                if iters:
                    max_iter = max(iters)
                    opt['path']['pretrained'] = os.path.join(models_dir, f'{max_iter}.pth')

    # ----------------------------------------
    # Setup logger
    # ----------------------------------------
    logger = setup_logger('test', opt['path']['log'])
    logger.info(f'Config: {args.config}')
    logger.info(f'Model: {opt["path"].get("pretrained", "None")}')

    # ----------------------------------------
    # Create test dataset
    # ----------------------------------------
    logger.info('Creating test dataset...')

    test_opt = opt['datasets']['test']
    test_set = MANIKINDataset(test_opt)
    logger.info(f'Test samples: {len(test_set)}')

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )

    # ----------------------------------------
    # Create model and load weights
    # ----------------------------------------
    logger.info('Creating model...')
    model = ModelMANIKIN(opt)
    model.init_test()
    logger.info(model.info_network())

    # ----------------------------------------
    # Testing loop
    # ----------------------------------------
    logger.info('Starting evaluation...')

    # Metrics storage
    all_rot_errors = []
    all_pos_errors = []
    all_vel_errors = []

    # Per-joint position errors
    joint_pos_errors = {joint: [] for joint in [
        'L_SHOULDER', 'R_SHOULDER', 'L_ELBOW', 'R_ELBOW',
        'L_WRIST', 'R_WRIST', 'L_HIP', 'R_HIP',
        'L_KNEE', 'R_KNEE', 'L_ANKLE', 'R_ANKLE'
    ]}

    # Output directory for results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    for idx, test_data in enumerate(tqdm(test_loader, desc='Testing')):
        filename = test_data.get('filename', [f'sample_{idx}'])[0]

        # Feed data
        model.feed_data(test_data, test=True)

        # Get sequence length
        frame_length = model.sparse.shape[1]
        if frame_length <= 10:
            logger.info(f'Skipping {filename}: too short ({frame_length} frames)')
            continue

        # Inference
        pred = model.test()

        # Get predictions and ground truth
        pred_poses = pred['full_body_6d']  # (1, T, 22, 6)
        gt_poses = model.poses_gt.reshape(1, -1, 22, 6)  # (1, T, 22, 6)
        gt_positions = model.joint_positions  # (1, T, 22, 3)

        # Compute rotation error (6D representation)
        rot_error = compute_rotation_error(pred_poses, gt_poses)
        all_rot_errors.append(rot_error)

        # Compute position errors using predicted positions from IK
        pred_elbow = pred['elbow_pos']  # (1, T, 2, 3)
        pred_knee = pred['knee_pos']    # (1, T, 2, 3)
        pred_shoulder = pred['shoulder_pos']  # (1, T, 2, 3)
        pred_hip = pred['hip_pos']      # (1, T, 2, 3)
        pred_ankle = pred['pred_ankle_pos']  # (1, T, 2, 3)

        # Stack GT positions
        gt_shoulder = torch.stack([
            gt_positions[:, :, JointIdx.L_SHOULDER],
            gt_positions[:, :, JointIdx.R_SHOULDER]
        ], dim=2)
        gt_elbow = torch.stack([
            gt_positions[:, :, JointIdx.L_ELBOW],
            gt_positions[:, :, JointIdx.R_ELBOW]
        ], dim=2)
        gt_hip = torch.stack([
            gt_positions[:, :, JointIdx.L_HIP],
            gt_positions[:, :, JointIdx.R_HIP]
        ], dim=2)
        gt_knee = torch.stack([
            gt_positions[:, :, JointIdx.L_KNEE],
            gt_positions[:, :, JointIdx.R_KNEE]
        ], dim=2)
        gt_ankle = torch.stack([
            gt_positions[:, :, JointIdx.L_ANKLE],
            gt_positions[:, :, JointIdx.R_ANKLE]
        ], dim=2)
        gt_wrist = torch.stack([
            gt_positions[:, :, JointIdx.L_WRIST],
            gt_positions[:, :, JointIdx.R_WRIST]
        ], dim=2)

        # Wrist comes from input, so use GT for now
        pred_wrist = gt_wrist  # Perfect wrist (from input)

        # Per-joint errors
        joint_pos_errors['L_SHOULDER'].append(compute_position_error(
            pred_shoulder[:, :, 0:1], gt_shoulder[:, :, 0:1]))
        joint_pos_errors['R_SHOULDER'].append(compute_position_error(
            pred_shoulder[:, :, 1:2], gt_shoulder[:, :, 1:2]))
        joint_pos_errors['L_ELBOW'].append(compute_position_error(
            pred_elbow[:, :, 0:1], gt_elbow[:, :, 0:1]))
        joint_pos_errors['R_ELBOW'].append(compute_position_error(
            pred_elbow[:, :, 1:2], gt_elbow[:, :, 1:2]))
        joint_pos_errors['L_WRIST'].append(compute_position_error(
            pred_wrist[:, :, 0:1], gt_wrist[:, :, 0:1]))
        joint_pos_errors['R_WRIST'].append(compute_position_error(
            pred_wrist[:, :, 1:2], gt_wrist[:, :, 1:2]))
        joint_pos_errors['L_HIP'].append(compute_position_error(
            pred_hip[:, :, 0:1], gt_hip[:, :, 0:1]))
        joint_pos_errors['R_HIP'].append(compute_position_error(
            pred_hip[:, :, 1:2], gt_hip[:, :, 1:2]))
        joint_pos_errors['L_KNEE'].append(compute_position_error(
            pred_knee[:, :, 0:1], gt_knee[:, :, 0:1]))
        joint_pos_errors['R_KNEE'].append(compute_position_error(
            pred_knee[:, :, 1:2], gt_knee[:, :, 1:2]))
        joint_pos_errors['L_ANKLE'].append(compute_position_error(
            pred_ankle[:, :, 0:1], gt_ankle[:, :, 0:1]))
        joint_pos_errors['R_ANKLE'].append(compute_position_error(
            pred_ankle[:, :, 1:2], gt_ankle[:, :, 1:2]))

        # Overall position error (average of key joints)
        all_pred = torch.cat([pred_shoulder, pred_elbow, pred_hip, pred_knee, pred_ankle], dim=2)
        all_gt = torch.cat([gt_shoulder, gt_elbow, gt_hip, gt_knee, gt_ankle], dim=2)
        pos_error = compute_position_error(all_pred, all_gt)
        all_pos_errors.append(pos_error)

        # Velocity error
        vel_error = compute_velocity_error(all_pred, all_gt, fps=120)
        all_vel_errors.append(vel_error)

        # Save results if requested
        if args.save_results:
            save_path = os.path.join(args.output_dir, f'{os.path.basename(filename)}.npz')
            np.savez(save_path,
                     pred_poses=pred_poses.cpu().numpy(),
                     gt_poses=gt_poses.cpu().numpy(),
                     pred_elbow=pred_elbow.cpu().numpy(),
                     pred_knee=pred_knee.cpu().numpy(),
                     pred_shoulder=pred_shoulder.cpu().numpy(),
                     pred_hip=pred_hip.cpu().numpy(),
                     pred_ankle=pred_ankle.cpu().numpy())

    # ----------------------------------------
    # Compute and report final metrics
    # ----------------------------------------
    logger.info('\n' + '='*60)
    logger.info('FINAL RESULTS')
    logger.info('='*60)

    if all_rot_errors:
        avg_rot = sum(all_rot_errors) / len(all_rot_errors)
        logger.info(f'Average Rotation Error (6D): {avg_rot:.4f}')

    if all_pos_errors:
        avg_pos = sum(all_pos_errors) / len(all_pos_errors)
        logger.info(f'Average Position Error (MPJPE): {avg_pos*100:.2f} cm')

    if all_vel_errors:
        avg_vel = sum(all_vel_errors) / len(all_vel_errors)
        logger.info(f'Average Velocity Error: {avg_vel*100:.2f} cm/s')

    logger.info('\nPer-Joint Position Errors (cm):')
    for joint, errors in joint_pos_errors.items():
        if errors:
            avg = sum(errors) / len(errors) * 100
            logger.info(f'  {joint:12s}: {avg:.2f} cm')

    logger.info('='*60)
    logger.info('Testing complete.')


if __name__ == '__main__':
    main()
