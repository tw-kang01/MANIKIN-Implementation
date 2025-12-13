"""
MANIKIN Loss Module (Paper Eq. 15)

L_total = λ_ori·L_ori + λ_rot·L_rot + λ_foot·L_foot +
          λ_FK·L_FK_torso + λ_swivel·L_swivel + λ_m·L_m

where:
    - L_ori: pelvis global rotation error (L1)
    - L_rot: LOCAL rotation for 21 joints EXCLUDING pelvis (L1)
             (pelvis is already supervised by L_ori)
    - L_foot: ankle POSITION + ROTATION error (L1)
              Paper: "optimize the foot position and orientation"
    - L_FK_torso: shoulder/hip position error via FK (L1)
    - L_swivel: swivel angle (cos/sin) error (L1)
    - L_m: mid-joint (elbow/knee) position error (L1)

Default weights (from paper):
    λ_ori=0.05, λ_swivel=0.2, λ_m=0.2, all others=1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.manikin_core import JointIdx


class MANIKINLossJLM(nn.Module):
    """
    MANIKIN Loss Function for JLM Architecture (Eq. 15)

    Compatible with MANIKINModelJLM output format.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights: dict with loss weights. Default uses paper values.
        """
        super().__init__()

        # Default weights (from paper Eq. 15)
        self.weights = weights or {
            'ori': 0.05,
            'rot': 1.0,
            'foot': 1.0,
            'FK_torso': 1.0,
            'swivel': 0.2,
            'm': 0.2,
        }

    def forward(self, pred, gt_data):
        """
        Compute all losses

        Args:
            pred: dict from MANIKINModelJLM.forward()
                - full_body_6d: (B, T, 22, 6) - Predicted full body pose
                - elbow_pos: (B, T, 2, 3) - Predicted elbow positions [L, R]
                - knee_pos: (B, T, 2, 3) - Predicted knee positions [L, R]
                - shoulder_pos: (B, T, 2, 3) - Shoulder positions from FK [L, R]
                - hip_pos: (B, T, 2, 3) - Hip positions from FK [L, R]
                - pred_arm_swivel: (B, T, 4) - [L_cos, L_sin, R_cos, R_sin]
                - pred_leg_swivel: (B, T, 4) - [L_cos, L_sin, R_cos, R_sin]
                - global_orient: (B, T, 6) - Pelvis global rotation

            gt_data: dict with ground truth
                - poses_gt: (B, T, 132) or (B, T, 22, 6) - GT poses in 6D
                - joint_positions: (B, T, 22, 3) - GT joint positions
                - arm_swivel_gt: (B, 4) or (B, T, 4) - GT arm swivel [L_cos, L_sin, R_cos, R_sin]
                - leg_swivel_gt: (B, 4) or (B, T, 4) - GT leg swivel [L_cos, L_sin, R_cos, R_sin]

        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual loss values (for logging)
        """
        device = pred['full_body_6d'].device
        batch, seq_len = pred['full_body_6d'].shape[:2]

        # Parse GT
        gt_poses_6d = gt_data['poses_gt']
        if gt_poses_6d.dim() == 3 and gt_poses_6d.shape[-1] == 132:
            gt_poses_6d = gt_poses_6d.reshape(batch, seq_len, 22, 6)

        gt_positions = gt_data['joint_positions']
        if gt_positions.dim() == 3:  # (B, 22, 3) -> (B, T, 22, 3)
            gt_positions = gt_positions.unsqueeze(1).expand(-1, seq_len, -1, -1)

        gt_arm_swivel = gt_data['arm_swivel_gt']
        gt_leg_swivel = gt_data['leg_swivel_gt']

        # Expand swivel if per-sample (not per-frame)
        if gt_arm_swivel.dim() == 2:  # (B, 4) -> (B, T, 4)
            gt_arm_swivel = gt_arm_swivel.unsqueeze(1).expand(-1, seq_len, -1)
        if gt_leg_swivel.dim() == 2:
            gt_leg_swivel = gt_leg_swivel.unsqueeze(1).expand(-1, seq_len, -1)

        loss_dict = {}

        # ====================================================================
        # L_ori: Global orientation loss (pelvis)
        # ====================================================================
        pred_ori = pred['global_orient']  # (B, T, 6)
        gt_ori = gt_poses_6d[:, :, JointIdx.PELVIS]  # (B, T, 6)
        L_ori = F.l1_loss(pred_ori, gt_ori)
        loss_dict['L_ori'] = L_ori.item()

        # ====================================================================
        # L_rot: Local rotation loss (21 joints, EXCLUDING Pelvis)
        # NOTE: Pelvis is already supervised by L_ori, exclude to avoid duplication
        # ====================================================================
        pred_rot = pred['full_body_6d'][:, :, 1:]  # (B, T, 21, 6) - joints 1:22
        gt_rot = gt_poses_6d[:, :, 1:]  # (B, T, 21, 6)
        L_rot = F.l1_loss(pred_rot, gt_rot)
        loss_dict['L_rot'] = L_rot.item()

        # ====================================================================
        # L_foot: Foot (ankle) POSITION + ROTATION loss
        # Paper: "optimize the foot position and orientation"
        # pred_foot is 18D: [L_pos(3), L_rot(6), R_pos(3), R_rot(6)]
        # ====================================================================
        pred_foot = pred['pred_foot']  # (B, T, 18)

        # Rotation loss
        pred_foot_rot = torch.cat([
            pred_foot[:, :, 3:9],    # L_ankle rotation (6D)
            pred_foot[:, :, 12:18]   # R_ankle rotation (6D)
        ], dim=-1)  # (B, T, 12)
        gt_foot_rot = torch.cat([
            gt_poses_6d[:, :, JointIdx.L_ANKLE],
            gt_poses_6d[:, :, JointIdx.R_ANKLE]
        ], dim=-1)  # (B, T, 12)

        # Position loss
        pred_ankle_pos = pred['pred_ankle_pos']  # (B, T, 2, 3)
        gt_ankle_pos = torch.stack([
            gt_positions[:, :, JointIdx.L_ANKLE],
            gt_positions[:, :, JointIdx.R_ANKLE]
        ], dim=2)  # (B, T, 2, 3)

        L_foot = F.l1_loss(pred_foot_rot, gt_foot_rot) + F.l1_loss(pred_ankle_pos, gt_ankle_pos)
        loss_dict['L_foot'] = L_foot.item()
        loss_dict['L_foot_rot'] = F.l1_loss(pred_foot_rot, gt_foot_rot).item()
        loss_dict['L_foot_pos'] = F.l1_loss(pred_ankle_pos, gt_ankle_pos).item()

        # ====================================================================
        # L_FK_torso: Torso FK position loss (shoulder + hip positions)
        # ====================================================================
        # Shoulder positions
        pred_shoulder = pred['shoulder_pos']  # (B, T, 2, 3)
        gt_shoulder = torch.stack([
            gt_positions[:, :, JointIdx.L_SHOULDER],
            gt_positions[:, :, JointIdx.R_SHOULDER]
        ], dim=2)  # (B, T, 2, 3)
        L_shoulder = F.l1_loss(pred_shoulder, gt_shoulder)

        # Hip positions
        pred_hip = pred['hip_pos']  # (B, T, 2, 3)
        gt_hip = torch.stack([
            gt_positions[:, :, JointIdx.L_HIP],
            gt_positions[:, :, JointIdx.R_HIP]
        ], dim=2)  # (B, T, 2, 3)
        L_hip = F.l1_loss(pred_hip, gt_hip)

        L_FK_torso = L_shoulder + L_hip
        loss_dict['L_FK_torso'] = L_FK_torso.item()
        loss_dict['L_shoulder'] = L_shoulder.item()
        loss_dict['L_hip'] = L_hip.item()

        # ====================================================================
        # L_swivel: Swivel angle loss (cos/sin)
        # ====================================================================
        pred_arm_swivel = pred['pred_arm_swivel']  # (B, T, 4)
        pred_leg_swivel = pred['pred_leg_swivel']  # (B, T, 4)

        L_arm_swivel = F.l1_loss(pred_arm_swivel, gt_arm_swivel)
        L_leg_swivel = F.l1_loss(pred_leg_swivel, gt_leg_swivel)
        L_swivel = L_arm_swivel + L_leg_swivel

        loss_dict['L_swivel'] = L_swivel.item()
        loss_dict['L_arm_swivel'] = L_arm_swivel.item()
        loss_dict['L_leg_swivel'] = L_leg_swivel.item()

        # ====================================================================
        # L_m: Mid-joint position loss (elbow + knee)
        # ====================================================================
        pred_elbow = pred['elbow_pos']  # (B, T, 2, 3)
        gt_elbow = torch.stack([
            gt_positions[:, :, JointIdx.L_ELBOW],
            gt_positions[:, :, JointIdx.R_ELBOW]
        ], dim=2)  # (B, T, 2, 3)
        L_elbow = F.l1_loss(pred_elbow, gt_elbow)

        pred_knee = pred['knee_pos']  # (B, T, 2, 3)
        gt_knee = torch.stack([
            gt_positions[:, :, JointIdx.L_KNEE],
            gt_positions[:, :, JointIdx.R_KNEE]
        ], dim=2)  # (B, T, 2, 3)
        L_knee = F.l1_loss(pred_knee, gt_knee)

        L_m = L_elbow + L_knee
        loss_dict['L_m'] = L_m.item()
        loss_dict['L_elbow'] = L_elbow.item()
        loss_dict['L_knee'] = L_knee.item()

        # ====================================================================
        # Total Loss (Eq. 15)
        # ====================================================================
        total_loss = (
            self.weights['ori'] * L_ori +
            self.weights['rot'] * L_rot +
            self.weights['foot'] * L_foot +
            self.weights['FK_torso'] * L_FK_torso +
            self.weights['swivel'] * L_swivel +
            self.weights['m'] * L_m
        )

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def compute_mpjpe(pred_positions, gt_positions):
    """
    Compute Mean Per-Joint Position Error (MPJPE)

    Args:
        pred_positions: (batch, num_joints, 3)
        gt_positions: (batch, num_joints, 3)

    Returns:
        mpjpe: scalar (in mm if positions are in meters, multiply by 1000)
    """
    error = torch.norm(pred_positions - gt_positions, dim=-1)  # (batch, num_joints)
    mpjpe = error.mean()
    return mpjpe


def compute_pa_mpjpe(pred_positions, gt_positions):
    """
    Compute Procrustes-Aligned MPJPE (PA-MPJPE)

    Aligns predicted positions to GT using Procrustes analysis before computing MPJPE.

    Args:
        pred_positions: (batch, num_joints, 3)
        gt_positions: (batch, num_joints, 3)

    Returns:
        pa_mpjpe: scalar
    """
    batch = pred_positions.shape[0]
    total_error = 0.0

    for b in range(batch):
        pred = pred_positions[b]  # (J, 3)
        gt = gt_positions[b]      # (J, 3)

        # Center
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        gt_centered = gt - gt.mean(dim=0, keepdim=True)

        # Compute optimal rotation using SVD
        H = pred_centered.T @ gt_centered  # (3, 3)
        U, S, Vt = torch.linalg.svd(H)

        # Handle reflection
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1., 1., d], device=pred.device))

        R = Vt.T @ D @ U.T

        # Apply rotation
        pred_aligned = pred_centered @ R

        # Compute error
        error = torch.norm(pred_aligned - gt_centered, dim=-1)
        total_error += error.mean()

    pa_mpjpe = total_error / batch
    return pa_mpjpe


def compute_mpjve(pred_positions, gt_positions, fps=60):
    """
    Compute Mean Per-Joint Velocity Error (MPJVE)

    Args:
        pred_positions: (B, T, J, 3) predicted joint positions
        gt_positions: (B, T, J, 3) GT joint positions
        fps: frame rate (default 60)

    Returns:
        mpjve: scalar (in mm/s if positions are in meters, multiply by 1000)
    """
    # Compute velocity (position difference between frames)
    pred_vel = (pred_positions[:, 1:] - pred_positions[:, :-1]) * fps
    gt_vel = (gt_positions[:, 1:] - gt_positions[:, :-1]) * fps

    # L2 error per joint per frame
    vel_error = torch.norm(pred_vel - gt_vel, dim=-1)  # (B, T-1, J)
    mpjve = vel_error.mean()
    return mpjve


def compute_jitter(positions, fps=60):
    """
    Compute Jitter metric: Mean acceleration magnitude
    Lower = smoother motion

    Args:
        positions: (B, T, J, 3) joint positions
        fps: frame rate

    Returns:
        jitter: scalar (in mm/s² if positions are in meters, multiply by 1000)
    """
    # Velocity: first derivative
    vel = (positions[:, 1:] - positions[:, :-1]) * fps

    # Acceleration: second derivative
    accel = (vel[:, 1:] - vel[:, :-1]) * fps

    # Mean acceleration magnitude
    jitter = torch.norm(accel, dim=-1).mean()
    return jitter


def compute_joint_penetration(positions):
    """
    Simplified penetration check: Joint pairs that should not be too close

    Check distances between joints that shouldn't collide:
    - L_Wrist vs R_Wrist
    - L_Elbow vs R_Elbow
    - L_Knee vs R_Knee

    Args:
        positions: (B, T, J, 3) joint positions

    Returns:
        penetration_score: scalar (lower = better, 0 = no violations)
    """
    MIN_DISTANCE = 0.05  # 5cm minimum distance

    # Define joint pairs that shouldn't collide
    check_pairs = [
        (JointIdx.L_WRIST, JointIdx.R_WRIST),
        (JointIdx.L_ELBOW, JointIdx.R_ELBOW),
        (JointIdx.L_KNEE, JointIdx.R_KNEE),
    ]

    violations = 0.0
    for j1, j2 in check_pairs:
        dist = torch.norm(positions[:, :, j1] - positions[:, :, j2], dim=-1)
        violations += F.relu(MIN_DISTANCE - dist).mean()

    return violations / len(check_pairs)
