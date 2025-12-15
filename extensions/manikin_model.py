"""
MANIKINModelJLM - Integration Model for MANIKIN V5

Integrates:
    1. MANIKINNetworkJLM: Neural network prediction
    2. TorsoFK: Forward kinematics for torso chain
    3. AnalyticArmSolver, AnalyticLegSolver: Inverse kinematics for limbs

Key responsibility:
    - Local rotation conversion (Global → Local) is done HERE, not in solvers
    - 6D rotation conversion is done HERE
    - Full body assembly

Pipeline:
    sparse_input (B, T, 54)
        ↓
    MANIKINNetworkJLM
        ↓
    ┌───────┬───────┬───────┐
    torso   arm_    leg_    foot
    (42D)   swivel  swivel  (18D = pos + rot)
            (4D)    (4D)
        ↓
    TorsoFK (with betas) - uses 36D torso_angles
        ↓
    shoulder_pos, hip_pos, collar_global_quat
        ↓
    ┌───────┴───────┐
    Arm IK          Leg IK
    (wrist from     (ankle from
     sparse input)   NN prediction)
        ↓               ↓
    Global quaternions
        ↓
    Local rotation conversion (in THIS model)
        ↓
    6D conversion
        ↓
    Full body assembly (22 joints × 6D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manikin_core import (
    quat_identity, quat_normalize, quat_multiply, quat_conjugate,
    quat_to_sixd, sixd_to_quat, JointIdx, EPS
)
from .manikin_network import MANIKINNetworkJLM, MANIKINNetworkS
from .torso_fk import TorsoFK
from .analytic_solver import AnalyticArmSolver, AnalyticLegSolver, compute_bone_lengths_from_positions


# =============================================================================
# Helper Functions for Local Rotation Conversion
# =============================================================================

def to_local_rotation(global_quat, parent_global_quat):
    """
    Convert global rotation to local rotation relative to parent.

    local = inverse(parent_global) × global

    Args:
        global_quat: (batch, 4) - Joint's global rotation
        parent_global_quat: (batch, 4) - Parent joint's global rotation

    Returns:
        local_quat: (batch, 4) - Local rotation relative to parent
    """
    parent_inv = quat_conjugate(parent_global_quat)
    local_quat = quat_multiply(parent_inv, global_quat)
    return quat_normalize(local_quat)


# =============================================================================
# Integration Model
# =============================================================================

class MANIKINModelJLM(nn.Module):
    """
    MANIKIN Integration Model (JLM Architecture)

    Combines neural network with biomechanical corrections:
        1. Network predicts torso, swivel angles, foot rotations
        2. TorsoFK computes shoulder/hip positions
        3. Analytic IK computes arm/leg rotations from swivel angles
        4. Full body is assembled from all components

    All local rotation conversions and 6D conversions happen in this model,
    NOT in the individual solvers.
    """

    def __init__(self, body_model, config=None):
        """
        Args:
            body_model: SMPL-H body model
            config: Configuration dict with network parameters (optional)
        """
        super().__init__()

        # Parse config
        if config is None:
            config = {}

        netG = config.get('netG', {})
        net_type = netG.get('net_type', 'MANIKIN_L')  # 기본값은 L (Large)
        embed_dim = netG.get('embed_dim', 256)
        nhead = netG.get('nhead', 8)
        num_layer = netG.get('num_layer', 3)

        # Network selection
        if net_type == 'MANIKIN_S':
            # Small version: EgoPoser-based lightweight network
            self.network = MANIKINNetworkS(
                embed_dim=embed_dim,
                num_layer=num_layer,
                nhead=nhead,
            )
        else:
            # Large version (default): Token-based AlternativeST network
            feat_dim = netG.get('feat_dim', 256)
            self.network = MANIKINNetworkJLM(
                embed_dim=embed_dim,
                feat_dim=feat_dim,
                nhead=nhead,
                num_layer=num_layer
            )

        # FK/IK modules
        self.torso_fk = TorsoFK(body_model)
        self.arm_solver = AnalyticArmSolver()
        self.leg_solver = AnalyticLegSolver()

        # Store body model reference (for loss computation)
        self.body_model = body_model

        # SMPL-H kinematic tree parents
        self.register_buffer('parents', torch.tensor([
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
        ]))

    def forward(self, sparse_input, gt_data=None, betas=None, bone_lengths=None,
                use_gt_ankle=True):
        """
        Forward pass with biomechanical corrections

        Args:
            sparse_input: (batch, seq_len, 54) - Sparse input
                [0:18] Head: 6D rot + 6D rot_vel + 3D pos + 3D pos_vel
                [18:36] L_Wrist: same format
                [36:54] R_Wrist: same format
            gt_data: dict with ground truth data (for training)
                - 'trans': (batch, seq_len, 3) - Global translation
                - 'joint_positions': (batch, seq_len, 22, 3) - GT joint positions
                - 'poses_gt': (batch, seq_len, 132) - GT poses in 6D
            betas: (batch, 16) - Body shape parameters
            bone_lengths: dict with bone lengths (optional, uses neutral if None)
            use_gt_ankle: bool - Use GT ankle positions for leg IK (training mode)

        Returns:
            dict with:
                - full_body_6d: (batch, seq_len, 22, 6) - Full body pose
                - elbow_pos: (batch, seq_len, 2, 3) - Predicted elbow positions [L, R]
                - knee_pos: (batch, seq_len, 2, 3) - Predicted knee positions [L, R]
                - shoulder_pos: (batch, seq_len, 2, 3) - Shoulder positions from FK [L, R]
                - hip_pos: (batch, seq_len, 2, 3) - Hip positions from FK [L, R]
                - pred_arm_swivel: (batch, seq_len, 4) - Predicted arm swivel
                - pred_leg_swivel: (batch, seq_len, 4) - Predicted leg swivel
        """
        batch, seq_len, _ = sparse_input.shape
        device = sparse_input.device

        # Default values
        if betas is None:
            betas = torch.zeros(batch, 16, device=device)

        # =====================================================================
        # 1. Network prediction
        # =====================================================================
        net_out = self.network(sparse_input)
        # net_out: torso (B, T, 42), arm_swivel (B, T, 4), leg_swivel (B, T, 4), foot (B, T, 18)

        # =====================================================================
        # 2. Parse network output
        # =====================================================================
        torso_pred = net_out['torso']           # (B, T, 42)
        arm_swivel = net_out['arm_swivel']      # (B, T, 4) = [L_cos, L_sin, R_cos, R_sin]
        leg_swivel = net_out['leg_swivel']      # (B, T, 4) = [L_cos, L_sin, R_cos, R_sin]
        foot_pred = net_out['foot']             # (B, T, 18) = [L_ankle(9), R_ankle(9)]

        # Parse torso: pelvis(6) + 6 joints × 6D = 42D
        # torso_angles = [spine1, spine2, spine3, neck, L_collar, R_collar] × 6D
        # NOTE: L_shoulder, R_shoulder, L_hip, R_hip rotations computed by IK solver
        global_orient = torso_pred[:, :, 0:6]       # (B, T, 6)
        torso_angles = torso_pred[:, :, 6:42]       # (B, T, 36)

        # Parse foot: position(3) + rotation(6D) per ankle
        # Ankle position is predicted by NN (unlike wrist which is from sparse input)
        left_ankle_pos_pred = foot_pred[:, :, 0:3]      # (B, T, 3)
        left_ankle_6d = foot_pred[:, :, 3:9]            # (B, T, 6)
        right_ankle_pos_pred = foot_pred[:, :, 9:12]    # (B, T, 3)
        right_ankle_6d = foot_pred[:, :, 12:18]         # (B, T, 6)

        # =====================================================================
        # 3. Extract wrist info from sparse input
        # =====================================================================
        # Head [0:18]: 6D rot + 6D rot_vel + 3D pos + 3D pos_vel
        # L_Wrist [18:36]: 6D rot + 6D rot_vel + 3D pos + 3D pos_vel
        # R_Wrist [36:54]: 6D rot + 6D rot_vel + 3D pos + 3D pos_vel

        left_wrist_6d = sparse_input[:, :, 18:24]       # (B, T, 6)
        left_wrist_pos = sparse_input[:, :, 18+6+6:18+6+6+3]  # (B, T, 3)

        right_wrist_6d = sparse_input[:, :, 36:42]      # (B, T, 6)
        right_wrist_pos = sparse_input[:, :, 36+6+6:36+6+6+3]  # (B, T, 3)

        # Convert wrist 6D to quaternion
        left_wrist_quat = sixd_to_quat(left_wrist_6d.reshape(-1, 6)).reshape(batch, seq_len, 4)
        right_wrist_quat = sixd_to_quat(right_wrist_6d.reshape(-1, 6)).reshape(batch, seq_len, 4)

        # =====================================================================
        # 4. Get translation (from GT or zero)
        # =====================================================================
        if gt_data is not None and 'trans' in gt_data:
            trans = gt_data['trans']  # (B, T, 3)
        else:
            trans = torch.zeros(batch, seq_len, 3, device=device)

        # =====================================================================
        # 5. Compute bone_lengths from TorsoFK (betas-dependent)
        # Following AvatarPoser's fk_module pattern:
        #   body_pose = body_model(**{'pose_body':..., 'root_orient':...})
        #   joint_position = body_pose.Jtr
        # =====================================================================
        if bone_lengths is None:
            # Use first frame to compute bone lengths (same betas → same bone lengths)
            fk_out_init = self.torso_fk(
                global_orient_6d=global_orient[:, 0],      # (B, 6)
                torso_angles_6d=torso_angles[:, 0],        # (B, 36)
                trans=trans[:, 0],                          # (B, 3)
                betas=betas                                 # (B, 16)
            )
            # Compute bone lengths from joint positions (betas-dependent!)
            bone_lengths = compute_bone_lengths_from_positions(fk_out_init['joint_positions'])
            # bone_lengths: dict with (B,) tensors

        # =====================================================================
        # 6. Process each frame
        # =====================================================================
        # Initialize outputs
        full_body_6d = torch.zeros(batch, seq_len, 22, 6, device=device)
        elbow_pos_all = torch.zeros(batch, seq_len, 2, 3, device=device)
        knee_pos_all = torch.zeros(batch, seq_len, 2, 3, device=device)
        shoulder_pos_all = torch.zeros(batch, seq_len, 2, 3, device=device)
        hip_pos_all = torch.zeros(batch, seq_len, 2, 3, device=device)

        for t in range(seq_len):
            # -----------------------------------------------------------------
            # 5.1 TorsoFK
            # -----------------------------------------------------------------
            fk_out = self.torso_fk(
                global_orient_6d=global_orient[:, t],      # (B, 6)
                torso_angles_6d=torso_angles[:, t],        # (B, 36)
                trans=trans[:, t],                          # (B, 3)
                betas=betas                                 # (B, 16)
            )

            # Extract positions
            left_shoulder_pos = fk_out['left_shoulder_pos']    # (B, 3)
            right_shoulder_pos = fk_out['right_shoulder_pos']  # (B, 3)
            left_hip_pos = fk_out['left_hip_pos']              # (B, 3)
            right_hip_pos = fk_out['right_hip_pos']            # (B, 3)

            # Store positions
            shoulder_pos_all[:, t, 0] = left_shoulder_pos
            shoulder_pos_all[:, t, 1] = right_shoulder_pos
            hip_pos_all[:, t, 0] = left_hip_pos
            hip_pos_all[:, t, 1] = right_hip_pos

            # Get collar global quaternions for local rotation conversion
            global_quats = fk_out['global_quats']  # (B, 22, 4)
            left_collar_global_quat = global_quats[:, JointIdx.L_COLLAR]   # (B, 4)
            right_collar_global_quat = global_quats[:, JointIdx.R_COLLAR]  # (B, 4)
            pelvis_global_quat = global_quats[:, JointIdx.PELVIS]          # (B, 4)

            # -----------------------------------------------------------------
            # 5.2 Arm IK
            # -----------------------------------------------------------------
            # Parse swivel angles
            left_arm_cos = arm_swivel[:, t, 0]   # (B,)
            left_arm_sin = arm_swivel[:, t, 1]   # (B,)
            right_arm_cos = arm_swivel[:, t, 2]  # (B,)
            right_arm_sin = arm_swivel[:, t, 3]  # (B,)

            # Left arm IK
            left_arm_out = self.arm_solver(
                p_shoulder=left_shoulder_pos,
                p_wrist=left_wrist_pos[:, t],
                wrist_quat=left_wrist_quat[:, t],
                cos_phi=left_arm_cos,
                sin_phi=left_arm_sin,
                L1=bone_lengths['left_humerus'],
                L2=bone_lengths['left_radius'],
                side='left'
            )

            # Right arm IK
            right_arm_out = self.arm_solver(
                p_shoulder=right_shoulder_pos,
                p_wrist=right_wrist_pos[:, t],
                wrist_quat=right_wrist_quat[:, t],
                cos_phi=right_arm_cos,
                sin_phi=right_arm_sin,
                L1=bone_lengths['right_humerus'],
                L2=bone_lengths['right_radius'],
                side='right'
            )

            # Store elbow positions
            elbow_pos_all[:, t, 0] = left_arm_out['elbow_pos']
            elbow_pos_all[:, t, 1] = right_arm_out['elbow_pos']

            # -----------------------------------------------------------------
            # 5.3 Leg IK
            # -----------------------------------------------------------------
            # Parse swivel angles
            left_leg_cos = leg_swivel[:, t, 0]   # (B,)
            left_leg_sin = leg_swivel[:, t, 1]   # (B,)
            right_leg_cos = leg_swivel[:, t, 2]  # (B,)
            right_leg_sin = leg_swivel[:, t, 3]  # (B,)

            # Get ankle positions:
            # - Training with GT: use GT for supervision
            # - Training/Inference: use NN prediction (always available)
            if use_gt_ankle and gt_data is not None and 'joint_positions' in gt_data:
                # Use GT ankle position for debugging / ablation
                left_ankle_pos = gt_data['joint_positions'][:, t, JointIdx.L_ANKLE]
                right_ankle_pos = gt_data['joint_positions'][:, t, JointIdx.R_ANKLE]
            else:
                # Use NN-predicted ankle position (normal operation)
                left_ankle_pos = left_ankle_pos_pred[:, t]      # (B, 3)
                right_ankle_pos = right_ankle_pos_pred[:, t]    # (B, 3)

            # Left leg IK
            left_leg_out = self.leg_solver(
                p_hip=left_hip_pos,
                p_ankle=left_ankle_pos,
                cos_phi=left_leg_cos,
                sin_phi=left_leg_sin,
                L1=bone_lengths['left_femur'],
                L2=bone_lengths['left_tibia'],
                side='left'
            )

            # Right leg IK
            right_leg_out = self.leg_solver(
                p_hip=right_hip_pos,
                p_ankle=right_ankle_pos,
                cos_phi=right_leg_cos,
                sin_phi=right_leg_sin,
                L1=bone_lengths['right_femur'],
                L2=bone_lengths['right_tibia'],
                side='right'
            )

            # Store knee positions
            knee_pos_all[:, t, 0] = left_leg_out['knee_pos']
            knee_pos_all[:, t, 1] = right_leg_out['knee_pos']

            # -----------------------------------------------------------------
            # 5.4 Local Rotation Conversion (done HERE, not in solvers!)
            # -----------------------------------------------------------------
            # Arm: shoulder local = shoulder_global × inverse(collar_global)
            left_shoulder_local = to_local_rotation(
                left_arm_out['shoulder_quat'], left_collar_global_quat
            )
            right_shoulder_local = to_local_rotation(
                right_arm_out['shoulder_quat'], right_collar_global_quat
            )

            # Elbow is already local (relative to shoulder) from solver
            left_elbow_local = left_arm_out['elbow_quat']
            right_elbow_local = right_arm_out['elbow_quat']

            # Wrist local = wrist_global × inverse(elbow_global)
            left_wrist_local = to_local_rotation(
                left_wrist_quat[:, t], left_arm_out['elbow_global_quat']
            )
            right_wrist_local = to_local_rotation(
                right_wrist_quat[:, t], right_arm_out['elbow_global_quat']
            )

            # Leg: hip local = hip_global × inverse(pelvis_global)
            left_hip_local = to_local_rotation(
                left_leg_out['hip_quat'], pelvis_global_quat
            )
            right_hip_local = to_local_rotation(
                right_leg_out['hip_quat'], pelvis_global_quat
            )

            # Knee is already local (relative to hip) from solver
            left_knee_local = left_leg_out['knee_quat']
            right_knee_local = right_leg_out['knee_quat']

            # -----------------------------------------------------------------
            # 5.5 Convert to 6D
            # -----------------------------------------------------------------
            left_shoulder_6d = quat_to_sixd(left_shoulder_local)
            right_shoulder_6d = quat_to_sixd(right_shoulder_local)
            left_elbow_6d = quat_to_sixd(left_elbow_local)
            right_elbow_6d = quat_to_sixd(right_elbow_local)
            left_wrist_6d_out = quat_to_sixd(left_wrist_local)
            right_wrist_6d_out = quat_to_sixd(right_wrist_local)
            left_hip_6d = quat_to_sixd(left_hip_local)
            right_hip_6d = quat_to_sixd(right_hip_local)
            left_knee_6d = quat_to_sixd(left_knee_local)
            right_knee_6d = quat_to_sixd(right_knee_local)

            # -----------------------------------------------------------------
            # 5.6 Assemble full body
            # -----------------------------------------------------------------
            # Torso (from network)
            full_body_6d[:, t, JointIdx.PELVIS] = global_orient[:, t]
            full_body_6d[:, t, JointIdx.SPINE1] = torso_angles[:, t, 0:6]
            full_body_6d[:, t, JointIdx.SPINE2] = torso_angles[:, t, 6:12]
            full_body_6d[:, t, JointIdx.SPINE3] = torso_angles[:, t, 12:18]
            full_body_6d[:, t, JointIdx.NECK] = torso_angles[:, t, 18:24]
            full_body_6d[:, t, JointIdx.L_COLLAR] = torso_angles[:, t, 24:30]
            full_body_6d[:, t, JointIdx.R_COLLAR] = torso_angles[:, t, 30:36]

            # Head (from sparse input - convert GLOBAL → LOCAL)
            head_global_quat = sixd_to_quat(sparse_input[:, t, 0:6])
            neck_global_quat = global_quats[:, JointIdx.NECK]
            head_local_quat = to_local_rotation(head_global_quat, neck_global_quat)
            head_6d = quat_to_sixd(head_local_quat)
            full_body_6d[:, t, JointIdx.HEAD] = head_6d

            # Arms (from IK)
            full_body_6d[:, t, JointIdx.L_SHOULDER] = left_shoulder_6d
            full_body_6d[:, t, JointIdx.R_SHOULDER] = right_shoulder_6d
            full_body_6d[:, t, JointIdx.L_ELBOW] = left_elbow_6d
            full_body_6d[:, t, JointIdx.R_ELBOW] = right_elbow_6d
            full_body_6d[:, t, JointIdx.L_WRIST] = left_wrist_6d_out
            full_body_6d[:, t, JointIdx.R_WRIST] = right_wrist_6d_out

            # Legs (from IK)
            full_body_6d[:, t, JointIdx.L_HIP] = left_hip_6d
            full_body_6d[:, t, JointIdx.R_HIP] = right_hip_6d
            full_body_6d[:, t, JointIdx.L_KNEE] = left_knee_6d
            full_body_6d[:, t, JointIdx.R_KNEE] = right_knee_6d

            # Feet (from network)
            full_body_6d[:, t, JointIdx.L_ANKLE] = left_ankle_6d[:, t]
            full_body_6d[:, t, JointIdx.R_ANKLE] = right_ankle_6d[:, t]

        # Stack predicted ankle positions for loss computation
        pred_ankle_pos = torch.stack([
            left_ankle_pos_pred,
            right_ankle_pos_pred
        ], dim=2)  # (B, T, 2, 3)

        return {
            'full_body_6d': full_body_6d,           # (B, T, 22, 6)
            'elbow_pos': elbow_pos_all,             # (B, T, 2, 3)
            'knee_pos': knee_pos_all,               # (B, T, 2, 3)
            'shoulder_pos': shoulder_pos_all,       # (B, T, 2, 3)
            'hip_pos': hip_pos_all,                 # (B, T, 2, 3)
            'pred_arm_swivel': arm_swivel,          # (B, T, 4)
            'pred_leg_swivel': leg_swivel,          # (B, T, 4)
            'pred_foot': foot_pred,                 # (B, T, 18) - pos + rot per ankle
            'pred_ankle_pos': pred_ankle_pos,       # (B, T, 2, 3) - for ankle position loss
            'global_orient': global_orient,         # (B, T, 6)
        }

    def get_bone_lengths(self):
        """Get bone lengths from TorsoFK"""
        return self.torso_fk.bone_lengths
