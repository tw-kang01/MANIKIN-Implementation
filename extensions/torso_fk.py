"""
MANIKIN Torso Forward Kinematics (Quaternion-based)
ECCV 2024 MANIKIN 논문 Eq.1 구현

FK Chain (SMPL-H):
Pelvis(0) → Spine1(3) → Spine2(6) → Spine3(9) → Neck(12) → Head(15)
                                              → L_Collar(13) → L_Shoulder(16)
                                              → R_Collar(14) → R_Shoulder(17)
         → L_Hip(1) → L_Knee(4) → L_Ankle(7)
         → R_Hip(2) → R_Knee(5) → R_Ankle(8)

Input: global_orient_6d (6D) + torso_angles_6d (36D)
       torso_angles = [spine1, spine2, spine3, neck, L_collar, R_collar] × 6D

       REMOVED: L_shoulder, R_shoulder, L_hip, R_hip
       → Shoulder/Hip ROTATIONS are computed by IK solver, not NN
       → Torso FK only needs parent chain to compute shoulder/hip POSITIONS

V5 Update: Collar joints (13, 14) are now included in torso_angles
           to properly propagate rotation to shoulder positions.

V6 Update: Removed analytical FK (~27mm error with neutral body).
           Now uses body model directly with betas for accurate FK (~0.0004mm error).

V7 Update: Added L_shoulder, R_shoulder, L_hip, R_hip to torso_angles
           for complete L_FK^torso loss (Eq.1).

V8 Update: Removed L_shoulder, R_shoulder, L_hip, R_hip from torso_angles.
           These rotations are computed by IK solver from positions + swivel.
           Torso FK computes POSITIONS only (needs parent chain rotations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manikin_core import (
    quat_identity, quat_normalize, quat_multiply, quat_rotate_vector,
    sixd_to_quat, quat_to_matrix, quat_from_matrix, JointIdx, EPS
)


class TorsoFK(nn.Module):
    """
    Torso Forward Kinematics using SMPL-H body model directly
    
    논문 Eq.1: L_FK^torso = ||FK(θ_torso^pred) - FK(θ_torso^gt)||₁
    
    Uses body model directly with betas for accurate joint positions (~0.0004mm error).
    """
    
    def __init__(self, body_model):
        """
        Args:
            body_model: SMPL-H body model
        """
        super().__init__()
        self.body_model = body_model
        
        # Extract bone lengths from neutral T-pose (for Analytic IK solver)
        self._init_bone_lengths()
    
    def _init_bone_lengths(self):
        """Extract bone lengths from neutral T-pose for IK solver"""
        with torch.no_grad():
            # Get T-pose joint positions
            batch_size = 1
            
            # Get device from body model buffers or default to CPU
            try:
                device = next(self.body_model.buffers()).device
            except StopIteration:
                try:
                    device = next(self.body_model.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
            
            # Zero pose = T-pose
            body_output = self.body_model(
                root_orient=torch.zeros(batch_size, 3, device=device),
                pose_body=torch.zeros(batch_size, 63, device=device),
                betas=torch.zeros(batch_size, 16, device=device),
            )
            
            joints = body_output.Jtr[0, :22]  # (22, 3)
            
            # Compute relative offsets
            offsets = torch.zeros(22, 3, device=device)
            parents = [
                -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
            ]
            for i in range(1, 22):
                offsets[i] = joints[i] - joints[parents[i]]
            
            self.register_buffer('bone_offsets', offsets)
            
            # Store bone lengths for IK
            bone_lengths = {
                'left_humerus': torch.norm(offsets[18]).item(),   # L_Elbow - L_Shoulder
                'left_radius': torch.norm(offsets[20]).item(),    # L_Wrist - L_Elbow
                'right_humerus': torch.norm(offsets[19]).item(),  # R_Elbow - R_Shoulder
                'right_radius': torch.norm(offsets[21]).item(),   # R_Wrist - R_Elbow
                'left_femur': torch.norm(offsets[4]).item(),      # L_Knee - L_Hip
                'left_tibia': torch.norm(offsets[7]).item(),      # L_Ankle - L_Knee
                'right_femur': torch.norm(offsets[5]).item(),     # R_Knee - R_Hip
                'right_tibia': torch.norm(offsets[8]).item(),     # R_Ankle - R_Knee
            }
            self.bone_lengths = bone_lengths
    
    def forward(self, global_orient_6d, torso_angles_6d, trans=None, betas=None):
        """
        Forward kinematics using SMPL-H body model directly

        MANIKIN Pipeline: Torso FK computes Shoulder & Hip POSITIONS for Analytic IK
        - NN predicts torso (spine, collar) rotations + swivel angles
        - Torso FK computes shoulder/hip positions from parent chain
        - IK solver computes shoulder/hip rotations from positions + swivel

        Args:
            global_orient_6d: (batch, 6) - Pelvis global rotation
            torso_angles_6d: (batch, 36) - 6 torso joints × 6D
                             [spine1, spine2, spine3, neck, L_collar, R_collar] × 6D
            trans: (batch, 3) - Global translation (optional)
            betas: (batch, 16) - Body shape parameters (required for accurate FK)

        Returns:
            dict with:
                - joint_positions: (batch, 22, 3) - All joint positions
                - left_shoulder_pos: (batch, 3) - For Arm IK
                - right_shoulder_pos: (batch, 3) - For Arm IK
                - left_hip_pos: (batch, 3) - For Leg IK
                - right_hip_pos: (batch, 3) - For Leg IK
                - neck_quat: (batch, 4) - Neck global orientation (for head local)
                - left_collar_quat: (batch, 4) - L_Collar global (for shoulder local conversion)
                - right_collar_quat: (batch, 4) - R_Collar global (for shoulder local conversion)
                - pelvis_quat: (batch, 4) - Pelvis global (for hip local conversion)
                - global_quats: (batch, 22, 4) - All global quaternions
        """
        batch = global_orient_6d.shape[0]
        device = global_orient_6d.device

        if trans is None:
            trans = torch.zeros(batch, 3, device=device)
        if betas is None:
            betas = torch.zeros(batch, 16, device=device)

        # Build full pose (22 joints × 3 axis-angle)
        full_pose_aa = torch.zeros(batch, 22, 3, device=device)

        # Global orient
        global_quat = sixd_to_quat(global_orient_6d)
        global_mat = quat_to_matrix(global_quat)
        full_pose_aa[:, 0] = self._matrix_to_axis_angle(global_mat)

        # Torso joints: 6 joints × 6D = 36D
        # [spine1, spine2, spine3, neck, L_collar, R_collar]
        # NOTE: L_shoulder, R_shoulder, L_hip, R_hip rotations are computed by IK solver
        torso_6d = torso_angles_6d.reshape(batch, 6, 6)
        torso_joint_indices = [
            JointIdx.SPINE1, JointIdx.SPINE2, JointIdx.SPINE3,
            JointIdx.NECK, JointIdx.L_COLLAR, JointIdx.R_COLLAR
        ]
        
        # Build local quaternions for global quat computation
        local_quats = quat_identity(batch * 22, device).reshape(batch, 22, 4)
        local_quats[:, 0] = global_quat
        
        for i, joint_idx in enumerate(torso_joint_indices):
            quat = sixd_to_quat(torso_6d[:, i])
            mat = quat_to_matrix(quat)
            full_pose_aa[:, joint_idx] = self._matrix_to_axis_angle(mat)
            local_quats[:, joint_idx] = quat

        # Call body model
        body_output = self.body_model(
            root_orient=full_pose_aa[:, 0],
            pose_body=full_pose_aa[:, 1:].reshape(batch, -1),
            trans=trans,
            betas=betas,
        )

        positions = body_output.Jtr[:, :22]  # (batch, 22, 3)

        # Compute global quaternions via FK chain
        global_quats = self._compute_global_quats(local_quats)
        
        return {
            'joint_positions': positions,
            'left_shoulder_pos': positions[:, JointIdx.L_SHOULDER],
            'right_shoulder_pos': positions[:, JointIdx.R_SHOULDER],
            'left_hip_pos': positions[:, JointIdx.L_HIP],
            'right_hip_pos': positions[:, JointIdx.R_HIP],
            'neck_quat': global_quats[:, JointIdx.NECK],
            'left_collar_quat': global_quats[:, JointIdx.L_COLLAR],
            'right_collar_quat': global_quats[:, JointIdx.R_COLLAR],
            'pelvis_quat': global_quats[:, JointIdx.PELVIS],
            'global_quats': global_quats,
        }
    
    def _matrix_to_axis_angle(self, matrix):
        """Convert rotation matrix to axis-angle"""
        quat = quat_from_matrix(matrix)
        
        # Quaternion to axis-angle
        xyz = quat[..., 1:]
        w = quat[..., 0:1]
        
        # angle = 2 * atan2(||xyz||, w)
        xyz_norm = torch.norm(xyz, dim=-1, keepdim=True)
        angle = 2 * torch.atan2(xyz_norm, w)
        
        # axis = xyz / ||xyz||
        axis = F.normalize(xyz, dim=-1, eps=EPS)
        
        return axis * angle
    
    def _compute_global_quats(self, local_quats):
        """Compute global quaternions from local"""
        batch = local_quats.shape[0]
        device = local_quats.device
        
        parents = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
        ]
        
        global_quats = torch.zeros(batch, 22, 4, device=device)
        global_quats[:, 0] = local_quats[:, 0]
        
        for i in range(1, 22):
            parent = parents[i]
            global_quats[:, i] = quat_multiply(
                global_quats[:, parent],
                local_quats[:, i]
            )
        
        return global_quats
    
    def compute_head_local_quat(self, head_global_quat, neck_global_quat):
        """
        Convert head global rotation to local rotation
        
        R_head_local = R_neck_global^(-1) × R_head_global
        q_head_local = q_neck_global^(-1) * q_head_global
        
        Args:
            head_global_quat: (batch, 4) - Head global orientation from sparse input
            neck_global_quat: (batch, 4) - Neck global orientation from torso FK
        
        Returns:
            head_local_quat: (batch, 4)
        """
        from .manikin_core import quat_conjugate
        
        neck_inv = quat_conjugate(neck_global_quat)
        head_local = quat_multiply(neck_inv, head_global_quat)
        
        return quat_normalize(head_local)


# Backward compatibility alias
TorsoFKWithBodyModel = TorsoFK


# ============================================================
# Standalone function wrappers for backward compatibility
# ============================================================

_torso_fk_instance = None
_body_model_cache = None


def torso_forward_kinematics(global_orient, torso_angles, body_model, trans=None, betas=None):
    """
    Standalone function for torso forward kinematics.

    Args:
        global_orient: (batch, 6) - 6D rotation for pelvis
        torso_angles: (batch, 36) - 6 torso joints × 6D
                      [spine1, spine2, spine3, neck, L_collar, R_collar] × 6D
        body_model: SMPL-H body model
        trans: (batch, 3) - optional translation
        betas: (batch, 16) - body shape parameters (required for accurate FK)

    Returns:
        dict with:
            - left_shoulder_pos, right_shoulder_pos
            - left_hip_pos, right_hip_pos
            - neck_quat (global orientation)
            - joint_positions (batch, 22, 3)
    """
    global _torso_fk_instance, _body_model_cache

    # Cache TorsoFK instance
    if _torso_fk_instance is None or _body_model_cache is not body_model:
        _torso_fk_instance = TorsoFK(body_model)
        _body_model_cache = body_model
        _torso_fk_instance = _torso_fk_instance.to(global_orient.device)

    return _torso_fk_instance(global_orient, torso_angles, trans, betas)
