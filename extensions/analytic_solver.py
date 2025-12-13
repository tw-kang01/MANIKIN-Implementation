"""
MANIKIN Analytic Solvers (Quaternion-based)
ECCV 2024 MANIKIN Paper Eq.7-14 Implementation

Arm: 7-DOF = shoulder(3) + elbow_flexion(1) + elbow_twist(1) + wrist(2)
Leg: 6-DOF = hip(3) + knee_flexion(1) + ankle(2)

V6 Update: Solver now accepts parent_global_quat and returns BOTH:
  - GLOBAL quaternions (for position computation)
  - LOCAL quaternions (for SMPL compatibility)

Key insight: IK computes "swing from T-pose" which differs from SMPL's FK chain.
The LOCAL rotations are computed using SMPL FK chain, not IK's swing-based globals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .manikin_core import (
    quat_identity, quat_normalize, quat_multiply, quat_conjugate,
    quat_rotate_vector, quat_from_axis_angle, quat_from_two_vectors,
    quat_to_matrix, quat_from_matrix, quat_to_sixd, sixd_to_quat,
    compute_swivel_frame, compute_mid_joint_position, compute_flexion_angle,
    compute_swing_quat, compute_twist_angle,
    safe_acos, safe_sqrt, safe_normalize, JointIdx, EPS, EPS_ACOS
)


def to_local_rotation(child_global_quat, parent_global_quat):
    """
    Convert global rotation to local rotation relative to parent.

    Formula: local = inverse(parent_global) × child_global

    This is the standard FK chain inversion for SMPL compatibility.

    Args:
        child_global_quat: (batch, 4) - Child joint global rotation
        parent_global_quat: (batch, 4) - Parent joint global rotation

    Returns:
        local_quat: (batch, 4) - Child's local rotation relative to parent
    """
    parent_inv = quat_conjugate(parent_global_quat)
    local_quat = quat_multiply(parent_inv, child_global_quat)
    return quat_normalize(local_quat)


class AnalyticArmSolver(nn.Module):
    """
    Quaternion-based Arm IK Solver
    
    MANIKIN Eq. 7-14:
        Input: p_shoulder, p_wrist, wrist_quat, swivel_angle, bone_lengths
        Output: shoulder_quat, elbow_quat (flexion+twist), wrist_quat (swing)
    
    Coordinate system: Z-up (AMASS)
        - T-pose arms extend along ±Y axis
        - Reference vector (gravity): -Z
    
    Note: bone_lengths는 forward()에서 동적으로 전달받음 (subject별로 다름)
    """
    
    def __init__(self, bone_offsets=None, v_ref=None):
        """
        Args:
            bone_offsets: dict or tensor with actual bone offset vectors from T-pose.
                         If provided, uses actual bone directions instead of idealized [±1, 0, 0].
                         Keys: 'left_humerus', 'right_humerus' (each is (3,) tensor)
            v_ref: (3,) tensor. Reference direction for swivel angle. 
                   Defaults to [0, 0, -1] (Gravity in Z-up, Backward in Y-up).

        Note: bone_lengths는 forward()에서 동적으로 전달
        """
        super().__init__()

        # Reference direction
        if v_ref is not None:
            self.register_buffer('v_ref', v_ref)
        else:
            # Default: gravity (down = -Z in Z-up system)
            self.register_buffer('v_ref', torch.tensor([0., 0., -1.]))

        # T-pose arm directions from ACTUAL bone offsets (important for accuracy!)
        # Using actual bone offset direction instead of idealized [±1, 0, 0]
        # reduces angular error from ~7 degrees to ~0 degrees
        if bone_offsets is not None and 'left_humerus' in bone_offsets:
            left_dir = F.normalize(bone_offsets['left_humerus'].float(), dim=-1)
            right_dir = F.normalize(bone_offsets['right_humerus'].float(), dim=-1)
            self.register_buffer('tpose_left_arm', left_dir)
            self.register_buffer('tpose_right_arm', right_dir)
        else:
            # Fallback to idealized directions for SMPL Z-up coordinate system
            # Arms extend along Y axis: left=+Y, right=-Y
            self.register_buffer('tpose_left_arm', torch.tensor([0., 1., 0.]))   # +Y (left arm)
            self.register_buffer('tpose_right_arm', torch.tensor([0., -1., 0.])) # -Y (right arm)
    
    def forward(self, p_shoulder, p_wrist, wrist_quat, cos_phi, sin_phi, L1, L2,
                collar_global_quat=None, side='left'):
        """
        Solve arm IK

        Args:
            p_shoulder: (batch, 3) - From torso FK
            p_wrist: (batch, 3) - From sparse input
            wrist_quat: (batch, 4) - Wrist global orientation from sparse input
            cos_phi: (batch,) - Swivel angle cosine
            sin_phi: (batch,) - Swivel angle sine
            L1: (batch,) or float - Humerus length (shoulder→elbow)
            L2: (batch,) or float - Radius length (elbow→wrist)
            collar_global_quat: (batch, 4) - Parent (collar) global rotation from TorsoFK.
                               If provided, LOCAL quaternions are computed for SMPL.
            side: 'left' or 'right'

        Returns:
            dict with:
                - elbow_pos: (batch, 3) - Computed mid-joint position
                - shoulder_quat: (batch, 4) - Shoulder GLOBAL (swing from T-pose)
                - elbow_quat: (batch, 4) - Elbow LOCAL rotation (flexion + twist)
                - elbow_global_quat: (batch, 4) - Elbow GLOBAL (via SMPL FK chain if collar provided)
                - wrist_swing_quat: (batch, 4) - Wrist swing component
                - elbow_flexion: (batch,) - Flexion angle (debug)
                - shoulder_twist: (batch,) - Twist angle (debug)

                If collar_global_quat is provided (V6 SMPL compatibility):
                - shoulder_local: (batch, 4) - Shoulder LOCAL (relative to collar)
                - wrist_local: (batch, 4) - Wrist LOCAL (relative to elbow via SMPL FK)
        """
        batch = p_shoulder.shape[0]
        device = p_shoulder.device
        
        # T-pose direction
        if side == 'left':
            tpose_dir = self.tpose_left_arm
        else:
            tpose_dir = self.tpose_right_arm
        
        # Ensure L1, L2 are tensors with proper shape for broadcasting
        if not isinstance(L1, torch.Tensor):
            L1 = torch.tensor(L1, device=device, dtype=p_shoulder.dtype)
        if not isinstance(L2, torch.Tensor):
            L2 = torch.tensor(L2, device=device, dtype=p_shoulder.dtype)
        
        # For scalar values, expand to batch
        if L1.dim() == 0:
            L1 = L1.expand(batch)
        if L2.dim() == 0:
            L2 = L2.expand(batch)
        
        # ============================================================
        # Eq. 7: Elbow position from swivel angle
        # ============================================================
        p_elbow = compute_mid_joint_position(
            p_shoulder, p_wrist, L1, L2, cos_phi, sin_phi, self.v_ref
        )
        
        # ============================================================
        # Eq. 9: Elbow flexion angle (1-DOF constraint)
        # ============================================================
        d = torch.norm(p_wrist - p_shoulder, dim=-1)
        theta_flexion = compute_flexion_angle(L1, L2, d)
        
        # ============================================================
        # Eq. 10: Shoulder swing rotation
        # ============================================================
        # Rotate T-pose arm direction to actual elbow direction
        vec_shoulder_elbow = p_elbow - p_shoulder
        # tpose_dir: (3,) -> (batch, 3), L1: (batch,) -> (batch, 1)
        tpose_arm_vec = tpose_dir.expand(batch, -1) * L1.unsqueeze(-1)
        
        q_shoulder_swing = compute_swing_quat(tpose_arm_vec, vec_shoulder_elbow)
        
        # ============================================================
        # Eq. 11, 12: Shoulder twist rotation
        # ============================================================
        # Compute where wrist would be after only swing (no twist)
        # forearm_tpose: (batch, 3), L2: (batch,) -> (batch, 1)
        forearm_tpose = tpose_dir.expand(batch, -1) * L2.unsqueeze(-1)
        forearm_after_swing = quat_rotate_vector(
            q_shoulder_swing, forearm_tpose
        )
        
        # Wrist position after flexion at elbow (no twist)
        # Apply flexion rotation around local X axis
        flexion_axis = torch.tensor([1., 0., 0.], device=device)
        q_flexion_local = quat_from_axis_angle(
            flexion_axis.expand(batch, -1), 
            theta_flexion
        )
        
        # Rotate forearm by flexion
        forearm_after_flexion = quat_rotate_vector(q_flexion_local, forearm_tpose)
        # Then by shoulder swing
        forearm_after_swing_flexion = quat_rotate_vector(q_shoulder_swing, forearm_after_flexion)
        
        p_wrist_after_swing = p_elbow + forearm_after_swing_flexion
        
        # Twist angle from chord length
        # L2: (batch,), sin(theta_flexion): (batch,) -> r_orbit: (batch,)
        r_orbit = L2 * torch.sin(theta_flexion)  # orbit radius for wrist

        # Handle singularity: when arm is fully extended (theta_flexion ≈ 0), r_orbit ≈ 0
        # In this case, twist is undefined, so we set it to 0
        is_singular = r_orbit < EPS
        r_orbit_safe = torch.where(is_singular, torch.ones_like(r_orbit), r_orbit)

        # Twist axis = shoulder-to-elbow direction (compute before twist angle)
        twist_axis = safe_normalize(vec_shoulder_elbow)

        # Compute SIGNED twist angle using twist_axis for direction
        theta_twist = compute_twist_angle(
            p_elbow, p_wrist, p_wrist_after_swing, r_orbit_safe,
            twist_axis=twist_axis  # Pass axis for sign determination
        )
        theta_twist = torch.where(is_singular, torch.zeros_like(theta_twist), theta_twist)

        q_shoulder_twist = quat_from_axis_angle(twist_axis, theta_twist)
        
        # Compose shoulder rotation: twist × swing
        q_shoulder = quat_multiply(q_shoulder_twist, q_shoulder_swing)
        
        # ============================================================
        # Eq. 13: Elbow twist (decompose wrist orientation)
        # Paper: q_twist = (q_w * ||v||^2, v_x*(v.q), v_y*(v.q), v_z*(v.q)) normalized
        # where v = twist axis (elbow->wrist), q = wrist quaternion
        # ============================================================
        vec_elbow_wrist = p_wrist - p_elbow
        v_twist = safe_normalize(vec_elbow_wrist)  # (batch, 3)
        
        # Extract wrist quaternion components
        q_w = wrist_quat[:, 0:1]  # (batch, 1) - scalar part
        q_xyz = wrist_quat[:, 1:]  # (batch, 3) - vector part
        
        # Compute v · q (dot product of twist axis with vector part)
        v_dot_q = (v_twist * q_xyz).sum(dim=-1, keepdim=True)  # (batch, 1)
        
        # Compute ||v||² (should be 1 since normalized, but keep for clarity)
        v_norm_sq = (v_twist * v_twist).sum(dim=-1, keepdim=True)  # (batch, 1)
        
        # Eq. 13: q_twist_elbow = (q_w * ||v||², v_x*(v·q), v_y*(v·q), v_z*(v·q))
        twist_w = q_w * v_norm_sq  # (batch, 1)
        twist_xyz = v_twist * v_dot_q  # (batch, 3) - each component scaled by (v·q)
        
        q_elbow_twist = torch.cat([twist_w, twist_xyz], dim=-1)  # (batch, 4)
        q_elbow_twist = quat_normalize(q_elbow_twist)
        
        # ============================================================
        # Eq. 14: Wrist swing = q_global_wrist * conjugate(q_twist_elbow)
        # ============================================================
        q_wrist_swing = quat_multiply(wrist_quat, quat_conjugate(q_elbow_twist))
        
        # ============================================================
        # Compose elbow rotation: flexion × twist
        # ============================================================
        # Flexion is around local X axis
        q_elbow = quat_multiply(q_elbow_twist, q_flexion_local)

        # ============================================================
        # Build return dict
        # ============================================================
        result = {
            'elbow_pos': p_elbow,
            # IK-computed quaternions
            'shoulder_quat': q_shoulder,           # Shoulder GLOBAL (swing from T-pose)
            'elbow_quat': q_elbow,                 # Elbow LOCAL (flexion + twist)
            'wrist_swing_quat': q_wrist_swing,     # Wrist swing component
            # Debug info
            'elbow_flexion': theta_flexion,
            'shoulder_twist': theta_twist,
        }

        # ============================================================
        # V6: Compute SMPL-compatible LOCAL quaternions
        # ============================================================
        if collar_global_quat is not None:
            # Step 1: Convert shoulder swing → LOCAL (relative to collar)
            shoulder_local = to_local_rotation(q_shoulder, collar_global_quat)
            result['shoulder_local'] = shoulder_local

            # Step 2: Recompute elbow_global using SMPL FK chain
            # IMPORTANT: Use collar × shoulder_local × elbow_local
            # NOT: IK's q_shoulder × q_elbow (different frame!)
            shoulder_global_smpl = quat_multiply(collar_global_quat, shoulder_local)
            elbow_global_smpl = quat_multiply(shoulder_global_smpl, q_elbow)
            result['elbow_global_quat'] = elbow_global_smpl

            # Step 3: Convert wrist GLOBAL → LOCAL (using correct elbow_global)
            wrist_local = to_local_rotation(wrist_quat, elbow_global_smpl)
            result['wrist_local'] = wrist_local
        else:
            # Legacy mode: use IK's swing-based global (less accurate for SMPL)
            q_elbow_global = quat_multiply(q_shoulder, q_elbow)
            result['elbow_global_quat'] = q_elbow_global

        return result


class AnalyticLegSolver(nn.Module):
    """
    Quaternion-based Leg IK Solver
    
    MANIKIN Eq. 7-12:
        Input: p_hip, p_ankle, swivel_angle, bone_lengths
        Output: hip_quat, knee_quat (flexion only)
    
    Coordinate system: Z-up (AMASS)
        - T-pose legs extend along -Z axis
        - Reference vector (gravity): -Z (same as leg direction in T-pose)
    
    Note: bone_lengths는 forward()에서 동적으로 전달받음 (subject별로 다름)
    """
    
    def __init__(self, bone_offsets=None, v_ref=None):
        """
        Args:
            bone_offsets: dict with actual bone offset vectors from T-pose.
                         If provided, uses actual bone directions instead of idealized [0, -1, 0].
                         Keys: 'left_femur', 'right_femur' (each is (3,) tensor)
            v_ref: (3,) tensor. Reference direction for swivel angle.
                   Defaults to [0, 1, 0] (Forward in Z-up). 
                   NOTE: For Y-up data (SMPL), pass [0, 0, -1] to avoid singularity with leg axis (-Y).

        Note: bone_lengths는 forward()에서 동적으로 전달
        """
        super().__init__()

        # Reference direction
        if v_ref is not None:
            self.register_buffer('v_ref', v_ref)
        else:
            # Reference direction: Forward (+Y in Z-up system)
            # Using -Z (gravity) causes singularity when leg is vertical (n || v_ref)
            # Using +Y makes phi=0 correspond to natural forward knee bend
            self.register_buffer('v_ref', torch.tensor([0., 1., 0.]))

        # T-pose leg direction from ACTUAL bone offsets
        # Using actual bone offset direction instead of idealized [0, -1, 0]
        # reduces angular error significantly
        if bone_offsets is not None and 'left_femur' in bone_offsets:
            left_dir = F.normalize(bone_offsets['left_femur'].float(), dim=-1)
            right_dir = F.normalize(bone_offsets['right_femur'].float(), dim=-1)
            self.register_buffer('tpose_left_leg', left_dir)
            self.register_buffer('tpose_right_leg', right_dir)
        else:
            # Fallback to idealized directions for SMPL Z-up coordinate system
            # Legs extend downward along -Z axis (gravity direction)
            self.register_buffer('tpose_left_leg', torch.tensor([0., 0., -1.]))   # -Z (down)
            self.register_buffer('tpose_right_leg', torch.tensor([0., 0., -1.]))  # -Z (down)
    
    def forward(self, p_hip, p_ankle, cos_phi, sin_phi, L1, L2,
                pelvis_global_quat=None, side='left'):
        """
        Solve leg IK

        Args:
            p_hip: (batch, 3) - From torso FK
            p_ankle: (batch, 3) - From foot FK or estimation
            cos_phi: (batch,) - Swivel angle cosine
            sin_phi: (batch,) - Swivel angle sine
            L1: (batch,) or float - Femur length (hip→knee)
            L2: (batch,) or float - Tibia length (knee→ankle)
            pelvis_global_quat: (batch, 4) - Parent (pelvis) global rotation from TorsoFK.
                               If provided, LOCAL quaternions are computed for SMPL.
            side: 'left' or 'right'

        Returns:
            dict with:
                - knee_pos: (batch, 3)
                - hip_quat: (batch, 4) - GLOBAL rotation (swing from T-pose)
                - knee_quat: (batch, 4) - LOCAL rotation (flexion only, relative to hip)
                - knee_global_quat: (batch, 4) - GLOBAL rotation (via SMPL FK chain if pelvis provided)

                If pelvis_global_quat is provided (V6 SMPL compatibility):
                - hip_local: (batch, 4) - Hip LOCAL (relative to pelvis)
        """
        batch = p_hip.shape[0]
        device = p_hip.device

        # Select T-pose direction based on side
        if side == 'left':
            tpose_dir = self.tpose_left_leg
        else:
            tpose_dir = self.tpose_right_leg

        # Ensure L1, L2 are tensors with proper shape for broadcasting
        if not isinstance(L1, torch.Tensor):
            L1 = torch.tensor(L1, device=device, dtype=p_hip.dtype)
        if not isinstance(L2, torch.Tensor):
            L2 = torch.tensor(L2, device=device, dtype=p_hip.dtype)

        # For scalar values, expand to batch
        if L1.dim() == 0:
            L1 = L1.expand(batch)
        if L2.dim() == 0:
            L2 = L2.expand(batch)

        # ============================================================
        # Eq. 7: Knee position from swivel angle
        # ============================================================
        p_knee = compute_mid_joint_position(
            p_hip, p_ankle, L1, L2, cos_phi, sin_phi, self.v_ref
        )

        # ============================================================
        # Eq. 9: Knee flexion angle (1-DOF constraint)
        # ============================================================
        d = torch.norm(p_ankle - p_hip, dim=-1)
        theta_flexion = compute_flexion_angle(L1, L2, d)

        # ============================================================
        # Eq. 10: Hip swing rotation
        # ============================================================
        vec_hip_knee = p_knee - p_hip
        # tpose_dir: (3,) -> (batch, 3), L1: (batch,) -> (batch, 1)
        tpose_leg_vec = tpose_dir.expand(batch, -1) * L1.unsqueeze(-1)

        q_hip_swing = compute_swing_quat(tpose_leg_vec, vec_hip_knee)

        # ============================================================
        # Eq. 11, 12: Hip twist rotation
        # ============================================================
        # Compute ankle position after only swing (no twist)
        # tibia_tpose: (batch, 3), L2: (batch,) -> (batch, 1)
        tibia_tpose = tpose_dir.expand(batch, -1) * L2.unsqueeze(-1)
        
        # Apply flexion at knee
        flexion_axis = torch.tensor([1., 0., 0.], device=device)
        q_flexion_local = quat_from_axis_angle(
            flexion_axis.expand(batch, -1),
            theta_flexion
        )
        
        # Tibia after knee flexion
        tibia_after_flexion = quat_rotate_vector(q_flexion_local, tibia_tpose)
        # Then by hip swing
        tibia_after_swing = quat_rotate_vector(q_hip_swing, tibia_after_flexion)
        
        p_ankle_after_swing = p_knee + tibia_after_swing
        
        # Twist angle
        # L2: (batch,), sin(theta_flexion): (batch,) -> r_orbit: (batch,)
        r_orbit = L2 * torch.sin(theta_flexion)

        # Handle singularity: when leg is fully extended (theta_flexion ≈ 0), r_orbit ≈ 0
        is_singular = r_orbit < EPS
        r_orbit_safe = torch.where(is_singular, torch.ones_like(r_orbit), r_orbit)

        # Twist axis = hip-to-knee direction (compute before twist angle)
        twist_axis = safe_normalize(vec_hip_knee)

        # Compute SIGNED twist angle using twist_axis for direction
        theta_twist = compute_twist_angle(
            p_knee, p_ankle, p_ankle_after_swing, r_orbit_safe,
            twist_axis=twist_axis  # Pass axis for sign determination
        )
        theta_twist = torch.where(is_singular, torch.zeros_like(theta_twist), theta_twist)

        q_hip_twist = quat_from_axis_angle(twist_axis, theta_twist)
        
        # Compose hip rotation: twist × swing
        q_hip = quat_multiply(q_hip_twist, q_hip_swing)
        
        # ============================================================
        # Knee: Only flexion (no twist for knee)
        # ============================================================
        q_knee = q_flexion_local

        # ============================================================
        # Build return dict
        # ============================================================
        result = {
            'knee_pos': p_knee,
            'hip_quat': q_hip,              # (batch, 4) Hip GLOBAL (swing from T-pose)
            'knee_quat': q_knee,            # (batch, 4) Knee LOCAL (flexion only)
            'knee_flexion': theta_flexion,  # (batch,) for debugging
            'hip_twist': theta_twist,       # (batch,) for debugging
        }

        # ============================================================
        # V6: Compute SMPL-compatible LOCAL quaternions
        # ============================================================
        if pelvis_global_quat is not None:
            # Step 1: Convert hip swing → LOCAL (relative to pelvis)
            hip_local = to_local_rotation(q_hip, pelvis_global_quat)
            result['hip_local'] = hip_local

            # Step 2: Recompute knee_global using SMPL FK chain
            # IMPORTANT: Use pelvis × hip_local × knee_local
            # NOT: IK's q_hip × q_knee (different frame!)
            hip_global_smpl = quat_multiply(pelvis_global_quat, hip_local)
            knee_global_smpl = quat_multiply(hip_global_smpl, q_knee)
            result['knee_global_quat'] = knee_global_smpl
        else:
            # Legacy mode: use IK's swing-based global (less accurate for SMPL)
            q_knee_global = quat_multiply(q_hip, q_knee)
            result['knee_global_quat'] = q_knee_global

        return result


def compute_gt_swivel_angle(p_base, p_mid, p_end, v_ref):
    """
    Compute ground truth swivel angle from joint positions
    
    Used for:
    1. Computing swivel GT during data preparation
    2. Computing swivel loss during training
    
    Args:
        p_base: (batch, 3) - Base joint (shoulder/hip)
        p_mid: (batch, 3) - Mid joint (elbow/knee)
        p_end: (batch, 3) - End joint (wrist/ankle)
        v_ref: (3,) - Reference direction (-Z for gravity)
    
    Returns:
        cos_phi: (batch,)
        sin_phi: (batch,)
    """
    device = p_base.device
    v_ref = v_ref.to(device)
    
    # Get coordinate frame
    n, u, v = compute_swivel_frame(p_base, p_end, v_ref)
    
    # Vector from orbit center to mid joint
    vec_base_end = p_end - p_base
    d = torch.norm(vec_base_end, dim=-1, keepdim=True)
    
    L1_sq = torch.norm(p_mid - p_base, dim=-1, keepdim=True) ** 2
    L2_sq = torch.norm(p_end - p_mid, dim=-1, keepdim=True) ** 2
    d_sq = d ** 2
    
    # cos(α) = (L1² + d² - L2²) / (2 * L1 * d)
    # But we just need the orbit center position
    L1 = torch.norm(p_mid - p_base, dim=-1, keepdim=True)
    d_safe = torch.clamp(d, min=EPS)
    cos_alpha = (L1_sq + d_sq - L2_sq) / (2 * L1 * d_safe)
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    
    # Orbit center
    p_c = p_base + L1 * cos_alpha * n
    
    # Vector from orbit center to mid joint
    vec_c_mid = p_mid - p_c
    vec_c_mid_norm = safe_normalize(vec_c_mid)
    
    # Project onto u, v
    cos_phi = (vec_c_mid_norm * u).sum(dim=-1)
    sin_phi = (vec_c_mid_norm * v).sum(dim=-1)
    
    # Normalize (in case of numerical errors)
    norm = safe_sqrt(cos_phi ** 2 + sin_phi ** 2)
    cos_phi = cos_phi / (norm + EPS)
    sin_phi = sin_phi / (norm + EPS)
    
    return cos_phi, sin_phi


def compute_bone_offsets_from_betas(body_model, betas, device='cpu'):
    """
    Compute T-pose bone offset directions from betas.

    실제 AMASS 데이터는 완벽한 T-pose가 아닐 수 있으므로,
    betas로 계산한 실제 bone offset 방향을 사용해야 정확도가 높아짐.

    Args:
        body_model: SMPL-H BodyModel instance
        betas: (16,) or (B, 16) body shape parameters
        device: torch device

    Returns:
        dict with bone offset unit vectors:
            'left_humerus': (3,) direction from L_SHOULDER to L_ELBOW
            'right_humerus': (3,) direction from R_SHOULDER to R_ELBOW
            'left_femur': (3,) direction from L_HIP to L_KNEE
            'right_femur': (3,) direction from R_HIP to R_KNEE

    Usage:
        # In model initialization
        bone_offsets = compute_bone_offsets_from_betas(body_model, betas)
        arm_solver = AnalyticArmSolver(bone_offsets=bone_offsets)
        leg_solver = AnalyticLegSolver(bone_offsets=bone_offsets)
    """
    if not isinstance(betas, torch.Tensor):
        betas = torch.tensor(betas, device=device, dtype=torch.float32)

    if betas.dim() == 1:
        betas = betas.unsqueeze(0)  # (1, 16)

    betas = betas.to(device)

    # T-pose = zero pose
    with torch.no_grad():
        body_output = body_model(
            root_orient=torch.zeros(1, 3, device=device),
            pose_body=torch.zeros(1, 63, device=device),
            trans=torch.zeros(1, 3, device=device),
            betas=betas[:1],  # Use first betas only
        )

    positions = body_output.Jtr[0, :22]  # (22, 3)

    # Compute bone directions (normalize)
    bone_offsets = {
        'left_humerus': F.normalize(
            positions[JointIdx.L_ELBOW] - positions[JointIdx.L_SHOULDER], dim=-1
        ),
        'right_humerus': F.normalize(
            positions[JointIdx.R_ELBOW] - positions[JointIdx.R_SHOULDER], dim=-1
        ),
        'left_femur': F.normalize(
            positions[JointIdx.L_KNEE] - positions[JointIdx.L_HIP], dim=-1
        ),
        'right_femur': F.normalize(
            positions[JointIdx.R_KNEE] - positions[JointIdx.R_HIP], dim=-1
        ),
    }

    return bone_offsets


def compute_bone_lengths_from_positions(positions):
    """
    GT joint positions에서 bone lengths를 실시간 계산

    SMPL-H는 motion 내에서 bone length가 일정하므로,
    첫 프레임에서 계산해도 되고, 매 프레임 계산해도 됨

    Args:
        positions: (batch, 22, 3) or (22, 3) - Joint positions

    Returns:
        dict with bone lengths (batch,) tensors or scalar if single frame
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)  # (1, 22, 3)
    
    # Arm bone lengths
    left_humerus = torch.norm(
        positions[:, JointIdx.L_ELBOW] - positions[:, JointIdx.L_SHOULDER], dim=-1
    )
    left_radius = torch.norm(
        positions[:, JointIdx.L_WRIST] - positions[:, JointIdx.L_ELBOW], dim=-1
    )
    right_humerus = torch.norm(
        positions[:, JointIdx.R_ELBOW] - positions[:, JointIdx.R_SHOULDER], dim=-1
    )
    right_radius = torch.norm(
        positions[:, JointIdx.R_WRIST] - positions[:, JointIdx.R_ELBOW], dim=-1
    )
    
    # Leg bone lengths
    left_femur = torch.norm(
        positions[:, JointIdx.L_KNEE] - positions[:, JointIdx.L_HIP], dim=-1
    )
    left_tibia = torch.norm(
        positions[:, JointIdx.L_ANKLE] - positions[:, JointIdx.L_KNEE], dim=-1
    )
    right_femur = torch.norm(
        positions[:, JointIdx.R_KNEE] - positions[:, JointIdx.R_HIP], dim=-1
    )
    right_tibia = torch.norm(
        positions[:, JointIdx.R_ANKLE] - positions[:, JointIdx.R_KNEE], dim=-1
    )
    
    return {
        'left_humerus': left_humerus,
        'left_radius': left_radius,
        'right_humerus': right_humerus,
        'right_radius': right_radius,
        'left_femur': left_femur,
        'left_tibia': left_tibia,
        'right_femur': right_femur,
        'right_tibia': right_tibia,
    }
