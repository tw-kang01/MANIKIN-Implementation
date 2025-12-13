"""
Swivel Angle Ground Truth Computation
MANIKIN 논문 Equations 2-5 구현

NaN 방지: SAGE/AvatarPoser 스타일의 안정화 함수 사용
- safe_acos, safe_normalize (analytic_solver에서 import)
- atan2 기반 각도 계산 사용
"""

import torch
import numpy as np
from .body_output_utils import get_joint_positions
from .manikin_core import safe_acos, safe_normalize, safe_sqrt, EPS, EPS_ACOS

# Alias for backward compatibility
EPS_NORM = EPS


def compute_bone_lengths_from_tpose(body_model, device=None):
    """
    T-pose에서 bone lengths 계산
    
    Args:
        body_model: SMPL-H body model
        device: torch device (optional, auto-detected if not provided)
    
    Returns:
        dict: bone lengths for arms and legs
    """
    # Safer device detection
    if device is None:
        # Try parameters first
        for param in body_model.parameters():
            device = param.device
            break
        else:
            # Fallback to buffers
            for buffer in body_model.buffers():
                device = buffer.device
                break
            else:
                # Last resort: use CUDA if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Debug logging removed - use proper logging if needed
    
    # T-pose: all joint angles = 0
    zero_pose = torch.zeros(1, 22, 3, device=device)
    
    # Explicitly provide hand pose to avoid device mismatch with defaults
    # SMPL-H: 15 joints * 3 dims * 2 hands = 90
    zero_hand = torch.zeros(1, 90, device=device)
    
    with torch.no_grad():
        body_output = body_model(
            root_orient=zero_pose[:, 0],
            pose_body=zero_pose[:, 1:].reshape(1, -1),
            pose_hand=zero_hand
        )
    
    # Extract joint positions safely (handles both dict and object)
    joint_positions = get_joint_positions(body_output)  # returns numpy (T, 22, 3)
    
    # Convert back to tensor on the correct device
    if isinstance(joint_positions, np.ndarray):
        joint_pos = torch.from_numpy(joint_positions[0]).to(device)  # (22, 3)
    else:
        joint_pos = joint_positions[0].to(device)
    
    bone_lengths = {
        # Arms: shoulder(16,17) -> elbow(18,19) -> wrist(20,21)
        'left_humerus': torch.norm(joint_pos[18] - joint_pos[16]).item(),   # shoulder -> elbow
        'left_radius': torch.norm(joint_pos[20] - joint_pos[18]).item(),    # elbow -> wrist
        'right_humerus': torch.norm(joint_pos[19] - joint_pos[17]).item(),  # shoulder -> elbow
        'right_radius': torch.norm(joint_pos[21] - joint_pos[19]).item(),   # elbow -> wrist
        # Legs: hip(1,2) -> knee(4,5) -> ankle(7,8)
        'left_femur': torch.norm(joint_pos[4] - joint_pos[1]).item(),       # hip -> knee
        'left_tibia': torch.norm(joint_pos[7] - joint_pos[4]).item(),       # knee -> ankle
        'right_femur': torch.norm(joint_pos[5] - joint_pos[2]).item(),      # hip -> knee
        'right_tibia': torch.norm(joint_pos[8] - joint_pos[5]).item(),      # knee -> ankle
    }
    
    return bone_lengths


def compute_swivel_angle_gt(p_base, p_mid, p_end, v_ref, L1, L2):
    """
    논문 Equations 2-5: Swivel angle ground truth 계산
    
    Args:
        p_base: base joint position (shoulder/hip) - (3,) or (N, 3)
        p_mid: mid joint position (elbow/knee) - (3,) or (N, 3)
        p_end: end joint position (wrist/ankle) - (3,) or (N, 3)
        v_ref: reference vector - (3,)
            Z-up 좌표계 (AMASS): torch.tensor([0., 0., -1.]) (down is -Z)
            Arm: body longitudinal axis, downward
            Leg: sagittal axis
        L1: upper segment length (humerus/femur)
        L2: lower segment length (radius/tibia)
    
    Returns:
        cos_phi, sin_phi: continuous swivel angle representation
    """
    # Handle both single and batch inputs
    if p_base.dim() == 1:
        p_base = p_base.unsqueeze(0)
        p_mid = p_mid.unsqueeze(0)
        p_end = p_end.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Eq. 2: Compute n, u, v
    vec_be = p_end - p_base
    n = safe_normalize(vec_be)
    
    # u: projection of -v_ref onto orbit plane
    v_ref = v_ref.to(p_base.device)
    if v_ref.dim() == 1:
        v_ref = v_ref.unsqueeze(0)
    
    dot_vref_n = (v_ref * n).sum(dim=-1, keepdim=True)
    u = -v_ref + dot_vref_n * n
    u = safe_normalize(u)
    
    # v: perpendicular to u and n (using torch.linalg.cross for stability)
    v = torch.linalg.cross(u, n, dim=-1)
    
    # Eq. 3: angle α (law of cosines)
    d = torch.norm(vec_be, dim=-1, keepdim=True)
    d_safe = torch.clamp(d, min=EPS_NORM)
    cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d_safe)
    # Use safe_acos
    alpha = safe_acos(cos_alpha)
    
    # Eq. 4: orbit circle center
    p_c_m = p_base + L1 * n * torch.cos(alpha)
    r_c_m = L1 * torch.sin(alpha)
    
    # Eq. 5: swivel angle from mid joint
    r = p_mid - p_c_m
    r_norm = torch.norm(r, dim=-1, keepdim=True)
    
    # Avoid division by zero
    r_norm = torch.clamp(r_norm, min=1e-6)
    
    cos_phi = (u * r).sum(dim=-1) / r_norm.squeeze(-1)
    sin_phi = (v * r).sum(dim=-1) / r_norm.squeeze(-1)
    
    # Normalize to ensure cos² + sin² = 1 (prevents numerical drift)
    norm = torch.sqrt(cos_phi**2 + sin_phi**2 + 1e-12)
    cos_phi = cos_phi / norm
    sin_phi = sin_phi / norm
    
    if squeeze_output:
        cos_phi = cos_phi.squeeze(0)
        sin_phi = sin_phi.squeeze(0)
    
    return cos_phi, sin_phi


def predict_mid_joint_from_swivel(p_base, p_end, cos_phi, sin_phi, L1, L2, v_ref):
    """
    논문 Eq. 7: p_m^pred(φ) 계산
    
    Given swivel angle (as cos/sin), compute mid joint position
    This is the inverse of compute_swivel_angle_gt
    
    Args:
        p_base, p_end: (batch, 3)
        cos_phi, sin_phi: (batch,)
        L1, L2: float
        v_ref: (3,)
    
    Returns:
        p_mid: (batch, 3) predicted mid joint position
    """
    # Eq. 2
    vec_be = p_end - p_base
    n = safe_normalize(vec_be)
    
    v_ref = v_ref.to(p_base.device).unsqueeze(0)
    dot_vref_n = (v_ref * n).sum(dim=-1, keepdim=True)
    u = -v_ref + dot_vref_n * n
    u = safe_normalize(u)
    v = torch.linalg.cross(u, n, dim=-1)
    
    # Eq. 3, 4
    d = torch.norm(vec_be, dim=-1, keepdim=True)
    d_safe = torch.clamp(d, min=EPS_NORM)
    
    # Handle degenerate case: when d is very small, mid joint is at base
    is_degenerate = d < EPS_NORM
    
    cos_alpha = (L1**2 + d_safe**2 - L2**2) / (2 * L1 * d_safe)
    # Clamp to valid range for acos
    cos_alpha = torch.clamp(cos_alpha, -1.0 + 1e-6, 1.0 - 1e-6)
    # Use safe_acos
    alpha = safe_acos(cos_alpha)
    
    p_c_m = p_base + L1 * n * torch.cos(alpha)
    r_c_m = L1 * torch.sin(alpha)
    
    # Eq. 7
    p_m_pred = p_c_m + r_c_m * (u * cos_phi.unsqueeze(-1) + v * sin_phi.unsqueeze(-1))
    
    # For degenerate cases, return base position
    p_m_pred = torch.where(is_degenerate.expand_as(p_m_pred), p_base, p_m_pred)
    
    return p_m_pred

