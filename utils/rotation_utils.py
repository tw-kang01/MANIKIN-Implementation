"""
MANIKIN Rotation Utils
SAGE/AvatarPoser의 안정화된 회전 변환 함수들을 통합

Features:
- SAGE rotation_conversions.py의 PyTorch3D 스타일 함수들
- AvatarPoser utils_transform.py의 6D representation 함수들
- NaN-safe 수학 연산 함수들
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# SAGE Rotation Functions (PyTorch3D style)
# Source: SAGE/utils/rotation_conversions.py
# ============================================================

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from a,
    with sign taken from b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x, eps=1e-12):
    """
    Returns torch.sqrt(torch.max(0, x)) with safe gradient near zero.
    SAGE L82-87: NaN-safe sqrt
    
    Key insight: sqrt(x).grad = 1/(2*sqrt(x)) → ∞ as x → 0
    Solution: Add small eps BEFORE sqrt to bound the gradient.
    """
    # Clamp to positive + eps to avoid:
    # 1. sqrt of negative → NaN
    # 2. gradient explosion at sqrt(0+) → 1/(2*sqrt(eps)) is bounded
    x_safe = torch.clamp(x, min=eps)
    return torch.sqrt(x_safe) * (x > 0).float()  # Zero output for x <= 0


def quaternion_to_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.
    Args: quaternions: (..., 4) with real part first
    Returns: (..., 3, 3) rotation matrices
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    """
    Convert rotation matrices to quaternions.
    Args: matrix: (..., 3, 3) rotation matrices
    Returns: (..., 4) quaternions with real part first
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle to quaternions.
    SAGE L449-479: includes small angle Taylor series approximation
    
    Args: axis_angle: (..., 3)
    Returns: (..., 4) quaternions with real part first
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Taylor series: sin(x/2)/x ≈ 1/2 - x²/48 for small x
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert quaternions to axis-angle.
    SAGE L481-509: atan2-based for numerical stability
    
    Args: quaternions: (..., 4) with real part first
    Returns: (..., 3) axis-angle
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Taylor series for small angles
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle to rotation matrix.
    Args: axis_angle: (..., 3)
    Returns: (..., 3, 3) rotation matrix
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle.
    Args: matrix: (..., 3, 3)
    Returns: (..., 3) axis-angle
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions (raw, no standardization).
    Args: a, b: (..., 4) quaternions with real part first
    Returns: (..., 4) product quaternion
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def standardize_quaternion(quaternions):
    """
    Convert quaternion to standard form with non-negative real part.
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions and standardize.
    Args: a, b: (..., 4) quaternions with real part first
    Returns: (..., 4) standardized product quaternion
    """
    return standardize_quaternion(quaternion_raw_multiply(a, b))


def quaternion_invert(quaternion):
    """
    Get inverse quaternion (conjugate for unit quaternions).
    Args: quaternion: (..., 4) with real part first
    Returns: (..., 4) inverse quaternion
    """
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply quaternion rotation to 3D points.
    Args:
        quaternion: (..., 4) with real part first
        point: (..., 3)
    Returns: (..., 3) rotated points
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D: {point.shape}")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to rotation matrix.
    SAGE L512-531: Gram-Schmidt orthogonalization

    6D format: first two COLUMNS of rotation matrix [col0, col1]
    Each column is a 3D vector, so 6D = [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z]

    Args: d6: (..., 6)
    Returns: (..., 3, 3) rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as COLUMNS (dim=-1) - b1, b2, b3 become columns of the matrix
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix):
    """
    Convert rotation matrix to 6D representation.
    6D = first two COLUMNS of rotation matrix

    Args: matrix: (..., 3, 3)
    Returns: (..., 6)
    """
    # Extract first two columns and concatenate
    col0 = matrix[..., :, 0]  # (..., 3)
    col1 = matrix[..., :, 1]  # (..., 3)
    return torch.cat([col0, col1], dim=-1)  # (..., 6)


# ============================================================
# AvatarPoser 6D Functions
# Source: AvatarPoser/utils/utils_transform.py
# ============================================================

def bgs(d6s):
    """
    Batch Gram-Schmidt orthogonalization for 6D rotation.
    AvatarPoser style: F.normalize based
    
    Args: d6s: (N, 6) or (N, 2, 3)
    Returns: (N, 3, 3) rotation matrices
    """
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    c = torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1
    b2 = F.normalize(a2 - c, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


# NOTE: sixd2matrot 삭제됨 - rotation_6d_to_matrix 사용 (Gram-Schmidt 정규화 포함)
# 기존 sixd2matrot은 정규화 없이 cross product만 사용하여 수치적으로 불안정


def matrot2sixd(pose_matrot):
    """
    Convert rotation matrix to 6D representation.
    Args: pose_matrot: (N, 3, 3)
    Returns: (N, 6)
    """
    return torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)


# ============================================================
# Safe Math Functions (NaN-free)
# ============================================================

EPS_ACOS = 1e-6
EPS_NORM = 1e-8


def safe_acos(x, eps=EPS_ACOS):
    """
    Safe acos that prevents NaN gradient at ±1.
    """
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def safe_asin(x, eps=EPS_ACOS):
    """
    Safe asin that prevents NaN gradient at ±1.
    """
    return torch.asin(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def safe_sqrt(x):
    """
    Safe sqrt that handles negative values (returns 0).
    Same as _sqrt_positive_part.
    """
    return _sqrt_positive_part(x)


def safe_normalize(x, dim=-1, eps=EPS_NORM):
    """
    Safe normalization using F.normalize.
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


def safe_div(num, denom, eps=EPS_NORM):
    """
    Safe division that prevents division by zero.
    """
    return num / (denom + eps)


def compute_angle_atan2(v1, v2, eps=EPS_NORM):
    """
    Compute angle between two vectors using atan2 (more stable than acos).
    
    Args:
        v1, v2: (..., 3) vectors
    Returns:
        (...,) angles in radians
    """
    v1_n = safe_normalize(v1)
    v2_n = safe_normalize(v2)
    
    dot = (v1_n * v2_n).sum(dim=-1)
    cross = torch.linalg.cross(v1_n, v2_n, dim=-1)
    cross_norm = torch.norm(cross, dim=-1)
    
    return torch.atan2(cross_norm, dot)


def compute_swing_rotation(v_from, v_to, eps=EPS_NORM):
    """
    Compute axis-angle rotation from v_from to v_to using atan2.
    
    Args:
        v_from, v_to: (..., 3) vectors
    Returns:
        (..., 3) axis-angle rotation
    """
    v_from_n = safe_normalize(v_from)
    v_to_n = safe_normalize(v_to)
    
    cross = torch.linalg.cross(v_from_n, v_to_n, dim=-1)
    cross_norm = torch.norm(cross, dim=-1, keepdim=True)
    dot = (v_from_n * v_to_n).sum(dim=-1, keepdim=True)
    
    angle = torch.atan2(cross_norm, dot)
    
    # Handle parallel vectors
    fallback = torch.tensor([[1., 0., 0.]], device=v_from.device).expand_as(cross)
    is_parallel = cross_norm < eps
    
    safe_cross = torch.where(is_parallel, fallback, cross)
    axis = safe_normalize(safe_cross)
    
    # Handle anti-parallel
    is_antiparallel = dot < -1.0 + eps
    axis = torch.where(is_antiparallel, fallback, axis)
    angle = torch.where(is_antiparallel, torch.full_like(angle, math.pi), angle)
    
    return axis * angle


# ============================================================
# Convenience Aliases
# ============================================================

# SAGE-style naming
aa_to_quat = axis_angle_to_quaternion
quat_to_aa = quaternion_to_axis_angle
aa_to_rotmat = axis_angle_to_matrix
rotmat_to_aa = matrix_to_axis_angle
quat_mul = quaternion_multiply
quat_inv = quaternion_invert


# ============================================================
# AvatarPoser/EgoPoser/SAGE Compatible Functions
# ============================================================

def sixd2aa(pose_6d, batch=False):
    """
    Convert 6D rotation representation to axis-angle.
    Compatible with EgoPoser/AvatarPoser/SAGE utils_transform.sixd2aa
    
    Shape:
        batch=False:
            - Input: (N, 6) where N is number of rotations
            - Output: (N, 3)
        batch=True:
            - Input: (B, J, 6) where B is batch size, J is number of joints
            - Output: (B, J, 3)
    
    Args:
        pose_6d: 6D rotation tensor
        batch: If True, expects (B, J, 6) input and returns (B, J, 3)
               If False, expects (N, 6) input and returns (N, 3)
    
    Returns:
        axis-angle representation
    
    Example:
        >>> # Single batch of rotations
        >>> pose_6d = torch.randn(10, 6)  # 10 rotations
        >>> aa = sixd2aa(pose_6d)  # (10, 3)
        >>> 
        >>> # Batched joint rotations  
        >>> pose_6d = torch.randn(8, 22, 6)  # batch=8, joints=22
        >>> aa = sixd2aa(pose_6d, batch=True)  # (8, 22, 3)
    """
    if batch:
        # Input: (B, J, 6) -> reshape to (B*J, 6) -> convert -> reshape to (B, J, 3)
        original_shape = pose_6d.shape[:-1]  # (B, J)
        pose_6d_flat = pose_6d.reshape(-1, 6)  # (B*J, 6)
        mat = rotation_6d_to_matrix(pose_6d_flat)  # (B*J, 3, 3)
        aa_flat = matrix_to_axis_angle(mat)  # (B*J, 3)
        return aa_flat.reshape(*original_shape, 3)  # (B, J, 3)
    else:
        # Input: (N, 6) -> Output: (N, 3)
        mat = rotation_6d_to_matrix(pose_6d)  # (N, 3, 3) or (..., 3, 3)
        return matrix_to_axis_angle(mat)  # (N, 3) or (..., 3)


def aa2sixd(pose_aa):
    """
    Convert axis-angle to 6D rotation representation.
    Compatible with EgoPoser/AvatarPoser/SAGE utils_transform.aa2sixd
    
    Shape:
        - Input: (N, 3) or (..., 3)
        - Output: (N, 6) or (..., 6)
    
    Args:
        pose_aa: axis-angle tensor
    
    Returns:
        6D rotation representation
    """
    mat = axis_angle_to_matrix(pose_aa)  # (..., 3, 3)
    return matrix_to_rotation_6d(mat)  # (..., 6)


def sixd2quat(pose_6d):
    """
    Convert 6D rotation to quaternion.
    
    Shape:
        - Input: (N, 6) or (..., 6)
        - Output: (N, 4) or (..., 4) with format (w, x, y, z)
    
    Args:
        pose_6d: 6D rotation tensor
    
    Returns:
        quaternion with real part first (w, x, y, z)
    """
    mat = rotation_6d_to_matrix(pose_6d)  # (..., 3, 3)
    return matrix_to_quaternion(mat)  # (..., 4)


def quat2sixd(quat):
    """
    Convert quaternion to 6D rotation representation.
    
    Shape:
        - Input: (N, 4) or (..., 4) with format (w, x, y, z)
        - Output: (N, 6) or (..., 6)
    
    Args:
        quat: quaternion with real part first (w, x, y, z)
    
    Returns:
        6D rotation representation
    """
    mat = quaternion_to_matrix(quat)  # (..., 3, 3)
    return matrix_to_rotation_6d(mat)  # (..., 6)


# Aliases for backward compatibility
sixd2matrot = rotation_6d_to_matrix  # (N, 6) -> (N, 3, 3) with Gram-Schmidt
matrot2sixd = matrix_to_rotation_6d  # (N, 3, 3) -> (N, 6)

