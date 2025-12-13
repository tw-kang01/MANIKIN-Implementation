"""
MANIKIN Core - Quaternion-based Rotation Operations
ECCV 2024 MANIKIN 논문 구현

모든 내부 회전 계산을 Quaternion으로 통일
6D representation은 NN I/O용으로만 사용

Manikin 자체 rotation_utils 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Use Manikin's own rotation utils instead of SAGE
from Manikin.utils.rotation_utils import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_raw_multiply,
    quaternion_multiply,
    quaternion_invert,
    quaternion_apply,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    standardize_quaternion,
)

# ============================================================
# Constants
# ============================================================

EPS = 1e-7  # For numerical stability
EPS_ACOS = 1e-6  # For safe acos/asin


# ============================================================
# Quaternion Operations - Using SAGE utils
# All quaternions use (w, x, y, z) format (real part first)
# ============================================================

def quat_identity(batch_size, device):
    """Identity quaternion: (1, 0, 0, 0)"""
    return torch.tensor([[1., 0., 0., 0.]], device=device).expand(batch_size, -1).clone()


def quat_normalize(q):
    """Normalize quaternion to unit length"""
    return F.normalize(q, p=2, dim=-1, eps=EPS)


# Use SAGE's quaternion_invert directly
quat_conjugate = quaternion_invert


def quat_multiply(q1, q2):
    """
    Hamilton product using SAGE's quaternion_raw_multiply
    Clone inputs to avoid inplace modification issues with autograd
    """
    return quaternion_raw_multiply(q1.clone(), q2.clone())


def quat_rotate_vector(q, v):
    """
    Rotate vector v by quaternion q
    Using SAGE's quaternion_apply
    
    Args:
        q: (..., 4) unit quaternion
        v: (..., 3) vector
    Returns:
        (..., 3) rotated vector
    """
    return quaternion_apply(q, v)


def quat_from_axis_angle(axis, angle):
    """
    Create quaternion from axis and angle
    Using SAGE's axis_angle_to_quaternion
    
    Args:
        axis: (..., 3) unit axis
        angle: (...,) or (..., 1) angle in radians
    Returns:
        (..., 4) quaternion
    """
    if angle.dim() == axis.dim() - 1:
        angle = angle.unsqueeze(-1)
    
    # Convert to axis-angle format (axis * angle)
    axis_angle = axis * angle
    return axis_angle_to_quaternion(axis_angle)


def quat_from_two_vectors(v_from, v_to):
    """
    Compute quaternion that rotates v_from to v_to
    
    Args:
        v_from, v_to: (..., 3) vectors (will be normalized)
    Returns:
        (..., 4) quaternion
    """
    v_from = F.normalize(v_from, dim=-1, eps=EPS)
    v_to = F.normalize(v_to, dim=-1, eps=EPS)
    
    # Cross product = axis * sin(angle)
    cross = torch.linalg.cross(v_from, v_to, dim=-1)
    
    # Dot product = cos(angle)
    dot = (v_from * v_to).sum(dim=-1, keepdim=True)
    
    # For numerical stability, use: q = (1 + dot, cross) then normalize
    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)
    
    # Handle anti-parallel case (dot ≈ -1)
    is_antiparallel = dot < -1.0 + EPS_ACOS
    
    # For anti-parallel, use any perpendicular axis
    fallback_axis = torch.zeros_like(v_from)
    abs_v = torch.abs(v_from)
    min_idx = torch.argmin(abs_v, dim=-1, keepdim=True)
    fallback_axis.scatter_(-1, min_idx, 1.0)
    fallback_axis = torch.linalg.cross(v_from, fallback_axis, dim=-1)
    fallback_axis = F.normalize(fallback_axis, dim=-1, eps=EPS)
    
    # q for 180° rotation: (0, axis)
    q_antiparallel = F.pad(fallback_axis, (1, 0), value=0)
    
    q = torch.where(is_antiparallel.expand_as(q), q_antiparallel, q)
    
    return quat_normalize(q)


# Use SAGE's quaternion_to_matrix
quat_to_matrix = quaternion_to_matrix


# Use SAGE's matrix_to_quaternion
quat_from_matrix = matrix_to_quaternion


# ============================================================
# 6D Representation Conversions (for NN I/O only)
# Using SAGE's rotation_6d_to_matrix and matrix_to_rotation_6d
# ============================================================

def sixd_to_quat(sixd):
    """
    Convert 6D rotation representation to quaternion
    6D → Matrix (Gram-Schmidt) → Quaternion
    Using SAGE's functions
    
    Args:
        sixd: (..., 6) 6D rotation
    Returns:
        (..., 4) quaternion
    """
    matrix = rotation_6d_to_matrix(sixd)
    return matrix_to_quaternion(matrix)


def quat_to_sixd(q):
    """
    Convert quaternion to 6D rotation representation
    Quaternion → Matrix → 6D (first two columns)
    Using SAGE's functions
    
    Args:
        q: (..., 4) quaternion
    Returns:
        (..., 6) 6D rotation
    """
    matrix = quaternion_to_matrix(q)
    return matrix_to_rotation_6d(matrix)


def sixd_identity(batch_size, device):
    """
    Identity 6D rotation: first two columns of I
    [1, 0, 0, 0, 1, 0]
    """
    return torch.tensor([[1., 0., 0., 0., 1., 0.]], device=device).expand(batch_size, -1).clone()


# ============================================================
# SMPL-H Joint Indices
# ============================================================

SMPLH_JOINT_NAMES = [
    'Pelvis',      # 0 - Root
    'L_Hip',       # 1
    'R_Hip',       # 2
    'Spine1',      # 3
    'L_Knee',      # 4
    'R_Knee',      # 5
    'Spine2',      # 6
    'L_Ankle',     # 7
    'R_Ankle',     # 8
    'Spine3',      # 9
    'L_Foot',      # 10 (toe)
    'R_Foot',      # 11 (toe)
    'Neck',        # 12
    'L_Collar',    # 13
    'R_Collar',    # 14
    'Head',        # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',     # 18
    'R_Elbow',     # 19
    'L_Wrist',     # 20
    'R_Wrist',     # 21
]

# Kinematic parent indices
SMPLH_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip <- Pelvis
    0,   # 2: R_Hip <- Pelvis
    0,   # 3: Spine1 <- Pelvis
    1,   # 4: L_Knee <- L_Hip
    2,   # 5: R_Knee <- R_Hip
    3,   # 6: Spine2 <- Spine1
    4,   # 7: L_Ankle <- L_Knee
    5,   # 8: R_Ankle <- R_Knee
    6,   # 9: Spine3 <- Spine2
    7,   # 10: L_Foot <- L_Ankle
    8,   # 11: R_Foot <- R_Ankle
    9,   # 12: Neck <- Spine3
    9,   # 13: L_Collar <- Spine3
    9,   # 14: R_Collar <- Spine3
    12,  # 15: Head <- Neck
    13,  # 16: L_Shoulder <- L_Collar
    14,  # 17: R_Shoulder <- R_Collar
    16,  # 18: L_Elbow <- L_Shoulder
    17,  # 19: R_Elbow <- R_Shoulder
    18,  # 20: L_Wrist <- L_Elbow
    19,  # 21: R_Wrist <- R_Elbow
]

# Joint index constants for convenience
class JointIdx:
    PELVIS = 0
    L_HIP = 1
    R_HIP = 2
    SPINE1 = 3
    L_KNEE = 4
    R_KNEE = 5
    SPINE2 = 6
    L_ANKLE = 7
    R_ANKLE = 8
    SPINE3 = 9
    L_FOOT = 10
    R_FOOT = 11
    NECK = 12
    L_COLLAR = 13
    R_COLLAR = 14
    HEAD = 15
    L_SHOULDER = 16
    R_SHOULDER = 17
    L_ELBOW = 18
    R_ELBOW = 19
    L_WRIST = 20
    R_WRIST = 21
    
    # Grouped indices
    TORSO = [SPINE1, SPINE2, SPINE3, NECK]  # 3, 6, 9, 12
    LEFT_ARM = [L_SHOULDER, L_ELBOW, L_WRIST]  # 16, 18, 20
    RIGHT_ARM = [R_SHOULDER, R_ELBOW, R_WRIST]  # 17, 19, 21
    LEFT_LEG = [L_HIP, L_KNEE, L_ANKLE, L_FOOT]  # 1, 4, 7, 10
    RIGHT_LEG = [R_HIP, R_KNEE, R_ANKLE, R_FOOT]  # 2, 5, 8, 11
    COLLAR = [L_COLLAR, R_COLLAR]  # 13, 14
    
    # Sparse input joints (from HMD)
    SPARSE = [HEAD, L_WRIST, R_WRIST]  # 15, 20, 21


# ============================================================
# Safe Math Operations
# Using SAGE-style numerical stability patterns
# ============================================================

def safe_acos(x, eps=EPS_ACOS):
    """acos with clamped input to prevent NaN gradient"""
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def safe_asin(x, eps=EPS_ACOS):
    """asin with clamped input to prevent NaN gradient at ±1"""
    return torch.asin(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def safe_sqrt(x, eps=1e-12):
    """
    sqrt with bounded gradient near zero.
    Prevents NaN gradients when sqrt(0) is encountered.
    
    Key: sqrt(x).grad = 1/(2*sqrt(x)) → ∞ as x → 0
    Solution: Add eps before sqrt to bound gradient.
    """
    x_safe = torch.clamp(x, min=eps)
    return torch.sqrt(x_safe) * (x > 0).float()


def safe_normalize(v, dim=-1, eps=EPS):
    """Normalize vector with epsilon for zero-length handling"""
    return F.normalize(v, p=2, dim=dim, eps=eps)


def safe_div(a, b, eps=EPS):
    """Safe division to prevent division by zero"""
    return a / torch.clamp(b, min=eps)


# ============================================================
# Analytic IK Helper Functions (Quaternion-based)
# MANIKIN Eq. 7-14
# ============================================================

def compute_swivel_frame(p_base, p_end, v_ref):
    """
    Compute the local coordinate frame for swivel angle (Paper Eq. 2)

    Paper Eq. 2:
        n = (p_e - p_b) / ||p_e - p_b||
        u = (-v_ref + (v_ref · n) * n) / ||-v_ref + (v_ref · n) * n||
        v = u × n

    Args:
        p_base: (batch, 3) base joint position (shoulder/hip)
        p_end: (batch, 3) end joint position (wrist/ankle)
        v_ref: (3,) reference direction (gravity = -Z for Z-up)

    Returns:
        n: (batch, 3) unit vector from base to end
        u: (batch, 3) perpendicular to n, projection of -v_ref onto orbit plane
        v: (batch, 3) perpendicular to both n and u (v = u × n)
    """
    device = p_base.device
    v_ref = v_ref.to(device)

    vec = p_end - p_base
    n = safe_normalize(vec)

    # Eq. 2: u = (-v_ref + (v_ref · n) * n) / ||...||
    # Project -v_ref onto plane perpendicular to n
    dot = (v_ref * n).sum(dim=-1, keepdim=True)
    u = -v_ref + dot * n  # FIXED: use -v_ref (paper Eq. 2)
    u = safe_normalize(u)

    # Eq. 2: v = u × n (right-handed coordinate system)
    v = torch.linalg.cross(u, n, dim=-1)  # FIXED: cross(u, n), not cross(n, u)

    return n, u, v


def compute_mid_joint_position(p_base, p_end, L1, L2, cos_phi, sin_phi, v_ref):
    """
    Compute mid-joint (elbow/knee) position from swivel angle (Eq. 3, 4, 7)
    
    Args:
        p_base: (batch, 3) base joint (shoulder/hip)
        p_end: (batch, 3) end joint (wrist/ankle)
        L1: (batch,) or float, first bone length (humerus/femur)
        L2: (batch,) or float, second bone length (radius/tibia)
        cos_phi, sin_phi: (batch,) swivel angle
        v_ref: (3,) reference direction
    
    Returns:
        p_mid: (batch, 3) mid joint position (elbow/knee)
    """
    n, u, v = compute_swivel_frame(p_base, p_end, v_ref)
    
    # Ensure L1, L2 are tensors with proper shape
    if not isinstance(L1, torch.Tensor):
        L1 = torch.tensor(L1, device=p_base.device, dtype=p_base.dtype)
    if not isinstance(L2, torch.Tensor):
        L2 = torch.tensor(L2, device=p_base.device, dtype=p_base.dtype)
    
    # Make L1, L2 broadcastable: (batch,) -> (batch, 1)
    if L1.dim() == 1:
        L1 = L1.unsqueeze(-1)  # (batch, 1)
    if L2.dim() == 1:
        L2 = L2.unsqueeze(-1)  # (batch, 1)
    
    # Distance from base to end
    d = torch.norm(p_end - p_base, dim=-1, keepdim=True)  # (batch, 1)
    d_safe = torch.clamp(d, min=EPS)
    
    # Handle degenerate case: when d is very small (base ≈ end)
    is_degenerate = d < EPS
    
    # Eq. 3: cos(α) = (L1² + d² - L2²) / (2 * L1 * d)
    cos_alpha = (L1**2 + d_safe**2 - L2**2) / (2 * L1 * d_safe)  # (batch, 1)
    cos_alpha = torch.clamp(cos_alpha, -1.0 + EPS_ACOS, 1.0 - EPS_ACOS)
    sin_alpha = safe_sqrt(1.0 - cos_alpha**2)  # (batch, 1)
    
    # Eq. 4: Orbit center and radius
    # L1 * cos_alpha: (batch, 1), n: (batch, 3) -> broadcast correctly
    p_c = p_base + (L1 * cos_alpha) * n  # (batch, 3)
    r_c = L1 * sin_alpha  # (batch, 1)
    
    # Eq. 7: Mid joint on orbit circle
    p_mid = p_c + r_c * (cos_phi.unsqueeze(-1) * u + sin_phi.unsqueeze(-1) * v)
    
    # For degenerate cases, return base position
    p_mid = torch.where(is_degenerate.expand_as(p_mid), p_base, p_mid)
    
    return p_mid


def compute_flexion_angle(L1, L2, d):
    """
    Compute flexion angle at mid joint (Eq. 9)
    
    논문 Eq. 9:
        θ_flexion = π - arccos((l_bm² + l_me² - ||p_e - p_b||²) / (2 * l_bm * l_me))
    
    여기서:
        l_bm = L1 (base to mid, e.g., humerus/femur)
        l_me = L2 (mid to end, e.g., radius/tibia)
        d = ||p_e - p_b|| (distance from base to end)
    
    Args:
        L1, L2: (batch,) or float - bone lengths (l_bm, l_me)
        d: (batch,) or (batch, 1) distance from base to end joint
    
    Returns:
        theta: (batch,) flexion angle in radians
    """
    # Ensure d is 1D
    if d.dim() > 1:
        d = d.squeeze(-1)
    
    # Ensure L1, L2 are tensors
    if not isinstance(L1, torch.Tensor):
        L1 = torch.tensor(L1, device=d.device, dtype=d.dtype)
    if not isinstance(L2, torch.Tensor):
        L2 = torch.tensor(L2, device=d.device, dtype=d.dtype)
    
    # Squeeze if needed
    if L1.dim() > 1:
        L1 = L1.squeeze(-1)
    if L2.dim() > 1:
        L2 = L2.squeeze(-1)
    
    # Eq. 9: cos(θ_internal) = (L1² + L2² - d²) / (2 * L1 * L2)
    cos_theta = (L1**2 + L2**2 - d**2) / (2 * L1 * L2 + EPS)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # θ_flexion = π - θ_internal (flexion is the external angle)
    theta_flexion = math.pi - safe_acos(cos_theta)
    
    return theta_flexion


def compute_swing_quat(v_from, v_to):
    """
    Compute quaternion for swing rotation (Eq. 10)
    
    논문 Eq. 10:
        axis = (p_im - p_b) × (p_m - p_b) / ||(p_im - p_b) × (p_m - p_b)||
        angle = arccos((p_im - p_b) · (p_m - p_b) / l²_bm)
    
    즉, v_from에서 v_to로 회전시키는 quaternion
    
    Handles edge cases:
        - Parallel vectors (angle ≈ 0): return identity quaternion
        - Anti-parallel vectors (angle ≈ π): use perpendicular fallback axis
    
    Args:
        v_from, v_to: (batch, 3) vectors (T-pose 방향, 실제 방향)
    Returns:
        q: (batch, 4) quaternion
    """
    v_from_norm = safe_normalize(v_from)
    v_to_norm = safe_normalize(v_to)
    
    # Cross product = rotation axis × sin(angle)
    cross = torch.linalg.cross(v_from_norm, v_to_norm, dim=-1)
    cross_norm = torch.norm(cross, dim=-1, keepdim=True)
    
    # Dot product = cos(angle)
    dot = (v_from_norm * v_to_norm).sum(dim=-1, keepdim=True)
    
    # === Handle edge cases ===
    batch = v_from.shape[0]
    device = v_from.device
    dtype = v_from.dtype
    
    # Case 1: Nearly parallel (angle ≈ 0) -> identity quaternion
    is_parallel = cross_norm.squeeze(-1) < EPS
    
    # Case 2: Nearly anti-parallel (dot ≈ -1) -> 180° rotation
    is_antiparallel = dot.squeeze(-1) < -1.0 + EPS_ACOS
    
    # For anti-parallel, find perpendicular axis
    # Use the axis with smallest component of v_from
    abs_v = torch.abs(v_from_norm)
    # Create fallback axis perpendicular to v_from
    fallback = torch.zeros_like(v_from_norm)
    min_idx = torch.argmin(abs_v, dim=-1, keepdim=True)
    fallback.scatter_(-1, min_idx, 1.0)
    fallback_axis = torch.linalg.cross(v_from_norm, fallback, dim=-1)
    fallback_axis = safe_normalize(fallback_axis)
    
    # Normal case: rotation axis (normalized)
    axis = safe_normalize(cross)
    
    # Rotation angle from dot product
    angle = safe_acos(dot.squeeze(-1))  # (batch,)
    
    # Compute quaternion for normal case
    q_normal = quat_from_axis_angle(axis, angle)
    
    # Identity quaternion for parallel case
    q_identity = torch.tensor([1., 0., 0., 0.], device=device, dtype=dtype).expand(batch, -1)
    
    # 180° rotation quaternion for anti-parallel case: (0, axis)
    q_antiparallel = torch.cat([
        torch.zeros(batch, 1, device=device, dtype=dtype),
        fallback_axis
    ], dim=-1)
    
    # Select appropriate quaternion
    q = torch.where(
        is_parallel.unsqueeze(-1).expand(-1, 4),
        q_identity,
        torch.where(
            is_antiparallel.unsqueeze(-1).expand(-1, 4),
            q_antiparallel,
            q_normal
        )
    )
    
    return quat_normalize(q)


def compute_twist_angle(p_mid, p_end, p_end_after_swing, r_orbit, twist_axis=None):
    """
    Compute SIGNED twist angle around limb axis (Eq. 11, 12)

    논문 Eq. 11: r_ec = l_me * sin(θ_flexion)  (orbit radius)
    논문 Eq. 12: θ_twist = 2 * arcsin(||p_e - p_k_e|| / (2 * r_ec))

    Args:
        p_mid: (batch, 3) mid joint position (elbow/knee)
        p_end: (batch, 3) actual end position (wrist/ankle)
        p_end_after_swing: (batch, 3) end position after swing only (p_k_e)
        r_orbit: (batch,) or (batch, 1) orbit radius (r_ec from Eq. 11)
        twist_axis: (batch, 3) optional - limb axis to determine rotation sign
                    If None, returns unsigned magnitude only

    Returns:
        theta: (batch,) SIGNED twist angle in radians (if twist_axis provided)
    """
    if r_orbit.dim() > 1:
        r_orbit = r_orbit.squeeze(-1)

    # ||p_e - p_k_e|| = chord length between actual and swing-only positions
    chord = torch.norm(p_end - p_end_after_swing, dim=-1)

    # Eq. 12: θ_twist = 2 * arcsin(chord / (2 * r_ec))
    # Safely compute sin(θ/2) = chord / (2 * r_ec)
    r_safe = torch.clamp(r_orbit, min=EPS)
    sin_half = chord / (2 * r_safe)

    # θ = 2 * arcsin(sin(θ/2)) - using safe_asin to prevent NaN gradient
    theta_mag = 2.0 * safe_asin(sin_half)

    # Determine sign using cross product if twist_axis is provided
    if twist_axis is not None:
        # vec1: p_mid → p_end_after_swing (reference direction)
        # vec2: p_mid → p_end (actual direction)
        vec1 = p_end_after_swing - p_mid
        vec2 = p_end - p_mid
        # Cross product gives rotation direction
        cross = torch.linalg.cross(vec1, vec2, dim=-1)
        # Sign = dot(cross, twist_axis) > 0 means positive rotation
        sign = torch.sign((cross * twist_axis).sum(dim=-1))
        # Handle zero case (no rotation)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        theta = theta_mag * sign
    else:
        theta = theta_mag

    return theta
