"""
MANIKIN Utils Package

Includes:
- manikin_logger: Training logger with forward/backward tracking
- manikin_loss_module: Eq. 15 loss computation
- rotation_utils: SAGE/AvatarPoser rotation functions
"""

from .manikin_logger import MANIKINLogger, GradientTracker, create_logger
# NOTE: manikin_loss_module is NOT imported here to avoid circular imports
# Import directly: from utils.manikin_loss_module import MANIKINLossJLM
from .rotation_utils import (
    # SAGE quaternion functions
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_multiply,
    quaternion_invert,
    quaternion_apply,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    
    # Safe math
    safe_acos,
    safe_asin,
    safe_sqrt,
    safe_normalize,
    safe_div,
    compute_angle_atan2,
    compute_swing_rotation,
    _sqrt_positive_part,
    
    # AvatarPoser style
    bgs,
    sixd2matrot,
    matrot2sixd,
    
    # Aliases
    aa_to_quat,
    quat_to_aa,
    aa_to_rotmat,
    rotmat_to_aa,
    quat_mul,
    quat_inv,
)

__all__ = [
    # Logger
    'MANIKINLogger',
    'GradientTracker',
    'create_logger',

    # Rotation
    'quaternion_to_matrix',
    'matrix_to_quaternion',
    'axis_angle_to_quaternion',
    'quaternion_to_axis_angle',
    'axis_angle_to_matrix',
    'matrix_to_axis_angle',
    'quaternion_multiply',
    'quaternion_invert',
    'quaternion_apply',
    'rotation_6d_to_matrix',
    'matrix_to_rotation_6d',
    
    # Safe math
    'safe_acos',
    'safe_asin',
    'safe_sqrt',
    'safe_normalize',
    'safe_div',
    'compute_angle_atan2',
    'compute_swing_rotation',
    '_sqrt_positive_part',
    
    # AvatarPoser
    'bgs',
    'sixd2matrot',
    'matrot2sixd',
]
