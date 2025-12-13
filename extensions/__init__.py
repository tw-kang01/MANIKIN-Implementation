"""
MANIKIN Extensions
ECCV 2024 MANIKIN - Biomechanically accurate neural IK

Current Architecture (V5 - AvatarJLM based):
- MANIKINNetworkJLM: AlternativeST Transformer (62D output)
  - torso: 42D (pelvis + spine1-3 + neck + collar)
  - swivel: 8D (arm + leg)
  - foot: 12D (L/R ankle)
- TorsoFK: Forward kinematics for shoulder/hip positions
- AnalyticSolver: IK for arms and legs

Data Flow:
  sparse_input (54D) → Network → torso (42D) + swivel (8D) + foot (12D)
                                    ↓
                              TorsoFK → shoulder/hip positions
                                    ↓
                              AnalyticIK → full body pose
"""

# ============================================================
# New Network (V5 - AvatarJLM based)
# ============================================================
from .manikin_network import MANIKINNetworkJLM, AlternativeST

# ============================================================
# Integration Model (V5)
# ============================================================
try:
    from .manikin_model import MANIKINModelJLM, to_local_rotation
except ImportError:
    pass

# ============================================================
# Core Components
# ============================================================
from .analytic_solver import AnalyticArmSolver, AnalyticLegSolver
from .torso_fk import TorsoFK, TorsoFKWithBodyModel

from .manikin_core import (
    quat_identity, quat_normalize, quat_multiply, quat_conjugate,
    quat_rotate_vector, quat_to_sixd, sixd_to_quat,
    JointIdx, SMPLH_PARENTS
)

# ============================================================
# Helper functions
# ============================================================
try:
    from .swivel_gt import compute_swivel_angle_gt, compute_bone_lengths_from_tpose, predict_mid_joint_from_swivel
except ImportError:
    pass

try:
    from .body_output_utils import get_joint_positions, get_vertices, get_faces, safe_to_numpy, BodyOutputWrapper
except ImportError:
    pass

__all__ = [
    # New Network (V5)
    'MANIKINNetworkJLM',
    'AlternativeST',
    # Core components
    'AnalyticArmSolver',
    'AnalyticLegSolver',
    'TorsoFK',
    'TorsoFKWithBodyModel',
    # Quaternion utils
    'quat_identity',
    'quat_normalize',
    'quat_multiply',
    'quat_conjugate',
    'quat_rotate_vector',
    'quat_to_sixd',
    'sixd_to_quat',
    'JointIdx',
    'SMPLH_PARENTS',
]
