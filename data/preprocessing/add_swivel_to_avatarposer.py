"""
Add Swivel Angles to AvatarPoser Data

This script augments AvatarPoser's processed PKL files with:
- joint_positions: (T, 22, 3) - recomputed from body_parms via FK
- arm_swivel_cos_sin: (T, 4) - [l_cos, l_sin, r_cos, r_sin]
- leg_swivel_cos_sin: (T, 4) - [l_cos, l_sin, r_cos, r_sin]
- bone_lengths: dict - per-subject bone lengths computed from T-pose
- betas: (16,) - body shape parameters (if available, else zeros)

V2 Update: Now computes per-subject bone_lengths using betas from pkl data.
           If betas not available, falls back to neutral body (betas=0).
"""

import sys
import os
import torch
import pickle
import glob
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append('.')
sys.path.append('AvatarPoser')

from human_body_prior.body_model.body_model import BodyModel


def compute_swivel_angle_batch(p_root, p_mid, p_effector, v_ref):
    """
    Compute swivel angle (cos, sin) for a limb chain.

    Args:
        p_root: (T, 3) shoulder/hip positions
        p_mid: (T, 3) elbow/knee positions
        p_effector: (T, 3) wrist/ankle positions
        v_ref: (3,) reference direction (gravity = -Z)

    Returns:
        cos_phi: (T,)
        sin_phi: (T,)
    """
    EPS = 1e-8
    device = p_root.device
    v_ref = v_ref.to(device)

    # Direction from root to effector (n-axis)
    d = p_effector - p_root  # (T, 3)
    d_norm = torch.norm(d, dim=-1, keepdim=True).clamp(min=EPS)
    n = d / d_norm  # (T, 3)

    # Project v_ref onto plane perpendicular to n (u-axis)
    v_dot_n = (v_ref.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
    u = v_ref.unsqueeze(0) - v_dot_n * n
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=EPS)
    u = u / u_norm  # (T, 3)

    # v = n × u (completes right-handed frame)
    v = torch.cross(n, u, dim=-1)
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=EPS)
    v = v / v_norm  # (T, 3)

    # Mid joint position relative to root
    e = p_mid - p_root  # (T, 3)

    # Project e onto plane perpendicular to n
    e_along_n = (e * n).sum(dim=-1, keepdim=True) * n
    e_perp = e - e_along_n
    e_perp_norm = torch.norm(e_perp, dim=-1, keepdim=True).clamp(min=EPS)
    e_hat = e_perp / e_perp_norm  # (T, 3)

    # Swivel angle: cos(phi) = e_hat · u, sin(phi) = e_hat · v
    cos_phi = (e_hat * u).sum(dim=-1)
    sin_phi = (e_hat * v).sum(dim=-1)

    return cos_phi, sin_phi


def add_swivel_to_avatarposer(
    input_dir="AvatarPoser/data_fps60",
    output_dir="data_manikin",
    subsets=["BioMotionLab_NTroje", "CMU", "MPI_HDM05"],
):
    """
    Augment AvatarPoser PKL files with swivel angles and joint positions.
    """
    # Load body model (neutral - no betas, like AvatarPoser)
    support_dir = 'AvatarPoser/support_data/'
    bm_fname = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
    dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/male/model.npz')

    print("Loading body model...")
    bm = BodyModel(
        bm_fname=bm_fname,
        num_betas=16,
        num_dmpls=8,
        dmpl_fname=dmpl_fname
    )

    # Reference vectors for swivel angle computation
    # Arms: gravity (-Z) is safe since arms point sideways in most poses
    # Legs: forward (+Y) is safe since legs point down in most poses
    # Using the same v_ref for legs causes NaN when standing (legs parallel to -Z)
    v_ref_arm = torch.tensor([0., 0., -1.])  # Gravity direction
    v_ref_leg = torch.tensor([0., 1., 0.])   # Forward direction

    # Joint indices (SMPL-H 22 joints)
    L_SHOULDER, R_SHOULDER = 16, 17
    L_ELBOW, R_ELBOW = 18, 19
    L_WRIST, R_WRIST = 20, 21
    L_HIP, R_HIP = 1, 2
    L_KNEE, R_KNEE = 4, 5
    L_ANKLE, R_ANKLE = 7, 8

    total_processed = 0

    for subset in subsets:
        for phase in ["train", "test"]:
            input_path = os.path.join(input_dir, subset, phase)
            output_path = os.path.join(output_dir, subset, phase)

            if not os.path.exists(input_path):
                print(f"Skipping {input_path} (not found)")
                continue

            os.makedirs(output_path, exist_ok=True)

            pkl_files = sorted(glob.glob(os.path.join(input_path, "*.pkl")))
            print(f"\nProcessing {subset}/{phase}: {len(pkl_files)} files")

            for pkl_file in tqdm(pkl_files, desc=f"{subset}/{phase}"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)

                    # Get body params
                    body_parms = data['body_parms_list']

                    # Recompute joint positions via FK (NO betas - neutral body)
                    with torch.no_grad():
                        body_output = bm(**{k: v for k, v in body_parms.items()
                                           if k in ['pose_body', 'root_orient', 'trans']})
                        joint_positions = body_output.Jtr[:, :22, :]  # (T, 22, 3)

                    # Skip first frame (like AvatarPoser)
                    joint_pos = joint_positions[1:]  # (T-1, 22, 3)

                    # Compute swivel angles
                    # Left arm: shoulder(16) → elbow(18) → wrist(20)
                    la_cos, la_sin = compute_swivel_angle_batch(
                        joint_pos[:, L_SHOULDER],
                        joint_pos[:, L_ELBOW],
                        joint_pos[:, L_WRIST],
                        v_ref_arm
                    )

                    # Right arm: shoulder(17) → elbow(19) → wrist(21)
                    ra_cos, ra_sin = compute_swivel_angle_batch(
                        joint_pos[:, R_SHOULDER],
                        joint_pos[:, R_ELBOW],
                        joint_pos[:, R_WRIST],
                        v_ref_arm
                    )

                    # Left leg: hip(1) → knee(4) → ankle(7)
                    # Uses v_ref_leg (+Y forward) to avoid singularity when standing
                    ll_cos, ll_sin = compute_swivel_angle_batch(
                        joint_pos[:, L_HIP],
                        joint_pos[:, L_KNEE],
                        joint_pos[:, L_ANKLE],
                        v_ref_leg
                    )

                    # Right leg: hip(2) → knee(5) → ankle(8)
                    # Uses v_ref_leg (+Y forward) to avoid singularity when standing
                    rl_cos, rl_sin = compute_swivel_angle_batch(
                        joint_pos[:, R_HIP],
                        joint_pos[:, R_KNEE],
                        joint_pos[:, R_ANKLE],
                        v_ref_leg
                    )

                    # Pack swivel angles
                    arm_swivel_cos_sin = torch.stack([la_cos, la_sin, ra_cos, ra_sin], dim=-1)
                    leg_swivel_cos_sin = torch.stack([ll_cos, ll_sin, rl_cos, rl_sin], dim=-1)

                    # === Compute bone_lengths with per-subject betas ===
                    betas = data.get('betas', None)
                    if betas is not None and not isinstance(betas, torch.Tensor):
                        betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)  # (1, 16)

                    with torch.no_grad():
                        if betas is not None:
                            tpose_output = bm(betas=betas)
                        else:
                            tpose_output = bm()  # Neutral body fallback
                        tpose_joints = tpose_output.Jtr[0, :22, :]  # (22, 3)

                    bone_lengths = {
                        'left_humerus': torch.norm(tpose_joints[L_ELBOW] - tpose_joints[L_SHOULDER]).item(),
                        'left_radius': torch.norm(tpose_joints[L_WRIST] - tpose_joints[L_ELBOW]).item(),
                        'right_humerus': torch.norm(tpose_joints[R_ELBOW] - tpose_joints[R_SHOULDER]).item(),
                        'right_radius': torch.norm(tpose_joints[R_WRIST] - tpose_joints[R_ELBOW]).item(),
                        'left_femur': torch.norm(tpose_joints[L_KNEE] - tpose_joints[L_HIP]).item(),
                        'left_tibia': torch.norm(tpose_joints[L_ANKLE] - tpose_joints[L_KNEE]).item(),
                        'right_femur': torch.norm(tpose_joints[R_KNEE] - tpose_joints[R_HIP]).item(),
                        'right_tibia': torch.norm(tpose_joints[R_ANKLE] - tpose_joints[R_KNEE]).item(),
                    }

                    # Add new fields to data
                    data['joint_positions'] = joint_pos.numpy()  # (T-1, 22, 3)
                    data['arm_swivel_cos_sin'] = arm_swivel_cos_sin.numpy()  # (T-1, 4)
                    data['leg_swivel_cos_sin'] = leg_swivel_cos_sin.numpy()  # (T-1, 4)
                    data['bone_lengths'] = bone_lengths
                    data['betas'] = betas.squeeze(0).numpy() if betas is not None else np.zeros(16)

                    # Save augmented data
                    output_file = os.path.join(output_path, os.path.basename(pkl_file))
                    with open(output_file, 'wb') as f:
                        pickle.dump(data, f)

                    total_processed += 1

                except Exception as e:
                    print(f"Error processing {pkl_file}: {e}")
                    continue

    print(f"\n{'='*60}")
    print(f"Done! Processed {total_processed} files")
    print(f"Output saved to: {output_dir}")
    print('='*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='AvatarPoser/data_fps60')
    parser.add_argument('--output_dir', type=str, default='data_manikin')
    parser.add_argument('--subsets', nargs='+',
                        default=['BioMotionLab_NTroje', 'CMU', 'MPI_HDM05'])
    args = parser.parse_args()

    add_swivel_to_avatarposer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subsets=args.subsets,
    )
