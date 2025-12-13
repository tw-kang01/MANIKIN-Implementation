"""
MANIKIN Data Preparation
Based on EgoPoser prepare_data.py with MANIKIN-specific additions:
- swivel angle GT computation
- joint positions (with subject betas for accuracy)
- bone lengths (per-subject from T-pose with betas)

Key: Uses subject betas (like EgoPoser) for accurate joint positions
     This gives ~48mm better accuracy than neutral body approach
"""

import sys
import os
import torch
import numpy as np
import pickle

# Add paths
sys.path.append('AvatarPoser')
sys.path.append('.')

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from utils import utils_transform


def compute_swivel_angle_gt_batch(p_root, p_mid, p_effector, v_ref):
    """
    Batched swivel angle computation (Paper Eq. 2-5)

    Paper Eq. 2:
        n = (p_e - p_b) / ||p_e - p_b||
        u = (-v_ref + (v_ref · n) * n) / ||-v_ref + (v_ref · n) * n||
        v = u × n

    Args:
        p_root: (T, 3) shoulder/hip positions (p_b)
        p_mid: (T, 3) elbow/knee positions (p_m)
        p_effector: (T, 3) wrist/ankle positions (p_e)
        v_ref: (3,) reference direction (-Z for arm, +Y for leg)

    Returns:
        cos_phi: (T,)
        sin_phi: (T,)
    """
    EPS = 1e-8
    device = p_root.device
    v_ref = v_ref.to(device)

    # Eq. 2: n = (p_e - p_b) / ||p_e - p_b||
    d = p_effector - p_root  # (T, 3)
    d_norm = torch.norm(d, dim=-1, keepdim=True).clamp(min=EPS)
    n = d / d_norm  # (T, 3)

    # Eq. 2: u = (-v_ref + (v_ref · n) * n) / ||...||
    # Project -v_ref onto plane perpendicular to n
    v_dot_n = (v_ref.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
    u = -v_ref.unsqueeze(0) + v_dot_n * n  # FIXED: use -v_ref (paper Eq. 2)
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=EPS)
    u = u / u_norm  # (T, 3)

    # Eq. 2: v = u × n (right-handed coordinate system)
    v = torch.cross(u, n, dim=-1)  # FIXED: cross(u, n), not cross(n, u)
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


def compute_bone_lengths_from_positions(positions):
    """
    Compute bone lengths from joint positions

    Args:
        positions: (22, 3) T-pose joint positions

    Returns:
        dict with bone lengths
    """
    return {
        'left_humerus': torch.norm(positions[18] - positions[16]).item(),   # L_Elbow - L_Shoulder
        'left_radius': torch.norm(positions[20] - positions[18]).item(),    # L_Wrist - L_Elbow
        'right_humerus': torch.norm(positions[19] - positions[17]).item(),  # R_Elbow - R_Shoulder
        'right_radius': torch.norm(positions[21] - positions[19]).item(),   # R_Wrist - R_Elbow
        'left_femur': torch.norm(positions[4] - positions[1]).item(),       # L_Knee - L_Hip
        'left_tibia': torch.norm(positions[7] - positions[4]).item(),       # L_Ankle - L_Knee
        'right_femur': torch.norm(positions[5] - positions[2]).item(),      # R_Knee - R_Hip
        'right_tibia': torch.norm(positions[8] - positions[5]).item(),      # R_Ankle - R_Knee
    }


def prepare_manikin_data(
    dataroot_amass="AvatarPoser/amass",
    output_dir="data_manikin",
    subsets=["BioMotionLab_NTroje"],
):
    """
    Prepare MANIKIN data based on EgoPoser format (with subject betas)

    Output per file:
        - rotation_local_full_gt_list: (T, 132) 6D rotations for 22 joints
        - hmd_position_global_full_gt_list: (T, 54) sparse input
        - head_global_trans_list: (T, 4, 4) head transformation
        - body_parms_list: dict with root_orient, pose_body, trans
        - betas: (16,) subject-specific body shape parameters
        - arm_swivel_cos_sin: (T, 4) arm swivel angles [l_cos, l_sin, r_cos, r_sin]
        - leg_swivel_cos_sin: (T, 4) leg swivel angles [l_cos, l_sin, r_cos, r_sin]
        - joint_positions: (T, 22, 3) joint positions (computed WITH betas)
        - bone_lengths: dict (per-subject, from T-pose with betas)
        - framerate, gender, filepath
    """

    # Body model (with betas support, like EgoPoser)
    support_dir = 'AvatarPoser/support_data/'
    bm_fname_male = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
    dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/male/model.npz')

    num_betas = 16
    num_dmpls = 8

    print("Loading body model...")
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male
    )

    # Note: Bone lengths will be computed per-subject with their betas
    print("Bone lengths will be computed per-subject using betas")

    # Reference vectors for swivel angle computation
    # Arms: gravity (-Z) is safe since arms point sideways in most poses
    # Legs: forward (+Y) is safe since legs point down in most poses
    # Using the same v_ref for legs causes NaN when standing (legs parallel to -Z)
    v_ref_arm = torch.tensor([0., 0., -1.])  # Gravity direction
    v_ref_leg = torch.tensor([0., 1., 0.])   # Forward direction

    for dataroot_subset in subsets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataroot_subset}")
        print('='*60)

        for phase in ["train", "test"]:
            print(f"\n--- Phase: {phase} ---")

            savedir = os.path.join(output_dir, dataroot_subset, phase)
            os.makedirs(savedir, exist_ok=True)

            # Load split file
            split_file_paths = [
                os.path.join("AvatarPoser/data_split", dataroot_subset, f"{phase}_split.txt"),
                os.path.join("EgoPoser/data_split", dataroot_subset, f"{phase}_split.txt"),
            ]

            filepaths = []
            for split_file in split_file_paths:
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        filepaths = [line.rstrip('\n') for line in f]
                    print(f"Loaded split file: {split_file}")
                    break

            if not filepaths:
                print(f"Warning: No split file found for {dataroot_subset}/{phase}")
                continue

            idx = 0
            for filepath in filepaths:
                try:
                    # Normalize path separators for Windows
                    filepath = filepath.replace('/', os.sep).replace('\\', os.sep)

                    # Handle relative paths
                    if not os.path.isabs(filepath):
                        possible_paths = [
                            filepath,
                            os.path.join(dataroot_amass, filepath),
                            os.path.join("AvatarPoser", filepath),
                        ]
                        for p in possible_paths:
                            p = p.replace('/', os.sep).replace('\\', os.sep)
                            if os.path.exists(p):
                                filepath = p
                                break

                    if not os.path.exists(filepath):
                        continue

                    bdata = np.load(filepath, allow_pickle=True)

                    # Skip shape files
                    try:
                        framerate = bdata["mocap_framerate"]
                    except:
                        continue

                    # Stride for 60fps output
                    if framerate == 120:
                        stride = 2
                    elif framerate == 60:
                        stride = 1
                    else:
                        stride = max(1, int(framerate / 60))

                    bdata_poses = bdata["poses"][::stride, ...]
                    bdata_trans = bdata["trans"][::stride, ...]
                    subject_gender = str(bdata.get("gender", "male"))
                    if isinstance(subject_gender, bytes):
                        subject_gender = subject_gender.decode('utf-8')

                    # Load betas (subject-specific body shape) - like EgoPoser
                    bdata_betas = bdata.get("betas", np.zeros(num_betas))
                    if isinstance(bdata_betas, np.ndarray):
                        bdata_betas = bdata_betas[:num_betas]  # Ensure 16 dims
                    betas_tensor = torch.Tensor(bdata_betas).unsqueeze(0)  # (1, 16)

                    # Skip short sequences
                    if bdata_poses.shape[0] < 50:
                        continue

                    # Use male model (like AvatarPoser/EgoPoser)
                    bm = bm_male

                    body_parms = {
                        'root_orient': torch.Tensor(bdata_poses[:, :3]),
                        'pose_body': torch.Tensor(bdata_poses[:, 3:66]),
                        'trans': torch.Tensor(bdata_trans),
                    }

                    # Forward kinematics WITH BETAS (like EgoPoser) for accurate joint positions
                    # Expand betas to match sequence length
                    T = bdata_poses.shape[0]
                    betas_expanded = betas_tensor.expand(T, -1)  # (T, 16)

                    body_pose_world = bm(
                        pose_body=body_parms['pose_body'],
                        root_orient=body_parms['root_orient'],
                        trans=body_parms['trans'],
                        betas=betas_expanded
                    )

                    # Compute per-subject bone lengths from T-pose with betas
                    with torch.no_grad():
                        tpose_output = bm(
                            root_orient=torch.zeros(1, 3),
                            pose_body=torch.zeros(1, 63),
                            trans=torch.zeros(1, 3),
                            betas=betas_tensor
                        )
                        tpose_joints = tpose_output.Jtr[0, :22]
                    bone_lengths = compute_bone_lengths_from_positions(tpose_joints)

                    # ====== rotation_local_full_gt_list (132D) ======
                    output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1, 3)
                    output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0], -1)
                    rotation_local_full_gt_list = output_6d[1:]  # Skip first frame

                    # ====== Global rotations ======
                    rotation_local_matrot = aa2matrot(
                        torch.tensor(bdata_poses, dtype=torch.float32).reshape(-1, 3)
                    ).reshape(bdata_poses.shape[0], -1, 9)

                    rotation_global_matrot = local2global_pose(
                        rotation_local_matrot,
                        bm.kintree_table[0].long()
                    )

                    head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

                    # Global 6D rotations
                    rotation_global_6d = utils_transform.matrot2sixd(
                        rotation_global_matrot.reshape(-1, 3, 3)
                    ).reshape(rotation_global_matrot.shape[0], -1, 6)
                    input_rotation_global_6d = rotation_global_6d[1:, [15, 20, 21], :]

                    # Rotation velocity
                    rotation_velocity_global_matrot = torch.matmul(
                        torch.inverse(rotation_global_matrot[:-1]),
                        rotation_global_matrot[1:]
                    )
                    rotation_velocity_global_6d = utils_transform.matrot2sixd(
                        rotation_velocity_global_matrot.reshape(-1, 3, 3)
                    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
                    input_rotation_velocity_global_6d = rotation_velocity_global_6d[:, [15, 20, 21], :]

                    # ====== Joint positions ======
                    position_global_full_gt_world = body_pose_world.Jtr[:, :22, :]

                    # Head transformation
                    head_global_trans = torch.eye(4).repeat(position_global_full_gt_world.shape[0], 1, 1)
                    head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
                    head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]
                    head_global_trans_list = head_global_trans[1:]

                    # ====== 54D Sparse Input ======
                    num_frames = position_global_full_gt_world.shape[0] - 1

                    hmd_position_global_full_gt_list = torch.cat([
                        input_rotation_global_6d.reshape(num_frames, -1),           # 0-18
                        input_rotation_velocity_global_6d.reshape(num_frames, -1),  # 18-36
                        position_global_full_gt_world[1:, [15, 20, 21], :].reshape(num_frames, -1),  # 36-45
                        (position_global_full_gt_world[1:, [15, 20, 21], :] -
                         position_global_full_gt_world[:-1, [15, 20, 21], :]).reshape(num_frames, -1)  # 45-54
                    ], dim=-1)

                    # ====== MANIKIN: Swivel Angles ======
                    joint_pos = position_global_full_gt_world[1:]  # Skip first frame

                    # Left arm: shoulder(16) → elbow(18) → wrist(20)
                    la_cos, la_sin = compute_swivel_angle_gt_batch(
                        joint_pos[:, 16], joint_pos[:, 18], joint_pos[:, 20], v_ref_arm
                    )

                    # Right arm: shoulder(17) → elbow(19) → wrist(21)
                    ra_cos, ra_sin = compute_swivel_angle_gt_batch(
                        joint_pos[:, 17], joint_pos[:, 19], joint_pos[:, 21], v_ref_arm
                    )

                    # Left leg: hip(1) → knee(4) → ankle(7)
                    # Uses v_ref_leg (+Y forward) to avoid singularity when standing
                    ll_cos, ll_sin = compute_swivel_angle_gt_batch(
                        joint_pos[:, 1], joint_pos[:, 4], joint_pos[:, 7], v_ref_leg
                    )

                    # Right leg: hip(2) → knee(5) → ankle(8)
                    # Uses v_ref_leg (+Y forward) to avoid singularity when standing
                    rl_cos, rl_sin = compute_swivel_angle_gt_batch(
                        joint_pos[:, 2], joint_pos[:, 5], joint_pos[:, 8], v_ref_leg
                    )

                    arm_swivel_cos_sin = torch.stack([la_cos, la_sin, ra_cos, ra_sin], dim=-1).numpy()
                    leg_swivel_cos_sin = torch.stack([ll_cos, ll_sin, rl_cos, rl_sin], dim=-1).numpy()

                    # ====== Save data ======
                    data = {
                        # AvatarPoser/EgoPoser compatible
                        'rotation_local_full_gt_list': rotation_local_full_gt_list,
                        'hmd_position_global_full_gt_list': hmd_position_global_full_gt_list,
                        'head_global_trans_list': head_global_trans_list,
                        'body_parms_list': body_parms,

                        # Subject shape (EgoPoser style)
                        'betas': bdata_betas,  # (16,) subject-specific body shape

                        # MANIKIN specific (computed with betas for accuracy)
                        'arm_swivel_cos_sin': arm_swivel_cos_sin,
                        'leg_swivel_cos_sin': leg_swivel_cos_sin,
                        'joint_positions': joint_pos.numpy(),  # (T, 22, 3) with betas
                        'bone_lengths': bone_lengths,  # Per-subject bone lengths

                        # Metadata
                        'framerate': 60,
                        'gender': subject_gender,
                        'filepath': filepath,
                    }

                    save_path = os.path.join(savedir, f'{idx}.pkl')
                    with open(save_path, 'wb') as f:
                        pickle.dump(data, f)

                    idx += 1
                    if idx % 20 == 0:
                        print(f"  Processed {idx} files...")

                except Exception as e:
                    print(f"  Error processing {filepath}: {e}")
                    continue

            print(f"  Saved {idx} files to {savedir}")

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data_manikin')
    parser.add_argument('--subsets', nargs='+', default=['BioMotionLab_NTroje'])
    args = parser.parse_args()

    prepare_manikin_data(
        output_dir=args.output_dir,
        subsets=args.subsets,
    )
