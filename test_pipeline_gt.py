"""
MANIKIN Pipeline Verification with GT Values

Tests that the pipeline produces correct outputs when:
1. NN outputs = GT values (perfect prediction)
2. IK solver receives GT swivel angles + GT positions

Expected: Output should match GT with near-zero error.
If error is large, indicates a bug in the pipeline.
"""

import sys
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('AvatarPoser')

from human_body_prior.body_model.body_model import BodyModel

from Manikin.extensions.manikin_core import JointIdx
from Manikin.extensions.torso_fk import TorsoFK
from Manikin.extensions.analytic_solver import AnalyticArmSolver, AnalyticLegSolver
from Manikin.utils.rotation_utils import (
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    quaternion_to_matrix, matrix_to_quaternion
)


def load_body_model():
    """Load SMPL-H body model"""
    support_dir = 'AvatarPoser/support_data/'
    bm_fname = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
    dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/male/model.npz')

    bm = BodyModel(
        bm_fname=bm_fname,
        num_betas=16,
        num_dmpls=8,
        dmpl_fname=dmpl_fname
    )
    return bm


# SMPL-H skeleton connections (22 joints)
SKELETON_CONNECTIONS = [
    # Spine chain
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Pelvis → Head
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),  # Spine3 → L_Collar → L_Shoulder → L_Elbow → L_Wrist
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),  # Spine3 → R_Collar → R_Shoulder → R_Elbow → R_Wrist
    # Left leg
    (0, 1), (1, 4), (4, 7),  # Pelvis → L_Hip → L_Knee → L_Ankle
    # Right leg
    (0, 2), (2, 5), (5, 8),  # Pelvis → R_Hip → R_Knee → R_Ankle
]


def visualize_skeleton_comparison(gt_positions, pred_positions, frame_idx, save_path=None):
    """
    Visualize GT vs Predicted skeleton

    Args:
        gt_positions: (22, 3) GT joint positions
        pred_positions: dict with pred joint positions for specific joints
        frame_idx: frame number for title
        save_path: optional path to save figure
    """
    fig = plt.figure(figsize=(16, 6))

    # View 1: Front view (XZ plane)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'Frame {frame_idx}: Front View (X-Z)')

    # View 2: Side view (YZ plane)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'Frame {frame_idx}: Side View (Y-Z)')

    # View 3: Top view (XY plane)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(f'Frame {frame_idx}: Top View (X-Y)')

    gt_np = gt_positions.numpy() if isinstance(gt_positions, torch.Tensor) else gt_positions

    for ax, elev, azim in [(ax1, 0, 0), (ax2, 0, 90), (ax3, 90, 0)]:
        # Draw GT skeleton (blue)
        for i, j in SKELETON_CONNECTIONS:
            ax.plot3D([gt_np[i, 0], gt_np[j, 0]],
                     [gt_np[i, 1], gt_np[j, 1]],
                     [gt_np[i, 2], gt_np[j, 2]], 'b-', linewidth=2, alpha=0.7)

        # Draw GT joints (blue circles)
        ax.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
                  c='blue', s=30, label='GT', alpha=0.7)

        # Draw predicted key joints (red)
        key_joints = ['L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip',
                     'L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee']
        joint_indices = [16, 17, 1, 2, 18, 19, 4, 5]

        if pred_positions:
            pred_pts = []
            for name, idx in zip(key_joints, joint_indices):
                if name.lower().replace('_', '') in str(pred_positions.keys()).lower():
                    for key in pred_positions:
                        if name.lower().replace('_', '') in key.lower():
                            pos = pred_positions[key]
                            if isinstance(pos, torch.Tensor):
                                pos = pos.cpu().numpy()
                            pred_pts.append(pos)
                            break

            if pred_pts:
                pred_arr = np.array(pred_pts)
                ax.scatter(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2],
                          c='red', s=50, marker='x', label='Pred', linewidths=2)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_error_over_time(errors_over_time, save_path=None):
    """
    Visualize position errors over time

    Args:
        errors_over_time: dict with lists of errors per joint type
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = ['Shoulder Position Error', 'Hip Position Error',
              'Elbow Position Error', 'Knee Position Error']
    keys = ['shoulder_pos', 'hip_pos', 'elbow_pos', 'knee_pos']

    for ax, title, key in zip(axes.flat, titles, keys):
        if key in errors_over_time and errors_over_time[key]:
            errors = errors_over_time[key]
            # Pair L/R errors
            n_frames = len(errors) // 2
            l_errors = [errors[i*2] for i in range(n_frames)]
            r_errors = [errors[i*2+1] for i in range(n_frames)]

            ax.plot(l_errors, 'b-', label='Left', alpha=0.7)
            ax.plot(r_errors, 'r-', label='Right', alpha=0.7)
            ax.axhline(y=np.mean(errors), color='g', linestyle='--',
                      label=f'Mean: {np.mean(errors):.2f}mm')

        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Error (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Pipeline Verification: Position Errors Over Time', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error plot to {save_path}")
    else:
        plt.show()

    plt.close()


def test_pipeline_with_gt(data_path, device='cpu'):
    """
    Test pipeline with GT values

    Args:
        data_path: path to pkl file from prepare_manikin_data
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'='*60}")
    print(f"Testing Pipeline with GT Values")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Extract GT values
    gt_poses_6d_raw = data['rotation_local_full_gt_list']
    if isinstance(gt_poses_6d_raw, torch.Tensor):
        gt_poses_6d = gt_poses_6d_raw.float()
    else:
        gt_poses_6d = torch.tensor(gt_poses_6d_raw, dtype=torch.float32)  # (T, 132)

    gt_positions_raw = data['joint_positions']
    if isinstance(gt_positions_raw, torch.Tensor):
        gt_positions = gt_positions_raw.float()
    else:
        gt_positions = torch.tensor(gt_positions_raw, dtype=torch.float32)  # (T, 22, 3)

    arm_swivel_gt = torch.tensor(data['arm_swivel_cos_sin'], dtype=torch.float32)  # (T, 4)
    leg_swivel_gt = torch.tensor(data['leg_swivel_cos_sin'], dtype=torch.float32)  # (T, 4)
    bone_lengths = data['bone_lengths']
    betas_raw = data['betas']
    if isinstance(betas_raw, torch.Tensor):
        betas = betas_raw.float().unsqueeze(0)
    else:
        betas = torch.tensor(betas_raw, dtype=torch.float32).unsqueeze(0)  # (1, 16)

    # Extract trans from body_parms (GT positions are in world space with trans)
    body_parms = data['body_parms_list']
    gt_trans = body_parms['trans'][1:]  # Skip first frame like joint_positions
    if isinstance(gt_trans, torch.Tensor):
        gt_trans = gt_trans.float()
    else:
        gt_trans = torch.tensor(gt_trans, dtype=torch.float32)  # (T, 3)

    T = gt_poses_6d.shape[0]
    gt_poses_6d = gt_poses_6d.reshape(T, 22, 6)

    print(f"Sequence length: {T} frames")
    print(f"GT poses shape: {gt_poses_6d.shape}")
    print(f"GT positions shape: {gt_positions.shape}")
    print(f"Bone lengths: {bone_lengths}")

    # Load body model for TorsoFK
    bm = load_body_model()

    # Initialize modules
    torso_fk = TorsoFK(bm).to(device)
    arm_solver = AnalyticArmSolver().to(device)
    leg_solver = AnalyticLegSolver().to(device)

    # Convert bone lengths to tensors (batch=1)
    bone_lengths_tensor = {k: torch.tensor([v], dtype=torch.float32, device=device)
                          for k, v in bone_lengths.items()}

    # Test with a few frames
    test_frames = min(10, T)

    errors = {
        'shoulder_pos': [],
        'hip_pos': [],
        'elbow_pos': [],
        'knee_pos': [],
        'wrist_pos': [],
        'ankle_pos': [],
    }

    for t in range(test_frames):
        print(f"\n--- Frame {t} ---")

        # ====================================
        # 1. Extract GT torso from poses
        # ====================================
        # GT poses: (22, 6)
        gt_frame_6d = gt_poses_6d[t].unsqueeze(0).to(device)  # (1, 22, 6)

        # Pelvis (global orient)
        global_orient = gt_frame_6d[:, JointIdx.PELVIS]  # (1, 6)

        # Torso angles: spine1, spine2, spine3, neck, L_collar, R_collar
        torso_indices = [JointIdx.SPINE1, JointIdx.SPINE2, JointIdx.SPINE3,
                        JointIdx.NECK, JointIdx.L_COLLAR, JointIdx.R_COLLAR]
        torso_angles = gt_frame_6d[:, torso_indices].reshape(1, 36)  # (1, 36)

        # ====================================
        # 2. TorsoFK with GT torso rotations + GT trans
        # ====================================
        frame_trans = gt_trans[t].unsqueeze(0).to(device)  # (1, 3)

        fk_out = torso_fk(
            global_orient_6d=global_orient,
            torso_angles_6d=torso_angles,
            trans=frame_trans,
            betas=betas.to(device)
        )

        # Compare shoulder/hip positions
        pred_l_shoulder = fk_out['left_shoulder_pos'][0]
        pred_r_shoulder = fk_out['right_shoulder_pos'][0]
        pred_l_hip = fk_out['left_hip_pos'][0]
        pred_r_hip = fk_out['right_hip_pos'][0]

        gt_l_shoulder = gt_positions[t, JointIdx.L_SHOULDER]
        gt_r_shoulder = gt_positions[t, JointIdx.R_SHOULDER]
        gt_l_hip = gt_positions[t, JointIdx.L_HIP]
        gt_r_hip = gt_positions[t, JointIdx.R_HIP]

        err_l_shoulder = torch.norm(pred_l_shoulder.cpu() - gt_l_shoulder).item() * 1000
        err_r_shoulder = torch.norm(pred_r_shoulder.cpu() - gt_r_shoulder).item() * 1000
        err_l_hip = torch.norm(pred_l_hip.cpu() - gt_l_hip).item() * 1000
        err_r_hip = torch.norm(pred_r_hip.cpu() - gt_r_hip).item() * 1000

        errors['shoulder_pos'].extend([err_l_shoulder, err_r_shoulder])
        errors['hip_pos'].extend([err_l_hip, err_r_hip])

        print(f"  TorsoFK Shoulder Error: L={err_l_shoulder:.2f}mm, R={err_r_shoulder:.2f}mm")
        print(f"  TorsoFK Hip Error: L={err_l_hip:.2f}mm, R={err_r_hip:.2f}mm")

        # ====================================
        # 3. Arm IK with GT swivel + GT wrist
        # ====================================
        # GT swivel angles
        la_cos, la_sin = arm_swivel_gt[t, 0], arm_swivel_gt[t, 1]
        ra_cos, ra_sin = arm_swivel_gt[t, 2], arm_swivel_gt[t, 3]

        # GT wrist positions
        gt_l_wrist = gt_positions[t, JointIdx.L_WRIST].unsqueeze(0).to(device)
        gt_r_wrist = gt_positions[t, JointIdx.R_WRIST].unsqueeze(0).to(device)

        # GT wrist rotations (6D)
        gt_l_wrist_6d = gt_frame_6d[:, JointIdx.L_WRIST]
        gt_r_wrist_6d = gt_frame_6d[:, JointIdx.R_WRIST]

        # Convert to quaternion for IK
        gt_l_wrist_mat = rotation_6d_to_matrix(gt_l_wrist_6d)
        gt_r_wrist_mat = rotation_6d_to_matrix(gt_r_wrist_6d)
        gt_l_wrist_quat = matrix_to_quaternion(gt_l_wrist_mat)
        gt_r_wrist_quat = matrix_to_quaternion(gt_r_wrist_mat)

        # Left arm IK
        left_arm_out = arm_solver(
            p_shoulder=fk_out['left_shoulder_pos'],
            p_wrist=gt_l_wrist,
            wrist_quat=gt_l_wrist_quat,
            cos_phi=la_cos.unsqueeze(0).to(device),
            sin_phi=la_sin.unsqueeze(0).to(device),
            L1=bone_lengths_tensor['left_humerus'],
            L2=bone_lengths_tensor['left_radius'],
            side='left',
        )

        # Right arm IK
        right_arm_out = arm_solver(
            p_shoulder=fk_out['right_shoulder_pos'],
            p_wrist=gt_r_wrist,
            wrist_quat=gt_r_wrist_quat,
            cos_phi=ra_cos.unsqueeze(0).to(device),
            sin_phi=ra_sin.unsqueeze(0).to(device),
            L1=bone_lengths_tensor['right_humerus'],
            L2=bone_lengths_tensor['right_radius'],
            side='right',
        )

        # Compare elbow positions
        pred_l_elbow = left_arm_out['elbow_pos'][0]
        pred_r_elbow = right_arm_out['elbow_pos'][0]
        gt_l_elbow = gt_positions[t, JointIdx.L_ELBOW]
        gt_r_elbow = gt_positions[t, JointIdx.R_ELBOW]

        err_l_elbow = torch.norm(pred_l_elbow.cpu() - gt_l_elbow).item() * 1000
        err_r_elbow = torch.norm(pred_r_elbow.cpu() - gt_r_elbow).item() * 1000

        errors['elbow_pos'].extend([err_l_elbow, err_r_elbow])
        print(f"  Arm IK Elbow Error: L={err_l_elbow:.2f}mm, R={err_r_elbow:.2f}mm")

        # ====================================
        # 4. Leg IK with GT swivel + GT ankle
        # ====================================
        ll_cos, ll_sin = leg_swivel_gt[t, 0], leg_swivel_gt[t, 1]
        rl_cos, rl_sin = leg_swivel_gt[t, 2], leg_swivel_gt[t, 3]

        gt_l_ankle = gt_positions[t, JointIdx.L_ANKLE].unsqueeze(0).to(device)
        gt_r_ankle = gt_positions[t, JointIdx.R_ANKLE].unsqueeze(0).to(device)

        # Left leg IK
        left_leg_out = leg_solver(
            p_hip=fk_out['left_hip_pos'],
            p_ankle=gt_l_ankle,
            cos_phi=ll_cos.unsqueeze(0).to(device),
            sin_phi=ll_sin.unsqueeze(0).to(device),
            L1=bone_lengths_tensor['left_femur'],
            L2=bone_lengths_tensor['left_tibia'],
            side='left',
        )

        # Right leg IK
        right_leg_out = leg_solver(
            p_hip=fk_out['right_hip_pos'],
            p_ankle=gt_r_ankle,
            cos_phi=rl_cos.unsqueeze(0).to(device),
            sin_phi=rl_sin.unsqueeze(0).to(device),
            L1=bone_lengths_tensor['right_femur'],
            L2=bone_lengths_tensor['right_tibia'],
            side='right',
        )

        # Compare knee positions
        pred_l_knee = left_leg_out['knee_pos'][0]
        pred_r_knee = right_leg_out['knee_pos'][0]
        gt_l_knee = gt_positions[t, JointIdx.L_KNEE]
        gt_r_knee = gt_positions[t, JointIdx.R_KNEE]

        err_l_knee = torch.norm(pred_l_knee.cpu() - gt_l_knee).item() * 1000
        err_r_knee = torch.norm(pred_r_knee.cpu() - gt_r_knee).item() * 1000

        errors['knee_pos'].extend([err_l_knee, err_r_knee])
        print(f"  Leg IK Knee Error: L={err_l_knee:.2f}mm, R={err_r_knee:.2f}mm")

    # ====================================
    # Summary
    # ====================================
    print(f"\n{'='*60}")
    print("SUMMARY (Mean Errors in mm)")
    print(f"{'='*60}")

    for key, vals in errors.items():
        if vals:
            mean_err = np.mean(vals)
            max_err = np.max(vals)
            print(f"  {key:15s}: mean={mean_err:8.2f}mm, max={max_err:8.2f}mm")

    total_mean = np.mean([v for vals in errors.values() for v in vals])
    print(f"\n  TOTAL MEAN ERROR: {total_mean:.2f}mm")

    if total_mean < 10:
        print("\n  [PASS] Pipeline is working correctly!")
    elif total_mean < 50:
        print("\n  [WARNING] Some error in pipeline, but may be acceptable")
    else:
        print("\n  [FAIL] Large error indicates bug in pipeline")

    # Create output directory
    output_dir = 'Manikin/outputs/pipeline_test'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize errors over time
    visualize_error_over_time(
        errors,
        save_path=os.path.join(output_dir, 'error_over_time.png')
    )

    # Visualize skeleton for first frame
    if test_frames > 0:
        # Get first frame GT positions
        first_frame_gt = gt_positions[0]

        # Collect predicted positions for first frame (re-run for visualization)
        gt_frame_6d = gt_poses_6d[0].unsqueeze(0).to(device)
        global_orient = gt_frame_6d[:, JointIdx.PELVIS]
        torso_indices = [JointIdx.SPINE1, JointIdx.SPINE2, JointIdx.SPINE3,
                        JointIdx.NECK, JointIdx.L_COLLAR, JointIdx.R_COLLAR]
        torso_angles = gt_frame_6d[:, torso_indices].reshape(1, 36)

        frame_trans = gt_trans[0].unsqueeze(0).to(device)
        fk_out = torso_fk(
            global_orient_6d=global_orient,
            torso_angles_6d=torso_angles,
            trans=frame_trans,
            betas=betas.to(device)
        )

        pred_positions_dict = {
            'l_shoulder': fk_out['left_shoulder_pos'][0].cpu().numpy(),
            'r_shoulder': fk_out['right_shoulder_pos'][0].cpu().numpy(),
            'l_hip': fk_out['left_hip_pos'][0].cpu().numpy(),
            'r_hip': fk_out['right_hip_pos'][0].cpu().numpy(),
        }

        visualize_skeleton_comparison(
            first_frame_gt,
            pred_positions_dict,
            frame_idx=0,
            save_path=os.path.join(output_dir, 'skeleton_frame0.png')
        )

    print(f"\nVisualizations saved to {output_dir}/")

    return errors


if __name__ == '__main__':
    import glob

    # Find a test data file
    data_files = glob.glob('data_manikin/**/*.pkl', recursive=True)

    if not data_files:
        print("No data files found in data_manikin/")
        print("Please run: python Manikin/data/preprocessing/prepare_manikin_data.py first")
    else:
        # Test with first file
        test_file = data_files[0]
        print(f"Using test file: {test_file}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")

        test_pipeline_with_gt(test_file, device=device)
