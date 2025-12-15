"""
MANIKIN Dataset
AvatarPoser/EgoPoser의 AMASS_Dataset 구조를 따르며, MANIKIN용 데이터 로드
"""

import sys
import os
import torch
import pickle
import glob
import random
import logging
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('.')
sys.path.append('AvatarPoser')


class MANIKINDataset(Dataset):
    """
    MANIKIN용 데이터셋
    AvatarPoser/EgoPoser 데이터 format 호환 + swivel angle GT 추가
    """
    
    def __init__(self, opt):
        """
        Args:
            opt: dict with dataset options
                - dataroot: list of data directories
                - window_size: number of frames in sliding window
                - phase: 'train' or 'test'
                - dataloader_batch_size: batch size
        """
        self.opt = opt
        self.window_size = opt.get('window_size', 40)
        self.batch_size = opt.get('dataloader_batch_size', 32)
        self.phase = opt.get('phase', 'train')
        
        # Handle dataroot as list or string
        dataroot_list = opt.get('dataroot', ['data_manikin'])
        if isinstance(dataroot_list, str):
            dataroot_list = [dataroot_list]
        
        # Find all pkl files
        self.filename_list = []
        for dataroot in dataroot_list:
            pattern = os.path.join(dataroot, '**', '*.pkl')
            self.filename_list += glob.glob(pattern, recursive=True)
        
        # Filter by phase if directories are structured
        if self.phase == 'train':
            self.filename_list = [f for f in self.filename_list 
                                  if 'test' not in f.lower() or 'train' in f.lower()]
        else:
            self.filename_list = [f for f in self.filename_list 
                                  if 'test' in f.lower()]
        
        # Fallback: if no files found with phase filter, use all
        if len(self.filename_list) == 0:
            for dataroot in dataroot_list:
                pattern = os.path.join(dataroot, '**', '*.pkl')
                self.filename_list += glob.glob(pattern, recursive=True)
        
        self.filename_list.sort()

        if len(self.filename_list) == 0:
            print(f"Warning: No data files found in {dataroot_list}")
            print("Please run: python Manikin/data/prepare_manikin_data.py first")
        else:
            print(f"Found {len(self.filename_list)} files for {self.phase}")

        # RAM 캐싱: 전체 데이터를 메모리에 로드 (I/O 병목 해결)
        self.data_cache = {}
        cache_in_memory = opt.get('cache_in_memory', True)
        if cache_in_memory and len(self.filename_list) > 0:
            print(f"Loading {len(self.filename_list)} files into memory...")
            for filename in tqdm(self.filename_list, desc="Caching data"):
                with open(filename, 'rb') as f:
                    self.data_cache[filename] = pickle.load(f)
            print(f"Cached {len(self.data_cache)} files in RAM")

        # Test용 sliding window 인덱스 사전 계산 (overlap 없음)
        if self.phase == 'test' and len(self.filename_list) > 0:
            self.test_windows = []  # [(file_idx, start, end), ...]

            for file_idx, filename in enumerate(self.filename_list):
                # 캐시에서 로드 (없으면 디스크에서)
                if filename in self.data_cache:
                    data = self.data_cache[filename]
                else:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                seq_len = data['hmd_position_global_full_gt_list'].shape[0]

                # 시퀀스가 window_size보다 작으면 전체를 하나의 window로
                if seq_len <= self.window_size:
                    self.test_windows.append((file_idx, 0, seq_len))
                else:
                    # Non-overlapping windows (stride = window_size)
                    for start in range(0, seq_len, self.window_size):
                        end = min(start + self.window_size, seq_len)
                        # 마지막 window가 너무 짧으면 skip (최소 10프레임)
                        if end - start >= 10:
                            self.test_windows.append((file_idx, start, end))

            print(f"Test: {len(self.filename_list)} files -> {len(self.test_windows)} windows")
    
    def __len__(self):
        if self.phase == 'test' and hasattr(self, 'test_windows'):
            return len(self.test_windows)
        return max(len(self.filename_list), self.batch_size)
    
    def __getitem__(self, idx):
        """
        Returns data for one training sample
        
        For training: returns windowed data
        For testing: returns full sequence
        """
        if len(self.filename_list) == 0:
            # Warn and return dummy data if no files found
            logging.warning("MANIKINDataset: No data files found - returning dummy data")
            return self._get_dummy_data()
        
        idx = idx % len(self.filename_list)
        filename = self.filename_list[idx]

        # 캐시에서 로드 (없으면 디스크에서)
        if filename in self.data_cache:
            data = self.data_cache[filename]
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

        # Get sequence length
        sparse_input_full = data['hmd_position_global_full_gt_list']  # (T, 54)
        seq_len = sparse_input_full.shape[0]

        if self.phase == 'train':
            # Skip sequences shorter than window size
            while seq_len <= self.window_size:
                idx = random.randint(0, len(self.filename_list) - 1)
                filename = self.filename_list[idx]
                # 캐시에서 로드
                if filename in self.data_cache:
                    data = self.data_cache[filename]
                else:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                sparse_input_full = data['hmd_position_global_full_gt_list']
                seq_len = sparse_input_full.shape[0]
            
            # Random window for training
            frame = random.randint(0, seq_len - self.window_size - 1)
            
            # Extract windowed data
            sparse_input = sparse_input_full[frame:frame + self.window_size]
            poses_gt = data['rotation_local_full_gt_list'][frame:frame + self.window_size]
            head_trans = data['head_global_trans_list'][frame + self.window_size - 1]
            
            # MANIKIN specific: swivel angles and joint positions for FULL WINDOW
            # Note: NN outputs (B, T, Features), so GT should also be (T, Features)
            arm_swivel = data['arm_swivel_cos_sin'][frame : frame + self.window_size]  # (T, 4)
            leg_swivel = data['leg_swivel_cos_sin'][frame : frame + self.window_size]  # (T, 4)
            joint_pos = data['joint_positions'][frame : frame + self.window_size]      # (T, 22, 3)
            
            # Convert to tensors (always ensure float32)
            if isinstance(sparse_input, torch.Tensor):
                sparse_input = sparse_input.float()
            else:
                sparse_input = torch.tensor(sparse_input, dtype=torch.float32)
            
            if isinstance(poses_gt, torch.Tensor):
                poses_gt = poses_gt.float()
            else:
                poses_gt = torch.tensor(poses_gt, dtype=torch.float32)
            
            if isinstance(head_trans, torch.Tensor):
                head_trans = head_trans.float()
            else:
                head_trans = torch.tensor(head_trans, dtype=torch.float32)
            
            # Get betas (subject-specific body shape)
            betas = data.get('betas', np.zeros(16))
            if isinstance(betas, np.ndarray):
                betas = torch.tensor(betas, dtype=torch.float32)
            
            return {
                # AvatarPoser 호환
                'L': sparse_input,  # (window_size, 54)
                'H': poses_gt[-1],  # (132,) - last frame GT
                'Head_trans_global': head_trans,  # (4, 4)

                # For full sequence GT
                'sparse': sparse_input,
                'poses_gt': poses_gt,

                # MANIKIN specific - FULL WINDOW (T, Features)
                'arm_swivel_gt': torch.tensor(arm_swivel, dtype=torch.float32),  # (T, 4)
                'leg_swivel_gt': torch.tensor(leg_swivel, dtype=torch.float32),  # (T, 4)
                'joint_positions': torch.tensor(joint_pos, dtype=torch.float32),  # (T, 22, 3)
                'bone_lengths': data['bone_lengths'],
                'betas': betas,  # (16,) subject-specific body shape

                # Joint positions for loss computation - FULL WINDOW (T, 3)
                'left_elbow_pos': torch.tensor(joint_pos[:, 18], dtype=torch.float32),   # (T, 3)
                'right_elbow_pos': torch.tensor(joint_pos[:, 19], dtype=torch.float32),  # (T, 3)
                'left_knee_pos': torch.tensor(joint_pos[:, 4], dtype=torch.float32),     # (T, 3)
                'right_knee_pos': torch.tensor(joint_pos[:, 5], dtype=torch.float32),    # (T, 3)
                'left_wrist_pos': torch.tensor(joint_pos[:, 20], dtype=torch.float32),   # (T, 3)
                'right_wrist_pos': torch.tensor(joint_pos[:, 21], dtype=torch.float32),  # (T, 3)
                'left_ankle_pos': torch.tensor(joint_pos[:, 7], dtype=torch.float32),    # (T, 3)
                'right_ankle_pos': torch.tensor(joint_pos[:, 8], dtype=torch.float32),   # (T, 3)

                # Shoulder/hip positions for L_FK_torso loss (paper Eq. 1) - FULL WINDOW (T, 3)
                'left_shoulder_pos': torch.tensor(joint_pos[:, 16], dtype=torch.float32),  # (T, 3)
                'right_shoulder_pos': torch.tensor(joint_pos[:, 17], dtype=torch.float32), # (T, 3)
                'left_hip_pos': torch.tensor(joint_pos[:, 1], dtype=torch.float32),        # (T, 3)
                'right_hip_pos': torch.tensor(joint_pos[:, 2], dtype=torch.float32),       # (T, 3)

                'filename': filename,
            }
        
        else:
            # Test: return windowed data (same as train for max_seq_len compatibility)
            file_idx, start, end = self.test_windows[idx]
            filename = self.filename_list[file_idx]

            # 캐시에서 로드 (없으면 디스크에서)
            if filename in self.data_cache:
                data = self.data_cache[filename]
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

            sparse_input_full = data['hmd_position_global_full_gt_list']
            total_seq_len = sparse_input_full.shape[0]

            # Extract windowed data
            sparse_input = sparse_input_full[start:end]
            poses_gt = data['rotation_local_full_gt_list'][start:end]
            head_trans = data['head_global_trans_list'][start:end]

            # MANIKIN specific
            arm_swivel = data['arm_swivel_cos_sin'][start:end]
            leg_swivel = data['leg_swivel_cos_sin'][start:end]
            joint_pos = data['joint_positions'][start:end]

            # Convert to tensors
            if isinstance(sparse_input, torch.Tensor):
                sparse_input = sparse_input.float()
            else:
                sparse_input = torch.tensor(sparse_input, dtype=torch.float32)

            if isinstance(poses_gt, torch.Tensor):
                poses_gt = poses_gt.float()
            else:
                poses_gt = torch.tensor(poses_gt, dtype=torch.float32)

            if isinstance(head_trans, torch.Tensor):
                head_trans = head_trans.float()
            else:
                head_trans = torch.tensor(head_trans, dtype=torch.float32)

            # Get betas (subject-specific body shape)
            betas = data.get('betas', np.zeros(16))
            if isinstance(betas, np.ndarray):
                betas = torch.tensor(betas, dtype=torch.float32)

            return {
                # AvatarPoser 호환
                'L': sparse_input,  # (window_size, 54)
                'H': poses_gt,      # (window_size, 132)
                'Head_trans_global': head_trans,

                # For full sequence
                'sparse': sparse_input,
                'poses_gt': poses_gt,

                # MANIKIN specific
                'arm_swivel_gt': torch.tensor(arm_swivel, dtype=torch.float32),
                'leg_swivel_gt': torch.tensor(leg_swivel, dtype=torch.float32),
                'joint_positions': torch.tensor(joint_pos, dtype=torch.float32),
                'bone_lengths': data['bone_lengths'],
                'betas': betas,  # (16,) subject-specific body shape

                # Window metadata (for result reconstruction)
                'filename': filename,
                'window_start': start,
                'window_end': end,
                'seq_len': total_seq_len,
            }
    
    def _get_dummy_data(self):
        """Return dummy data when no files are available"""
        window = self.window_size if self.phase == 'train' else 100

        return {
            'L': torch.randn(window, 54),
            'H': torch.randn(132) if self.phase == 'train' else torch.randn(window, 132),
            'P': {
                'root_orient': torch.zeros(window, 3),
                'pose_body': torch.zeros(window, 63),
                'trans': torch.zeros(window, 3),
            },
            'Head_trans_global': torch.eye(4) if self.phase == 'train' else torch.eye(4).unsqueeze(0).repeat(window, 1, 1),
            'sparse': torch.randn(window, 54),
            'poses_gt': torch.randn(window, 132),  # Always (T, 132)
            'arm_swivel_gt': torch.randn(window, 4),  # Always (T, 4)
            'leg_swivel_gt': torch.randn(window, 4),  # Always (T, 4)
            'joint_positions': torch.randn(window, 22, 3),  # Always (T, 22, 3)
            'bone_lengths': {
                'left_humerus': 0.3, 'left_radius': 0.25,
                'right_humerus': 0.3, 'right_radius': 0.25,
                'left_femur': 0.4, 'left_tibia': 0.4,
                'right_femur': 0.4, 'right_tibia': 0.4,
            },
            'betas': torch.zeros(16),
            # Joint positions - FULL WINDOW (T, 3)
            'left_elbow_pos': torch.zeros(window, 3),
            'right_elbow_pos': torch.zeros(window, 3),
            'left_knee_pos': torch.zeros(window, 3),
            'right_knee_pos': torch.zeros(window, 3),
            'left_wrist_pos': torch.zeros(window, 3),
            'right_wrist_pos': torch.zeros(window, 3),
            'left_ankle_pos': torch.zeros(window, 3),
            'right_ankle_pos': torch.zeros(window, 3),
            'left_shoulder_pos': torch.zeros(window, 3),
            'right_shoulder_pos': torch.zeros(window, 3),
            'left_hip_pos': torch.zeros(window, 3),
            'right_hip_pos': torch.zeros(window, 3),
            'filename': 'dummy',
            # Test metadata
            'window_start': 0,
            'window_end': window,
            'seq_len': window,
        }


def define_Dataset(dataset_opt):
    """
    Factory function for dataset creation (AvatarPoser 호환)
    """
    dataset_type = dataset_opt.get('dataset_type', 'manikin')

    if dataset_type == 'manikin':
        return MANIKINDataset(dataset_opt)
    else:
        raise NotImplementedError(f"Dataset type [{dataset_type}] is not implemented")
