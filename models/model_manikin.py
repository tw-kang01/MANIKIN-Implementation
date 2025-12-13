"""
MANIKIN Model Training Wrapper
Based on EgoPoser/AvatarPoser model pattern

Integrates:
- MANIKINModelJLM (Network + TorsoFK + Analytic IK)
- MANIKINLossJLM (6-term loss function, Eq. 15)
- Body model loading (SMPL-H)
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_base import ModelBase
from extensions.manikin_model import MANIKINModelJLM
from utils.manikin_loss_module import MANIKINLossJLM


def load_body_model(support_dir, device, gender='male'):
    """
    Load SMPL-H body model from AvatarPoser support_data

    Args:
        support_dir: Path to support_data directory (contains body_models/)
        device: torch device
        gender: 'male', 'female', or 'neutral'

    Returns:
        BodyModel instance
    """
    # Import from human_body_prior
    # Try AvatarPoser's human_body_prior first
    try:
        avatarposer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'AvatarPoser'
        )
        sys.path.insert(0, avatarposer_path)
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError:
        # Try local Manikin human_body_prior
        from Manikin.human_body_prior.body_model.body_model import BodyModel

    bm_fname = os.path.join(support_dir, f'body_models/smplh/{gender}/model.npz')
    dmpl_fname = os.path.join(support_dir, f'body_models/dmpls/{gender}/model.npz')

    if not os.path.exists(bm_fname):
        raise FileNotFoundError(f"Body model not found: {bm_fname}")

    body_model = BodyModel(
        bm_fname=bm_fname,
        num_betas=16,
        num_dmpls=8,
        dmpl_fname=dmpl_fname if os.path.exists(dmpl_fname) else None
    ).to(device)

    return body_model


class ModelMANIKIN(ModelBase):
    """
    MANIKIN Training Model Wrapper

    Integrates MANIKINModelJLM with training infrastructure.
    """

    def __init__(self, opt):
        super().__init__(opt)

        self.opt_train = opt.get('train', {})
        self.opt_net = opt.get('netG', {})

        # Load body model
        support_dir = opt.get('support_dir', 'AvatarPoser/support_data/')
        self.body_model = load_body_model(support_dir, self.device)

        # Create MANIKIN model
        self.net = MANIKINModelJLM(self.body_model, config=opt)
        self.net = self.net.to(self.device)

        # Initialize log dict
        self.log_dict = OrderedDict()

    # ----------------------------------------
    # Training Initialization
    # ----------------------------------------

    def init_train(self):
        """Initialize training: load, define loss/optimizer/scheduler"""
        self.load()
        self.net.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def init_test(self):
        """Initialize for testing"""
        self.load(test=True)
        self.net.eval()
        self.log_dict = OrderedDict()

    # ----------------------------------------
    # Load/Save
    # ----------------------------------------

    def load(self, test=False):
        """Load pretrained model"""
        if test:
            load_path = self.opt['path'].get('pretrained')
        else:
            load_path = self.opt['path'].get('pretrained_netG')

        if load_path is not None and os.path.exists(load_path):
            print(f'Loading model from [{load_path}]...')
            self.load_network(load_path, self.net, strict=False)

    def save(self, iter_label):
        """Save model checkpoint"""
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_network(self.save_dir, self.net, iter_label)

        # Save optimizer if configured
        if self.opt_train.get('G_optimizer_reuse', False):
            self.save_optimizer(self.save_dir, self.optimizer, 'optimizer', iter_label)

    # ----------------------------------------
    # Loss Definition
    # ----------------------------------------

    def define_loss(self):
        """Define loss function (6-term, Eq. 15)"""
        weights = self.opt_train.get('lambda_weights', {})
        self.loss_fn = MANIKINLossJLM(weights).to(self.device)

    # ----------------------------------------
    # Optimizer Definition
    # ----------------------------------------

    def define_optimizer(self):
        """Define Adam optimizer"""
        optim_params = [p for p in self.net.parameters() if p.requires_grad]
        lr = self.opt_train.get('G_optimizer_lr', 1e-4)
        self.optimizer = Adam(optim_params, lr=lr, weight_decay=0)

    # ----------------------------------------
    # Scheduler Definition
    # ----------------------------------------

    def define_scheduler(self):
        """Define MultiStepLR scheduler"""
        milestones = self.opt_train.get('G_scheduler_milestones', [10000, 20000, 30000])
        gamma = self.opt_train.get('G_scheduler_gamma', 0.5)
        self.schedulers.append(MultiStepLR(self.optimizer, milestones, gamma))

    # ----------------------------------------
    # Feed Data
    # ----------------------------------------

    def feed_data(self, data, test=False):
        """
        Load batch data to device

        Args:
            data: dict from DataLoader with custom collate_fn
            test: bool, whether in test mode
        """
        # Sparse input
        self.sparse = data['sparse'].to(self.device)  # (B, T, 54)

        # Ground truth
        self.poses_gt = data['poses_gt'].to(self.device)  # (B, T, 132)
        self.joint_positions = data['joint_positions'].to(self.device)  # (B, T, 22, 3)
        self.arm_swivel_gt = data['arm_swivel_gt'].to(self.device)  # (B, T, 4)
        self.leg_swivel_gt = data['leg_swivel_gt'].to(self.device)  # (B, T, 4)

        # Body parameters
        self.betas = data['betas'].to(self.device)  # (B, 16)

        # Translation (optional)
        if 'trans' in data:
            self.trans = data['trans'].to(self.device)  # (B, T, 3)
        else:
            batch, seq_len = self.sparse.shape[:2]
            self.trans = torch.zeros(batch, seq_len, 3, device=self.device)

        # bone_lengths: dict with (B,) tensors (from custom collate_fn)
        self.bone_lengths = {
            k: v.to(self.device) for k, v in data['bone_lengths'].items()
        }

        # Prepare gt_data dict for model forward
        self.gt_data = {
            'poses_gt': self.poses_gt,
            'joint_positions': self.joint_positions,
            'arm_swivel_gt': self.arm_swivel_gt,
            'leg_swivel_gt': self.leg_swivel_gt,
            'trans': self.trans,
        }

        # Store filename for logging
        self.filename = data.get('filename', None)

    # ----------------------------------------
    # Optimization Step
    # ----------------------------------------

    def optimize_parameters(self, current_step):
        """
        Forward → Loss → Backward → Optimizer Step

        Args:
            current_step: current training iteration
        """
        self.optimizer.zero_grad()

        # Forward pass
        pred = self.net(
            sparse_input=self.sparse,
            gt_data=self.gt_data,
            betas=self.betas,
            bone_lengths=self.bone_lengths,
            use_gt_ankle=True  # Use GT ankle positions during training
        )

        # Compute loss (6 terms)
        total_loss, loss_dict = self.loss_fn(pred, self.gt_data)

        # Backward
        total_loss.backward()

        # Gradient clipping
        clipgrad = self.opt_train.get('G_optimizer_clipgrad', 0)
        if clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clipgrad)

        # Optimizer step
        self.optimizer.step()

        # Update log dict
        self.log_dict = OrderedDict()
        self.log_dict['total_loss'] = total_loss.item()
        for k, v in loss_dict.items():
            self.log_dict[k] = v

    # ----------------------------------------
    # Testing
    # ----------------------------------------

    def test(self):
        """Run inference in eval mode"""
        self.net.eval()

        with torch.no_grad():
            pred = self.net(
                sparse_input=self.sparse,
                gt_data=None,  # No GT in test (use predicted ankle)
                betas=self.betas,
                bone_lengths=self.bone_lengths,
                use_gt_ankle=False  # Use predicted ankle positions
            )

        self.pred = pred
        return pred

    # ----------------------------------------
    # Logging
    # ----------------------------------------

    def current_log(self):
        """Return current loss log dict"""
        return self.log_dict

    def current_prediction(self):
        """Return current prediction (after test())"""
        if hasattr(self, 'pred'):
            return self.pred
        return None

    def current_gt(self):
        """Return current ground truth data"""
        return {
            'poses_gt': self.poses_gt,
            'joint_positions': self.joint_positions,
            'arm_swivel_gt': self.arm_swivel_gt,
            'leg_swivel_gt': self.leg_swivel_gt,
            'trans': self.trans,
        }

    # ----------------------------------------
    # Network Info
    # ----------------------------------------

    def info_network(self):
        """Return network info string"""
        return self.describe_network(self.net)

    def info_params(self):
        """Return parameter info string"""
        return self.describe_params(self.net)
