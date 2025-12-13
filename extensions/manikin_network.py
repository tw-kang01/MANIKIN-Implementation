"""
MANIKINNetworkJLM - AvatarJLM 기반 MANIKIN 네트워크

AvatarJLM의 AlternativeST (Spatial-Temporal Transformer) 구조를 MANIKIN에 적용.

Input: (batch, seq_len, 54) - 3 sparse joints (Head, L_Wrist, R_Wrist) x 18D
Output:
    - torso: (batch, seq_len, 42) - 7 joints x 6D
        pelvis(6) + spine1-3(18) + neck(6) + collar(12)
        NOTE: shoulder/hip rotations are computed by IK solver, not NN
    - arm_swivel: (batch, seq_len, 4) - L_arm(cos,sin) + R_arm(cos,sin)
    - leg_swivel: (batch, seq_len, 4) - L_leg(cos,sin) + R_leg(cos,sin)
    - foot: (batch, seq_len, 18) - L_ankle(9) + R_ankle(9)
        Each ankle: position(3) + rotation(6D)
        Position is needed for Leg IK target (unlike arms where wrist is known)

Torso FK uses torso angles (36D = 6 joints) to compute shoulder/hip POSITIONS.
IK solver computes shoulder/hip ROTATIONS from positions + swivel angles.
Leg IK uses predicted ankle position as target (arm IK uses known wrist position).

Architecture:
    Stage 1: Initial prediction (SimpleSMPL-like)
    Stage 2: Tokenization + AlternativeST + Output heads
"""

import math
import warnings
import torch
import torch.nn as nn


# ============================================================================
# Helper Functions (from AvatarJLM)
# ============================================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization (from AvatarJLM/module.py)"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills tensor with truncated normal distribution values."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================================
# AlternativeST: Spatial-Temporal Alternating Transformer (from AvatarJLM)
# ============================================================================

class AlternativeST(nn.Module):
    """
    Spatial-Temporal Alternating Transformer from AvatarJLM.

    Alternates between:
    - Spatial Transformer Block (STB): attention across joints within same frame
    - Temporal Transformer Block (TTB): attention across time for same joint

    Args:
        repeat_time: Number of ST block repetitions
        s_layer: Number of layers in spatial transformer
        t_layer: Number of layers in temporal transformer
        embed_dim: Feature dimension
        nhead: Number of attention heads
    """

    def __init__(self, repeat_time=1, s_layer=2, t_layer=2, embed_dim=256, nhead=8):
        super().__init__()
        self.num_layer = repeat_time
        self.s_layer = s_layer
        self.t_layer = t_layer
        self.STB = nn.ModuleList()
        self.TTB = nn.ModuleList()

        for _ in range(repeat_time):
            if self.s_layer != 0:
                spatial_layer = nn.TransformerEncoderLayer(
                    embed_dim, nhead=nhead, batch_first=True
                )
                self.STB.append(nn.TransformerEncoder(spatial_layer, num_layers=s_layer))
            if self.t_layer != 0:
                temporal_layer = nn.TransformerEncoderLayer(
                    embed_dim, nhead=nhead, batch_first=True
                )
                self.TTB.append(nn.TransformerEncoder(temporal_layer, num_layers=t_layer))

    def forward(self, feat):
        """
        Args:
            feat: (batch, seq_len, token_num, feat_dim)
        Returns:
            feat: (batch, seq_len, token_num, feat_dim)
        """
        assert len(feat.shape) == 4, \
            'Input shape should be 4D: (batch, seq_len, token_num, feat_dim)'

        batch, seq_len, token_num, feat_dim = feat.shape

        for i in range(self.num_layer):
            # Spatial attention: across tokens within same frame
            if self.s_layer != 0:
                # Reshape: (batch*seq_len, token_num, feat_dim)
                feat = self.STB[i](
                    feat.reshape(batch * seq_len, token_num, -1)
                ).reshape(batch, seq_len, token_num, -1)

            # Temporal attention: across frames for same token
            if self.t_layer != 0:
                # Reshape: (batch*token_num, seq_len, feat_dim)
                feat = feat.permute(0, 2, 1, 3)  # (batch, token_num, seq_len, feat_dim)
                feat = self.TTB[i](
                    feat.reshape(batch * token_num, seq_len, -1)
                ).reshape(batch, token_num, seq_len, -1)
                feat = feat.permute(0, 2, 1, 3)  # (batch, seq_len, token_num, feat_dim)

        return feat


# ============================================================================
# MANIKINNetworkJLM: Main Network
# ============================================================================

class MANIKINNetworkJLM(nn.Module):
    """
    AvatarJLM 기반 MANIKIN 네트워크

    Token Structure (16 tokens total):
        [0:3]   Input tokens:  Head, L_Wrist, R_Wrist (from VR tracker)
        [3:10]  Torso tokens:  Pelvis, Spine1, Spine2, Spine3, Neck, L_Collar, R_Collar
        [10:12] Foot tokens:   L_Ankle, R_Ankle
        [12:14] Swivel tokens: L_Arm, R_Arm
        [14:16] Swivel tokens: L_Leg, R_Leg

    NOTE: Shoulder/Hip rotations are NOT predicted by NN. They are computed by IK solver
          using shoulder/hip positions (from Torso FK) and swivel angles (from NN).

    Input: (batch, seq_len, 54) - 3 joints x 18D
        Each joint: rot_6d(6) + rot_vel_6d(6) + pos_3d(3) + pos_vel_3d(3)

    Output:
        - torso: (batch, seq_len, 42) - 7 joints x 6D rotation
            [0:6]   pelvis
            [6:12]  spine1
            [12:18] spine2
            [18:24] spine3
            [24:30] neck
            [30:36] L_collar
            [36:42] R_collar
        - arm_swivel: (batch, seq_len, 4) - L/R x (cos, sin)
        - leg_swivel: (batch, seq_len, 4) - L/R x (cos, sin)
        - foot: (batch, seq_len, 18) - 2 joints x (3D position + 6D rotation)
            [0:3]   L_ankle position
            [3:9]   L_ankle rotation (6D)
            [9:12]  R_ankle position
            [12:18] R_ankle rotation (6D)
    """

    def __init__(
        self,
        embed_dim=512,
        joint_embed_dim=256,
        num_layer=6,
        s_layer=1,
        t_layer=1,
        nhead=8,
        reg_hidden_dim=1024,
    ):
        super().__init__()

        # =============== Constants ===============
        self.input_dim = 3 * 18  # 54D: 3 sparse joints x 18D each
        self.output_dim = 68    # 42 + 8 + 18 (torso + swivel + foot with position)
        self.token_num = 16     # 3 input + 7 torso + 2 foot + 4 swivel
        self.feat_dim = joint_embed_dim * 2  # 512

        # =============== Stage 1: Initial Prediction ===============
        # Sparse input → initial 68D output (like SimpleSMPL in AvatarJLM)
        self.linear_embedding = nn.Linear(self.input_dim, embed_dim)
        self.init_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, self.output_dim)  # 68D
        )

        # =============== Stage 2: Token Embeddings ===============
        # Different embedding layers for each token type
        self.input_token_embed = nn.Linear(18, self.feat_dim)   # per sparse joint
        self.torso_token_embed = nn.Linear(6, self.feat_dim)    # 6D rotation (7 joints)
        self.foot_token_embed = nn.Linear(9, self.feat_dim)     # 3D pos + 6D rot per ankle
        self.swivel_token_embed = nn.Linear(2, self.feat_dim)   # cos, sin

        # =============== Stage 2: AlternativeST Transformer ===============
        self.transformer = AlternativeST(
            repeat_time=num_layer,
            s_layer=s_layer,
            t_layer=t_layer,
            embed_dim=self.feat_dim,
            nhead=nhead
        )

        # Positional embeddings
        max_seq_len = 200
        self.temp_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, self.feat_dim))
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, self.token_num, self.feat_dim))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.joint_embed, std=.02)

        # =============== Output Heads ===============
        # Torso head: uses tokens [3:10] (7 torso joints)
        self.torso_head = nn.Sequential(
            nn.Linear(self.feat_dim * 7, reg_hidden_dim),
            nn.GroupNorm(8, reg_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(reg_hidden_dim, 42)  # 7 joints x 6D
        )

        # Foot head: uses tokens [10:12] (2 foot joints)
        # Output: 2 ankles x (3D position + 6D rotation) = 18D
        self.foot_head = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 18)  # 2 joints x (3D pos + 6D rot)
        )

        # Arm swivel head: uses tokens [12:14] (L_Arm, R_Arm)
        self.arm_swivel_head = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 4)  # L/R x (cos, sin)
        )

        # Leg swivel head: uses tokens [14:16] (L_Leg, R_Leg)
        self.leg_swivel_head = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 4)  # L/R x (cos, sin)
        )

    def tokenize(self, sparse_input, init_output):
        """
        Convert inputs to token representation.

        Args:
            sparse_input: (batch, seq_len, 54) - 3 sparse joints x 18D
            init_output: (batch, seq_len, 68) - Stage 1 initial prediction

        Returns:
            tokens: (batch, seq_len, 16, feat_dim)
        """
        batch, seq_len = sparse_input.shape[:2]

        # Parse sparse input: 3 joints x 18D
        sparse_3joint = sparse_input.reshape(batch, seq_len, 3, 18)

        # Parse initial output (68D total)
        # torso: 0:42 (7 joints x 6D)
        # arm_swivel: 42:46 (2 x 2)
        # leg_swivel: 46:50 (2 x 2)
        # foot: 50:68 (2 joints x (3D pos + 6D rot) = 18D)
        init_torso = init_output[:, :, :42].reshape(batch, seq_len, 7, 6)
        init_arm_swivel = init_output[:, :, 42:46].reshape(batch, seq_len, 2, 2)
        init_leg_swivel = init_output[:, :, 46:50].reshape(batch, seq_len, 2, 2)
        init_foot = init_output[:, :, 50:68].reshape(batch, seq_len, 2, 9)  # 3D pos + 6D rot

        # Build tokens
        tokens_list = []

        # [0:3] Input tokens from sparse joints
        input_tokens = self.input_token_embed(
            sparse_3joint.reshape(batch * seq_len, 3, 18)
        )  # (batch*seq_len, 3, feat_dim)
        tokens_list.append(input_tokens)

        # [3:10] Torso tokens from initial prediction (7 joints)
        torso_tokens = self.torso_token_embed(
            init_torso.reshape(batch * seq_len, 7, 6)
        )  # (batch*seq_len, 7, feat_dim)
        tokens_list.append(torso_tokens)

        # [10:12] Foot tokens from initial prediction (3D pos + 6D rot = 9D per ankle)
        foot_tokens = self.foot_token_embed(
            init_foot.reshape(batch * seq_len, 2, 9)
        )  # (batch*seq_len, 2, feat_dim)
        tokens_list.append(foot_tokens)

        # [12:14] Arm swivel tokens
        arm_swivel_tokens = self.swivel_token_embed(
            init_arm_swivel.reshape(batch * seq_len, 2, 2)
        )  # (batch*seq_len, 2, feat_dim)
        tokens_list.append(arm_swivel_tokens)

        # [14:16] Leg swivel tokens
        leg_swivel_tokens = self.swivel_token_embed(
            init_leg_swivel.reshape(batch * seq_len, 2, 2)
        )  # (batch*seq_len, 2, feat_dim)
        tokens_list.append(leg_swivel_tokens)

        # Concatenate all tokens: (batch*seq_len, 16, feat_dim)
        tokens = torch.cat(tokens_list, dim=1)

        return tokens.reshape(batch, seq_len, self.token_num, self.feat_dim)

    def forward(self, sparse_input):
        """
        Forward pass.

        Args:
            sparse_input: (batch, seq_len, 54) - 3 sparse joints x 18D
                Each joint: rot_6d(6) + rot_vel(6) + pos(3) + pos_vel(3)

        Returns:
            dict with:
                - torso: (batch, seq_len, 42) - 7 joints
                    [0:6]   pelvis
                    [6:12]  spine1
                    [12:18] spine2
                    [18:24] spine3
                    [24:30] neck
                    [30:36] L_collar
                    [36:42] R_collar
                - arm_swivel: (batch, seq_len, 4)
                - leg_swivel: (batch, seq_len, 4)
                - foot: (batch, seq_len, 18) - position + rotation per ankle
                    [0:3]   L_ankle position
                    [3:9]   L_ankle rotation (6D)
                    [9:12]  R_ankle position
                    [12:18] R_ankle rotation (6D)
                - init_output: (batch, seq_len, 68) for Stage 1 supervision
        """
        batch, seq_len = sparse_input.shape[:2]

        # =============== Stage 1: Initial Prediction ===============
        static_embed = self.linear_embedding(sparse_input)  # (batch, seq_len, embed_dim)
        init_output = self.init_regressor(static_embed)     # (batch, seq_len, 62)

        # =============== Stage 2: Tokenization ===============
        tokens = self.tokenize(sparse_input, init_output)
        # tokens: (batch, seq_len, 16, feat_dim)

        # Add positional embeddings
        tokens = tokens + self.joint_embed + self.temp_embed[:, :seq_len]

        # =============== Stage 2: Transformer ===============
        refined = self.transformer(tokens)  # (batch, seq_len, 16, feat_dim)

        # =============== Output Heads ===============
        # Flatten tokens for each head

        # Torso: tokens [3:10] - 7 joints
        torso_feat = refined[:, :, 3:10].reshape(batch * seq_len, -1)
        torso = self.torso_head(torso_feat).reshape(batch, seq_len, 42)

        # Foot: tokens [10:12] - outputs position + rotation per ankle
        foot_feat = refined[:, :, 10:12].reshape(batch * seq_len, -1)
        foot = self.foot_head(foot_feat).reshape(batch, seq_len, 18)

        # Arm swivel: tokens [12:14]
        arm_swivel_feat = refined[:, :, 12:14].reshape(batch * seq_len, -1)
        arm_swivel = self.arm_swivel_head(arm_swivel_feat).reshape(batch, seq_len, 4)

        # Leg swivel: tokens [14:16]
        leg_swivel_feat = refined[:, :, 14:16].reshape(batch * seq_len, -1)
        leg_swivel = self.leg_swivel_head(leg_swivel_feat).reshape(batch, seq_len, 4)

        return {
            'torso': torso,              # (batch, seq_len, 42) - 7 joints
            'arm_swivel': arm_swivel,    # (batch, seq_len, 4)
            'leg_swivel': leg_swivel,    # (batch, seq_len, 4)
            'foot': foot,                # (batch, seq_len, 18) - pos + rot per ankle
            'init_output': init_output,  # (batch, seq_len, 68) for Stage 1 loss
        }


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    # Test network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create network
    net = MANIKINNetworkJLM(
        embed_dim=512,
        joint_embed_dim=256,
        num_layer=6,
        s_layer=1,
        t_layer=1,
        nhead=8,
        reg_hidden_dim=1024,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total_params:,}')

    # Test forward pass
    batch_size = 4
    seq_len = 40
    sparse_input = torch.randn(batch_size, seq_len, 54).to(device)

    output = net(sparse_input)

    print(f'\nInput shape: {sparse_input.shape}')
    print(f'Output shapes:')
    print(f'  - torso: {output["torso"].shape} (7 joints: pelvis + spine1-3 + neck + collar)')
    print(f'  - arm_swivel: {output["arm_swivel"].shape}')
    print(f'  - leg_swivel: {output["leg_swivel"].shape}')
    print(f'  - foot: {output["foot"].shape} (2 ankles: pos(3) + rot(6D) each)')
    print(f'  - init_output: {output["init_output"].shape}')

    # Verify output dimensions
    assert output['torso'].shape == (batch_size, seq_len, 42), \
        f"Expected torso shape (batch, seq, 42), got {output['torso'].shape}"
    assert output['arm_swivel'].shape == (batch_size, seq_len, 4)
    assert output['leg_swivel'].shape == (batch_size, seq_len, 4)
    assert output['foot'].shape == (batch_size, seq_len, 18), \
        f"Expected foot shape (batch, seq, 18), got {output['foot'].shape}"
    assert output['init_output'].shape == (batch_size, seq_len, 68), \
        f"Expected init_output shape (batch, seq, 68), got {output['init_output'].shape}"

    print('\nAll tests passed!')
