"""
Franka dynamics model: predicts future joint positions given current state and
a chunk of joint-velocity + gripper actions.

Normalization bounds are derived from 1%/99% percentiles of the training data:
  joint_vel    [-0.408, 0.490] … [-0.692, 0.810]  (per-joint)
  joint_delta  [-0.280, 0.283] … [-0.451, 0.470]  (per-joint)
  gripper_pos  [0.057, 0.996]
  gripper_delta [-0.938, 0.938]

Gripper actions are binarized (< 0.5 → 0, ≥ 0.5 → 1) before being fed to the
network.  The gripper loss is weighted 5× higher than the joint loss during
training (see train.py).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Dynamics(nn.Module):
    """MLP that maps (current state, action chunk) → future joint + gripper positions."""

    def __init__(self, action_dim: int, action_num: int, hidden_size: int,
                 gripper_loss_weight: float = 5.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.gripper_loss_weight = gripper_loss_weight

        # Per-joint normalization bounds (1st / 99th percentile)
        self.joint_vel_01 = np.array([-0.4077107, -0.79047304, -0.47850373, -0.8666644,
                                       -0.6729502, -0.5602032, -0.692411])[None, :]
        self.joint_vel_99 = np.array([ 0.4900636,  0.7259861,  0.45910007,  0.79220384,
                                        0.69864315,  0.648198,  0.810115])[None, :]
        self.joint_delta_01 = np.array([-0.2801219, -0.397792, -0.22935797, -0.3351759,
                                         -0.42025003, -0.36825255, -0.450706])[None, :]
        self.joint_delta_99 = np.array([ 0.2827909,  0.42184818,  0.33529875,  0.35958457,
                                          0.375613,   0.44463825,  0.4697690])[None, :]

        # Gripper normalization bounds
        self.gripper_pos_01   = np.array([0.057269])
        self.gripper_pos_99   = np.array([0.995595])
        self.gripper_delta_01 = np.array([-0.938326])
        self.gripper_delta_99 = np.array([ 0.938326])

        input_dim  = int((action_dim + 1) * (action_num + 1))
        output_dim = int(action_num * (action_dim + 1))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_dim),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, joint, joint_vel, joint_delta, gripper, gripper_action,
                gripper_delta, training: bool = True):
        """
        joint:          (B, action_dim)  or (B, 1, action_dim) — current joint positions
        joint_vel:      (action_num, action_dim) or (B, action_num, action_dim)
        joint_delta:    same shape as joint_vel — target deltas (training only)
        gripper:        (B, 1) or (B, 1, 1) — current gripper position
        gripper_action: (action_num, 1) or (B, action_num, 1)
        gripper_delta:  same shape as gripper_action — target deltas (training only)

        Returns (training=True):  scalar loss
        Returns (training=False): (joint_future, gripper_future) numpy arrays
        """
        # Normalise input shapes to (B, T, D)
        if joint.ndim == 2:
            joint = joint[None, :]
        if joint_vel.ndim == 2:
            joint_vel = joint_vel[None, :]
        if gripper.ndim == 1:
            gripper = gripper[None, None, :]
        elif gripper.ndim == 2:
            gripper = gripper[:, None, :]
        if gripper_action.ndim == 2:
            gripper_action = gripper_action[None, :]

        joint       = torch.as_tensor(joint,       dtype=torch.float32).to(self.device)
        joint_vel   = torch.as_tensor(
            self._norm(joint_vel, self.joint_vel_01, self.joint_vel_99),
            dtype=torch.float32,
        ).to(self.device)
        gripper_n   = torch.as_tensor(
            self._norm(gripper, self.gripper_pos_01, self.gripper_pos_99),
            dtype=torch.float32,
        ).to(self.device)
        # Binarize gripper action (open/close signal)
        gripper_act = torch.as_tensor(
            (gripper_action >= 0.5).astype(np.float32), dtype=torch.float32
        ).to(self.device)

        B = joint.shape[0]
        x = torch.cat([joint.reshape(B, -1), gripper_n.reshape(B, -1),
                        joint_vel.reshape(B, -1), gripper_act.reshape(B, -1)], dim=1)
        pred = self.net(x)
        pred = einops.rearrange(pred, "b (t d) -> b t d", t=self.action_num, d=self.action_dim + 1)

        pred_joint   = pred[:, :, :self.action_dim]
        pred_gripper = pred[:, :, self.action_dim:]

        if training:
            jd = torch.as_tensor(
                self._norm(joint_delta, self.joint_delta_01, self.joint_delta_99),
                dtype=torch.float32,
            ).to(self.device)
            gd = torch.as_tensor(
                self._norm(gripper_delta, self.gripper_delta_01, self.gripper_delta_99),
                dtype=torch.float32,
            ).to(self.device)
            return F.mse_loss(pred_joint, jd) + self.gripper_loss_weight * F.mse_loss(pred_gripper, gd)

        # Inference: denormalize and integrate
        pj = self._denorm(pred_joint.detach().cpu().numpy(), self.joint_delta_01, self.joint_delta_99)
        pg = self._denorm(pred_gripper.detach().cpu().numpy(), self.gripper_delta_01, self.gripper_delta_99)

        joint_np   = joint.detach().cpu().numpy().reshape(B, self.action_dim)
        gripper_np = self._denorm(
            gripper_n.detach().cpu().numpy().reshape(B, 1),
            self.gripper_pos_01, self.gripper_pos_99,
        )

        return (joint_np[:, None, :] + pj)[0], (gripper_np[:, None, :] + pg)[0]

    @staticmethod
    def _norm(data, lo, hi, eps: float = 1e-8):
        return 2 * (data - lo) / (hi - lo + eps) - 1

    @staticmethod
    def _denorm(data, lo, hi, eps: float = 1e-8):
        return (data + 1) / 2 * (hi - lo + eps) + lo
