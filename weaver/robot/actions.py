"""PI policy ↔ WEAVER action conversion utilities.

The PI policy outputs joint-velocity actions at control rate (15 Hz).
WEAVER operates at video rate (5 Hz, RGB_SKIP=3 control steps per frame).
These helpers convert between the two representations.

Two conversion paths:
  - Velocity path (no dynamics model): sum joint velocities over RGB_SKIP steps.
  - Dynamics path (with dynamics model): predict future absolute positions via
    the Ctrl-World dynamics model, then take state differences at video rate.
"""

from __future__ import annotations

import numpy as np
import torch

RGB_SKIP = 3       # control steps per video frame
DYNAMICS_STEPS = 15  # dynamics model horizon


# ---------------------------------------------------------------------------
# Image preprocessing for PI observations
# ---------------------------------------------------------------------------

def preprocess_pi_image(img: np.ndarray) -> np.ndarray:
    """Resize uint8 (H, W, 3) → (180, 320) then zero-pad to (224, 224)."""
    from openpi_client import image_tools
    ten = torch.from_numpy(img).to(torch.uint8)
    ten = torch.nn.functional.interpolate(
        ten.permute(2, 0, 1).unsqueeze(0).float(),
        size=(180, 320), mode="bilinear", align_corners=False,
    ).squeeze(0).permute(1, 2, 0).to(torch.uint8)
    return image_tools.resize_with_pad(ten.cpu().numpy(), 224, 224)


# ---------------------------------------------------------------------------
# Dynamics model helper
# ---------------------------------------------------------------------------

def joint_vel_to_positions(dynamics_model, current_joint, current_gripper,
                             joint_vel_chunk, gripper_action_chunk,
                             n_steps: int = DYNAMICS_STEPS):
    """Predict future absolute joint + gripper positions via the dynamics model.

    Pads the chunk to n_steps if shorter; returns only up to len(chunk) frames.
    """
    n_pi = len(joint_vel_chunk)
    if n_pi < n_steps:
        pad = n_steps - n_pi
        jv = np.concatenate([joint_vel_chunk,      np.tile(joint_vel_chunk[-1:],      (pad, 1))], axis=0)
        gp = np.concatenate([gripper_action_chunk, np.tile(gripper_action_chunk[-1:], (pad, 1))], axis=0)
    else:
        jv, gp = joint_vel_chunk[:n_steps], gripper_action_chunk[:n_steps]

    future_j, future_g = dynamics_model(
        current_joint[None, :], jv, None,
        gripper=current_gripper, gripper_action=gp, gripper_delta=None, training=False,
    )
    valid = min(n_pi, n_steps)
    return future_j[:valid], future_g[:valid]


# ---------------------------------------------------------------------------
# Batched PI chunk → normalized WM action conversion
# ---------------------------------------------------------------------------

def pi_chunks_to_wm_actions(
    chunks_BN: np.ndarray,       # (B*N, n_ctrl, 8)
    action_mean: np.ndarray,
    action_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    state_norms_BN: np.ndarray,  # (B*N, 8) normalized current states
    horizon: int,
    dynamics_model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert B*N PI action chunks to normalized WM video-rate actions and states.

    Velocity path (dynamics_model=None):
        Each video-rate step sums joint velocities over RGB_SKIP control steps.
        Gripper is taken from the last step in each window.

    Dynamics path (dynamics_model provided):
        Runs the Ctrl-World dynamics model to predict absolute future positions,
        then subsamples at video rate and computes state differences.

    Returns:
        a_norm: (B*N, T, 8) normalized WM actions
        s_norm: (B*N, T, 8) normalized WM states
    """
    BN, n_ctrl = chunks_BN.shape[:2]
    curr_states = state_norms_BN * state_std + state_mean

    if dynamics_model is not None:
        joint_vel, gripper_pos = chunks_BN[:, :, :7], chunks_BN[:, :, 7:]
        a_list, s_list = [], []
        for bn in range(BN):
            cur_j, cur_g = curr_states[bn, :7], curr_states[bn, 7:8]
            future_j, future_g = joint_vel_to_positions(
                dynamics_model, cur_j, cur_g, joint_vel[bn], gripper_pos[bn]
            )
            idx = np.arange(RGB_SKIP - 1, len(future_j), RGB_SKIP)
            if len(idx) == 0:
                a_list.append(np.zeros((0, 8), np.float32))
                s_list.append(np.zeros((0, 8), np.float32))
                continue
            fj, fg = future_j[idx], future_g[idx]
            all_j = np.concatenate([cur_j[None], fj], axis=0)
            all_g = np.concatenate([cur_g[None], fg], axis=0)
            a_list.append(np.concatenate([all_j[1:] - all_j[:-1], all_g[1:] - all_g[:-1]], axis=-1).astype(np.float32))
            s_list.append(np.concatenate([fj, fg], axis=-1).astype(np.float32))
        T = max((len(a) for a in a_list), default=0)
        if T == 0:
            return np.zeros((BN, 0, 8), np.float32), np.zeros((BN, 0, 8), np.float32)
        def _pad(arr):
            if len(arr) < T:
                arr = np.concatenate([arr, np.zeros((T - len(arr), 8), np.float32)], axis=0)
            return arr[:T]
        actions = np.stack([_pad(a) for a in a_list], axis=0)[:, :horizon]
        states  = np.stack([_pad(s) for s in s_list], axis=0)[:, :horizon]
    else:
        joint_vel, gripper_pos = chunks_BN[:, :, :7], chunks_BN[:, :, 7:]
        running_joints = curr_states[:, :7].copy()
        a_list, s_list = [], []
        for i in np.arange(0, n_ctrl, RGB_SKIP):
            end   = min(i + RGB_SKIP, n_ctrl)
            a_sum = joint_vel[:, i:end].sum(axis=1)
            g_last = gripper_pos[:, end - 1, 0]
            a_list.append(np.concatenate([a_sum, g_last[:, None]], axis=-1))
            running_joints += a_sum
            s_list.append(np.concatenate([running_joints.copy(), g_last[:, None]], axis=-1))
        if not a_list:
            return np.zeros((BN, 0, 8), np.float32), np.zeros((BN, 0, 8), np.float32)
        actions = np.stack(a_list, axis=1)[:, :horizon]
        states  = np.stack(s_list, axis=1)[:, :horizon]

    a_norm = ((actions - action_mean) / (action_std + 1e-8)).astype(np.float32)
    s_norm = ((states  - state_mean)  / (state_std  + 1e-8)).astype(np.float32)
    return a_norm, s_norm
