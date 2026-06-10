#!/usr/bin/env python3
"""
Real-robot deploy loop: interleaved PI policy execution + WEAVER imagination.

At each chunk boundary:
  1. Queries PI for N action samples.
  2. Converts each to WEAVER video-rate actions and rolls out the WM.
  3. Scores each rollout with reward model (+ critic for advantage mode).
  4. Executes the best sample on the robot and saves imagined frames.

KV-cache mode (--use-kv-cache) reuses prefix K/V across denoising steps via
WEAVER.generate_latent_rollouts_cached for faster imagination.

Usage:
    python -m weaver.deploy_pi_sample \\
        --checkpoint /path/to/chkpts \\
        --output-dir ./deploy_out \\
        --task "pick up the cup" \\
        --use-ema --open-loop-horizon 9
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys.experimental.motion_utils import reset_joints_to
from easydict import EasyDict
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R

from .robot.actions import pi_chunks_to_wm_actions, preprocess_pi_image, RGB_SKIP
from .robot.panda import prevent_keyboard_interrupt, step
from .utils.viz import resize_to, save_video_mp4
from .utils.wm_eval import (
    load_eval_config, build_model, load_norm_stats,
    load_checkpoint_into_model, wm_rollout_and_score, load_dynamics_model,
)
from .wm.model import WEAVER


# ---------------------------------------------------------------------------
# KV-cache rollout wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_cached(wm: WEAVER, context, hist_actions, future_actions, future_states,
                     memory_tokens=None) -> dict:
    """Rollout via KV-cache (no reward scoring).

    Builds the combined (context + future) input tensors and calls
    generate_latent_rollouts_cached, which reuses prefix K/V across
    denoising steps for faster inference at deploy time.
    """
    B, Hf, _ = future_actions.shape
    if Hf <= 0:
        return context
    x1 = {k: torch.cat([context[k], torch.zeros(B, Hf, *context[k].shape[2:],
                         device=context[k].device, dtype=context[k].dtype)], dim=1)
          for k in wm._img_keys}
    x1["states"] = torch.cat([context["states"], future_states], dim=1)
    hist = (hist_actions if hist_actions is not None
            else torch.zeros(B, wm._n_history, wm.wm._n_actions,
                             device=future_actions.device, dtype=future_actions.dtype))
    return wm.generate_latent_rollouts_cached(x1, torch.cat([hist, future_actions], dim=1),
                                               memory_tokens=memory_tokens)


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def next_traj_index(output_dir: str) -> int:
    root = Path(output_dir)
    if not root.is_dir():
        return 0
    m = max((int(p.name[5:]) for p in root.iterdir()
             if p.is_dir() and p.name.startswith("traj_") and p.name[5:].isdigit()), default=-1)
    return m + 1


def eef_pose_to_state(T_matrix, gripper):
    T = np.array(T_matrix).reshape(4, 4).T
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_euler('xyz')])


def get_observation(cr_interfaces, robot_interface, camera_ids):
    last_state   = robot_interface._state_buffer[-1]
    last_gripper = robot_interface._gripper_state_buffer[-1]
    imgs = {f"camera_{cid}": cr_interfaces[cid].get_img()['color'] for cid in camera_ids}
    return {
        "joint_position":      np.array(last_state.q),
        "gripper_position":    1 - np.array(last_gripper.width) / 0.085,
        "gripper_position_pi": np.array(last_gripper.width),
        "eef_pos":             eef_pose_to_state(last_state.O_T_EE, last_gripper.width),
        "wrist_image":         imgs["camera_0"],
        "left_image":          imgs["camera_1"],
        "right_image":         imgs["camera_2"],
    }


def encode_real_frame(wm, obs_rgb, state_norm, image_size, img_keys, device):
    """Encode a single real observation frame into WM latent features."""
    x = {}
    for k in img_keys:
        frame = obs_rgb["wrist_image"] if k == "wrist_left" else obs_rgb["right_image"]
        ten = torch.from_numpy(frame).permute(2, 0, 1).float().to(device) / 255.0 * 2 - 1
        ten = torch.nn.functional.interpolate(
            ten.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
        )
        ten = (ten / 2.0 + 0.5).clamp(0, 1).unsqueeze(0)
        x[k] = wm.im_encoder(ten)
    x["states"] = torch.from_numpy(state_norm).float().to(device).view(1, 1, -1)
    return x


def build_memory_tokens_from_real(wm, real_frame_list, img_keys, device):
    """Build memory tokens from t_memory-spaced real observation history."""
    if not wm._use_memory or wm._n_memory_frames <= 0:
        return None
    n_mem, t_mem = wm._n_memory_frames, wm._t_memory
    head = len(real_frame_list)
    mem_frames = [real_frame_list[max(0, head - (n_mem - j) * t_mem)] for j in range(n_mem)]
    mem = {f"{k}_features": torch.cat([f[k] for f in mem_frames], dim=1) for k in img_keys}
    mem["states"] = torch.cat([f["states"] for f in mem_frames], dim=1)
    with torch.no_grad():
        return wm.encode_memory_obs(mem)


# ---------------------------------------------------------------------------
# Per-chunk imagination + selection
# ---------------------------------------------------------------------------

def _prepare_context(wm, obs_buffer, hist_actions_buf, real_frame_list, img_keys, device):
    """Build context dict, history action window, and memory tokens from real-frame buffers."""
    context = {k: torch.cat([f[k] for f in obs_buffer], dim=1) for k in img_keys}
    context["states"] = torch.cat([f["states"] for f in obs_buffer], dim=1)
    hist_actions = torch.cat(list(hist_actions_buf), dim=1)
    mem_tokens   = build_memory_tokens_from_real(wm, real_frame_list, img_keys, device)
    return context, hist_actions, mem_tokens


def _run_new_chunk(wm, obs, context, hist_actions, mem_tokens,
                   instruction, chunk_idx, traj_dir, args,
                   img_keys, device, gamma, n_states, bootstrap,
                   current_state_norm, state_mean, state_std,
                   action_mean, action_std, dynamics_model, policy):
    """Query PI, imagine all N samples, score, and return the best sample's execution state.

    Returns:
        pred_action_chunk: (act_horizon, 8) control-rate actions to execute.
        fut_a:             (1, exec_len, A) normalized WM actions for the best sample.
        new_state_norm:    (8,) updated current state norm after execution window.
        pending_panels:    list of exec_len decoded frame dicts for visualization.
    """
    N = args.num_samples

    # ── Query PI for N action samples using the current real observation ──────
    req = {
        "observation/exterior_image_1_left": preprocess_pi_image(obs["right_image"]),
        "observation/wrist_image_left":      preprocess_pi_image(obs["wrist_image"]),
        "observation/joint_position":        obs["joint_position"],
        "observation/gripper_position":      obs["gripper_position"],
        "prompt":                            instruction,
    }
    with prevent_keyboard_interrupt():
        t0 = time.time()
        all_chunks = policy.infer_parallel(req, num_samples=N)["actions"]
        print(f"  PI inference: {time.time()-t0:.2f}s")

    # ── Convert PI control-rate chunks → normalized WM video-rate actions ─────
    state_norms = np.tile(current_state_norm[None], (N, 1))
    a_norms, s_norms = pi_chunks_to_wm_actions(
        all_chunks[:, :args.act_horizon],
        action_mean, action_std, state_mean, state_std,
        state_norms, wm._n_horizon, dynamics_model,
    )
    exec_len    = min(bootstrap, a_norms.shape[1])
    fut_a_batch = torch.from_numpy(a_norms[:, :exec_len]).to(device)
    fut_s_batch = torch.from_numpy(s_norms[:, :exec_len]).to(device)

    # Expand single-batch context to N for batched rollout
    ctx_batch  = {k: context[k].expand(N, -1, -1, -1) for k in img_keys}
    ctx_batch["states"] = context["states"].expand(N, -1, -1)
    hist_batch = hist_actions.expand(N, -1, -1)
    mem_batch  = mem_tokens.expand(N, -1, -1) if mem_tokens is not None else None
    task_embed = wm.encode_task({"text": [instruction]}).expand(N, -1)

    # ── Imagine: roll out WEAVER for all N samples, then score ────────────────
    t0 = time.time()
    if args.use_kv_cache:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            xt_batch     = _rollout_cached(wm, ctx_batch, hist_batch, fut_a_batch, fut_s_batch, mem_batch)
            actions_full = torch.cat([hist_batch, fut_a_batch], dim=1)
            rewards_all  = wm.rm(xt_batch, actions_full, task_embed)
            values_all   = (wm.critic(xt_batch, actions_full, task_embed)
                            if args.selection_criterion == "advantage" else None)
    else:
        xt_batch, rewards_all, values_all = wm_rollout_and_score(
            wm, ctx_batch, hist_batch, fut_a_batch, fut_s_batch, task_embed, mem_batch
        )
    print(f"  WM + score: {time.time()-t0:.2f}s")

    # ── Select best sample by reward sum or discounted advantage ──────────────
    chunk_rewards = rewards_all[:, wm._n_history - 1:-1].float()
    if args.selection_criterion == "advantage":
        H  = chunk_rewards.shape[1]
        gp = torch.tensor([gamma**i for i in range(H)], device=device)
        v0 = values_all[:, wm._n_history - 1].float()
        vH = values_all[:, -1].float()
        scores = (chunk_rewards * gp).sum(1) + (gamma**H) * vH - v0
    else:
        scores = chunk_rewards.sum(1)

    best = int(scores.argmax().item())
    print(f"  [chunk {chunk_idx}] scores={[f'{v:.3f}' for v in scores.tolist()]}  best={best}")

    # ── Save debug grid: last imagined frame per sample with reward label ──────
    with torch.no_grad():
        dec_last = wm.decode_obs(
            {**{k: xt_batch[k][:, -1:] for k in img_keys},
             "states": xt_batch["states"][:, -1:]},
            chunk_size=N,
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    for si in range(N):
        views = [(dec_last[k][si, 0].float().cpu().numpy().transpose(1, 2, 0) * 255)
                  .clip(0, 255).astype(np.uint8) for k in img_keys]
        panel = np.ascontiguousarray(np.concatenate(views, axis=1))
        color = (0, 255, 0) if si == best else (255, 255, 255)
        cv2.putText(panel, f"s{si} R={chunk_rewards[si].sum():.3f}{' *' if si==best else ''}",
                    (5, 20), font, 0.5, color, 1)
        rows.append(panel)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(traj_dir, f"chunk{chunk_idx:04d}_{ts}.png"),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    # ── Extract best sample state for robot execution ─────────────────────────
    pred_action_chunk  = all_chunks[best][:args.act_horizon]
    fut_a              = fut_a_batch[best:best+1]
    new_state_norm     = s_norms[best, min(exec_len - 1, s_norms.shape[1] - 1)]

    # Decode imagined future frames of the best sample for visualization overlay
    xt_best    = {k: xt_batch[k][best:best+1] for k in img_keys}
    xt_best["states"] = xt_batch["states"][best:best+1]
    pred_feats = {**{k: xt_best[k][:, wm._n_history:wm._n_history+exec_len] for k in img_keys},
                  "states": xt_best["states"][:, wm._n_history:wm._n_history+exec_len]}
    with torch.no_grad():
        dec = wm.decode_obs(pred_feats, chunk_size=16)
    pending_panels = [
        {k: (dec[k][0, i].float().cpu().numpy().transpose(1, 2, 0) * 255)
              .clip(0, 255).astype(np.uint8) for k in img_keys}
        for i in range(exec_len)
    ]

    return pred_action_chunk, fut_a, new_state_norm, pending_panels


def _commit_video_frame(wm, obs, state_mean, state_std, image_size, img_keys, device,
                         obs_buffer, real_frame_list, hist_actions_buf,
                         fut_a, actions_from_chunk,
                         gt_triplets, imagined_panels, pending_panels):
    """Commit one video-rate frame: update the real-frame buffers and sync GT/imagined panels.

    Called every RGB_SKIP control steps to keep the video-rate observation
    buffers and visualization lists aligned.
    """
    s_raw  = np.concatenate([obs["joint_position"], [obs["gripper_position"]]]).astype(np.float32)
    s_norm = (s_raw - state_mean) / (state_std + 1e-8)
    x_new  = encode_real_frame(wm, obs, s_norm, image_size, img_keys, device)
    obs_buffer.append(x_new)
    real_frame_list.append(x_new)
    if fut_a is not None:
        vi = min((actions_from_chunk - 1) // RGB_SKIP, fut_a.shape[1] - 1)
        hist_actions_buf.append(fut_a[:, vi:vi+1])
    if pending_panels:
        gt_triplets.append((obs["right_image"], obs["left_image"], obs["wrist_image"]))
        imagined_panels.append(pending_panels.pop(0))


def _save_trial_video(gt_triplets, imagined_panels, img_keys, traj_dir, fps):
    """Save side-by-side GT (top) / imagined WM (bottom) video for one trial."""
    n = min(len(gt_triplets), len(imagined_panels))
    if n == 0:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frames = []
    for i in range(n):
        gt_r, gt_l, gt_w = gt_triplets[i]
        pan = imagined_panels[i]
        im_l = resize_to(pan.get('exterior_1_left'), gt_l.shape[:2])
        im_w = resize_to(pan.get('wrist_left'),      gt_w.shape[:2])
        frames.append(np.concatenate([
            np.concatenate([gt_r,  gt_l,  gt_w],  axis=1),
            np.concatenate([im_l, np.zeros_like(gt_r), im_w], axis=1),
        ], axis=0))
    save_video_mp4(os.path.join(traj_dir, f"combined_{ts}.mp4"), np.stack(frames), fps)
    print("Saved combined video.")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Interleaved PI deploy with WEAVER imagination")
    p.add_argument("--checkpoint",        type=str, required=True)
    p.add_argument("--dynamics-model",    type=str, default=None)
    p.add_argument("--output-dir",        type=str, required=True)
    p.add_argument("--task",              type=str, required=True)
    p.add_argument("--pi-host",           type=str, default="intent-chai.lan.local.cmu.edu")
    p.add_argument("--pi-port",           type=int, default=8000)
    p.add_argument("--max-steps",         type=int, default=700)
    p.add_argument("--open-loop-horizon", type=int, default=8)
    p.add_argument("--act-horizon",       type=int, default=15)
    p.add_argument("--fps",               type=int, default=10)
    p.add_argument("--num-samples",       type=int, default=1)
    p.add_argument("--controller-type",   type=str, default="JOINT_IMPEDANCE")
    p.add_argument("--controller-cfg",    type=str, default="joint-impedance-controller.yml")
    p.add_argument("--interface-cfg",     type=str, default="charmander.yml")
    p.add_argument("--init-joints",       type=float, nargs=7,
                   default=[0.0933692, 0.07232527, -0.03192432, -2.17384338,
                             -0.01927867, 2.26411851, 0.07160476])
    p.add_argument("--use-ema",           action="store_true")
    p.add_argument("--torch-compile",     action="store_true")
    p.add_argument("--selection-criterion", type=str, default="reward",
                   choices=["reward", "advantage"])
    p.add_argument("--use-kv-cache",      action="store_true")
    p.add_argument("--dataset-path",      type=str, default="/data/yilin/world_model_data_ours_v2")
    p.add_argument("--relabel-action",    action="store_true")
    p.add_argument("--overrides",         nargs="*", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = load_eval_config(args.checkpoint, args.overrides)
    wm, img_keys, image_size = build_model(
        cfg, device,
        img_keys=["wrist_left", "exterior_1_left"],
        val_steps=4,
        inference_overrides={"pyramid_stagger_width": 0},
    )
    load_checkpoint_into_model(wm, args.checkpoint, device, use_ema=args.use_ema)
    if not args.use_kv_cache and args.torch_compile and hasattr(torch, "compile"):
        wm = torch.compile(wm)
    wm.eval()

    bootstrap = args.open_loop_horizon // RGB_SKIP
    gamma     = getattr(cfg.model, "discount_factor", 0.99)
    n_states  = getattr(cfg.dataset, "n_states", 8)
    print(f"bootstrap={bootstrap}  gamma={gamma}")

    # ── Norm stats + optional dynamics model ──────────────────────────────────
    state_mean, state_std, action_mean, action_std = load_norm_stats(
        args.dataset_path, args.relabel_action
    )
    dynamics_model = None
    if args.relabel_action:
        if args.dynamics_model is None:
            raise ValueError("--dynamics-model required with --relabel-action")
        dynamics_model = load_dynamics_model(args.dynamics_model, device)

    # ── Robot + camera setup ──────────────────────────────────────────────────
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), control_freq=15)
    init_joints     = np.array(args.init_joints, dtype=np.float64)
    reset_joints_to(robot_interface, init_joints)

    camera_ids  = [0, 1, 2]
    cr_interfaces = {}
    camera_info = EasyDict({"camera_id": camera_ids,
                             "camera_type": ["zed", "zed", "zed"],
                             "camera_name": [f"camera_zed_{i}" for i in camera_ids]})
    for cid in camera_ids:
        ci = CameraRedisSubInterface(camera_info=camera_info, camera_id=cid)
        ci.start()
        cr_interfaces[cid] = ci

    controller_cfg = YamlConfig(os.path.join(config_root, args.controller_cfg)).as_easydict()
    policy         = websocket_client_policy.WebsocketClientPolicy(args.pi_host, args.pi_port)

    # ── Trial loop ────────────────────────────────────────────────────────────
    while True:
        instruction = input(f"Enter instruction (default: {args.task!r}): ").strip() or args.task
        traj_n   = next_traj_index(args.output_dir)
        traj_dir = os.path.join(args.output_dir, f"traj_{traj_n:03d}")
        os.makedirs(traj_dir, exist_ok=True)
        print(f"Trial traj_{traj_n:03d}: {instruction!r}")

        # Initialise from the first real observation
        obs0    = get_observation(cr_interfaces, robot_interface, camera_ids)
        s0      = np.concatenate([obs0["joint_position"], [obs0["gripper_position"]]]).astype(np.float32)
        s0_norm = (s0 - state_mean) / (state_std + 1e-8)
        x_real  = encode_real_frame(wm, obs0, s0_norm, image_size, img_keys, device)

        obs_buffer       = collections.deque([x_real] * wm._n_history, maxlen=wm._n_history)
        real_frame_list  = [x_real]
        hist_actions_buf = collections.deque(
            [torch.zeros(1, 1, wm.wm._n_actions, device=device)] * wm._n_history,
            maxlen=wm._n_history,
        )

        current_state_norm  = s0_norm.copy()
        pred_action_chunk   = None
        actions_from_chunk  = 0
        chunk_idx           = 0
        fut_a               = None
        pending_panels      = []
        steps_since_commit  = 0
        gt_triplets         = []
        imagined_panels     = []
        obs                 = obs0

        # ── Per-step execution loop ───────────────────────────────────────────
        for t in range(args.max_steps):
            try:
                # At each chunk boundary: imagine N futures, select the best one
                if pred_action_chunk is None or actions_from_chunk >= args.open_loop_horizon:
                    actions_from_chunk = 0
                    chunk_idx += 1
                    context, hist_actions, mem_tokens = _prepare_context(
                        wm, obs_buffer, hist_actions_buf, real_frame_list, img_keys, device
                    )
                    pred_action_chunk, fut_a, current_state_norm, pending_panels = _run_new_chunk(
                        wm, obs, context, hist_actions, mem_tokens,
                        instruction, chunk_idx, traj_dir, args,
                        img_keys, device, gamma, n_states, bootstrap,
                        current_state_norm, state_mean, state_std,
                        action_mean, action_std, dynamics_model, policy,
                    )

                # Execute one control action on the robot
                act = pred_action_chunk[actions_from_chunk].copy()
                act[-1] = 1.0 if act[-1] > 0.5 else -1.0
                actions_from_chunk += 1
                step(np.clip(act, -1, 1), robot_interface=robot_interface,
                     controller_cfg=controller_cfg, controller_type=args.controller_type,
                     step_count=t)
                time.sleep(max(0.0, 1 / 15.0))

                obs = get_observation(cr_interfaces, robot_interface, camera_ids)
                steps_since_commit += 1

                # Every RGB_SKIP control steps = one video-rate frame
                if steps_since_commit >= RGB_SKIP:
                    _commit_video_frame(
                        wm, obs, state_mean, state_std, image_size, img_keys, device,
                        obs_buffer, real_frame_list, hist_actions_buf,
                        fut_a, actions_from_chunk,
                        gt_triplets, imagined_panels, pending_panels,
                    )
                    steps_since_commit = 0

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt — ending trial.")
                break

        # ── Post-trial: save video and prompt for next action ─────────────────
        if input("Save combined video (GT top, WM bottom)? (y/n): ").strip().lower() == "y":
            _save_trial_video(gt_triplets, imagined_panels, img_keys, traj_dir, args.fps)

        if input("Reset robot? (y/n): ").strip().lower() == "y":
            reset_joints_to(robot_interface, init_joints)

        if input("Run another trial? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
