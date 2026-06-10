#!/usr/bin/env python3
"""
Batch synthetic trajectory generation via PI policy + WEAVER imagination.

For each batch of B dataset segments:
  1. Encode n_history context frames and query PI for N action samples per segment.
  2. Expand B → B*N and run M imagination chunks through WEAVER.
  3. Select the best sample per segment by reward or advantage.
  4. Decode the winner, build annotation, and save video + JSON.

Usage:
    python -m weaver.synth_data_gen \\
        --checkpoint /path/to/chkpts \\
        --dataset-path /path/to/droid \\
        --output-dir ./data_gen_out \\
        --open-loop-horizon 9 --use-ema --num-trajectories 1000 \\
        --filter-episode-id stack --selection-criterion advantage
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .datasets.dataset import create_synth_dataloader
from .dynamics.model import Dynamics
from openpi_client import image_tools, websocket_client_policy
from .robot.actions import pi_chunks_to_wm_actions, preprocess_pi_image, RGB_SKIP
from .robot.fk import compute_fk_cartesian
from .utils.viz import stamp_reward_on_frames, resize_frames, save_video_mp4, save_trajectory
from .utils.wm_eval import (
    load_eval_config, build_model, load_norm_stats,
    load_dynamics_model, load_checkpoint_into_model, wm_rollout_and_score,
)


CAM_H, CAM_W = 720, 1280

_thread_local = threading.local()

TASK_MAP = {
    "marker": "pick up the marker and place it in the cup",
    "towel":  "put the towel in the basket",
    "bag":    "pick up the bag of chips and place it on top of the green plate",
    "beans":  "pick up the cup of beans and place them in the bowl",
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """All accumulated data from the M-chunk imagination loop for one batch."""
    all_states:          np.ndarray    # (B*N, T, S) normalized states
    all_actions:         np.ndarray    # (B*N, T, A) normalized WM actions
    all_raw_pi:          np.ndarray    # (B*N, T_ctrl, 8) raw PI control-rate actions
    all_rewards:         np.ndarray    # (B*N, T) per-step reward model outputs
    all_values:          np.ndarray    # (B*N, T) per-step critic outputs
    all_latents:         dict          # {img_key: (B*N, T, N_patches, D)} predicted latents
    accumulated_rewards: torch.Tensor  # (B, N) sum of per-step rewards across all chunks
    bootstrap_values:    torch.Tensor  # (B*N,) critic value at the last predicted step


# ---------------------------------------------------------------------------
# PI helpers
# ---------------------------------------------------------------------------

def _get_policy(host, port):
    if not hasattr(_thread_local, "policy"):
        _thread_local.policy = websocket_client_policy.WebsocketClientPolicy(host, port)
    return _thread_local.policy


def _resolve_task_text(args_task, data_texts):
    raw = args_task if args_task else (data_texts[0] if data_texts else "")
    return next((v for k, v in TASK_MAP.items() if k in raw), raw)


def _build_pi_reqs_step1(obs, state_raw_B, task_text, wrist_key, ext_key, n_history, B):
    """Build B PI request dicts using real history frames (no WM decode needed)."""
    hi = n_history - 1
    wrist_B, ext_B = obs[wrist_key][:, hi], obs[ext_key][:, hi]
    return [{
        "observation/exterior_image_1_left": preprocess_pi_image(_t2u8(ext_B[b])),
        "observation/wrist_image_left":      preprocess_pi_image(_t2u8(wrist_B[b])),
        "observation/joint_position":        state_raw_B[b, :7].cpu().numpy(),
        "observation/gripper_position":      state_raw_B[b, 7:8].cpu().numpy(),
        "prompt":                            task_text,
    } for b in range(B)]


def _build_pi_reqs_subsequent(dec_last, last_states_norm, state_std, state_mean,
                               task_text, wrist_key, ext_key, N):
    """Build B*N PI request dicts from decoded imagined frames."""
    BN = last_states_norm.shape[0]
    state_raws = last_states_norm * state_std + state_mean
    wrist_imgs = (dec_last[wrist_key][:, 0].float().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    ext_imgs   = (dec_last[ext_key][:, 0].float().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    return [{
        "observation/exterior_image_1_left": preprocess_pi_image(ext_imgs[bn]),
        "observation/wrist_image_left":      preprocess_pi_image(wrist_imgs[bn]),
        "observation/joint_position":        state_raws[bn, :7],
        "observation/gripper_position":      state_raws[bn, 7:8],
        "prompt":                            task_text,
    } for bn in range(BN)]


def _t2u8(t):
    return (t.float().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def _chunks_to_wm_tensors(chunks_np, state_norms, action_mean, action_std, state_mean, state_std,
                            horizon, n_actions, n_states, exec_len, device, dynamics_model):
    """Convert PI action chunks (numpy) to WEAVER-ready (fut_a, fut_s) torch tensors."""
    a_norms, s_norms = pi_chunks_to_wm_actions(
        chunks_np, action_mean, action_std, state_mean, state_std,
        state_norms, horizon, dynamics_model=dynamics_model,
    )
    BN = chunks_np.shape[0]
    fut_a = torch.zeros(BN, exec_len, n_actions, device=device)
    fut_s = torch.zeros(BN, exec_len, n_states,  device=device)
    el = min(exec_len, a_norms.shape[1])
    if el > 0:
        fut_a[:, :el] = torch.from_numpy(a_norms[:, :el]).to(device)
        fut_s[:, :el] = torch.from_numpy(s_norms[:, :el]).to(device)
    return fut_a, fut_s


# ---------------------------------------------------------------------------
# Batch processing — the four core phases of each generation batch
# ---------------------------------------------------------------------------

def _prepare_batch_context(data, wm, img_keys, n_history, B, N, device,
                            state_std_t, state_mean_t, task_text,
                            executor, query_parallel,
                            wrist_key, ext_key, exec_ctrl_len, exec_len,
                            n_actions, n_states, action_mean, action_std,
                            state_mean, state_std, dynamics_model):
    """Phase 1 — Encode the real context and query PI for the first action chunk.

    Encodes the n_history real frames as WM context, queries PI for N action samples
    using those real frames (no imagination needed yet), then expands everything from
    B segments to B*N for batched rollout.

    Returns a tuple:
        (context_BN, task_embed_BN, hist_actions_BN,
         state_norm_B, state_raw_B, state_norms_BN,
         raw_pi_chunks, fut_a_BN, fut_s_BN)
    """
    obs = data["obs"]

    raw_obs = {f"{k}_features": obs[f"{k}_features"][:, :n_history].to(device) for k in img_keys}
    raw_obs["states"] = obs["states"][:, :n_history].to(device)
    with torch.no_grad():
        context_B = wm.encode_obs(raw_obs)

    state_norm_B = obs["states"][:, n_history - 1].to(device)
    state_raw_B  = state_norm_B * state_std_t + state_mean_t

    task_feats   = data["task"].get("features")
    task_embed_B = (task_feats.to(device).float() if task_feats is not None
                    else wm.encode_task({"text": [task_text] * B}))

    # First PI query uses real frames — no WM decode needed
    reqs_B       = _build_pi_reqs_step1(obs, state_raw_B, task_text, wrist_key, ext_key, n_history, B)
    chunks_BN_np = np.concatenate(list(executor.map(query_parallel, reqs_B)), axis=0)

    # Expand B → B*N for batched WM rollout
    context_BN      = {k: context_B[k].repeat_interleave(N, dim=0) for k in img_keys}
    context_BN["states"] = context_B["states"].repeat_interleave(N, dim=0)
    task_embed_BN   = task_embed_B.repeat_interleave(N, dim=0)
    hist_actions_BN = data["actions"][:, :n_history].to(device).repeat_interleave(N, dim=0)

    state_norms_BN = state_norm_B.cpu().numpy().repeat(N, axis=0)
    fut_a_BN, fut_s_BN = _chunks_to_wm_tensors(
        chunks_BN_np, state_norms_BN, action_mean, action_std, state_mean, state_std,
        wm._n_horizon, n_actions, n_states, exec_len, device, dynamics_model,
    )

    return (context_BN, task_embed_BN, hist_actions_BN,
            state_norm_B, state_raw_B, state_norms_BN,
            [chunks_BN_np[:, :exec_ctrl_len].copy()], fut_a_BN, fut_s_BN)


def _run_imagination_loop(wm, context_BN, hist_actions_BN, task_embed_BN,
                           fut_a_BN, fut_s_BN, raw_pi_chunks,
                           M, exec_len, n_history, B, N, n_actions, n_states,
                           img_keys, device, action_mean, action_std, state_mean, state_std,
                           dynamics_model, exec_ctrl_len,
                           task_text, wrist_key, ext_key, executor, query_single) -> RolloutResult:
    """Phase 2 — Run M imagination chunks through WEAVER, interleaved with PI queries.

    Each chunk:
      a. Rolls out WEAVER for exec_len video-rate steps and scores with reward + critic.
      b. Updates the t_memory-spaced frame buffer for memory tokens.
      c. Decodes the last imagined frame, queries PI, converts to WM actions for the
         next chunk (skipped on the final chunk).

    Returns a RolloutResult with data concatenated across all M chunks.
    """
    t_mem = wm._t_memory
    n_mem = wm._n_memory_frames if wm._use_memory else 0
    frame_buffer = {i: {**{k: context_BN[k][:, i].detach() for k in img_keys},
                         "states": context_BN["states"][:, i].detach()}
                    for i in range(n_history)}

    def get_memory_tokens(head):
        if not wm._use_memory or n_mem == 0:
            return None
        positions = [max(0, head - (n_mem - j) * t_mem) for j in range(n_mem)]
        mem_raw = {f"{k}_features": torch.stack([frame_buffer[p][k] for p in positions], dim=1)
                   for k in img_keys}
        mem_raw["states"] = torch.stack([frame_buffer[p]["states"] for p in positions], dim=1)
        with torch.no_grad():
            return wm.encode_memory_obs(mem_raw)

    states_chunks  = []
    actions_chunks = []
    latents_chunks = {k: [] for k in img_keys}
    rewards_chunks = []
    values_chunks  = []
    accumulated_rewards = torch.zeros(B, N, device=device)
    bootstrap_values_BN = None

    for m in range(M):
        head = n_history + m * exec_len

        # Imagine and score
        xt_batch, rewards_all, values_all = wm_rollout_and_score(
            wm, context_BN, hist_actions_BN, fut_a_BN, fut_s_BN,
            task_embed_BN, memory_tokens=get_memory_tokens(head),
        )
        print(f"  Chunk {m+1}/{M}")

        chunk_rewards = rewards_all[:, n_history - 1:-1]
        accumulated_rewards += chunk_rewards.sum(1).reshape(B, N)

        states_chunks.append(xt_batch["states"][:, n_history:n_history+exec_len].cpu().numpy())
        actions_chunks.append(fut_a_BN.float().cpu().numpy())
        rewards_chunks.append(chunk_rewards.cpu().numpy())
        values_chunks.append(values_all[:, n_history-1:-1].cpu().numpy())
        for k in img_keys:
            latents_chunks[k].append(xt_batch[k][:, n_history:n_history+exec_len].cpu().numpy())

        if m == M - 1:
            bootstrap_values_BN = values_all[:, -1]
            break

        # Update frame buffer and slide context window for next chunk
        for i in range(exec_len):
            pos = head + i
            frame_buffer[pos] = {k: xt_batch[k][:, n_history + i].detach() for k in img_keys}
            frame_buffer[pos]["states"] = xt_batch["states"][:, n_history + i].detach()

        context_BN      = {k: xt_batch[k][:, -n_history:].detach() for k in img_keys}
        context_BN["states"] = xt_batch["states"][:, -n_history:].detach()
        hist_actions_BN = torch.cat([hist_actions_BN, fut_a_BN], 1)[:, -n_history:].detach()

        # Decode last imagined frame → query PI → convert to WM actions
        with torch.no_grad():
            dec_last = wm.decode_obs(
                {**{k: xt_batch[k][:, -1:] for k in img_keys}, "states": xt_batch["states"][:, -1:]},
                chunk_size=B * N,
            )
        last_states_norm = xt_batch["states"][:, -1].cpu().numpy()
        reqs_BN = _build_pi_reqs_subsequent(
            dec_last, last_states_norm, state_std, state_mean, task_text, wrist_key, ext_key, N
        )
        chunks_BN_np = np.stack(list(executor.map(query_single, reqs_BN)), axis=0)
        raw_pi_chunks.append(chunks_BN_np[:, :exec_ctrl_len].copy())
        fut_a_BN, fut_s_BN = _chunks_to_wm_tensors(
            chunks_BN_np, last_states_norm, action_mean, action_std, state_mean, state_std,
            wm._n_horizon, n_actions, n_states, exec_len, device, dynamics_model,
        )

    return RolloutResult(
        all_states=np.concatenate(states_chunks,  axis=1),
        all_actions=np.concatenate(actions_chunks, axis=1),
        all_raw_pi=np.concatenate(raw_pi_chunks,   axis=1),
        all_rewards=np.concatenate(rewards_chunks, axis=1),
        all_values=np.concatenate(values_chunks,   axis=1),
        all_latents={k: np.concatenate(latents_chunks[k], axis=1) for k in img_keys},
        accumulated_rewards=accumulated_rewards,
        bootstrap_values=bootstrap_values_BN,
    )


def _compute_selection_scores(result: RolloutResult, B, N, gamma, selection_criterion):
    """Phase 3 — Compute per-segment selection scores and return the best sample index.

    Returns:
        best_ns:   (B,) int tensor — best sample index per segment.
        advantages: (B, N) float tensor (advantage criterion) or None (reward criterion).
    """
    if selection_criterion == "advantage":
        H  = result.all_rewards.shape[1]
        gp = gamma ** np.arange(H, dtype=np.float64)
        advantages = torch.from_numpy(
            ((result.all_rewards.astype(np.float64) * gp[None]).sum(1)
             + (gamma ** H) * result.bootstrap_values.cpu().numpy().astype(np.float64)
             - result.all_values[:, 0].astype(np.float64)
            ).reshape(B, N)
        ).float()
        print(f"  Advantages: {advantages.tolist()}")
        return advantages.argmax(dim=1), advantages
    else:
        print(f"  Returns: {result.accumulated_rewards.cpu().tolist()}")
        return result.accumulated_rewards.argmax(dim=1), None


def _save_debug_videos(result: RolloutResult, wm, best_ns, B, N, img_keys,
                        device, output_dir, batch_idx, fps) -> dict:
    """Decode all B*N imagined trajectories and save per-segment thumbnail comparison videos."""
    all_lat_t = {k: torch.from_numpy(result.all_latents[k]).to(device) for k in img_keys}
    all_lat_t["states"] = torch.from_numpy(result.all_states).to(device)
    with torch.no_grad():
        dec_all = wm.decode_obs(all_lat_t, chunk_size=B * N)
    all_frames = {k: dec_all[k].float().cpu().numpy() for k in img_keys}

    debug_dir = os.path.join(output_dir, "debug", f"batch{batch_idx:03d}")
    os.makedirs(debug_dir, exist_ok=True)
    DH, DW = CAM_H // 4, CAM_W // 4
    for b in range(B):
        best_n = int(best_ns[b].item())
        strips = []
        for n in range(N):
            bn = b * N + n
            frames_raw = {k: (all_frames[k][bn].transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                          for k in img_keys}
            strip = np.concatenate([resize_frames(frames_raw[k], DH, DW) for k in img_keys], axis=2)
            strip = stamp_reward_on_frames(strip, result.accumulated_rewards[b, n].item(), n == best_n,
                                           result.all_rewards[bn], result.all_values[bn])
            strips.append(strip)
        save_video_mp4(os.path.join(debug_dir, f"seg{b:02d}.mp4"),
                       np.concatenate(strips, axis=2), fps)
    return all_frames


def _save_winning_segment(b, bn, best_n, result: RolloutResult, advantages,
                           wm, state_raw_B, img_keys, ext_key, wrist_key,
                           exec_len, device, state_std, state_mean,
                           args, global_traj_idx, task_text, batch_idx, N, M,
                           all_frames=None) -> str | None:
    """Phase 4 — Decode, annotate, and save one winning trajectory.

    Returns the saved video path, or None if the segment was filtered out.
    """
    # Quality filter
    if args.selection_criterion == "advantage":
        best_adv = advantages[b, best_n].item()
        if best_adv < 0.1:
            print(f"  Segment {b}: SKIPPED — advantage {best_adv:.3f} < 0.1")
            return None
    else:
        if float(result.all_values[bn][-1]) - float(result.all_values[bn][0]) < -0.75:
            print(f"  Segment {b}: SKIPPED — critic Δv < -0.75")
            return None

    T          = result.all_states.shape[1]
    states_raw = result.all_states[bn] * state_std + state_mean

    # Decode the winning latents to pixel frames
    if all_frames is not None:
        frames_hwc_raw = {k: (all_frames[k][bn].transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                          for k in img_keys}
    else:
        best_feat = {k: torch.from_numpy(result.all_latents[k][bn:bn+1]).to(device) for k in img_keys}
        best_feat["states"] = torch.from_numpy(result.all_states[bn:bn+1]).to(device)
        with torch.no_grad():
            dec_best = wm.decode_obs(best_feat, chunk_size=exec_len)
        frames_hwc_raw = {k: (dec_best[k][0].float().cpu().numpy().transpose(0, 2, 3, 1) * 255)
                              .clip(0, 255).astype(np.uint8) for k in img_keys}
    frames_hwc = {k: resize_frames(frames_hwc_raw[k], CAM_H, CAM_W) for k in img_keys}
    video = np.concatenate([frames_hwc[ext_key], np.zeros((T, CAM_H, CAM_W, 3), np.uint8),
                             frames_hwc[wrist_key]], axis=2)

    # Compute state/action trajectories for the annotation
    init_j = state_raw_B[b, :7].cpu().numpy()
    init_g = state_raw_B[b, 7:8].cpu().numpy()
    obs_j  = np.concatenate([init_j[None], states_raw[:-1, :7]], axis=0)
    obs_g  = np.concatenate([init_g[None], states_raw[:-1, 7:8]], axis=0)
    raw_pi = result.all_raw_pi[bn]
    act_joint_pos = np.cumsum(raw_pi[:, :7], axis=0) + init_j[None]

    annotation = _build_annotation(
        global_idx=global_traj_idx, task_text=task_text, T=T,
        T_ctrl=raw_pi.shape[0], obs_joints=obs_j, obs_gripper=obs_g,
        obs_cartesian=compute_fk_cartesian(obs_j),
        act_joint_pos=act_joint_pos, raw_joint_vel=raw_pi[:, :7],
        raw_gripper=raw_pi[:, 7:8], act_cartesian=compute_fk_cartesian(act_joint_pos),
        rewards_per_step=result.all_rewards[bn].tolist(),
        critic_values_per_step=result.all_values[bn].tolist(),
        sample_returns=result.accumulated_rewards[b].cpu().tolist(),
        meta={
            "num_cameras": max(3, len(img_keys)),
            "batch_idx": batch_idx, "segment_in_batch": b,
            "best_sample_idx": best_n, "num_samples": N, "num_chunks": M,
            "open_loop_horizon": args.open_loop_horizon,
            "selection_criterion": args.selection_criterion,
            **({"sample_advantages": advantages[b].tolist()}
               if args.selection_criterion == "advantage" else {}),
        },
    )
    return save_trajectory(args.output_dir, global_traj_idx, video, annotation, args.fps)


# ---------------------------------------------------------------------------
# Annotation builder
# ---------------------------------------------------------------------------

def _build_annotation(global_idx, task_text, T, T_ctrl, obs_joints, obs_gripper, obs_cartesian,
                       act_joint_pos, raw_joint_vel, raw_gripper, act_cartesian,
                       rewards_per_step, critic_values_per_step, sample_returns, meta):
    return {
        "texts":           [task_text],
        "episode_id":      global_idx,
        "success":         0,
        "num_cameras":     meta.pop("num_cameras"),
        "video_path":      f"videos/{global_idx}.mp4",
        "video_length":    T,
        "raw_length":      T_ctrl,
        **meta,
        "sample_returns":             sample_returns,
        "rewards_per_step":           rewards_per_step,
        "critic_values_per_step":     critic_values_per_step,
        "observation.state.cartesian_position": obs_cartesian.tolist(),
        "observation.state.joint_position":     obs_joints.tolist(),
        "observation.state.gripper_position":   obs_gripper.tolist(),
        "action.joint_position":     act_joint_pos.tolist(),
        "action.joint_velocity":     raw_joint_vel.tolist(),
        "action.gripper_position":   raw_gripper.tolist(),
        "action.cartesian_position": act_cartesian.tolist(),
    }


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Batch synthetic traj generation via WEAVER + PI")
    p.add_argument("--checkpoint",        type=str, required=True)
    p.add_argument("--dataset-path",      type=str, required=True)
    p.add_argument("--output-dir",        type=str, required=True)
    p.add_argument("--task",              type=str, default=None,
                   help="Override instruction for all segments (uses dataset text if not set).")
    p.add_argument("--pi-host",           type=str, default="intent-chai.lan.local.cmu.edu")
    p.add_argument("--pi-port",           type=int, default=8000)
    p.add_argument("--batch-size",        type=int, default=4)
    p.add_argument("--num-samples",       type=int, default=5)
    p.add_argument("--num-chunks",        type=int, default=4)
    p.add_argument("--num-trajectories",  type=int, default=1000)
    p.add_argument("--num-batches",       type=int, default=None)
    p.add_argument("--segments-per-traj", type=int, default=None)
    p.add_argument("--open-loop-horizon", type=int, default=9)
    p.add_argument("--act-horizon",       type=int, default=15)
    p.add_argument("--fps",               type=int, default=5)
    p.add_argument("--use-ema",           action="store_true")
    p.add_argument("--torch-compile",     action="store_true")
    p.add_argument("--pred-actions",      action="store_true")
    p.add_argument("--dynamics-model",    type=str, default=None,
                   help="Path to dynamics model .pth (required with --pred-actions).")
    p.add_argument("--debug",             action="store_true")
    p.add_argument("--filter-episode-id", type=str, default=None)
    p.add_argument("--filter-success",    action="store_true")
    p.add_argument("--selection-criterion", type=str, default="reward",
                   choices=["reward", "advantage"])
    p.add_argument("--overrides",         nargs="*", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.debug:
        args.num_batches = 5
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = load_eval_config(args.checkpoint, args.overrides)
    wm, img_keys, _ = build_model(cfg, device, val_steps=16)
    load_checkpoint_into_model(wm, args.checkpoint, device, use_ema=args.use_ema)
    if args.torch_compile and hasattr(torch, "compile"):
        wm = torch.compile(wm)
    wm.eval()

    n_history = wm._n_history
    n_actions = wm.wm._n_actions
    n_states  = getattr(cfg.dataset, "n_states", 8)
    wrist_key = next((k for k in img_keys if "wrist"    in k), img_keys[-1])
    ext_key   = next((k for k in img_keys if "exterior" in k), img_keys[0])
    gamma     = getattr(cfg.model, "discount_factor", 0.99)

    # ── Norm stats + dynamics model ──────────────────────────────────────────
    state_mean, state_std, action_mean, action_std = load_norm_stats(args.dataset_path, args.pred_actions)
    dynamics_model = load_dynamics_model(args.dynamics_model, device) if args.pred_actions else None
    state_mean_t = torch.from_numpy(state_mean).to(device)
    state_std_t  = torch.from_numpy(state_std).to(device)

    # ── Dataloader ────────────────────────────────────────────────────────────
    B = args.batch_size
    cfg.dataset.path = args.dataset_path
    loader = create_synth_dataloader(
        cfg.dataset, n_history, cfg.horizon, B, 4, cfg.im_encoder.name,
        filter_episode_id=args.filter_episode_id, filter_success=args.filter_success,
    )
    target_trajs = args.num_trajectories
    max_batches  = args.num_batches
    if args.segments_per_traj is not None:
        max_batches = math.ceil(len(loader.dataset) * args.segments_per_traj / B)

    from .utils.tools import cycle
    loader_iter = iter(cycle(loader))

    # ── PI thread pool ────────────────────────────────────────────────────────
    N, M = args.num_samples, args.num_chunks
    executor      = concurrent.futures.ThreadPoolExecutor(max_workers=max(B, B * N))
    query_parallel = lambda req: _get_policy(args.pi_host, args.pi_port).infer_parallel(req, num_samples=N)["actions"]
    query_single   = lambda req: _get_policy(args.pi_host, args.pi_port).infer(req)["actions"]
    exec_ctrl_len  = min(args.open_loop_horizon, args.act_horizon)
    exec_len       = int(math.ceil(exec_ctrl_len / RGB_SKIP))
    print(f"B={B}  N={N}  M={M}  exec_len={exec_len}  γ={gamma}")

    global_traj_idx = 0
    batch_idx       = 0

    # ── Generation loop ───────────────────────────────────────────────────────
    while global_traj_idx < target_trajs:
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached max_batches={max_batches}; stopping.")
            break
        batch_idx += 1
        t_batch   = time.time()

        data      = next(loader_iter)
        task_text = _resolve_task_text(args.task, list(data["task"]["text"]))
        print(f"\n{'='*60}\nBatch {batch_idx}  saved={global_traj_idx}/{target_trajs}  Task: {task_text!r}")

        # Phase 1: encode real context + first PI query
        (context_BN, task_embed_BN, hist_actions_BN,
         state_norm_B, state_raw_B, state_norms_BN,
         raw_pi_chunks, fut_a_BN, fut_s_BN) = _prepare_batch_context(
            data, wm, img_keys, n_history, B, N, device,
            state_std_t, state_mean_t, task_text, executor, query_parallel,
            wrist_key, ext_key, exec_ctrl_len, exec_len, n_actions, n_states,
            action_mean, action_std, state_mean, state_std, dynamics_model,
        )

        # Phase 2: M-chunk WM imagination interleaved with PI queries
        result = _run_imagination_loop(
            wm, context_BN, hist_actions_BN, task_embed_BN, fut_a_BN, fut_s_BN,
            raw_pi_chunks, M, exec_len, n_history, B, N, n_actions, n_states,
            img_keys, device, action_mean, action_std, state_mean, state_std,
            dynamics_model, exec_ctrl_len, task_text, wrist_key, ext_key,
            executor, query_single,
        )

        # Phase 3: select best sample per segment by reward or advantage
        best_ns, advantages = _compute_selection_scores(result, B, N, gamma, args.selection_criterion)

        # Optional debug: decode and save all B*N samples as thumbnail videos
        all_frames = _save_debug_videos(result, wm, best_ns, B, N, img_keys,
                                         device, args.output_dir, batch_idx, args.fps) if args.debug else None

        # Phase 4: decode winner + build annotation + save video/JSON per segment
        for b in range(B):
            best_n = int(best_ns[b].item())
            print(f"  Segment {b}: best_n={best_n}  returns={[f'{v:.3f}' for v in result.accumulated_rewards[b].tolist()]}")
            vid_path = _save_winning_segment(
                b, b * N + best_n, best_n, result, advantages,
                wm, state_raw_B, img_keys, ext_key, wrist_key,
                exec_len, device, state_std, state_mean,
                args, global_traj_idx, task_text, batch_idx, N, M,
                all_frames=all_frames,
            )
            if vid_path is not None:
                print(f"  → Saved traj {global_traj_idx}: {vid_path}")
                global_traj_idx += 1

        print(f"Batch time: {time.time() - t_batch:.1f}s")

    executor.shutdown(wait=False)
    print(f"\nDone. Saved {global_traj_idx} trajectories to {args.output_dir}")


if __name__ == "__main__":
    main()
