#!/usr/bin/env python3
"""
Replay saved DROID trajectories through the WEAVER world model.

Autoregressively predicts future frames from real history context and
overlays per-frame reward curves on the comparison video (GT top, predicted
bottom).  Optionally saves inferred rewards to annotation JSON files and
produces a GT-vs-inferred reward comparison figure.

Usage:
    python -m weaver.replay_traj_reward \\
        --checkpoint /path/to/chkpts \\
        --dataset-path /path/to/droid \\
        --traj-ids 0 1 5 \\
        --output-dir ./eval_replay_reward \\
        --show-reward --use-ema
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from .datasets.droid import DROIDTrajectory
from .utils.viz import render_reward_curve, plot_reward_comparison, save_comparison_video
from .utils.wm_eval import (
    load_eval_config, build_model, load_norm_stats,
    load_checkpoint_into_model, wm_rollout_and_score,
)
from .wm.model import WEAVER


RGB_SKIP = 3


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Replay DROID trajectories through WEAVER + reward overlay")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--dataset-path",  type=str, required=True)
    p.add_argument("--output-dir",    type=str, default=None)
    p.add_argument("--traj-ids",      type=int, nargs="+", default=None)
    p.add_argument("--num-trajs",     type=int, default=10)
    p.add_argument("--split",         type=str, default="val", choices=["train", "val"])
    p.add_argument("--start-frame",   type=int, default=0)
    p.add_argument("--max-frames",    type=int, default=None)
    p.add_argument("--relabel-actions", action="store_true")
    p.add_argument("--use-ema",       action="store_true")
    p.add_argument("--fps",           type=int, default=5)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--annotation-dir", type=str, default="annotations")
    p.add_argument("--show-reward",   action="store_true",
                   help="Overlay per-frame reward curve at the bottom of each video.")
    p.add_argument("--save-rewards",  action="store_true",
                   help="Write inferred per-frame rewards into annotation JSON files.")
    p.add_argument("--no-plot",       action="store_true",
                   help="Skip saving the GT vs inferred reward comparison figure.")
    p.add_argument("--overrides",     nargs="*", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def _preprocess_states_actions(annotation: dict, relabel_actions: bool):
    """Extract video-rate states and actions from a raw DROID annotation dict."""
    joint_pos = np.array(annotation["observation.state.joint_position"])
    grip_pos  = np.array(annotation["observation.state.gripper_position"])[:, None]
    if grip_pos.ndim == 3:
        grip_pos = grip_pos[:, 0, :]
    full_states = np.concatenate([joint_pos, grip_pos], axis=-1)

    idx    = np.arange(0, len(joint_pos), RGB_SKIP)
    states = full_states[idx]

    if relabel_actions:
        next_idx = np.clip(idx + RGB_SKIP, 0, len(full_states) - 1)
        actions  = full_states[next_idx] - full_states[idx]
    else:
        jv = np.array(annotation["action.joint_velocity"])
        gp = np.array(annotation["action.gripper_position"])
        actions = []
        for i in idx:
            end = min(i + RGB_SKIP, len(jv))
            actions.append(np.concatenate([jv[i:end].sum(0), gp[end - 1]], axis=-1))
        actions = np.stack(actions, axis=0)

    return torch.from_numpy(states).float(), torch.from_numpy(actions).float()


def load_trajectory(
    dataset_path, traj_id, data_type, img_keys,
    start_frame, max_frames, relabel_actions,
    state_mean, state_std, action_mean, action_std,
    encoder_type="svd", n_memory_frames=0, t_memory=1,
    annotation_dir="annotations",
):
    """Load a full trajectory: latent obs, GT video frames, states, actions, and memory.

    obs and actions span [0, end_frame] so the generation loop can use any
    start_frame as context.  GT RGB is loaded only for the prediction window
    [start_frame+1, end_frame) for side-by-side comparison.
    """
    state_mean_t  = torch.from_numpy(state_mean)
    state_std_t   = torch.from_numpy(state_std)
    action_mean_t = torch.from_numpy(action_mean)
    action_std_t  = torch.from_numpy(action_std)

    traj = DROIDTrajectory(data_root=dataset_path, traj_id=traj_id, data_type=data_type,
                            load_on_init=True, encoder_type=encoder_type,
                            annotation_dir=annotation_dir)
    traj_len = len(traj)
    states, actions = _preprocess_states_actions(traj.annotation, relabel_actions)

    end_frame   = traj_len if max_frames is None else min(traj_len, start_frame + max_frames)
    end_frame   = min(end_frame, len(states))
    start_frame = min(start_frame, len(states) - 1)
    print(f"  Trajectory {traj_id}: {traj_len} frames, "
          f"context [0, {start_frame+1}), predicting [{start_frame+1}, {end_frame})")

    # Normalized obs (full window)
    obs = {}
    for key in img_keys:
        actual = key if key != "exterior_rand" else "exterior_1_left"
        if actual not in traj.latents:
            raise ValueError(f"Camera {actual} not found in trajectory {traj_id}")
        obs[f"{key}_features"] = torch.from_numpy(traj.latents[actual][0:end_frame]).unsqueeze(0)
    obs["states"] = ((states[0:end_frame] - state_mean_t) / state_std_t).unsqueeze(0)
    actions_norm  = ((actions[0:end_frame] - action_mean_t) / action_std_t).unsqueeze(0)

    # GT RGB (prediction window only)
    gt_frames_np = traj.get_video_frames(start_frame + 1, end_frame)
    gt_obs = {}
    for key in img_keys:
        actual = key if key != "exterior_rand" else "exterior_1_left"
        frames = torch.from_numpy(gt_frames_np[actual]).float() / 255.0
        gt_obs[key] = rearrange(frames, "t h w c -> 1 t c h w")

    # Optional memory frames before start_frame
    memory = None
    if n_memory_frames > 0:
        if start_frame > 0:
            mem_idx = np.arange(0, min(start_frame, len(states)), max(1, t_memory))
            if len(mem_idx) > n_memory_frames:
                mem_idx = mem_idx[-n_memory_frames:]
        else:
            mem_idx = np.empty(0, dtype=int)

        n_have = len(mem_idx)
        n_pad  = n_memory_frames - n_have
        all_s  = (states - state_mean_t) / state_std_t
        seed_s = all_s[start_frame:start_frame + 1]
        prior_s = all_s[mem_idx] if n_have > 0 else torch.empty(0, states.shape[-1])
        mem_states = torch.cat([seed_s.expand(n_pad, -1), prior_s], dim=0) if n_pad > 0 else prior_s
        memory = {"states": mem_states.unsqueeze(0)}
        for key in img_keys:
            actual = key if key != "exterior_rand" else "exterior_1_left"
            seed_l  = torch.from_numpy(traj.latents[actual][start_frame:start_frame + 1])
            prior_l = torch.from_numpy(traj.latents[actual][mem_idx]) if n_have > 0 else torch.empty(0, *seed_l.shape[1:])
            mem_lat = torch.cat([seed_l.expand(n_pad, *[-1]*(seed_l.ndim-1)), prior_l], dim=0) if n_pad > 0 else prior_l
            memory[f"{key}_features"] = mem_lat.unsqueeze(0)

    annotation_path = os.path.join(dataset_path, f"{traj.annotation_dir}/{data_type}/{traj_id}.json")
    return obs, gt_obs, actions_norm, traj.annotation.get("texts", [""])[0], \
           memory, traj.annotation, annotation_path


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def load_gt_reward_progress(annotation: dict) -> np.ndarray | None:
    rewards = annotation.get("reward_progress")
    if rewards is None:
        return None
    rewards = np.array(rewards, dtype=np.float32)
    video_len = annotation.get("video_length")
    raw_len   = annotation.get("raw_length", len(rewards))
    if video_len is not None and len(rewards) == raw_len and raw_len != video_len:
        rewards = rewards[np.arange(0, raw_len, RGB_SKIP)]
    return rewards - 1


def save_inferred_rewards(annotation_path, reward_values, pred_start_frame, video_length):
    with open(annotation_path) as f:
        ann = json.load(f)
    full = [None] * video_length
    for i, r in enumerate(reward_values):
        if pred_start_frame + i < video_length:
            full[pred_start_frame + i] = float(r)
    ann["inferred_reward_wm"]          = full
    ann["inferred_reward_pred_start"]  = pred_start_frame
    with open(annotation_path, "w") as f:
        json.dump(ann, f, indent=2)
    print(f"  Saved inferred rewards → {annotation_path}")


# ---------------------------------------------------------------------------
# WM generation with reward collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_videos_full_with_rewards(wm: WEAVER, obs, actions, task_embed,
                                       horizon=None, memory=None, bootstrap=None,
                                       start_frame=0):
    """Autoregressively predict future frames and return per-frame reward scores.

    The trajectory is divided into chunks of `bootstrap` frames. Each chunk is
    imagined by the WM conditioned on the previous n_history frames as context.
    After all chunks are generated, the reward model scores every predicted frame
    in a single forward pass (possible because rm is per-frame with no temporal
    coupling between frames).

    Args:
        obs:         Full obs dict from frame 0 to end_frame,
                     {key_features: (1, T, N, D), "states": (1, T, S)}.
        actions:     Normalized action tensor (1, T, A) spanning the same range.
        task_embed:  (1, D) task embedding for reward scoring; pass None to skip.
        start_frame: Last context frame — prediction begins at start_frame + 1.
        horizon:     WM rollout length per chunk (defaults to wm._n_horizon).
        bootstrap:   Frames kept per chunk; <= horizon (defaults to horizon).
        memory:      Optional pre-loaded memory obs dict for the memory module.

    Returns:
        decoded:       dict (1, T_pred, C, H, W) — pixel-decoded predicted frames.
        reward_values: list[float] length T_pred, or None if task_embed is None.
        pred_start:    int — the video-frame index where prediction begins.
    """
    horizon    = horizon or wm._n_horizon
    n_hist     = wm._n_history
    bootstrap  = bootstrap or horizon
    # pred_start: first frame to predict (must have at least n_hist real context frames before it)
    # ctx_start:  first frame of the n_hist-frame context window that precedes pred_start
    pred_start = max(n_hist, start_frame + 1)
    ctx_start  = pred_start - n_hist

    # Pre-encode memory once; updated after each chunk by rolling in the latest predicted frame
    memory_tokens = wm.encode_memory_obs(memory) if (wm._use_memory and memory is not None) else None
    n_memory      = wm._n_memory_frames if wm._use_memory else 0

    all_xt_chunks      = []  # predicted latents, one entry per chunk (history stripped)
    all_actions_chunks = []  # corresponding GT actions, collected for end-of-loop reward scoring
    xt_chunk = x1_chunk = None

    for chunk_idx, t_start in enumerate(range(pred_start, actions.shape[1], bootstrap)):
        # ── Build the (context + future) input window for this chunk ────────────
        # Chunk 0 uses real encoded observations as context.
        # Later chunks reuse the last n_hist predicted latents as the new context
        # so the model conditions on its own imagination rather than real frames.
        chunk_start   = ctx_start if chunk_idx == 0 else t_start - n_hist
        actions_chunk = actions[:, chunk_start:chunk_start + n_hist + horizon]
        if actions_chunk.shape[1] < n_hist + horizon:
            break  # not enough actions left for a full chunk

        if chunk_idx == 0:
            x1_chunk = wm.encode_obs(
                {k: v[:, chunk_start:chunk_start + n_hist + horizon] for k, v in obs.items()}
            )
        else:
            # Slide context window: overwrite the history slots with the last n_hist
            # predicted frames from the previous chunk, leaving future slots untouched.
            for k in xt_chunk:
                x1_chunk[k][:, :n_hist] = xt_chunk[k][:, -n_hist:]

        # Zero out future slots so the WM treats them as targets to denoise
        for k in x1_chunk:
            x1_chunk[k][:, n_hist:] *= 0

        # ── Imagination: denoise the future slots given the context ─────────────
        xt_chunk = wm.generate_latent_rollouts(x1_chunk, actions_chunk, memory_tokens=memory_tokens)

        # Store only the predicted frames (strip the history prefix) and their actions
        all_xt_chunks.append({k: v[:, n_hist:n_hist + bootstrap].clone() for k, v in xt_chunk.items()})
        all_actions_chunks.append(actions_chunk[:, n_hist:n_hist + bootstrap])

        # Trim xt_chunk so the next iteration's context slide works correctly
        for k in xt_chunk:
            xt_chunk[k] = xt_chunk[k][:, :n_hist + bootstrap]

        # ── Memory update: drop the oldest frame, append the latest predicted one ─
        # encode_memory_obs handles the full projection pipeline (patch encoding,
        # zero-action token, t=1 timestep embedding) so we only need to roll the buffer.
        if memory_tokens is not None and n_memory > 0:
            N_per = memory_tokens.shape[1] // n_memory
            new_frame = {f"{k}_features": xt_chunk[k][:, n_hist - 1:n_hist] for k in wm._img_keys}
            new_frame["states"] = xt_chunk["states"][:, n_hist - 1:n_hist]
            memory_tokens = torch.cat([memory_tokens[:, N_per:], wm.encode_memory_obs(new_frame)], dim=1)

    # ── Post-loop: decode pixels and score rewards in one pass ──────────────────
    all_xt  = {k: torch.cat([c[k] for c in all_xt_chunks], dim=1) for k in all_xt_chunks[0]}
    decoded = wm.decode_obs(all_xt, chunk_size=16)

    # wm.rm is a per-frame scorer (MLP with no temporal attention), so calling it on
    # the full concatenated trajectory is equivalent to calling it chunk-by-chunk.
    reward_values = None
    if task_embed is not None:
        amp = "cuda" if str(actions.device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=amp, dtype=torch.bfloat16):
            r = wm.rm(all_xt, torch.cat(all_actions_chunks, dim=1), task_embed)
        reward_values = r[0].float().cpu().tolist()

    return decoded, reward_values, pred_start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint)
    output_dir = (os.path.abspath(args.output_dir) if args.output_dir
                  else os.path.join(os.path.dirname(checkpoint_dir), "eval_replay_reward"))
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = load_eval_config(checkpoint_dir, args.overrides)
    wm, img_keys, _ = build_model(cfg, device, val_steps=10)
    load_checkpoint_into_model(wm, checkpoint_dir, device, use_ema=args.use_ema)
    if args.torch_compile and hasattr(torch, "compile"):
        wm = torch.compile(wm)
    wm.eval()
    print(f"Image keys: {img_keys}")

    # ── Norm stats ────────────────────────────────────────────────────────────
    state_mean, state_std, action_mean, action_std = load_norm_stats(args.dataset_path, args.relabel_actions)

    encoder_type = "sd3" if ("stable-diffusion-3" in cfg.im_encoder.name or
                              "sd3" in cfg.im_encoder.name) else "svd"

    # ── Trajectory list ───────────────────────────────────────────────────────
    data_type = "val" if args.split == "val" else "train"
    if args.traj_ids is not None:
        traj_ids = args.traj_ids
    else:
        anno_dir = Path(args.dataset_path) / args.annotation_dir / data_type
        traj_ids = sorted(int(p.stem) for p in anno_dir.glob("*.json"))[: args.num_trajs]
    print(f"Evaluating {len(traj_ids)} trajectories")

    eval_horizon = getattr(cfg, "eval_horizon", cfg.horizon)
    bootstrap    = getattr(cfg, "eval_bootstrap", None)
    need_rewards = args.show_reward or args.save_rewards or not args.no_plot

    # ── Per-trajectory loop ───────────────────────────────────────────────────
    for traj_id in tqdm(traj_ids, desc="Trajectories"):
        print(f"\n=== Trajectory {traj_id} ===")
        try:
            obs, gt_obs, actions, text, memory, annotation, annotation_path = load_trajectory(
                dataset_path=args.dataset_path, traj_id=traj_id, data_type=data_type,
                img_keys=img_keys, start_frame=args.start_frame, max_frames=args.max_frames,
                relabel_actions=args.relabel_actions,
                state_mean=state_mean, state_std=state_std,
                action_mean=action_mean, action_std=action_std,
                encoder_type=encoder_type,
                n_memory_frames=getattr(cfg, "n_memory_frames", 0),
                t_memory=getattr(cfg, "t_memory", 1),
                annotation_dir=args.annotation_dir,
            )
        except Exception as e:
            print(f"  Skipping trajectory {traj_id}: {e}")
            continue

        from .utils.tools import move_tensors_to_device
        obs     = move_tensors_to_device(obs, device=device)
        actions = actions.to(device)
        if memory is not None:
            memory = move_tensors_to_device(memory, device=device)
        print(f"  Task: {text!r}  |  frames: {actions.shape[1]}")

        task_embed = None
        if need_rewards:
            with torch.no_grad():
                task_embed = wm.encode_task({"text": [text or ""]})

        with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                              dtype=torch.bfloat16):
            decoded_pred, reward_values, pred_start = generate_videos_full_with_rewards(
                wm=wm, obs=obs, actions=actions, task_embed=task_embed,
                horizon=eval_horizon, memory=memory, bootstrap=bootstrap,
                start_frame=args.start_frame,
            )

        if reward_values:
            print(f"  Reward [{min(reward_values):.4f}, {max(reward_values):.4f}]  "
                  f"mean={sum(reward_values)/len(reward_values):.4f}")

        safe_text = text.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "")[:30]

        if args.save_rewards and reward_values:
            save_inferred_rewards(annotation_path, reward_values, pred_start,
                                  annotation.get("video_length", actions.shape[1]))

        if not args.no_plot and reward_values:
            plot_reward_comparison(
                gt_rewards=load_gt_reward_progress(annotation),
                inferred_rewards=reward_values,
                pred_start_frame=pred_start, traj_id=traj_id,
                out_path=os.path.join(output_dir, f"traj_{traj_id}_{safe_text}_reward_cmp.png"),
            )

        save_comparison_video(
            gt_obs, decoded_pred, img_keys,
            out_path=os.path.join(output_dir, f"traj_{traj_id}_{safe_text}.mp4"),
            fps=args.fps,
            reward_values=reward_values if args.show_reward else None,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
